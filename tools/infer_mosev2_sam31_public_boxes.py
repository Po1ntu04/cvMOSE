#!/usr/bin/env python3
"""Run MOSEv2 inference with SAM 3.1's public video API using GT boxes.

This is the low-intrusion control route: first-frame GT instance masks are only
used to derive per-object bounding boxes.  The boxes are submitted through the
public ``handle_request(add_prompt)`` surface, then SAM 3.1 propagates normally.
No facebookresearch/sam3 source is patched and no private mask-conditioning path
is used, except for a tiny session-start workaround for the current SAM3.1
``offload_state_to_cpu`` signature mismatch.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PublicPromptResult:
    frame_index: int
    outputs: Mapping[str, object]
    internal_to_gt: dict[int, int]
    match_iou: dict[int, float]
    source: str = "public_box_prompt"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_workspace = here.parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=default_workspace)
    p.add_argument("--sam3-root", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--pred-root", type=Path, default=None)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--videos", nargs="*", default=None, help="Optional subset of video names")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--overwrite-submission", action="store_true")
    p.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-video-to-cpu", dest="offload_video_to_cpu", action="store_false")
    p.add_argument("--max-num-objects", type=int, default=64)
    p.add_argument("--multiplex-count", type=int, default=16)
    p.add_argument("--use-fa3", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--async-loading-frames", action="store_true", default=False)
    p.add_argument("--prompt-mode", choices=["boxes", "points"], default="boxes", help="Public prompt type derived from first-frame GT")
    p.add_argument("--box-padding-px", type=float, default=0.0, help="Optional padding around GT boxes before normalization")
    p.add_argument("--points-per-object", type=int, default=1, help="Number of positive GT-derived points per object for prompt-mode=points")
    p.add_argument("--include-box-points", action="store_true", help="For prompt-mode=points, prepend bbox corner prompts with SAM-style labels 2/3")
    p.add_argument("--postprocess-batch-size", type=int, default=1, help="SAM3.1 frame postprocess batch size; 1 minimizes VRAM")
    p.add_argument("--batched-grounding-batch-size", type=int, default=1, help="SAM3.1 grounding batch size; 1 minimizes VRAM")
    p.add_argument("--disable-batched-grounding", action="store_true", help="Disable SAM3.1 batched grounding to reduce VRAM")
    p.add_argument("--trace-jsonl", type=Path, default=None)
    p.add_argument("--metrics-json", type=Path, default=None)
    return p.parse_args()


def complete_paths(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.sam3_root = (args.sam3_root or ws / "5_19" / "sam3").resolve()
    args.checkpoint = (args.checkpoint or ws / "5_19" / "data" / "sam3.1" / "sam3.1_multiplex.pt").resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.pred_root = (args.pred_root or ws / "homework" / "pred_sam31_public_boxes_b101").resolve()
    args.submit_root = (args.submit_root or ws / "homework" / "submission_433_sam31_public_boxes").resolve()
    args.zip_path = (args.zip_path or ws / "homework" / "submission_mosev2_sam31_public_boxes.zip").resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logs = ws / "homework" / "logs"
    args.trace_jsonl = (args.trace_jsonl or logs / f"sam31_public_boxes_trace_{stamp}.jsonl").resolve()
    args.metrics_json = (args.metrics_json or logs / f"sam31_public_boxes_metrics_{stamp}.json").resolve()
    return args


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(f"No frames found in {video_dir}")
    return frames


def load_first_annotation(ann_dir: Path) -> tuple[np.ndarray, list[int] | None]:
    ann_path = ann_dir / "00000.png"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing first-frame annotation: {ann_path}")
    img = Image.open(ann_path)
    palette = img.getpalette()
    ann = np.array(img)
    if ann.ndim != 2:
        raise ValueError(f"Annotation must be single-channel label PNG: {ann_path}, got shape {ann.shape}")
    return ann, palette


def save_label_png(path: Path, label: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if label.dtype != np.uint8:
        label = label.astype(np.uint8)
    img = Image.fromarray(label, mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def label_to_masks(label: np.ndarray, obj_ids: Sequence[int]) -> dict[int, np.ndarray]:
    return {int(obj_id): (label == int(obj_id)) for obj_id in obj_ids if np.any(label == int(obj_id))}


def masks_to_normalized_xywh_boxes(
    masks_by_obj_id: Mapping[int, np.ndarray],
    shape: tuple[int, int],
    *,
    padding_px: float = 0.0,
) -> tuple[list[int], np.ndarray]:
    h, w = shape
    boxes: list[list[float]] = []
    obj_ids: list[int] = []
    for obj_id, mask in masks_by_obj_id.items():
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        x0 = max(0.0, float(xs.min()) - padding_px)
        y0 = max(0.0, float(ys.min()) - padding_px)
        x1 = min(float(w), float(xs.max() + 1) + padding_px)
        y1 = min(float(h), float(ys.max() + 1) + padding_px)
        bw = max(1.0 / w, x1 - x0)
        bh = max(1.0 / h, y1 - y0)
        boxes.append([x0 / w, y0 / h, bw / w, bh / h])
        obj_ids.append(int(obj_id))
    if not boxes:
        raise ValueError("No non-empty object boxes derived from GT mask")
    arr = np.asarray(boxes, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    # Keep x+w/y+h in range after clipping.
    arr[:, 2] = np.minimum(arr[:, 2], 1.0 - arr[:, 0])
    arr[:, 3] = np.minimum(arr[:, 3], 1.0 - arr[:, 1])
    arr[:, 2:] = np.maximum(arr[:, 2:], 1e-6)
    return obj_ids, arr


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    return float(inter / union) if union > 0 else 0.0


def greedy_internal_to_gt_mapping(
    out_obj_ids: np.ndarray,
    out_masks: np.ndarray,
    gt_masks: Mapping[int, np.ndarray],
) -> tuple[dict[int, int], dict[int, float]]:
    pairs: list[tuple[float, int, int]] = []
    for out_idx, internal_id in enumerate([int(x) for x in out_obj_ids.tolist()]):
        if out_idx >= len(out_masks):
            continue
        for gt_id, gt_mask in gt_masks.items():
            pairs.append((compute_iou(out_masks[out_idx], gt_mask), internal_id, int(gt_id)))
    pairs.sort(reverse=True, key=lambda x: x[0])
    mapping: dict[int, int] = {}
    match_iou: dict[int, float] = {}
    used_gt: set[int] = set()
    for iou, internal_id, gt_id in pairs:
        if internal_id in mapping or gt_id in used_gt:
            continue
        mapping[internal_id] = gt_id
        match_iou[internal_id] = float(iou)
        used_gt.add(gt_id)
    return mapping, match_iou


def positive_points_from_mask(mask: np.ndarray, shape: tuple[int, int], count: int = 1) -> np.ndarray:
    """Return normalized (x, y) positive points inside a GT mask."""
    h, w = shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("Cannot sample points from an empty mask")
    cy = float(ys.mean())
    cx = float(xs.mean())
    candidates: list[tuple[float, float]] = [(cx, cy)]
    if count > 1:
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        candidates.extend(
            [
                ((x0 + x1) * 0.5, (y0 + y1) * 0.5),
                (x0 + (x1 - x0) * 0.25, y0 + (y1 - y0) * 0.25),
                (x0 + (x1 - x0) * 0.75, y0 + (y1 - y0) * 0.25),
                (x0 + (x1 - x0) * 0.25, y0 + (y1 - y0) * 0.75),
                (x0 + (x1 - x0) * 0.75, y0 + (y1 - y0) * 0.75),
            ]
        )
    selected: list[tuple[float, float]] = []
    used: set[tuple[int, int]] = set()
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    for cand_x, cand_y in candidates:
        d2 = (coords[:, 0] - cand_x) ** 2 + (coords[:, 1] - cand_y) ** 2
        for idx in np.argsort(d2):
            px, py = int(xs[idx]), int(ys[idx])
            if (px, py) not in used:
                used.add((px, py))
                selected.append(((px + 0.5) / w, (py + 0.5) / h))
                break
        if len(selected) >= count:
            break
    while len(selected) < count:
        idx = len(selected) % ys.size
        px, py = int(xs[idx]), int(ys[idx])
        selected.append(((px + 0.5) / w, (py + 0.5) / h))
    return np.asarray(selected[:count], dtype=np.float32)


def normalized_box_corner_points(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("Cannot build box points from an empty mask")
    x0 = (float(xs.min()) + 0.5) / w
    y0 = (float(ys.min()) + 0.5) / h
    x1 = (float(xs.max()) + 0.5) / w
    y1 = (float(ys.max()) + 0.5) / h
    return np.asarray([[x0, y0], [x1, y1]], dtype=np.float32)


def outputs_to_label(outputs: Mapping[str, object], shape: tuple[int, int], internal_to_gt: Mapping[int, int]) -> np.ndarray:
    label = np.zeros(shape, dtype=np.uint16)
    masks = np.asarray(outputs.get("out_binary_masks", np.zeros((0, *shape), dtype=bool)), dtype=bool)
    out_obj_ids = np.asarray(outputs.get("out_obj_ids", np.zeros(0, dtype=np.int64))).reshape(-1)
    probs = outputs.get("out_probs", None)
    order = np.arange(len(out_obj_ids))
    if probs is not None and len(np.asarray(probs).reshape(-1)) == len(order):
        order = np.argsort(np.asarray(probs).reshape(-1))
    for idx in order:
        internal_id = int(out_obj_ids[int(idx)])
        gt_id = internal_to_gt.get(internal_id)
        if gt_id is None or int(idx) >= len(masks):
            continue
        label[masks[int(idx)]] = int(gt_id)
    if label.max(initial=0) <= 255:
        return label.astype(np.uint8)
    return label


def cuda_memory_snapshot() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Sam31PublicBoxRunner:
    def __init__(self, predictor):
        self.predictor = predictor
        self.model = predictor.model

    @classmethod
    def build(
        cls,
        *,
        sam3_root: Path,
        checkpoint_path: Path,
        use_fa3: bool = False,
        compile_model: bool = False,
        max_num_objects: int = 64,
        multiplex_count: int = 16,
        async_loading_frames: bool = False,
        postprocess_batch_size: int = 1,
        batched_grounding_batch_size: int = 1,
        disable_batched_grounding: bool = False,
    ) -> "Sam31PublicBoxRunner":
        sys.path.insert(0, str(sam3_root))
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_path),
            max_num_objects=max_num_objects,
            multiplex_count=multiplex_count,
            use_fa3=use_fa3,
            compile=compile_model,
            warm_up=False,
            async_loading_frames=async_loading_frames,
        )
        if hasattr(predictor.model, "postprocess_batch_size"):
            predictor.model.postprocess_batch_size = int(postprocess_batch_size)
        if hasattr(predictor.model, "batched_grounding_batch_size"):
            predictor.model.batched_grounding_batch_size = int(batched_grounding_batch_size)
        if disable_batched_grounding and hasattr(predictor.model, "use_batched_grounding"):
            predictor.model.use_batched_grounding = False
        predictor.model.eval()
        return cls(predictor)

    def start_session(self, video_dir: Path | str, *, offload_video_to_cpu: bool = True) -> str:
        # Work around the current SAM3.1 public predictor forwarding an unsupported
        # offload_state_to_cpu kwarg into Sam3MultiplexTrackingWithInteractivity.init_state.
        init_kwargs = {
            "resource_path": str(video_dir),
            "offload_video_to_cpu": offload_video_to_cpu,
        }
        if hasattr(self.predictor, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.predictor.async_loading_frames
        state = self.model.init_state(**init_kwargs)
        session_id = str(uuid.uuid4())
        self.predictor._all_inference_states[session_id] = {
            "state": state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return session_id

    def close_session(self, session_id: str) -> None:
        self.predictor.handle_request({"type": "close_session", "session_id": session_id})

    def add_first_frame_boxes(
        self,
        session_id: str,
        ann: np.ndarray,
        obj_ids: Sequence[int],
        *,
        box_padding_px: float = 0.0,
    ) -> PublicPromptResult:
        gt_masks = label_to_masks(ann, obj_ids)
        gt_obj_ids, boxes = masks_to_normalized_xywh_boxes(gt_masks, ann.shape, padding_px=box_padding_px)
        response = self.predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "bounding_boxes": boxes,
                "bounding_box_labels": np.ones(len(gt_obj_ids), dtype=np.int32),
            }
        )
        outputs = response["outputs"]
        out_obj_ids = np.asarray(outputs.get("out_obj_ids", np.zeros(0, dtype=np.int64))).reshape(-1)
        out_masks = np.asarray(outputs.get("out_binary_masks", np.zeros((0, *ann.shape), dtype=bool)), dtype=bool)
        internal_to_gt, match_iou = greedy_internal_to_gt_mapping(out_obj_ids, out_masks, gt_masks)
        return PublicPromptResult(
            frame_index=int(response["frame_index"]),
            outputs=outputs,
            internal_to_gt=internal_to_gt,
            match_iou=match_iou,
        )

    def add_first_frame_points(
        self,
        session_id: str,
        ann: np.ndarray,
        obj_ids: Sequence[int],
        *,
        points_per_object: int = 1,
        include_box_points: bool = False,
    ) -> PublicPromptResult:
        gt_masks = label_to_masks(ann, obj_ids)
        last_response = None
        for obj_id in obj_ids:
            pts = positive_points_from_mask(gt_masks[int(obj_id)], ann.shape, count=max(1, points_per_object))
            labels = np.ones(len(pts), dtype=np.int32)
            if include_box_points:
                box_pts = normalized_box_corner_points(gt_masks[int(obj_id)], ann.shape)
                pts = np.concatenate([box_pts, pts], axis=0)
                labels = np.concatenate([np.asarray([2, 3], dtype=np.int32), labels], axis=0)
            last_response = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "points": pts,
                    "point_labels": labels,
                    "obj_id": int(obj_id),
                    "rel_coordinates": True,
                }
            )
        if last_response is None:
            raise ValueError("No object ids supplied for point prompts")
        outputs = last_response["outputs"]
        out_obj_ids = np.asarray(outputs.get("out_obj_ids", np.zeros(0, dtype=np.int64))).reshape(-1)
        out_masks = np.asarray(outputs.get("out_binary_masks", np.zeros((0, *ann.shape), dtype=bool)), dtype=bool)
        identity = {int(x): int(x) for x in out_obj_ids.tolist() if int(x) in set(map(int, obj_ids))}
        match_iou = {
            int(obj_id): compute_iou(out_masks[idx], gt_masks[int(obj_id)])
            for idx, obj_id in enumerate([int(x) for x in out_obj_ids.tolist()])
            if idx < len(out_masks) and int(obj_id) in gt_masks
        }
        return PublicPromptResult(
            frame_index=int(last_response["frame_index"]),
            outputs=outputs,
            internal_to_gt=identity,
            match_iou=match_iou,
            source="public_point_prompt",
        )

    def propagate_forward(self, session_id: str, *, start_frame_idx: int = 0):
        yield from self.predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": start_frame_idx,
            }
        )


def frame_trace_record(video: str, frame_path: Path, frame_idx: int, outputs: Mapping[str, object], seconds: float, *, source: str, internal_to_gt: Mapping[int, int] | None = None, match_iou: Mapping[int, float] | None = None) -> dict:
    masks = np.asarray(outputs.get("out_binary_masks", np.zeros((0, 0, 0), dtype=bool)), dtype=bool)
    out_obj_ids = np.asarray(outputs.get("out_obj_ids", np.zeros(0, dtype=np.int64))).reshape(-1)
    rec = {
        "event": "frame",
        "video": video,
        "frame_index": int(frame_idx),
        "frame_name": frame_path.name,
        "source": source,
        "seconds_since_video_start": round(seconds, 4),
        "internal_object_ids": [int(x) for x in out_obj_ids.tolist()],
        "mapped_object_ids": [int(internal_to_gt[int(x)]) for x in out_obj_ids.tolist() if internal_to_gt and int(x) in internal_to_gt],
        "mask_areas": masks.reshape(masks.shape[0], -1).sum(axis=1).astype(int).tolist() if masks.size else [],
    }
    if "out_probs" in outputs:
        rec["out_probs"] = [float(x) for x in np.asarray(outputs["out_probs"]).reshape(-1).tolist()]
    if internal_to_gt is not None:
        rec["internal_to_gt"] = {str(k): int(v) for k, v in internal_to_gt.items()}
    if match_iou is not None:
        rec["prompt_match_iou"] = {str(k): float(v) for k, v in match_iou.items()}
    return rec


def run_video(runner: Sam31PublicBoxRunner, args: argparse.Namespace, video_name: str) -> dict:
    import torch

    video_dir = args.jpeg_root / video_name
    ann_dir = args.ann_root / video_name
    out_dir = args.pred_root / video_name
    frames = list_frames(video_dir)
    ann, palette = load_first_annotation(ann_dir)
    obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
    if not obj_ids:
        raise ValueError(f"No foreground object ids in {ann_dir / '00000.png'}")

    if args.skip_existing and len(sorted(out_dir.glob("*.png"))) == len(frames):
        return {"video": video_name, "status": "skipped", "frames": len(frames), "objects": obj_ids}

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    video_started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    session_id = runner.start_session(video_dir, offload_video_to_cpu=args.offload_video_to_cpu)
    seen: set[int] = set()
    try:
        if args.prompt_mode == "points":
            prompt_result = runner.add_first_frame_points(
                session_id,
                ann,
                obj_ids,
                points_per_object=args.points_per_object,
                include_box_points=args.include_box_points,
            )
        else:
            prompt_result = runner.add_first_frame_boxes(
                session_id,
                ann,
                obj_ids,
                box_padding_px=args.box_padding_px,
            )
        save_label_png(out_dir / f"{frames[0].stem}.png", ann.copy(), palette)
        seen.add(0)
        write_jsonl(
            args.trace_jsonl,
            frame_trace_record(
                video_name,
                frames[0],
                0,
                prompt_result.outputs,
                time.time() - video_started,
                source=prompt_result.source,
                internal_to_gt=prompt_result.internal_to_gt,
                match_iou=prompt_result.match_iou,
            ),
        )

        for response in runner.propagate_forward(session_id, start_frame_idx=0):
            frame_idx = int(response["frame_index"])
            if frame_idx == 0:
                continue
            if frame_idx < 0 or frame_idx >= len(frames):
                raise IndexError(f"SAM3.1 returned frame_idx={frame_idx} outside 0..{len(frames)-1}")
            outputs = response["outputs"]
            label = outputs_to_label(outputs, ann.shape, prompt_result.internal_to_gt)
            save_label_png(out_dir / f"{frames[frame_idx].stem}.png", label, palette)
            seen.add(frame_idx)
            write_jsonl(
                args.trace_jsonl,
                frame_trace_record(
                    video_name,
                    frames[frame_idx],
                    frame_idx,
                    outputs,
                    time.time() - video_started,
                    source="public_box_propagate",
                    internal_to_gt=prompt_result.internal_to_gt,
                ),
            )
    finally:
        runner.close_session(session_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing = [i for i in range(len(frames)) if i not in seen]
    if missing:
        raise RuntimeError(f"{video_name}: missing predicted frames {missing[:20]} (total {len(missing)})")

    outputs = sorted(out_dir.glob("*.png"))
    expected_names = [f"{f.stem}.png" for f in frames]
    got_names = [f.name for f in outputs]
    if got_names != expected_names:
        raise RuntimeError(f"{video_name}: output names mismatch")
    for sample_idx in [0, len(outputs) - 1]:
        out_img = Image.open(outputs[sample_idx])
        frame_img = Image.open(frames[sample_idx])
        if out_img.size != frame_img.size:
            raise RuntimeError(f"{video_name}: size mismatch at {outputs[sample_idx].name}: {out_img.size} vs {frame_img.size}")

    seconds = time.time() - video_started
    return {
        "video": video_name,
        "status": "done",
        "frames": len(frames),
        "objects": obj_ids,
        "mapped_objects": sorted(set(prompt_result.internal_to_gt.values())),
        "prompt_match_iou": {str(k): round(v, 6) for k, v in prompt_result.match_iou.items()},
        "seconds": round(seconds, 3),
        "fps": round(len(frames) / seconds, 3) if seconds > 0 else None,
        "cuda": cuda_memory_snapshot(),
    }


def make_submission(args: argparse.Namespace) -> None:
    if args.submit_root.exists() and args.overwrite_submission:
        shutil.rmtree(args.submit_root)
    args.submit_root.mkdir(parents=True, exist_ok=True)

    for src_root in [args.provided_output_root, args.pred_root]:
        if not src_root.is_dir():
            raise FileNotFoundError(src_root)
        for video_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
            dst = args.submit_root / video_dir.name
            shutil.copytree(video_dir, dst, dirs_exist_ok=True)

    video_dirs = sorted(p for p in args.submit_root.iterdir() if p.is_dir())
    png_count = sum(1 for _ in args.submit_root.rglob("*.png"))
    if len(video_dirs) != 433:
        raise RuntimeError(f"Submission must contain 433 video dirs; got {len(video_dirs)} at {args.submit_root}")
    if png_count != 66526:
        raise RuntimeError(f"Submission must contain 66526 PNGs; got {png_count} at {args.submit_root}")

    if not args.no_zip:
        if args.zip_path.exists():
            args.zip_path.unlink()
        with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(args.submit_root.rglob("*.png")):
                zf.write(file, file.relative_to(args.submit_root).as_posix())


def main() -> None:
    args = complete_paths(parse_args())
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_jsonl.exists():
        args.trace_jsonl.unlink()

    missing = [p for p in [args.sam3_root, args.checkpoint, args.jpeg_root, args.ann_root] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths: " + ", ".join(map(str, missing)))

    import torch

    build_started = time.time()
    runner = Sam31PublicBoxRunner.build(
        sam3_root=args.sam3_root,
        checkpoint_path=args.checkpoint,
        use_fa3=args.use_fa3,
        compile_model=args.compile,
        max_num_objects=args.max_num_objects,
        multiplex_count=args.multiplex_count,
        async_loading_frames=args.async_loading_frames,
        postprocess_batch_size=args.postprocess_batch_size,
        batched_grounding_batch_size=args.batched_grounding_batch_size,
        disable_batched_grounding=args.disable_batched_grounding,
    )
    build_seconds = time.time() - build_started

    videos = args.videos or sorted(p.name for p in args.ann_root.iterdir() if p.is_dir())
    run_started = time.time()
    results = []
    for video_name in videos:
        result = run_video(runner, args, video_name)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    if args.make_submission:
        make_submission(args)

    total_seconds = time.time() - run_started
    metrics = {
        "route": "sam31_public_boxes",
        "workspace": str(args.workspace),
        "sam3_root": str(args.sam3_root),
        "checkpoint": str(args.checkpoint),
        "pred_root": str(args.pred_root),
        "submit_root": str(args.submit_root),
        "zip_path": str(args.zip_path) if args.make_submission and not args.no_zip else None,
        "trace_jsonl": str(args.trace_jsonl),
        "build_seconds": round(build_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "videos": len(videos),
        "frames": int(sum(r.get("frames", 0) for r in results)),
        "fps": round(sum(r.get("frames", 0) for r in results) / total_seconds, 3) if total_seconds > 0 else None,
        "prompt_mode": args.prompt_mode,
        "box_padding_px": args.box_padding_px,
        "points_per_object": args.points_per_object,
        "include_box_points": args.include_box_points,
        "postprocess_batch_size": args.postprocess_batch_size,
        "batched_grounding_batch_size": args.batched_grounding_batch_size,
        "disable_batched_grounding": args.disable_batched_grounding,
        "cuda_final": cuda_memory_snapshot(),
        "results": results,
    }
    peaks = [r.get("cuda", {}) for r in results if r.get("status") == "done"]
    if peaks:
        metrics["max_video_peak_cuda"] = max(peaks, key=lambda x: x.get("max_allocated_bytes", -1))
    args.metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics_json": str(args.metrics_json), "summary": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

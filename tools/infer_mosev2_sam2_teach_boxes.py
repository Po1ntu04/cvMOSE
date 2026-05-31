#!/usr/bin/env python3
"""Probe MLLM teach-SAM box prompts with SAM2 bounded propagation.

This is an experimental M13 tool.  It consumes split-harness event-story JSON,
adds selected non-frame0 MLLM boxes as SAM2 video prompts, propagates, then
merges only selected object/window regions onto a conservative baseline root.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class BoxAction:
    video: str
    obj_id: int
    frame_idx: int
    box_norm_1000: list[float]
    confidence: float
    source: str
    location_description: str
    event_type: str
    diagnosis: str
    prompt_type: str = "box"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--sam2-root", type=Path, default=None)
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--baseline-root", type=Path, default=None)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--targets", nargs="*", default=None, help="video:obj filters")
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-confidence", type=float, default=0.80)
    p.add_argument("--max-actions-per-object", type=int, default=2)
    p.add_argument("--prefer-last-anchor", action="store_true", default=True)
    p.add_argument("--no-prefer-last-anchor", dest="prefer_last_anchor", action="store_false")
    p.add_argument("--merge-radius", type=int, default=12)
    p.add_argument("--merge-mode", choices=["window", "from_anchor", "all_after_first_anchor"], default="window")
    p.add_argument("--propagate-direction", choices=["forward", "reverse", "both"], default="forward", help="SAM2 propagation direction after adding MLLM anchors. both enables late-anchor backward recovery.")
    p.add_argument("--reverse-max-frames", type=int, default=0, help="0 means reverse all the way to frame 0; otherwise bound reverse propagation length.")
    p.add_argument("--clip-to-anchor-box", action="store_true", help="When merging target masks, keep only pixels inside expanded MLLM anchor boxes.")
    p.add_argument("--clip-pad-frac", type=float, default=0.35, help="Fractional box padding for --clip-to-anchor-box.")
    p.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-video-to-cpu", dest="offload_video_to_cpu", action="store_false")
    p.add_argument("--offload-state-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-state-to-cpu", dest="offload_state_to_cpu", action="store_false")
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--overwrite-submission", action="store_true")
    return p.parse_args()


def complete(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve(); args.workspace = ws
    args.sam2_root = (args.sam2_root or ws / "5_19" / "sam2").resolve()
    args.checkpoint = (args.checkpoint or ws / "5_19" / "data" / "sam2" / "sam2.1_hiera_base_plus.pt").resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.baseline_root = (args.baseline_root or ws / "homework" / "pred_m7_part_balanced").resolve()
    if args.submit_root is None:
        args.submit_root = ws / "homework" / "submission_433_m13_teach_boxes"
    if args.zip_path is None:
        args.zip_path = ws / "homework" / "submission_mosev2_m13_teach_boxes.zip"
    args.pred_root = args.pred_root.resolve(); args.submit_root = args.submit_root.resolve(); args.zip_path = args.zip_path.resolve(); args.audit_json = args.audit_json.resolve()
    return args


def parse_target_filter(items: list[str] | None) -> set[tuple[str, int]] | None:
    if not items:
        return None
    out: set[tuple[str, int]] = set()
    for item in items:
        video, obj = item.split(":", 1)
        out.add((video, int(obj)))
    return out


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(video_dir)
    return frames


def load_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr


def load_first_annotation(ann_dir: Path) -> tuple[np.ndarray, list[int] | None]:
    path = ann_dir / "00000.png"
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr, img.getpalette()


def save_label_png(path: Path, label: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(label.astype(np.uint8), mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def logits_to_label(mask_logits, obj_ids: list[int]) -> np.ndarray:
    import torch
    if isinstance(mask_logits, torch.Tensor):
        logits = mask_logits.detach().float().cpu().numpy()
    else:
        logits = np.asarray(mask_logits)
    if logits.ndim == 4:
        logits = logits[:, 0]
    ids = np.asarray([int(x) for x in obj_ids], dtype=np.uint16)
    best_idx = np.argmax(logits, axis=0)
    best_score = np.max(logits, axis=0)
    out = np.zeros(logits.shape[1:], dtype=np.uint16)
    pos = best_score > 0
    out[pos] = ids[best_idx[pos]]
    return out.astype(np.uint8) if out.max(initial=0) <= 255 else out


def import_sam2(root: Path):
    sys.path.insert(0, str(root))
    from sam2.build_sam import build_sam2_video_predictor
    return build_sam2_video_predictor


def norm_to_xyxy(box_norm: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(x) for x in box_norm]
    x1 = max(0.0, min(width - 1.0, x1 * width / 1000.0))
    x2 = max(0.0, min(width - 1.0, x2 * width / 1000.0))
    y1 = max(0.0, min(height - 1.0, y1 * height / 1000.0))
    y2 = max(0.0, min(height - 1.0, y2 * height / 1000.0))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def collect_actions(args: argparse.Namespace) -> dict[str, dict[int, list[BoxAction]]]:
    data = json.loads(args.split_json.read_text(encoding="utf-8"))
    filt = parse_target_filter(args.targets)
    out: dict[str, dict[int, list[BoxAction]]] = {}
    for rec in data.get("records", []):
        video = str(rec.get("video")); obj_id = int(rec.get("obj_id"))
        if args.videos and video not in set(args.videos):
            continue
        if filt and (video, obj_id) not in filt:
            continue
        j = rec.get("judgment", {}) or {}
        event = str(j.get("event_type") or "")
        diagnosis = str(j.get("current_prediction_diagnosis") or "")
        actions: list[BoxAction] = []
        for idx, plan in enumerate(j.get("positive_prompt_plan") or []):
            if plan.get("prompt_type", "box") != "box":
                continue
            frame_idx = int(plan.get("frame_idx") or 0)
            if frame_idx <= 0:
                continue
            conf = float(plan.get("confidence") or 0.0)
            box = plan.get("box_norm_1000") or plan.get("bbox_norm")
            if conf < args.min_confidence or not isinstance(box, list) or len(box) != 4:
                continue
            actions.append(BoxAction(video, obj_id, frame_idx, [float(x) for x in box], conf, f"split_plan:{idx}", str(plan.get("location_description") or ""), event, diagnosis, str(plan.get("prompt_type") or "box")))
        actions.sort(key=lambda a: (a.frame_idx if args.prefer_last_anchor else -a.confidence), reverse=args.prefer_last_anchor)
        actions = actions[: max(0, int(args.max_actions_per_object))]
        if actions:
            out.setdefault(video, {})[obj_id] = sorted(actions, key=lambda a: a.frame_idx)
    return out



def expand_xyxy_int(box: list[float], width: int, height: int, pad_frac: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(x) for x in box]
    bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)
    pad = max(bw, bh) * float(pad_frac)
    xx1 = max(0, int(round(x1 - pad))); yy1 = max(0, int(round(y1 - pad)))
    xx2 = min(width - 1, int(round(x2 + pad))); yy2 = min(height - 1, int(round(y2 + pad)))
    return xx1, yy1, xx2, yy2


def anchor_clip_masks(actions_by_obj: dict[int, list[BoxAction]], width: int, height: int, pad_frac: float) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for obj_id, actions in actions_by_obj.items():
        mask = np.zeros((height, width), dtype=bool)
        for action in actions:
            box = norm_to_xyxy(action.box_norm_1000, width, height)
            x1, y1, x2, y2 = expand_xyxy_int(box, width, height, pad_frac)
            mask[y1 : y2 + 1, x1 : x2 + 1] = True
        out[int(obj_id)] = mask
    return out

def merge_window(frames_count: int, actions: list[BoxAction], mode: str, radius: int) -> set[int]:
    if not actions:
        return set()
    if mode == "all_after_first_anchor":
        lo = min(a.frame_idx for a in actions)
        return set(range(max(1, lo), frames_count))
    out: set[int] = set()
    for a in actions:
        if mode == "from_anchor":
            out.update(range(max(1, a.frame_idx), frames_count))
        else:
            out.update(range(max(1, a.frame_idx - radius), min(frames_count - 1, a.frame_idx + radius) + 1))
    return out


def run_video(predictor, args: argparse.Namespace, video: str, actions_by_obj: dict[int, list[BoxAction]]) -> dict[str, Any]:
    import torch
    video_dir = args.jpeg_root / video
    frames = list_frames(video_dir)
    ann, palette = load_first_annotation(args.ann_root / video)
    obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
    state = predictor.init_state(video_path=str(video_dir), offload_video_to_cpu=args.offload_video_to_cpu, offload_state_to_cpu=args.offload_state_to_cpu)
    predictor.reset_state(state)
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if str(args.device).startswith("cuda") and torch.cuda.is_available() else contextlib.nullcontext()
    action_audit: list[dict[str, Any]] = []
    with torch.inference_mode(), autocast_ctx:
        for obj_id in obj_ids:
            predictor.add_new_mask(state, frame_idx=0, obj_id=int(obj_id), mask=(ann == int(obj_id)))
        with Image.open(frames[0]) as img0:
            width, height = img0.size
        for obj_id, actions in actions_by_obj.items():
            for action in actions:
                if action.frame_idx >= len(frames):
                    continue
                box_xyxy = norm_to_xyxy(action.box_norm_1000, width, height)
                predictor.add_new_points_or_box(state, frame_idx=int(action.frame_idx), obj_id=int(obj_id), box=box_xyxy, clear_old_points=True, normalize_coords=True)
                d = asdict(action); d["box_xyxy"] = [round(float(x), 2) for x in box_xyxy]
                action_audit.append(d)
        reprop_labels: dict[int, np.ndarray] = {}
        propagation_audit: list[dict[str, Any]] = []
        if args.propagate_direction in {"forward", "both"}:
            count = 0
            for frame_idx, cur_obj_ids, mask_logits in predictor.propagate_in_video(state):
                reprop_labels[int(frame_idx)] = logits_to_label(mask_logits, [int(x) for x in cur_obj_ids]) if frame_idx != 0 else ann.copy()
                count += 1
            propagation_audit.append({"direction": "forward", "start_frame_idx": None, "frames": count})
        if args.propagate_direction in {"reverse", "both"}:
            reverse_starts = sorted({int(a.frame_idx) for acts in actions_by_obj.values() for a in acts if int(a.frame_idx) > 0}, reverse=True)
            for start_idx in reverse_starts:
                max_track = None if int(args.reverse_max_frames) <= 0 else int(args.reverse_max_frames)
                count = 0
                for frame_idx, cur_obj_ids, mask_logits in predictor.propagate_in_video(state, start_frame_idx=start_idx, max_frame_num_to_track=max_track, reverse=True):
                    reprop_labels[int(frame_idx)] = logits_to_label(mask_logits, [int(x) for x in cur_obj_ids]) if frame_idx != 0 else ann.copy()
                    count += 1
                propagation_audit.append({"direction": "reverse", "start_frame_idx": start_idx, "frames": count, "max_frame_num_to_track": max_track})
    out_dir = args.pred_root / video
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(frames[0]) as img0:
        width, height = img0.size
    windows_by_obj: dict[int, set[int]] = {obj_id: merge_window(len(frames), acts, args.merge_mode, args.merge_radius) for obj_id, acts in actions_by_obj.items()}
    clip_masks = anchor_clip_masks(actions_by_obj, width, height, args.clip_pad_frac) if args.clip_to_anchor_box else {}
    changed_frames: dict[str, list[int]] = {str(obj_id): [] for obj_id in windows_by_obj}
    for i, frame in enumerate(frames):
        if i == 0:
            final = ann.copy()
        else:
            base_path = args.baseline_root / video / f"{frame.stem}.png"
            final = load_label(base_path).copy() if base_path.is_file() else reprop_labels.get(i, np.zeros_like(ann)).copy()
            rp = reprop_labels.get(i)
            if rp is not None:
                for obj_id, win in windows_by_obj.items():
                    if i in win:
                        before = final.copy()
                        final[final == int(obj_id)] = 0
                        candidate_mask = (rp == int(obj_id))
                        if int(obj_id) in clip_masks:
                            candidate_mask = candidate_mask & clip_masks[int(obj_id)]
                        final[candidate_mask] = int(obj_id)
                        if not np.array_equal(before == int(obj_id), final == int(obj_id)):
                            changed_frames[str(obj_id)].append(i)
        save_label_png(out_dir / f"{frame.stem}.png", final, palette)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"video": video, "frames": len(frames), "objects": obj_ids, "actions": action_audit, "propagation": propagation_audit, "windows": {str(k): sorted(v) for k, v in windows_by_obj.items()}, "changed_frames": changed_frames}


def copy_video_from_baseline(args: argparse.Namespace, video: str) -> dict[str, Any]:
    src = args.baseline_root / video
    dst = args.pred_root / video
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return {"video": video, "copied_baseline": True, "frames": len(list(dst.glob('*.png')))}


def expected_pred_names(jpeg_root: Path, video: str) -> list[str]:
    return [f"{p.stem}.png" for p in list_frames(jpeg_root / video)]


def ensure_pred_root_complete(args: argparse.Namespace) -> None:
    """Fail closed by filling non-experiment videos from the baseline root.

    M13 smoke runs often process only a subset of the 15 homework videos.
    A submission zip is valid only when all 15 predicted videos are present,
    so missing untouched videos are copied from the conservative baseline.
    Existing but incomplete video directories are treated as errors rather than
    silently patched, because that would hide broken inference output.
    """
    for video_dir in sorted(p for p in args.jpeg_root.iterdir() if p.is_dir()):
        video = video_dir.name
        dst = args.pred_root / video
        expected = expected_pred_names(args.jpeg_root, video)
        if not dst.exists():
            src = args.baseline_root / video
            if not src.is_dir():
                raise FileNotFoundError(f"missing baseline video for submission fill: {src}")
            shutil.copytree(src, dst)
        got = sorted(p.name for p in dst.glob("*.png"))
        if got != expected:
            raise RuntimeError(f"pred_root incomplete for {video}: expected {len(expected)} frames, got {len(got)}")


def validate_submission_or_die(args: argparse.Namespace) -> None:
    from validate_mose_submission import validate_zip

    result = validate_zip(args.zip_path, args.provided_output_root, args.jpeg_root, args.ann_root, 433, 66526, 15, False)
    if not result.get("ok"):
        raise SystemExit("submission validation failed: " + json.dumps(result, ensure_ascii=False))


def make_submission(args: argparse.Namespace) -> None:
    ensure_pred_root_complete(args)
    if args.submit_root.exists() and args.overwrite_submission:
        shutil.rmtree(args.submit_root)
    args.submit_root.mkdir(parents=True, exist_ok=True)
    for root in [args.provided_output_root, args.pred_root]:
        for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            dst = args.submit_root / video_dir.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(video_dir, dst)
    if args.zip_path.exists() and args.overwrite_submission:
        args.zip_path.unlink()
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for png in sorted(args.submit_root.rglob("*.png")):
            zf.write(png, png.relative_to(args.submit_root).as_posix())
    validate_submission_or_die(args)


def main() -> None:
    args = complete(parse_args())
    actions = collect_actions(args)
    videos = sorted(set(args.videos or actions.keys()))
    if not videos:
        raise SystemExit("no videos/actions selected")
    args.pred_root.mkdir(parents=True, exist_ok=True)
    build = import_sam2(args.sam2_root)
    predictor = build(args.model_cfg, str(args.checkpoint), device=args.device)
    results = []
    started = time.time()
    for video in videos:
        if video in actions:
            print(f"teach_boxes video={video} actions={sum(len(v) for v in actions[video].values())}", flush=True)
            results.append(run_video(predictor, args, video, actions[video]))
        else:
            results.append(copy_video_from_baseline(args, video))
    summary = {"method": "m13_teach_sam_boxes", "videos": len(videos), "action_count": sum(sum(len(v) for v in d.values()) for d in actions.values()), "elapsed_sec": round(time.time() - started, 2)}
    payload = {"summary": summary, "split_json": str(args.split_json), "baseline_root": str(args.baseline_root), "pred_root": str(args.pred_root), "merge_mode": args.merge_mode, "merge_radius": args.merge_radius, "propagate_direction": args.propagate_direction, "reverse_max_frames": args.reverse_max_frames, "clip_to_anchor_box": bool(args.clip_to_anchor_box), "clip_pad_frac": float(args.clip_pad_frac), "results": results}
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.make_submission:
        make_submission(args)
    print("teach_boxes_summary=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

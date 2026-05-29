#!/usr/bin/env python3
"""Run original SAM2 with a training-free reliability gate on non-conditioning memory writes.

M2 keeps SAM2's prediction path intact: every propagated frame still emits the
current mask logits.  The only intervention is after a non-conditioning frame is
predicted and encoded: if the predicted mask looks unreliable, we do not store
that frame in SAM2's ``non_cond_frame_outputs`` memory bank.  This targets the
autoregressive failure mode where a bad prediction becomes future memory.

This script intentionally does not depend on the M1/M1.1 post-processing paths.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Reuse the baseline I/O and submission helpers so M2 differs only in tracker logic.
from infer_mosev2_sam2 import (  # noqa: E402
    import_sam2,
    list_frames,
    load_first_annotation,
    logits_to_label,
    make_submission,
    save_label_png,
)


@dataclass
class MemoryGateConfig:
    enabled: bool = True
    mask_thr: float = 0.50
    perturb: float = 0.05
    obj_thr: float = 0.35
    iou_thr: float = 0.55  # reserved for future SAM2 internals exposing iou score
    stability_thr: float = 0.60
    min_area_pixels: float = 3.0
    max_area_frac: float = 0.80
    min_ratio: float = 0.20
    max_ratio: float = 4.0
    max_motion_px_floor: float = 40.0
    motion_area_scale: float = 3.0
    tiny_area_relaxed_motion: float = 0.001
    reference: str = "previous"  # previous | last_reliable_on_empty | last_reliable
    update_prev_on_unreliable: bool = True


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_workspace = here.parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=default_workspace)
    p.add_argument("--sam2-root", type=Path, default=None)
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--pred-root", type=Path, default=None)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--videos", nargs="*", default=None, help="Optional subset of video names")
    p.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu, ...")
    p.add_argument("--skip-existing", action="store_true", help="Skip a video if output count already matches frame count")
    p.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-video-to-cpu", dest="offload_video_to_cpu", action="store_false")
    p.add_argument("--offload-state-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-state-to-cpu", dest="offload_state_to_cpu", action="store_false")
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--no-zip", action="store_true", help="Build submission directory but skip zip creation")
    p.add_argument("--overwrite-submission", action="store_true")

    p.add_argument("--m2-disable-memory-gate", dest="m2_gate_enabled", action="store_false", default=True)
    p.add_argument("--m2-mask-thr", type=float, default=0.50)
    p.add_argument("--m2-perturb", type=float, default=0.05)
    p.add_argument("--m2-obj-thr", type=float, default=0.35)
    p.add_argument("--m2-iou-thr", type=float, default=0.55)
    p.add_argument("--m2-stability-thr", type=float, default=0.60)
    p.add_argument("--m2-min-area-pixels", type=float, default=3.0)
    p.add_argument("--m2-max-area-frac", type=float, default=0.80)
    p.add_argument("--m2-min-ratio", type=float, default=0.20)
    p.add_argument("--m2-max-ratio", type=float, default=4.0)
    p.add_argument("--m2-max-motion-px-floor", type=float, default=40.0)
    p.add_argument("--m2-motion-area-scale", type=float, default=3.0)
    p.add_argument("--m2-tiny-area-relaxed-motion", type=float, default=0.001)
    p.add_argument(
        "--m2-reference",
        choices=["previous", "last_reliable_on_empty", "last_reliable"],
        default="previous",
        help=(
            "Mask used for area/motion sanity. 'previous' matches the base M2 pseudo-code; "
            "'last_reliable_on_empty' falls back to the last memory-written mask when the previous output is empty."
        ),
    )
    p.add_argument(
        "--m2-no-update-prev-on-unreliable",
        dest="m2_update_prev_on_unreliable",
        action="store_false",
        default=True,
        help="If set, an unreliable frame does not become the next frame's prev_mask reference.",
    )
    p.add_argument("--m2-audit-json", type=Path, default=None, help="Optional aggregate JSON audit path for this process")
    p.add_argument("--m2-audit-dir", type=Path, default=None, help="Optional per-video JSON audit directory; safe for parallel workers")
    return p.parse_args()


def complete_paths(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.sam2_root = (args.sam2_root or ws / "5_19" / "sam2").resolve()
    args.checkpoint = (args.checkpoint or ws / "5_19" / "data" / "sam2" / "sam2.1_hiera_base_plus.pt").resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.pred_root = (args.pred_root or ws / "homework" / "pred_sam2_m2_memory_gate").resolve()
    args.submit_root = (args.submit_root or ws / "homework" / "submission_433_m2_memory_gate").resolve()
    args.zip_path = (args.zip_path or ws / "homework" / "submission_mosev2_m2_memory_gate.zip").resolve()
    if args.m2_audit_json is None:
        args.m2_audit_json = (ws / "homework" / "logs" / "m2_memory_gate_latest.json").resolve()
    else:
        args.m2_audit_json = args.m2_audit_json.resolve()
    if args.m2_audit_dir is None:
        args.m2_audit_dir = (ws / "homework" / "logs" / "m2_memory_gate_by_video").resolve()
    else:
        args.m2_audit_dir = args.m2_audit_dir.resolve()
    return args


def config_from_args(args: argparse.Namespace) -> MemoryGateConfig:
    return MemoryGateConfig(
        enabled=bool(args.m2_gate_enabled),
        mask_thr=float(args.m2_mask_thr),
        perturb=float(args.m2_perturb),
        obj_thr=float(args.m2_obj_thr),
        iou_thr=float(args.m2_iou_thr),
        stability_thr=float(args.m2_stability_thr),
        min_area_pixels=float(args.m2_min_area_pixels),
        max_area_frac=float(args.m2_max_area_frac),
        min_ratio=float(args.m2_min_ratio),
        max_ratio=float(args.m2_max_ratio),
        max_motion_px_floor=float(args.m2_max_motion_px_floor),
        motion_area_scale=float(args.m2_motion_area_scale),
        tiny_area_relaxed_motion=float(args.m2_tiny_area_relaxed_motion),
        reference=args.m2_reference,
        update_prev_on_unreliable=bool(args.m2_update_prev_on_unreliable),
    )


def _as_2d_logits(mask_logits):
    logits = mask_logits.detach()
    if logits.ndim == 4:
        logits = logits[0, 0]
    elif logits.ndim == 3:
        logits = logits[0]
    elif logits.ndim != 2:
        raise ValueError(f"Expected mask logits [1,1,H,W], [1,H,W], or [H,W], got {tuple(logits.shape)}")
    return logits.float()


def _iou_bool(a, b) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def _centroid(mask) -> tuple[float, float] | None:
    nz = mask.nonzero(as_tuple=False)
    if nz.numel() == 0:
        return None
    # nonzero returns (y, x).  Keep pixel coordinates in the current mask resolution.
    mean = nz.float().mean(dim=0)
    return float(mean[1].item()), float(mean[0].item())


def _distance(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float:
    if a is None or b is None:
        return float("inf")
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _obj_score_value(object_score_logits: Any | None) -> float | None:
    if object_score_logits is None:
        return None
    try:
        import torch

        if isinstance(object_score_logits, torch.Tensor):
            if object_score_logits.numel() == 0:
                return None
            return float(torch.sigmoid(object_score_logits.detach().float()).mean().item())
    except Exception:
        pass
    try:
        return float(object_score_logits)
    except Exception:
        return None


def _choose_reference(
    cfg: MemoryGateConfig,
    prev_mask,
    last_reliable_mask,
    min_area_pixels: float,
):
    if cfg.reference == "last_reliable":
        return last_reliable_mask, "last_reliable"
    if cfg.reference == "last_reliable_on_empty":
        if prev_mask is None or float(prev_mask.sum().item()) < min_area_pixels:
            return last_reliable_mask, "last_reliable"
        return prev_mask, "previous"
    return prev_mask, "previous"


def reliable_for_memory(
    mask_logits,
    prev_mask,
    last_reliable_mask,
    object_score_logits: Any | None,
    cfg: MemoryGateConfig,
) -> tuple[bool, dict[str, Any], object]:
    """Return reliability, audit metrics, and the binary current mask tensor."""
    import torch

    logits = _as_2d_logits(mask_logits)
    prob = torch.sigmoid(logits)
    mask = prob > cfg.mask_thr
    h, w = int(mask.shape[-2]), int(mask.shape[-1])
    numel = float(h * w)
    min_area_frac = float(cfg.min_area_pixels) / max(numel, 1.0)

    area_pixels = float(mask.sum().item())
    area_frac = area_pixels / max(numel, 1.0)

    mask_lo = prob > max(0.0, cfg.mask_thr - cfg.perturb)
    mask_hi = prob > min(1.0, cfg.mask_thr + cfg.perturb)
    stability = _iou_bool(mask_lo, mask_hi)

    ref_mask, reference_used = _choose_reference(cfg, prev_mask, last_reliable_mask, cfg.min_area_pixels)
    if ref_mask is None:
        prev_area_pixels = area_pixels
        prev_area_frac = area_frac
        area_ratio = 1.0
        displacement = 0.0
        ref_centroid = _centroid(mask)
    else:
        prev_area_pixels = float(ref_mask.sum().item())
        prev_area_frac = prev_area_pixels / max(numel, 1.0)
        if prev_area_pixels < 1.0:
            area_ratio = 1.0 if area_pixels < 1.0 else float("inf")
        else:
            area_ratio = area_pixels / max(prev_area_pixels, 1e-6)
        ref_centroid = _centroid(ref_mask)
        displacement = _distance(_centroid(mask), ref_centroid)

    max_motion_px = max(cfg.max_motion_px_floor, cfg.motion_area_scale * math.sqrt(max(prev_area_pixels, 0.0)))
    obj_score = _obj_score_value(object_score_logits)

    checks = {
        "obj_ok": True if obj_score is None else obj_score > cfg.obj_thr,
        "area_abs_ok": min_area_frac <= area_frac <= cfg.max_area_frac,
        "area_ratio_ok": cfg.min_ratio <= area_ratio <= cfg.max_ratio,
        "stable_ok": stability > cfg.stability_thr,
        "motion_ok": displacement < max_motion_px or area_frac < cfg.tiny_area_relaxed_motion,
    }
    reliable = bool(cfg.enabled and all(checks.values()))
    reasons = [name for name, ok in checks.items() if not ok]

    metrics: dict[str, Any] = {
        "reliable": reliable,
        "reasons": reasons,
        "reference": reference_used,
        "shape_hw": [h, w],
        "mask_thr": cfg.mask_thr,
        "area_pixels": area_pixels,
        "area_frac": area_frac,
        "min_area_frac": min_area_frac,
        "prev_area_pixels": prev_area_pixels,
        "prev_area_frac": prev_area_frac,
        "area_ratio": area_ratio if math.isfinite(area_ratio) else "inf",
        "stability": stability,
        "displacement_px": displacement if math.isfinite(displacement) else "inf",
        "max_motion_px": max_motion_px,
        "obj_score": obj_score,
        "checks": checks,
    }
    return reliable, metrics, mask.detach()


def _empty_video_audit(video_name: str, obj_ids: list[int], cfg: MemoryGateConfig) -> dict[str, Any]:
    return {
        "video": video_name,
        "config": asdict(cfg),
        "objects": {str(int(obj_id)): {"frames": [], "summary": {}} for obj_id in obj_ids},
        "summary": {},
    }


def _append_audit_record(predictor, obj_id: int, record: dict[str, Any]) -> None:
    video_name = getattr(predictor, "_m2_memory_gate_current_video", "unknown") or "unknown"
    audit = predictor._m2_memory_gate_audit.setdefault(video_name, {"objects": {}})
    obj_key = str(int(obj_id))
    audit.setdefault("objects", {}).setdefault(obj_key, {"frames": [], "summary": {}})
    audit["objects"][obj_key]["frames"].append(record)


def _finalize_video_audit(audit: dict[str, Any]) -> dict[str, Any]:
    total_noncond = 0
    total_written = 0
    total_skipped = 0
    reason_counts: dict[str, int] = {}
    per_obj = audit.get("objects", {})
    for obj_key, obj_data in per_obj.items():
        frames = obj_data.get("frames", [])
        noncond = [f for f in frames if f.get("storage") == "non_cond_frame_outputs"]
        written = [f for f in noncond if f.get("memory_written")]
        skipped = [f for f in noncond if not f.get("memory_written")]
        for rec in skipped:
            for reason in rec.get("reasons", []):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        obj_data["summary"] = {
            "frames_total": len(frames),
            "noncond_total": len(noncond),
            "memory_written": len(written),
            "memory_skipped": len(skipped),
            "skip_ratio": (len(skipped) / len(noncond)) if noncond else 0.0,
        }
        total_noncond += len(noncond)
        total_written += len(written)
        total_skipped += len(skipped)
    audit["summary"] = {
        "objects": len(per_obj),
        "noncond_total": total_noncond,
        "memory_written": total_written,
        "memory_skipped": total_skipped,
        "skip_ratio": (total_skipped / total_noncond) if total_noncond else 0.0,
        "skip_reason_counts": dict(sorted(reason_counts.items())),
    }
    return audit


def install_reliable_memory_gate(predictor, cfg: MemoryGateConfig, audit: dict[str, Any]) -> None:
    """Monkey-patch SAM2VideoPredictor.propagate_in_video for M2 memory gating."""
    import torch
    from tqdm import tqdm

    predictor._m2_memory_gate_cfg = cfg
    predictor._m2_memory_gate_audit = audit
    predictor._m2_memory_gate_current_video = None

    @torch.inference_mode()
    def propagate_in_video_m2(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate while skipping unreliable non-conditioning memory writes."""
        self.propagate_in_video_preflight(inference_state)

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        if start_frame_idx is None:
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        prev_masks = [None] * batch_size
        last_reliable_masks = [None] * batch_size

        for frame_idx in tqdm(processing_order, desc="propagate in video (m2 memory gate)"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                obj_id = int(obj_ids[obj_idx])
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    current_mask = (_as_2d_logits(pred_masks) > 0).detach()
                    prev_masks[obj_idx] = current_mask
                    last_reliable_masks[obj_idx] = current_mask
                    record = {
                        "frame_idx": int(frame_idx),
                        "storage": storage_key,
                        "reliable": True,
                        "memory_written": True,
                        "reasons": [],
                        "is_conditioning": True,
                    }
                    _append_audit_record(self, obj_id, record)
                    if self.clear_non_cond_mem_around_input:
                        self._clear_obj_non_cond_mem_around_input(inference_state, frame_idx, obj_idx)
                else:
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    if cfg.enabled:
                        reliable, metrics, current_mask = reliable_for_memory(
                            pred_masks,
                            prev_masks[obj_idx],
                            last_reliable_masks[obj_idx],
                            current_out.get("object_score_logits"),
                            cfg,
                        )
                    else:
                        current_mask = (_as_2d_logits(pred_masks) > 0).detach()
                        metrics = {
                            "reliable": True,
                            "reasons": [],
                            "reference": "disabled",
                        }
                        reliable = True

                    current_out["is_reliable_memory"] = bool(reliable)
                    if reliable:
                        obj_output_dict[storage_key][frame_idx] = current_out
                        last_reliable_masks[obj_idx] = current_mask
                    # Critical M2 intervention: unreliable current masks are still yielded,
                    # but are not inserted into non_cond_frame_outputs, so future SAM2
                    # memory attention cannot read them as mask memory or object pointers.
                    if reliable or cfg.update_prev_on_unreliable:
                        prev_masks[obj_idx] = current_mask

                    record = {
                        "frame_idx": int(frame_idx),
                        "storage": storage_key,
                        "reliable": bool(reliable),
                        "memory_written": bool(reliable),
                        "is_conditioning": False,
                        **metrics,
                    }
                    _append_audit_record(self, obj_id, record)

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}
                pred_masks_per_obj[obj_idx] = pred_masks

            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(inference_state, all_pred_masks)
            yield frame_idx, obj_ids, video_res_masks

    predictor.propagate_in_video = types.MethodType(propagate_in_video_m2, predictor)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_video(predictor, args: argparse.Namespace, cfg: MemoryGateConfig, video_name: str) -> dict[str, Any]:
    import torch

    video_dir = args.jpeg_root / video_name
    ann_dir = args.ann_root / video_name
    out_dir = args.pred_root / video_name
    frames = list_frames(video_dir)
    ann, palette = load_first_annotation(ann_dir)
    obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
    if not obj_ids:
        raise ValueError(f"No foreground object ids in {ann_dir / '00000.png'}")

    if args.skip_existing:
        existing = sorted(out_dir.glob("*.png"))
        if len(existing) == len(frames):
            return {"video": video_name, "status": "skipped", "frames": len(frames), "objects": obj_ids}

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    predictor._m2_memory_gate_current_video = video_name
    predictor._m2_memory_gate_audit[video_name] = _empty_video_audit(video_name, obj_ids, cfg)

    state = predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )
    predictor.reset_state(state)

    with torch.inference_mode():
        for obj_id in obj_ids:
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                mask=(ann == obj_id),
            )

    seen: set[int] = set()
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        for frame_idx, cur_obj_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_idx < 0 or frame_idx >= len(frames):
                raise IndexError(f"SAM2 returned frame_idx={frame_idx} outside 0..{len(frames)-1}")
            if frame_idx == 0:
                label = ann.copy()
            else:
                label = logits_to_label(mask_logits, [int(x) for x in cur_obj_ids])
            save_label_png(out_dir / f"{frames[frame_idx].stem}.png", label, palette)
            seen.add(int(frame_idx))

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
            raise RuntimeError(
                f"{video_name}: size mismatch at {outputs[sample_idx].name}: {out_img.size} vs {frame_img.size}"
            )

    audit = _finalize_video_audit(predictor._m2_memory_gate_audit[video_name])
    args.m2_audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.m2_audit_dir / f"{video_name}.json", audit)

    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result: dict[str, Any] = {"video": video_name, "status": "done", "frames": len(frames), "objects": obj_ids}
    result.update({f"m2_{k}": v for k, v in audit.get("summary", {}).items() if k != "skip_reason_counts"})
    return result


def aggregate_audit(audit_by_video: dict[str, Any], cfg: MemoryGateConfig, results: list[dict[str, Any]]) -> dict[str, Any]:
    total_noncond = 0
    total_written = 0
    total_skipped = 0
    reason_counts: dict[str, int] = {}
    videos: dict[str, Any] = {}
    for video_name in sorted(audit_by_video):
        audit = _finalize_video_audit(audit_by_video[video_name])
        videos[video_name] = audit
        summary = audit.get("summary", {})
        total_noncond += int(summary.get("noncond_total", 0))
        total_written += int(summary.get("memory_written", 0))
        total_skipped += int(summary.get("memory_skipped", 0))
        for reason, count in summary.get("skip_reason_counts", {}).items():
            reason_counts[reason] = reason_counts.get(reason, 0) + int(count)
    return {
        "method": "m2_reliable_memory_gate",
        "principle": "original SAM2 prediction path; skip unreliable non-conditioning memory writes",
        "config": asdict(cfg),
        "results": results,
        "summary": {
            "videos": len(videos),
            "noncond_total": total_noncond,
            "memory_written": total_written,
            "memory_skipped": total_skipped,
            "skip_ratio": (total_skipped / total_noncond) if total_noncond else 0.0,
            "skip_reason_counts": dict(sorted(reason_counts.items())),
        },
        "videos": videos,
    }


def main() -> None:
    args = complete_paths(parse_args())
    cfg = config_from_args(args)
    for required in [args.sam2_root, args.checkpoint, args.jpeg_root, args.ann_root]:
        if not required.exists():
            raise FileNotFoundError(required)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    build_sam2_video_predictor = import_sam2(args.sam2_root)
    print(f"workspace={args.workspace}")
    print(f"sam2_root={args.sam2_root}")
    print(f"checkpoint={args.checkpoint}")
    print(f"device={args.device} cuda_available={torch.cuda.is_available()} gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"m2_memory_gate_config={json.dumps(asdict(cfg), ensure_ascii=False, sort_keys=True)}", flush=True)

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=args.device)
    predictor.eval()
    audit_by_video: dict[str, Any] = {}
    install_reliable_memory_gate(predictor, cfg, audit_by_video)

    videos = args.videos or sorted(p.name for p in args.jpeg_root.iterdir() if p.is_dir())
    print(f"videos={len(videos)} {videos}", flush=True)
    started = time.time()
    results: list[dict[str, Any]] = []
    for idx, video_name in enumerate(videos, 1):
        t0 = time.time()
        print(f"[{idx}/{len(videos)}] {video_name} start", flush=True)
        result = run_video(predictor, args, cfg, video_name)
        result["seconds"] = round(time.time() - t0, 2)
        results.append(result)
        print(f"[{idx}/{len(videos)}] {video_name} {result}", flush=True)

    total_frames = sum(int(r["frames"]) for r in results if r["status"] in {"done", "skipped"})
    elapsed = time.time() - started
    aggregate = aggregate_audit(audit_by_video, cfg, results)
    aggregate["runtime"] = {"seconds": elapsed, "frames": total_frames, "fps": (total_frames / elapsed) if elapsed > 0 else None}
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        aggregate["runtime"].update(
            {
                "cuda_max_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "cuda_max_memory_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            }
        )
    write_json(args.m2_audit_json, aggregate)
    print(
        "inference_complete "
        f"videos={len(results)} frames={total_frames} seconds={elapsed:.2f} "
        f"m2_summary={json.dumps(aggregate['summary'], ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )
    if "cuda_max_memory_allocated_mib" in aggregate["runtime"]:
        print(
            "cuda_peak_mib "
            f"allocated={aggregate['runtime']['cuda_max_memory_allocated_mib']:.1f} "
            f"reserved={aggregate['runtime']['cuda_max_memory_reserved_mib']:.1f}",
            flush=True,
        )
    print(f"m2_audit_json={args.m2_audit_json}", flush=True)
    print(f"m2_audit_dir={args.m2_audit_dir}", flush=True)

    if args.make_submission:
        make_submission(args)


if __name__ == "__main__":
    main()

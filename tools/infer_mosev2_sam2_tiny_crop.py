#!/usr/bin/env python3
"""Generate training-free SAM2 tiny-crop candidate masks for MOSEv2.

This is a candidate-source generator, not a final submission method.  For each
small first-frame object, it builds a per-object crop video guided by an existing
mask source (usually SAM2 baseline), runs frozen SAM2 on the crop at higher
relative target resolution, and maps the crop prediction back to full-frame
coordinates.  Non-eligible objects remain empty so downstream selectors can use
this root as a candidate/evidence source without silently replacing baseline.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
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
class CropFrame:
    frame_idx: int
    frame_name: str
    source: str
    center: tuple[float, float]
    side: int
    box: tuple[int, int, int, int]
    guide_area: int


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
    p.add_argument("--guide-root", type=Path, default=None, help="Mask root used only for crop centers; default M11 cycle-gated SAM2 when available")
    p.add_argument("--out-pred-root", type=Path, required=True)
    p.add_argument("--work-root", type=Path, default=None)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, default=None)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--min-side", type=int, default=160)
    p.add_argument("--max-side", type=int, default=768)
    p.add_argument("--side-scale", type=float, default=9.0)
    p.add_argument("--eligible-area-frac", type=float, default=0.02)
    p.add_argument("--max-guide-area-ratio", type=float, default=10.0)
    p.add_argument("--velocity-damping", type=float, default=0.75)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--make-submission", action="store_true", help="Disabled for this candidate-only source; use M3 selector to build submissions")
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--overwrite-submission", action="store_true")
    p.add_argument("--no-zip", action="store_true")
    return p.parse_args()


def complete_args(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.sam2_root = (args.sam2_root or ws / "5_19" / "sam2").resolve()
    args.checkpoint = (args.checkpoint or ws / "5_19" / "data" / "sam2" / "sam2.1_hiera_base_plus.pt").resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.guide_root = (args.guide_root or ws / "homework" / "pred_sam2_m11_cycle").resolve()
    args.out_pred_root = args.out_pred_root.resolve()
    args.work_root = (args.work_root or ws / "homework" / "tiny_crop_work").resolve()
    args.audit_json = args.audit_json.resolve()
    if args.audit_dir is not None:
        args.audit_dir = args.audit_dir.resolve()
    if args.submit_root is None:
        args.submit_root = ws / "homework" / "submission_433_tiny_crop_candidate"
    args.submit_root = args.submit_root.resolve()
    if args.zip_path is None:
        args.zip_path = ws / "homework" / "submission_mosev2_tiny_crop_candidate.zip"
    args.zip_path = args.zip_path.resolve()
    return args


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(video_dir)
    return frames


def load_first_annotation(ann_dir: Path) -> tuple[np.ndarray, list[int] | None]:
    path = ann_dir / "00000.png"
    img = Image.open(path)
    palette = img.getpalette()
    arr = np.array(img)
    if arr.ndim != 2:
        raise ValueError(f"annotation must be single channel: {path}")
    return arr, palette


def load_label(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img)
    if arr.ndim != 2:
        raise ValueError(f"label must be single channel: {path}")
    return arr


def save_label_png(path: Path, label: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if label.dtype != np.uint8:
        label = label.astype(np.uint8)
    img = Image.fromarray(label, mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def bbox_and_centroid(mask: np.ndarray) -> tuple[list[int] | None, tuple[float, float] | None]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], (float(xs.mean()), float(ys.mean()))


def crop_box(center: tuple[float, float], side: int, width: int, height: int) -> tuple[int, int, int, int]:
    side = int(min(max(1, side), width, height))
    cx, cy = center
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = max(0, min(width - side, x0))
    y0 = max(0, min(height - side, y0))
    return x0, y0, x0 + side, y0 + side


def resize_crop(img: Image.Image, box: tuple[int, int, int, int], size: int, nearest: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return img.crop(box).resize((size, size), resample)


def import_sam2(sam2_root: Path):
    sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor


def plan_crop_sequence(
    *,
    frames: list[Path],
    ann: np.ndarray,
    obj_id: int,
    guide_root: Path,
    video: str,
    args: argparse.Namespace,
) -> tuple[list[CropFrame], np.ndarray]:
    init_mask = ann == obj_id
    init_bbox, init_center = bbox_and_centroid(init_mask)
    if init_bbox is None or init_center is None:
        raise ValueError(f"{video} id={obj_id}: empty init mask")
    x0, y0, x1, y1 = init_bbox
    init_area = int(init_mask.sum())
    init_side = max(x1 - x0, y1 - y0, math.sqrt(max(init_area, 1)))
    side = int(round(min(args.max_side, max(args.min_side, args.side_scale * init_side))))
    h, w = ann.shape
    plan: list[CropFrame] = []
    centers: list[tuple[float, float]] = []
    last_center = init_center
    prev_center: tuple[float, float] | None = None
    for idx, frame in enumerate(frames):
        source = "init" if idx == 0 else "prediction"
        guide_area = 0
        center = last_center
        if idx > 0:
            guide_path = guide_root / video / f"{frame.stem}.png"
            if guide_path.is_file():
                guide = load_label(guide_path)
                mask = guide == obj_id
                guide_area = int(mask.sum())
                _, guide_center = bbox_and_centroid(mask)
                if guide_center is not None and 0 < guide_area <= args.max_guide_area_ratio * max(init_area, 1):
                    center = guide_center
                    source = "guide"
                elif prev_center is not None:
                    vx = (last_center[0] - prev_center[0]) * args.velocity_damping
                    vy = (last_center[1] - prev_center[1]) * args.velocity_damping
                    center = (last_center[0] + vx, last_center[1] + vy)
                    source = "velocity"
                else:
                    source = "last"
            elif prev_center is not None:
                vx = (last_center[0] - prev_center[0]) * args.velocity_damping
                vy = (last_center[1] - prev_center[1]) * args.velocity_damping
                center = (last_center[0] + vx, last_center[1] + vy)
                source = "velocity_no_guide"
        box = crop_box(center, side, w, h)
        center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        plan.append(CropFrame(idx, frame.name, source, center, box[2] - box[0], box, guide_area))
        prev_center = last_center
        last_center = center
        centers.append(center)
    first_box = plan[0].box
    crop_mask = resize_crop(Image.fromarray(init_mask.astype(np.uint8) * 255), first_box, args.crop_size, nearest=True)
    crop_mask_arr = np.array(crop_mask) > 0
    return plan, crop_mask_arr


def build_crop_video(frames: list[Path], plan: list[CropFrame], crop_dir: Path, crop_size: int) -> None:
    if crop_dir.exists():
        shutil.rmtree(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    for frame, rec in zip(frames, plan, strict=True):
        img = Image.open(frame).convert("RGB")
        crop = resize_crop(img, rec.box, crop_size, nearest=False)
        crop.save(crop_dir / frame.name, quality=95)


def run_crop_object(predictor, args: argparse.Namespace, video: str, obj_id: int, frames: list[Path], plan: list[CropFrame], init_crop_mask: np.ndarray) -> tuple[dict[int, np.ndarray], list[int]]:
    import torch

    crop_dir = args.work_root / f"{video}_obj{obj_id}"
    build_crop_video(frames, plan, crop_dir, args.crop_size)
    state = predictor.init_state(video_path=str(crop_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
    predictor.reset_state(state)
    with torch.inference_mode():
        predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=init_crop_mask)
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if str(args.device).startswith("cuda") and torch.cuda.is_available() else contextlib.nullcontext()
    crop_masks: dict[int, np.ndarray] = {}
    missing_obj_frames: list[int] = []
    with torch.inference_mode(), autocast_ctx:
        for frame_idx, cur_obj_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_idx < 0 or frame_idx >= len(frames):
                raise RuntimeError(f"{video} id={obj_id}: bad frame_idx={frame_idx}")
            logits = mask_logits.detach().float().cpu().numpy()
            if logits.ndim == 4:
                logits = logits[:, 0]
            ids = [int(x) for x in cur_obj_ids]
            if obj_id in ids:
                k = ids.index(obj_id)
                crop_masks[int(frame_idx)] = logits[k] > 0
            else:
                # Never substitute a different propagated object under this id.
                # A missing expected id is an empty candidate for this frame.
                missing_obj_frames.append(int(frame_idx))
                crop_masks[int(frame_idx)] = np.zeros((args.crop_size, args.crop_size), dtype=bool)
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return crop_masks, missing_obj_frames


def map_crop_mask_to_full(mask_crop: np.ndarray, rec: CropFrame, full_shape: tuple[int, int]) -> np.ndarray:
    h, w = full_shape
    x0, y0, x1, y1 = rec.box
    side = x1 - x0
    img = Image.fromarray(mask_crop.astype(np.uint8) * 255)
    resized = np.array(img.resize((side, side), Image.Resampling.NEAREST)) > 0
    out = np.zeros((h, w), dtype=bool)
    out[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    return out


def run_video(predictor, args: argparse.Namespace, video: str) -> dict[str, Any]:
    frames = list_frames(args.jpeg_root / video)
    ann, palette = load_first_annotation(args.ann_root / video)
    h, w = ann.shape
    out_dir = args.out_pred_root / video
    if args.skip_existing and len(list(out_dir.glob("*.png"))) == len(frames):
        return {"video": video, "status": "skipped", "frames": len(frames)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.png"):
        p.unlink()
    obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
    eligible: list[int] = []
    object_audits: dict[str, Any] = {}
    per_frame_masks: dict[int, dict[int, np.ndarray]] = {i: {} for i in range(len(frames))}
    for obj_id in obj_ids:
        init_area = int((ann == obj_id).sum())
        area_frac = init_area / max(h * w, 1)
        object_audits[str(obj_id)] = {"init_area": init_area, "init_area_frac": area_frac, "eligible": area_frac <= args.eligible_area_frac}
        if area_frac <= args.eligible_area_frac:
            eligible.append(obj_id)
    for obj_id in eligible:
        plan, init_crop_mask = plan_crop_sequence(frames=frames, ann=ann, obj_id=obj_id, guide_root=args.guide_root, video=video, args=args)
        crop_masks, missing_obj_frames = run_crop_object(predictor, args, video, obj_id, frames, plan, init_crop_mask)
        nonempty = 0
        for idx, mask_crop in crop_masks.items():
            full_mask = map_crop_mask_to_full(mask_crop, plan[idx], ann.shape)
            if full_mask.any():
                nonempty += 1
            per_frame_masks[idx][obj_id] = full_mask
        object_audits[str(obj_id)].update({
            "crop_side": plan[0].side,
            "crop_size": args.crop_size,
            "nonempty_frames": nonempty,
            "missing_obj_frames": missing_obj_frames,
            "plan_sources": {src: sum(1 for r in plan if r.source == src) for src in sorted({r.source for r in plan})},
            "sample_plan": [asdict(r) for r in plan[:3] + plan[-3:]],
        })
    conflict_frames = 0
    for idx, frame in enumerate(frames):
        label = np.zeros(ann.shape, dtype=np.uint16)
        # Preserve first-frame GT for eligible objects only; this is a candidate source.
        for obj_id in sorted(per_frame_masks[idx]):
            mask = per_frame_masks[idx][obj_id]
            if idx == 0:
                mask = ann == obj_id
            overlap = mask & (label != 0)
            if overlap.any():
                conflict_frames += 1
                mask = mask & (label == 0)
            label[mask] = obj_id
        if label.max(initial=0) <= 255:
            label = label.astype(np.uint8)
        save_label_png(out_dir / f"{frame.stem}.png", label, palette)
    return {
        "video": video,
        "status": "done",
        "frames": len(frames),
        "objects": obj_ids,
        "eligible_objects": eligible,
        "eligible_count": len(eligible),
        "conflict_frames": conflict_frames,
        "object_audits": object_audits,
    }


def copy_tree_contents(src_root: Path, submit_root: Path) -> None:
    for video_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        shutil.copytree(video_dir, submit_root / video_dir.name, dirs_exist_ok=True)


def make_submission(args: argparse.Namespace) -> dict[str, Any]:
    if args.submit_root.exists() and args.overwrite_submission:
        shutil.rmtree(args.submit_root)
    args.submit_root.mkdir(parents=True, exist_ok=True)
    copy_tree_contents(args.provided_output_root, args.submit_root)
    copy_tree_contents(args.out_pred_root, args.submit_root)
    video_dirs = sorted(p for p in args.submit_root.iterdir() if p.is_dir())
    png_count = sum(1 for _ in args.submit_root.rglob("*.png"))
    if len(video_dirs) != 433 or png_count != 66526:
        raise RuntimeError(f"bad submission shape videos={len(video_dirs)} pngs={png_count}")
    payload = {"submit_root": str(args.submit_root), "video_dirs": len(video_dirs), "pngs": png_count}
    print(f"submission_dir={args.submit_root} videos={len(video_dirs)} pngs={png_count}")
    if not args.no_zip:
        if args.zip_path.exists():
            args.zip_path.unlink()
        with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(args.submit_root.rglob("*.png")):
                zf.write(path, path.relative_to(args.submit_root).as_posix())
        with zipfile.ZipFile(args.zip_path) as zf:
            bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"bad zip member {bad}")
        payload.update({"zip_path": str(args.zip_path), "zip_size": args.zip_path.stat().st_size})
        print(f"zip_path={args.zip_path} size={args.zip_path.stat().st_size}")
    return payload


def main() -> None:
    start = time.time()
    args = complete_args(parse_args())
    if args.make_submission:
        raise RuntimeError("tiny-crop outputs are candidate-only and intentionally omit non-eligible objects; merge/select via M3 before making a submission")
    for required in [args.sam2_root, args.checkpoint, args.jpeg_root, args.ann_root, args.guide_root]:
        if not required.exists():
            raise FileNotFoundError(required)
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    build_predictor = import_sam2(args.sam2_root)
    print(f"workspace={args.workspace}")
    print(f"guide_root={args.guide_root}")
    print(f"tiny_crop_config={json.dumps({k: getattr(args, k) for k in ['crop_size','min_side','max_side','side_scale','eligible_area_frac','max_guide_area_ratio']}, sort_keys=True)}")
    predictor = build_predictor(args.model_cfg, str(args.checkpoint), device=args.device)
    predictor.eval()
    args.out_pred_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    if args.audit_dir:
        args.audit_dir.mkdir(parents=True, exist_ok=True)
    videos = args.videos or sorted(p.name for p in args.jpeg_root.iterdir() if p.is_dir())
    results = []
    for i, video in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] {video} start", flush=True)
        res = run_video(predictor, args, video)
        print(f"[{i}/{len(videos)}] {video} {res}", flush=True)
        results.append(res)
        if args.audit_dir:
            (args.audit_dir / f"{video}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    submission = None
    summary = {
        "videos": len(results),
        "frames": sum(int(r.get("frames", 0)) for r in results),
        "eligible_objects": sum(int(r.get("eligible_count", 0)) for r in results),
        "conflict_frames": sum(int(r.get("conflict_frames", 0)) for r in results),
    }
    runtime = {"seconds": round(time.time() - start, 3)}
    if torch.cuda.is_available():
        runtime.update({
            "cuda_max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
            "cuda_max_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 1024 / 1024, 2),
        })
    payload = {
        "method": "m4_tiny_crop_candidate_source",
        "principle": "training-free SAM2 crop rerun for small first-frame objects; candidate source only",
        "config": {k: getattr(args, k) for k in ["crop_size", "min_side", "max_side", "side_scale", "eligible_area_frac", "max_guide_area_ratio", "velocity_damping"]},
        "provenance": {
            "git_sha": os.environ.get("CVMOSE_GIT_SHA"),
            "git_branch": os.environ.get("CVMOSE_GIT_BRANCH"),
            "git_dirty": os.environ.get("CVMOSE_GIT_DIRTY"),
            "git_diff_hash": os.environ.get("CVMOSE_GIT_DIFF_HASH"),
            "guide_root": str(args.guide_root),
        },
        "summary": summary,
        "runtime": runtime,
        "outputs": {"out_pred_root": str(args.out_pred_root)},
        "submission": submission,
        "results": results,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("tiny_crop_audit_json=" + str(args.audit_json))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    print("runtime=" + json.dumps(runtime, sort_keys=True))


if __name__ == "__main__":
    main()

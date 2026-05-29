#!/usr/bin/env python3
"""Run SAM2 video object segmentation for the MOSEv2 homework split.

Expected workspace layout:
  MOSEv2/
    5_19/sam2/                         # SAM2 repo
    5_19/data/sam2/sam2.1_hiera_base_plus.pt
    homework/JPEGImages/<video>/*.jpg
    homework/Annotations/<video>/00000.png
    homework/output/<418 provided videos>/*.png

The script writes one single-channel palette PNG per frame and can optionally
merge the provided 418 outputs plus the new 15 outputs into a Codabench zip.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


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
    return p.parse_args()


def complete_paths(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.sam2_root = (args.sam2_root or ws / "5_19" / "sam2").resolve()
    args.checkpoint = (args.checkpoint or ws / "5_19" / "data" / "sam2" / "sam2.1_hiera_base_plus.pt").resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.pred_root = (args.pred_root or ws / "homework" / "pred_sam2_b101").resolve()
    args.submit_root = (args.submit_root or ws / "homework" / "submission_433").resolve()
    args.zip_path = (args.zip_path or ws / "homework" / "submission_mosev2_sam2.zip").resolve()
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
        # MOSE labels in this homework are small object ids; uint8 is expected by the provided masks.
        label = label.astype(np.uint8)
    img = Image.fromarray(label, mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def logits_to_label(mask_logits, obj_ids: list[int], original_ann: np.ndarray | None = None) -> np.ndarray:
    """Convert SAM2 [N,1,H,W] logits into one label map.

    For each pixel, keep the object with the largest positive logit. Pixels whose
    best logit is <= 0 remain background. This avoids overlapping object masks.
    """
    import torch

    if isinstance(mask_logits, torch.Tensor):
        logits = mask_logits.detach().float().cpu().numpy()
    else:
        logits = np.asarray(mask_logits)
    if logits.ndim == 4:
        logits = logits[:, 0]
    if logits.ndim != 3:
        raise ValueError(f"Expected mask logits [N,H,W] or [N,1,H,W], got {logits.shape}")
    ids = np.asarray([int(x) for x in obj_ids], dtype=np.uint16)
    best_idx = np.argmax(logits, axis=0)
    best_score = np.max(logits, axis=0)
    out = np.zeros(logits.shape[1:], dtype=np.uint16)
    positive = best_score > 0
    out[positive] = ids[best_idx[positive]]
    if out.max(initial=0) <= 255:
        out = out.astype(np.uint8)
    return out


def import_sam2(sam2_root: Path):
    # Add the SAM2 repo root so Python imports the package at sam2_root/sam2.
    sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor


def run_video(predictor, args: argparse.Namespace, video_name: str) -> dict:
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
                label = ann.copy()  # preserve exact GT first frame and palette ids
            else:
                label = logits_to_label(mask_logits, [int(x) for x in cur_obj_ids])
            save_label_png(out_dir / f"{frames[frame_idx].stem}.png", label, palette)
            seen.add(int(frame_idx))

    missing = [i for i in range(len(frames)) if i not in seen]
    if missing:
        raise RuntimeError(f"{video_name}: missing predicted frames {missing[:20]} (total {len(missing)})")

    # Basic size/name validation.
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

    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"video": video_name, "status": "done", "frames": len(frames), "objects": obj_ids}


def make_submission(args: argparse.Namespace) -> None:
    if args.submit_root.exists() and args.overwrite_submission:
        shutil.rmtree(args.submit_root)
    args.submit_root.mkdir(parents=True, exist_ok=True)

    # Copy provided 418 outputs first, then overlay the newly predicted 15 videos.
    for src_root in [args.provided_output_root, args.pred_root]:
        if not src_root.is_dir():
            raise FileNotFoundError(src_root)
        for video_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
            dst = args.submit_root / video_dir.name
            shutil.copytree(video_dir, dst, dirs_exist_ok=True)

    video_dirs = sorted(p for p in args.submit_root.iterdir() if p.is_dir())
    png_count = sum(1 for _ in args.submit_root.rglob("*.png"))
    print(f"submission_dir={args.submit_root} videos={len(video_dirs)} pngs={png_count}", flush=True)
    if len(video_dirs) != 433:
        raise RuntimeError(f"Expected 433 video directories in submission, got {len(video_dirs)}")

    if args.no_zip:
        return
    if args.zip_path.exists():
        args.zip_path.unlink()
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(args.submit_root.rglob("*.png")):
            zf.write(path, path.relative_to(args.submit_root).as_posix())
    print(f"zip_path={args.zip_path} size={args.zip_path.stat().st_size}", flush=True)


def main() -> None:
    args = complete_paths(parse_args())
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

    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=args.device)
    predictor.eval()

    videos = args.videos or sorted(p.name for p in args.jpeg_root.iterdir() if p.is_dir())
    print(f"videos={len(videos)} {videos}", flush=True)
    started = time.time()
    results = []
    for idx, video_name in enumerate(videos, 1):
        t0 = time.time()
        print(f"[{idx}/{len(videos)}] {video_name} start", flush=True)
        result = run_video(predictor, args, video_name)
        result["seconds"] = round(time.time() - t0, 2)
        results.append(result)
        print(f"[{idx}/{len(videos)}] {video_name} {result}", flush=True)

    total_frames = sum(int(r["frames"]) for r in results if r["status"] in {"done", "skipped"})
    print(f"inference_complete videos={len(results)} frames={total_frames} seconds={time.time()-started:.2f}", flush=True)

    if args.make_submission:
        make_submission(args)


if __name__ == "__main__":
    main()

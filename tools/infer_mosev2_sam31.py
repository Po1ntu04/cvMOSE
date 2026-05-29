#!/usr/bin/env python3
"""Run MOSEv2 homework inference with SAM 3.1 using first-frame GT masks.

This script intentionally mirrors ``infer_mosev2_sam2.py`` for dataset/output
handling, while moving all SAM 3.1 private-mask plumbing into
``sam31_gt_mask_adapter.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from sam31_gt_mask_adapter import Sam31FrameOutput, Sam31GtMaskAdapter


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
    p.add_argument("--skip-existing", action="store_true", help="Skip if output count already matches frame count")
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--no-zip", action="store_true", help="Build submission directory but skip zip creation")
    p.add_argument("--overwrite-submission", action="store_true")
    p.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-video-to-cpu", dest="offload_video_to_cpu", action="store_false")
    p.add_argument("--offload-state-to-cpu", action="store_true", default=False)
    p.add_argument("--max-num-objects", type=int, default=64)
    p.add_argument("--multiplex-count", type=int, default=16)
    p.add_argument("--use-fa3", action="store_true", help="Enable FlashAttention3 kernels; default false for portability")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile; default false for smoke/reproducibility")
    p.add_argument("--async-loading-frames", action="store_true", default=False)
    p.add_argument("--trace-jsonl", type=Path, default=None, help="Per-frame prediction trace JSONL")
    p.add_argument("--metrics-json", type=Path, default=None, help="Run/video metrics JSON")
    return p.parse_args()


def complete_paths(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.sam3_root = (args.sam3_root or ws / "5_19" / "sam3").resolve()
    args.checkpoint = (args.checkpoint or ws / "5_19" / "data" / "sam3.1" / "sam3.1_multiplex.pt").resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.pred_root = (args.pred_root or ws / "homework" / "pred_sam31_b101").resolve()
    args.submit_root = (args.submit_root or ws / "homework" / "submission_433_sam31").resolve()
    args.zip_path = (args.zip_path or ws / "homework" / "submission_mosev2_sam31.zip").resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    logs = ws / "homework" / "logs"
    args.trace_jsonl = (args.trace_jsonl or logs / f"sam31_frame_trace_{stamp}.jsonl").resolve()
    args.metrics_json = (args.metrics_json or logs / f"sam31_metrics_{stamp}.json").resolve()
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


def sam31_output_to_label(output: Sam31FrameOutput, shape: tuple[int, int]) -> np.ndarray:
    label = np.zeros(shape, dtype=np.uint16)
    masks = output.out_binary_masks
    obj_ids = [int(x) for x in output.out_obj_ids.tolist()]
    if masks.size == 0 or not obj_ids:
        return label.astype(np.uint8)
    if masks.ndim != 3:
        raise ValueError(f"Expected out_binary_masks [N,H,W], got {masks.shape}")
    if masks.shape[1:] != shape:
        raise ValueError(f"SAM3.1 mask shape {masks.shape[1:]} != expected {shape}")

    # SAM3.1 postprocess already applies object-wise non-overlap.  If any
    # overlap remains, later masks overwrite earlier ones in score/object order.
    order = np.arange(len(obj_ids))
    if output.out_probs is not None and len(output.out_probs) == len(order):
        order = np.argsort(output.out_probs)  # high confidence overwrites low confidence
    for idx in order:
        label[masks[idx]] = obj_ids[int(idx)]
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


def frame_trace_record(video: str, frame_path: Path, output: Sam31FrameOutput, seconds: float) -> dict:
    masks = output.out_binary_masks
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(int).tolist() if masks.size else []
    rec = {
        "event": "frame",
        "video": video,
        "frame_index": int(output.frame_index),
        "frame_name": frame_path.name,
        "source": output.source,
        "seconds_since_video_start": round(seconds, 4),
        "object_ids": [int(x) for x in output.out_obj_ids.tolist()],
        "mask_areas": areas,
    }
    if output.out_probs is not None:
        rec["out_probs"] = [float(x) for x in np.asarray(output.out_probs).reshape(-1).tolist()]
    if output.raw_scores is not None:
        rec["raw_scores"] = [float(x) for x in np.asarray(output.raw_scores).reshape(-1).tolist()]
    return rec


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_video(adapter: Sam31GtMaskAdapter, args: argparse.Namespace, video_name: str) -> dict:
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

    video_started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    session_id = adapter.start_session(
        video_dir,
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )
    seen: set[int] = set()
    try:
        masks_by_obj_id = adapter.masks_from_label_map(ann, obj_ids)
        add_output = adapter.add_first_frame_masks(session_id, masks_by_obj_id, frame_idx=0)
        save_label_png(out_dir / f"{frames[0].stem}.png", ann.copy(), palette)
        seen.add(0)
        write_jsonl(args.trace_jsonl, frame_trace_record(video_name, frames[0], add_output, time.time() - video_started))

        for output in adapter.propagate_forward(session_id, start_frame_idx=0):
            frame_idx = int(output.frame_index)
            if frame_idx == 0:
                continue  # already saved exact GT first frame
            if frame_idx < 0 or frame_idx >= len(frames):
                raise IndexError(f"SAM3.1 returned frame_idx={frame_idx} outside 0..{len(frames)-1}")
            label = sam31_output_to_label(output, ann.shape)
            save_label_png(out_dir / f"{frames[frame_idx].stem}.png", label, palette)
            seen.add(frame_idx)
            write_jsonl(args.trace_jsonl, frame_trace_record(video_name, frames[frame_idx], output, time.time() - video_started))
    finally:
        adapter.close_session(session_id)
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
            raise RuntimeError(
                f"{video_name}: size mismatch at {outputs[sample_idx].name}: {out_img.size} vs {frame_img.size}"
            )

    seconds = time.time() - video_started
    return {
        "video": video_name,
        "status": "done",
        "frames": len(frames),
        "objects": obj_ids,
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
    for required in [args.sam3_root, args.checkpoint, args.jpeg_root, args.ann_root]:
        if not required.exists():
            raise FileNotFoundError(required)
    args.trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_jsonl.exists():
        args.trace_jsonl.unlink()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SAM3.1 mask adapter currently requires CUDA")

    print(f"workspace={args.workspace}")
    print(f"sam3_root={args.sam3_root}")
    print(f"checkpoint={args.checkpoint}")
    print(
        f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
        f"gpus={torch.cuda.device_count()} device={torch.cuda.get_device_name(0)}"
    )
    print(f"trace_jsonl={args.trace_jsonl}")
    print(f"metrics_json={args.metrics_json}")

    run_started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    adapter = Sam31GtMaskAdapter.build(
        sam3_root=args.sam3_root,
        checkpoint_path=args.checkpoint,
        use_fa3=args.use_fa3,
        compile_model=args.compile,
        max_num_objects=args.max_num_objects,
        multiplex_count=args.multiplex_count,
        async_loading_frames=args.async_loading_frames,
    )
    build_seconds = time.time() - run_started
    print(f"sam31_adapter_ready build_seconds={build_seconds:.2f}", flush=True)

    videos = args.videos or sorted(p.name for p in args.jpeg_root.iterdir() if p.is_dir())
    print(f"videos={len(videos)} {videos}", flush=True)
    results = []
    for idx, video_name in enumerate(videos, 1):
        t0 = time.time()
        print(f"[{idx}/{len(videos)}] {video_name} start", flush=True)
        write_jsonl(args.trace_jsonl, {"event": "video_start", "video": video_name, "index": idx, "total": len(videos)})
        result = run_video(adapter, args, video_name)
        result.setdefault("seconds", round(time.time() - t0, 3))
        results.append(result)
        write_jsonl(args.trace_jsonl, {"event": "video_done", **result})
        print(f"[{idx}/{len(videos)}] {video_name} {result}", flush=True)

    total_seconds = time.time() - run_started
    total_frames = sum(int(r["frames"]) for r in results if r["status"] in {"done", "skipped"})
    peak_alloc_result = max(
        (r for r in results if isinstance(r.get("cuda"), dict)),
        key=lambda r: int(r["cuda"].get("max_allocated_bytes", 0)),
        default=None,
    )
    peak_reserved_result = max(
        (r for r in results if isinstance(r.get("cuda"), dict)),
        key=lambda r: int(r["cuda"].get("max_reserved_bytes", 0)),
        default=None,
    )
    max_video_peak_cuda = {
        "max_allocated_bytes": int(peak_alloc_result["cuda"].get("max_allocated_bytes", 0)) if peak_alloc_result else None,
        "max_allocated_video": peak_alloc_result.get("video") if peak_alloc_result else None,
        "max_reserved_bytes": int(peak_reserved_result["cuda"].get("max_reserved_bytes", 0)) if peak_reserved_result else None,
        "max_reserved_video": peak_reserved_result.get("video") if peak_reserved_result else None,
    }

    metrics = {
        "model": "sam3.1_multiplex_gt_mask_adapter",
        "workspace": str(args.workspace),
        "sam3_root": str(args.sam3_root),
        "checkpoint": str(args.checkpoint),
        "build_seconds": round(build_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "videos": len(results),
        "frames": total_frames,
        "fps": round(total_frames / total_seconds, 3) if total_seconds > 0 else None,
        "cuda": cuda_memory_snapshot(),
        "max_video_peak_cuda": max_video_peak_cuda,
        "results": results,
        "trace_jsonl": str(args.trace_jsonl),
    }
    args.metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"inference_complete videos={len(results)} frames={total_frames} "
        f"seconds={total_seconds:.2f} metrics={args.metrics_json}",
        flush=True,
    )

    if args.make_submission:
        make_submission(args)


if __name__ == "__main__":
    main()

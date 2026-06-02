#!/usr/bin/env python3
"""Run SAM2Long as an external training-free candidate on MOSEv2 homework videos.

This adapter deliberately keeps SAM2Long out of the cvMOSE package tree: it
invokes the external repository's ``tools/vos_inference.py`` with absolute
MOSEv2 paths, then normalizes the outputs back to the homework label-PNG
contract and optionally builds a 433-video submission zip by copying the 418
provided outputs unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    default_workspace = here.parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=default_workspace)
    p.add_argument("--sam2long-root", type=Path, required=True)
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--pred-root", type=Path, default=None)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-pathway", type=int, default=3)
    p.add_argument("--iou-thre", type=float, default=0.3)
    p.add_argument("--uncertainty", type=float, default=1.0)
    p.add_argument("--score-thresh", type=float, default=0.0)
    p.add_argument("--apply-postprocessing", action="store_true")
    p.add_argument("--track-object-appearing-later-in-video", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--overwrite-submission", action="store_true")
    return p.parse_args()


def complete_paths(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.sam2long_root = args.sam2long_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.pred_root = (args.pred_root or ws / "homework" / "pred_m6_sam2long").resolve()
    args.submit_root = (args.submit_root or ws / "homework" / "submission_433_m6_sam2long").resolve()
    args.zip_path = (args.zip_path or ws / "homework" / "submission_mosev2_sam2long.zip").resolve()
    args.audit_json = args.audit_json.resolve()
    return args


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(f"No frames found in {video_dir}")
    return frames


def load_label(path: Path) -> tuple[np.ndarray, list[int] | None]:
    img = Image.open(path)
    return np.array(img), img.getpalette()


def save_label(path: Path, arr: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr, mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def label_stats(arr: np.ndarray, obj_ids: list[int]) -> dict[str, Any]:
    total = int(arr.size)
    return {
        "unique": [int(x) for x in np.unique(arr)],
        "empty_by_object": {str(obj_id): int(not np.any(arr == obj_id)) for obj_id in obj_ids},
        "area_by_object": {str(obj_id): int(np.sum(arr == obj_id)) for obj_id in obj_ids},
        "area_frac_by_object": {str(obj_id): float(np.sum(arr == obj_id) / max(total, 1)) for obj_id in obj_ids},
    }


def write_video_list(path: Path, videos: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(videos) + "\n", encoding="utf-8")


def run_external(args: argparse.Namespace, videos: list[str]) -> tuple[int, str, str, Path]:
    video_list = args.audit_json.with_suffix(".videos.txt")
    write_video_list(video_list, videos)
    cmd = [
        args.python,
        str(args.sam2long_root / "tools" / "vos_inference.py"),
        "--sam2_cfg",
        args.model_cfg,
        "--sam2_checkpoint",
        str(args.checkpoint),
        "--base_video_dir",
        str(args.jpeg_root),
        "--input_mask_dir",
        str(args.ann_root),
        "--video_list_file",
        str(video_list),
        "--output_mask_dir",
        str(args.pred_root),
        "--num_pathway",
        str(args.num_pathway),
        "--iou_thre",
        str(args.iou_thre),
        "--uncertainty",
        str(args.uncertainty),
        "--score_thresh",
        str(args.score_thresh),
    ]
    if args.apply_postprocessing:
        cmd.append("--apply_postprocessing")
    if args.track_object_appearing_later_in_video:
        cmd.append("--track_object_appearing_later_in_video")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.sam2long_root) + os.pathsep + env.get("PYTHONPATH", "")
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(args.sam2long_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.time() - started
    return proc.returncode, proc.stdout, proc.stderr, video_list


def normalize_and_audit_outputs(args: argparse.Namespace, videos: list[str]) -> list[dict[str, Any]]:
    per_video: list[dict[str, Any]] = []
    for video in videos:
        frames = list_frames(args.jpeg_root / video)
        ann, palette = load_label(args.ann_root / video / "00000.png")
        obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
        out_dir = args.pred_root / video
        if not out_dir.is_dir():
            raise RuntimeError(f"SAM2Long did not create output dir: {out_dir}")
        expected_names = [f"{f.stem}.png" for f in frames]
        got_names = sorted(p.name for p in out_dir.glob("*.png"))
        if got_names != expected_names:
            raise RuntimeError(f"{video}: output names/count mismatch expected={len(expected_names)} got={len(got_names)}")

        # The submission invariant requires exact first-frame GT, not a re-decoded tracker output.
        save_label(out_dir / expected_names[0], ann, palette)

        empty_counts = {str(obj_id): 0 for obj_id in obj_ids}
        invalid_labels: list[str] = []
        first_frame_ok = True
        for name in expected_names:
            arr, pal = load_label(out_dir / name)
            if name == expected_names[0]:
                first_frame_ok = bool(np.array_equal(arr, ann))
            allowed = set([0, *obj_ids])
            unique = {int(x) for x in np.unique(arr)}
            if not unique.issubset(allowed):
                invalid_labels.append(f"{name}:{sorted(unique - allowed)}")
            for obj_id in obj_ids:
                if not np.any(arr == obj_id):
                    empty_counts[str(obj_id)] += 1
            # Normalize palette so visual/zip tools get stable indexed PNGs.
            if pal is None and palette:
                save_label(out_dir / name, arr, palette)
        first_arr, _ = load_label(out_dir / expected_names[0])
        last_arr, _ = load_label(out_dir / expected_names[-1])
        per_video.append(
            {
                "video": video,
                "frames": len(frames),
                "objects": obj_ids,
                "pngs": len(got_names),
                "first_frame_preserved": first_frame_ok,
                "invalid_labels": invalid_labels[:20],
                "empty_counts": empty_counts,
                "first_frame_stats": label_stats(first_arr, obj_ids),
                "last_frame_stats": label_stats(last_arr, obj_ids),
            }
        )
    return per_video


def make_submission(args: argparse.Namespace) -> None:
    if args.submit_root.exists() and args.overwrite_submission:
        shutil.rmtree(args.submit_root)
    args.submit_root.mkdir(parents=True, exist_ok=True)
    for src_root in [args.provided_output_root, args.pred_root]:
        if not src_root.is_dir():
            raise FileNotFoundError(src_root)
        for video_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
            shutil.copytree(video_dir, args.submit_root / video_dir.name, dirs_exist_ok=True)
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
    for required in [args.sam2long_root, args.checkpoint, args.jpeg_root, args.ann_root, args.provided_output_root]:
        if not required.exists():
            raise FileNotFoundError(required)
    videos = args.videos or sorted(p.name for p in args.jpeg_root.iterdir() if p.is_dir())
    if not videos:
        raise RuntimeError("No videos selected")
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    if args.pred_root.exists():
        shutil.rmtree(args.pred_root)
    args.pred_root.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "sam2long_root": str(args.sam2long_root),
                "checkpoint": str(args.checkpoint),
                "pred_root": str(args.pred_root),
                "videos": videos,
                "num_pathway": args.num_pathway,
                "iou_thre": args.iou_thre,
                "uncertainty": args.uncertainty,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    started = time.time()
    code, stdout, stderr, video_list = run_external(args, videos)
    if code != 0:
        args.audit_json.write_text(
            json.dumps(
                {
                    "ok": False,
                    "stage": "external_run",
                    "returncode": code,
                    "stdout_tail": stdout[-8000:],
                    "stderr_tail": stderr[-8000:],
                    "video_list_file": str(video_list),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(stdout[-8000:])
        print(stderr[-8000:], file=sys.stderr)
        raise SystemExit(code)

    per_video = normalize_and_audit_outputs(args, videos)
    if args.make_submission:
        make_submission(args)
    audit = {
        "ok": True,
        "method": "sam2long",
        "git_sha": os.environ.get("CVMOSE_GIT_SHA"),
        "git_branch": os.environ.get("CVMOSE_GIT_BRANCH"),
        "sam2long_root": str(args.sam2long_root),
        "checkpoint": str(args.checkpoint),
        "pred_root": str(args.pred_root),
        "submit_root": str(args.submit_root) if args.make_submission else None,
        "zip_path": str(args.zip_path) if args.make_submission and not args.no_zip else None,
        "videos": videos,
        "params": {"num_pathway": args.num_pathway, "iou_thre": args.iou_thre, "uncertainty": args.uncertainty, "score_thresh": args.score_thresh},
        "seconds": round(time.time() - started, 2),
        "stdout_tail": stdout[-12000:],
        "stderr_tail": stderr[-12000:],
        "per_video": per_video,
        "summary": {
            "videos": len(per_video),
            "pngs": int(sum(v["pngs"] for v in per_video)),
            "first_frame_failures": [v["video"] for v in per_video if not v["first_frame_preserved"]],
            "invalid_label_videos": [v["video"] for v in per_video if v["invalid_labels"]],
        },
    }
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("audit_json=" + str(args.audit_json), flush=True)
    print("summary=" + json.dumps(audit["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

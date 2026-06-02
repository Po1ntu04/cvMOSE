#!/usr/bin/env python3
"""Thin non-vendored SAAS VOS adapter for MOSEv2 homework.

SAAS weights were not reliably available during M6, so this adapter is kept as a
ready smoke harness: it invokes an external SAAS checkout, normalizes frame-0 GT,
and can build a 433-video zip once a checkpoint is supplied.
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

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--saas-root", type=Path, required=True)
    p.add_argument("--model-cfg", default="configs/saas/saas_hiera_b+.yaml")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--per-obj-png-file", action="store_true")
    p.add_argument("--track-object-appearing-later-in-video", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--overwrite-submission", action="store_true")
    return p.parse_args()


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(video_dir)
    return frames


def load_label(path: Path) -> tuple[np.ndarray, list[int] | None]:
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr, img.getpalette()


def save_label(path: Path, arr: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr, mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def write_video_list(path: Path, videos: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(videos) + "\n", encoding="utf-8")


def make_submission(ws: Path, pred_root: Path, provided_root: Path, submit_root: Path, zip_path: Path, overwrite: bool) -> dict:
    if submit_root.exists() and overwrite:
        shutil.rmtree(submit_root)
    submit_root.mkdir(parents=True, exist_ok=True)
    for root in [provided_root, pred_root]:
        for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            shutil.copytree(video_dir, submit_root / video_dir.name, dirs_exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(submit_root.rglob("*.png")):
            zf.write(path, path.relative_to(submit_root).as_posix())
    return {"submit_root": str(submit_root), "zip_path": str(zip_path), "video_dirs": len([p for p in submit_root.iterdir() if p.is_dir()]), "pngs": sum(1 for _ in submit_root.rglob("*.png")), "zip_size": zip_path.stat().st_size}


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    provided_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    submit_root = (args.submit_root or ws / "homework" / "submission_433_m6_saas").resolve()
    zip_path = (args.zip_path or ws / "homework" / "submission_mosev2_saas.zip").resolve()
    videos = args.videos or sorted(p.name for p in jpeg_root.iterdir() if p.is_dir())
    for required in [args.saas_root, args.checkpoint, jpeg_root, ann_root, provided_root]:
        if not required.exists():
            raise FileNotFoundError(required)
    if args.pred_root.exists():
        shutil.rmtree(args.pred_root)
    args.pred_root.mkdir(parents=True, exist_ok=True)
    video_list = args.audit_json.with_suffix(".videos.txt")
    write_video_list(video_list, videos)
    cmd = [args.python, str(args.saas_root / "tools" / "vos_inference.py"), "--sam2_cfg", args.model_cfg, "--sam2_checkpoint", str(args.checkpoint), "--base_video_dir", str(jpeg_root), "--input_mask_dir", str(ann_root), "--video_list_file", str(video_list), "--output_mask_dir", str(args.pred_root)]
    if args.per_obj_png_file:
        cmd.append("--per_obj_png_file")
    if args.track_object_appearing_later_in_video:
        cmd.append("--track_object_appearing_later_in_video")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.saas_root) + os.pathsep + env.get("PYTHONPATH", "")
    started = time.time()
    proc = subprocess.run(cmd, cwd=str(args.saas_root), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps({"ok": False, "returncode": proc.returncode, "stdout_tail": proc.stdout[-8000:], "stderr_tail": proc.stderr[-8000:]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(proc.stdout[-8000:]); print(proc.stderr[-8000:], file=sys.stderr)
        raise SystemExit(proc.returncode)
    per_video = []
    for video in videos:
        frames = list_frames(jpeg_root / video)
        ann, palette = load_label(ann_root / video / "00000.png")
        out_dir = args.pred_root / video
        names = [f"{f.stem}.png" for f in frames]
        got = sorted(p.name for p in out_dir.glob("*.png")) if out_dir.is_dir() else []
        if got != names:
            raise RuntimeError(f"{video}: output mismatch expected={len(names)} got={len(got)}")
        save_label(out_dir / names[0], ann, palette)
        per_video.append({"video": video, "frames": len(frames), "objects": [int(x) for x in np.unique(ann) if int(x) != 0]})
    sub = make_submission(ws, args.pred_root, provided_root, submit_root, zip_path, args.overwrite_submission) if args.make_submission else None
    audit = {"ok": True, "method": "saas_adapter", "seconds": round(time.time() - started, 2), "videos": videos, "pred_root": str(args.pred_root), "submission": sub, "per_video": per_video, "stdout_tail": proc.stdout[-12000:], "stderr_tail": proc.stderr[-12000:]}
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_json": str(args.audit_json), "videos": len(per_video), "submission": sub}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate MOSEv2 homework submission invariants.

The validator is intentionally method-agnostic.  It checks the Codabench zip
and/or a submission directory built from the 418 provided outputs plus the 15
predicted homework videos.  The central invariant is that the 418 provided
videos are copied unchanged while only the 15 prediction videos may vary.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PngInfo:
    size: tuple[int, int]
    mode: str
    unique_values: list[int]
    sha256: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--pred-root", type=Path, default=None, help="Optional direct 15-video prediction root to validate")
    p.add_argument("--submit-root", type=Path, default=None, help="Optional expanded 433-video submission directory")
    p.add_argument("--zip-path", type=Path, default=None, help="Optional submission zip")
    p.add_argument("--target-ann-root", type=Path, default=None)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--expected-videos", type=int, default=433)
    p.add_argument("--expected-pngs", type=int, default=66526)
    p.add_argument("--expected-pred-videos", type=int, default=15)
    p.add_argument("--allow-array-identical-provided", action="store_true", help="Allow provided PNG byte differences if decoded arrays match")
    return p.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def image_info(data: bytes) -> PngInfo:
    with Image.open(io.BytesIO(data)) as img:
        arr = np.array(img)
        if arr.ndim != 2:
            raise ValueError(f"expected single-channel label PNG, got shape {arr.shape}")
        unique = [int(x) for x in np.unique(arr)]
        return PngInfo(size=img.size, mode=img.mode, unique_values=unique, sha256=sha256_bytes(data))


def list_video_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_pngs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.png") if p.is_file())


def zip_png_entries(zf: zipfile.ZipFile) -> list[str]:
    return sorted(name for name in zf.namelist() if name.lower().endswith(".png") and not name.endswith("/"))


def zip_video_dirs(entries: Iterable[str]) -> list[str]:
    return sorted({name.split("/", 1)[0] for name in entries if "/" in name})


def frame_stems(video_dir: Path) -> list[str]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(f"no frames in {video_dir}")
    return [p.stem for p in frames]


def annotation_ids(ann_root: Path, video: str) -> set[int]:
    ann_path = ann_root / video / "00000.png"
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)
    with Image.open(ann_path) as img:
        arr = np.array(img)
    return {int(x) for x in np.unique(arr)}


def frame_size(jpeg_root: Path, video: str, stem: str) -> tuple[int, int]:
    for suffix in (".jpg", ".jpeg", ".png"):
        p = jpeg_root / video / f"{stem}{suffix}"
        if p.is_file():
            with Image.open(p) as img:
                return img.size
    raise FileNotFoundError(f"missing RGB frame for {video}/{stem}")


def validate_submit_root(
    submit_root: Path,
    provided_root: Path,
    jpeg_root: Path,
    ann_root: Path,
    expected_videos: int,
    expected_pngs: int,
    expected_pred_videos: int,
    allow_array_identical_provided: bool,
) -> dict[str, Any]:
    video_dirs = list_video_dirs(submit_root)
    pngs = list_pngs(submit_root)
    errors: list[str] = []
    if len(video_dirs) != expected_videos:
        errors.append(f"submit-root expected {expected_videos} video dirs, got {len(video_dirs)}")
    if len(pngs) != expected_pngs:
        errors.append(f"submit-root expected {expected_pngs} PNGs, got {len(pngs)}")

    provided_videos = list_video_dirs(provided_root)
    if len(provided_videos) != 418:
        errors.append(f"provided-output-root expected 418 video dirs, got {len(provided_videos)}")
    provided_changed: list[str] = []
    provided_array_equal_only: list[str] = []
    for video in provided_videos:
        src_dir = provided_root / video
        dst_dir = submit_root / video
        if not dst_dir.is_dir():
            provided_changed.append(f"{video}:missing_dir")
            continue
        src_files = sorted(p.relative_to(src_dir).as_posix() for p in src_dir.rglob("*.png"))
        dst_files = sorted(p.relative_to(dst_dir).as_posix() for p in dst_dir.rglob("*.png"))
        if src_files != dst_files:
            provided_changed.append(f"{video}:file_set_mismatch")
            continue
        for rel in src_files:
            src_bytes = read_bytes(src_dir / rel)
            dst_bytes = read_bytes(dst_dir / rel)
            if src_bytes == dst_bytes:
                continue
            if allow_array_identical_provided:
                src_arr = np.array(Image.open(io.BytesIO(src_bytes)))
                dst_arr = np.array(Image.open(io.BytesIO(dst_bytes)))
                if src_arr.shape == dst_arr.shape and np.array_equal(src_arr, dst_arr):
                    provided_array_equal_only.append(f"{video}/{rel}")
                    continue
            provided_changed.append(f"{video}/{rel}:sha256_mismatch")
            break
    if provided_changed:
        errors.append(f"provided outputs changed: {provided_changed[:20]}")

    predicted_videos = list_video_dirs(jpeg_root)
    if len(predicted_videos) != expected_pred_videos:
        errors.append(f"jpeg-root expected {expected_pred_videos} predicted video dirs, got {len(predicted_videos)}")
    predicted_errors: list[str] = []
    for video in predicted_videos:
        expected_names = [f"{stem}.png" for stem in frame_stems(jpeg_root / video)]
        dst_dir = submit_root / video
        got_names = sorted(p.name for p in dst_dir.glob("*.png")) if dst_dir.is_dir() else []
        if got_names != expected_names:
            predicted_errors.append(f"{video}:names/count mismatch expected={len(expected_names)} got={len(got_names)}")
            continue
        allowed_ids = annotation_ids(ann_root, video)
        for name in expected_names:
            p = dst_dir / name
            info = image_info(read_bytes(p))
            if info.size != frame_size(jpeg_root, video, Path(name).stem):
                predicted_errors.append(f"{video}/{name}:size {info.size} != RGB")
                break
            if not set(info.unique_values).issubset(allowed_ids):
                predicted_errors.append(f"{video}/{name}:labels {info.unique_values} not subset {sorted(allowed_ids)}")
                break
    if predicted_errors:
        errors.append(f"predicted videos invalid: {predicted_errors[:20]}")

    return {
        "kind": "submit_root",
        "path": str(submit_root),
        "video_dirs": len(video_dirs),
        "pngs": len(pngs),
        "provided_videos": len(provided_videos),
        "provided_changed_count": len(provided_changed),
        "provided_changed_samples": provided_changed[:20],
        "provided_array_equal_only_count": len(provided_array_equal_only),
        "predicted_videos": len(predicted_videos),
        "predicted_error_count": len(predicted_errors),
        "predicted_error_samples": predicted_errors[:20],
        "ok": not errors,
        "errors": errors,
    }


def validate_zip(
    zip_path: Path,
    provided_root: Path,
    jpeg_root: Path,
    ann_root: Path,
    expected_videos: int,
    expected_pngs: int,
    expected_pred_videos: int,
    allow_array_identical_provided: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        entries = zip_png_entries(zf)
        video_dirs = zip_video_dirs(entries)
        names_set = set(entries)
        if bad is not None:
            errors.append(f"zipfile.testzip bad member: {bad}")
        if len(video_dirs) != expected_videos:
            errors.append(f"zip expected {expected_videos} video dirs, got {len(video_dirs)}")
        if len(entries) != expected_pngs:
            errors.append(f"zip expected {expected_pngs} PNGs, got {len(entries)}")

        provided_videos = list_video_dirs(provided_root)
        provided_changed: list[str] = []
        provided_array_equal_only: list[str] = []
        for video in provided_videos:
            src_dir = provided_root / video
            src_files = sorted(p.relative_to(src_dir).as_posix() for p in src_dir.rglob("*.png"))
            zip_files = sorted(name.split("/", 1)[1] for name in entries if name.startswith(f"{video}/"))
            if src_files != zip_files:
                provided_changed.append(f"{video}:file_set_mismatch")
                continue
            for rel in src_files:
                src_bytes = read_bytes(src_dir / rel)
                dst_bytes = zf.read(f"{video}/{rel}")
                if src_bytes == dst_bytes:
                    continue
                if allow_array_identical_provided:
                    src_arr = np.array(Image.open(io.BytesIO(src_bytes)))
                    dst_arr = np.array(Image.open(io.BytesIO(dst_bytes)))
                    if src_arr.shape == dst_arr.shape and np.array_equal(src_arr, dst_arr):
                        provided_array_equal_only.append(f"{video}/{rel}")
                        continue
                provided_changed.append(f"{video}/{rel}:sha256_mismatch")
                break
        if provided_changed:
            errors.append(f"provided outputs changed in zip: {provided_changed[:20]}")

        predicted_videos = list_video_dirs(jpeg_root)
        if len(predicted_videos) != expected_pred_videos:
            errors.append(f"jpeg-root expected {expected_pred_videos} predicted video dirs, got {len(predicted_videos)}")
        predicted_errors: list[str] = []
        for video in predicted_videos:
            expected_names = [f"{stem}.png" for stem in frame_stems(jpeg_root / video)]
            zip_names = sorted(name.split("/", 1)[1] for name in entries if name.startswith(f"{video}/"))
            if zip_names != expected_names:
                predicted_errors.append(f"{video}:names/count mismatch expected={len(expected_names)} got={len(zip_names)}")
                continue
            allowed_ids = annotation_ids(ann_root, video)
            for name in expected_names:
                entry = f"{video}/{name}"
                if entry not in names_set:
                    predicted_errors.append(f"{entry}:missing")
                    break
                info = image_info(zf.read(entry))
                if info.size != frame_size(jpeg_root, video, Path(name).stem):
                    predicted_errors.append(f"{entry}:size {info.size} != RGB")
                    break
                if not set(info.unique_values).issubset(allowed_ids):
                    predicted_errors.append(f"{entry}:labels {info.unique_values} not subset {sorted(allowed_ids)}")
                    break
        if predicted_errors:
            errors.append(f"predicted videos invalid in zip: {predicted_errors[:20]}")

    return {
        "kind": "zip",
        "path": str(zip_path),
        "size": zip_path.stat().st_size,
        "video_dirs": len(video_dirs),
        "pngs": len(entries),
        "testzip_bad_member": bad,
        "provided_videos": len(provided_videos),
        "provided_changed_count": len(provided_changed),
        "provided_changed_samples": provided_changed[:20],
        "provided_array_equal_only_count": len(provided_array_equal_only),
        "predicted_videos": len(predicted_videos),
        "predicted_error_count": len(predicted_errors),
        "predicted_error_samples": predicted_errors[:20],
        "ok": not errors,
        "errors": errors,
    }


def validate_pred_root(pred_root: Path, jpeg_root: Path, ann_root: Path, expected_pred_videos: int) -> dict[str, Any]:
    if pred_root is None:
        return {"kind": "pred_root", "ok": True, "skipped": True}
    if not pred_root.is_dir():
        return {"kind": "pred_root", "path": str(pred_root), "ok": False, "errors": ["missing pred root"]}
    errors: list[str] = []
    videos = list_video_dirs(jpeg_root)
    pred_videos = list_video_dirs(pred_root)
    if len(videos) != expected_pred_videos:
        errors.append(f"jpeg-root expected {expected_pred_videos} predicted video dirs, got {len(videos)}")
    if pred_videos != videos:
        errors.append(f"pred video set mismatch expected={videos} got={pred_videos}")
    png_count = 0
    for video in videos:
        expected_names = [f"{stem}.png" for stem in frame_stems(jpeg_root / video)]
        out_dir = pred_root / video
        got_names = sorted(p.name for p in out_dir.glob("*.png")) if out_dir.is_dir() else []
        png_count += len(got_names)
        if got_names != expected_names:
            errors.append(f"{video}:names/count mismatch expected={len(expected_names)} got={len(got_names)}")
            continue
        allowed_ids = annotation_ids(ann_root, video)
        for name in expected_names:
            info = image_info(read_bytes(out_dir / name))
            if info.size != frame_size(jpeg_root, video, Path(name).stem):
                errors.append(f"{video}/{name}:size {info.size} != RGB")
                break
            if not set(info.unique_values).issubset(allowed_ids):
                errors.append(f"{video}/{name}:labels {info.unique_values} not subset {sorted(allowed_ids)}")
                break
    return {
        "kind": "pred_root",
        "path": str(pred_root),
        "video_dirs": len(pred_videos),
        "pngs": png_count,
        "ok": not errors,
        "errors": errors[:20],
    }


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    provided_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    ann_root = (args.target_ann_root or ws / "homework" / "Annotations").resolve()
    submit_root = args.submit_root.resolve() if args.submit_root else None
    zip_path = args.zip_path.resolve() if args.zip_path else None
    pred_root = args.pred_root.resolve() if args.pred_root else None

    if submit_root is None and zip_path is None and pred_root is None:
        raise SystemExit("provide at least one of --submit-root, --zip-path, or --pred-root")

    results: dict[str, Any] = {
        "workspace": str(ws),
        "provided_output_root": str(provided_root),
        "jpeg_root": str(jpeg_root),
        "target_ann_root": str(ann_root),
        "expected_videos": args.expected_videos,
        "expected_pngs": args.expected_pngs,
        "expected_pred_videos": args.expected_pred_videos,
        "checks": [],
    }
    if pred_root is not None:
        results["checks"].append(validate_pred_root(pred_root, jpeg_root, ann_root, args.expected_pred_videos))
    if submit_root is not None:
        results["checks"].append(
            validate_submit_root(
                submit_root,
                provided_root,
                jpeg_root,
                ann_root,
                args.expected_videos,
                args.expected_pngs,
                args.expected_pred_videos,
                args.allow_array_identical_provided,
            )
        )
    if zip_path is not None:
        results["checks"].append(
            validate_zip(
                zip_path,
                provided_root,
                jpeg_root,
                ann_root,
                args.expected_videos,
                args.expected_pngs,
                args.expected_pred_videos,
                args.allow_array_identical_provided,
            )
        )

    ok = all(check.get("ok") for check in results["checks"])
    results["ok"] = ok
    text = json.dumps(results, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

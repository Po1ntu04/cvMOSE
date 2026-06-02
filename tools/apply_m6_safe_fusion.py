#!/usr/bin/env python3
"""Build M6 conservative fusion prediction roots and Codabench zips.

Fusion is intentionally policy-based, not pixel voting.  The default root is
copied for every target video (usually M11).  Optional replacements can swap an
entire video or a single object id from a candidate root after visual review.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_rule(text: str) -> tuple[str, int | None, Path, str]:
    # Forms: video=/path, video:obj=/path, video=name=/path is not supported to avoid ambiguity.
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"replacement must be video[::obj]=/path, got {text}")
    left, path = text.split("=", 1)
    if ":" in left:
        video, obj = left.split(":", 1)
        return video, int(obj), Path(path).expanduser(), text
    return left, None, Path(path).expanduser(), text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--default-root", type=Path, required=True)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--submit-root", type=Path, required=True)
    p.add_argument("--zip-path", type=Path, required=True)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--replace", action="append", default=[], help="video=/root or video:obj_id=/root; repeatable")
    p.add_argument("--note", action="append", default=[])
    p.add_argument("--overwrite", action="store_true")
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


def make_submission(workspace: Path, pred_root: Path, submit_root: Path, zip_path: Path, overwrite: bool) -> dict[str, Any]:
    provided_root = workspace / "homework" / "output"
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
    return {
        "submit_root": str(submit_root),
        "zip_path": str(zip_path),
        "video_dirs": len([p for p in submit_root.iterdir() if p.is_dir()]),
        "pngs": sum(1 for _ in submit_root.rglob("*.png")),
        "zip_size": zip_path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = ws / "homework" / "JPEGImages"
    ann_root = ws / "homework" / "Annotations"
    default_root = args.default_root.resolve()
    pred_root = args.pred_root.resolve()
    rules = [parse_rule(x) for x in args.replace]
    by_video: dict[str, list[tuple[int | None, Path, str]]] = {}
    for video, obj, root, raw in rules:
        by_video.setdefault(video, []).append((obj, root.resolve(), raw))
    if pred_root.exists() and args.overwrite:
        shutil.rmtree(pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "workspace": str(ws),
        "default_root": str(default_root),
        "pred_root": str(pred_root),
        "rules": args.replace,
        "notes": args.note,
        "videos": {},
    }
    for video_dir in sorted(p for p in jpeg_root.iterdir() if p.is_dir()):
        video = video_dir.name
        frames = list_frames(video_dir)
        ann, palette = load_label(ann_root / video / "00000.png")
        allowed = {int(x) for x in np.unique(ann)}
        obj_ids = [int(x) for x in sorted(allowed) if int(x) != 0]
        out_dir = pred_root / video
        out_dir.mkdir(parents=True, exist_ok=True)
        replacements = by_video.get(video, [])
        changed_frames = 0
        for frame_idx, frame in enumerate(frames):
            base, base_pal = load_label(default_root / video / f"{frame.stem}.png")
            label = base.copy()
            applied: list[str] = []
            if frame_idx == 0:
                label = ann.copy()
            else:
                for obj, root, raw in replacements:
                    src_path = root / video / f"{frame.stem}.png"
                    if not src_path.is_file():
                        continue
                    src, _ = load_label(src_path)
                    if src.shape != label.shape:
                        continue
                    if obj is None:
                        label = src.copy()
                        applied.append(raw)
                    else:
                        if obj not in obj_ids:
                            continue
                        label[label == obj] = 0
                        label[(label == 0) & (src == obj)] = obj
                        applied.append(raw)
            invalid = sorted({int(x) for x in np.unique(label)} - allowed)
            if invalid:
                raise RuntimeError(f"{video}/{frame.stem}: invalid labels {invalid}")
            if not np.array_equal(label, base):
                changed_frames += 1
            save_label(out_dir / f"{frame.stem}.png", label, palette or base_pal)
        audit["videos"][video] = {
            "frames": len(frames),
            "objects": obj_ids,
            "rules": [raw for _, _, raw in replacements],
            "changed_vs_default": changed_frames,
            "source": "default" if not replacements else "policy_replaced",
        }
    submit_info = make_submission(ws, pred_root, args.submit_root.resolve(), args.zip_path.resolve(), args.overwrite)
    audit["submission"] = submit_info
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_json": str(args.audit_json), **submit_info}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

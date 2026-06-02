#!/usr/bin/env python3
"""Generate M6 visual comparison sheets for all MOSEv2 target videos.

Each sheet uses real RGB frames and label overlays.  Columns:
RGB, SAM2 baseline, M11, candidate A/B/C, delta vs SAM2, and zoom crop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    1: np.array([255, 50, 45], dtype=np.float32),
    2: np.array([35, 220, 60], dtype=np.float32),
    3: np.array([45, 120, 255], dtype=np.float32),
    4: np.array([255, 220, 35], dtype=np.float32),
    5: np.array([255, 0, 220], dtype=np.float32),
}

DEEP_ZOOM_DEFAULT = {
    "r13u5z4y",
    "q0sizv6m",
    "msinig6m",
    "1qlssuz2",
    "2smf7uq9",
    "8jsm23a7",
    "lcgc29va",
    "4vznweiu",
    "4f98052b",
    "pe0d85lk",
    "z6dx46qr",
}


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = font(16)
SMALL = font(12)


def parse_named_root(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name.strip(), Path(path).expanduser()
    path = Path(text).expanduser()
    return path.name, path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--m11-root", type=Path, required=True)
    p.add_argument("--candidate-roots", nargs="*", default=[], help="Up to 3 roots as name=/path")
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--deep-zoom-videos", nargs="*", default=sorted(DEEP_ZOOM_DEFAULT))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--frames-per-video", type=int, default=9)
    p.add_argument("--thumb-width", type=int, default=260)
    p.add_argument("--zoom-width", type=int, default=260)
    return p.parse_args()


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(video_dir)
    return frames


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_label(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    if arr.shape != shape:
        return np.zeros(shape, dtype=np.uint8)
    return arr


def overlay(rgb: np.ndarray, label: np.ndarray) -> Image.Image:
    out = rgb.astype(np.float32).copy()
    for oid in [int(x) for x in np.unique(label) if int(x) != 0]:
        color = COLORS.get(oid, np.array([255, 0, 255], dtype=np.float32))
        mask = label == oid
        out[mask] = out[mask] * 0.42 + color * 0.58
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def delta(rgb: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> Image.Image:
    out = rgb.astype(np.float32) * 0.43
    same = (baseline > 0) & (baseline == candidate)
    removed = (baseline > 0) & (candidate == 0)
    added = (baseline == 0) & (candidate > 0)
    changed = (baseline > 0) & (candidate > 0) & (baseline != candidate)
    out[same] = out[same] * 0.35 + np.array([255, 255, 255]) * 0.65
    out[removed] = np.array([255, 80, 45])
    out[added] = np.array([40, 230, 80])
    out[changed] = np.array([70, 130, 255])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def resize(img: Image.Image, width: int) -> Image.Image:
    h = max(1, round(img.height * width / img.width))
    return img.resize((width, h), Image.Resampling.BILINEAR)


def annotate(img: Image.Image, title: str, sub: str = "") -> Image.Image:
    bar_h = 44 if sub else 26
    out = Image.new("RGB", (img.width, img.height + bar_h), "white")
    out.paste(img, (0, bar_h))
    d = ImageDraw.Draw(out)
    d.text((5, 3), title[:42], fill=(0, 0, 0), font=FONT)
    if sub:
        d.text((5, 25), sub[:80], fill=(40, 40, 40), font=SMALL)
    return out


def crop_box(labels: list[np.ndarray], pad: int = 40) -> tuple[int, int, int, int] | None:
    if not labels:
        return None
    union = np.zeros_like(labels[0], dtype=bool)
    for label in labels:
        union |= label > 0
    ys, xs = np.nonzero(union)
    if len(xs) == 0:
        return None
    h, w = union.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = max(0, x1 - pad), min(w, x2 + pad)
    y1, y2 = max(0, y1 - pad), min(h, y2 + pad)
    side = max(96, x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
    x2, y2 = min(w, x1 + side), min(h, y1 + side)
    x1, y1 = max(0, x2 - side), max(0, y2 - side)
    return (x1, y1, x2, y2)


def stack(rows: list[list[Image.Image]]) -> Image.Image:
    if not rows:
        return Image.new("RGB", (1, 1), "white")
    cols = max(len(r) for r in rows)
    widths = [max((r[c].width for r in rows if c < len(r)), default=1) for c in range(cols)]
    heights = [max(cell.height for cell in row) for row in rows]
    out = Image.new("RGB", (sum(widths), sum(heights)), "white")
    y = 0
    for row, rh in zip(rows, heights):
        x = 0
        for c, cell in enumerate(row):
            out.paste(cell, (x, y))
            x += widths[c]
        y += rh
    return out


def selected_frames(video: str, frames: list[Path], roots: list[tuple[str, Path]], shape: tuple[int, int], max_rows: int) -> list[int]:
    chosen = {0, len(frames) - 1}
    baseline_root = roots[0][1]
    for idx, frame in enumerate(frames):
        base = load_label(baseline_root / video / f"{frame.stem}.png", shape)
        if any(not np.array_equal(base, load_label(root / video / f"{frame.stem}.png", shape)) for _, root in roots[1:]):
            chosen.add(idx)
    for idx in np.linspace(0, len(frames) - 1, max_rows, dtype=int).tolist():
        chosen.add(int(idx))
    ordered = sorted(chosen)
    if len(ordered) > max_rows:
        keep = np.linspace(0, len(ordered) - 1, max_rows).round().astype(int)
        ordered = [ordered[int(i)] for i in keep]
    return ordered


def make_video(args: argparse.Namespace, video: str, roots: list[tuple[str, Path]], candidate_names: list[str]) -> dict[str, Any]:
    jpeg_root = args.workspace / "homework" / "JPEGImages"
    frames = list_frames(jpeg_root / video)
    first = load_rgb(frames[0])
    shape = first.shape[:2]
    idxs = selected_frames(video, frames, roots, shape, args.frames_per_video)
    rows: list[list[Image.Image]] = []
    labels = [name for name, _ in roots]
    for idx in idxs:
        frame = frames[idx]
        rgb = load_rgb(frame)
        shape = rgb.shape[:2]
        labels_by_root = [(name, load_label(root / video / f"{frame.stem}.png", shape)) for name, root in roots]
        baseline = labels_by_root[0][1]
        cells = [annotate(resize(Image.fromarray(rgb), args.thumb_width), "RGB", f"f={frame.stem}")]
        for name, lab in labels_by_root:
            cells.append(annotate(resize(overlay(rgb, lab), args.thumb_width), name, f"area={int((lab>0).sum())}"))
        delta_ref = labels_by_root[-1][1] if len(labels_by_root) > 1 else baseline
        cells.append(annotate(resize(delta(rgb, baseline, delta_ref), args.thumb_width), "delta last vs SAM2", "green add / red remove"))
        if video in set(args.deep_zoom_videos):
            box = crop_box([lab for _, lab in labels_by_root])
            if box is not None:
                x1, y1, x2, y2 = box
                zoom_img = Image.fromarray(rgb[y1:y2, x1:x2])
                cells.append(annotate(resize(zoom_img, args.zoom_width), "zoom crop", f"{x1},{y1},{x2},{y2}"))
        rows.append(cells)
    sheet = stack(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{video}_m6_compare.jpg"
    sheet.save(out, quality=92)
    return {"video": video, "frames": len(frames), "selected_frames": idxs, "columns": ["RGB", *labels, "delta last vs SAM2", "zoom"], "path": str(out)}


def main() -> None:
    args = parse_args()
    args.workspace = args.workspace.resolve()
    candidates = [parse_named_root(x) for x in args.candidate_roots][:3]
    roots = [("SAM2", args.baseline_root.resolve()), ("M11", args.m11_root.resolve()), *[(n, p.resolve()) for n, p in candidates]]
    videos = args.videos or sorted(p.name for p in (args.workspace / "homework" / "JPEGImages").iterdir() if p.is_dir())
    index = {
        "workspace": str(args.workspace),
        "roots": {name: str(path) for name, path in roots},
        "sheets": [make_video(args, video, roots, [n for n, _ in candidates]) for video in videos],
    }
    (args.out_dir / "sheet_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(index['sheets'])} sheets to {args.out_dir}")


if __name__ == "__main__":
    main()

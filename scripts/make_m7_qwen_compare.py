#!/usr/bin/env python3
"""Create lightweight visual compare sheets for M7 Qwen-VL verifier outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--baseline-root", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_b101"))
    p.add_argument("--m11-root", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_m11_cycle"))
    p.add_argument("--m5r-root", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_m5r_dino_key"))
    p.add_argument("--qwen-veto-root", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_m7_qwen_veto_key"))
    p.add_argument("--qwen-support-root", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_m7_qwen_support_key"))
    p.add_argument("--candidate-judgments-json", type=Path, default=Path("artifacts/m7_qwen_vl/candidate_judgments.json"))
    p.add_argument("--tracklet-judgments-json", type=Path, default=Path("artifacts/m7_qwen_vl/tracklet_judgments.json"))
    p.add_argument("--out-dir", type=Path, default=Path("docs/assets/m7_qwen_vl/key_compare"))
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--max-rows", type=int, default=8)
    return p.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

FONT = font(18); SMALL = font(14)


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
    img = rgb.astype(np.float32).copy()
    colors = {1: np.array([255, 40, 30], dtype=np.float32), 2: np.array([40, 220, 60], dtype=np.float32), 3: np.array([40, 120, 255], dtype=np.float32), 4: np.array([255, 220, 40], dtype=np.float32)}
    for oid in [int(x) for x in np.unique(label) if int(x) != 0]:
        mask = label == oid
        color = colors.get(oid, np.array([255, 0, 255], dtype=np.float32))
        img[mask] = img[mask] * 0.42 + color * 0.58
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def delta_panel(rgb: np.ndarray, base: np.ndarray, cand: np.ndarray) -> Image.Image:
    img = rgb.astype(np.float32).copy() * 0.45
    removed = (base > 0) & (cand == 0)
    added = (base == 0) & (cand > 0)
    changed = (base > 0) & (cand > 0) & (base != cand)
    same = (base > 0) & (base == cand)
    img[same] = img[same] * 0.4 + np.array([255, 255, 255]) * 0.6
    img[removed] = np.array([255, 80, 40]); img[added] = np.array([40, 220, 80]); img[changed] = np.array([80, 120, 255])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def fit(img: Image.Image, width: int = 280) -> Image.Image:
    scale = width / img.width
    return img.resize((width, max(1, int(img.height * scale))), Image.Resampling.BILINEAR)


def label(img: Image.Image, title: str, sub: str = "") -> Image.Image:
    bar = 48 if sub else 30
    out = Image.new("RGB", (img.width, img.height + bar), "white")
    out.paste(img, (0, bar)); d = ImageDraw.Draw(out)
    d.text((6, 4), title, fill=(0, 0, 0), font=FONT)
    if sub: d.text((6, 28), sub[:90], fill=(40, 40, 40), font=SMALL)
    return out


def stack(rows: list[list[Image.Image]]) -> Image.Image:
    if not rows:
        return Image.new("RGB", (800, 100), "white")
    col_w = [max(row[i].width for row in rows) for i in range(len(rows[0]))]
    row_h = [max(cell.height for cell in row) for row in rows]
    out = Image.new("RGB", (sum(col_w), sum(row_h)), "white")
    y = 0
    for r, row in enumerate(rows):
        x = 0
        for c, cell in enumerate(row):
            out.paste(cell, (x, y)); x += col_w[c]
        y += row_h[r]
    return out


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file(): return {}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}


def choose_frames(video: str, frames: list[Path], args: argparse.Namespace, candidate_json: dict[str, Any]) -> list[int]:
    chosen = {0}
    for rec in candidate_json.get("records", []):
        if rec.get("video") == video:
            f = int(rec.get("frame_idx", 0)); chosen.update([max(0, f-1), f, min(len(frames)-1, f+1)])
    # Add first/middle/last changed frames for Qwen roots when available.
    changed = []
    for i, frame in enumerate(frames):
        shape = load_rgb(frame).shape[:2]
        base = load_label(args.baseline_root / video / f"{frame.stem}.png", shape)
        for root in [args.qwen_veto_root, args.qwen_support_root]:
            lab = load_label(root / video / f"{frame.stem}.png", shape)
            if lab.shape == base.shape and not np.array_equal(base, lab):
                changed.append(i); break
    if changed:
        chosen.update([changed[0], changed[len(changed)//2], changed[-1]])
    out = sorted(x for x in chosen if 0 <= x < len(frames))
    if len(out) > args.max_rows:
        idxs = np.linspace(0, len(out) - 1, args.max_rows).round().astype(int)
        out = [out[int(i)] for i in idxs]
    return out


def main() -> None:
    args = parse_args(); jpeg_root = args.workspace / "homework" / "JPEGImages"; args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_json = read_json(args.candidate_judgments_json)
    videos = args.videos or sorted(p.name for p in jpeg_root.iterdir() if p.is_dir())
    metrics: dict[str, Any] = {"videos": {}}
    for video in videos:
        frames = sorted((jpeg_root / video).glob("*.jpg"))
        if not frames: frames = sorted((jpeg_root / video).glob("*.png"))
        if not frames: continue
        selected = choose_frames(video, frames, args, candidate_json)
        rows: list[list[Image.Image]] = []
        for idx in selected:
            frame = frames[idx]; rgb = load_rgb(frame); shape = rgb.shape[:2]
            base = load_label(args.baseline_root / video / f"{frame.stem}.png", shape)
            m11 = load_label(args.m11_root / video / f"{frame.stem}.png", shape)
            m5r = load_label(args.m5r_root / video / f"{frame.stem}.png", shape)
            veto = load_label(args.qwen_veto_root / video / f"{frame.stem}.png", shape)
            support = load_label(args.qwen_support_root / video / f"{frame.stem}.png", shape)
            sub = f"f={idx:05d} Δveto={int(np.count_nonzero(base != veto))} Δsupport={int(np.count_nonzero(base != support))}"
            rows.append([
                label(fit(Image.fromarray(rgb)), "RGB", sub),
                label(fit(overlay(rgb, base)), "SAM2 baseline"),
                label(fit(overlay(rgb, m11)), "M11"),
                label(fit(overlay(rgb, m5r)), "M5R-C/DINO"),
                label(fit(overlay(rgb, veto)), "M7 Qwen veto"),
                label(fit(overlay(rgb, support)), "M7 Qwen support"),
                label(fit(delta_panel(rgb, base, support)), "Δ support vs SAM2"),
            ])
        sheet = stack(rows)
        out = args.out_dir / f"{video}_m7_qwen_compare.jpg"
        sheet.save(out, quality=92)
        metrics["videos"][video] = {"selected_frames": selected, "sheet": str(out)}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(metrics['videos'])} M7 sheets to {args.out_dir}")


if __name__ == "__main__":
    main()

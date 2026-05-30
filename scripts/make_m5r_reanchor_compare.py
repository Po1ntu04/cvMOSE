#!/usr/bin/env python3
"""Create visual audit sheets for M5R-C re-anchor experiments.

Rows are selected around accepted anchors and changed frames.  Columns compare
RGB, SAM2 baseline, M11 safety gate, RAR-state scaffold, M5R-C, and M5R-vs-SAM2
delta.  This is intentionally lightweight and data-local; it does not require
remote GPUs.
"""
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
    p.add_argument("--audit-dir", type=Path, default=Path("artifacts/m5r_reanchor/key_by_video"))
    p.add_argument("--baseline-root", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_b101"))
    p.add_argument("--m11-root", type=Path, default=Path("artifacts/m5r_reanchor/source_preds/m11"))
    p.add_argument("--rar-root", type=Path, default=Path("artifacts/m5r_reanchor/source_preds/rar_state"))
    p.add_argument("--m5r-root", type=Path, default=Path("artifacts/m5r_reanchor/key_pred"))
    p.add_argument("--out-dir", type=Path, default=Path("docs/assets/m5r_reanchor/key_compare"))
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--max-rows", type=int, default=9)
    return p.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = font(18)
SMALL = font(14)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_label(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        return np.zeros(shape, dtype=np.uint8)
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr


def overlay(rgb: np.ndarray, label: np.ndarray) -> Image.Image:
    img = rgb.astype(np.float32).copy()
    colors = {
        1: np.array([255, 40, 30], dtype=np.float32),
        2: np.array([40, 220, 60], dtype=np.float32),
        3: np.array([40, 120, 255], dtype=np.float32),
        4: np.array([255, 220, 40], dtype=np.float32),
    }
    for oid in [int(x) for x in np.unique(label) if int(x) != 0]:
        color = colors.get(oid, np.array([255, 0, 255], dtype=np.float32))
        mask = label == oid
        img[mask] = img[mask] * 0.42 + color * 0.58
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def delta_panel(rgb: np.ndarray, base: np.ndarray, m5r: np.ndarray) -> Image.Image:
    img = rgb.astype(np.float32).copy() * 0.45
    removed = (base > 0) & (m5r == 0)
    added = (base == 0) & (m5r > 0)
    changed_id = (base > 0) & (m5r > 0) & (base != m5r)
    same = (base > 0) & (m5r == base)
    img[same] = img[same] * 0.4 + np.array([255, 255, 255]) * 0.6
    img[removed] = np.array([255, 80, 40])
    img[added] = np.array([40, 220, 80])
    img[changed_id] = np.array([80, 120, 255])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def fit_panel(img: Image.Image, width: int = 320) -> Image.Image:
    scale = width / img.width
    return img.resize((width, max(1, int(img.height * scale))), Image.Resampling.BILINEAR)


def add_label(img: Image.Image, title: str, sub: str = "") -> Image.Image:
    bar_h = 48 if sub else 30
    out = Image.new("RGB", (img.width, img.height + bar_h), "white")
    out.paste(img, (0, bar_h))
    d = ImageDraw.Draw(out)
    d.text((6, 4), title, fill=(0, 0, 0), font=FONT)
    if sub:
        d.text((6, 28), sub[:80], fill=(40, 40, 40), font=SMALL)
    return out


def stack_grid(rows: list[list[Image.Image]]) -> Image.Image:
    if not rows:
        return Image.new("RGB", (800, 200), "white")
    col_w = [max(row[c].width for row in rows) for c in range(len(rows[0]))]
    row_h = [max(cell.height for cell in row) for row in rows]
    out = Image.new("RGB", (sum(col_w), sum(row_h)), "white")
    y = 0
    for r, row in enumerate(rows):
        x = 0
        for c, cell in enumerate(row):
            out.paste(cell, (x, y))
            x += col_w[c]
        y += row_h[r]
    return out


def choose_frames(audit: dict[str, Any], frames: list[Path], baseline_root: Path, m5r_root: Path, video: str, max_rows: int) -> list[int]:
    chosen: set[int] = {0}
    for obj in audit.get("objects_audit", {}).values():
        for a in obj.get("anchors", []):
            f = int(a.get("frame_idx", 0))
            chosen.update([max(0, f - 2), f, min(len(frames) - 1, f + 2), min(len(frames) - 1, f + 8)])
        ev = obj.get("event_frames", [])
        if ev:
            chosen.update([int(ev[0]), int(ev[min(len(ev) - 1, len(ev) // 2)]), int(ev[-1])])
    changed = []
    for i, frame in enumerate(frames):
        b = load_label(baseline_root / video / f"{frame.stem}.png", (1, 1))
        m = load_label(m5r_root / video / f"{frame.stem}.png", b.shape)
        if b.shape == m.shape and not np.array_equal(b, m):
            changed.append(i)
    if changed:
        chosen.update([changed[0], changed[len(changed) // 2], changed[-1]])
    ordered = sorted(x for x in chosen if 0 <= x < len(frames))
    if len(ordered) > max_rows:
        # Keep temporal coverage with anchor/change bias already represented.
        idxs = np.linspace(0, len(ordered) - 1, max_rows).round().astype(int)
        ordered = [ordered[int(i)] for i in idxs]
    return ordered


def main() -> None:
    args = parse_args()
    jpeg_root = args.workspace / "homework" / "JPEGImages"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos = args.videos or sorted(p.stem for p in args.audit_dir.glob("*.json"))
    metrics: dict[str, Any] = {"videos": {}}
    for video in videos:
        audit_path = args.audit_dir / f"{video}.json"
        if not audit_path.is_file():
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        frames = sorted((jpeg_root / video).glob("*.jpg"))
        if not frames:
            continue
        selected = choose_frames(audit, frames, args.baseline_root, args.m5r_root, video, args.max_rows)
        rows: list[list[Image.Image]] = []
        header = Image.new("RGB", (320 * 6, 70), "white")
        d = ImageDraw.Draw(header)
        summary = audit.get("summary", {})
        d.text((8, 8), f"{video}: anchors={summary.get('accepted_anchor_count')} changed={summary.get('changed_vs_baseline')} descriptor={audit.get('descriptor')}", fill=(0, 0, 0), font=FONT)
        object_line = []
        for obj_id, obj in audit.get("objects_audit", {}).items():
            anchors = obj.get("anchors", [])
            anchor_txt = ",".join(f"obj{obj_id}@{a['frame_idx']}:{a['source']} m={a['margin']}" for a in anchors) or f"obj{obj_id}:none"
            object_line.append(anchor_txt)
        d.text((8, 38), " | ".join(object_line)[:190], fill=(30, 30, 30), font=SMALL)
        rows.append([header])
        for idx in selected:
            frame = frames[idx]
            rgb = load_rgb(frame)
            shape = rgb.shape[:2]
            base = load_label(args.baseline_root / video / f"{frame.stem}.png", shape)
            m11 = load_label(args.m11_root / video / f"{frame.stem}.png", shape)
            rar = load_label(args.rar_root / video / f"{frame.stem}.png", shape)
            m5r = load_label(args.m5r_root / video / f"{frame.stem}.png", shape)
            b_area = int((base > 0).sum())
            m_area = int((m5r > 0).sum())
            sub = f"f={idx:05d} base_area={b_area} m5r_area={m_area} Δpx={int(np.count_nonzero(base != m5r))}"
            row = [
                add_label(fit_panel(Image.fromarray(rgb)), "RGB", sub),
                add_label(fit_panel(overlay(rgb, base)), "SAM2 baseline"),
                add_label(fit_panel(overlay(rgb, m11)), "M11 safety"),
                add_label(fit_panel(overlay(rgb, rar)), "RAR-state"),
                add_label(fit_panel(overlay(rgb, m5r)), "M5R-C"),
                add_label(fit_panel(delta_panel(rgb, base, m5r)), "Δ M5R vs SAM2", "green add / red remove / white same"),
            ]
            rows.append(row)
        # Header row has one cell; expand it to width by placing independently.
        body = stack_grid(rows[1:])
        final = Image.new("RGB", (max(body.width, header.width), header.height + body.height), "white")
        final.paste(header, (0, 0))
        final.paste(body, (0, header.height))
        out = args.out_dir / f"{video}_m5r_compare.jpg"
        final.save(out, quality=92)
        metrics["videos"][video] = {
            "selected_frames": selected,
            "accepted_anchor_count": summary.get("accepted_anchor_count"),
            "changed_vs_baseline": summary.get("changed_vs_baseline"),
            "sheet": str(out),
        }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(metrics['videos'])} sheets to {args.out_dir}")


if __name__ == "__main__":
    main()

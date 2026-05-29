#!/usr/bin/env python3
"""Create M3 comparison sheets for real visual inspection.

Rows per sampled frame: RGB, baseline, M2-light, M11, SAM3 adapter (if present), M3.
The script intentionally uses actual RGB frames and masks, not only mask stats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {
    1: (0, 255, 0),
    2: (255, 64, 64),
    3: (64, 128, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--m3-root", type=Path, required=True)
    p.add_argument("--m3-audit-json", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, default=None)
    p.add_argument("--m2-light-root", type=Path, default=None)
    p.add_argument("--m11-root", type=Path, default=None)
    p.add_argument("--sam31-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--frames-per-video", type=int, default=8)
    p.add_argument("--thumb-width", type=int, default=240)
    p.add_argument("--zoom", action="store_true", help="Generate crop/zoom sheets in addition to full-frame sheets")
    return p.parse_args()


def list_frames(video_dir: Path) -> list[Path]:
    return sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_label(path: Path | None, shape: tuple[int, int]) -> np.ndarray:
    if path is None or not path.is_file():
        return np.zeros(shape, dtype=np.uint8)
    with Image.open(path) as img:
        return np.array(img)


def overlay_mask(rgb: Image.Image, label: np.ndarray, alpha: float = 0.48) -> Image.Image:
    base = rgb.convert("RGBA")
    arr = np.array(base)
    mask_arr = np.zeros_like(arr)
    ids = [int(x) for x in np.unique(label) if int(x) != 0]
    for oid in ids:
        color = COLORS.get(oid, (255, 255, 0))
        m = label == oid
        mask_arr[m, :3] = color
        mask_arr[m, 3] = int(255 * alpha)
    out = Image.alpha_composite(base, Image.fromarray(mask_arr, mode="RGBA")).convert("RGB")
    return out


def draw_text(img: Image.Image, text: str, fill=(255, 255, 255), bg=(0, 0, 0)) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    lines = []
    for raw in text.split("\n"):
        while len(raw) > 42:
            lines.append(raw[:42])
            raw = raw[42:]
        lines.append(raw)
    y = 3
    for line in lines[:5]:
        box = d.textbbox((3, y), line, font=font)
        d.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=bg)
        d.text((3, y), line, fill=fill, font=font)
        y += box[3] - box[1] + 3
    return out


def resize_keep(img: Image.Image, width: int) -> Image.Image:
    w, h = img.size
    new_h = max(1, int(round(h * width / w)))
    return img.resize((width, new_h), Image.Resampling.BILINEAR)


def frame_decisions(video_audit: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for fr in video_audit.get("frames", []):
        idx = int(fr.get("frame_idx", -1))
        if idx < 0:
            continue
        if idx == 0:
            out[idx] = "GT"
            continue
        parts = []
        for obj_s, obj in fr.get("objects", {}).items():
            parts.append(f"id{obj_s}:{obj.get('selected_source')}:{obj.get('output_class')}")
        if fr.get("conflicts"):
            parts.append("conflict")
        out[idx] = " | ".join(parts)
    return out


def selected_frame_indices(video_audit: dict[str, Any], total: int, n: int) -> list[int]:
    must = {0, max(0, total - 1)}
    interesting = []
    for fr in video_audit.get("frames", []):
        idx = int(fr.get("frame_idx", -1))
        if idx <= 0:
            continue
        objs = fr.get("objects", {})
        if any(obj.get("selected_source") not in {"baseline", None} or obj.get("output_class") in {"reject", "empty"} for obj in objs.values()):
            interesting.append(idx)
    # Spread out interesting frames first, then uniform fill.
    for idx in interesting[:: max(1, len(interesting) // max(n - len(must), 1))]:
        must.add(idx)
        if len(must) >= n:
            break
    for idx in np.linspace(0, max(0, total - 1), num=n, dtype=int).tolist():
        must.add(int(idx))
        if len(must) >= n:
            break
    return sorted(must)[:n]


def crop_box_for_labels(labels: list[np.ndarray], pad: int = 24) -> tuple[int, int, int, int] | None:
    union = np.zeros_like(labels[0], dtype=bool)
    for lab in labels:
        union |= lab != 0
    ys, xs = np.nonzero(union)
    if len(xs) == 0:
        return None
    h, w = union.shape
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    # Make box at least 80px and roughly square.
    bw, bh = x1 - x0, y1 - y0
    target = max(80, bw, bh)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0 = max(0, cx - target // 2); x1 = min(w, x0 + target); x0 = max(0, x1 - target)
    y0 = max(0, cy - target // 2); y1 = min(h, y0 + target); y0 = max(0, y1 - target)
    return (x0, y0, x1, y1)


def paste_grid(cells: list[list[Image.Image]], row_labels: list[str], col_labels: list[str]) -> Image.Image:
    if not cells or not cells[0]:
        raise ValueError("empty grid")
    cell_w = max(img.width for row in cells for img in row)
    cell_h = max(img.height for row in cells for img in row)
    left = 110
    top = 32
    out = Image.new("RGB", (left + cell_w * len(cells[0]), top + cell_h * len(cells)), (245, 245, 245))
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for j, label in enumerate(col_labels):
        d.text((left + j * cell_w + 4, 8), label, fill=(0, 0, 0), font=font)
    for i, label in enumerate(row_labels):
        d.text((4, top + i * cell_h + 6), label, fill=(0, 0, 0), font=font)
    for i, row in enumerate(cells):
        for j, img in enumerate(row):
            canvas = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
            canvas.paste(img, (0, 0))
            out.paste(canvas, (left + j * cell_w, top + i * cell_h))
    return out


def make_video_sheet(args: argparse.Namespace, video: str, video_audit: dict[str, Any]) -> dict[str, Any]:
    jpeg_root = args.workspace / "homework" / "JPEGImages"
    frames = list_frames(jpeg_root / video)
    idxs = selected_frame_indices(video_audit, len(frames), args.frames_per_video)
    roots = [
        ("RGB", None),
        ("SAM2", args.baseline_root),
        ("M2-light", args.m2_light_root),
        ("M11", args.m11_root),
        ("SAM3", args.sam31_root),
        ("M3", args.m3_root),
    ]
    roots = [(name, root) for name, root in roots if root is None or root.is_dir()]
    decisions = frame_decisions(video_audit)
    rows: list[list[Image.Image]] = []
    row_labels: list[str] = []
    for name, root in roots:
        row: list[Image.Image] = []
        for idx in idxs:
            rgb = load_rgb(frames[idx])
            shape = (rgb.height, rgb.width)
            if root is None:
                img = rgb
            else:
                label = load_label(root / video / f"{frames[idx].stem}.png", shape)
                img = overlay_mask(rgb, label)
            img = resize_keep(img, args.thumb_width)
            if name == "M3":
                img = draw_text(img, decisions.get(idx, ""))
            row.append(img)
        rows.append(row)
        row_labels.append(name)
    sheet = paste_grid(rows, row_labels, [frames[i].stem for i in idxs])
    out_full = args.out_dir / "full" / f"{video}_m3_compare.jpg"
    out_full.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_full, quality=92)

    out_zoom = None
    if args.zoom:
        z_rows: list[list[Image.Image]] = []
        z_labels: list[str] = []
        for name, root in roots:
            zrow: list[Image.Image] = []
            for idx in idxs:
                rgb = load_rgb(frames[idx])
                shape = (rgb.height, rgb.width)
                label_list = []
                for _, r in roots:
                    if r is not None and r.is_dir():
                        label_list.append(load_label(r / video / f"{frames[idx].stem}.png", shape))
                box = crop_box_for_labels(label_list) if label_list else None
                if root is None:
                    img = rgb
                else:
                    label = load_label(root / video / f"{frames[idx].stem}.png", shape)
                    img = overlay_mask(rgb, label)
                if box is not None:
                    img = img.crop(box)
                img = img.resize((args.thumb_width, args.thumb_width), Image.Resampling.BILINEAR)
                if name == "M3":
                    img = draw_text(img, decisions.get(idx, ""))
                zrow.append(img)
            z_rows.append(zrow)
            z_labels.append(name)
        zoom_sheet = paste_grid(z_rows, z_labels, [frames[i].stem for i in idxs])
        out_zoom = args.out_dir / "zooms" / f"{video}_m3_zoom.jpg"
        out_zoom.parent.mkdir(parents=True, exist_ok=True)
        zoom_sheet.save(out_zoom, quality=92)
    return {"video": video, "frames": [int(i) for i in idxs], "full_sheet": str(out_full), "zoom_sheet": str(out_zoom) if out_zoom else None}


def main() -> None:
    args = parse_args()
    args.workspace = args.workspace.resolve()
    args.m3_root = args.m3_root.resolve()
    args.m3_audit_json = args.m3_audit_json.resolve()
    args.out_dir = args.out_dir.resolve()
    args.baseline_root = (args.baseline_root or args.workspace / "homework" / "pred_sam2_b101").resolve()
    if args.m2_light_root is not None: args.m2_light_root = args.m2_light_root.resolve()
    if args.m11_root is not None: args.m11_root = args.m11_root.resolve()
    if args.sam31_root is not None: args.sam31_root = args.sam31_root.resolve()
    data = json.loads(args.m3_audit_json.read_text(encoding="utf-8"))
    result_by_video = {item["video"]: item for item in data.get("results", [])}
    videos = args.videos or sorted(result_by_video)
    summaries = [make_video_sheet(args, v, result_by_video[v]) for v in videos if v in result_by_video]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sheet_index.json").write_text(json.dumps({"summary": data.get("summary", {}), "sheets": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("sheet_index=" + str(args.out_dir / "sheet_index.json"))


if __name__ == "__main__":
    main()

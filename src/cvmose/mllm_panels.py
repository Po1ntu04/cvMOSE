"""Panel renderers for M7 Qwen-VL verifier.

The MLLM is only a verifier.  These helpers standardize visual evidence so
Qwen-VL sees the same reference/candidate layout across target profiling,
single-frame candidate judging, and multi-frame tracklet judging.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GREEN = (30, 220, 80)
RED = (230, 50, 50)
YELLOW = (255, 215, 0)
BLUE = (50, 120, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (230, 230, 230)


@dataclass(slots=True)
class CandidateVisual:
    candidate_id: str
    source: str
    frame_idx: int
    obj_id: int
    mask: np.ndarray
    score: float | None = None
    bbox: list[int] | None = None
    metadata: dict[str, Any] | None = None

    def to_index(self) -> dict[str, Any]:
        box = self.bbox or bbox_from_mask(self.mask)
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "frame_idx": int(self.frame_idx),
            "obj_id": int(self.obj_id),
            "area": int(np.asarray(self.mask).astype(bool).sum()),
            "bbox": box,
            "score": self.score,
            "metadata": self.metadata or {},
        }


def font(size: int = 18) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def homework_roots(workspace: str | Path) -> tuple[Path, Path]:
    ws = Path(workspace)
    jpeg = ws / "homework" / "JPEGImages"
    ann = ws / "homework" / "Annotations"
    if not jpeg.is_dir() and (ws / "JPEGImages").is_dir():
        jpeg = ws / "JPEGImages"
    if not ann.is_dir() and (ws / "Annotations").is_dir():
        ann = ws / "Annotations"
    return jpeg, ann


def list_frames(jpeg_root: Path, video: str) -> list[Path]:
    root = jpeg_root / video
    frames = sorted([p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not frames:
        raise FileNotFoundError(f"no frames for {video}: {root}")
    return frames


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_label(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr


def first_annotation(ann_root: Path, video: str) -> tuple[Path, np.ndarray]:
    files = sorted((ann_root / video).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"no annotation for {video}: {ann_root / video}")
    return files[0], load_label(files[0])


def frame_path_by_idx(frames: list[Path], frame_idx: int) -> Path:
    if frame_idx < 0 or frame_idx >= len(frames):
        raise IndexError(frame_idx)
    return frames[frame_idx]


def pred_label_path(root: Path, video: str, frame_path: Path) -> Path:
    return root / video / f"{frame_path.stem}.png"


def bbox_from_mask(mask: np.ndarray | None) -> list[int] | None:
    if mask is None:
        return None
    ys, xs = np.nonzero(np.asarray(mask).astype(bool))
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def edge_touch(box: list[int] | None, shape: tuple[int, int], margin: int = 4) -> bool:
    if box is None:
        return False
    h, w = shape
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin


def expand_box(box: list[int] | None, shape: tuple[int, int], pad: int | float = 16, square: bool = False, min_side: int = 0) -> list[int]:
    h, w = shape
    if box is None:
        return [0, 0, w, h]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    if isinstance(pad, float):
        p = int(round(max(bw, bh) * pad))
    else:
        p = int(pad)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if square:
        side = max(bw, bh, int(min_side)) + 2 * p
        x1 = int(round(cx - side / 2))
        x2 = x1 + side
        y1 = int(round(cy - side / 2))
        y2 = y1 + side
    else:
        x1, x2 = x1 - p, x2 + p
        y1, y2 = y1 - p, y2 + p
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return [0, 0, w, h]
    return [x1, y1, x2, y2]


def crop_pil(img: Image.Image, box: list[int]) -> Image.Image:
    return img.crop(tuple(int(v) for v in box))


def crop_mask(mask: np.ndarray, box: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    return np.asarray(mask).astype(bool)[y1:y2, x1:x2]


def overlay_mask(img: Image.Image, mask: np.ndarray | None, color: tuple[int, int, int] = GREEN, alpha: float = 0.45) -> Image.Image:
    out = img.convert("RGBA")
    if mask is None:
        return out.convert("RGB")
    m = np.asarray(mask).astype(bool)
    if m.shape[:2] != (out.height, out.width):
        m_img = Image.fromarray((m.astype(np.uint8) * 255)).resize(out.size, Image.Resampling.NEAREST)
        m = np.asarray(m_img) > 0
    layer = Image.new("RGBA", out.size, color + (0,))
    arr = np.asarray(layer).copy()
    arr[m, 3] = int(255 * alpha)
    out = Image.alpha_composite(out, Image.fromarray(arr, "RGBA"))
    return out.convert("RGB")


def draw_bbox(img: Image.Image, box: list[int] | None, color: tuple[int, int, int] = YELLOW, width: int = 4, label: str | None = None) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    if box is not None:
        d.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], outline=color, width=width)
        if label:
            draw_label(d, (box[0], max(0, box[1] - 24)), label, fill=color)
    return out


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int] = YELLOW, size: int = 18) -> None:
    f = font(size)
    x, y = xy
    try:
        box = draw.textbbox((x, y), text, font=f)
    except Exception:
        box = (x, y, x + len(text) * 8, y + 20)
    pad = 3
    draw.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], fill=(0, 0, 0))
    draw.text((x, y), text, font=f, fill=fill)


def add_caption(img: Image.Image, caption: str, size: int = 18) -> Image.Image:
    img = img.convert("RGB")
    cap_h = max(30, size + 12)
    out = Image.new("RGB", (img.width, img.height + cap_h), WHITE)
    out.paste(img, (0, cap_h))
    d = ImageDraw.Draw(out)
    draw_label(d, (6, 5), caption[:120], fill=YELLOW, size=size)
    return out


def fit_cell(img: Image.Image, cell: tuple[int, int], bg: tuple[int, int, int] = WHITE) -> Image.Image:
    cw, ch = cell
    img = img.convert("RGB")
    scale = min(cw / img.width, ch / img.height)
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (cw, ch), bg)
    out.paste(resized, ((cw - nw) // 2, (ch - nh) // 2))
    return out


def grid(images: list[Image.Image], cols: int = 2, cell: tuple[int, int] = (520, 420), bg: tuple[int, int, int] = WHITE) -> Image.Image:
    if not images:
        images = [add_caption(Image.new("RGB", (cell[0], cell[1]), GRAY), "EMPTY")]
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * cell[0], rows * cell[1]), bg)
    for i, img in enumerate(images):
        out.paste(fit_cell(img, cell, bg), ((i % cols) * cell[0], (i // cols) * cell[1]))
    return out


def target_metrics(mask: np.ndarray, image_shape: tuple[int, int]) -> dict[str, Any]:
    box = bbox_from_mask(mask)
    h, w = image_shape
    area = int(np.asarray(mask).astype(bool).sum())
    frac = area / max(float(h * w), 1.0)
    tiny = frac <= 0.01 or area < 250
    return {
        "area": area,
        "area_fraction": frac,
        "bbox": box,
        "is_tiny_by_area": tiny,
        "is_edge_or_partial_by_bbox": edge_touch(box, image_shape),
    }


def render_target_profile_panel(
    *,
    workspace: str | Path,
    video: str,
    obj_id: int,
    out_path: str | Path,
    zoom_factor: int | None = None,
) -> dict[str, Any]:
    jpeg_root, ann_root = homework_roots(workspace)
    frames = list_frames(jpeg_root, video)
    _, ann = first_annotation(ann_root, video)
    rgb = load_rgb(frames[0])
    mask = ann == int(obj_id)
    box = bbox_from_mask(mask)
    metrics = target_metrics(mask, ann.shape)
    min_side = 96 if metrics["is_tiny_by_area"] else 64
    crop_box = expand_box(box, ann.shape, pad=0.75, square=True, min_side=min_side)
    full_box = draw_bbox(rgb, box, YELLOW, width=5, label=f"video={video} obj={obj_id} frame0")
    full_overlay = overlay_mask(rgb, mask, GREEN, alpha=0.45)
    full_overlay = draw_bbox(full_overlay, box, YELLOW, width=4, label="GT mask overlay")
    crop_raw = crop_pil(rgb, crop_box)
    crop_overlay = overlay_mask(crop_pil(rgb, crop_box), crop_mask(mask, crop_box), GREEN, alpha=0.55)
    panels = [
        add_caption(full_box, "frame0 RGB + target bbox"),
        add_caption(full_overlay, "frame0 GT mask overlay (green)"),
        add_caption(crop_raw, "target crop raw"),
        add_caption(crop_overlay, "target crop + mask overlay"),
    ]
    if metrics["is_tiny_by_area"]:
        z = int(zoom_factor or 4)
        zoom = crop_overlay.resize((crop_overlay.width * z, crop_overlay.height * z), Image.Resampling.NEAREST)
        panels.append(add_caption(zoom, f"tiny target {z}x zoom"))
    sheet = grid(panels, cols=2, cell=(620, 520))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    return {
        "video": video,
        "obj_id": int(obj_id),
        "frame_idx": 0,
        "panel_path": str(out),
        "frame_path": str(frames[0]),
        "crop_box": crop_box,
        **metrics,
    }


def render_candidate_judge_panel(
    *,
    workspace: str | Path,
    video: str,
    obj_id: int,
    frame_idx: int,
    candidates: list[CandidateVisual],
    out_path: str | Path,
    pre_gap_frame_idx: int | None = None,
    max_candidates: int = 6,
) -> dict[str, Any]:
    jpeg_root, ann_root = homework_roots(workspace)
    frames = list_frames(jpeg_root, video)
    _, ann = first_annotation(ann_root, video)
    ref_rgb = load_rgb(frames[0])
    ref_mask = ann == int(obj_id)
    ref_box = expand_box(bbox_from_mask(ref_mask), ann.shape, pad=0.75, square=True, min_side=96)
    cur_rgb = load_rgb(frame_path_by_idx(frames, frame_idx))
    panels: list[Image.Image] = []
    panels.append(add_caption(overlay_mask(crop_pil(ref_rgb, ref_box), crop_mask(ref_mask, ref_box), GREEN), "REF: first-frame target instance"))
    if pre_gap_frame_idx is not None and 0 <= pre_gap_frame_idx < len(frames):
        pre_rgb = load_rgb(frames[pre_gap_frame_idx])
        panels.append(add_caption(pre_rgb.resize((min(520, pre_rgb.width), int(pre_rgb.height * min(520, pre_rgb.width) / pre_rgb.width)), Image.Resampling.LANCZOS), f"PRE-GAP context frame {pre_gap_frame_idx}"))
    else:
        panels.append(add_caption(Image.new("RGB", (520, 360), GRAY), "NO PRE-GAP REF"))

    limited = candidates[:max_candidates]
    wide = cur_rgb.copy()
    d = ImageDraw.Draw(wide)
    for cand in limited:
        box = cand.bbox or bbox_from_mask(cand.mask)
        wide = draw_bbox(wide, box, YELLOW if cand.candidate_id == "A" else BLUE, width=5, label=f"{cand.candidate_id}:{cand.source}")
    panels.append(add_caption(wide, f"CURRENT frame {frame_idx}: candidates marked A/B/C"))

    for cand in limited:
        box = cand.bbox or bbox_from_mask(cand.mask)
        crop_box = expand_box(box, cand.mask.shape, pad=0.80, square=True, min_side=112)
        raw = crop_pil(cur_rgb, crop_box)
        over = overlay_mask(raw, crop_mask(cand.mask, crop_box), GREEN, alpha=0.55)
        pair = Image.new("RGB", (raw.width + over.width, max(raw.height, over.height)), WHITE)
        pair.paste(raw, (0, 0)); pair.paste(over, (raw.width, 0))
        panels.append(add_caption(pair, f"Candidate {cand.candidate_id}: {cand.source} raw | overlay"))
    panels.append(add_caption(Image.new("RGB", (520, 360), (245, 245, 245)), "EMPTY / TARGET ABSENT OPTION"))
    sheet = grid(panels, cols=2, cell=(660, 540))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    return {
        "video": video,
        "obj_id": int(obj_id),
        "frame_idx": int(frame_idx),
        "panel_path": str(out),
        "pre_gap_frame_idx": pre_gap_frame_idx,
        "candidates": [c.to_index() for c in limited],
    }


def render_tracklet_judge_panel(
    *,
    workspace: str | Path,
    video: str,
    obj_id: int,
    anchor_frame: int,
    candidate_source: str,
    masks_by_frame: dict[int, np.ndarray],
    out_path: str | Path,
) -> dict[str, Any]:
    jpeg_root, ann_root = homework_roots(workspace)
    frames = list_frames(jpeg_root, video)
    _, ann = first_annotation(ann_root, video)
    ref_rgb = load_rgb(frames[0])
    ref_mask = ann == int(obj_id)
    ref_box = expand_box(bbox_from_mask(ref_mask), ann.shape, pad=0.75, square=True, min_side=96)
    panels: list[Image.Image] = [
        add_caption(overlay_mask(crop_pil(ref_rgb, ref_box), crop_mask(ref_mask, ref_box), GREEN), "REF: first-frame target")
    ]
    frame_records: list[dict[str, Any]] = []
    for idx in sorted(masks_by_frame)[:3]:
        if idx < 0 or idx >= len(frames):
            continue
        rgb = load_rgb(frames[idx])
        mask = np.asarray(masks_by_frame[idx]).astype(bool)
        box = bbox_from_mask(mask)
        crop_box = expand_box(box, mask.shape, pad=0.80, square=True, min_side=112)
        raw = crop_pil(rgb, crop_box)
        over = overlay_mask(raw, crop_mask(mask, crop_box), GREEN, alpha=0.55)
        pair = Image.new("RGB", (raw.width + over.width, max(raw.height, over.height)), WHITE)
        pair.paste(raw, (0, 0)); pair.paste(over, (raw.width, 0))
        panels.append(add_caption(pair, f"{candidate_source} frame {idx}: raw | overlay"))
        frame_records.append({"frame_idx": int(idx), "area": int(mask.sum()), "bbox": box})
    if anchor_frame < len(frames):
        wide = load_rgb(frames[anchor_frame])
        for rec in frame_records:
            if rec["bbox"]:
                wide = draw_bbox(wide, rec["bbox"], YELLOW, width=4, label=f"t={rec['frame_idx']}")
        panels.append(add_caption(wide, f"wide context strip around anchor frame {anchor_frame}"))
    sheet = grid(panels, cols=2, cell=(660, 540))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=92)
    return {
        "video": video,
        "obj_id": int(obj_id),
        "anchor_frame": int(anchor_frame),
        "candidate_source": candidate_source,
        "panel_path": str(out),
        "frames": frame_records,
    }


def write_sheet_index(records: list[dict[str, Any]], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sheets": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

"""Training-free candidate-pool utilities for MOSEv2 re-anchor experiments.

The M8 layer separates **candidate recall** from **identity commit**.  This
module intentionally has no model dependency: it audits existing prediction
roots, connected foreground components, first-frame positives, hard negatives,
and Qwen/M7 judgments before any expensive SAM2 re-propagation is attempted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(slots=True)
class SourceSpec:
    name: str
    root: Path


@dataclass(slots=True)
class MaskStats:
    area: int
    bbox: list[int] | None
    centroid: list[float] | None
    area_frac: float


@dataclass(slots=True)
class CandidateRecord:
    video: str
    obj_id: int
    frame_idx: int
    source: str
    root_name: str
    area: int
    bbox: list[int] | None
    centroid: list[float] | None
    area_ratio_init: float
    area_ratio_default: float | None
    iou_default: float | None
    pos_sim: float = 0.0
    neg_sim: float = 0.0
    margin: float = 0.0
    temporal_support: int = 0
    score: float = 0.0
    qwen_support: bool = False
    qwen_veto: bool = False
    qwen_confidence: float | None = None
    qwen_reason: str | None = None
    qwen_panel: str | None = None
    rejected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "obj_id": self.obj_id,
            "frame_idx": self.frame_idx,
            "source": self.source,
            "root_name": self.root_name,
            "area": self.area,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "area_ratio_init": round(float(self.area_ratio_init), 5),
            "area_ratio_default": None if self.area_ratio_default is None else round(float(self.area_ratio_default), 5),
            "iou_default": None if self.iou_default is None else round(float(self.iou_default), 5),
            "pos_sim": round(float(self.pos_sim), 5),
            "neg_sim": round(float(self.neg_sim), 5),
            "margin": round(float(self.margin), 5),
            "temporal_support": int(self.temporal_support),
            "score": round(float(self.score), 5),
            "qwen_support": bool(self.qwen_support),
            "qwen_veto": bool(self.qwen_veto),
            "qwen_confidence": None if self.qwen_confidence is None else round(float(self.qwen_confidence), 5),
            "qwen_reason": self.qwen_reason,
            "qwen_panel": self.qwen_panel,
            "rejected": list(self.rejected),
        }


def list_frames(video_dir: Path) -> list[Path]:
    return sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_label(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    if not path.is_file():
        return None
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    if shape is not None and arr.shape != shape:
        return None
    return arr


def label_path(root: Path, video: str, frame_stem: str) -> Path:
    return root / video / f"{frame_stem}.png"


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def centroid_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def mask_stats(mask: np.ndarray) -> MaskStats:
    area = int(mask.sum())
    h, w = mask.shape
    return MaskStats(area=area, bbox=bbox_from_mask(mask), centroid=centroid_from_mask(mask), area_frac=area / max(1.0, float(h * w)))


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    aa = a.astype(bool); bb = b.astype(bool)
    inter = np.logical_and(aa, bb).sum(); union = np.logical_or(aa, bb).sum()
    return float(inter / union) if union else 1.0


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-8 else vec


def descriptor(rgb: np.ndarray, mask: np.ndarray, bins: int = 12) -> np.ndarray | None:
    """Small, deterministic object descriptor for fast candidate audits.

    This is not meant to replace DINO.  It gives a cheap first pass and keeps
    source-selection conservative until a DINO/SAM2/Qwen verifier certifies an
    anchor.
    """
    if int(mask.sum()) <= 0:
        return None
    pix = rgb[mask]
    if pix.size == 0:
        return None
    # Deterministic sub-sampling keeps audits fast for large person/background
    # blobs while retaining enough coarse appearance evidence.
    if pix.shape[0] > 4096:
        step = max(1, pix.shape[0] // 4096)
        pix = pix[::step][:4096]
    chunks: list[np.ndarray] = []
    for c in range(3):
        hist, _ = np.histogram(pix[:, c], bins=bins, range=(0, 256))
        chunks.append(l2_normalize(hist.astype(np.float32)))
    # Cheap vectorized color/intensity cues.  Avoid per-pixel Python HSV conversion:
    # M8 pool building scores thousands of masks and must stay interactive.
    gray_pix = pix.astype(np.float32).mean(axis=1)
    hist, _ = np.histogram(gray_pix, bins=bins, range=(0, 256))
    chunks.append(l2_normalize(hist.astype(np.float32)))
    # Chromatic contrast ratios are enough to separate many same-class distractors
    # without pretending this replaces DINO.
    denom = np.maximum(pix.astype(np.float32).sum(axis=1), 1.0)
    for c in range(3):
        ratio = pix[:, c].astype(np.float32) / denom
        hist, _ = np.histogram(ratio, bins=bins, range=(0, 1))
        chunks.append(l2_normalize(hist.astype(np.float32)))
    # Edge orientation was intentionally omitted here: the pool builder scores
    # many masks interactively, while DINO/SAM2 stages handle stronger identity.
    h, w = mask.shape
    box = bbox_from_mask(mask)
    if box is None:
        shape = np.zeros(8, dtype=np.float32)
    else:
        x1, y1, x2, y2 = box
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        edge_touch = float(x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1)
        shape = np.asarray([
            math.log1p(float(mask.sum())) / 12.0,
            float(mask.sum()) / max(float(h * w), 1.0),
            bw / max(float(w), 1.0),
            bh / max(float(h), 1.0),
            min(5.0, bw / max(float(bh), 1.0)) / 5.0,
            min(5.0, bh / max(float(bw), 1.0)) / 5.0,
            ((x1 + x2) / 2.0) / max(float(w), 1.0),
            edge_touch,
        ], dtype=np.float32)
    chunks.append(l2_normalize(shape))
    return l2_normalize(np.concatenate(chunks))


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    if a.size != b.size:
        n = min(int(a.size), int(b.size))
        a = a[:n]; b = b[:n]
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8))


def connected_components(mask: np.ndarray, min_area: int = 12, max_count: int = 12) -> list[np.ndarray]:
    mask = mask.astype(bool)
    if int(mask.sum()) < min_area:
        return []
    try:
        import cv2  # type: ignore
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        comps: list[tuple[int, np.ndarray]] = []
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area >= min_area:
                comps.append((area, labels == idx))
        comps.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in comps[:max_count]]
    except Exception:
        pass
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]; coords = []; seen[sy, sx] = True
        while stack:
            y, x = stack.pop(); coords.append((y, x))
            for yy in range(max(0, y - 1), min(h, y + 2)):
                for xx in range(max(0, x - 1), min(w, x + 2)):
                    if mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True; stack.append((yy, xx))
        if len(coords) >= min_area:
            comp = np.zeros_like(mask, dtype=bool)
            yy, xx = zip(*coords); comp[np.asarray(yy), np.asarray(xx)] = True
            comps.append((len(coords), comp))
    comps.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in comps[:max_count]]


def ring_mask(mask: np.ndarray, pad: int | None = None) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    box = bbox_from_mask(mask)
    if box is None:
        return out
    h, w = mask.shape
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1)
    p = int(pad if pad is not None else max(8, min(96, 2 * side)))
    xx1, yy1 = max(0, x1 - p), max(0, y1 - p)
    xx2, yy2 = min(w, x2 + p), min(h, y2 + p)
    out[yy1:yy2, xx1:xx2] = True
    out[mask.astype(bool)] = False
    return out


def parse_source_roots(items: list[str]) -> list[SourceSpec]:
    out: list[SourceSpec] = []
    for item in items:
        if "=" not in item:
            p = Path(item).expanduser().resolve(); name = p.name
        else:
            name, raw = item.split("=", 1); p = Path(raw).expanduser().resolve(); name = name.strip()
        if p.is_dir():
            out.append(SourceSpec(name=name, root=p))
    return out

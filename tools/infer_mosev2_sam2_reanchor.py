#!/usr/bin/env python3
"""M5R-C: retrieve verified later-frame anchors, inject them into SAM2, and re-propagate.

This is the first complete training-free re-anchor loop after the RAR A/B scaffold:

1. build a high-recall candidate pool from existing prediction sources and foreground components;
2. score candidates with object-level descriptors, including optional SAM2 image-encoder masked pooling;
3. compare positives against hard negatives by margin;
4. call SAM2 ``add_new_mask`` on verified later-frame anchors and rerun propagation;
5. merge re-propagated masks back into the baseline only around accepted anchor windows.

It is intentionally conservative: if no anchor passes identity margin checks, the output is copied
from the SAM2 baseline and the audit explains why.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from infer_mosev2_sam2 import (  # noqa: E402
    import_sam2,
    list_frames,
    load_first_annotation,
    logits_to_label,
    make_submission,
    save_label_png,
)


SOURCE_DEFAULTS = {
    "baseline": "pred_sam2_b101",
    "m11": "pred_sam2_m11_cycle",
    "m2_light": "pred_sam2_m2_light",
    "tiny_crop": "pred_sam2_tiny_crop_candidate",
    "sam31": "pred_sam31_public_boxes",
    "rar_rcms": "pred_sam2_rar",
    "rar_state": "pred_sam2_rar_state",
}


@dataclass(slots=True)
class Descriptor:
    vector: np.ndarray
    parts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class Candidate:
    video: str
    frame_idx: int
    obj_id: int
    source: str
    mask: np.ndarray
    area: int
    bbox: list[int] | None
    centroid: list[float] | None
    descriptor: Descriptor | None = None
    pos_sim: float = 0.0
    neg_sim: float = 0.0
    margin: float = 0.0
    score: float = 0.0
    rejected: list[str] = field(default_factory=list)

    def brief(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "obj_id": self.obj_id,
            "source": self.source,
            "area": self.area,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "pos_sim": round(float(self.pos_sim), 4),
            "neg_sim": round(float(self.neg_sim), 4),
            "margin": round(float(self.margin), 4),
            "score": round(float(self.score), 4),
            "rejected": list(self.rejected),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=REPO_ROOT)
    p.add_argument("--sam2-root", type=Path, default=None)
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--pred-root", type=Path, default=None)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-video-to-cpu", dest="offload_video_to_cpu", action="store_false")
    p.add_argument("--offload-state-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-state-to-cpu", dest="offload_state_to_cpu", action="store_false")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--overwrite-submission", action="store_true")
    p.add_argument("--no-zip", action="store_true")

    p.add_argument("--baseline-root", type=Path, default=None)
    p.add_argument("--m11-root", type=Path, default=None)
    p.add_argument("--m2-light-root", type=Path, default=None)
    p.add_argument("--tiny-crop-root", type=Path, default=None)
    p.add_argument("--sam31-root", type=Path, default=None)
    p.add_argument("--rar-rcms-root", type=Path, default=None)
    p.add_argument("--rar-state-root", type=Path, default=None)
    p.add_argument("--rar-audit-json", type=Path, default=None)
    p.add_argument("--audit-json", type=Path, default=None)
    p.add_argument("--audit-dir", type=Path, default=None)

    p.add_argument("--descriptor", choices=["rgb", "rgb_sam2"], default="rgb_sam2")
    p.add_argument("--include-any-fg-components", action="store_true", default=True)
    p.add_argument("--no-any-fg-components", dest="include_any_fg_components", action="store_false")
    p.add_argument("--sam2-auto-mask-candidates", action="store_true", help="Add SAM2 automatic-mask proposals on recovery frames")
    p.add_argument("--auto-mask-max-frames-per-object", type=int, default=4)
    p.add_argument("--auto-mask-points-per-side", type=int, default=16)
    p.add_argument("--auto-mask-pred-iou-thr", type=float, default=0.70)
    p.add_argument("--auto-mask-stability-thr", type=float, default=0.70)
    p.add_argument("--auto-mask-min-area", type=int, default=12)
    p.add_argument("--auto-mask-max-count", type=int, default=24)
    p.add_argument("--component-min-area", type=int, default=12)
    p.add_argument("--component-max-count", type=int, default=24)
    p.add_argument("--candidate-pad", type=int, default=3, help="Add +/- pad around recovery/event frames")
    p.add_argument("--candidate-frame-stride", type=int, default=0, help="Also sample every N frames; 0 disables")
    p.add_argument("--positive-max-frames", type=int, default=8)
    p.add_argument("--negative-max-items", type=int, default=48)
    p.add_argument("--max-candidates-per-object", type=int, default=300)
    p.add_argument("--max-anchors-per-object", type=int, default=2)
    p.add_argument("--min-anchor-frame", type=int, default=1)
    p.add_argument("--anchor-after-hard-event", action="store_true", default=True)
    p.add_argument("--no-anchor-after-hard-event", dest="anchor_after_hard_event", action="store_false")
    p.add_argument("--anchor-delay-after-hard-event", type=int, default=2)
    p.add_argument("--hard-event-area-drop-ratio", type=float, default=0.20)
    p.add_argument("--min-anchor-separation", type=int, default=6)
    p.add_argument("--identity-margin", type=float, default=0.08)
    p.add_argument("--positive-thr", type=float, default=0.50)
    p.add_argument("--negative-thr", type=float, default=0.92)
    p.add_argument("--max-area-ratio", type=float, default=8.0)
    p.add_argument("--min-area-ratio", type=float, default=0.02)
    p.add_argument("--merge-policy", choices=["anchor_window", "all_reprop", "baseline_only"], default="anchor_window")
    p.add_argument("--merge-radius", type=int, default=14)
    p.add_argument("--copy-baseline-on-no-anchor", action="store_true", default=True)
    return p.parse_args()


def complete_paths(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.sam2_root = (args.sam2_root or ws / "5_19" / "sam2").resolve()
    args.checkpoint = (args.checkpoint or ws / "5_19" / "data" / "sam2" / "sam2.1_hiera_base_plus.pt").resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.pred_root = (args.pred_root or ws / "homework" / "pred_sam2_m5r_reanchor").resolve()
    args.submit_root = (args.submit_root or ws / "homework" / "submission_433_m5r_reanchor").resolve()
    args.zip_path = (args.zip_path or ws / "homework" / "submission_mosev2_m5r_reanchor.zip").resolve()
    for key, sub in SOURCE_DEFAULTS.items():
        attr = f"{key.replace('_', '-')}-root"
        dest = f"{key}_root"
        current = getattr(args, dest, None)
        setattr(args, dest, (current or ws / "homework" / sub).resolve())
    args.rar_audit_json = (args.rar_audit_json or ws / "homework" / "logs" / "rar_state_latest.json").resolve()
    args.audit_json = (args.audit_json or ws / "homework" / "logs" / "m5r_reanchor_latest.json").resolve()
    args.audit_dir = (args.audit_dir or ws / "homework" / "logs" / "m5r_reanchor_by_video").resolve()
    return args


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


def label_path(root: Path, video: str, frame: Path) -> Path:
    return root / video / f"{frame.stem}.png"


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    return float(inter / union) if union else 1.0


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


def bbox_ring_mask(mask: np.ndarray, pad: int | None = None) -> np.ndarray:
    """A local background/context negative around a mask.

    This is a fallback hard-negative source for videos with a single object id
    and no obvious rejected tracklets.  It makes "positive-only" matching fail
    closed instead of accepting an anchor solely because it resembles the first
    frame crop.
    """
    out = np.zeros_like(mask, dtype=bool)
    box = bbox_from_mask(mask)
    if box is None:
        return out
    h, w = mask.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    p = int(pad if pad is not None else max(8, min(96, round(0.75 * max(bw, bh)))))
    xx1, yy1 = max(0, x1 - p), max(0, y1 - p)
    xx2, yy2 = min(w, x2 + p), min(h, y2 + p)
    out[yy1:yy2, xx1:xx2] = True
    out[mask.astype(bool)] = False
    return out


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    denom = float(np.linalg.norm(vec))
    return vec / denom if denom > 1e-8 else vec


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    # SAM2 feature extraction can legitimately fail for a tiny mask if it
    # vanishes at feature resolution.  Descriptors then fall back to the fixed
    # RGB/HSV/edge/shape prefix.  Compare the shared prefix instead of crashing;
    # this preserves high-recall candidate generation while making the audit
    # reveal which items lacked SAM2 image-encoder evidence.
    if a.size != b.size:
        n = min(int(a.size), int(b.size))
        a = l2_normalize(a[:n])
        b = l2_normalize(b[:n])
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8))


def rgb_descriptor(rgb: np.ndarray, mask: np.ndarray, bins: int = 12) -> np.ndarray | None:
    if mask.sum() <= 0:
        return None
    pixels = rgb[mask]
    if pixels.size == 0:
        return None
    chunks: list[np.ndarray] = []
    for c in range(3):
        hist, _ = np.histogram(pixels[:, c], bins=bins, range=(0, 256))
        chunks.append(l2_normalize(hist.astype(np.float32)))
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"))
    hpix = hsv[mask]
    for c, rng in enumerate([(0, 256), (0, 256), (0, 256)]):
        hist, _ = np.histogram(hpix[:, c], bins=bins, range=rng)
        chunks.append(l2_normalize(hist.astype(np.float32)))
    gray = rgb.astype(np.float32).mean(axis=2)
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) + math.pi) / (2 * math.pi)
    hist, _ = np.histogram(ang[mask], bins=9, range=(0, 1), weights=mag[mask])
    chunks.append(l2_normalize(hist.astype(np.float32)))
    h, w = mask.shape
    box = bbox_from_mask(mask)
    if box is None:
        shape = np.zeros(6, dtype=np.float32)
    else:
        x1, y1, x2, y2 = box
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        shape = np.asarray([
            math.log1p(float(mask.sum())) / 12.0,
            float(mask.sum()) / max(float(h * w), 1.0),
            bw / max(float(w), 1.0),
            bh / max(float(h), 1.0),
            (bw / max(float(bh), 1.0)) / 5.0,
            float(np.count_nonzero(mask & border_mask(mask))) / max(float(mask.sum()), 1.0),
        ], dtype=np.float32)
    chunks.append(l2_normalize(shape))
    return l2_normalize(np.concatenate(chunks))


def border_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    if mask.size == 0:
        return out
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = True
    return out


class Sam2DescriptorExtractor:
    def __init__(self, predictor: Any, state: dict[str, Any], enabled: bool):
        self.predictor = predictor
        self.state = state
        self.enabled = enabled
        self.cache: dict[tuple[int, int], np.ndarray | None] = {}

    def describe(self, frame_idx: int, mask: np.ndarray) -> np.ndarray | None:
        if not self.enabled or mask.sum() <= 0:
            return None
        key = (frame_idx, hash(mask.tobytes()))
        if key in self.cache:
            return self.cache[key]
        try:
            import torch
            import torch.nn.functional as F

            with torch.inference_mode():
                _, _, feats, _, feat_sizes = self.predictor._get_image_feature(self.state, frame_idx, 1)
                chunks = []
                # Use the two coarsest levels when available: stable semantics plus some mid-level detail.
                levels = [-1, -2] if len(feats) >= 2 else [-1]
                for level in levels:
                    h, w = feat_sizes[level]
                    feat = feats[level][:, 0, :].permute(1, 0).reshape(-1, h, w).float()
                    m = torch.as_tensor(mask, device=feat.device, dtype=torch.float32)[None, None]
                    m = F.interpolate(m, size=(h, w), mode="nearest")[0, 0] > 0.5
                    if not bool(m.any()):
                        continue
                    obj = feat[:, m].mean(dim=1)
                    dil = F.max_pool2d(m.float()[None, None], kernel_size=5, stride=1, padding=2)[0, 0] > 0.5
                    ring = dil & (~m)
                    if bool(ring.any()):
                        ctx = feat[:, ring].mean(dim=1)
                        vec = torch.cat([obj, obj - ctx], dim=0)
                    else:
                        vec = obj
                    chunks.append(vec.detach().float().cpu().numpy())
                if not chunks:
                    self.cache[key] = None
                    return None
                out = l2_normalize(np.concatenate([l2_normalize(c) for c in chunks]))
                self.cache[key] = out
                return out
        except Exception:
            self.cache[key] = None
            return None


class Sam2AutoMaskCandidateGenerator:
    """Thin optional wrapper around upstream SAM2 automatic masks.

    The user-requested high-recall pool must not rely only on a single current
    tracker path.  This generator supplies an independent per-frame proposal
    source on recovery frames.  It is intentionally bounded because automatic
    masks are expensive and noisy on MOSEv2's tiny/occluded objects.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.generator: Any | None = None
        self.cache: dict[int, list[np.ndarray]] = {}

    def _ensure(self) -> Any:
        if self.generator is not None:
            return self.generator
        import torch

        sys.path.insert(0, str(self.args.sam2_root))
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        model = build_sam2(self.args.model_cfg, str(self.args.checkpoint), device=self.args.device)
        model.eval()
        self.generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=self.args.auto_mask_points_per_side,
            pred_iou_thresh=self.args.auto_mask_pred_iou_thr,
            stability_score_thresh=self.args.auto_mask_stability_thr,
            crop_n_layers=0,
            min_mask_region_area=0,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.generator

    def generate(self, frame_idx: int, rgb: np.ndarray) -> list[np.ndarray]:
        if frame_idx in self.cache:
            return self.cache[frame_idx]
        try:
            records = self._ensure().generate(rgb)
        except Exception as exc:
            print(f"warning: SAM2 automatic-mask proposals failed at frame {frame_idx}: {exc}", file=sys.stderr, flush=True)
            self.cache[frame_idx] = []
            return []
        masks: list[tuple[int, np.ndarray]] = []
        for rec in records:
            seg = rec.get("segmentation") if isinstance(rec, dict) else None
            if seg is None:
                continue
            mask = np.asarray(seg).astype(bool)
            area = int(mask.sum())
            if area >= self.args.auto_mask_min_area:
                masks.append((area, mask))
        masks.sort(key=lambda x: x[0], reverse=True)
        out = [m for _, m in masks[: self.args.auto_mask_max_count]]
        self.cache[frame_idx] = out
        return out


def make_descriptor(rgb: np.ndarray, mask: np.ndarray, sam2_desc: np.ndarray | None) -> Descriptor | None:
    parts: dict[str, int] = {}
    chunks = []
    r = rgb_descriptor(rgb, mask)
    if r is not None:
        parts["rgb_hsv_edge_shape"] = int(r.size)
        chunks.append(r * 0.75)
    if sam2_desc is not None:
        parts["sam2_image_encoder"] = int(sam2_desc.size)
        chunks.append(l2_normalize(sam2_desc) * 1.25)
    if not chunks:
        return None
    return Descriptor(vector=l2_normalize(np.concatenate(chunks)), parts=parts)


def connected_components(mask: np.ndarray, min_area: int, max_count: int) -> list[np.ndarray]:
    mask = mask.astype(bool)
    if mask.sum() < min_area:
        return []
    try:
        import cv2  # type: ignore

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        comps = []
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area >= min_area:
                comps.append((area, labels == idx))
        comps.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in comps[:max_count]]
    except Exception:
        pass
    try:
        from scipy import ndimage  # type: ignore

        labels, n = ndimage.label(mask)
        comps = []
        for idx in range(1, n + 1):
            comp = labels == idx
            area = int(comp.sum())
            if area >= min_area:
                comps.append((area, comp))
        comps.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in comps[:max_count]]
    except Exception:
        # Slow fallback for rare environments without cv2/scipy.
        seen = np.zeros_like(mask, dtype=bool)
        comps: list[tuple[int, np.ndarray]] = []
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        for sy, sx in zip(ys.tolist(), xs.tolist()):
            if seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            coords = []
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                coords.append((y, x))
                for yy in range(max(0, y - 1), min(h, y + 2)):
                    for xx in range(max(0, x - 1), min(w, x + 2)):
                        if mask[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True
                            stack.append((yy, xx))
            if len(coords) >= min_area:
                comp = np.zeros_like(mask, dtype=bool)
                yy, xx = zip(*coords)
                comp[np.asarray(yy), np.asarray(xx)] = True
                comps.append((len(coords), comp))
        comps.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in comps[:max_count]]


def source_roots(args: argparse.Namespace) -> dict[str, Path]:
    roots = {
        "baseline": args.baseline_root,
        "m11": args.m11_root,
        "m2_light": args.m2_light_root,
        "tiny_crop": args.tiny_crop_root,
        "sam31": args.sam31_root,
        "rar_rcms": args.rar_rcms_root,
        "rar_state": args.rar_state_root,
    }
    return {name: root for name, root in roots.items() if root and root.is_dir()}


def load_rar_states(path: Path, video: str) -> dict[tuple[int, int], str]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    audit = data.get("videos", {}).get(video, {}) if "videos" in data else data
    out: dict[tuple[int, int], str] = {}
    for obj_id, obj in audit.get("objects", {}).items():
        for rec in obj.get("frames", []):
            out[(int(obj_id), int(rec.get("frame_idx", -1)))] = str(rec.get("state", ""))
    return out


def event_frames(
    frames: list[Path],
    labels_by_source: dict[str, list[np.ndarray | None]],
    obj_id: int,
    rar_states: dict[tuple[int, int], str],
    args: argparse.Namespace,
) -> set[int]:
    events: set[int] = set()
    prev_area: int | None = None
    baseline = labels_by_source.get("baseline", [])
    m11 = labels_by_source.get("m11", [])
    for idx, _ in enumerate(frames):
        base = (baseline[idx] == obj_id) if idx < len(baseline) and baseline[idx] is not None else None
        m11_mask = (m11[idx] == obj_id) if idx < len(m11) and m11[idx] is not None else None
        area = int(base.sum()) if base is not None else 0
        if idx > 0:
            if area == 0:
                events.add(idx)
            if m11_mask is not None and base is not None and area > 0 and int(m11_mask.sum()) == 0:
                events.add(idx)
            if prev_area and prev_area > 0:
                ratio = area / max(prev_area, 1)
                if ratio > 4.0 or ratio < 0.20:
                    events.add(idx)
            state = rar_states.get((obj_id, idx), "")
            if state and state != "stable":
                events.add(idx)
        if area > 0:
            prev_area = area
    padded = set(events)
    for idx in list(events):
        for j in range(max(1, idx - args.candidate_pad), min(len(frames), idx + args.candidate_pad + 1)):
            padded.add(j)
    if args.candidate_frame_stride and args.candidate_frame_stride > 0:
        padded.update(range(1, len(frames), args.candidate_frame_stride))
    return padded


def anchor_floor_frame(
    frames: list[Path],
    labels_by_source: dict[str, list[np.ndarray | None]],
    obj_id: int,
    init_area: int,
    args: argparse.Namespace,
) -> tuple[int, str]:
    """Return the earliest frame allowed for a *recovered* later-frame anchor.

    Pre-disappearance frames are useful positives/RCMS references, but promoting
    them as re-anchors does not solve reappearance and can perturb an already
    good prefix.  By default, require at least one hard visibility break
    (empty/near-empty, M11 suppression, or severe area collapse) before a
    candidate is eligible for delayed commit.
    """
    floor = max(1, int(args.min_anchor_frame))
    if not args.anchor_after_hard_event:
        return floor, "disabled"
    baseline = labels_by_source.get("baseline", [])
    m11 = labels_by_source.get("m11", [])
    min_area = max(int(args.component_min_area), int(init_area * args.hard_event_area_drop_ratio))
    for idx in range(1, len(frames)):
        base = (baseline[idx] == obj_id) if idx < len(baseline) and baseline[idx] is not None else None
        area = int(base.sum()) if base is not None else 0
        if area <= min_area:
            return max(floor, idx + int(args.anchor_delay_after_hard_event)), f"baseline_area_drop:{idx}:{area}<={min_area}"
        if idx < len(m11) and m11[idx] is not None and base is not None and area > min_area and int((m11[idx] == obj_id).sum()) == 0:
            return max(floor, idx + int(args.anchor_delay_after_hard_event)), f"m11_suppressed:{idx}"
    return floor, "no_hard_event"


def read_labels_for_video(roots: dict[str, Path], video: str, frames: list[Path], shape: tuple[int, int]) -> dict[str, list[np.ndarray | None]]:
    out: dict[str, list[np.ndarray | None]] = {}
    for name, root in roots.items():
        out[name] = [load_label(label_path(root, video, frame), shape) for frame in frames]
    return out


def collect_candidates(
    video: str,
    frames: list[Path],
    labels_by_source: dict[str, list[np.ndarray | None]],
    obj_id: int,
    selected_frames: set[int],
    auto_generator: Sam2AutoMaskCandidateGenerator | None,
    auto_min_frame: int,
    args: argparse.Namespace,
) -> list[Candidate]:
    cands: list[Candidate] = []
    auto_frame_budget = args.auto_mask_max_frames_per_object
    auto_frames: set[int] = set()
    if auto_generator is not None and auto_frame_budget > 0:
        evidence_frames: list[int] = []
        fallback_frames: list[int] = []
        for idx in sorted(selected_frames):
            if idx < auto_min_frame or idx <= 0 or idx >= len(frames):
                continue
            fallback_frames.append(idx)
            best_obj_area = 0
            best_fg_area = 0
            for labels in labels_by_source.values():
                if idx >= len(labels) or labels[idx] is None:
                    continue
                label = labels[idx]
                assert label is not None
                best_obj_area = max(best_obj_area, int((label == obj_id).sum()))
                best_fg_area = max(best_fg_area, int((label > 0).sum()))
            # Prefer the first likely reappearance frames where any independent
            # source has foreground evidence after the hard event.  Do not sort
            # by area: in MOSEv2, a huge late mask is often the *wrong* same-class
            # object or background blob, while the correct reappearance starts
            # small/partial.
            if best_obj_area >= args.component_min_area or best_fg_area >= args.component_min_area:
                evidence_frames.append(idx)
        auto_frames = set(evidence_frames[:auto_frame_budget])
        if not auto_frames:
            auto_frames = set(fallback_frames[:auto_frame_budget])
    for idx in sorted(selected_frames):
        if idx <= 0 or idx >= len(frames):
            continue
        seen_masks: list[np.ndarray] = []
        for source, labels in labels_by_source.items():
            if idx >= len(labels) or labels[idx] is None:
                continue
            label = labels[idx]
            assert label is not None
            masks: list[tuple[str, np.ndarray]] = [(source, label == obj_id)]
            if args.include_any_fg_components:
                for k, comp in enumerate(connected_components(label > 0, args.component_min_area, args.component_max_count)):
                    masks.append((f"{source}:anyfg:{k}", comp))
            for src_name, mask in masks:
                area = int(mask.sum())
                if area < args.component_min_area:
                    continue
                if any(mask_iou(mask, old) > 0.985 for old in seen_masks):
                    continue
                seen_masks.append(mask.copy())
                cands.append(Candidate(
                    video=video,
                    frame_idx=idx,
                    obj_id=obj_id,
                    source=src_name,
                    mask=mask.astype(bool),
                    area=area,
                    bbox=bbox_from_mask(mask),
                    centroid=centroid_from_mask(mask),
                ))
                if len(cands) >= args.max_candidates_per_object:
                    return cands
        if auto_generator is not None and idx in auto_frames:
            rgb = load_rgb(frames[idx])
            for k, mask in enumerate(auto_generator.generate(idx, rgb)):
                area = int(mask.sum())
                if area < args.auto_mask_min_area:
                    continue
                if any(mask_iou(mask, old) > 0.985 for old in seen_masks):
                    continue
                seen_masks.append(mask.copy())
                cands.append(Candidate(
                    video=video,
                    frame_idx=idx,
                    obj_id=obj_id,
                    source=f"sam2_auto:{k}",
                    mask=mask.astype(bool),
                    area=area,
                    bbox=bbox_from_mask(mask),
                    centroid=centroid_from_mask(mask),
                ))
                if len(cands) >= args.max_candidates_per_object:
                    return cands
    return cands


def build_positive_negative_banks(
    video: str,
    frames: list[Path],
    ann: np.ndarray,
    obj_id: int,
    labels_by_source: dict[str, list[np.ndarray | None]],
    events: set[int],
    extractor: Sam2DescriptorExtractor,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    audit: list[dict[str, Any]] = []
    pos: list[np.ndarray] = []
    neg: list[np.ndarray] = []

    def add_desc(kind: str, frame_idx: int, source: str, mask: np.ndarray) -> None:
        rgb = load_rgb(frames[frame_idx])
        desc = make_descriptor(
            rgb,
            mask,
            extractor.describe(frame_idx, mask) if args.descriptor == "rgb_sam2" else None,
        )
        if desc is None:
            return
        target = pos if kind == "positive" else neg
        limit = args.positive_max_frames + 1 if kind == "positive" else args.negative_max_items
        if len(target) < limit:
            target.append(desc.vector)
            audit.append({"kind": kind, "frame_idx": frame_idx, "source": source, "area": int(mask.sum()), "parts": desc.parts})

    add_desc("positive", 0, "first_frame_gt", ann == obj_id)
    first_ring = bbox_ring_mask(ann == obj_id)
    if int(first_ring.sum()) >= args.component_min_area:
        add_desc("negative", 0, "first_frame_context_ring", first_ring)
    baseline = labels_by_source.get("baseline", [])
    m11 = labels_by_source.get("m11", [])
    for idx in range(1, len(frames)):
        if len(pos) >= args.positive_max_frames + 1:
            break
        if idx in events:
            continue
        if idx >= len(baseline) or baseline[idx] is None:
            continue
        mask = baseline[idx] == obj_id
        if int(mask.sum()) < args.component_min_area:
            continue
        if idx < len(m11) and m11[idx] is not None and int((m11[idx] == obj_id).sum()) == 0:
            continue
        add_desc("positive", idx, "baseline_stable", mask)
        if len(neg) < max(2, args.negative_max_items // 8):
            ring = bbox_ring_mask(mask)
            if int(ring.sum()) >= args.component_min_area:
                add_desc("negative", idx, "baseline_stable_context_ring", ring)

    # hard negatives: M11-suppressed baseline masks, same-frame other object ids, and any-FG components not overlapping the object.
    for idx in range(1, len(frames)):
        if len(neg) >= args.negative_max_items:
            break
        if idx < len(baseline) and baseline[idx] is not None:
            bmask = baseline[idx] == obj_id
            if idx < len(m11) and m11[idx] is not None and int(bmask.sum()) >= args.component_min_area and int((m11[idx] == obj_id).sum()) == 0:
                add_desc("negative", idx, "m11_rejected_baseline", bmask)
            for other_id in [int(x) for x in np.unique(baseline[idx]) if int(x) not in {0, obj_id}]:
                omask = baseline[idx] == other_id
                if int(omask.sum()) >= args.component_min_area:
                    add_desc("negative", idx, f"baseline_other_id:{other_id}", omask)
        for source, labels in labels_by_source.items():
            if len(neg) >= args.negative_max_items:
                break
            if idx >= len(labels) or labels[idx] is None:
                continue
            label = labels[idx]
            assert label is not None
            obj_mask = label == obj_id
            for k, comp in enumerate(connected_components(label > 0, args.component_min_area, min(8, args.component_max_count))):
                if mask_iou(comp, obj_mask) < 0.10:
                    add_desc("negative", idx, f"{source}:nearby_anyfg:{k}", comp)
                    if len(neg) >= args.negative_max_items:
                        break
    return pos, neg, audit


def score_candidates(
    cands: list[Candidate],
    pos_bank: list[np.ndarray],
    neg_bank: list[np.ndarray],
    init_area: int,
    extractor: Sam2DescriptorExtractor,
    frames: list[Path],
    args: argparse.Namespace,
) -> None:
    for cand in cands:
        ratio = cand.area / max(float(init_area), 1.0)
        if ratio > args.max_area_ratio:
            cand.rejected.append("area_too_large")
        if ratio < args.min_area_ratio:
            cand.rejected.append("area_too_small")
        rgb = load_rgb(frames[cand.frame_idx])
        cand.descriptor = make_descriptor(
            rgb,
            cand.mask,
            extractor.describe(cand.frame_idx, cand.mask) if args.descriptor == "rgb_sam2" else None,
        )
        if cand.descriptor is None:
            cand.rejected.append("no_descriptor")
            continue
        cand.pos_sim = max((cosine(cand.descriptor.vector, p) for p in pos_bank), default=0.0)
        cand.neg_sim = max((cosine(cand.descriptor.vector, n) for n in neg_bank), default=0.0)
        cand.margin = cand.pos_sim - cand.neg_sim
        cand.score = cand.margin + 0.05 * math.log1p(cand.area)
        if cand.pos_sim < args.positive_thr:
            cand.rejected.append("low_positive_similarity")
        if cand.neg_sim > args.negative_thr and cand.margin < args.identity_margin * 2:
            cand.rejected.append("too_close_to_negative")
        if cand.margin < args.identity_margin:
            cand.rejected.append("low_identity_margin")


def choose_anchors(cands: list[Candidate], min_frame: int, args: argparse.Namespace) -> list[Candidate]:
    valid = [c for c in cands if c.frame_idx >= min_frame and not c.rejected]
    valid.sort(key=lambda c: (c.score, c.margin, c.pos_sim, c.area), reverse=True)
    anchors: list[Candidate] = []
    for cand in valid:
        if any(abs(cand.frame_idx - old.frame_idx) < args.min_anchor_separation for old in anchors):
            continue
        anchors.append(cand)
        if len(anchors) >= args.max_anchors_per_object:
            break
    return anchors


def repropagate_with_anchors(
    predictor: Any,
    args: argparse.Namespace,
    video: str,
    frames: list[Path],
    ann: np.ndarray,
    palette: list[int] | None,
    anchors_by_obj: dict[int, list[Candidate]],
    baseline_labels: list[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    import torch

    video_dir = args.jpeg_root / video
    state = predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )
    predictor.reset_state(state)
    obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
    with torch.inference_mode():
        for obj_id in obj_ids:
            predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=(ann == obj_id))
        for obj_id, anchors in anchors_by_obj.items():
            for anchor in anchors:
                predictor.add_new_mask(state, frame_idx=int(anchor.frame_idx), obj_id=int(obj_id), mask=anchor.mask)

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    reprop: list[np.ndarray | None] = [None] * len(frames)
    with torch.inference_mode(), autocast_ctx:
        for frame_idx, cur_obj_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_idx == 0:
                label = ann.copy()
            else:
                label = logits_to_label(mask_logits, [int(x) for x in cur_obj_ids])
            reprop[int(frame_idx)] = label.astype(ann.dtype, copy=False)
    missing = [i for i, x in enumerate(reprop) if x is None]
    if missing:
        raise RuntimeError(f"{video}: reprop missing frames {missing[:20]}")
    reprop_labels = [x for x in reprop if x is not None]

    if args.merge_policy == "all_reprop":
        final = reprop_labels
    elif args.merge_policy == "baseline_only" or not any(anchors_by_obj.values()):
        final = [x.copy() for x in baseline_labels]
    else:
        final = [x.copy() for x in baseline_labels]
        windows: dict[int, set[int]] = {}
        for obj_id, anchors in anchors_by_obj.items():
            windows[obj_id] = set()
            for anchor in anchors:
                lo = max(1, anchor.frame_idx - args.merge_radius)
                hi = min(len(frames) - 1, anchor.frame_idx + args.merge_radius)
                windows[obj_id].update(range(lo, hi + 1))
        for idx in range(1, len(frames)):
            label = final[idx].copy()
            for obj_id in sorted(windows):
                if idx not in windows[obj_id]:
                    continue
                new_mask = reprop_labels[idx] == obj_id
                label[label == obj_id] = 0
                label[(label == 0) & new_mask] = obj_id
            final[idx] = label.astype(np.uint8 if label.max(initial=0) <= 255 else np.uint16)
    final[0] = ann.copy()
    audit = {
        "reprop_frames": len(reprop_labels),
        "merge_policy": args.merge_policy,
        "merge_radius": args.merge_radius,
        "anchor_windows": {
            str(obj_id): sorted(int(x) for x in sorted({f for a in anchors for f in range(max(1, a.frame_idx - args.merge_radius), min(len(frames) - 1, a.frame_idx + args.merge_radius) + 1)}))
            for obj_id, anchors in anchors_by_obj.items()
        },
    }
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final, audit


def run_video(predictor: Any, args: argparse.Namespace, video: str, roots: dict[str, Path]) -> dict[str, Any]:
    import torch

    frames = list_frames(args.jpeg_root / video)
    ann, palette = load_first_annotation(args.ann_root / video)
    shape = ann.shape
    obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
    out_dir = args.pred_root / video
    if args.skip_existing and len(sorted(out_dir.glob("*.png"))) == len(frames):
        return {"video": video, "status": "skipped", "frames": len(frames), "objects": obj_ids}
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    labels_by_source = read_labels_for_video(roots, video, frames, shape)
    if "baseline" not in labels_by_source or any(x is None for x in labels_by_source["baseline"]):
        raise FileNotFoundError(f"baseline predictions missing for {video}: {roots.get('baseline')}")
    baseline_labels = [x.copy() for x in labels_by_source["baseline"] if x is not None]
    rar_states = load_rar_states(args.rar_audit_json, video)

    # A descriptor-only state; later repropagation uses a fresh state so prompt outputs are clean.
    desc_state = predictor.init_state(
        video_path=str(args.jpeg_root / video),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )
    extractor = Sam2DescriptorExtractor(predictor, desc_state, enabled=args.descriptor == "rgb_sam2")
    auto_generator = Sam2AutoMaskCandidateGenerator(args) if args.sam2_auto_mask_candidates else None

    anchors_by_obj: dict[int, list[Candidate]] = {}
    video_audit: dict[str, Any] = {
        "video": video,
        "frames": len(frames),
        "objects": obj_ids,
        "sources": {k: str(v) for k, v in roots.items()},
        "descriptor": args.descriptor,
        "objects_audit": {},
    }
    for obj_id in obj_ids:
        events = event_frames(frames, labels_by_source, obj_id, rar_states, args)
        init_area = int((ann == obj_id).sum())
        pos_bank, neg_bank, bank_audit = build_positive_negative_banks(
            video, frames, ann, obj_id, labels_by_source, events, extractor, args
        )
        anchor_floor, anchor_floor_reason = anchor_floor_frame(frames, labels_by_source, obj_id, init_area, args)
        cands = collect_candidates(video, frames, labels_by_source, obj_id, events, auto_generator, anchor_floor, args)
        score_candidates(cands, pos_bank, neg_bank, init_area, extractor, frames, args)
        anchors = choose_anchors(cands, anchor_floor, args)
        anchors_by_obj[obj_id] = anchors
        rejected_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for cand in cands:
            source_counts[cand.source] = source_counts.get(cand.source, 0) + 1
            for reason in cand.rejected:
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
        ranked = sorted(cands, key=lambda c: (c.score, c.margin, c.pos_sim), reverse=True)[:20]
        video_audit["objects_audit"][str(obj_id)] = {
            "init_area": init_area,
            "event_frames": sorted(int(x) for x in events),
            "positive_bank": len(pos_bank),
            "negative_bank": len(neg_bank),
            "bank_items": bank_audit[:80],
            "candidate_count": len(cands),
            "anchor_floor_frame": int(anchor_floor),
            "anchor_floor_reason": anchor_floor_reason,
            "candidate_source_counts": dict(sorted(source_counts.items())),
            "rejected_counts": dict(sorted(rejected_counts.items())),
            "anchors": [a.brief() for a in anchors],
            "top_candidates": [c.brief() for c in ranked],
        }
    del desc_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if any(anchors_by_obj.values()):
        final_labels, reprop_audit = repropagate_with_anchors(
            predictor, args, video, frames, ann, palette, anchors_by_obj, baseline_labels
        )
    else:
        final_labels = [x.copy() for x in baseline_labels]
        final_labels[0] = ann.copy()
        reprop_audit = {"reprop_frames": 0, "merge_policy": "baseline_copy_no_anchor", "merge_radius": args.merge_radius, "anchor_windows": {}}

    changed = 0
    for idx, frame in enumerate(frames):
        label = final_labels[idx]
        if idx == 0:
            label = ann.copy()
        if not np.array_equal(label, baseline_labels[idx]):
            changed += 1
        save_label_png(out_dir / f"{frame.stem}.png", label, palette)

    video_audit["summary"] = {
        "accepted_anchor_count": sum(len(v) for v in anchors_by_obj.values()),
        "changed_vs_baseline": changed,
        "reprop": reprop_audit,
    }
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / f"{video}.json").write_text(json.dumps(video_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"video": video, "status": "done", "frames": len(frames), "objects": obj_ids, **video_audit["summary"]}


def collect_provenance(args: argparse.Namespace) -> dict[str, Any]:
    def run_git(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "git_sha": os.environ.get("CVMOSE_GIT_SHA") or run_git(["git", "rev-parse", "HEAD"]),
        "git_branch": os.environ.get("CVMOSE_GIT_BRANCH") or run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "script": str(Path(__file__).resolve()),
        "workspace": str(args.workspace),
        "descriptor": args.descriptor,
        "pred_root": str(args.pred_root),
    }


def validate_submission(args: argparse.Namespace) -> None:
    output_json = args.audit_json.with_name(args.audit_json.stem + "_validation.json")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "validate_mose_submission.py"),
        "--workspace", str(args.workspace),
        "--pred-root", str(args.pred_root),
        "--submit-root", str(args.submit_root),
        "--output-json", str(output_json),
    ]
    if not args.no_zip:
        cmd.extend(["--zip-path", str(args.zip_path)])
    subprocess.run(cmd, check=True)


def main() -> None:
    args = complete_paths(parse_args())
    roots = source_roots(args)
    if "baseline" not in roots:
        raise FileNotFoundError(f"baseline root missing: {args.baseline_root}")
    for required in [args.sam2_root, args.checkpoint, args.jpeg_root, args.ann_root]:
        if not required.exists():
            raise FileNotFoundError(required)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    build_sam2_video_predictor = import_sam2(args.sam2_root)
    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=args.device)
    predictor.eval()

    args.pred_root.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    videos = args.videos or sorted(p.name for p in args.jpeg_root.iterdir() if p.is_dir())
    started = time.time()
    results: list[dict[str, Any]] = []
    print(f"m5r_reanchor videos={len(videos)} descriptor={args.descriptor} sources={sorted(roots)}", flush=True)
    for i, video in enumerate(videos, 1):
        t0 = time.time()
        print(f"[{i}/{len(videos)}] {video} start", flush=True)
        result = run_video(predictor, args, video, roots)
        result["seconds"] = round(time.time() - t0, 2)
        print(f"[{i}/{len(videos)}] {video} {result}", flush=True)
        results.append(result)

    elapsed = time.time() - started
    summary = {
        "videos": len(results),
        "frames": sum(int(r.get("frames", 0)) for r in results),
        "accepted_anchor_count": sum(int(r.get("accepted_anchor_count", 0)) for r in results),
        "changed_vs_baseline": sum(int(r.get("changed_vs_baseline", 0)) for r in results),
    }
    runtime = {
        "seconds": elapsed,
        "fps": summary["frames"] / elapsed if elapsed else None,
    }
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        runtime.update({
            "cuda_max_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
            "cuda_max_memory_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        })
    aggregate = {
        "method": "m5r_candidate_retrieval_reanchor",
        "principle": "high-recall candidates + object-level positive/negative identity margin + SAM2 add_new_mask re-propagation",
        "summary": summary,
        "runtime": runtime,
        "provenance": collect_provenance(args),
        "config": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
        "source_roots": {k: str(v) for k, v in roots.items()},
        "results": results,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print("m5r_reanchor_summary=" + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    print(f"audit_json={args.audit_json}", flush=True)

    if args.make_submission:
        make_submission(args)
        validate_submission(args)


if __name__ == "__main__":
    main()

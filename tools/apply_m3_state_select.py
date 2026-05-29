#!/usr/bin/env python3
"""Training-free M3-state candidate selector for MOSEv2 homework outputs.

This script does not rerun SAM2.  It combines already-produced candidate label
maps (baseline SAM2, M2, M2-light, M11, optional SAM3 adapter) with a small,
auditable identity/presence state machine.  The goal is to make identity-risk
handling explicit rather than tuning a single weighted score.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SOURCE_ORDER = ["baseline", "m2", "m2_light", "m11", "sam31"]
PREFERRED_SOURCE_ORDER = {
    "conservative": ["baseline", "m2_light", "m2", "m11", "sam31"],
    "balanced": ["baseline", "m2_light", "m2", "m11", "sam31"],
    "aggressive": ["baseline", "m2", "m2_light", "sam31", "m11"],
    "surgical": ["baseline", "m11", "m2_light", "m2", "sam31"],
}
STATE_CONFIRMED = "CONFIRMED_VISIBLE"
STATE_UNCERTAIN = "UNCERTAIN"
STATE_OCCLUDED = "LIKELY_OCCLUDED"
STATE_REAPPEARING = "REAPPEARING_CANDIDATE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve()
    default_ws = here.parents[1]
    p.add_argument("--workspace", type=Path, default=default_ws)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--baseline-root", type=Path, default=None, required=False)
    p.add_argument("--m2-root", type=Path, default=None)
    p.add_argument("--m2-light-root", type=Path, default=None)
    p.add_argument("--m11-root", type=Path, default=None)
    p.add_argument("--sam31-adapter-root", type=Path, default=None)
    p.add_argument("--m2-audit-json", type=Path, default=None)
    p.add_argument("--m2-light-audit-json", type=Path, default=None)
    p.add_argument("--m11-audit-json", type=Path, default=None)
    p.add_argument("--out-pred-root", type=Path, required=True)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--audit-dir", type=Path, default=None)
    p.add_argument("--variant", choices=["conservative", "balanced", "aggressive", "surgical"], default="conservative")
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--overwrite-submission", action="store_true")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--confirm-delay", type=int, default=None)
    p.add_argument("--identity-high", type=float, default=0.58)
    p.add_argument("--identity-weak", type=float, default=0.42)
    p.add_argument("--negative-margin", type=float, default=0.04)
    p.add_argument("--agreement-iou", type=float, default=0.30)
    p.add_argument("--min-area-pixels", type=int, default=1)
    p.add_argument("--tiny-area-frac", type=float, default=0.001)
    p.add_argument("--edge-margin-frac", type=float, default=0.015)
    p.add_argument("--max-init-area-ratio", type=float, default=8.0)
    p.add_argument("--max-last-area-ratio", type=float, default=6.0)
    p.add_argument("--max-motion-floor", type=float, default=65.0)
    p.add_argument("--max-motion-scale", type=float, default=5.0)
    p.add_argument("--history-frames", type=int, default=5)
    return p.parse_args()


def complete_args(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.baseline_root = (args.baseline_root or ws / "homework" / "pred_sam2_b101").resolve()
    args.out_pred_root = args.out_pred_root.resolve()
    args.audit_json = args.audit_json.resolve()
    if args.audit_dir is not None:
        args.audit_dir = args.audit_dir.resolve()
    if args.submit_root is None:
        args.submit_root = ws / "homework" / f"submission_433_m3_state_{args.variant}"
    args.submit_root = args.submit_root.resolve()
    if args.zip_path is None:
        args.zip_path = ws / "homework" / f"submission_mosev2_m3_state_{args.variant}.zip"
    args.zip_path = args.zip_path.resolve()
    for attr in ["m2_root", "m2_light_root", "m11_root", "sam31_adapter_root", "m2_audit_json", "m2_light_audit_json", "m11_audit_json"]:
        value = getattr(args, attr)
        if value is not None:
            setattr(args, attr, value.resolve())
    if args.confirm_delay is None:
        args.confirm_delay = {"conservative": 2, "balanced": 2, "aggressive": 1, "surgical": 2}[args.variant]
    if args.variant == "conservative":
        args.identity_high = max(args.identity_high, 0.60)
        args.identity_weak = max(args.identity_weak, 0.45)
    elif args.variant == "aggressive":
        args.identity_high = min(args.identity_high, 0.52)
        args.identity_weak = min(args.identity_weak, 0.36)
    elif args.variant == "surgical":
        # Surgical is a baseline-preserving safety valve: it suppresses only
        # high-risk baseline frames with independent absence/rejection evidence
        # and never imports alternative masks as a new final source.
        args.identity_high = max(args.identity_high, 0.60)
        args.identity_weak = max(args.identity_weak, 0.45)
    return args


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(f"No frames found in {video_dir}")
    return frames


def load_label(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img)
    if arr.ndim != 2:
        raise ValueError(f"label PNG must be single-channel: {path} got shape={arr.shape}")
    return arr


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def load_first_annotation(ann_dir: Path) -> tuple[np.ndarray, list[int] | None]:
    path = ann_dir / "00000.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    img = Image.open(path)
    palette = img.getpalette()
    arr = np.array(img)
    if arr.ndim != 2:
        raise ValueError(f"annotation must be single-channel: {path}")
    return arr, palette


def save_label_png(path: Path, label: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if label.dtype != np.uint8:
        label = label.astype(np.uint8)
    img = Image.fromarray(label, mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
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


def edge_touch(mask: np.ndarray, margin_frac: float) -> bool:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return False
    h, w = mask.shape
    mx = max(1, int(round(w * margin_frac)))
    my = max(1, int(round(h * margin_frac)))
    return bool(xs.min() < mx or ys.min() < my or xs.max() >= w - mx or ys.max() >= h - my)


def rgb_hist(rgb: np.ndarray, mask: np.ndarray, bins: int = 8) -> np.ndarray | None:
    if mask.sum() <= 0:
        return None
    pixels = rgb[mask]
    if pixels.size == 0:
        return None
    chunks = []
    for c in range(3):
        hist, _ = np.histogram(pixels[:, c], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        denom = float(hist.sum())
        if denom > 0:
            hist /= denom
        chunks.append(hist)
    out = np.concatenate(chunks)
    return out.astype(np.float32)


def hist_intersection(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.minimum(a, b).sum() / 3.0)  # three separately normalized channel histograms


def max_hist_intersection(hist: np.ndarray | None, bank: list[np.ndarray]) -> float:
    if hist is None or not bank:
        return 0.0
    return max(hist_intersection(hist, ref) for ref in bank)


def clamp_ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("inf") if num > 0 else 1.0
    return float(num / den)


def safe_round(value: float | None, ndigits: int = 4) -> float | None:
    if value is None:
        return None
    if math.isinf(value) or math.isnan(value):
        return None
    return round(float(value), ndigits)


@dataclass
class Candidate:
    source: str
    mask: np.ndarray
    present: bool
    area: int
    area_frac: float
    bbox: list[int] | None
    centroid: list[float] | None
    edge_touch: bool
    hist: np.ndarray | None
    id_sim: float = 0.0
    neg_sim: float = 0.0
    other_id_sim: float = 0.0
    area_ratio_init: float | None = None
    area_ratio_last: float | None = None
    displacement_px: float | None = None
    agreement_count: int = 0
    agreement_sources: list[str] = field(default_factory=list)
    baseline_iou: float | None = None
    m2_iou: float | None = None
    m2_light_iou: float | None = None
    m11_iou: float | None = None
    sam31_iou: float | None = None
    flags: list[str] = field(default_factory=list)
    candidate_class: str = "uncertain"

    def audit(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "present": self.present,
            "area": self.area,
            "area_frac": safe_round(self.area_frac, 8),
            "bbox": self.bbox,
            "centroid": [safe_round(x, 2) for x in self.centroid] if self.centroid else None,
            "edge_touch": self.edge_touch,
            "identity_sim": safe_round(self.id_sim),
            "negative_sim": safe_round(self.neg_sim),
            "other_identity_sim": safe_round(self.other_id_sim),
            "area_ratio_init": safe_round(self.area_ratio_init),
            "area_ratio_last": safe_round(self.area_ratio_last),
            "displacement_px": safe_round(self.displacement_px, 2),
            "agreement_count": self.agreement_count,
            "agreement_sources": self.agreement_sources,
            "overlap_iou": {
                "baseline": safe_round(self.baseline_iou),
                "m2": safe_round(self.m2_iou),
                "m2_light": safe_round(self.m2_light_iou),
                "m11": safe_round(self.m11_iou),
                "sam31": safe_round(self.sam31_iou),
            },
            "flags": self.flags,
            "candidate_class": self.candidate_class,
        }


@dataclass
class ObjState:
    obj_id: int
    init_area: int
    init_hist: np.ndarray | None
    init_mask: np.ndarray
    other_init_hists: list[np.ndarray]
    identity_bank: list[np.ndarray] = field(default_factory=list)
    negative_bank: list[np.ndarray] = field(default_factory=list)
    last_confirmed_mask: np.ndarray | None = None
    last_confirmed_area: int = 0
    last_confirmed_centroid: list[float] | None = None
    absent_streak: int = 0
    candidate_streak: int = 0
    state: str = STATE_CONFIRMED
    accepted_anchor_frames: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.init_hist is not None:
            self.identity_bank.append(self.init_hist)
        self.last_confirmed_mask = self.init_mask.copy()
        self.last_confirmed_area = self.init_area
        self.last_confirmed_centroid = centroid_from_mask(self.init_mask)
        for h in self.other_init_hists:
            if h is not None:
                self.negative_bank.append(h)

    def update_identity(self, cand: Candidate, max_bank: int = 8) -> None:
        if cand.hist is not None:
            self.identity_bank.append(cand.hist)
            if len(self.identity_bank) > max_bank:
                # Keep the first-frame anchor and the freshest references.
                self.identity_bank = self.identity_bank[:1] + self.identity_bank[-(max_bank - 1) :]
        self.last_confirmed_mask = cand.mask.copy()
        self.last_confirmed_area = cand.area
        self.last_confirmed_centroid = cand.centroid

    def add_negative(self, cand: Candidate, max_bank: int = 12) -> None:
        if cand.hist is not None:
            self.negative_bank.append(cand.hist)
            if len(self.negative_bank) > max_bank:
                self.negative_bank = self.negative_bank[-max_bank:]


def load_source_labels(source_roots: dict[str, Path], video: str, frame_name: str) -> dict[str, np.ndarray]:
    labels: dict[str, np.ndarray] = {}
    for source, root in source_roots.items():
        path = root / video / f"{Path(frame_name).stem}.png"
        if path.is_file():
            labels[source] = load_label(path)
    return labels


def candidate_from_label(source: str, label: np.ndarray | None, obj_id: int, rgb: np.ndarray, args: argparse.Namespace) -> Candidate:
    if label is None:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
    else:
        mask = np.asarray(label == obj_id, dtype=bool)
    area = int(mask.sum())
    present = area >= int(args.min_area_pixels)
    if not present:
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        area = 0
    h, w = mask.shape
    return Candidate(
        source=source,
        mask=mask,
        present=present,
        area=area,
        area_frac=float(area / max(h * w, 1)),
        bbox=bbox_from_mask(mask),
        centroid=centroid_from_mask(mask),
        edge_touch=edge_touch(mask, args.edge_margin_frac),
        hist=rgb_hist(rgb, mask) if present else None,
    )


def source_roots_from_args(args: argparse.Namespace) -> dict[str, Path]:
    roots = {"baseline": args.baseline_root}
    optional = {
        "m2": args.m2_root,
        "m2_light": args.m2_light_root,
        "m11": args.m11_root,
        "sam31": args.sam31_adapter_root,
    }
    for name, path in optional.items():
        if path is not None and path.is_dir():
            roots[name] = path
    return {k: v for k, v in roots.items() if v is not None and v.is_dir()}


def read_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_m2_state_index(path: Path | None) -> dict[tuple[str, int, int], dict[str, Any]]:
    data = read_json(path)
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not isinstance(data, dict):
        return out
    videos = data.get("videos", {})
    if not isinstance(videos, dict):
        return out
    for video, vdata in videos.items():
        objects = vdata.get("objects", {}) if isinstance(vdata, dict) else {}
        for obj_s, obj_data in objects.items():
            try:
                obj_id = int(obj_s)
            except Exception:
                continue
            for fr in obj_data.get("frames", []) if isinstance(obj_data, dict) else []:
                if not isinstance(fr, dict) or "frame_idx" not in fr:
                    continue
                idx = int(fr["frame_idx"])
                state = fr.get("state")
                if state is None:
                    if bool(fr.get("memory_written")):
                        state = "memory_write"
                    elif bool(fr.get("reliable", True)):
                        state = "output_only_uncertain"
                    else:
                        state = "likely_absent"
                out[(video, obj_id, idx)] = {
                    "state": state,
                    "reliable": bool(fr.get("reliable", state == "memory_write")),
                    "memory_written": bool(fr.get("memory_written", state == "memory_write")),
                    "reasons": fr.get("reasons", []),
                    "checks": fr.get("checks", {}),
                }
    return out


def load_m11_suppressed_index(path: Path | None) -> dict[tuple[str, int], set[int]]:
    data = read_json(path)
    out: dict[tuple[str, int], set[int]] = {}
    if not isinstance(data, dict):
        return out
    results = data.get("results", [])
    if not isinstance(results, list):
        return out
    for item in results:
        if not isinstance(item, dict):
            continue
        video = item.get("video")
        per_object = item.get("per_object", {})
        if not video or not isinstance(per_object, dict):
            continue
        for obj_s, obj_data in per_object.items():
            try:
                obj_id = int(obj_s)
            except Exception:
                continue
            frames = obj_data.get("suppressed_frames", []) if isinstance(obj_data, dict) else []
            out[(video, obj_id)] = {int(x) for x in frames}
    return out


def enrich_candidates(cands: list[Candidate], state: ObjState, all_obj_states: dict[int, ObjState], args: argparse.Namespace) -> None:
    source_by_name = {c.source: c for c in cands}
    for cand in cands:
        if cand.present:
            cand.id_sim = max_hist_intersection(cand.hist, state.identity_bank)
            cand.neg_sim = max_hist_intersection(cand.hist, state.negative_bank)
            other_banks: list[np.ndarray] = []
            for oid, ostate in all_obj_states.items():
                if oid != state.obj_id:
                    other_banks.extend(ostate.identity_bank[:3])
            cand.other_id_sim = max_hist_intersection(cand.hist, other_banks)
            cand.area_ratio_init = clamp_ratio(cand.area, state.init_area)
            cand.area_ratio_last = clamp_ratio(cand.area, state.last_confirmed_area)
            if cand.centroid is not None and state.last_confirmed_centroid is not None:
                dx = cand.centroid[0] - state.last_confirmed_centroid[0]
                dy = cand.centroid[1] - state.last_confirmed_centroid[1]
                cand.displacement_px = float(math.hypot(dx, dy))
        for src in SOURCE_ORDER:
            other = source_by_name.get(src)
            iou = mask_iou(cand.mask, other.mask) if other is not None else None
            setattr(cand, f"{src}_iou", iou)
        agree_sources: list[str] = []
        if cand.present:
            for other in cands:
                if other.source == cand.source or not other.present:
                    continue
                if mask_iou(cand.mask, other.mask) >= args.agreement_iou:
                    agree_sources.append(other.source)
        cand.agreement_sources = agree_sources
        cand.agreement_count = 1 + len(agree_sources) if cand.present else 0


def flag_candidate(
    cand: Candidate,
    state: ObjState,
    video: str,
    frame_idx: int,
    args: argparse.Namespace,
    m2_state: dict[tuple[str, int, int], dict[str, Any]],
    m2_light_state: dict[tuple[str, int, int], dict[str, Any]],
    m11_suppressed: dict[tuple[str, int], set[int]],
) -> None:
    flags = cand.flags
    if not cand.present:
        cand.candidate_class = "empty"
        return
    if cand.area < args.min_area_pixels:
        flags.append("too_small")
    if cand.area_ratio_init is not None and cand.area_ratio_init > args.max_init_area_ratio and not cand.edge_touch:
        flags.append("large_area_jump_vs_init")
    if cand.area_ratio_last is not None and cand.area_ratio_last > args.max_last_area_ratio and not cand.edge_touch:
        flags.append("large_area_jump_vs_last")
    max_motion = max(args.max_motion_floor, args.max_motion_scale * math.sqrt(max(state.last_confirmed_area, 1)))
    if cand.displacement_px is not None and cand.displacement_px > max_motion and cand.area_frac > args.tiny_area_frac:
        flags.append("large_motion_vs_last_confirmed")
    if cand.id_sim < args.identity_weak:
        flags.append("low_identity_similarity")
    if cand.neg_sim > cand.id_sim + args.negative_margin:
        flags.append("closer_to_negative_bank")
    if cand.other_id_sim > cand.id_sim + args.negative_margin:
        flags.append("closer_to_other_obj_identity")
    if cand.agreement_count <= 1 and cand.source != "baseline":
        flags.append("single_source_candidate")
    cur_m2 = m2_state.get((video, state.obj_id, frame_idx), {})
    cur_light = m2_light_state.get((video, state.obj_id, frame_idx), {})
    if cand.source == "m2" and cur_m2.get("state") == "likely_absent":
        flags.append("m2_audit_likely_absent")
    if cand.source in {"baseline", "m2", "m2_light"} and cur_light.get("state") == "likely_absent":
        flags.append("m2_light_audit_likely_absent")
    if frame_idx in m11_suppressed.get((video, state.obj_id), set()):
        flags.append("m11_cycle_suppressed_frame")
    # Class is assigned by selector after all source-level evidence is available.


def sort_candidates(cands: list[Candidate], variant: str) -> list[Candidate]:
    order = PREFERRED_SOURCE_ORDER[variant]
    idx = {name: i for i, name in enumerate(order)}
    return sorted(
        [c for c in cands if c.present],
        key=lambda c: (
            -c.agreement_count,
            -(1 if c.id_sim >= 0.58 else 0),
            -c.id_sim,
            c.neg_sim,
            idx.get(c.source, 99),
        ),
    )


def source_empty_vote(
    video: str,
    obj_id: int,
    frame_idx: int,
    cands: list[Candidate],
    m2_light_state: dict[tuple[str, int, int], dict[str, Any]],
    m11_suppressed: dict[tuple[str, int], set[int]],
) -> tuple[int, list[str]]:
    votes = 0
    reasons: list[str] = []
    cand_by_source = {c.source: c for c in cands}
    if not cand_by_source.get("baseline", Candidate("baseline", np.zeros((1, 1), bool), False, 0, 0, None, None, False, None)).present:
        votes += 1
        reasons.append("baseline_empty")
    light = m2_light_state.get((video, obj_id, frame_idx), {})
    if light.get("state") == "likely_absent":
        votes += 1
        reasons.append("m2_light_likely_absent")
    if frame_idx in m11_suppressed.get((video, obj_id), set()):
        votes += 1
        reasons.append("m11_suppressed")
    if "m11" in cand_by_source and not cand_by_source["m11"].present and cand_by_source.get("baseline") and cand_by_source["baseline"].present:
        votes += 1
        reasons.append("m11_empty_vs_baseline")
    return votes, reasons


def classify_and_select(
    *,
    video: str,
    frame_idx: int,
    obj_id: int,
    cands: list[Candidate],
    state: ObjState,
    args: argparse.Namespace,
    m2_light_state: dict[tuple[str, int, int], dict[str, Any]],
    m11_suppressed: dict[tuple[str, int], set[int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    cand_by_source = {c.source: c for c in cands}
    baseline = cand_by_source.get("baseline")
    ranked = sort_candidates(cands, args.variant)
    empty_votes, empty_reasons = source_empty_vote(video, obj_id, frame_idx, cands, m2_light_state, m11_suppressed)
    post_gap = state.absent_streak >= 2 or state.state in {STATE_OCCLUDED, STATE_REAPPEARING}
    init_tiny = (state.init_area / max(state.init_mask.size, 1)) <= args.tiny_area_frac

    selected: Candidate | None = None
    decision = "empty"
    decision_reasons: list[str] = []
    next_state = STATE_OCCLUDED
    output_class = "reject"
    update_identity = False
    accepted_anchor = False

    def is_hard_reject(c: Candidate) -> bool:
        hard = {
            "closer_to_negative_bank",
            "closer_to_other_obj_identity",
            "large_area_jump_vs_init",
            "large_area_jump_vs_last",
        }
        if any(f in c.flags for f in hard):
            return True
        # Cycle fail is negative evidence, but for tiny/edge targets it is not a single-frame hard reject.
        if "m11_cycle_suppressed_frame" in c.flags and not (c.edge_touch or init_tiny):
            return True
        return False

    def is_suspicious(c: Candidate) -> bool:
        if is_hard_reject(c):
            return True
        soft = {"low_identity_similarity", "large_motion_vs_last_confirmed", "single_source_candidate"}
        return any(f in c.flags for f in soft)

    def can_accept_visible(c: Candidate) -> bool:
        if is_hard_reject(c):
            return False
        if args.variant == "conservative":
            # Conservative mode treats masked RGB appearance as necessary but
            # not sufficient evidence.  A single-source high-color match after
            # a gap is exactly how same-class distractors can enter the state
            # machine, so require at least one agreeing source unless the
            # low-risk baseline itself is being preserved.
            if c.source != "baseline" and c.agreement_count < 2:
                return False
            if post_gap and c.agreement_count < 2:
                return False
        if c.id_sim >= args.identity_high:
            return True
        if c.agreement_count >= 3 and c.id_sim >= args.identity_weak and c.neg_sim <= c.id_sim + args.negative_margin:
            return True
        if not post_gap and c.source == "baseline" and c.agreement_count >= 2 and c.id_sim >= args.identity_weak:
            return True
        return False

    def can_output_only(c: Candidate) -> bool:
        if is_hard_reject(c):
            return False
        if args.variant == "conservative" and post_gap and c.source != "baseline" and c.agreement_count < 2:
            return False
        if c.id_sim >= args.identity_weak and c.agreement_count >= 2:
            return True
        if args.variant == "aggressive" and c.id_sim >= 0.34 and not (c.neg_sim > c.id_sim + args.negative_margin):
            return True
        if not post_gap and c.source == "baseline" and not (c.neg_sim > c.id_sim + args.negative_margin):
            return True
        return False

    if args.variant == "surgical":
        # Baseline-preserving variant: M3 acts only as a high-risk rejector.
        # This is the safest response to the observed conservative/balanced
        # over-empty behavior: do not import SAM3/M2 masks as final masks, but
        # still use M11 cycle and M2-light absence evidence to prevent obvious
        # post-occlusion wrong-tracklet continuation.
        if baseline is not None and baseline.present:
            prev_absent = state.absent_streak
            prev_candidate = state.candidate_streak
            hard_baseline_reject = is_hard_reject(baseline)
            cycle_reject = "m11_cycle_suppressed_frame" in baseline.flags
            consensus_absent = post_gap and empty_votes >= 2
            if cycle_reject or consensus_absent or hard_baseline_reject:
                decision = "empty"
                output_class = "reject"
                next_state = STATE_OCCLUDED
                decision_reasons.extend(empty_reasons)
                if cycle_reject:
                    decision_reasons.append("surgical_m11_cycle_reject")
                if consensus_absent:
                    decision_reasons.append("surgical_consensus_absent")
                if hard_baseline_reject:
                    decision_reasons.append("surgical_hard_baseline_reject")
                state.absent_streak += 1
                state.candidate_streak = 0
                state.state = next_state
                selected_mask = np.zeros_like(state.init_mask, dtype=bool)
                for cand in ranked:
                    if cand.present and (cand.source == "baseline" or is_hard_reject(cand)):
                        cand.candidate_class = "reject"
                        state.add_negative(cand)
                audit = {
                    "frame_idx": frame_idx,
                    "obj_id": obj_id,
                    "prev_absent_streak": prev_absent,
                    "post_gap": post_gap,
                    "empty_votes": empty_votes,
                    "empty_reasons": empty_reasons,
                    "decision": decision,
                    "decision_reasons": decision_reasons,
                    "output_class": output_class,
                    "state_after": state.state,
                    "accepted_anchor": False,
                    "candidate_streak": state.candidate_streak,
                    "selected_source": "empty",
                    "selected_area": 0,
                    "selected_bbox": None,
                    "candidates": [c.audit() for c in cands],
                }
                return selected_mask, audit

            selected = baseline
            decision = "baseline"
            decision_reasons.append("surgical_keep_baseline")
            candidate_ok = not is_suspicious(selected)
            if post_gap:
                state.candidate_streak = prev_candidate + 1 if candidate_ok else 0
                state.absent_streak = max(prev_absent, 1)
                if candidate_ok and state.candidate_streak >= args.confirm_delay:
                    output_class = "accept_visible"
                    next_state = STATE_CONFIRMED
                    accepted_anchor = True
                    state.absent_streak = 0
                    update_identity = True
                    state.update_identity(selected)
                    decision_reasons.append("surgical_confirmed_after_delay")
                else:
                    output_class = "output_only"
                    next_state = STATE_REAPPEARING
                    decision_reasons.append("surgical_reappearing_candidate_delay")
            else:
                state.absent_streak = 0
                state.candidate_streak = 0
                output_class = "accept_visible"
                next_state = STATE_CONFIRMED
                if candidate_ok:
                    update_identity = True
                    state.update_identity(selected)
            selected.candidate_class = output_class
            state.state = next_state
            audit = {
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "prev_absent_streak": prev_absent,
                "post_gap": post_gap,
                "empty_votes": empty_votes,
                "empty_reasons": empty_reasons,
                "decision": decision,
                "decision_reasons": decision_reasons,
                "output_class": output_class,
                "state_after": state.state,
                "accepted_anchor": accepted_anchor,
                "candidate_streak": state.candidate_streak,
                "selected_source": selected.source,
                "selected_area": selected.area,
                "selected_bbox": selected.bbox,
                "candidates": [c.audit() for c in cands],
            }
            return selected.mask, audit

        decision = "empty"
        output_class = "empty"
        next_state = STATE_OCCLUDED
        decision_reasons.extend(empty_reasons or ["surgical_baseline_empty"])
        state.absent_streak += 1
        state.candidate_streak = 0
        state.state = next_state
        audit = {
            "frame_idx": frame_idx,
            "obj_id": obj_id,
            "prev_absent_streak": max(0, state.absent_streak - 1),
            "post_gap": post_gap,
            "empty_votes": empty_votes,
            "empty_reasons": empty_reasons,
            "decision": decision,
            "decision_reasons": decision_reasons,
            "output_class": output_class,
            "state_after": state.state,
            "accepted_anchor": False,
            "candidate_streak": state.candidate_streak,
            "selected_source": "empty",
            "selected_area": 0,
            "selected_bbox": None,
            "candidates": [c.audit() for c in cands],
        }
        return np.zeros_like(state.init_mask, dtype=bool), audit

    # Conservative empty is an explicit state decision, not a low-priority source.
    if args.variant == "conservative" and post_gap and empty_votes >= 2:
        usable = [c for c in ranked if can_accept_visible(c)]
        if usable:
            selected = usable[0]
            decision = selected.source
            decision_reasons.append("post_gap_empty_votes_overridden_by_strong_identity")
            output_class = "accept_visible"
            next_state = STATE_CONFIRMED
            update_identity = True
        else:
            decision = "empty"
            decision_reasons.extend(["post_gap_empty_votes"] + empty_reasons)
            output_class = "reject"
            next_state = STATE_OCCLUDED
    else:
        # Baseline is kept when it is low-risk; it is not blindly top-priority.
        candidates_to_try = ranked
        for cand in candidates_to_try:
            if can_accept_visible(cand):
                selected = cand
                decision = cand.source
                output_class = "accept_visible"
                next_state = STATE_CONFIRMED
                update_identity = True
                decision_reasons.append("accepted_visible_rule")
                break
        if selected is None:
            for cand in candidates_to_try:
                if can_output_only(cand):
                    selected = cand
                    decision = cand.source
                    output_class = "output_only"
                    next_state = STATE_REAPPEARING if post_gap else STATE_UNCERTAIN
                    decision_reasons.append("output_only_uncertain_rule")
                    break
        if selected is None:
            if baseline is not None and baseline.present and args.variant == "aggressive" and not is_hard_reject(baseline):
                selected = baseline
                decision = baseline.source
                output_class = "output_only"
                next_state = STATE_UNCERTAIN
                decision_reasons.append("aggressive_kept_baseline_despite_weak_identity")
            elif baseline is not None and baseline.present and not post_gap and args.variant != "conservative" and not is_hard_reject(baseline):
                selected = baseline
                decision = baseline.source
                output_class = "output_only"
                next_state = STATE_UNCERTAIN
                decision_reasons.append("pre_gap_baseline_fallback")
            else:
                decision = "empty"
                output_class = "reject" if ranked else "empty"
                next_state = STATE_OCCLUDED if empty_votes or post_gap else STATE_UNCERTAIN
                decision_reasons.extend(empty_reasons or ["no_candidate_passed_rules"])

    if selected is not None:
        if output_class == "accept_visible":
            state.candidate_streak = state.candidate_streak + 1 if post_gap else args.confirm_delay
            if state.candidate_streak >= args.confirm_delay:
                accepted_anchor = True
                if frame_idx not in state.accepted_anchor_frames:
                    state.accepted_anchor_frames.append(frame_idx)
        else:
            state.candidate_streak = 0
        state.absent_streak = 0
        if update_identity:
            state.update_identity(selected)
        selected.candidate_class = "accept_anchor_candidate" if accepted_anchor else output_class
        selected_mask = selected.mask
    else:
        state.absent_streak += 1
        state.candidate_streak = 0
        selected_mask = np.zeros_like(state.init_mask, dtype=bool)

    state.state = next_state
    # Rejected visible candidates become hard negatives after decision, but avoid adding tiny/edge uncertain cases too eagerly.
    for cand in ranked:
        if selected is not None and cand.source == selected.source:
            continue
        if is_hard_reject(cand) or (decision == "empty" and cand.id_sim < args.identity_weak and not cand.edge_touch and not init_tiny):
            cand.candidate_class = "reject"
            state.add_negative(cand)
        elif cand.candidate_class == "uncertain":
            cand.candidate_class = "uncertain"

    audit = {
        "frame_idx": frame_idx,
        "obj_id": obj_id,
        "prev_absent_streak": max(0, state.absent_streak - (1 if selected is None else 0)),
        "post_gap": post_gap,
        "empty_votes": empty_votes,
        "empty_reasons": empty_reasons,
        "decision": decision,
        "decision_reasons": decision_reasons,
        "output_class": "accept_anchor_candidate" if accepted_anchor else output_class,
        "state_after": state.state,
        "accepted_anchor": accepted_anchor,
        "candidate_streak": state.candidate_streak,
        "selected_source": selected.source if selected is not None else "empty",
        "selected_area": selected.area if selected is not None else 0,
        "selected_bbox": selected.bbox if selected is not None else None,
        "candidates": [c.audit() for c in cands],
    }
    return selected_mask, audit


def label_sources_for_frame(
    source_roots: dict[str, Path], video: str, frame_path: Path, expected_shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    labels = load_source_labels(source_roots, video, frame_path.name)
    for source, arr in labels.items():
        if arr.shape != expected_shape:
            raise ValueError(f"{video}/{frame_path.name}: {source} shape {arr.shape} != expected {expected_shape}")
    return labels


def init_obj_states(ann: np.ndarray, rgb0: np.ndarray, args: argparse.Namespace) -> dict[int, ObjState]:
    obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
    init_hists: dict[int, np.ndarray | None] = {}
    init_masks: dict[int, np.ndarray] = {}
    init_areas: dict[int, int] = {}
    for oid in obj_ids:
        mask = ann == oid
        init_masks[oid] = mask
        init_areas[oid] = int(mask.sum())
        init_hists[oid] = rgb_hist(rgb0, mask)
    states: dict[int, ObjState] = {}
    for oid in obj_ids:
        other = [hist for oo, hist in init_hists.items() if oo != oid and hist is not None]
        states[oid] = ObjState(
            obj_id=oid,
            init_area=init_areas[oid],
            init_hist=init_hists[oid],
            init_mask=init_masks[oid],
            other_init_hists=other,
        )
    return states


def combine_masks(
    selected_masks: dict[int, np.ndarray],
    obj_decisions: dict[int, dict[str, Any]],
    ann_shape: tuple[int, int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    label = np.zeros(ann_shape, dtype=np.uint16)
    conflicts: list[dict[str, Any]] = []

    def rank(obj_id: int) -> tuple[int, int, int]:
        d = obj_decisions[obj_id]
        cls = d.get("output_class", "")
        class_rank = {"accept_anchor_candidate": 4, "accept_visible": 3, "output_only": 2, "uncertain": 1, "reject": 0, "empty": 0}.get(cls, 0)
        area = int(d.get("selected_area", 0))
        # Smaller area wins inside equal class to reduce blobs swallowing tiny ids.
        return (class_rank, -area, -obj_id)

    for oid in sorted(selected_masks, key=rank, reverse=True):
        mask = selected_masks[oid].astype(bool)
        if not mask.any():
            continue
        overlap = mask & (label != 0)
        if overlap.any():
            conflicts.append({"obj_id": oid, "overlap_pixels": int(overlap.sum()), "with_labels": [int(x) for x in np.unique(label[overlap]) if int(x) != 0]})
            mask = mask & (label == 0)
        label[mask] = oid
    if label.max(initial=0) <= 255:
        label = label.astype(np.uint8)
    return label, conflicts


def run_video(
    video: str,
    args: argparse.Namespace,
    source_roots: dict[str, Path],
    m2_state: dict[tuple[str, int, int], dict[str, Any]],
    m2_light_state: dict[tuple[str, int, int], dict[str, Any]],
    m11_suppressed: dict[tuple[str, int], set[int]],
) -> dict[str, Any]:
    frames = list_frames(args.jpeg_root / video)
    ann, palette = load_first_annotation(args.ann_root / video)
    out_dir = args.out_pred_root / video
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()
    rgb0 = load_rgb(frames[0])
    obj_states = init_obj_states(ann, rgb0, args)
    obj_ids = sorted(obj_states)
    summary = {
        "frames": len(frames),
        "objects": obj_ids,
        "changed_vs_baseline": 0,
        "empty_outputs": {str(oid): 0 for oid in obj_ids},
        "source_counts": {},
        "class_counts": {},
        "state_counts": {},
        "conflict_frames": 0,
        "accepted_anchor_counts": {str(oid): 0 for oid in obj_ids},
    }
    frame_audits: list[dict[str, Any]] = []

    for frame_idx, frame_path in enumerate(frames):
        rgb = load_rgb(frame_path)
        if rgb.shape[:2] != ann.shape:
            raise ValueError(f"{video}/{frame_path.name}: RGB shape {rgb.shape[:2]} != ann shape {ann.shape}")
        if frame_idx == 0:
            save_label_png(out_dir / f"{frame_path.stem}.png", ann, palette)
            for oid in obj_ids:
                area0 = int((ann == oid).sum())
                summary["source_counts"]["first_frame_gt"] = summary["source_counts"].get("first_frame_gt", 0) + 1
                summary["class_counts"]["gt_anchor"] = summary["class_counts"].get("gt_anchor", 0) + 1
                summary["state_counts"][STATE_CONFIRMED] = summary["state_counts"].get(STATE_CONFIRMED, 0) + 1
                if area0 == 0:
                    summary["empty_outputs"][str(oid)] += 1
            frame_audits.append({
                "frame_idx": 0,
                "frame_name": frame_path.name,
                "decision": "first_frame_gt",
                "objects": {
                    str(oid): {
                        "decision": "first_frame_gt",
                        "selected_source": "first_frame_gt",
                        "output_class": "gt_anchor",
                        "state_after": STATE_CONFIRMED,
                        "selected_area": int((ann == oid).sum()),
                        "selected_bbox": bbox_from_mask(ann == oid),
                    }
                    for oid in obj_ids
                },
                "conflicts": [],
            })
            continue
        labels = label_sources_for_frame(source_roots, video, frame_path, ann.shape)
        selected_masks: dict[int, np.ndarray] = {}
        obj_decisions: dict[int, dict[str, Any]] = {}
        obj_audits: dict[str, Any] = {}
        for oid in obj_ids:
            state = obj_states[oid]
            cands = [candidate_from_label("empty", None, oid, rgb, args)]
            for source in SOURCE_ORDER:
                if source not in source_roots:
                    continue
                cands.append(candidate_from_label(source, labels.get(source), oid, rgb, args))
            enrich_candidates(cands, state, obj_states, args)
            for cand in cands:
                flag_candidate(cand, state, video, frame_idx, args, m2_state, m2_light_state, m11_suppressed)
            mask, audit = classify_and_select(
                video=video,
                frame_idx=frame_idx,
                obj_id=oid,
                cands=cands,
                state=state,
                args=args,
                m2_light_state=m2_light_state,
                m11_suppressed=m11_suppressed,
            )
            selected_masks[oid] = mask
            obj_decisions[oid] = audit
            obj_audits[str(oid)] = audit
            source = audit.get("selected_source", "empty")
            cls = audit.get("output_class", "empty")
            st = audit.get("state_after", "")
            summary["source_counts"][source] = summary["source_counts"].get(source, 0) + 1
            summary["class_counts"][cls] = summary["class_counts"].get(cls, 0) + 1
            summary["state_counts"][st] = summary["state_counts"].get(st, 0) + 1
            if audit.get("selected_area", 0) == 0:
                summary["empty_outputs"][str(oid)] += 1
            if audit.get("accepted_anchor"):
                summary["accepted_anchor_counts"][str(oid)] += 1
        label, conflicts = combine_masks(selected_masks, obj_decisions, ann.shape)
        if conflicts:
            summary["conflict_frames"] += 1
        save_label_png(out_dir / f"{frame_path.stem}.png", label, palette)
        baseline_path = source_roots["baseline"] / video / f"{frame_path.stem}.png"
        if baseline_path.is_file():
            base_label = load_label(baseline_path)
            if not np.array_equal(label, base_label):
                summary["changed_vs_baseline"] += 1
        frame_audits.append({
            "frame_idx": frame_idx,
            "frame_name": frame_path.name,
            "objects": obj_audits,
            "conflicts": conflicts,
        })

    # Validate names/count/sizes for this video.
    out_names = [p.name for p in sorted(out_dir.glob("*.png"))]
    expected_names = [f"{p.stem}.png" for p in frames]
    if out_names != expected_names:
        raise RuntimeError(f"{video}: output names mismatch")
    return {
        "video": video,
        "variant": args.variant,
        "summary": summary,
        "accepted_anchor_frames": {str(oid): obj_states[oid].accepted_anchor_frames for oid in obj_ids},
        "frames": frame_audits,
    }


def copy_tree_contents(src_root: Path, submit_root: Path) -> None:
    if not src_root.is_dir():
        raise FileNotFoundError(src_root)
    for video_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        shutil.copytree(video_dir, submit_root / video_dir.name, dirs_exist_ok=True)


def list_video_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def expected_pred_png_count(jpeg_root: Path) -> int:
    total = 0
    for video_dir in sorted(p for p in jpeg_root.iterdir() if p.is_dir()):
        total += len(list_frames(video_dir))
    return total


def make_submission(args: argparse.Namespace) -> dict[str, Any]:
    predicted_videos = set(list_video_dirs(args.jpeg_root))
    provided_videos = set(list_video_dirs(args.provided_output_root))
    actual_pred_videos = set(list_video_dirs(args.out_pred_root))
    if len(predicted_videos) != 15:
        raise RuntimeError(f"expected 15 predicted videos under jpeg-root, got {len(predicted_videos)}")
    if actual_pred_videos != predicted_videos:
        raise RuntimeError(
            "out-pred-root must contain exactly the 15 predicted videos; "
            f"missing={sorted(predicted_videos - actual_pred_videos)} extra={sorted(actual_pred_videos - predicted_videos)}"
        )
    overlap = actual_pred_videos & provided_videos
    if overlap:
        raise RuntimeError(f"out-pred-root would overwrite provided videos: {sorted(overlap)[:20]}")
    pred_png_count = sum(1 for _ in args.out_pred_root.rglob("*.png"))
    expected_pred_pngs = expected_pred_png_count(args.jpeg_root)
    if pred_png_count != expected_pred_pngs:
        raise RuntimeError(f"pred PNG count mismatch expected={expected_pred_pngs} got={pred_png_count}")

    if args.submit_root.exists() and args.overwrite_submission:
        shutil.rmtree(args.submit_root)
    args.submit_root.mkdir(parents=True, exist_ok=True)
    copy_tree_contents(args.provided_output_root, args.submit_root)
    copy_tree_contents(args.out_pred_root, args.submit_root)
    video_dirs = sorted(p for p in args.submit_root.iterdir() if p.is_dir())
    png_count = sum(1 for _ in args.submit_root.rglob("*.png"))
    payload = {"submit_root": str(args.submit_root), "video_dirs": len(video_dirs), "pngs": png_count}
    print(f"submission_dir={args.submit_root} videos={len(video_dirs)} pngs={png_count}", flush=True)
    if len(video_dirs) != 433 or png_count != 66526:
        raise RuntimeError(f"expected 433 video dirs / 66526 PNGs, got {len(video_dirs)} / {png_count}")
    if not args.no_zip:
        if args.zip_path.exists():
            args.zip_path.unlink()
        with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in sorted(args.submit_root.rglob("*.png")):
                zf.write(path, path.relative_to(args.submit_root).as_posix())
        with zipfile.ZipFile(args.zip_path) as zf:
            bad = zf.testzip()
            png_entries = [name for name in zf.namelist() if name.endswith(".png")]
        if bad is not None or len(png_entries) != 66526:
            raise RuntimeError(f"zip validation failed bad={bad} png_entries={len(png_entries)}")
        payload.update({"zip_path": str(args.zip_path), "zip_size": args.zip_path.stat().st_size})
        print(f"zip_path={args.zip_path} size={args.zip_path.stat().st_size}", flush=True)
    return payload


def main() -> None:
    start = time.time()
    args = complete_args(parse_args())
    for required in [args.jpeg_root, args.ann_root, args.baseline_root, args.provided_output_root]:
        if not required.exists():
            raise FileNotFoundError(required)
    source_roots = source_roots_from_args(args)
    if "baseline" not in source_roots:
        raise FileNotFoundError(f"baseline-root missing: {args.baseline_root}")
    args.out_pred_root.mkdir(parents=True, exist_ok=True)
    if args.audit_dir is not None:
        args.audit_dir.mkdir(parents=True, exist_ok=True)
    videos = args.videos or sorted(p.name for p in args.jpeg_root.iterdir() if p.is_dir())
    m2_state = load_m2_state_index(args.m2_audit_json)
    m2_light_state = load_m2_state_index(args.m2_light_audit_json)
    m11_suppressed = load_m11_suppressed_index(args.m11_audit_json)
    results = []
    aggregate = {
        "videos": 0,
        "frames": 0,
        "objects": 0,
        "changed_vs_baseline": 0,
        "source_counts": {},
        "class_counts": {},
        "state_counts": {},
        "conflict_frames": 0,
        "empty_outputs": {},
        "accepted_anchor_counts": {},
    }
    for video in videos:
        result = run_video(video, args, source_roots, m2_state, m2_light_state, m11_suppressed)
        results.append(result)
        if args.audit_dir is not None:
            (args.audit_dir / f"{video}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        s = result["summary"]
        aggregate["videos"] += 1
        aggregate["frames"] += int(s["frames"])
        aggregate["objects"] += len(s["objects"])
        aggregate["changed_vs_baseline"] += int(s.get("changed_vs_baseline", 0))
        aggregate["conflict_frames"] += int(s.get("conflict_frames", 0))
        for key in ["source_counts", "class_counts", "state_counts", "empty_outputs", "accepted_anchor_counts"]:
            for name, count in s.get(key, {}).items():
                aggregate[key][name] = aggregate[key].get(name, 0) + int(count)
    submission = make_submission(args) if args.make_submission else None
    payload = {
        "method": "m3_state_candidate_selector",
        "variant": args.variant,
        "principle": "training-free explicit presence/identity state; no hidden labels; no weighted final_score",
        "provenance": {
            "git_sha": os.environ.get("CVMOSE_GIT_SHA"),
            "git_branch": os.environ.get("CVMOSE_GIT_BRANCH"),
            "workspace": str(args.workspace),
            "source_roots": {k: str(v) for k, v in source_roots.items()},
            "audits": {
                "m2": str(args.m2_audit_json) if args.m2_audit_json else None,
                "m2_light": str(args.m2_light_audit_json) if args.m2_light_audit_json else None,
                "m11": str(args.m11_audit_json) if args.m11_audit_json else None,
            },
        },
        "config": {
            "variant": args.variant,
            "confirm_delay": args.confirm_delay,
            "identity_high": args.identity_high,
            "identity_weak": args.identity_weak,
            "negative_margin": args.negative_margin,
            "agreement_iou": args.agreement_iou,
            "max_init_area_ratio": args.max_init_area_ratio,
            "max_last_area_ratio": args.max_last_area_ratio,
            "max_motion_floor": args.max_motion_floor,
            "max_motion_scale": args.max_motion_scale,
        },
        "runtime": {"seconds": round(time.time() - start, 3)},
        "summary": aggregate,
        "submission": submission,
        "results": results,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("m3_audit_json=" + str(args.audit_json), flush=True)
    print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""M1: visibility/identity gate for SAM2 MOSEv2 predictions.

The method is intentionally model-agnostic and reversible.  It reads raw SAM2
label PNG predictions and suppresses object masks that are more consistent with
post-occlusion distractor drift than with a confirmed continuation of the first-
frame instance.

Principle: in semi-supervised VOS, an empty mask during uncertainty is often
preferable to a confident mask on a wrong same-class instance, because a wrong
instance contributes both false positives and false negatives while violating the
identity contract.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:  # optional; if absent, fragmentation checks are skipped.
    from scipy import ndimage as ndi  # type: ignore
except Exception:  # pragma: no cover - depends on runtime env
    ndi = None


@dataclass
class MaskFeature:
    present: bool
    area: int = 0
    bbox: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2 inclusive-exclusive
    centroid: tuple[float, float] | None = None
    hist: np.ndarray | None = None
    largest_component_ratio: float = 1.0


@dataclass
class ObjectState:
    obj_id: int
    init_feature: MaskFeature
    last_confirmed: MaskFeature
    last_frame: int = 0
    last_velocity: tuple[float, float] = (0.0, 0.0)
    invisible_gap: int = 0
    suppressed_frames: list[dict[str, Any]] = field(default_factory=list)
    raw_nonempty: int = 0
    gated_nonempty: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--raw-pred-root", type=Path, default=None)
    p.add_argument("--out-pred-root", type=Path, default=None)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--audit-json", type=Path, default=None)
    # Geometry/identity thresholds. Defaults are identity-protective but not
    # globally destructive: suppression generally requires post-gap uncertainty
    # or multiple independent anomaly signals.
    p.add_argument("--strict-gap", type=int, default=4, help="Consecutive empty raw frames before reappearance is treated as identity-uncertain.")
    p.add_argument("--reappear-dist-frac", type=float, default=0.10, help="Distance as fraction of image diagonal that flags post-gap reappearance as suspicious.")
    p.add_argument("--motion-scale", type=float, default=3.0, help="Multiplier on target scale for normal motion allowance.")
    p.add_argument("--motion-diag-frac", type=float, default=0.025, help="Minimum motion allowance as fraction of image diagonal.")
    p.add_argument("--gap-motion-growth", type=float, default=0.18, help="Motion allowance growth per invisible frame, capped internally.")
    p.add_argument("--min-step-area-ratio", type=float, default=0.10)
    p.add_argument("--max-step-area-ratio", type=float, default=8.0)
    p.add_argument("--max-frame-area-frac", type=float, default=0.55)
    p.add_argument("--appearance-threshold", type=float, default=0.18)
    p.add_argument("--tiny-appearance-pixels", type=int, default=80)
    p.add_argument("--fragment-threshold", type=float, default=0.35)
    p.add_argument("--hist-bins", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def complete_args(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.raw_pred_root = (args.raw_pred_root or ws / "homework" / "pred_sam2_b101").resolve()
    args.out_pred_root = (args.out_pred_root or ws / "homework" / "pred_sam2_m1_visibility").resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    args.audit_json = (args.audit_json or ws / "homework" / "logs" / f"m1_visibility_gate_{stamp}.json").resolve()
    return args


def is_same_or_nested(a: Path, b: Path) -> bool:
    """Return True when either path is the same as or nested under the other."""
    a = a.resolve()
    b = b.resolve()
    if a == b:
        return True
    try:
        a.relative_to(b)
        return True
    except ValueError:
        pass
    try:
        b.relative_to(a)
        return True
    except ValueError:
        return False


def validate_output_root(args: argparse.Namespace) -> None:
    protected = {
        "raw_pred_root": args.raw_pred_root,
        "jpeg_root": args.jpeg_root,
        "ann_root": args.ann_root,
    }
    for name, protected_path in protected.items():
        if is_same_or_nested(args.out_pred_root, protected_path):
            raise ValueError(
                f"unsafe --out-pred-root {args.out_pred_root}: overlaps protected {name} {protected_path}"
            )


def validate_raw_labels(raw: np.ndarray, obj_ids: list[int], path: Path) -> None:
    known = {0, *obj_ids}
    unknown = sorted(int(x) for x in np.unique(raw) if int(x) not in known)
    if unknown:
        raise ValueError(f"{path}: unexpected raw label ids {unknown}; expected subset of {sorted(known)}")


def list_frame_paths(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(f"no frames in {video_dir}")
    return frames


def load_label(path: Path) -> tuple[np.ndarray, list[int] | None]:
    img = Image.open(path)
    palette = img.getpalette()
    arr = np.array(img)
    if arr.ndim != 2:
        raise ValueError(f"expected single-channel label PNG: {path}, got {arr.shape}")
    return arr, palette


def save_label(path: Path, arr: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr, mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def masked_rgb_hist(rgb: np.ndarray, mask: np.ndarray, bins: int) -> np.ndarray | None:
    pixels = rgb[mask]
    if pixels.shape[0] == 0:
        return None
    # Normalize to [0, 1] and use a compact 3D RGB histogram.  RGB is not a
    # perfect appearance model, but it gives an independent low-cost signal.
    vals = pixels.astype(np.float32) / 255.0
    hist, _ = np.histogramdd(vals, bins=(bins, bins, bins), range=((0, 1), (0, 1), (0, 1)))
    hist = hist.astype(np.float32).ravel()
    s = float(hist.sum())
    if s <= 0:
        return None
    return hist / s


def hist_intersection(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.minimum(a, b).sum())


def largest_component_ratio(mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area == 0 or ndi is None:
        return 1.0
    labels, n = ndi.label(mask)
    if n <= 1:
        return 1.0
    counts = np.bincount(labels.ravel())
    if counts.shape[0] <= 1:
        return 1.0
    largest = int(counts[1:].max(initial=0))
    return float(largest / max(area, 1))


def compute_feature(label: np.ndarray, obj_id: int, rgb: np.ndarray | None, bins: int) -> MaskFeature:
    mask = label == obj_id
    ys, xs = np.nonzero(mask)
    area = int(xs.size)
    if area == 0:
        return MaskFeature(False)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    hist = masked_rgb_hist(rgb, mask, bins) if rgb is not None else None
    return MaskFeature(
        present=True,
        area=area,
        bbox=(x1, y1, x2, y2),
        centroid=(float(xs.mean()), float(ys.mean())),
        hist=hist,
        largest_component_ratio=largest_component_ratio(mask),
    )


def bbox_scale(f: MaskFeature) -> float:
    if f.bbox is None:
        return math.sqrt(max(f.area, 1))
    x1, y1, x2, y2 = f.bbox
    return max(float(x2 - x1), float(y2 - y1), math.sqrt(max(f.area, 1)))


def centroid_distance(a: MaskFeature, b: MaskFeature) -> float | None:
    if a.centroid is None or b.centroid is None:
        return None
    return math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1])


def should_suppress(
    state: ObjectState,
    cur: MaskFeature,
    frame_idx: int,
    image_shape: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[bool, list[str], dict[str, float | int | None]]:
    H, W = image_shape
    diag = math.hypot(W, H)
    reasons: list[str] = []
    metrics: dict[str, float | int | None] = {}

    dist = centroid_distance(cur, state.last_confirmed)
    metrics["centroid_dist"] = dist
    scale = max(bbox_scale(state.last_confirmed), bbox_scale(state.init_feature), 1.0)
    vx, vy = state.last_velocity
    velocity = math.hypot(vx, vy)
    gap = state.invisible_gap
    gap_growth = 1.0 + min(gap, 10) * args.gap_motion_growth
    motion_allow = max(args.motion_scale * scale, args.motion_diag_frac * diag) * gap_growth + 1.5 * velocity
    metrics["motion_allow"] = motion_allow
    metrics["gap"] = gap
    if dist is not None and dist > motion_allow:
        reasons.append("motion_jump")

    if state.last_confirmed.area > 0 and cur.area > 0:
        step_ratio = cur.area / max(state.last_confirmed.area, 1)
    else:
        step_ratio = None
    metrics["step_area_ratio"] = step_ratio
    if step_ratio is not None and (step_ratio < args.min_step_area_ratio or step_ratio > args.max_step_area_ratio):
        reasons.append("area_jump")

    frame_area_frac = cur.area / max(H * W, 1)
    metrics["frame_area_frac"] = frame_area_frac
    if frame_area_frac > args.max_frame_area_frac:
        reasons.append("implausibly_large_frame_fraction")

    app_last = hist_intersection(cur.hist, state.last_confirmed.hist)
    app_init = hist_intersection(cur.hist, state.init_feature.hist)
    metrics["appearance_to_last"] = app_last
    metrics["appearance_to_init"] = app_init
    if cur.area >= args.tiny_appearance_pixels:
        known_apps = [x for x in [app_last, app_init] if x is not None]
        if known_apps and max(known_apps) < args.appearance_threshold:
            reasons.append("appearance_drift")

    metrics["largest_component_ratio"] = cur.largest_component_ratio
    if cur.largest_component_ratio < args.fragment_threshold:
        reasons.append("fragmented_mask")

    reappear_dist_limit = args.reappear_dist_frac * diag
    metrics["reappear_dist_limit"] = reappear_dist_limit
    post_gap_suspicious = gap >= args.strict_gap and dist is not None and dist > reappear_dist_limit
    if post_gap_suspicious:
        reasons.append("unconfirmed_far_reappearance")

    # Suppression logic is intentionally conservative except after an invisible
    # gap, where identity uncertainty is the main failure mode.
    suppress = False
    hard = set(reasons)
    if "implausibly_large_frame_fraction" in hard or "fragmented_mask" in hard:
        suppress = True
    elif gap >= args.strict_gap and ("unconfirmed_far_reappearance" in hard or "motion_jump" in hard):
        suppress = True
    elif gap >= args.strict_gap and len(hard.intersection({"area_jump", "appearance_drift"})) >= 2:
        suppress = True
    elif len(hard.intersection({"motion_jump", "area_jump", "appearance_drift"})) >= 2 and gap > 0:
        suppress = True

    return suppress, reasons, metrics


def apply_video(args: argparse.Namespace, video: str) -> dict[str, Any]:
    frame_paths = list_frame_paths(args.jpeg_root / video)
    ann, palette = load_label(args.ann_root / video / "00000.png")
    obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
    if not obj_ids:
        raise ValueError(f"{video}: no foreground ids in first annotation")

    raw_dir = args.raw_pred_root / video
    out_dir = args.out_pred_root / video
    if not raw_dir.is_dir():
        raise FileNotFoundError(raw_dir)
    raw_paths = [raw_dir / f"{p.stem}.png" for p in frame_paths]
    missing = [p for p in raw_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{video}: missing raw predictions: {missing[:5]}")

    if not args.dry_run:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    H, W = ann.shape
    states: dict[int, ObjectState] = {}
    rgb0 = np.array(Image.open(frame_paths[0]).convert("RGB"))
    for obj_id in obj_ids:
        init = compute_feature(ann, obj_id, rgb0, args.hist_bins)
        states[obj_id] = ObjectState(obj_id=obj_id, init_feature=init, last_confirmed=init)

    video_audit: dict[str, Any] = {
        "video": video,
        "frames": len(frame_paths),
        "objects": obj_ids,
        "suppressed_total": 0,
        "per_object": {},
    }

    for frame_idx, (frame_path, raw_path) in enumerate(zip(frame_paths, raw_paths)):
        raw, raw_palette = load_label(raw_path)
        if raw.shape != ann.shape:
            raise ValueError(f"{video}/{raw_path.name}: shape {raw.shape} != annotation shape {ann.shape}")
        validate_raw_labels(raw, obj_ids, raw_path)
        if frame_idx == 0:
            out = ann.copy()
            if not args.dry_run:
                save_label(out_dir / f"{frame_path.stem}.png", out, palette or raw_palette)
            for obj_id, state in states.items():
                state.raw_nonempty += 1
                state.gated_nonempty += 1
            continue

        rgb = np.array(Image.open(frame_path).convert("RGB"))
        out = raw.copy()
        for obj_id, state in states.items():
            cur = compute_feature(raw, obj_id, rgb, args.hist_bins)
            if cur.present:
                state.raw_nonempty += 1
                suppress, reasons, metrics = should_suppress(state, cur, frame_idx, (H, W), args)
                if suppress:
                    out[out == obj_id] = 0
                    event = {
                        "frame": frame_idx,
                        "frame_name": frame_path.stem,
                        "reasons": reasons,
                        "metrics": metrics,
                        "area": cur.area,
                        "bbox": cur.bbox,
                    }
                    state.suppressed_frames.append(event)
                    video_audit["suppressed_total"] += 1
                    state.invisible_gap += 1
                else:
                    if state.last_confirmed.centroid is not None and cur.centroid is not None:
                        dt = max(frame_idx - state.last_frame, 1)
                        state.last_velocity = (
                            (cur.centroid[0] - state.last_confirmed.centroid[0]) / dt,
                            (cur.centroid[1] - state.last_confirmed.centroid[1]) / dt,
                        )
                    state.last_confirmed = cur
                    state.last_frame = frame_idx
                    state.invisible_gap = 0
                    state.gated_nonempty += 1
            else:
                state.invisible_gap += 1
        if not args.dry_run:
            save_label(out_dir / f"{frame_path.stem}.png", out, palette or raw_palette)

    for obj_id, state in states.items():
        video_audit["per_object"][str(obj_id)] = {
            "raw_nonempty": state.raw_nonempty,
            "gated_nonempty": state.gated_nonempty,
            "suppressed": len(state.suppressed_frames),
            "suppressed_frames": state.suppressed_frames,
            "init_area": state.init_feature.area,
            "init_bbox": state.init_feature.bbox,
        }
    return video_audit


def main() -> None:
    args = complete_args(parse_args())
    validate_output_root(args)
    for required in [args.raw_pred_root, args.jpeg_root, args.ann_root]:
        if not required.exists():
            raise FileNotFoundError(required)
    videos = args.videos or sorted(p.name for p in args.raw_pred_root.iterdir() if p.is_dir())
    started = time.time()
    audits = []
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] M1 gate {video}", flush=True)
        audits.append(apply_video(args, video))

    summary = {
        "method": "M1_visibility_identity_gate",
        "workspace": str(args.workspace),
        "raw_pred_root": str(args.raw_pred_root),
        "out_pred_root": str(args.out_pred_root),
        "videos": len(audits),
        "frames": sum(int(a["frames"]) for a in audits),
        "suppressed_total": sum(int(a["suppressed_total"]) for a in audits),
        "seconds": round(time.time() - started, 3),
        "thresholds": {
            "strict_gap": args.strict_gap,
            "reappear_dist_frac": args.reappear_dist_frac,
            "motion_scale": args.motion_scale,
            "motion_diag_frac": args.motion_diag_frac,
            "gap_motion_growth": args.gap_motion_growth,
            "min_step_area_ratio": args.min_step_area_ratio,
            "max_step_area_ratio": args.max_step_area_ratio,
            "max_frame_area_frac": args.max_frame_area_frac,
            "appearance_threshold": args.appearance_threshold,
            "fragment_threshold": args.fragment_threshold,
        },
        "results": audits,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ["method", "videos", "frames", "suppressed_total", "seconds"]}, ensure_ascii=False), flush=True)
    print(f"audit_json={args.audit_json}", flush=True)


if __name__ == "__main__":
    main()

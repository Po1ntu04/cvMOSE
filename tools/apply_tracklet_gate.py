#!/usr/bin/env python3
"""M1.1: training-free tracklet-level identity gate for SAM2 MOSEv2 masks.

M1 was intentionally broad: it suppressed frame-by-frame after a suspicious
post-occlusion reappearance.  That was useful as a probe, but visually it
false-emptied camera-motion / edge-target cases.  M1.1 keeps the same core
training-free principle while changing the decision unit:

- build raw non-empty tracklets separated by *raw* empty gaps;
- judge each post-gap tracklet once using multi-frame evidence;
- protect edge-truncated and large/camera-motion-prone targets from far-only
  suppression;
- suppress only when a post-gap tracklet has both long-gap/far-reappearance and
  persistent identity-inconsistent size/shape evidence.

This is still a reversible post-process over existing SAM2 label PNGs.  It does
not use learned training, fine-tuning, or SAM2 confidence/logits.
"""
from __future__ import annotations

import argparse
import json
import math
import contextlib
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image

# Reuse low-level image/mask utilities, not M1's suppression policy.
from apply_visibility_gate import (  # type: ignore
    MaskFeature,
    compute_feature,
    hist_intersection,
    is_same_or_nested,
    list_frame_paths,
    load_label,
    save_label,
    validate_raw_labels,
)


@dataclass
class TrackletDecision:
    start: int
    end: int
    gap: int
    suppress: bool
    reasons: list[str]
    metrics: dict[str, Any]


@dataclass
class ObjectAudit:
    obj_id: int
    raw_nonempty: int = 0
    gated_nonempty: int = 0
    suppressed_frames: list[int] = field(default_factory=list)
    tracklets: list[dict[str, Any]] = field(default_factory=list)
    init_area: int = 0
    init_bbox: tuple[int, int, int, int] | None = None
    init_area_frac: float = 0.0
    edge_protected: bool = False
    large_protected: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--raw-pred-root", type=Path, default=None)
    p.add_argument("--out-pred-root", type=Path, default=None)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--audit-json", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")

    # Tracklet evidence thresholds.  Defaults deliberately avoid M1's far-only
    # suppression: a post-gap tracklet must be far AND persistently size/shape
    # inconsistent, unless it is a protected edge/large target.
    p.add_argument("--strict-gap", type=int, default=4)
    p.add_argument("--far-dist-frac", type=float, default=0.14)
    p.add_argument("--area-grow-ratio", type=float, default=8.0)
    p.add_argument("--area-shrink-ratio", type=float, default=0.25)
    p.add_argument("--large-target-frac", type=float, default=0.05)
    p.add_argument("--edge-margin-frac", type=float, default=0.025)
    p.add_argument("--min-tracklet-frames", type=int, default=2)
    p.add_argument("--evidence-window", type=int, default=5)
    p.add_argument("--reject-bridge-gap", type=int, default=3)
    p.add_argument("--reject-continue-dist-frac", type=float, default=0.18)
    p.add_argument("--fragment-threshold", type=float, default=0.35)
    p.add_argument("--max-frame-area-frac", type=float, default=0.55)
    p.add_argument("--hist-bins", type=int, default=4)

    # Optional frozen-SAM2 cycle consistency verification.  This remains
    # training-free: a candidate tracklet is prompted as a mask at its
    # reappearance frame and propagated backward to frame 0.  Low IoU with the
    # first-frame GT is stronger evidence that the candidate is not the original
    # instance.
    p.add_argument("--cycle-verify", action="store_true")
    p.add_argument("--cycle-iou-threshold", type=float, default=0.10)
    p.add_argument("--sam2-root", type=Path, default=None)
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-video-to-cpu", dest="offload_video_to_cpu", action="store_false")
    p.add_argument("--offload-state-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-state-to-cpu", dest="offload_state_to_cpu", action="store_false")
    return p.parse_args()


def complete_args(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.raw_pred_root = (args.raw_pred_root or ws / "homework" / "pred_sam2_b101").resolve()
    args.out_pred_root = (args.out_pred_root or ws / "homework" / "pred_sam2_m11_tracklet").resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    args.audit_json = (args.audit_json or ws / "homework" / "logs" / f"m11_tracklet_gate_{stamp}.json").resolve()
    args.sam2_root = (args.sam2_root or ws / "5_19" / "sam2").resolve()
    args.checkpoint = (args.checkpoint or ws / "5_19" / "data" / "sam2" / "sam2.1_hiera_base_plus.pt").resolve()
    return args


def validate_output_root(args: argparse.Namespace) -> None:
    for name, protected_path in {
        "raw_pred_root": args.raw_pred_root,
        "jpeg_root": args.jpeg_root,
        "ann_root": args.ann_root,
    }.items():
        if is_same_or_nested(args.out_pred_root, protected_path):
            raise ValueError(f"unsafe --out-pred-root {args.out_pred_root}: overlaps protected {name} {protected_path}")


def finite_median(vals: list[float | None]) -> float | None:
    xs = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if not xs:
        return None
    return float(median(xs))


def centroid_dist(a: MaskFeature, b: MaskFeature) -> float | None:
    if a.centroid is None or b.centroid is None:
        return None
    return math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1])


def touches_edge(f: MaskFeature, image_shape: tuple[int, int], edge_margin_frac: float) -> bool:
    if f.bbox is None:
        return False
    h, w = image_shape
    margin = max(2, int(round(edge_margin_frac * math.hypot(w, h))))
    x1, y1, x2, y2 = f.bbox
    return x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin


def build_tracklets(present: list[bool]) -> list[tuple[int, int, int]]:
    """Return [(start, end_inclusive, raw_empty_gap_before), ...]."""
    out: list[tuple[int, int, int]] = []
    i = 1  # frame 0 is GT anchor, not a candidate tracklet.
    last_end = 0
    n = len(present)
    while i < n:
        while i < n and not present[i]:
            i += 1
        if i >= n:
            break
        start = i
        while i + 1 < n and present[i + 1]:
            i += 1
        end = i
        gap = max(start - last_end - 1, 0)
        out.append((start, end, gap))
        last_end = end
        i += 1
    return out


def decide_tracklet(
    start: int,
    end: int,
    gap: int,
    features: list[MaskFeature],
    anchor: MaskFeature,
    init_feature: MaskFeature,
    image_shape: tuple[int, int],
    args: argparse.Namespace,
) -> TrackletDecision:
    h, w = image_shape
    diag = math.hypot(w, h)
    window = features[start : min(end + 1, start + max(args.evidence_window, 1))]
    length = end - start + 1
    init_area_frac = init_feature.area / max(h * w, 1)
    edge_protected = touches_edge(init_feature, image_shape, args.edge_margin_frac)
    large_protected = init_area_frac >= args.large_target_frac
    protected = edge_protected or large_protected

    dists = [centroid_dist(f, anchor) for f in window]
    dist_frac = finite_median([None if d is None else d / diag for d in dists])
    area_ratios = [f.area / max(anchor.area, 1) for f in window if f.present and anchor.area > 0]
    area_ratio = finite_median(area_ratios)
    app_to_anchor = finite_median([hist_intersection(f.hist, anchor.hist) for f in window])
    app_to_init = finite_median([hist_intersection(f.hist, init_feature.hist) for f in window])
    app_best = finite_median([x for x in [app_to_anchor, app_to_init] if x is not None])
    component_ratio = finite_median([f.largest_component_ratio for f in window])
    frame_area_frac = finite_median([f.area / max(h * w, 1) for f in window])

    internal_steps = []
    for prev, cur in zip(window, window[1:]):
        d = centroid_dist(cur, prev)
        if d is not None:
            internal_steps.append(d / diag)
    internal_step_frac = finite_median(internal_steps)

    far = dist_frac is not None and dist_frac >= args.far_dist_frac
    area_mismatch = area_ratio is not None and (
        area_ratio >= args.area_grow_ratio or area_ratio <= args.area_shrink_ratio
    )
    fragmented = component_ratio is not None and component_ratio < args.fragment_threshold
    huge = frame_area_frac is not None and frame_area_frac > args.max_frame_area_frac
    has_tracklet_evidence = length >= args.min_tracklet_frames
    post_gap = gap >= args.strict_gap

    reasons: list[str] = []
    if post_gap:
        reasons.append("post_raw_empty_gap")
    if far:
        reasons.append("far_from_last_confirmed_tracklet")
    if area_mismatch:
        reasons.append("persistent_area_mismatch")
    if fragmented:
        reasons.append("fragmented_tracklet")
    if huge:
        reasons.append("implausibly_large_tracklet")
    if protected:
        reasons.append("protected_edge_or_large_target")
    if not has_tracklet_evidence:
        reasons.append("insufficient_tracklet_evidence")

    # Training-free policy: suppress only high-precision identity-risk tracklets.
    # Far-only is explicitly not enough, because it caused M1's camera-motion
    # false empties.  Edge-truncated and large targets are protected unless the
    # mask itself looks corrupt by independent signals.
    suppress = False
    if post_gap and has_tracklet_evidence:
        if huge and not protected:
            suppress = True
        elif far and area_mismatch and not protected:
            suppress = True
        elif far and fragmented and area_mismatch:
            suppress = True

    metrics = {
        "length": length,
        "gap": gap,
        "dist_frac_median": dist_frac,
        "area_ratio_median": area_ratio,
        "appearance_to_anchor_median": app_to_anchor,
        "appearance_to_init_median": app_to_init,
        "appearance_best_median": app_best,
        "component_ratio_median": component_ratio,
        "frame_area_frac_median": frame_area_frac,
        "internal_step_frac_median": internal_step_frac,
        "init_area_frac": init_area_frac,
        "edge_protected": edge_protected,
        "large_protected": large_protected,
        "protected": protected,
    }
    return TrackletDecision(start, end, gap, suppress, reasons, metrics)



class CycleVerifier:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for cycle verification but unavailable")
        if not args.sam2_root.exists():
            raise FileNotFoundError(args.sam2_root)
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)
        sys.path.insert(0, str(args.sam2_root))
        from sam2.build_sam import build_sam2_video_predictor

        self.args = args
        self.torch = torch
        self.predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=args.device)
        self.predictor.eval()
        self.cache: dict[tuple[str, int, int], dict[str, Any]] = {}

    def verify(
        self,
        video: str,
        video_dir: Path,
        frame_idx: int,
        obj_id: int,
        candidate_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> dict[str, Any]:
        key = (video, obj_id, frame_idx)
        if key in self.cache:
            return self.cache[key]
        torch = self.torch
        state = self.predictor.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=self.args.offload_video_to_cpu,
            offload_state_to_cpu=self.args.offload_state_to_cpu,
        )
        self.predictor.reset_state(state)
        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if str(self.args.device).startswith("cuda") and torch.cuda.is_available()
            else contextlib.nullcontext()
        )
        got_frame0 = False
        pred0 = None
        with torch.inference_mode(), autocast_ctx:
            self.predictor.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=1,
                mask=candidate_mask.astype(bool),
            )
            for out_frame_idx, _cur_obj_ids, mask_logits in self.predictor.propagate_in_video(
                state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=frame_idx + 1,
                reverse=True,
            ):
                if int(out_frame_idx) == 0:
                    logits = mask_logits.detach().float().cpu().numpy()
                    if logits.ndim == 4:
                        logits = logits[0, 0]
                    elif logits.ndim == 3:
                        logits = logits[0]
                    pred0 = logits > 0
                    got_frame0 = True
                    break
        if pred0 is None:
            iou = 0.0
            inter = union = 0
        else:
            gt = gt_mask.astype(bool)
            inter = int(np.logical_and(pred0, gt).sum())
            union = int(np.logical_or(pred0, gt).sum())
            iou = float(inter / union) if union else 0.0
        result = {
            "cycle_iou_to_first_gt": iou,
            "cycle_intersection": inter,
            "cycle_union": union,
            "cycle_got_frame0": got_frame0,
            "cycle_prompt_frame": frame_idx,
        }
        self.cache[key] = result
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


def apply_video(args: argparse.Namespace, video: str, cycle_verifier: CycleVerifier | None = None) -> dict[str, Any]:
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

    raws: list[np.ndarray] = []
    raw_palettes: list[list[int] | None] = []
    rgbs: list[np.ndarray] = []
    for frame_path, raw_path in zip(frame_paths, raw_paths):
        raw, raw_palette = load_label(raw_path)
        if raw.shape != ann.shape:
            raise ValueError(f"{video}/{raw_path.name}: shape {raw.shape} != annotation shape {ann.shape}")
        validate_raw_labels(raw, obj_ids, raw_path)
        raws.append(raw)
        raw_palettes.append(raw_palette)
        rgbs.append(np.array(Image.open(frame_path).convert("RGB")))

    h, w = ann.shape
    features_by_obj: dict[int, list[MaskFeature]] = {}
    for obj_id in obj_ids:
        feats: list[MaskFeature] = []
        for i, raw in enumerate(raws):
            label = ann if i == 0 else raw
            feats.append(compute_feature(label, obj_id, rgbs[i], args.hist_bins))
        features_by_obj[obj_id] = feats

    # Output starts as raw SAM2, with exact GT restored at frame 0.
    outs = [x.copy() for x in raws]
    outs[0] = ann.copy()

    video_audit: dict[str, Any] = {
        "video": video,
        "frames": len(frame_paths),
        "objects": obj_ids,
        "suppressed_total": 0,
        "per_object": {},
    }

    for obj_id in obj_ids:
        feats = features_by_obj[obj_id]
        present = [f.present for f in feats]
        init_feature = feats[0]
        anchor = init_feature
        audit = ObjectAudit(
            obj_id=obj_id,
            raw_nonempty=sum(1 for x in present if x),
            init_area=init_feature.area,
            init_bbox=init_feature.bbox,
            init_area_frac=init_feature.area / max(h * w, 1),
            edge_protected=touches_edge(init_feature, (h, w), args.edge_margin_frac),
            large_protected=(init_feature.area / max(h * w, 1)) >= args.large_target_frac,
        )

        # Frame 0 is always kept.
        audit.gated_nonempty = 1 if present[0] else 0
        rejected_tail: MaskFeature | None = None
        for start, end, gap in build_tracklets(present):
            decision = decide_tracklet(start, end, gap, feats, anchor, init_feature, (h, w), args)

            # If a rejected candidate fragments into adjacent raw tracklets, keep
            # rejecting the short-gap continuation. This is not the M1 latch: the
            # accepted anchor is still not updated by rejected candidates, and
            # the bridge only applies across a small raw-empty gap.
            if not decision.suppress and rejected_tail is not None and gap <= args.reject_bridge_gap:
                h_img, w_img = ann.shape
                diag = math.hypot(w_img, h_img)
                cont_dist = finite_median([
                    None if (d := centroid_dist(f, rejected_tail)) is None else d / diag
                    for f in feats[start : min(end + 1, start + max(args.evidence_window, 1))]
                ])
                far_from_anchor = decision.metrics.get("dist_frac_median")
                if (cont_dist is not None and cont_dist <= args.reject_continue_dist_frac) or (
                    far_from_anchor is not None and far_from_anchor >= args.far_dist_frac
                ):
                    decision.suppress = True
                    decision.reasons.append("rejected_tracklet_continuation")
                    decision.metrics["dist_frac_to_rejected_median"] = cont_dist

            if decision.suppress and cycle_verifier is not None:
                cycle = cycle_verifier.verify(
                    video=video,
                    video_dir=args.jpeg_root / video,
                    frame_idx=start,
                    obj_id=obj_id,
                    candidate_mask=(raws[start] == obj_id),
                    gt_mask=(ann == obj_id),
                )
                decision.metrics.update(cycle)
                if cycle["cycle_iou_to_first_gt"] >= args.cycle_iou_threshold:
                    decision.suppress = False
                    decision.reasons.append("cycle_consistent_keep")
                else:
                    decision.reasons.append("cycle_inconsistent_reject")

            event = {
                "start": start,
                "end": end,
                "gap": gap,
                "suppress": decision.suppress,
                "reasons": decision.reasons,
                "metrics": decision.metrics,
            }
            audit.tracklets.append(event)
            if decision.suppress:
                for frame_idx in range(start, end + 1):
                    outs[frame_idx][outs[frame_idx] == obj_id] = 0
                    audit.suppressed_frames.append(frame_idx)
                video_audit["suppressed_total"] += end - start + 1
                rejected_tail = feats[end]
            else:
                for frame_idx in range(start, end + 1):
                    audit.gated_nonempty += 1
                # Update anchor only after an accepted raw tracklet, preventing
                # suppressed candidates from poisoning future identity state.
                anchor = feats[end]
                rejected_tail = None

        video_audit["per_object"][str(obj_id)] = {
            "raw_nonempty": audit.raw_nonempty,
            "gated_nonempty": audit.gated_nonempty,
            "suppressed": len(audit.suppressed_frames),
            "suppressed_frames": audit.suppressed_frames,
            "tracklets": audit.tracklets,
            "init_area": audit.init_area,
            "init_bbox": audit.init_bbox,
            "init_area_frac": audit.init_area_frac,
            "edge_protected": audit.edge_protected,
            "large_protected": audit.large_protected,
        }

    if not args.dry_run:
        for frame_path, out, raw_palette in zip(frame_paths, outs, raw_palettes):
            save_label(out_dir / f"{frame_path.stem}.png", out, palette or raw_palette)

    return video_audit


def main() -> None:
    args = complete_args(parse_args())
    validate_output_root(args)
    for required in [args.raw_pred_root, args.jpeg_root, args.ann_root]:
        if not required.exists():
            raise FileNotFoundError(required)
    cycle_verifier = CycleVerifier(args) if args.cycle_verify else None
    videos = args.videos or sorted(p.name for p in args.raw_pred_root.iterdir() if p.is_dir())
    started = time.time()
    audits = []
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] M1.1 tracklet gate {video}", flush=True)
        audits.append(apply_video(args, video, cycle_verifier=cycle_verifier))

    summary = {
        "method": "M1.1_training_free_tracklet_identity_gate",
        "workspace": str(args.workspace),
        "raw_pred_root": str(args.raw_pred_root),
        "out_pred_root": str(args.out_pred_root),
        "videos": len(audits),
        "frames": sum(int(a["frames"]) for a in audits),
        "suppressed_total": sum(int(a["suppressed_total"]) for a in audits),
        "seconds": round(time.time() - started, 3),
        "thresholds": {
            "strict_gap": args.strict_gap,
            "far_dist_frac": args.far_dist_frac,
            "area_grow_ratio": args.area_grow_ratio,
            "area_shrink_ratio": args.area_shrink_ratio,
            "large_target_frac": args.large_target_frac,
            "edge_margin_frac": args.edge_margin_frac,
            "min_tracklet_frames": args.min_tracklet_frames,
            "evidence_window": args.evidence_window,
            "reject_bridge_gap": args.reject_bridge_gap,
            "reject_continue_dist_frac": args.reject_continue_dist_frac,
            "fragment_threshold": args.fragment_threshold,
            "max_frame_area_frac": args.max_frame_area_frac,
            "cycle_verify": args.cycle_verify,
            "cycle_iou_threshold": args.cycle_iou_threshold,
        },
        "results": audits,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ["method", "videos", "frames", "suppressed_total", "seconds"]}, ensure_ascii=False), flush=True)
    print(f"audit_json={args.audit_json}", flush=True)


if __name__ == "__main__":
    main()

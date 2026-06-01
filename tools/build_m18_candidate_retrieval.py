#!/usr/bin/env python3
"""Build M18 distractor-aware candidate retrieval cards.

This is a lightweight, auditable first pass over existing prediction roots.  It
uses cheap RGB/shape descriptors by default, but applies the same identity logic
as future DINO/SAM2 descriptor routes:

    score = positive similarity - distractor similarity
            + temporal story compatibility
            + source independence bonus
            - composite/background risk

The tool does not modify predictions or ledgers unless a downstream script
explicitly consumes the JSON promotion cards.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.candidate_pool import (  # noqa: E402
    connected_components,
    descriptor,
    label_path,
    list_frames,
    load_label,
    load_rgb,
    mask_iou,
    mask_stats,
    parse_source_roots,
    ring_mask,
)
from cvmose.retrieval_memory import (  # noqa: E402
    CandidateSignals,
    IdentityMemory,
    PromotionPolicy,
    cheap_story_compatibility,
    descriptor_item_from_vector,
    risk_from_area_ratio,
    score_candidate,
)
from cvmose.temporal_ledger import ledger_key, load_ledgers  # noqa: E402


def parse_target(text: str) -> tuple[str, int]:
    video, obj = text.split(":", 1)
    return video, int(obj.replace("obj", ""))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--ledger-dir", type=Path, default=Path("artifacts/m18_temporal_ledger"))
    p.add_argument("--source-root", action="append", default=[], help="name=/path; repeatable")
    p.add_argument("--targets", nargs="*", default=["q0sizv6m:2", "8jsm23a7:1", "r13u5z4y:1"])
    p.add_argument("--out-json", type=Path, default=Path("artifacts/m18_candidate_retrieval/candidates.json"))
    p.add_argument("--out-csv", type=Path, default=Path("artifacts/m18_candidate_retrieval/candidates.csv"))
    p.add_argument("--component-min-area", type=int, default=12)
    p.add_argument("--component-max-count", type=int, default=8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-frames-per-object", type=int, default=80)
    p.add_argument("--same-class-min-margin", type=float, default=0.18)
    p.add_argument("--min-margin", type=float, default=0.08)
    return p.parse_args()


def frame_indices_from_ledger(ledger: dict[str, Any], frame_count: int, max_frames: int) -> list[int]:
    wanted: list[int] = []
    for entry in ledger.get("event_story", []):
        allowed = str(entry.get("allowed_output", "review_only"))
        if allowed == "keep_current":
            continue
        frames = entry.get("frames")
        if not (isinstance(frames, list) and len(frames) == 2):
            continue
        start, end = max(1, int(frames[0])), min(frame_count - 1, int(frames[1]))
        for idx in range(start, end + 1):
            wanted.append(idx)
    for win in ledger.get("next_review_windows", []):
        frames = win.get("frames")
        if isinstance(frames, list):
            if len(frames) == 2 and all(isinstance(x, int) for x in frames):
                for idx in range(max(1, frames[0]), min(frame_count - 1, frames[1]) + 1):
                    wanted.append(idx)
            else:
                for idx in frames:
                    if isinstance(idx, int) and 1 <= idx < frame_count:
                        wanted.append(idx)
    unique = sorted(dict.fromkeys(wanted))
    if len(unique) <= max_frames:
        return unique
    # Preserve endpoints and evenly sample the rest to keep the artifact compact.
    picks = np.linspace(0, len(unique) - 1, max_frames).round().astype(int).tolist()
    return sorted(dict.fromkeys(unique[i] for i in picks))


def box_to_mask(box: list[int] | None, shape: tuple[int, int]) -> np.ndarray | None:
    if not box or len(box) != 4:
        return None
    h, w = shape
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def add_memory_desc(memory: IdentityMemory, kind: str, item_id: str, source: str, frame_idx: int, rgb: np.ndarray, mask: np.ndarray, tags: list[str] | None = None, weight: float = 1.0) -> bool:
    desc = descriptor(rgb, mask)
    if desc is None:
        return False
    item = descriptor_item_from_vector(item_id, desc, source=source, frame_idx=frame_idx, tags=tags or [], weight=weight)
    if kind == "positive":
        memory.add_positive(item)
    elif kind == "distractor":
        memory.add_distractor(item)
    else:
        memory.add_rejected(item)
    return True


def resolve_label_source(source: str, labels: dict[str, list[np.ndarray | None]]) -> str | None:
    """Resolve a ledger source name to an available prediction-root name.

    Ledger entries are human-readable and sometimes include suffixes such as
    ``m15_safe_baseline`` or ``m13_zofficial_balanced_official_large_interval``.
    The retrieval pass should still be able to reuse the corresponding local
    root (``m15_safe`` / ``official_large``) as identity memory when available.
    """
    source = str(source)
    if source in labels:
        return source
    for name in labels:
        if name in source or source in name:
            return name
    return None


def sampled_memory_frames(frames_spec: list[Any], frame_count: int) -> list[int]:
    """Return compact, deterministic samples from a ledger memory frame spec."""
    ints = [int(x) for x in frames_spec if isinstance(x, int)]
    if not ints:
        return [0]
    if len(ints) == 2 and ints[0] <= ints[1]:
        start = max(0, min(frame_count - 1, ints[0]))
        end = max(0, min(frame_count - 1, ints[1]))
        mid = (start + end) // 2
        return sorted(dict.fromkeys([start, mid, end]))
    return sorted(dict.fromkeys(max(0, min(frame_count - 1, x)) for x in ints))[:5]


def add_ledger_memory_item(
    memory: IdentityMemory,
    audit: list[dict[str, Any]],
    *,
    kind: str,
    item: dict[str, Any],
    labels: dict[str, list[np.ndarray | None]],
    frames: list[Path],
    ann_shape: tuple[int, int],
    obj_id: int,
    component_min_area: int,
    default_weight: float,
) -> None:
    source = str(item.get("source", "ledger"))
    root_name = resolve_label_source(source, labels)
    samples = sampled_memory_frames(item.get("frames") or [], len(frames))
    added = 0
    for idx in samples:
        mask = box_to_mask(item.get("bbox"), ann_shape)
        if mask is None and root_name is not None and idx < len(labels[root_name]) and labels[root_name][idx] is not None:
            mask = labels[root_name][idx] == int(obj_id)
        if mask is None or int(mask.sum()) < component_min_area:
            continue
        rgb = load_rgb(frames[idx])
        memory_kind = "positive" if kind == "positive" else ("rejected" if kind == "rejected" else "distractor")
        ok = add_memory_desc(
            memory,
            memory_kind,
            f"{item.get('memory_id', 'ledger_memory')}:{idx}",
            root_name or source,
            idx,
            rgb,
            mask,
            item.get("tags") or [],
            float(item.get("weight") or default_weight),
        )
        if ok:
            added += 1
            audit.append(
                {
                    "kind": memory_kind,
                    "source": source,
                    "resolved_source": root_name,
                    "frame_idx": idx,
                    "area": int(mask.sum()),
                    "from_ledger": True,
                    "memory_id": item.get("memory_id"),
                }
            )
    if added == 0 and item.get("bbox") is None and root_name is None:
        audit.append(
            {
                "kind": kind,
                "source": source,
                "from_ledger": True,
                "skipped": "no_bbox_and_no_matching_prediction_root",
                "memory_id": item.get("memory_id"),
            }
        )


def build_memory(
    workspace: Path,
    frames: list[Path],
    ann: np.ndarray,
    obj_id: int,
    ledger: dict[str, Any],
    labels: dict[str, list[np.ndarray | None]],
    component_min_area: int,
) -> tuple[IdentityMemory, list[dict[str, Any]]]:
    memory = IdentityMemory()
    audit: list[dict[str, Any]] = []
    init = ann == int(obj_id)
    rgb0 = load_rgb(frames[0])
    if add_memory_desc(memory, "positive", "first_frame_gt", "first_frame_gt", 0, rgb0, init, ["first_gt"], 1.0):
        audit.append({"kind": "positive", "source": "first_frame_gt", "frame_idx": 0, "area": int(init.sum())})
    ring = ring_mask(init)
    if int(ring.sum()) >= component_min_area and add_memory_desc(memory, "distractor", "first_frame_context_ring", "context_ring", 0, rgb0, ring, ["context"], 0.85):
        audit.append({"kind": "distractor", "source": "context_ring", "frame_idx": 0, "area": int(ring.sum())})
    for other in [int(x) for x in np.unique(ann) if int(x) not in {0, int(obj_id)}]:
        om = ann == other
        if int(om.sum()) >= component_min_area and add_memory_desc(memory, "distractor", f"first_frame_other_{other}", "first_frame_other_object", 0, rgb0, om, ["other_object"], 1.0):
            audit.append({"kind": "distractor", "source": f"first_frame_other:{other}", "frame_idx": 0, "area": int(om.sum())})
    # Ledger positive/distractor memories are the main M18 upgrade over the
    # earlier M5R/M8 pass: hidden-confirmed later anchors (q0/amfdu/z6) and
    # known same-class failures must affect scoring, not only the first frame.
    for item in ledger.get("positive_memory", []):
        if str(item.get("source")) == "first_frame_gt":
            continue
        add_ledger_memory_item(
            memory,
            audit,
            kind="positive",
            item=item,
            labels=labels,
            frames=frames,
            ann_shape=ann.shape,
            obj_id=obj_id,
            component_min_area=component_min_area,
            default_weight=0.95,
        )
    for item in ledger.get("distractor_memory", []):
        add_ledger_memory_item(
            memory,
            audit,
            kind="distractor",
            item=item,
            labels=labels,
            frames=frames,
            ann_shape=ann.shape,
            obj_id=obj_id,
            component_min_area=component_min_area,
            default_weight=0.80,
        )
    # Approved/promoted current-best windows can provide additional positives when labels exist locally.
    for cand in ledger.get("candidate_anchors", []):
        if cand.get("decision") != "promote" or cand.get("review_status") != "approved":
            continue
        src = str(cand.get("source"))
        if src not in labels:
            continue
        idx = int(cand.get("frame_idx") or 0)
        if 0 <= idx < len(frames) and labels[src][idx] is not None:
            mask = labels[src][idx] == int(obj_id)
            if int(mask.sum()) >= component_min_area:
                rgb = load_rgb(frames[idx])
                if add_memory_desc(memory, "positive", f"promoted_{src}_{idx}", src, idx, rgb, mask, ["ledger_promoted"], 0.95):
                    audit.append({"kind": "positive", "source": src, "frame_idx": idx, "area": int(mask.sum()), "from_ledger": True})
    return memory, audit


def load_labels(root: Path, video: str, frames: list[Path], shape: tuple[int, int]) -> list[np.ndarray | None]:
    out: list[np.ndarray | None] = []
    for frame in frames:
        p = label_path(root, video, frame.stem)
        out.append(load_label(p, shape) if p.is_file() else None)
    return out


def candidate_masks(
    lab: np.ndarray,
    obj_id: int,
    *,
    component_min_area: int,
    component_max_count: int,
) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    obj = lab == int(obj_id)
    if int(obj.sum()) >= component_min_area:
        out.append(("object_id", obj))
    # Include foreground components to catch cases where the target-looking item
    # was assigned another id or merged into a composite proposal.
    for idx, comp in enumerate(connected_components(lab > 0, component_min_area, component_max_count)):
        out.append((f"anyfg:{idx}", comp))
    return out


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = ws / "homework" / "JPEGImages"
    ann_root = ws / "homework" / "Annotations"
    ledgers = load_ledgers(args.ledger_dir)
    roots = parse_source_roots(args.source_root)
    if not roots:
        # Useful local defaults; missing paths are ignored by parse_source_roots.
        roots = parse_source_roots(
            [
                "m15_safe=/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m15_layered_safe",
                "m17_q0_full_box=/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m17_q0_full_box",
                "m16mask=/home/yu/projects/cv/cvMOSE/artifacts/m16_remaining/remote_downloads/pred_m16_remaining_maskbox",
                "m14_from8_noclip=/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m14_zofficial_amfdu_from8_noclip",
                "official_large=/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_official_large_mosev2",
            ]
        )
    policy = PromotionPolicy(min_margin=args.min_margin, same_class_min_margin=args.same_class_min_margin)
    results: dict[str, Any] = {
        "method": "m18_candidate_retrieval",
        "workspace": str(ws),
        "ledger_dir": str(args.ledger_dir),
        "source_roots": {r.name: str(r.root) for r in roots},
        "objects": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for target in args.targets:
        video, obj_id = parse_target(target)
        key = ledger_key(video, obj_id)
        ledger = ledgers.get(key)
        if ledger is None:
            raise SystemExit(f"missing ledger for {key} in {args.ledger_dir}")
        frames = list_frames(jpeg_root / video)
        ann = load_label(ann_root / video / "00000.png")
        if ann is None:
            continue
        shape = ann.shape
        labels = {
            root.name: load_labels(root.root, video, frames, shape)
            for root in roots
            if (root.root / video).is_dir()
        }
        memory, bank_audit = build_memory(ws, frames, ann, obj_id, ledger, labels, args.component_min_area)
        idxs = frame_indices_from_ledger(ledger, len(frames), args.max_frames_per_object)
        init_area = int((ann == int(obj_id)).sum())
        object_candidates: list[dict[str, Any]] = []
        seen_masks: dict[int, list[np.ndarray]] = {}
        for idx in idxs:
            rgb = load_rgb(frames[idx])
            for root_name, seq in labels.items():
                if idx >= len(seq) or seq[idx] is None:
                    continue
                lab = seq[idx]
                for mask_kind, mask in candidate_masks(
                    lab,
                    obj_id,
                    component_min_area=args.component_min_area,
                    component_max_count=args.component_max_count,
                ):
                    area = int(mask.sum())
                    if area < args.component_min_area:
                        continue
                    # Deduplicate per frame across roots; keep distinct components if IoU is low.
                    old = seen_masks.setdefault(idx, [])
                    if any(mask_iou(mask, prev) > 0.985 for prev in old):
                        continue
                    old.append(mask.copy())
                    desc = descriptor(rgb, mask)
                    if desc is None:
                        continue
                    stats = mask_stats(mask)
                    story_score, story_tags = cheap_story_compatibility(idx, ledger.get("event_story", []))
                    area_ratio_init = area / max(1.0, float(init_area))
                    composite_risk, risk_tags = risk_from_area_ratio(area_ratio_init)
                    temporal_votes = 0
                    for j in (idx - 1, idx + 1, idx + 2):
                        if 0 <= j < len(frames) and root_name in labels and labels[root_name][j] is not None:
                            near = labels[root_name][j] == int(obj_id)
                            if int(near.sum()) >= args.component_min_area:
                                temporal_votes += 1
                    signals = CandidateSignals(
                        candidate_id=f"{key}:{idx}:{root_name}:{mask_kind}",
                        vector=[float(x) for x in desc.tolist()],
                        source=f"{root_name}:{mask_kind}",
                        frame_idx=idx,
                        temporal_story_compatibility=story_score,
                        source_independence_bonus=0.08 if root_name not in {ledger.get("current_best_source"), "m15_safe"} else 0.0,
                        composite_background_risk=composite_risk,
                        temporal_consistency_votes=temporal_votes,
                        qwen_tracklet_support=False,
                        independent_source_agreement=temporal_votes >= 2,
                        same_class_dense=bool(ledger.get("target_profile", {}).get("same_class_dense")),
                        state=str(ledger.get("current_state", "AMBIGUOUS")),
                        hard_negative_hit=False,
                        risk_tags=risk_tags + story_tags,
                        evidence_tags=story_tags if story_score > 0.25 else [],
                    )
                    scored = score_candidate(signals, memory, policy)
                    current_name = str(ledger.get("current_best_source", ""))
                    is_current_reference = (
                        root_name == current_name
                        or root_name in current_name
                        or current_name in root_name
                        or root_name == "m15_safe"
                    )
                    if is_current_reference and scored.decision == "promote":
                        scored.decision = "output_only"
                        scored.reasons.append("current_best_reference_not_new_anchor")
                    if ledger.get("review_status") != "approved" and scored.decision == "promote":
                        scored.decision = "output_only"
                        scored.reasons.append("ledger_not_review_approved_no_auto_promote")
                    rec = {
                        "video": video,
                        "obj_id": int(obj_id),
                        "frame_idx": int(idx),
                        "source": root_name,
                        "mask_kind": mask_kind,
                        "area": area,
                        "bbox": stats.bbox,
                        "centroid": stats.centroid,
                        "area_ratio_init": round(float(area_ratio_init), 5),
                        "story_tags": story_tags,
                        "risk_tags": risk_tags,
                        "temporal_votes": temporal_votes,
                        "score_card": scored.to_json(),
                        "promotion_card": {
                            "video": video,
                            "obj_id": int(obj_id),
                            "frame_idx": int(idx),
                            "candidate_source": f"{root_name}:{mask_kind}",
                            "positive_evidence": ["cheap_descriptor_positive_pool", *signals.evidence_tags],
                            "negative_evidence": ["distractor_pool_checked", *risk_tags],
                            "temporal_story_compatibility": round(float(story_score), 5),
                            "descriptor_margin": round(float(scored.margin), 5),
                            "risk_tags": risk_tags + story_tags,
                            "decision": scored.decision,
                            "decision_reason": "; ".join(scored.reasons),
                            "review_status": "needs_more_evidence" if scored.decision != "reject" else "rejected",
                        },
                    }
                    object_candidates.append(rec)
        object_candidates.sort(
            key=lambda r: (
                r["score_card"]["decision"] == "promote",
                r["score_card"]["decision"] == "output_only",
                float(r["score_card"]["score"]),
                float(r["score_card"]["margin"]),
            ),
            reverse=True,
        )
        top = object_candidates[: args.top_k]
        results["objects"][key] = {
            "ledger_review_status": ledger.get("review_status"),
            "current_best_source": ledger.get("current_best_source"),
            "frames_considered": idxs,
            "identity_bank": {
                "positive": len(memory.positive_pool),
                "distractor": len(memory.distractor_pool),
                "rejected": len(memory.rejected_pool),
                "audit": bank_audit[:80],
            },
            "candidate_count": len(object_candidates),
            "top_candidates": top,
            "promotion_cards": [x["promotion_card"] for x in top],
        }
        for rec in top:
            card = rec["score_card"]
            csv_rows.append(
                {
                    "video": video,
                    "obj_id": obj_id,
                    "frame_idx": rec["frame_idx"],
                    "source": rec["source"],
                    "mask_kind": rec["mask_kind"],
                    "area": rec["area"],
                    "bbox": json.dumps(rec["bbox"]),
                    "area_ratio_init": rec["area_ratio_init"],
                    "pos_sim": card["pos_sim"],
                    "neg_sim": card["neg_sim"],
                    "margin": card["margin"],
                    "score": card["score"],
                    "decision": card["decision"],
                    "reasons": ";".join(card["reasons"]),
                }
            )
    results["summary"] = {
        "objects": len(results["objects"]),
        "candidate_count": sum(v["candidate_count"] for v in results["objects"].values()),
        "csv_rows": len(csv_rows),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video", "obj_id", "frame_idx", "source", "mask_kind", "area", "bbox", "area_ratio_init", "pos_sim", "neg_sim", "margin", "score", "decision", "reasons"]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"out_json": str(args.out_json), "out_csv": str(args.out_csv), "summary": results["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

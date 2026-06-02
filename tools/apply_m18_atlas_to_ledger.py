#!/usr/bin/env python3
"""Integrate Gate-B same-class atlas JSON into M18 temporal ledgers.

Qwen/MLLM atlas output is review evidence, not mask truth.  This tool converts
its normalized boxes into pixel-space ledger memories/candidate cards so the
retrieval/fusion stack can use them as explicit target-vs-distractor evidence.
All newly added positive/reanchor items default to ``needs_more_evidence``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.temporal_ledger import (  # noqa: E402
    ledger_filename,
    ledger_key,
    load_ledgers,
    save_ledger,
    write_index,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--atlas-json", type=Path, required=True)
    p.add_argument("--ledger-dir", type=Path, default=Path("artifacts/m18_temporal_ledger"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/m18_temporal_ledger"))
    p.add_argument("--backup-dir", type=Path, default=Path("artifacts/m18_temporal_ledger_before_gateb"))
    p.add_argument("--min-target-conf", type=float, default=0.55)
    p.add_argument("--min-distractor-conf", type=float, default=0.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def image_size(workspace: Path, video: str, frame_idx: int) -> tuple[int, int]:
    root = workspace / "homework" / "JPEGImages" / video
    frames = sorted([*root.glob("*.jpg"), *root.glob("*.jpeg"), *root.glob("*.png")])
    if not frames:
        raise FileNotFoundError(root)
    idx = max(0, min(len(frames) - 1, int(frame_idx)))
    with Image.open(frames[idx]) as img:
        return int(img.width), int(img.height)


def norm_to_pixel(box: Any, width: int, height: int) -> list[int] | None:
    if not (isinstance(box, list) and len(box) == 4):
        return None
    vals = []
    for v in box:
        try:
            vals.append(float(v))
        except Exception:
            return None
    if vals == [0.0, 0.0, 0.0, 0.0]:
        return None
    x1 = round(vals[0] * width / 1000.0)
    y1 = round(vals[1] * height / 1000.0)
    x2 = round(vals[2] * width / 1000.0)
    y2 = round(vals[3] * height / 1000.0)
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def add_unique(items: list[dict[str, Any]], item: dict[str, Any], key_fields: tuple[str, ...]) -> bool:
    key = tuple(item.get(k) for k in key_fields)
    for old in items:
        if tuple(old.get(k) for k in key_fields) == key:
            return False
    items.append(item)
    return True


def add_distractor(
    *,
    ledger: dict[str, Any],
    memory_id: str,
    source: str,
    frame_idx: int,
    bbox: list[int] | None,
    description: str,
    reason: str,
    tags: list[str],
    confidence: float | None = None,
) -> bool:
    if bbox is None:
        return False
    item = {
        "memory_id": memory_id,
        "kind": "distractor",
        "source": source,
        "frames": [int(frame_idx)],
        "bbox": bbox,
        "mask_ref": None,
        "description": description,
        "evidence": [reason] if reason else [],
        "confidence": confidence,
        "tags": tags,
    }
    return add_unique(ledger.setdefault("distractor_memory", []), item, ("memory_id",))


def add_candidate(
    *,
    ledger: dict[str, Any],
    anchor_id: str,
    source: str,
    frame_idx: int,
    bbox: list[int] | None,
    description: str,
    reason: str,
    confidence: float | None,
    risk_tags: list[str],
) -> bool:
    if bbox is None:
        return False
    item = {
        "anchor_id": anchor_id,
        "source": source,
        "frame_idx": int(frame_idx),
        "bbox": bbox,
        "mask_ref": None,
        "description": description,
        "positive_evidence": [reason] if reason else [],
        "negative_evidence": [],
        "temporal_story_compatibility": confidence,
        "descriptor_margin": None,
        "independent_sources": ["m18_sameclass_atlas"],
        "risk_tags": risk_tags,
        "decision": "needs_more_evidence",
        "decision_reason": "MLLM atlas candidate requires descriptor margin, hard-negative check, and delayed confirmation before promotion.",
        "review_status": "needs_more_evidence",
    }
    return add_unique(ledger.setdefault("candidate_anchors", []), item, ("anchor_id",))


def integrate_record(workspace: Path, ledger: dict[str, Any], record: dict[str, Any], args: argparse.Namespace) -> dict[str, int]:
    video = str(record["video"])
    obj_id = int(record["obj_id"])
    agg = record.get("aggregate") or {}
    counts = {"candidate_added": 0, "distractor_added": 0}
    same_class_dense = bool((ledger.get("target_profile") or {}).get("same_class_dense"))
    base_risks = ["mllm_atlas", "same_class_dense" if same_class_dense else "mllm_review"]

    def bbox_for(frame_idx: int, box_norm: Any) -> list[int] | None:
        w, h = image_size(workspace, video, frame_idx)
        return norm_to_pixel(box_norm, w, h)

    for item in ensure_list(agg.get("target_candidate_list")):
        frame_idx = int(item.get("frame_idx", 0))
        conf = float(item.get("confidence") or 0.0)
        if conf < float(args.min_target_conf):
            continue
        box = bbox_for(frame_idx, item.get("bbox_norm_1000"))
        added = add_candidate(
            ledger=ledger,
            anchor_id=f"m18_atlas_target_f{frame_idx}_{item.get('entity_id','E')}",
            source="m18_sameclass_atlas",
            frame_idx=frame_idx,
            bbox=box,
            description=str(item.get("entity_id", "target_candidate")),
            reason=str(item.get("reason", "")),
            confidence=conf,
            risk_tags=base_risks,
        )
        counts["candidate_added"] += int(added)

    for item in ensure_list(agg.get("recommended_reanchor_frames")):
        frame_idx = int(item.get("frame_idx", 0))
        conf = float(item.get("confidence") or 0.0)
        if conf < float(args.min_target_conf):
            continue
        box = bbox_for(frame_idx, item.get("box_norm_1000"))
        added = add_candidate(
            ledger=ledger,
            anchor_id=f"m18_atlas_reanchor_f{frame_idx}_{item.get('entity_id','E')}",
            source="m18_sameclass_atlas",
            frame_idx=frame_idx,
            bbox=box,
            description=str(item.get("entity_id", "recommended_reanchor")),
            reason=str(item.get("rationale", "")),
            confidence=conf,
            risk_tags=base_risks,
        )
        counts["candidate_added"] += int(added)

    for item in ensure_list(agg.get("positive_prompt_plan")):
        frame_idx = int(item.get("frame_idx", 0))
        conf = float(item.get("confidence") or 0.0)
        if conf < float(args.min_target_conf):
            continue
        box = bbox_for(frame_idx, item.get("box_norm_1000"))
        added = add_candidate(
            ledger=ledger,
            anchor_id=f"m18_atlas_prompt_f{frame_idx}",
            source="m18_sameclass_atlas",
            frame_idx=frame_idx,
            bbox=box,
            description=str(item.get("location_description", "positive_prompt_plan")),
            reason=str(item.get("rationale", "")),
            confidence=conf,
            risk_tags=base_risks + ["prompt_plan"],
        )
        counts["candidate_added"] += int(added)

    distractor_sources = [
        ("distractor_candidate_list", "hard_distractor", "bbox_norm_1000"),
        ("old_position_distractors", "old_position_distractor", "bbox_norm_1000"),
        ("same_category_temporally_impossible", "temporally_impossible", "bbox_norm_1000"),
        ("negative_boxes_for_verification", "negative_verification_box", "box_norm_1000"),
        ("candidate_likely_composite_background", "composite_background", "bbox_norm_1000"),
    ]
    for field, tag, box_field in distractor_sources:
        for idx, item in enumerate(ensure_list(agg.get(field))):
            frame_idx = int(item.get("frame_idx", 0))
            box = bbox_for(frame_idx, item.get(box_field))
            desc = str(item.get("description") or item.get("entity_id") or field)
            reason = str(item.get("reason") or item.get("rationale") or "")
            added = add_distractor(
                ledger=ledger,
                memory_id=f"m18_atlas_{tag}_f{frame_idx}_{idx}",
                source="m18_sameclass_atlas",
                frame_idx=frame_idx,
                bbox=box,
                description=desc,
                reason=reason,
                tags=["mllm_atlas", tag],
                confidence=float(item.get("confidence") or args.min_distractor_conf) if item.get("confidence") is not None else None,
            )
            counts["distractor_added"] += int(added)

    for idx, item in enumerate(ensure_list(agg.get("negative_memory_bank"))):
        frame_idx = int(item.get("frame_idx", 0))
        box = bbox_for(frame_idx, item.get("bbox_norm_1000"))
        added = add_distractor(
            ledger=ledger,
            memory_id=f"m18_atlas_negative_bank_f{frame_idx}_{idx}",
            source="m18_sameclass_atlas",
            frame_idx=frame_idx,
            bbox=box,
            description=str(item.get("description", "negative_memory_bank")),
            reason=str(item.get("reason", "")),
            tags=["mllm_atlas", "negative_memory_bank"],
            confidence=float(item.get("confidence")) if item.get("confidence") is not None else None,
        )
        counts["distractor_added"] += int(added)

    ledger.setdefault("notes", []).append(
        f"Gate-B atlas integrated from {Path(args.atlas_json).name}: "
        f"{counts['candidate_added']} candidates, {counts['distractor_added']} distractors. "
        "All MLLM positives remain needs_more_evidence."
    )
    return counts


def main() -> None:
    args = parse_args()
    payload = json.loads(args.atlas_json.read_text(encoding="utf-8"))
    ledgers = load_ledgers(args.ledger_dir)
    if args.out_dir.resolve() == args.ledger_dir.resolve() and args.backup_dir and not args.backup_dir.exists():
        shutil.copytree(args.ledger_dir, args.backup_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"atlas_json": str(args.atlas_json), "objects": {}}
    touched: list[dict[str, Any]] = []
    for record in payload.get("records", []):
        key = ledger_key(str(record["video"]), int(record["obj_id"]))
        if key not in ledgers:
            summary["objects"][key] = {"status": "missing_ledger"}
            continue
        ledger = ledgers[key]
        counts = integrate_record(args.workspace.resolve(), ledger, record, args)
        save_ledger(ledger, args.out_dir / ledger_filename(str(ledger["video"]), int(ledger["obj_id"])))
        touched.append(ledger)
        summary["objects"][key] = {"status": "updated", **counts}
    # Preserve untouched ledgers when writing to a fresh output directory.
    for key, ledger in ledgers.items():
        path = args.out_dir / ledger_filename(str(ledger["video"]), int(ledger["obj_id"]))
        if not path.exists():
            save_ledger(ledger, path)
            touched.append(ledger)
    index = write_index(args.out_dir, touched)
    summary["index"] = str(index)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

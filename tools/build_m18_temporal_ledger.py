#!/usr/bin/env python3
"""Build M18 object-centric temporal ledgers.

Round-1 purpose: persist compact per-object state, candidate/distractor memory,
hidden-score memory, and next review windows without modifying any prediction
outputs.  The generated JSON is the source of truth for later retrieval and
fusion tools.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.temporal_ledger import (  # noqa: E402
    LEDGER_SCHEMA_VERSION,
    ledger_filename,
    ledger_key,
    save_ledger,
    schema_sample,
    validate_ledger,
    write_index,
)

DEFAULT_TARGETS = [
    "8jsm23a7:1",
    "q0sizv6m:2",
    "r13u5z4y:1",
    "1qlssuz2:1",
    "4vznweiu:1",
    "amfdu83t:1",
    "z6dx46qr:1",
]

EVIDENCE_DOCS = {
    "metric_memory": "docs/test_latest_metric_memory.md",
    "m13_z6": "docs/m13_candidate_fusions.md",
    "m14_amfdu": "docs/m14_amfdu83t_kangaroo_reid.md",
    "m15_layered": "docs/m15_layered_video_optim_report.md",
    "m16_remaining": "docs/m16_remaining_reanchor_report.md",
    "m17_q0": "docs/m17_q0_temporal_continuity_report.md",
}


def parse_target(text: str) -> tuple[str, int]:
    video, obj = text.split(":", 1)
    return video, int(obj.replace("obj", ""))


def maybe_first_frame_stats(workspace: Path, video: str, obj_id: int) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on local env
        return {"error": f"PIL/numpy unavailable: {exc}"}
    ann_path = workspace / "homework" / "Annotations" / video / "00000.png"
    frame_root = workspace / "homework" / "JPEGImages" / video
    if not ann_path.is_file() or not frame_root.is_dir():
        return {"error": "workspace annotations or frames missing"}
    ann = np.asarray(Image.open(ann_path))
    if ann.ndim != 2:
        ann = ann[..., 0]
    mask = ann == int(obj_id)
    ys, xs = np.nonzero(mask)
    bbox = None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    frames = sorted(p for p in frame_root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    size = None
    if frames:
        with Image.open(frames[0]) as img:
            size = [int(img.width), int(img.height)]
    return {
        "frame_count": len(frames),
        "image_size": size,
        "object_ids": [int(x) for x in np.unique(ann) if int(x) != 0],
        "area": int(mask.sum()),
        "area_frac": round(float(mask.sum()) / max(1.0, float(mask.size)), 8),
        "bbox": bbox,
        "edge_touch": bool(bbox and (bbox[0] <= 1 or bbox[1] <= 1 or (size and bbox[2] >= size[0] - 1) or (size and bbox[3] >= size[1] - 1))),
    }


def base_ledger(video: str, obj_id: int, current_best_source: str, profile: dict[str, Any], story: list[dict[str, Any]], *, review_status: str = "needs_more_evidence", current_state: str = "AMBIGUOUS") -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "key": ledger_key(video, obj_id),
        "video": video,
        "obj_id": int(obj_id),
        "current_best_source": current_best_source,
        "current_state": current_state,
        "review_status": review_status,
        "target_profile": {
            "first_frame_description": profile.pop("first_frame_description"),
            "tiny": bool(profile.pop("tiny", False)),
            "same_class_dense": bool(profile.pop("same_class_dense", False)),
            "edge_partial": bool(profile.pop("edge_partial", False)),
            "category": profile.pop("category", None),
            "identity_cues": profile.pop("identity_cues", []),
            "first_frame_stats": profile.pop("first_frame_stats", {}),
        },
        "event_story": story,
        "positive_memory": [],
        "distractor_memory": [],
        "candidate_anchors": [],
        "score_history": [],
        "source_table": [],
        "next_review_windows": [],
        "hard_rules": [
            "MLLM coordinates are review evidence, not mask truth.",
            "Same-class dense candidates require delayed confirmation before promotion.",
            "If a candidate is closer to hard-negative memory than positive memory, reject it.",
        ],
        "notes": [],
    }


def add_memory(ledger: dict[str, Any], kind: str, memory_id: str, source: str, description: str, *, frames: list[int] | None = None, bbox: list[int] | None = None, confidence: float | None = None, tags: list[str] | None = None, evidence: list[str] | None = None) -> None:
    item = {
        "memory_id": memory_id,
        "kind": kind,
        "source": source,
        "frames": frames or [],
        "bbox": bbox,
        "mask_ref": None,
        "description": description,
        "evidence": evidence or [],
        "confidence": confidence,
        "tags": tags or [],
    }
    if kind == "positive":
        ledger["positive_memory"].append(item)
    elif kind == "distractor":
        ledger["distractor_memory"].append(item)
    else:
        ledger.setdefault("distractor_memory", []).append(item)


def add_candidate(ledger: dict[str, Any], anchor_id: str, source: str, frame_idx: int, description: str, *, bbox: list[int] | None = None, positive: list[str] | None = None, negative: list[str] | None = None, compatibility: float | None = None, margin: float | None = None, sources: list[str] | None = None, risks: list[str] | None = None, decision: str = "needs_more_evidence", reason: str = "", review_status: str = "unreviewed") -> None:
    ledger["candidate_anchors"].append(
        {
            "anchor_id": anchor_id,
            "source": source,
            "frame_idx": int(frame_idx),
            "bbox": bbox,
            "mask_ref": None,
            "description": description,
            "positive_evidence": positive or [],
            "negative_evidence": negative or [],
            "temporal_story_compatibility": compatibility,
            "descriptor_margin": margin,
            "independent_sources": sources or [],
            "risk_tags": risks or [],
            "decision": decision,
            "decision_reason": reason,
            "review_status": review_status,
        }
    )


def add_score(ledger: dict[str, Any], source: str, *, hidden: float | None = None, row: float | None = None, j: float | None = None, f: float | None = None, delta: float | None = None, note: str = "") -> None:
    ledger["score_history"].append(
        {
            "source": source,
            "hidden_jf_new": hidden,
            "row_jf_new": row,
            "row_j": j,
            "row_f_new": f,
            "delta_vs_previous_row_jf_new": delta,
            "note": note,
        }
    )


def catalog() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    q0 = base_ledger(
        "q0sizv6m",
        2,
        "m17_q0_full_box",
        {
            "first_frame_description": "cropped white/black animal at the left/bottom edge; not the mid-left herd animal",
            "tiny": False,
            "same_class_dense": True,
            "edge_partial": True,
            "category": "black-white animal",
            "identity_cues": ["left-edge partial first mask", "moves toward camera", "bottom/foreground reappearance", "late lower-rump/edge then absent"],
        },
        [
            {"frames": [0, 2], "state": "visible_initial", "summary": "first-frame partial left-edge animal remains the target", "evidence": [EVIDENCE_DOCS["m17_q0"]], "allowed_output": "keep_current"},
            {"frames": [3, 5], "state": "occluded_or_absent", "summary": "do not promote same-class masks while target is missing/occluded", "evidence": [EVIDENCE_DOCS["m17_q0"]], "allowed_output": "empty"},
            {"frames": [6, 12], "state": "reappears_bottom_foreground", "summary": "nose/head/body reappear from lower-left/bottom foreground; full-box probe has hidden support", "evidence": ["artifacts/q0_fix/q0_bottom_box_split.json", EVIDENCE_DOCS["m17_q0"]], "allowed_output": "candidate"},
            {"frames": [13, 29], "state": "foreground_continuation_high_risk", "summary": "full-box continuation explains hidden gain but may contain composite foreground masks", "evidence": ["artifacts/q0_fix/q0_story_trim_after34.json"], "allowed_output": "candidate"},
            {"frames": [30, 33], "state": "partial_rump_or_absent", "summary": "only possible rump/edge; trim/rump policies need hidden/visual review", "evidence": ["artifacts/q0_fix/q0_story_rump_31_33.json"], "allowed_output": "review_only"},
            {"frames": [34, 41], "state": "absent_after_disappearance", "summary": "late false positives should not be promoted; empty policy is under test", "evidence": [EVIDENCE_DOCS["m17_q0"]], "allowed_output": "empty"},
        ],
        review_status="approved",
        current_state="RECOVERY_BRANCH",
    )
    add_memory(q0, "positive", "q0_first_gt", "first_frame_gt", "partial left-edge target animal", frames=[0], tags=["first_gt", "edge_partial"])
    add_memory(q0, "positive", "q0_fullbox_hidden_good", "m17_q0_full_box", "hidden-confirmed bottom/foreground recovery path", frames=[6, 29], confidence=0.85, tags=["hidden_win"])
    add_memory(q0, "distractor", "q0_old_position_animal", "m15_safe_baseline", "mid-left same-class animal near old scene position after frame 13", frames=[13, 41], tags=["old_position_distractor", "same_class"])
    add_memory(q0, "distractor", "q0_late_fullbox_false_positive", "m17_q0_full_box", "late masks around 36 and 39-41 likely false after disappearance", frames=[34, 41], tags=["absent_window", "rejected_policy"])
    add_candidate(q0, "q0_fullbox_f6", "q0_bottom_box", 6, "lower-left/bottom foreground SAM2 box-prompt branch", positive=["hidden row improves 12.34 -> 68.29"], negative=["same-class/composite foreground risk late"], compatibility=0.85, sources=["manual_story", "SAM2_box", "hidden_feedback"], risks=["same_class_dense"], decision="promote", reason="Two independent signals: user/visual story plus hidden row lift; keep bounded by story schedule.", review_status="approved")
    add_candidate(q0, "q0_trim_after34", "m17_story_trim_after34", 30, "force empty after visible foreground interval", positive=["matches disappearance story"], negative=["not hidden-confirmed yet"], compatibility=0.65, sources=["visual_story"], risks=["score_probe"], decision="output_only", reason="Candidate is a schedule probe, not a memory anchor.", review_status="needs_more_evidence")
    add_candidate(q0, "q0_rump_31_33", "q0_color_balanced", 31, "color/ROI rump candidate before disappearance", positive=["matches partial-rump hypothesis"], negative=["fragmented color mask; no hidden feedback"], compatibility=0.45, risks=["composite_background", "single_signal"], decision="needs_more_evidence", reason="Same-class dense partial evidence cannot promote.", review_status="needs_more_evidence")
    add_score(q0, "m15_safe", hidden=43.44, row=12.34, j=12.50, f=12.50, note="pre-q0 baseline row")
    add_score(q0, "m17_q0_early_box", hidden=43.47, note="user-reported hidden score")
    add_score(q0, "m17_q0_full_box", hidden=43.54, row=68.29, j=66.96, f=67.43, delta=55.95, note="hidden-confirmed q0 obj2 recovery")
    q0["source_table"] = [
        {"frames": [6, 29], "source": "q0_bottom_box", "status": "approved for balanced/safe if no newer regression"},
        {"frames": [30, 33], "source": "trim/rump policy", "status": "needs hidden review"},
        {"frames": [34, 41], "source": "empty", "status": "story probe; do not promote memory"},
    ]
    q0["next_review_windows"] = [{"frames": [27, 34], "question": "choose trim/rump/empty schedule using hidden feedback and zoom sheet"}]
    q0["notes"].append("Gate-A sample object: expresses target, hard distractor, event story, and forbidden late replacements without chat context.")
    out[q0["key"]] = q0

    eight = base_ledger(
        "8jsm23a7",
        1,
        "m15_safe_unchanged_for_8js",
        {
            "first_frame_description": "single annotated Mahjong tile instance in a crowded tile scene; later picked/moved/placed among same-class tiles",
            "tiny": True,
            "same_class_dense": True,
            "edge_partial": False,
            "category": "Mahjong tile",
            "identity_cues": ["tile face markings", "picked-up physical tile", "front-row placement event", "avoid two-bamboo/front-back-row distractors"],
        },
        [
            {"frames": [0, 2], "state": "visible_initial", "summary": "first annotated tile establishes exact instance; do not infer from category alone", "evidence": [EVIDENCE_DOCS["m15_layered"]], "allowed_output": "keep_current"},
            {"frames": [3, 19], "state": "picked_moved_occluded_or_handheld", "summary": "object may be carried/occluded; old table position becomes a hard negative", "evidence": [EVIDENCE_DOCS["m16_remaining"]], "allowed_output": "review_only"},
            {"frames": [20, 48], "state": "placed_among_front_row_same_class", "summary": "candidate must distinguish seven-bamboo target from two-bamboo/front/back-row tiles", "evidence": [EVIDENCE_DOCS["m15_layered"], EVIDENCE_DOCS["m16_remaining"]], "allowed_output": "candidate"},
        ],
        review_status="needs_more_evidence",
        current_state="AMBIGUOUS",
    )
    add_memory(eight, "positive", "8js_first_gt", "first_frame_gt", "exact annotated tile face at frame0", frames=[0], tags=["first_gt", "tiny"])
    add_memory(eight, "distractor", "8js_m13_wrong_two_bamboo", "m13_reverse_anchor", "left-side two-bamboo tile in farther/upper row; visually plausible false positive", frames=[20, 48], tags=["hard_negative", "same_class", "old_false_positive"])
    add_memory(eight, "distractor", "8js_old_position", "event_story", "original table location after tile is picked up; temporally impossible target", frames=[3, 48], tags=["old_position_distractor", "temporally_impossible"])
    add_candidate(eight, "8js_m15_frame20", "m15_8js_frame20_fromanchor", 20, "front-row tile reanchor probe", positive=["mechanism aligned with placement event"], negative=["M15 balanced hidden-tied safe; exact identity ambiguous"], compatibility=0.55, sources=["Qwen_event", "SAM2"], risks=["same_class_dense", "tile_face_ambiguous"], decision="output_only", reason="Not safe: no hidden row lift and single visual/MLLM signal is insufficient.", review_status="needs_more_evidence")
    add_candidate(eight, "8js_m16_maskbox", "m16mask", 26, "near/front-row mask-box tile hypothesis", positive=["non-empty tile after placement"], negative=["may still target wrong face or part tile"], compatibility=0.55, sources=["manual_box", "SAM2_maskbox"], risks=["same_class_dense", "single_frame_prompt"], decision="needs_more_evidence", reason="Requires same-class atlas with at least two hard distractors before promotion.", review_status="needs_more_evidence")
    add_score(eight, "m15_safe", hidden=43.44, row=2.13, j=2.13, f=2.13, note="still unsolved")
    add_score(eight, "m15_balanced", hidden=43.44, row=2.13, note="extra 8js edit produced no measurable row gain")
    eight["next_review_windows"] = [{"frames": [0, 2, 20, 26, 48], "question": "build M18 Mahjong entity atlas with target hypothesis plus >=2 hard distractors"}]
    out[eight["key"]] = eight

    r13 = base_ledger(
        "r13u5z4y",
        1,
        "m15_safe_empty_after_occlusion",
        {
            "first_frame_description": "strawberry slice among many similar red/white slices; identity becomes ambiguous after occlusion/contact",
            "tiny": False,
            "same_class_dense": True,
            "edge_partial": False,
            "category": "strawberry slice",
            "identity_cues": ["specific cut shape", "local neighbor order", "avoid more salient reappearing slice unless identity proven"],
        },
        [
            {"frames": [0, 6], "state": "visible_initial", "summary": "target slice visible before multi-object contact", "evidence": [EVIDENCE_DOCS["m15_layered"]], "allowed_output": "keep_current"},
            {"frames": [7, 19], "state": "occluded_contact", "summary": "dense strawberry contact; empty is safer than wrong same-class slice", "evidence": [EVIDENCE_DOCS["m16_remaining"]], "allowed_output": "empty"},
            {"frames": [20, 44], "state": "possible_reappearance_same_class_dense", "summary": "accept only candidates distinguishable from hard-negative slices", "evidence": [EVIDENCE_DOCS["m16_remaining"]], "allowed_output": "candidate"},
        ],
        review_status="needs_more_evidence",
        current_state="OCCLUDED",
    )
    add_memory(r13, "positive", "r13_first_gt", "first_frame_gt", "initial strawberry slice", frames=[0], tags=["first_gt"])
    add_memory(r13, "distractor", "r13_more_salient_slice", "m16mask_or_visual", "more visible post-occlusion slice that may be a hard negative", frames=[20, 44], tags=["hard_negative", "same_class"])
    add_candidate(r13, "r13_m16_maskbox", "m16mask", 20, "left-slice reappearance mask-box hypothesis", positive=["non-empty after occlusion"], negative=["same-class slices cannot be distinguished yet"], compatibility=0.35, sources=["manual_box", "SAM2_maskbox"], risks=["same_class_dense", "hard_negative_similarity"], decision="needs_more_evidence", reason="If candidate is more like hard negative, keep empty.", review_status="needs_more_evidence")
    add_score(r13, "m15_safe", hidden=43.44, row=38.84, j=39.53, f=39.53, note="reappearance remains unsolved/empty")
    r13["next_review_windows"] = [{"frames": [0, 7, 19, 20, 44], "question": "strawberry atlas; only accept anchor if target slice can be distinguished from hard negatives"}]
    out[r13["key"]] = r13

    one = base_ledger(
        "1qlssuz2",
        1,
        "m15_safe_current_track",
        {
            "first_frame_description": "tiny white vehicle/road object; current track is coherent but low hidden row",
            "tiny": True,
            "same_class_dense": False,
            "edge_partial": False,
            "category": "tiny vehicle/road object",
            "identity_cues": ["small white vehicle", "road/camera-motion context", "avoid broad expansion"],
        },
        [
            {"frames": [0, 12], "state": "visible_or_coherent_current", "summary": "current SAM2 path is mostly coherent", "evidence": [EVIDENCE_DOCS["m16_remaining"]], "allowed_output": "keep_current"},
            {"frames": [13, 14], "state": "bridge_or_occlusion_uncertain", "summary": "possible gap window; needs targeted evidence", "evidence": [EVIDENCE_DOCS["m16_remaining"]], "allowed_output": "review_only"},
            {"frames": [15, 39], "state": "tiny_continuation_uncertain", "summary": "avoid broad aggressive replacement without detector support", "evidence": [EVIDENCE_DOCS["m15_layered"]], "allowed_output": "keep_current"},
        ],
        review_status="needs_more_evidence",
        current_state="AMBIGUOUS",
    )
    add_memory(one, "positive", "1ql_first_gt", "first_frame_gt", "tiny white vehicle reference", frames=[0], tags=["first_gt", "tiny"])
    add_memory(one, "distractor", "1ql_broad_window_rejected", "m15_1ql_window", "over-expanded aggressive window; likely background/adjacent road included", frames=[13, 39], tags=["rejected_aggressive", "composite_background"])
    add_candidate(one, "1ql_m15_window", "m15_1ql_window", 13, "aggressive tiny-car window", positive=["may fill visibility"], negative=["M15 aggressive lowered score; masks too large"], compatibility=0.15, sources=["Qwen_event", "SAM2"], risks=["overexpanded", "aggressive_regression"], decision="reject", reason="Aggressive package regressed and visual review found overlarge masks.", review_status="rejected")
    add_score(one, "m15_safe", hidden=43.44, row=36.88, j=40.36, f=42.11, note="unchanged safe row")
    add_score(one, "m15_aggressive", hidden=43.38, note="aggressive package with 1ql/4v additions regressed globally")
    one["next_review_windows"] = [{"frames": [13, 14, 37, 39], "question": "detector-level small vehicle proof; no broad SAM2 window replacement"}]
    out[one["key"]] = one

    four = base_ledger(
        "4vznweiu",
        1,
        "m15_safe_current_track",
        {
            "first_frame_description": "tiny semantic bead/letter-like object among adjacent beads/flowers",
            "tiny": True,
            "same_class_dense": True,
            "edge_partial": False,
            "category": "tiny bead/letter object",
            "identity_cues": ["T-like/letter bead relation", "adjacent bead cluster", "avoid flowers or neighboring letters"],
        },
        [
            {"frames": [0, 34], "state": "tiny_semantic_current_best", "summary": "current path is weak but safer than broad aggressive alternatives", "evidence": [EVIDENCE_DOCS["m15_layered"], EVIDENCE_DOCS["m16_remaining"]], "allowed_output": "keep_current"},
        ],
        review_status="needs_more_evidence",
        current_state="AMBIGUOUS",
    )
    add_memory(four, "positive", "4v_first_gt", "first_frame_gt", "tiny target bead/letter reference", frames=[0], tags=["first_gt", "tiny"])
    add_memory(four, "distractor", "4v_adjacent_letters_flowers", "m15_4v_window", "adjacent bead/flower candidates selected by aggressive probe", frames=[1, 34], tags=["hard_negative", "semantic_distractor"])
    add_candidate(four, "4v_m15_window", "m15_4v_window", 1, "aggressive bead/letter window", positive=["non-empty alternative"], negative=["shifts between bead cluster and flowers; aggressive regressed"], compatibility=0.10, sources=["Qwen_event", "SAM2"], risks=["same_class_dense", "semantic_swap", "aggressive_regression"], decision="reject", reason="Not final-allowable without high-resolution letter atlas.", review_status="rejected")
    add_score(four, "m15_safe", hidden=43.44, row=33.50, j=36.38, f=37.64, note="unchanged safe row")
    add_score(four, "m15_aggressive", hidden=43.38, note="aggressive package with 1ql/4v additions regressed globally")
    four["next_review_windows"] = [{"frames": [0, 1, 10, 20, 34], "question": "build high-resolution bead/letter atlas before any replacement"}]
    out[four["key"]] = four

    amf = base_ledger(
        "amfdu83t",
        1,
        "m15_safe_m14_from8_noclip",
        {
            "first_frame_description": "far-left edge kangaroo, only head/neck/front-foot visible before leaving view",
            "tiny": True,
            "same_class_dense": True,
            "edge_partial": True,
            "category": "kangaroo",
            "identity_cues": ["far-left edge start", "bottom-left near-camera reappearance", "overtaking path", "single kangaroo continuation after frame 11"],
        },
        [
            {"frames": [0, 2], "state": "visible_then_exits", "summary": "target starts at far-left edge then likely leaves view", "evidence": [EVIDENCE_DOCS["m14_amfdu"]], "allowed_output": "keep_current"},
            {"frames": [3, 7], "state": "absent_or_occluded", "summary": "do not write strong distractor as target", "evidence": [EVIDENCE_DOCS["m14_amfdu"]], "allowed_output": "empty"},
            {"frames": [8, 23], "state": "reappears_bottom_left_nearcamera", "summary": "M14 from8 no-clip continuation is hidden-confirmed and safe in M15", "evidence": [EVIDENCE_DOCS["m15_layered"]], "allowed_output": "candidate"},
        ],
        review_status="approved",
        current_state="RECONFIRMED",
    )
    add_memory(amf, "positive", "amf_first_gt", "first_frame_gt", "far-left edge kangaroo", frames=[0], tags=["first_gt", "edge_partial"])
    add_memory(amf, "positive", "amf_from8_hidden_good", "m14_from8_noclip", "hidden-confirmed near-camera reappearance/continuation", frames=[8, 23], confidence=0.90, tags=["hidden_win", "same_class_recovery"])
    add_memory(amf, "distractor", "amf_left_side_strong_distractor", "visual_review", "strong left-side kangaroo after disappearance not necessarily original target", frames=[3, 11], tags=["same_class", "hard_negative"])
    add_candidate(amf, "amf_from8_noclip", "m14_from8_noclip", 8, "bottom-left near-camera kangaroo reanchor and continuation", positive=["row lift 28.44 -> 84.84", "single plausible track frames 8-23"], negative=["identity risk after frame 11, but hidden confirms net win"], compatibility=0.90, sources=["visual_story", "SAM2_box", "hidden_feedback"], risks=["same_class_dense_but_confirmed"], decision="promote", reason="Hidden-confirmed same-class recovery; use as positive memory template.", review_status="approved")
    add_score(amf, "m13_zofficial_family", hidden=43.34, row=28.44, note="before amfdu recovery")
    add_score(amf, "m15_safe", hidden=43.44, row=84.84, j=91.39, f=97.94, delta=56.40, note="decisive M15 gain")
    amf["source_table"] = [{"frames": [8, 23], "source": "m14_from8_noclip", "status": "approved safe/balanced"}]
    amf["next_review_windows"] = [{"frames": [12, 23], "question": "monitor continuation; if regression appears, trim but keep positive memory"}]
    out[amf["key"]] = amf

    z6 = base_ledger(
        "z6dx46qr",
        1,
        "m13_zofficial_balanced_official_large_interval",
        {
            "first_frame_description": "low-contrast underwater/sand tiny target with two-dot/eye-like cues",
            "tiny": True,
            "same_class_dense": False,
            "edge_partial": False,
            "category": "low-contrast aquatic/underwater object",
            "identity_cues": ["two dark dot relation", "low-contrast body", "official-large interval continuity"],
        },
        [
            {"frames": [0, 7], "state": "visible_initial_low_contrast", "summary": "baseline/first GT gives positive memory", "evidence": [EVIDENCE_DOCS["m13_z6"]], "allowed_output": "keep_current"},
            {"frames": [8, 24], "state": "official_large_interval_supported", "summary": "official-large interval is hidden-confirmed; direct MLLM/SAM boxes oversegment", "evidence": [EVIDENCE_DOCS["metric_memory"], EVIDENCE_DOCS["m13_z6"]], "allowed_output": "candidate"},
            {"frames": [25, 37], "state": "low_contrast_uncertain", "summary": "do not extend without detector/retrieval support", "evidence": [EVIDENCE_DOCS["m13_z6"]], "allowed_output": "review_only"},
        ],
        review_status="approved",
        current_state="RECONFIRMED",
    )
    add_memory(z6, "positive", "z6_first_gt", "first_frame_gt", "low-contrast target reference", frames=[0], tags=["first_gt", "tiny"])
    add_memory(z6, "positive", "z6_official_interval", "official_large", "hidden-confirmed official-large interval frames 8-24", frames=[8, 24], confidence=0.85, tags=["hidden_win"])
    add_memory(z6, "distractor", "z6_direct_box_blob", "m13_direct_sam2_box", "direct box prompting produced large low-contrast blobs", frames=[8, 24], tags=["rejected_route", "oversegmentation"])
    add_candidate(z6, "z6_official_l_8_24", "official_large", 8, "official-large interval replacement", positive=["row lift 40.47 -> 65.14"], negative=["not proof for later extension"], compatibility=0.80, sources=["official_large", "hidden_feedback"], risks=["low_contrast"], decision="promote", reason="Hidden-confirmed interval; direct SAM2 box alternative rejected.", review_status="approved")
    add_score(z6, "m7_family", hidden=43.30, row=40.47, note="pre-zofficial row")
    add_score(z6, "m13_zofficial_balanced", hidden=43.34, row=65.14, j=61.77, f=79.12, delta=24.67, note="official-large interval gain")
    z6["source_table"] = [{"frames": [8, 24], "source": "official_large", "status": "approved safe/balanced"}]
    z6["next_review_windows"] = [{"frames": [25, 37], "question": "look for source-independent low-contrast continuation; do not direct-box prompt"}]
    out[z6["key"]] = z6

    return out


def inject_first_frame_stats(ledgers: dict[str, dict[str, Any]], workspace: Path) -> None:
    for ledger in ledgers.values():
        stats = maybe_first_frame_stats(workspace, str(ledger["video"]), int(ledger["obj_id"]))
        ledger["target_profile"]["first_frame_stats"] = stats
        if stats.get("area_frac", 1.0) < 0.002:
            ledger["target_profile"]["tiny"] = True
        if stats.get("edge_touch"):
            ledger["target_profile"]["edge_partial"] = True


def write_doc(path: Path, ledgers: list[dict[str, Any]], out_dir: Path) -> None:
    preserved_tail = ""
    if path.is_file():
        old = path.read_text(encoding="utf-8")
        marker = "\n## Review/optimization pass"
        if marker in old:
            preserved_tail = old[old.index(marker):].rstrip()
    lines = [
        "# M18 system re-anchor report",
        "",
        "This report is the Round-1 durable memory layer for the object-centric temporal re-anchor system. It does not modify prediction outputs; it creates reviewable JSON ledgers that later candidate retrieval and fusion tools can consume.",
        "",
        "## Gate A — ledger/schema review",
        "",
        f"- schema version: `{LEDGER_SCHEMA_VERSION}`",
        f"- ledger dir: `{out_dir}`",
        "- sample schema JSON: `artifacts/m18_temporal_ledger/schema_sample.json`",
        "- two required sample ledgers: `q0sizv6m_obj2.json`, `8jsm23a7_obj1.json`",
        "",
        "Pass condition coverage:",
        "",
        "- target profile and first-frame stats are explicit, including tiny/edge/same-class flags.",
        "- event story records visible/occluded/recovery/absent windows with allowed output modes.",
        "- positive and distractor memory are separated; rejected policies remain available as negative memory.",
        "- candidate anchors store positive evidence, negative evidence, risk tags, review status, and promotion decision.",
        "",
        "## Round-1 object ledger index",
        "",
        "| object | current best | review | state | score memory | next review window |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for led in sorted(ledgers, key=lambda x: x["key"]):
        scores = []
        for s in led.get("score_history", [])[-2:]:
            bit = str(s.get("source"))
            if s.get("row_jf_new") is not None:
                bit += f": row {float(s['row_jf_new']):.2f}"
            if s.get("hidden_jf_new") is not None:
                bit += f" / hidden {float(s['hidden_jf_new']):.2f}"
            scores.append(bit)
        nxt = "; ".join(
            f"{w.get('frames')}: {w.get('question')}" for w in led.get("next_review_windows", [])[:2]
        )
        lines.append(
            f"| `{led['key']}` | `{led['current_best_source']}` | {led['review_status']} | {led['current_state']} | {'<br>'.join(scores)} | {nxt} |"
        )
    lines += [
        "",
        "## Gate B/C/D hooks now represented in JSON",
        "",
        "- Gate B atlas results should append `distractor_memory` plus `candidate_anchors[*].negative_evidence` before any propagation.",
        "- Gate C promotion cards map directly to `candidate_anchors[*]`: source, frame, positive/negative evidence, story compatibility, descriptor margin, risk tags, and decision.",
        "- Gate D fusion must write a source table and validation JSON; `tools/apply_m18_reanchor_fusion.py` consumes these ledgers plus explicit policies.",
        "",
        "## Current conservative interpretation",
        "",
        "- Approved positive memory: `amfdu83t:1` M14/M15 from8 no-clip; `z6dx46qr:1` official-large interval; `q0sizv6m:2` M17 full-box bottom path with late schedule still under review.",
        "- Needs atlas before safe/balanced: `8jsm23a7:1`, `r13u5z4y:1`, `1qlssuz2:1`, `4vznweiu:1`.",
        "- Rejected/negative patterns: old-position Mahjong distractor, q0 mid-left herd animal, r13 more-salient strawberry slice, 1ql broad expansion, 4v bead/flower semantic swap, z6 direct-box blob.",
        "",
        "## No-output-change invariant",
        "",
        "Round 1 generated only docs and JSON ledgers. No prediction root or submission zip was modified.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip()
    if preserved_tail:
        text += "\n" + preserved_tail
    path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/m18_temporal_ledger"))
    p.add_argument("--out-doc", type=Path, default=Path("docs/m18_system_reanchor_report.md"))
    p.add_argument("--no-doc", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wanted = {ledger_key(*parse_target(t)) for t in args.targets}
    ledgers = catalog()
    selected = {k: v for k, v in ledgers.items() if k in wanted}
    missing = sorted(wanted - set(selected))
    if missing:
        raise SystemExit(f"unknown targets: {missing}")
    inject_first_frame_stats(selected, args.workspace.resolve())
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_path = out_dir / "schema_sample.json"
    sample = schema_sample()
    errors = validate_ledger(sample)
    if errors:
        raise SystemExit("schema sample invalid: " + "; ".join(errors))
    schema_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    written: list[dict[str, Any]] = []
    for led in selected.values():
        path = out_dir / ledger_filename(str(led["video"]), int(led["obj_id"]))
        written.append(save_ledger(led, path))
    index_path = write_index(out_dir, written)
    if not args.no_doc:
        write_doc(args.out_doc, written, out_dir)
    print(
        json.dumps(
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "out_dir": str(out_dir),
                "count": len(written),
                "index": str(index_path),
                "schema_sample": str(schema_path),
                "doc": None if args.no_doc else str(args.out_doc),
                "ledgers": [ledger_key(str(x["video"]), int(x["obj_id"])) for x in written],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

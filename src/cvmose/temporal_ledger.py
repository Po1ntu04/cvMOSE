"""Object-centric temporal ledgers for M18 re-anchor experiments.

The ledger is deliberately lightweight JSON.  It stores durable, reviewable
identity state per video/object so later retrieval/fusion scripts do not depend
on chat history or long prompts.  This module owns schema defaults, structural
validation, and round-trip helpers; experiment tools own visual evidence and
model I/O.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

LEDGER_SCHEMA_VERSION = "m18.temporal-ledger.v1"

ReviewStatus = Literal["unreviewed", "approved", "rejected", "needs_more_evidence"]
PromotionDecision = Literal["reject", "output_only", "promote", "needs_more_evidence"]
TemporalState = Literal[
    "STABLE",
    "AMBIGUOUS",
    "OCCLUDED",
    "RECOVERY_BRANCH",
    "RECONFIRMED",
    "REJECTED",
]

REVIEW_STATUSES = {"unreviewed", "approved", "rejected", "needs_more_evidence"}
PROMOTION_DECISIONS = {"reject", "output_only", "promote", "needs_more_evidence"}
TEMPORAL_STATES = {
    "STABLE",
    "AMBIGUOUS",
    "OCCLUDED",
    "RECOVERY_BRANCH",
    "RECONFIRMED",
    "REJECTED",
}


@dataclass(slots=True)
class FrameRange:
    start: int
    end: int

    def to_json(self) -> list[int]:
        return [int(self.start), int(self.end)]


@dataclass(slots=True)
class TargetProfile:
    first_frame_description: str
    tiny: bool = False
    same_class_dense: bool = False
    edge_partial: bool = False
    category: str | None = None
    identity_cues: list[str] = field(default_factory=list)
    first_frame_stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventStoryEntry:
    frames: list[int]
    state: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    allowed_output: Literal["keep_current", "candidate", "empty", "review_only"] = "review_only"


@dataclass(slots=True)
class MemoryItem:
    memory_id: str
    kind: Literal["positive", "distractor", "rejected"]
    source: str
    frames: list[int] = field(default_factory=list)
    bbox: list[int] | None = None
    mask_ref: str | None = None
    description: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence: float | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CandidateAnchor:
    anchor_id: str
    source: str
    frame_idx: int
    bbox: list[int] | None = None
    mask_ref: str | None = None
    description: str = ""
    positive_evidence: list[str] = field(default_factory=list)
    negative_evidence: list[str] = field(default_factory=list)
    temporal_story_compatibility: float | None = None
    descriptor_margin: float | None = None
    independent_sources: list[str] = field(default_factory=list)
    risk_tags: list[str] = field(default_factory=list)
    decision: PromotionDecision = "needs_more_evidence"
    decision_reason: str = ""
    review_status: ReviewStatus = "unreviewed"


@dataclass(slots=True)
class ScoreRecord:
    source: str
    hidden_jf_new: float | None = None
    row_jf_new: float | None = None
    row_j: float | None = None
    row_f_new: float | None = None
    delta_vs_previous_row_jf_new: float | None = None
    note: str = ""


@dataclass(slots=True)
class ObjectLedger:
    video: str
    obj_id: int
    current_best_source: str
    target_profile: TargetProfile
    event_story: list[EventStoryEntry]
    positive_memory: list[MemoryItem] = field(default_factory=list)
    distractor_memory: list[MemoryItem] = field(default_factory=list)
    candidate_anchors: list[CandidateAnchor] = field(default_factory=list)
    review_status: ReviewStatus = "unreviewed"
    schema_version: str = LEDGER_SCHEMA_VERSION
    current_state: TemporalState = "AMBIGUOUS"
    score_history: list[ScoreRecord] = field(default_factory=list)
    source_table: list[dict[str, Any]] = field(default_factory=list)
    next_review_windows: list[dict[str, Any]] = field(default_factory=list)
    hard_rules: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return ledger_key(self.video, self.obj_id)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["key"] = self.key
        return data


def ledger_key(video: str, obj_id: int | str) -> str:
    return f"{video}:obj{int(obj_id)}"


def ledger_filename(video: str, obj_id: int | str) -> str:
    return f"{video}_obj{int(obj_id)}.json"


def _expect_type(value: Any, typ: type, path: str, errors: list[str]) -> None:
    if not isinstance(value, typ):
        errors.append(f"{path}: expected {typ.__name__}, got {type(value).__name__}")


def validate_ledger(data: dict[str, Any]) -> list[str]:
    """Return structural errors for a ledger JSON object.

    This is intentionally dependency-free and validates only the contract used by
    scripts.  It is not a replacement for human review of visual evidence.
    """
    errors: list[str] = []
    if data.get("schema_version") != LEDGER_SCHEMA_VERSION:
        errors.append(
            f"schema_version: expected {LEDGER_SCHEMA_VERSION!r}, got {data.get('schema_version')!r}"
        )
    for key, typ in [("video", str), ("obj_id", int), ("current_best_source", str)]:
        _expect_type(data.get(key), typ, key, errors)
    status = data.get("review_status")
    if status not in REVIEW_STATUSES:
        errors.append(f"review_status: invalid {status!r}")
    state = data.get("current_state")
    if state not in TEMPORAL_STATES:
        errors.append(f"current_state: invalid {state!r}")
    tp = data.get("target_profile")
    if not isinstance(tp, dict):
        errors.append("target_profile: expected object")
    else:
        _expect_type(tp.get("first_frame_description"), str, "target_profile.first_frame_description", errors)
        for key in ["tiny", "same_class_dense", "edge_partial"]:
            _expect_type(tp.get(key), bool, f"target_profile.{key}", errors)
    story = data.get("event_story")
    if not isinstance(story, list) or not story:
        errors.append("event_story: expected non-empty list")
    elif isinstance(story, list):
        for i, item in enumerate(story):
            if not isinstance(item, dict):
                errors.append(f"event_story[{i}]: expected object")
                continue
            frames = item.get("frames")
            if not (
                isinstance(frames, list)
                and len(frames) == 2
                and all(isinstance(x, int) for x in frames)
                and frames[0] <= frames[1]
            ):
                errors.append(f"event_story[{i}].frames: expected [start,end] ints")
            _expect_type(item.get("state"), str, f"event_story[{i}].state", errors)
            _expect_type(item.get("summary"), str, f"event_story[{i}].summary", errors)
    for list_key in ["positive_memory", "distractor_memory", "candidate_anchors", "score_history"]:
        if not isinstance(data.get(list_key), list):
            errors.append(f"{list_key}: expected list")
    for i, item in enumerate(data.get("candidate_anchors") or []):
        if not isinstance(item, dict):
            errors.append(f"candidate_anchors[{i}]: expected object")
            continue
        decision = item.get("decision")
        if decision not in PROMOTION_DECISIONS:
            errors.append(f"candidate_anchors[{i}].decision: invalid {decision!r}")
        review_status = item.get("review_status")
        if review_status not in REVIEW_STATUSES:
            errors.append(f"candidate_anchors[{i}].review_status: invalid {review_status!r}")
    return errors


def assert_valid_ledger(data: dict[str, Any]) -> None:
    errors = validate_ledger(data)
    if errors:
        raise ValueError("invalid temporal ledger:\n- " + "\n- ".join(errors))


def save_ledger(ledger: ObjectLedger | dict[str, Any], path: Path) -> dict[str, Any]:
    data = ledger.to_json() if isinstance(ledger, ObjectLedger) else deepcopy(ledger)
    # The key is derived and useful in JSON, but validation should not require it.
    data.setdefault("key", ledger_key(str(data.get("video")), int(data.get("obj_id"))))
    assert_valid_ledger(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def load_ledger(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"ledger root must be object: {path}")
    assert_valid_ledger(data)
    return data


def load_ledgers(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        if path.name in {"index.json", "schema_sample.json"}:
            continue
        data = load_ledger(path)
        out[ledger_key(str(data["video"]), int(data["obj_id"]))] = data
    return out


def write_index(root: Path, ledgers: list[dict[str, Any]]) -> Path:
    payload = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "count": len(ledgers),
        "ledgers": [
            {
                "key": ledger_key(str(item["video"]), int(item["obj_id"])),
                "video": item["video"],
                "obj_id": item["obj_id"],
                "review_status": item.get("review_status"),
                "current_best_source": item.get("current_best_source"),
                "current_state": item.get("current_state"),
                "path": ledger_filename(str(item["video"]), int(item["obj_id"])),
                "next_review_windows": item.get("next_review_windows", []),
            }
            for item in ledgers
        ],
    }
    path = root / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def schema_sample() -> dict[str, Any]:
    """A compact example for Gate-A review and docs."""
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "key": "q0sizv6m:obj2",
        "video": "q0sizv6m",
        "obj_id": 2,
        "current_best_source": "m17_q0_full_box",
        "current_state": "RECOVERY_BRANCH",
        "review_status": "needs_more_evidence",
        "target_profile": {
            "first_frame_description": "cropped white/black animal at the left/bottom edge",
            "tiny": False,
            "same_class_dense": True,
            "edge_partial": True,
            "category": "animal",
            "identity_cues": ["left-edge partial body", "moves toward lower foreground"],
            "first_frame_stats": {"area": 7787, "bbox": [0, 552, 78, 701]},
        },
        "event_story": [
            {
                "frames": [0, 2],
                "state": "visible_initial",
                "summary": "target starts as the left-edge cropped animal",
                "evidence": ["first_frame_gt"],
                "allowed_output": "keep_current",
            },
            {
                "frames": [3, 5],
                "state": "occluded_or_absent",
                "summary": "no durable mask should be promoted",
                "evidence": ["m17_visual_review"],
                "allowed_output": "empty",
            },
            {
                "frames": [6, 29],
                "state": "reappears_bottom_foreground",
                "summary": "SAM2 full-box candidate has hidden-score support but remains composite-risk late",
                "evidence": ["m17_q0_full_box_hidden_row_68.29"],
                "allowed_output": "candidate",
            },
        ],
        "positive_memory": [],
        "distractor_memory": [],
        "candidate_anchors": [],
        "score_history": [],
        "source_table": [],
        "next_review_windows": [],
        "hard_rules": ["same-class dense requires delayed confirmation"],
        "notes": [],
    }


__all__ = [
    "LEDGER_SCHEMA_VERSION",
    "ObjectLedger",
    "TargetProfile",
    "EventStoryEntry",
    "MemoryItem",
    "CandidateAnchor",
    "ScoreRecord",
    "ledger_key",
    "ledger_filename",
    "validate_ledger",
    "assert_valid_ledger",
    "save_ledger",
    "load_ledger",
    "load_ledgers",
    "write_index",
    "schema_sample",
]

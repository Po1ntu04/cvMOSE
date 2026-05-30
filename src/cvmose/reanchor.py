"""Small training-free primitives for reappearance-aware re-anchoring.

The classes in this module intentionally do not depend on SAM2/SAM3.  They are
plain state containers and rule helpers used by experiment entrypoints.  Keeping
these primitives small makes the RAR path auditable: propagation code owns model
I/O, while this module owns state names, reservoir selection, and commit labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

STATE_STABLE = "stable"
STATE_AMBIGUOUS = "ambiguous"
STATE_RECOVERY = "recovery"
STATE_ABSENT = "absent"

COMMIT_WRITE_MAIN = "write_main_memory"
COMMIT_PROVISIONAL = "provisional_only"
COMMIT_PROMOTE_RCMS = "promote_rcms_anchors"
COMMIT_OUTPUT_EMPTY = "output_empty"
COMMIT_CONDITIONED_EXISTING = "conditioned_memory_existing"
COMMIT_NONCOND_BLOCKED = "noncond_memory_blocked"


@dataclass(slots=True)
class MaskStats:
    """Geometry summary for one object mask on one frame."""

    area_pixels: float
    area_frac: float
    bbox: list[int] | None
    centroid: list[float] | None
    edge_touch: bool = False

    @property
    def present(self) -> bool:
        return self.area_pixels > 0


@dataclass(slots=True)
class QualitySignals:
    """Tracker-quality signals before any identity/retrieval evidence."""

    objectness: float | None
    stability: float
    area_ratio: float | None
    displacement_px: float | None
    quality: float
    empty: bool
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ["area_ratio", "displacement_px"]:
            value = out.get(key)
            if value == float("inf"):
                out[key] = "inf"
        return out


@dataclass(slots=True)
class FrameAudit:
    """Per-frame RAR audit record requested by the experiment plan."""

    video: str
    obj_id: int
    frame_idx: int
    state: str
    base_mask_stats: MaskStats
    quality_signals: QualitySignals
    rcms_selected: list[int] = field(default_factory=list)
    candidate_count: int = 0
    best_candidate_score: float | None = None
    commit_decision: str = COMMIT_PROVISIONAL
    policy_decision: str = COMMIT_PROVISIONAL
    actual_memory_write: bool = False
    blocked_noncond_write: bool = False
    memory_storage_key: str = "none"
    promoted_cond_frames: list[int] = field(default_factory=list)
    used_cond_frames: list[int] = field(default_factory=list)
    output_policy: str = "provisional"
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "obj_id": self.obj_id,
            "frame_idx": self.frame_idx,
            "state": self.state,
            "base_mask_stats": asdict(self.base_mask_stats),
            "quality_signals": self.quality_signals.to_json(),
            "rcms_selected": list(self.rcms_selected),
            "candidate_count": int(self.candidate_count),
            "best_candidate_score": self.best_candidate_score,
            "commit_decision": self.commit_decision,
            "policy_decision": self.policy_decision,
            "actual_memory_write": bool(self.actual_memory_write),
            "blocked_noncond_write": bool(self.blocked_noncond_write),
            "memory_storage_key": self.memory_storage_key,
            "promoted_cond_frames": list(self.promoted_cond_frames),
            "used_cond_frames": list(self.used_cond_frames),
            "output_policy": self.output_policy,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class AnchorRecord:
    """A possible conditioned anchor.

    ``payload`` may hold a SAM2 output dict.  It is deliberately excluded from
    JSON so audits remain lightweight and serializable.
    """

    frame_idx: int
    obj_id: int
    stats: MaskStats
    quality: float
    source: Literal["init", "pre_disappearance", "retrieved", "negative"]
    payload: Any = None
    selected_as_conditioned: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "obj_id": self.obj_id,
            "stats": asdict(self.stats),
            "quality": float(self.quality),
            "source": self.source,
            "selected_as_conditioned": bool(self.selected_as_conditioned),
        }


@dataclass
class AnchorBank:
    """Per-object anchors with an RCMS-lite pre-disappearance reservoir."""

    obj_id: int
    init_anchor: AnchorRecord | None = None
    pre_disappearance_cond_reservoir: list[AnchorRecord] = field(default_factory=list)
    retrieved_anchors: list[AnchorRecord] = field(default_factory=list)
    negative_bank: list[AnchorRecord] = field(default_factory=list)
    max_pre_disappearance: int = 12
    min_quality: float = 0.60

    def set_init_anchor(self, anchor: AnchorRecord) -> None:
        self.init_anchor = anchor

    def add_pre_disappearance(self, anchor: AnchorRecord) -> bool:
        if anchor.quality < self.min_quality or not anchor.stats.present:
            return False
        already_present = any(
            existing.frame_idx == anchor.frame_idx
            for existing in self.pre_disappearance_cond_reservoir
        )
        if already_present:
            return False
        self.pre_disappearance_cond_reservoir.append(anchor)
        self.pre_disappearance_cond_reservoir.sort(key=lambda item: (-item.quality, item.frame_idx))
        del self.pre_disappearance_cond_reservoir[self.max_pre_disappearance :]
        return True

    def select_rcms(
        self,
        *,
        current_frame: int,
        max_anchors: int,
        min_quality: float | None = None,
    ) -> list[AnchorRecord]:
        """Select high-quality anchors nearest to disappearance/recovery."""

        threshold = self.min_quality if min_quality is None else min_quality
        eligible = [
            anchor
            for anchor in self.pre_disappearance_cond_reservoir
            if (
                anchor.frame_idx < current_frame
                and anchor.quality >= threshold
                and anchor.payload is not None
            )
        ]
        eligible.sort(key=lambda item: (abs(current_frame - item.frame_idx), -item.quality))
        selected = eligible[:max(0, max_anchors)]
        for anchor in selected:
            anchor.selected_as_conditioned = True
        return selected

    def to_json(self) -> dict[str, Any]:
        return {
            "obj_id": self.obj_id,
            "init_anchor": self.init_anchor.to_json() if self.init_anchor else None,
            "pre_disappearance_cond_reservoir": [
                a.to_json() for a in self.pre_disappearance_cond_reservoir
            ],
            "retrieved_anchors": [a.to_json() for a in self.retrieved_anchors],
            "negative_bank": [a.to_json() for a in self.negative_bank],
        }


@dataclass(slots=True)
class StateMachineConfig:
    """Thresholds for stable/ambiguous/recovery transitions."""

    stable_quality: float = 0.60
    ambiguous_quality: float = 0.35
    min_stable_area_pixels: float = 3.0
    empty_streak_to_recovery: int = 1
    ambiguous_streak_to_recovery: int = 3


@dataclass
class StateMachine:
    """Minimal stable -> ambiguous -> recovery state machine."""

    cfg: StateMachineConfig = field(default_factory=StateMachineConfig)
    state: str = STATE_STABLE
    empty_streak: int = 0
    ambiguous_streak: int = 0
    stable_streak: int = 0

    def update(self, stats: MaskStats, signals: QualitySignals) -> str:
        if signals.empty or stats.area_pixels < self.cfg.min_stable_area_pixels:
            self.empty_streak += 1
            self.ambiguous_streak = 0
            self.stable_streak = 0
            if self.empty_streak >= self.cfg.empty_streak_to_recovery:
                self.state = STATE_RECOVERY
            else:
                self.state = STATE_AMBIGUOUS
            return self.state

        self.empty_streak = 0
        if signals.quality >= self.cfg.stable_quality:
            self.stable_streak += 1
            self.ambiguous_streak = 0
            self.state = STATE_STABLE
        elif signals.quality >= self.cfg.ambiguous_quality:
            self.ambiguous_streak += 1
            self.stable_streak = 0
            self.state = STATE_AMBIGUOUS
        else:
            self.ambiguous_streak += 1
            self.stable_streak = 0
            self.state = (
                STATE_RECOVERY
                if self.ambiguous_streak >= self.cfg.ambiguous_streak_to_recovery
                else STATE_AMBIGUOUS
            )
        return self.state


@dataclass(slots=True)
class CommitPolicy:
    """Delayed promotion policy for current or retrieved candidates."""

    confirm_frames: int = 2
    min_candidate_score: float = 0.60
    candidate_streak: int = 0

    def current_decision(self, state: str, signals: QualitySignals) -> str:
        if state == STATE_STABLE and signals.quality >= self.min_candidate_score:
            self.candidate_streak = min(self.confirm_frames, self.candidate_streak + 1)
            return COMMIT_WRITE_MAIN
        self.candidate_streak = 0
        if state == STATE_RECOVERY:
            return COMMIT_PROMOTE_RCMS
        return COMMIT_PROVISIONAL

    def allow_retrieved_promotion(self, score: float, consistency_ok: bool) -> bool:
        if score >= self.min_candidate_score and consistency_ok:
            self.candidate_streak += 1
        else:
            self.candidate_streak = 0
        return self.candidate_streak >= self.confirm_frames


@dataclass(slots=True)
class SelectorDecision:
    action: Literal[
        "accept_main_path",
        "keep_uncertain_branch",
        "trigger_retrieval",
        "output_empty",
        "request_crop_refinement",
    ]
    reason: str


class Selector:
    """Rule-based first-version selector; no weighted global score."""

    def decide(
        self,
        state: str,
        signals: QualitySignals,
        *,
        has_retrieved_candidate: bool = False,
    ) -> SelectorDecision:
        if state == STATE_STABLE:
            return SelectorDecision("accept_main_path", "state_stable")
        if state == STATE_AMBIGUOUS:
            return SelectorDecision("keep_uncertain_branch", "state_ambiguous_no_commit")
        if has_retrieved_candidate:
            return SelectorDecision(
                "request_crop_refinement",
                "retrieved_candidate_needs_refinement",
            )
        if signals.empty:
            return SelectorDecision("output_empty", "recovery_empty_prediction")
        return SelectorDecision("trigger_retrieval", "recovery_nonempty_without_verified_anchor")

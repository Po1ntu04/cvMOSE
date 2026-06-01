"""Distractor-aware identity memory and delayed promotion rules.

M18 treats every later-frame proposal as a candidate that must beat the target
memory *and* hard-negative memory before it can become a tracker anchor.  This
module is model-agnostic: callers may feed DINOv2/DINOv3/SAM2/RGB descriptors;
the scoring and promotion gates remain auditable and unit-testable.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

PromotionAction = Literal["reject", "output_only", "promote", "needs_more_evidence"]
ObjectState = Literal[
    "STABLE",
    "AMBIGUOUS",
    "OCCLUDED",
    "RECOVERY_BRANCH",
    "RECONFIRMED",
    "REJECTED",
]


@dataclass(slots=True)
class DescriptorItem:
    item_id: str
    vector: list[float]
    source: str
    frame_idx: int | None = None
    tags: list[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass(slots=True)
class IdentityMemory:
    positive_pool: list[DescriptorItem] = field(default_factory=list)
    distractor_pool: list[DescriptorItem] = field(default_factory=list)
    rejected_pool: list[DescriptorItem] = field(default_factory=list)

    def add_positive(self, item: DescriptorItem) -> None:
        self.positive_pool.append(item)

    def add_distractor(self, item: DescriptorItem) -> None:
        self.distractor_pool.append(item)

    def add_rejected(self, item: DescriptorItem) -> None:
        self.rejected_pool.append(item)


@dataclass(slots=True)
class CandidateSignals:
    candidate_id: str
    vector: list[float] | None = None
    source: str = "unknown"
    frame_idx: int | None = None
    temporal_story_compatibility: float = 0.0
    source_independence_bonus: float = 0.0
    composite_background_risk: float = 0.0
    temporal_consistency_votes: int = 0
    qwen_tracklet_support: bool = False
    independent_source_agreement: bool = False
    same_class_dense: bool = False
    state: ObjectState = "AMBIGUOUS"
    hard_negative_hit: bool = False
    risk_tags: list[str] = field(default_factory=list)
    evidence_tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CandidateScore:
    candidate_id: str
    pos_sim: float
    neg_sim: float
    rejected_sim: float
    margin: float
    temporal_story_compatibility: float
    source_independence_bonus: float
    composite_background_risk: float
    score: float
    evidence_count: int
    decision: PromotionAction
    reasons: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        for key in [
            "pos_sim",
            "neg_sim",
            "rejected_sim",
            "margin",
            "temporal_story_compatibility",
            "source_independence_bonus",
            "composite_background_risk",
            "score",
        ]:
            out[key] = round(float(out[key]), 5)
        return out


@dataclass(slots=True)
class PromotionPolicy:
    min_positive_similarity: float = 0.40
    min_margin: float = 0.08
    same_class_min_margin: float = 0.18
    promote_score: float = 0.45
    output_only_score: float = 0.25
    reject_negative_margin: float = -0.02
    hard_negative_similarity: float = 0.72
    min_evidence_to_promote: int = 2
    same_class_min_evidence_to_promote: int = 2
    absent_allows_empty: bool = True


@dataclass(slots=True)
class StateTransition:
    previous_state: ObjectState
    new_state: ObjectState
    reason: str


class ObjectStateMachine:
    """Small ledger-level state machine used by retrieval/fusion tools."""

    def transition(
        self,
        previous: ObjectState,
        *,
        visible: bool,
        ambiguous: bool,
        candidate_score: CandidateScore | None = None,
    ) -> StateTransition:
        if previous == "REJECTED":
            return StateTransition(previous, "REJECTED", "already_rejected")
        if not visible:
            return StateTransition(previous, "OCCLUDED", "not_visible_or_empty_allowed")
        if ambiguous:
            return StateTransition(previous, "AMBIGUOUS", "visible_but_identity_ambiguous")
        if candidate_score and candidate_score.decision == "promote":
            return StateTransition(previous, "RECONFIRMED", "candidate_promoted_after_delayed_evidence")
        if candidate_score and candidate_score.decision == "output_only":
            return StateTransition(previous, "RECOVERY_BRANCH", "candidate_kept_output_only")
        return StateTransition(previous, "STABLE", "stable_current_best")


def l2_normalize(vec: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    denom = float(np.linalg.norm(arr))
    return arr / denom if denom > 1e-8 else arr


def cosine(a: Sequence[float] | np.ndarray | None, b: Sequence[float] | np.ndarray | None) -> float:
    av = l2_normalize(a)
    bv = l2_normalize(b)
    if av is None or bv is None:
        return 0.0
    if av.size != bv.size:
        n = min(int(av.size), int(bv.size))
        av = av[:n]
        bv = bv[:n]
    return float(np.dot(av, bv))


def weighted_pool_similarity(vec: Sequence[float] | np.ndarray | None, pool: Sequence[DescriptorItem]) -> float:
    if not pool:
        return 0.0
    best = 0.0
    for item in pool:
        sim = cosine(vec, item.vector) * max(0.0, float(item.weight))
        if sim > best:
            best = sim
    return float(best)


def evidence_count(signals: CandidateSignals, margin_ok: bool, story_ok: bool) -> int:
    count = 0
    if margin_ok:
        count += 1
    if story_ok:
        count += 1
    if int(signals.temporal_consistency_votes) >= 2:
        count += 1
    if signals.qwen_tracklet_support:
        count += 1
    if signals.independent_source_agreement or signals.source_independence_bonus > 0:
        count += 1
    # Explicit evidence tags allow non-descriptor external reviewers to add a
    # bounded signal without bypassing the delayed-confirmation count.
    count += min(1, len(signals.evidence_tags))
    return int(count)


def score_candidate(
    signals: CandidateSignals,
    memory: IdentityMemory,
    policy: PromotionPolicy | None = None,
) -> CandidateScore:
    """Score and gate one candidate against positive/negative identity memory."""
    policy = policy or PromotionPolicy()
    pos_sim = weighted_pool_similarity(signals.vector, memory.positive_pool)
    neg_sim = max(
        weighted_pool_similarity(signals.vector, memory.distractor_pool),
        weighted_pool_similarity(signals.vector, memory.rejected_pool),
    )
    rejected_sim = weighted_pool_similarity(signals.vector, memory.rejected_pool)
    margin = pos_sim - neg_sim
    story = float(max(-1.0, min(1.0, signals.temporal_story_compatibility)))
    source_bonus = float(max(0.0, signals.source_independence_bonus))
    composite_risk = float(max(0.0, signals.composite_background_risk))
    score = margin + 0.20 * story + source_bonus - composite_risk

    reasons: list[str] = []
    min_margin = policy.same_class_min_margin if signals.same_class_dense else policy.min_margin
    margin_ok = margin >= min_margin and pos_sim >= policy.min_positive_similarity
    story_ok = story > 0.25
    evid = evidence_count(signals, margin_ok, story_ok)

    if signals.state == "OCCLUDED":
        # A non-empty proposal inside an occluded/absent state is review
        # evidence, not an output candidate.  Empty-output policies are handled
        # explicitly by fusion tools with source="empty"; returning
        # needs_more_evidence here prevents accidental balanced fusion of a
        # visually plausible but temporally impossible same-class object.
        reasons.append("state_occluded_non_empty_candidate_review_only")
        return CandidateScore(
            signals.candidate_id,
            pos_sim,
            neg_sim,
            rejected_sim,
            margin,
            story,
            source_bonus,
            composite_risk,
            score,
            evid,
            "needs_more_evidence" if policy.absent_allows_empty else "reject",
            reasons,
        )

    hard_negative = (
        signals.hard_negative_hit
        or neg_sim >= policy.hard_negative_similarity
        or margin <= policy.reject_negative_margin
    )
    if hard_negative:
        reasons.append("closer_to_hard_negative_than_positive")
        return CandidateScore(
            signals.candidate_id,
            pos_sim,
            neg_sim,
            rejected_sim,
            margin,
            story,
            source_bonus,
            composite_risk,
            score,
            evid,
            "reject",
            reasons,
        )

    if not margin_ok:
        reasons.append(
            "low_same_class_margin" if signals.same_class_dense else "low_identity_margin"
        )
    if composite_risk > 0.20:
        reasons.append("composite_or_background_risk")
    if score < policy.output_only_score:
        reasons.append("score_below_output_only_threshold")
        return CandidateScore(
            signals.candidate_id,
            pos_sim,
            neg_sim,
            rejected_sim,
            margin,
            story,
            source_bonus,
            composite_risk,
            score,
            evid,
            "needs_more_evidence",
            reasons,
        )

    required_evidence = (
        policy.same_class_min_evidence_to_promote
        if signals.same_class_dense
        else policy.min_evidence_to_promote
    )
    if score >= policy.promote_score and evid >= required_evidence and composite_risk <= 0.20:
        reasons.append("delayed_promotion_evidence_met")
        return CandidateScore(
            signals.candidate_id,
            pos_sim,
            neg_sim,
            rejected_sim,
            margin,
            story,
            source_bonus,
            composite_risk,
            score,
            evid,
            "promote",
            reasons,
        )

    if signals.same_class_dense and evid < required_evidence:
        reasons.append("same_class_dense_single_evidence_no_promote")
    else:
        reasons.append("kept_as_recovery_branch_output_only")
    return CandidateScore(
        signals.candidate_id,
        pos_sim,
        neg_sim,
        rejected_sim,
        margin,
        story,
        source_bonus,
        composite_risk,
        score,
        evid,
        "output_only",
        reasons,
    )


def descriptor_item_from_vector(
    item_id: str,
    vector: Sequence[float] | np.ndarray,
    *,
    source: str,
    frame_idx: int | None = None,
    tags: list[str] | None = None,
    weight: float = 1.0,
) -> DescriptorItem:
    vec = l2_normalize(vector)
    if vec is None:
        raise ValueError("empty descriptor")
    return DescriptorItem(
        item_id=item_id,
        vector=[float(x) for x in vec.tolist()],
        source=source,
        frame_idx=frame_idx,
        tags=list(tags or []),
        weight=float(weight),
    )


def cheap_story_compatibility(frame_idx: int, event_story: Sequence[dict[str, Any]]) -> tuple[float, list[str]]:
    """Map a frame into the compact ledger story.

    Returns a bounded compatibility prior and tags.  This does not decide
    identity; it only prevents temporally impossible candidates from scoring as
    if they were plausible.
    """
    for entry in event_story:
        frames = entry.get("frames")
        if not (isinstance(frames, list) and len(frames) == 2):
            continue
        if int(frames[0]) <= int(frame_idx) <= int(frames[1]):
            allowed = str(entry.get("allowed_output", "review_only"))
            state = str(entry.get("state", ""))
            if allowed == "candidate":
                return 0.75, [f"story:{state}"]
            if allowed == "keep_current":
                return 0.40, [f"story:{state}"]
            if allowed == "empty":
                return -0.50, [f"story:{state}", "candidate_in_absent_window"]
            return 0.0, [f"story:{state}", "review_only_window"]
    return 0.0, ["story:unspecified"]


def risk_from_area_ratio(area_ratio_init: float | None, *, max_ratio: float = 8.0, min_ratio: float = 0.01) -> tuple[float, list[str]]:
    if area_ratio_init is None or not math.isfinite(float(area_ratio_init)):
        return 0.10, ["area_ratio_unknown"]
    ratio = float(area_ratio_init)
    if ratio <= 0:
        return 0.20, ["empty_or_zero_area_candidate"]
    risks: list[str] = []
    risk = 0.0
    if ratio > max_ratio:
        risk += min(0.45, 0.08 * math.log(max(1.0, ratio / max_ratio) + 1.0))
        risks.append("area_too_large_vs_first_frame")
    if ratio < min_ratio:
        risk += 0.20
        risks.append("area_too_small_vs_first_frame")
    return risk, risks


__all__ = [
    "DescriptorItem",
    "IdentityMemory",
    "CandidateSignals",
    "CandidateScore",
    "PromotionPolicy",
    "ObjectStateMachine",
    "StateTransition",
    "cosine",
    "descriptor_item_from_vector",
    "score_candidate",
    "cheap_story_compatibility",
    "risk_from_area_ratio",
]

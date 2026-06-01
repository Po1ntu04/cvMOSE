#!/usr/bin/env python3
"""Lightweight unit checks for M18 retrieval-memory policy."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.retrieval_memory import (  # noqa: E402
    CandidateSignals,
    IdentityMemory,
    PromotionPolicy,
    descriptor_item_from_vector,
    score_candidate,
)


def base_memory() -> IdentityMemory:
    mem = IdentityMemory()
    mem.add_positive(descriptor_item_from_vector("pos", [1, 0, 0], source="test"))
    mem.add_distractor(descriptor_item_from_vector("neg", [0, 1, 0], source="test"))
    return mem


def assert_case(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(name)


def main() -> None:
    mem = base_memory()
    policy = PromotionPolicy(promote_score=0.45, output_only_score=0.20)

    positive = score_candidate(
        CandidateSignals(
            candidate_id="positive_closer",
            vector=[0.95, 0.05, 0.0],
            temporal_story_compatibility=0.8,
            temporal_consistency_votes=2,
            independent_source_agreement=True,
        ),
        mem,
        policy,
    )
    assert_case("positive closer should be promotable", positive.decision == "promote")

    negative = score_candidate(
        CandidateSignals(
            candidate_id="negative_closer",
            vector=[0.05, 0.95, 0.0],
            temporal_story_compatibility=0.8,
            temporal_consistency_votes=2,
            independent_source_agreement=True,
        ),
        mem,
        policy,
    )
    assert_case("negative closer should reject", negative.decision == "reject")

    same_class_single = score_candidate(
        CandidateSignals(
            candidate_id="same_class_single",
            vector=[0.95, 0.05, 0.0],
            temporal_story_compatibility=0.0,
            temporal_consistency_votes=0,
            same_class_dense=True,
        ),
        mem,
        policy,
    )
    assert_case(
        "same-class dense single evidence should not promote",
        same_class_single.decision in {"output_only", "needs_more_evidence"},
    )

    absent = score_candidate(
        CandidateSignals(
            candidate_id="absent_empty_allowed",
            vector=[0.95, 0.05, 0.0],
            state="OCCLUDED",
            temporal_story_compatibility=-0.5,
        ),
        mem,
        policy,
    )
    assert_case(
        "occluded non-empty candidates should stay review-only",
        absent.decision == "needs_more_evidence",
    )

    print("M18 retrieval-memory unit checks passed")


if __name__ == "__main__":
    main()

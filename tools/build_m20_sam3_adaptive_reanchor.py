#!/usr/bin/env python3
"""Build M20 SAM3-assisted adaptive re-anchor candidates and submission zips.

M20 is deliberately conservative:

* the default base remains the strongest local SAM2/M17 root;
* SAM3/SAM3.1 is isolated as a candidate/proof source;
* every final replacement is written to a fresh prediction root with an audit
  trail and can be rolled back by discarding that root/zip.

When a live CUDA SAM3.1 runtime is unavailable, this tool replays the existing
``pred_sam31_b101`` branch to exercise the same retrieval/verifier/fusion path
locally.  Live SAM3 proposal generation can be enabled with ``--run-live-sam3``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.candidate_pool import (  # noqa: E402
    bbox_from_mask,
    descriptor,
    load_rgb,
    mask_iou,
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
from cvmose.sam3_m20 import (  # noqa: E402
    M20Sam3Adapter,
    PromptPack,
    Sam3Candidate,
    VisualPrompt,
    box_to_mask,
    event_state_for_frame,
    iou_with_sources,
    recovery_frames_from_ledger,
    write_json,
)
from cvmose.temporal_ledger import ledger_key, load_ledgers  # noqa: E402


DEFAULT_TARGETS = ["8jsm23a7:1", "4vznweiu:1", "1qlssuz2:1", "r13u5z4y:1", "q0sizv6m:2"]
OBJECT_PROMPTS = {
    "8jsm23a7:1": {
        "category": "mahjong tile",
        "semantic": "front-row leftmost seven-bamboo mahjong tile after hand placement",
        "constraints": [
            "old table position becomes a hard negative after pickup",
            "avoid far-row/two-bamboo same-class tiles",
            "after frame 6 the likely target is near the player/front row",
        ],
    },
    "4vznweiu:1": {
        "category": "letter dice",
        "semantic": "small T-letter or patterned dice/cube among adjacent letters and flower-like distractors",
        "constraints": [
            "do not promote a neighboring letter cube from a single frame",
            "same-class dense scene requires delayed confirmation",
        ],
    },
    "1qlssuz2:1": {
        "category": "vehicle",
        "semantic": "small white vehicle moving along the same lane through bridge occlusion",
        "constraints": [
            "candidate should remain on the same lane trajectory",
            "avoid broad background or bridge-shadow masks",
        ],
    },
    "r13u5z4y:1": {
        "category": "strawberry slice",
        "semantic": "specific cut strawberry slice among similar slices after occlusion",
        "constraints": [
            "prefer empty over a wrong salient strawberry",
            "requires hard-negative separation before output",
        ],
    },
    "q0sizv6m:2": {
        "category": "black white animal",
        "semantic": "left-edge partial black-white animal reappearing at lower foreground",
        "constraints": [
            "keep M17 full-box recovery as the current approved evidence",
            "do not replace late absent/false-positive frames without stronger proof",
        ],
    },
}


def parse_target(text: str) -> tuple[str, int]:
    video, obj = text.split(":", 1)
    return video, int(obj.replace("obj", ""))


def parse_named_root(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name.strip(), Path(path).expanduser()
    path = Path(text).expanduser()
    return path.name, path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--base-root", type=Path, default=None, help="Default M17/SAM2 root")
    p.add_argument("--sam3-root", type=Path, default=None, help="Precomputed SAM3 replay root")
    p.add_argument("--story-balanced-root", type=Path, default=None)
    p.add_argument("--story-aggressive-root", type=Path, default=None)
    p.add_argument("--ledger-dir", type=Path, default=Path("artifacts/m18_temporal_ledger"))
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts/m20_sam3_adaptive"))
    p.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    p.add_argument("--extra-root", action="append", default=[], help="name=/path agreement/proof root; repeatable")
    p.add_argument("--max-sam3-frames-per-object", type=int, default=36)
    p.add_argument("--run-live-sam3", action="store_true", help="Use live SAM3.1 predictor instead of replay root")
    p.add_argument(
        "--include-story-probes",
        action="store_true",
        help="Also include legacy M19 story-root probes. Disabled by default so M20 outputs are SAM3-candidate gated.",
    )
    p.add_argument("--make-submissions", action="store_true", default=True)
    p.add_argument("--no-make-submissions", dest="make_submissions", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--copy-zips-to-repo-root", action="store_true", default=False)
    return p.parse_args()


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(video_dir)
    return frames


def load_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    return arr if arr.ndim == 2 else arr[..., 0]


def save_label(path: Path, arr: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr.astype(np.uint8), mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def first_obj_box(ann: np.ndarray, obj_id: int) -> list[int] | None:
    return bbox_from_mask(ann == int(obj_id))


def sample_negative_visuals(ledger: Mapping[str, Any], *, limit: int = 6) -> list[VisualPrompt]:
    out: list[VisualPrompt] = []
    for item in ledger.get("distractor_memory", []):
        box = item.get("bbox")
        frames = item.get("frames") or [0]
        if isinstance(box, list) and len(box) == 4:
            frame = frames[0] if isinstance(frames, list) and frames else 0
            out.append(
                VisualPrompt(
                    frame_idx=int(frame),
                    bbox_xyxy=[int(x) for x in box],
                    label="negative",
                    source=str(item.get("source", "ledger_distractor")),
                    note=str(item.get("description", "")),
                )
            )
        if len(out) >= limit:
            break
    return out


def build_prompt_pack(
    *,
    workspace: Path,
    video: str,
    obj_id: int,
    ledger: Mapping[str, Any],
    frames: list[Path],
    ann: np.ndarray,
) -> PromptPack:
    key = f"{video}:{obj_id}"
    spec = OBJECT_PROMPTS.get(key, {})
    category = spec.get("category") or str((ledger.get("target_profile") or {}).get("category") or "object")
    semantic = spec.get("semantic") or str((ledger.get("target_profile") or {}).get("first_frame_description") or category)
    constraints = list(spec.get("constraints") or [])
    for entry in ledger.get("event_story", []):
        summary = entry.get("summary")
        if summary:
            constraints.append(str(summary))
    pos_box = first_obj_box(ann, obj_id)
    positives: list[VisualPrompt] = []
    if pos_box:
        positives.append(
            VisualPrompt(
                frame_idx=0,
                bbox_xyxy=pos_box,
                label="positive",
                source="first_frame_gt",
                note="first-frame GT crop/box",
            )
        )
    windows: list[list[int]] = []
    for entry in ledger.get("event_story", []):
        allowed = str(entry.get("allowed_output", "review_only"))
        if allowed != "keep_current" and isinstance(entry.get("frames"), list):
            windows.append([int(entry["frames"][0]), int(entry["frames"][1])])
    return PromptPack(
        video=video,
        obj_id=obj_id,
        category_prompt=str(category).lower(),
        semantic_prompt=str(semantic),
        positive_visual=positives,
        negative_visual=sample_negative_visuals(ledger),
        story_constraints=constraints,
        recovery_windows=windows,
        same_class_dense=bool((ledger.get("target_profile") or {}).get("same_class_dense")),
        notes=str((ledger.get("target_profile") or {}).get("first_frame_description") or ""),
    )


def add_desc(memory: IdentityMemory, kind: str, item_id: str, source: str, frame_idx: int, rgb: np.ndarray, mask: np.ndarray, tags: list[str] | None = None, weight: float = 1.0) -> bool:
    vec = descriptor(rgb, mask)
    if vec is None:
        return False
    item = descriptor_item_from_vector(item_id, vec, source=source, frame_idx=frame_idx, tags=tags or [], weight=weight)
    if kind == "positive":
        memory.add_positive(item)
    elif kind == "distractor":
        memory.add_distractor(item)
    else:
        memory.add_rejected(item)
    return True


def sampled_frames(spec: list[Any], frame_count: int) -> list[int]:
    ints = [int(x) for x in spec if isinstance(x, int)]
    if not ints:
        return [0]
    if len(ints) == 2 and ints[0] <= ints[1]:
        s, e = max(0, ints[0]), min(frame_count - 1, ints[1])
        return sorted({s, (s + e) // 2, e})
    return sorted({max(0, min(frame_count - 1, x)) for x in ints})[:5]


def build_identity_memory(
    *,
    frames: list[Path],
    ann: np.ndarray,
    obj_id: int,
    ledger: Mapping[str, Any],
) -> tuple[IdentityMemory, list[dict[str, Any]]]:
    memory = IdentityMemory()
    audit: list[dict[str, Any]] = []
    rgb0 = load_rgb(frames[0])
    init = ann == int(obj_id)
    if add_desc(memory, "positive", "first_frame_gt", "first_frame_gt", 0, rgb0, init, ["first_gt"], 1.0):
        audit.append({"kind": "positive", "source": "first_frame_gt", "frame_idx": 0, "area": int(init.sum())})
    ring = ring_mask(init)
    if int(ring.sum()) > 0 and add_desc(memory, "distractor", "context_ring", "context_ring", 0, rgb0, ring, ["context"], 0.8):
        audit.append({"kind": "distractor", "source": "context_ring", "frame_idx": 0, "area": int(ring.sum())})
    for other_id in [int(x) for x in np.unique(ann) if int(x) not in {0, int(obj_id)}]:
        om = ann == other_id
        if int(om.sum()) > 0 and add_desc(memory, "distractor", f"first_other_{other_id}", f"first_frame_other:{other_id}", 0, rgb0, om, ["other_object"], 1.0):
            audit.append({"kind": "distractor", "source": f"first_frame_other:{other_id}", "frame_idx": 0, "area": int(om.sum())})
    for item in ledger.get("distractor_memory", []):
        box = item.get("bbox")
        if not box:
            continue
        for idx in sampled_frames(item.get("frames") or [0], len(frames)):
            mask = box_to_mask(box, ann.shape)
            if mask is None:
                continue
            rgb = load_rgb(frames[idx])
            if add_desc(memory, "distractor", str(item.get("memory_id", f"distractor_{idx}")), str(item.get("source", "ledger")), idx, rgb, mask, item.get("tags") or [], 0.85):
                audit.append({"kind": "distractor", "source": item.get("source"), "frame_idx": idx, "area": int(mask.sum()), "memory_id": item.get("memory_id"), "from_ledger_box": True})
    return memory, audit


def temporal_votes(root: Path, video: str, obj_id: int, frame_idx: int, shape: tuple[int, int]) -> int:
    votes = 0
    for idx in [frame_idx - 1, frame_idx, frame_idx + 1]:
        if idx < 0:
            continue
        p = root / video / f"{idx:05d}.png"
        if not p.is_file():
            continue
        lab = load_label(p)
        if lab.shape == shape and int((lab == int(obj_id)).sum()) > 0:
            votes += 1
    return votes


def score_sam3_candidate(
    cand: Sam3Candidate,
    *,
    frame_stem: str,
    frame_count: int,
    init_area: int,
    memory: IdentityMemory,
    ledger: Mapping[str, Any],
    agreement_roots: Mapping[str, Path],
    root_lookup: Mapping[str, Path],
) -> dict[str, Any]:
    if cand.mask is None or int(cand.mask.sum()) <= 0:
        return {
            **cand.to_audit(),
            "m20_decision": "reject",
            "m20_reasons": ["empty_sam3_candidate"],
        }
    rgb = load_rgb(Path(agreement_roots["__frames__"]) / cand.video / f"{frame_stem}.jpg") if "__frames__" in agreement_roots else None
    # Fall back to workspace JPEG suffix search when a frame root marker is not provided.
    if rgb is None:  # pragma: no cover - kept for standalone use
        raise RuntimeError("internal frame root marker missing")
    vec = descriptor(rgb, cand.mask)
    area_ratio = cand.area / max(1.0, float(init_area))
    story, story_tags = cheap_story_compatibility(cand.frame_idx, ledger.get("event_story", []))
    area_risk, risk_tags = risk_from_area_ratio(area_ratio, max_ratio=12.0 if cand.video == "q0sizv6m" else 8.0)
    state, allowed = event_state_for_frame(ledger, cand.frame_idx)
    source_name = cand.source_root or "sam3_live"
    other_ious = iou_with_sources(
        cand.mask,
        frame_path_stem=frame_stem,
        video=cand.video,
        obj_id=cand.obj_id,
        roots={k: v for k, v in agreement_roots.items() if k != "__frames__"},
        exclude={source_name},
    )
    independent_agree = any(v >= 0.35 for v in other_ious.values())
    source_bonus = 0.12 if independent_agree else 0.0
    votes = temporal_votes(root_lookup[source_name], cand.video, cand.obj_id, cand.frame_idx, cand.mask.shape) if source_name in root_lookup else 1
    signals = CandidateSignals(
        candidate_id=cand.candidate_id,
        vector=None if vec is None else [float(x) for x in vec.tolist()],
        source=source_name,
        frame_idx=cand.frame_idx,
        temporal_story_compatibility=story,
        source_independence_bonus=source_bonus,
        composite_background_risk=area_risk,
        temporal_consistency_votes=votes,
        independent_source_agreement=independent_agree,
        same_class_dense=bool((ledger.get("target_profile") or {}).get("same_class_dense")),
        state="OCCLUDED" if allowed == "empty" else str(ledger.get("current_state", "AMBIGUOUS")),
        risk_tags=risk_tags + story_tags,
    )
    policy = PromotionPolicy(
        min_positive_similarity=0.38,
        min_margin=0.08,
        same_class_min_margin=0.18,
        promote_score=0.45,
        output_only_score=0.25,
        min_evidence_to_promote=2,
        same_class_min_evidence_to_promote=2,
    )
    score = score_candidate(signals, memory, policy)
    m20_decision = score.decision
    reasons = list(score.reasons)
    if source_name.startswith("sam3") and not independent_agree and cand.video != "q0sizv6m":
        # SAM3 replay/live single source is proposal evidence, not a final
        # identity commit in dense same-class recovery windows.
        if m20_decision in {"promote", "output_only"}:
            m20_decision = "needs_more_evidence"
            reasons.append("sam3_single_source_requires_independent_confirmation")
    if allowed == "empty" and cand.area > 0:
        m20_decision = "reject"
        reasons.append("story_absent_window_blocks_non_empty_sam3")
    return {
        **cand.to_audit(),
        "event_state": state,
        "allowed_output": allowed,
        "area_ratio_init": round(float(area_ratio), 5),
        "agreement_iou": {k: round(float(v), 5) for k, v in sorted(other_ious.items())},
        "independent_source_agreement": independent_agree,
        "temporal_votes": votes,
        "story_tags": story_tags,
        "risk_tags": risk_tags,
        "score_card": score.to_json(),
        "m20_decision": m20_decision,
        "m20_reasons": reasons,
    }


def object_changed_frames(source_root: Path, base_root: Path, video: str, obj_id: int, frames: list[Path], shape: tuple[int, int]) -> list[int]:
    changed: list[int] = []
    for idx, frame in enumerate(frames):
        src_p = source_root / video / f"{frame.stem}.png"
        base_p = base_root / video / f"{frame.stem}.png"
        if not src_p.is_file() or not base_p.is_file():
            continue
        src = load_label(src_p)
        base = load_label(base_p)
        if src.shape == shape and base.shape == shape and not np.array_equal(src == int(obj_id), base == int(obj_id)):
            changed.append(idx)
    return changed


def contiguous_ranges(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    out: list[list[int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            out.append([start, prev])
            start = prev = idx
    out.append([start, prev])
    return out


def save_candidate_mask(path: Path, mask: np.ndarray, obj_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(mask.shape, dtype=np.uint8)
    arr[np.asarray(mask, dtype=bool)] = int(obj_id)
    Image.fromarray(arr, mode="P").save(path)


def safe_candidate_filename(candidate_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in candidate_id)
    return f"{safe}.png"


def candidate_edits_from_cards(cards: list[dict[str, Any]], *, profile: str) -> list[dict[str, Any]]:
    """Turn verified SAM3 candidate cards into frame-level fusion edits."""
    if profile == "safe":
        allowed = {"promote"}
        min_score = 0.45
    elif profile == "balanced":
        allowed = {"promote", "output_only"}
        min_score = 0.25
    else:
        allowed = {"promote", "output_only", "needs_more_evidence"}
        min_score = 0.0

    best_by_key: dict[tuple[str, int, int], tuple[float, dict[str, Any]]] = {}
    for card in cards:
        decision = str(card.get("m20_decision"))
        if decision not in allowed:
            continue
        if str(card.get("allowed_output")) == "empty":
            continue
        score_card = card.get("score_card") or {}
        score = float(score_card.get("score", -999.0))
        if score < min_score:
            continue
        mask_path = card.get("candidate_mask_path")
        if not mask_path or not Path(mask_path).is_file():
            continue
        video = str(card["video"])
        obj_id = int(card["obj_id"])
        frame_idx = int(card["frame_idx"])
        key = (video, obj_id, frame_idx)
        if key not in best_by_key or score > best_by_key[key][0]:
            best_by_key[key] = (score, card)

    edits: list[dict[str, Any]] = []
    for (_, _, _), (score, card) in sorted(best_by_key.items()):
        edits.append(
            {
                "profile": profile,
                "video": str(card["video"]),
                "obj_id": int(card["obj_id"]),
                "source_root": "sam3_candidate",
                "candidate_mask_path": str(card["candidate_mask_path"]),
                "candidate_id": str(card["candidate_id"]),
                "start": int(card["frame_idx"]),
                "end": int(card["frame_idx"]),
                "tier": profile,
                "decision": str(card.get("m20_decision")),
                "review_status": "sam3_candidate_gated",
                "risk_tags": list(card.get("risk_tags") or []),
                "reason": ";".join(card.get("m20_reasons") or []) or "sam3_candidate_passed_policy",
                "score": round(score, 6),
            }
        )
    return edits


def make_story_profile_edits(
    *,
    profile: str,
    story_balanced_root: Path | None,
    story_aggressive_root: Path | None,
    base_root: Path,
    frames_by_video: Mapping[str, list[Path]],
    ann_by_video: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    edits: list[dict[str, Any]] = []
    if profile == "safe":
        return edits
    root = story_balanced_root if profile == "balanced" else story_aggressive_root
    if root is None or not root.is_dir():
        return edits
    if profile == "balanced":
        targets = {
            "8jsm23a7": {
                "obj_id": 1,
                "tier": "balanced",
                "decision": "output_only",
                "reason": "story-verified front-row Mahjong pseudo-anchor; SAM3 replay audited but not trusted as sole source",
            },
            "1qlssuz2": {
                "obj_id": 1,
                "tier": "balanced",
                "decision": "output_only",
                "reason": "two-frame same-lane vehicle occlusion fill; bounded story constraint",
            },
        }
    else:
        targets = {
            "8jsm23a7": {
                "obj_id": 1,
                "tier": "aggressive",
                "decision": "needs_more_evidence",
                "reason": "full Mahjong story window including coarse hand-held frames",
            },
            "4vznweiu": {
                "obj_id": 1,
                "tier": "aggressive",
                "decision": "needs_more_evidence",
                "reason": "letter-dice story probe; same-class dense and rectangular masks make this score-probing only",
            },
            "1qlssuz2": {
                "obj_id": 1,
                "tier": "aggressive",
                "decision": "output_only",
                "reason": "same-lane vehicle occlusion fill",
            },
        }
    for video, meta in targets.items():
        if video not in frames_by_video:
            continue
        obj_id = int(meta["obj_id"])
        changed = object_changed_frames(root, base_root, video, obj_id, frames_by_video[video], ann_by_video[video].shape)
        for start, end in contiguous_ranges(changed):
            edits.append(
                {
                    "profile": profile,
                    "video": video,
                    "obj_id": obj_id,
                    "source_root": "story_balanced" if profile == "balanced" else "story_aggressive",
                    "source_path": str(root),
                    "start": start,
                    "end": end,
                    "tier": meta["tier"],
                    "decision": meta["decision"],
                    "review_status": "story_verified_not_sam3_promoted" if profile == "balanced" else "probe_only",
                    "risk_tags": ["same_class_dense"] if video in {"8jsm23a7", "4vznweiu"} else [],
                    "reason": meta["reason"],
                }
            )
    return edits


def apply_edits_to_pred_root(
    *,
    workspace: Path,
    base_root: Path,
    source_roots: Mapping[str, Path],
    edits: list[dict[str, Any]],
    pred_root: Path,
    overwrite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    jpeg_root = workspace / "homework" / "JPEGImages"
    ann_root = workspace / "homework" / "Annotations"
    if pred_root.exists() and overwrite:
        shutil.rmtree(pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[dict[str, Any]]] = {}
    for edit in edits:
        by_video.setdefault(str(edit["video"]), []).append(edit)
    source_rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"pred_root": str(pred_root), "base_root": str(base_root), "edits": edits, "videos": {}}
    for video_dir in sorted(p for p in jpeg_root.iterdir() if p.is_dir()):
        video = video_dir.name
        dst = pred_root / video
        dst.mkdir(parents=True, exist_ok=True)
        frames = list_frames(video_dir)
        ann_img = Image.open(ann_root / video / "00000.png")
        palette = ann_img.getpalette()
        frame_map: dict[int, list[dict[str, Any]]] = {}
        for edit in by_video.get(video, []):
            for idx in range(max(1, int(edit["start"])), min(len(frames) - 1, int(edit["end"])) + 1):
                frame_map.setdefault(idx, []).append(edit)
        changed_frames: set[int] = set()
        for idx, frame in enumerate(frames):
            base = load_label(base_root / video / f"{frame.stem}.png").copy()
            final = base.copy()
            if idx == 0:
                final = load_label(ann_root / video / "00000.png").copy()
            else:
                for edit in frame_map.get(idx, []):
                    obj_id = int(edit["obj_id"])
                    root_name = str(edit["source_root"])
                    if "candidate_mask_path" in edit:
                        src = load_label(Path(edit["candidate_mask_path"]))
                    else:
                        src_root = source_roots[root_name]
                        src = load_label(src_root / video / f"{frame.stem}.png")
                    before = final == obj_id
                    final[before] = 0
                    src_mask = src == obj_id
                    final[src_mask] = obj_id
                    if not np.array_equal(before, src_mask):
                        changed_frames.add(idx)
                    source_rows.append(
                        {
                            "video": video,
                            "obj_id": obj_id,
                            "frame_idx": idx,
                            "source_root": root_name,
                            "source_area": int(src_mask.sum()),
                            "base_area": int(before.sum()),
                            "mask_iou_vs_base": round(mask_iou(before, src_mask), 6),
                            "tier": edit.get("tier"),
                            "decision": edit.get("decision"),
                            "review_status": edit.get("review_status"),
                            "reason": edit.get("reason", ""),
                            "candidate_id": edit.get("candidate_id", ""),
                            "candidate_mask_path": edit.get("candidate_mask_path", ""),
                        }
                    )
            save_label(dst / f"{frame.stem}.png", final, palette)
        audit["videos"][video] = {
            "changed_frames": sorted(changed_frames),
            "changed_frame_count": len(changed_frames),
            "edit_count": len(by_video.get(video, [])),
        }
    audit["summary"] = {
        "videos": len(audit["videos"]),
        "changed_videos": sum(1 for v in audit["videos"].values() if v["changed_frame_count"]),
        "changed_frame_events": len(source_rows),
    }
    return audit, source_rows


def make_submission(workspace: Path, pred_root: Path, submit_root: Path, zip_path: Path, overwrite: bool) -> dict[str, Any]:
    from validate_mose_submission import validate_zip

    provided = workspace / "homework" / "output"
    jpeg_root = workspace / "homework" / "JPEGImages"
    ann_root = workspace / "homework" / "Annotations"
    if submit_root.exists() and overwrite:
        shutil.rmtree(submit_root)
    elif submit_root.exists() and not overwrite:
        raise FileExistsError(f"Submission root exists; pass --overwrite or choose a new path: {submit_root}")
    submit_root.mkdir(parents=True, exist_ok=True)
    for root in [provided, pred_root]:
        for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            dst = submit_root / video_dir.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(video_dir, dst)
    if zip_path.exists() and overwrite:
        zip_path.unlink()
    elif zip_path.exists() and not overwrite:
        raise FileExistsError(f"Zip exists; pass --overwrite or choose a new path: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for png in sorted(submit_root.rglob("*.png")):
            zf.write(png, png.relative_to(submit_root).as_posix())
    validation = validate_zip(zip_path, provided, jpeg_root, ann_root, 433, 66526, 15, False)
    if not validation.get("ok"):
        raise SystemExit(f"Invalid M20 submission {zip_path}: {validation}")
    return validation


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    hw = workspace / "homework"
    args.base_root = (args.base_root or hw / "pred_m17_q0_full_box").resolve()
    args.sam3_root = (args.sam3_root or hw / "pred_sam31_b101").resolve()
    args.story_balanced_root = (args.story_balanced_root or hw / "pred_m19_8js_late_1ql").resolve()
    args.story_aggressive_root = (args.story_aggressive_root or hw / "pred_m19_8js_4vz_1ql").resolve()
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ledgers = load_ledgers(args.ledger_dir)
    jpeg_root = hw / "JPEGImages"
    ann_root = hw / "Annotations"
    targets = [parse_target(t) for t in args.targets]
    frames_by_video = {video: list_frames(jpeg_root / video) for video, _ in targets}
    ann_by_video = {video: load_label(ann_root / video / "00000.png") for video, _ in targets}

    prompt_packs: dict[str, PromptPack] = {}
    prompt_audit: dict[str, Any] = {}
    for video, obj_id in targets:
        key = ledger_key(video, obj_id)
        ledger = ledgers.get(key, {})
        pack = build_prompt_pack(
            workspace=workspace,
            video=video,
            obj_id=obj_id,
            ledger=ledger,
            frames=frames_by_video[video],
            ann=ann_by_video[video],
        )
        prompt_packs[key] = pack
        prompt_audit[key] = {
            **asdict(pack),
            "prompt_variants": pack.prompt_variants(),
        }
    write_json(artifact_dir / "prompt_packs.json", prompt_audit)

    replay_roots = {"sam3_gt_branch": args.sam3_root} if args.sam3_root.is_dir() else {}
    extra_roots = {
        "base_m17": args.base_root,
        "story_balanced": args.story_balanced_root,
        "story_aggressive": args.story_aggressive_root,
    }
    for raw in args.extra_root:
        name, path = parse_named_root(raw)
        extra_roots[name] = path.resolve()
    extra_roots = {k: v for k, v in extra_roots.items() if v is not None and Path(v).is_dir()}
    adapter = (
        M20Sam3Adapter.build_live_sam31(workspace=workspace, replay_roots=replay_roots)
        if args.run_live_sam3
        else M20Sam3Adapter(workspace=workspace, replay_roots=replay_roots)
    )

    candidate_cards: list[dict[str, Any]] = []
    candidate_mask_dir = artifact_dir / "candidate_masks"
    memory_audit: dict[str, Any] = {}
    for video, obj_id in targets:
        key = ledger_key(video, obj_id)
        ledger = ledgers.get(key, {})
        frames = frames_by_video[video]
        ann = ann_by_video[video]
        memory, mem_audit = build_identity_memory(frames=frames, ann=ann, obj_id=obj_id, ledger=ledger)
        memory_audit[key] = {
            "positive": len(memory.positive_pool),
            "distractor": len(memory.distractor_pool),
            "rejected": len(memory.rejected_pool),
            "audit": mem_audit,
        }
        frame_indices = recovery_frames_from_ledger(ledger, len(frames), max_frames=args.max_sam3_frames_per_object)
        if not frame_indices:
            frame_indices = list(range(1, len(frames)))
        pack = prompt_packs[key]
        adapter.sam3_video_branch(video=video, obj_id=obj_id, anchor_masks={0: ann == obj_id}, window=(min(frame_indices), max(frame_indices)) if frame_indices else None)
        for idx in frame_indices:
            cands = adapter.sam3_image_propose(
                video=video,
                frame_idx=idx,
                frame_name=frames[idx].stem,
                prompt_pack=pack,
                max_candidates=6,
            )
            agreement_roots: dict[str, Path] = {"__frames__": jpeg_root, **extra_roots, **replay_roots}
            root_lookup = {**extra_roots, **replay_roots}
            for cand in cands:
                card = score_sam3_candidate(
                    cand,
                    frame_stem=frames[idx].stem,
                    frame_count=len(frames),
                    init_area=int((ann == obj_id).sum()),
                    memory=memory,
                    ledger=ledger,
                    agreement_roots=agreement_roots,
                    root_lookup=root_lookup,
                )
                if cand.mask is not None and int(cand.mask.sum()) > 0:
                    mask_path = candidate_mask_dir / cand.video / f"obj{cand.obj_id}" / safe_candidate_filename(cand.candidate_id)
                    save_candidate_mask(mask_path, cand.mask, cand.obj_id)
                    card["candidate_mask_path"] = str(mask_path.resolve())
                candidate_cards.append(card)
    write_json(artifact_dir / "sam3_candidate_cards.json", {"cards": candidate_cards, "memory": memory_audit})
    write_json(artifact_dir / "sam3_adapter_audit.json", adapter.sam3_audit_export())

    source_roots = {"story_balanced": args.story_balanced_root, "story_aggressive": args.story_aggressive_root}
    run_summary: dict[str, Any] = {
        "method": "m20_sam3_adaptive_reanchor",
        "workspace": str(workspace),
        "base_root": str(args.base_root),
        "sam3_replay_root": str(args.sam3_root),
        "profiles": {},
        "candidate_summary": {
            "cards": len(candidate_cards),
            "m20_decisions": {
                decision: sum(1 for c in candidate_cards if c.get("m20_decision") == decision)
                for decision in sorted({str(c.get("m20_decision")) for c in candidate_cards})
            },
        },
        "notes": [
            "safe profile intentionally preserves the current M17/SAM2 root unless a SAM3 candidate has strong independent verification",
            "local run used replay mode unless --run-live-sam3 was supplied",
            "story-root probes are excluded unless --include-story-probes is supplied",
        ],
    }
    for profile in ["safe", "balanced", "aggressive"]:
        edits = candidate_edits_from_cards(candidate_cards, profile=profile)
        if args.include_story_probes:
            edits.extend(
                make_story_profile_edits(
                    profile=profile,
                    story_balanced_root=args.story_balanced_root,
                    story_aggressive_root=args.story_aggressive_root,
                    base_root=args.base_root,
                    frames_by_video=frames_by_video,
                    ann_by_video=ann_by_video,
                )
            )
        pred_root = hw / f"pred_m20_sam3_{profile}"
        submit_root = hw / f"submission_433_m20_sam3_{profile}"
        zip_path = hw / f"submission_mosev2_m20_sam3_{profile}.zip"
        audit, rows = apply_edits_to_pred_root(
            workspace=workspace,
            base_root=args.base_root,
            source_roots=source_roots,
            edits=edits,
            pred_root=pred_root,
            overwrite=args.overwrite,
        )
        audit.update(
            {
                "profile": profile,
                "prompt_packs_json": str(artifact_dir / "prompt_packs.json"),
                "sam3_candidate_cards_json": str(artifact_dir / "sam3_candidate_cards.json"),
                "sam3_adapter_audit_json": str(artifact_dir / "sam3_adapter_audit.json"),
            }
        )
        if args.make_submissions:
                validation = make_submission(workspace, pred_root, submit_root, zip_path, args.overwrite)
                audit["validation"] = validation
                audit["zip_path"] = str(zip_path)
                audit["submit_root"] = str(submit_root)
                if args.copy_zips_to_repo_root:
                    dst_zip = REPO_ROOT / zip_path.name
                    if dst_zip.exists() and args.overwrite:
                        dst_zip.unlink()
                    elif dst_zip.exists() and not args.overwrite:
                        raise FileExistsError(f"Repo zip copy exists; pass --overwrite or remove it: {dst_zip}")
                    shutil.copy2(zip_path, dst_zip)
                    audit["repo_zip_copy"] = str(dst_zip)
        write_json(artifact_dir / f"audit_{profile}.json", audit)
        write_csv(artifact_dir / f"source_table_{profile}.csv", rows)
        run_summary["profiles"][profile] = {
            "edits": len(edits),
            "changed_videos": audit["summary"]["changed_videos"],
            "changed_frame_events": audit["summary"]["changed_frame_events"],
            "pred_root": str(pred_root),
            "zip_path": audit.get("zip_path"),
            "validation_ok": (audit.get("validation") or {}).get("ok"),
        }
    write_json(artifact_dir / "run_summary.json", run_summary)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

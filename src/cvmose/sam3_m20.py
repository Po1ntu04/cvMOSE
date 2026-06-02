"""SAM3-assisted adaptive re-anchor utilities for M20.

This module keeps SAM3 as an isolated candidate/proof source.  It intentionally
does not mutate SAM2/M11/M17 prediction roots; callers must explicitly fuse a
verified candidate into a new prediction root.
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cvmose.candidate_pool import bbox_from_mask, centroid_from_mask, mask_iou


@dataclass(slots=True)
class VisualPrompt:
    """A positive or negative visual prompt box in absolute pixel coordinates."""

    frame_idx: int
    bbox_xyxy: list[int]
    label: str = "positive"
    source: str = "unknown"
    note: str = ""

    def to_normalized_xywh(self, width: int, height: int) -> list[float]:
        x1, y1, x2, y2 = [float(v) for v in self.bbox_xyxy]
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"degenerate prompt box {self.bbox_xyxy}")
        return [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]


@dataclass(slots=True)
class PromptPack:
    """Per video/object SAM3 prompt package."""

    video: str
    obj_id: int
    category_prompt: str
    semantic_prompt: str
    positive_visual: list[VisualPrompt] = field(default_factory=list)
    negative_visual: list[VisualPrompt] = field(default_factory=list)
    story_constraints: list[str] = field(default_factory=list)
    recovery_windows: list[list[int]] = field(default_factory=list)
    same_class_dense: bool = False
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.video}:obj{self.obj_id}"

    def prompt_variants(self, *, include_negative: bool = True) -> list[dict[str, Any]]:
        """Return the planned SAM3 prompt combinations for audit/replay."""
        text = self.category_prompt.strip()
        semantic = self.semantic_prompt.strip()
        variants: list[dict[str, Any]] = [
            {
                "name": "text_only",
                "text_prompt": text,
                "positive_visual": [],
                "negative_visual": [],
            },
            {
                "name": "semantic_text_only",
                "text_prompt": semantic or text,
                "positive_visual": [],
                "negative_visual": [],
            },
            {
                "name": "visual_only",
                "text_prompt": "visual",
                "positive_visual": [asdict(v) for v in self.positive_visual[:1]],
                "negative_visual": [],
            },
            {
                "name": "text_positive_visual",
                "text_prompt": text,
                "positive_visual": [asdict(v) for v in self.positive_visual[:1]],
                "negative_visual": [],
            },
        ]
        if include_negative and self.negative_visual:
            variants.append(
                {
                    "name": "text_positive_negative_visual",
                    "text_prompt": text,
                    "positive_visual": [asdict(v) for v in self.positive_visual[:1]],
                    "negative_visual": [asdict(v) for v in self.negative_visual[:4]],
                    "note": "negative boxes are used only for image proposal recall/audit, not assumed to be tracker semantics",
                }
            )
        return variants


@dataclass(slots=True)
class Sam3Candidate:
    """Normalized SAM3 proposal/tracker candidate record."""

    candidate_id: str
    video: str
    obj_id: int
    frame_idx: int
    mask: np.ndarray | None
    bbox: list[int] | None
    score: float | None
    prompt_source: str
    text_prompt: str | None = None
    visual_prompt: list[dict[str, Any]] = field(default_factory=list)
    source_root: str | None = None
    mask_kind: str = "object_id"
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else 0

    def to_audit(self, *, include_mask: bool = False) -> dict[str, Any]:
        out = {
            "candidate_id": self.candidate_id,
            "video": self.video,
            "obj_id": self.obj_id,
            "frame_idx": self.frame_idx,
            "area": self.area,
            "bbox": self.bbox,
            "centroid": None if self.mask is None else centroid_from_mask(self.mask),
            "score": self.score,
            "prompt_source": self.prompt_source,
            "text_prompt": self.text_prompt,
            "visual_prompt": self.visual_prompt,
            "source_root": self.source_root,
            "mask_kind": self.mask_kind,
            "audit": self.audit,
        }
        if include_mask and self.mask is not None:
            out["mask_rle"] = bool_mask_to_uncompressed_rle(self.mask)
        return out


def bool_mask_to_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    """Small JSON-safe uncompressed RLE for audit/debug only."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return {"shape": list(mask.shape), "counts": []}
    counts: list[int] = []
    current = int(flat[0])
    run = 1
    for val in flat[1:].tolist():
        val = int(val)
        if val == current:
            run += 1
        else:
            counts.extend([current, run])
            current = val
            run = 1
    counts.extend([current, run])
    return {"shape": list(mask.shape), "counts": counts}


def box_to_mask(box: Sequence[int] | None, shape: tuple[int, int]) -> np.ndarray | None:
    if not box or len(box) != 4:
        return None
    h, w = shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def mask_summary(mask: np.ndarray | None) -> dict[str, Any]:
    if mask is None:
        return {"area": 0, "bbox": None, "centroid": None}
    return {
        "area": int(mask.sum()),
        "bbox": bbox_from_mask(mask),
        "centroid": centroid_from_mask(mask),
    }


class M20Sam3Adapter:
    """Three-layer SAM3 adapter.

    The adapter has two operating modes:

    1. ``predictor`` mode, using a live SAM3/SAM3.1 video predictor with the
       official ``handle_request(add_prompt)`` API.
    2. ``replay`` mode, using precomputed candidate roots such as
       ``pred_sam31_b101``.  This keeps local validation possible when the CUDA
       b101 SAM3 runtime is unavailable.
    """

    def __init__(
        self,
        *,
        workspace: Path,
        predictor: Any | None = None,
        replay_roots: Mapping[str, Path] | None = None,
        confidence_threshold: float = 0.35,
    ) -> None:
        self.workspace = Path(workspace)
        self.predictor = predictor
        self.replay_roots = {str(k): Path(v) for k, v in (replay_roots or {}).items()}
        self.confidence_threshold = float(confidence_threshold)
        self.audit: list[dict[str, Any]] = []

    @classmethod
    def build_live_sam31(
        cls,
        *,
        workspace: Path,
        sam3_root: Path | None = None,
        checkpoint_path: Path | None = None,
        max_num_objects: int = 16,
        multiplex_count: int = 16,
        use_fa3: bool = False,
        compile_model: bool = False,
        async_loading_frames: bool = False,
        confidence_threshold: float = 0.35,
        replay_roots: Mapping[str, Path] | None = None,
    ) -> "M20Sam3Adapter":
        import sys

        workspace = Path(workspace).resolve()
        sam3_root = (sam3_root or workspace / "5_19" / "sam3").resolve()
        checkpoint_path = (
            checkpoint_path
            or workspace / "5_19" / "data" / "sam3.1" / "sam3.1_multiplex.pt"
        ).resolve()
        if str(sam3_root) not in sys.path:
            sys.path.insert(0, str(sam3_root))
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_path),
            max_num_objects=max_num_objects,
            multiplex_count=multiplex_count,
            use_fa3=use_fa3,
            compile=compile_model,
            warm_up=False,
            async_loading_frames=async_loading_frames,
            default_output_prob_thresh=confidence_threshold,
        )
        # Expose memory-selection knobs without requiring an upstream patch.
        model = getattr(predictor, "model", None)
        tracker = getattr(model, "tracker", None)
        if tracker is not None:
            raw_tracker = getattr(tracker, "model", None)
            if raw_tracker is not None and hasattr(raw_tracker, "use_memory_selection"):
                raw_tracker.use_memory_selection = bool(getattr(raw_tracker, "use_memory_selection", False))
        return cls(
            workspace=workspace,
            predictor=predictor,
            replay_roots=replay_roots,
            confidence_threshold=confidence_threshold,
        )

    def sam3_image_propose(
        self,
        *,
        video: str,
        frame_idx: int,
        prompt_pack: PromptPack,
        prompt_variant: dict[str, Any] | None = None,
        frame_name: str | None = None,
        max_candidates: int = 12,
    ) -> list[Sam3Candidate]:
        """Generate or replay later-frame SAM3 proposals."""
        if self.predictor is None:
            return self._replay_image_propose(
                video=video,
                obj_id=prompt_pack.obj_id,
                frame_idx=frame_idx,
                frame_name=frame_name,
                prompt_pack=prompt_pack,
                max_candidates=max_candidates,
            )
        if prompt_variant is not None:
            return self._live_image_propose(
                video=video,
                frame_idx=frame_idx,
                prompt_pack=prompt_pack,
                prompt_variant=prompt_variant,
                max_candidates=max_candidates,
            )

        # Prompt ablation is part of the SAM3 value proposition.  Run variants
        # independently and deduplicate near-identical masks by IoU so downstream
        # scoring can decide whether text, semantic text, or frame-local visual
        # prompts actually helped.
        merged: list[Sam3Candidate] = []
        for variant in prompt_pack.prompt_variants():
            for cand in self._live_image_propose(
                video=video,
                frame_idx=frame_idx,
                prompt_pack=prompt_pack,
                prompt_variant=variant,
                max_candidates=max_candidates,
            ):
                if cand.mask is None:
                    continue
                duplicate = False
                for prev in merged:
                    if prev.mask is not None and mask_iou(prev.mask, cand.mask) >= 0.95:
                        duplicate = True
                        break
                if not duplicate:
                    merged.append(cand)
                if len(merged) >= max_candidates:
                    return merged
        return merged

    def sam3_video_branch(
        self,
        *,
        video: str,
        obj_id: int,
        anchor_masks: Mapping[int, np.ndarray],
        branch_name: str = "sam3_gt_branch",
        window: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Return a bounded SAM3 branch audit.

        Live multi-anchor bounded propagation is deliberately not hidden behind
        a fake abstraction here.  In replay mode this describes the precomputed
        branch window; live callers should run the existing SAM3.1 adapter or a
        specialized branch script and then pass that root into fusion.
        """
        rec = {
            "event": "video_branch",
            "video": video,
            "obj_id": obj_id,
            "branch_name": branch_name,
            "anchor_frames": sorted(int(k) for k in anchor_masks),
            "window": list(window) if window else None,
            "mode": "live_unimplemented" if self.predictor is not None else "replay_audit",
            "rollback_policy": [
                "wrong_same_class_drift",
                "stable_frame_regression",
                "large_disagreement_without_verifier_support",
            ],
        }
        self.audit.append(rec)
        return rec

    def sam3_audit_export(self) -> dict[str, Any]:
        return {
            "adapter": "M20Sam3Adapter",
            "mode": "live" if self.predictor is not None else "replay",
            "replay_roots": {k: str(v) for k, v in self.replay_roots.items()},
            "confidence_threshold": self.confidence_threshold,
            "events": self.audit,
        }

    def _replay_image_propose(
        self,
        *,
        video: str,
        obj_id: int,
        frame_idx: int,
        frame_name: str | None,
        prompt_pack: PromptPack,
        max_candidates: int,
    ) -> list[Sam3Candidate]:
        from PIL import Image

        stem = frame_name or f"{frame_idx:05d}"
        out: list[Sam3Candidate] = []
        for root_name, root in self.replay_roots.items():
            path = root / video / f"{stem}.png"
            if not path.is_file():
                self.audit.append(
                    {
                        "event": "image_propose_replay_missing",
                        "video": video,
                        "frame_idx": frame_idx,
                        "root": root_name,
                        "path": str(path),
                    }
                )
                continue
            arr = np.asarray(Image.open(path))
            if arr.ndim != 2:
                arr = arr[..., 0]
            mask = arr == int(obj_id)
            if int(mask.sum()) <= 0:
                self.audit.append(
                    {
                        "event": "image_propose_replay_empty",
                        "video": video,
                        "obj_id": obj_id,
                        "frame_idx": frame_idx,
                        "root": root_name,
                        "prompt_pack": prompt_pack.key,
                    }
                )
                continue
            cand = Sam3Candidate(
                candidate_id=f"{video}:obj{obj_id}:{frame_idx}:{root_name}:object_id",
                video=video,
                obj_id=obj_id,
                frame_idx=frame_idx,
                mask=mask,
                bbox=bbox_from_mask(mask),
                score=None,
                prompt_source=f"replay:{root_name}",
                text_prompt=prompt_pack.category_prompt,
                visual_prompt=[asdict(v) for v in prompt_pack.positive_visual[:1]],
                source_root=root_name,
                mask_kind="object_id",
                audit={
                    "replay_path": str(path),
                    "story_constraints": prompt_pack.story_constraints,
                },
            )
            out.append(cand)
        out.sort(key=lambda c: c.area, reverse=True)
        self.audit.append(
            {
                "event": "image_propose_replay",
                "video": video,
                "obj_id": obj_id,
                "frame_idx": frame_idx,
                "candidate_count": len(out),
                "roots": sorted(self.replay_roots),
            }
        )
        return out[:max_candidates]

    def _live_image_propose(
        self,
        *,
        video: str,
        frame_idx: int,
        prompt_pack: PromptPack,
        prompt_variant: dict[str, Any] | None,
        max_candidates: int,
    ) -> list[Sam3Candidate]:
        import torch

        video_dir = self.workspace / "homework" / "JPEGImages" / video
        variant = prompt_variant or prompt_pack.prompt_variants()[0]
        request_id = str(uuid.uuid4())
        started = time.time()
        session_id = self._start_live_session(video_dir, request_id)
        try:
            frame_path = video_dir / f"{frame_idx:05d}.jpg"
            if not frame_path.is_file():
                frame_path = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.png")])[
                    frame_idx
                ]
            from PIL import Image

            with Image.open(frame_path) as img:
                width, height = img.size
            boxes: list[list[float]] = []
            labels: list[int] = []
            for vp in variant.get("positive_visual", []):
                visual = VisualPrompt(**vp)
                if int(visual.frame_idx) == int(frame_idx):
                    boxes.append(visual.to_normalized_xywh(width, height))
                    labels.append(1)
            for vp in variant.get("negative_visual", []):
                visual = VisualPrompt(**vp)
                if int(visual.frame_idx) == int(frame_idx):
                    boxes.append(visual.to_normalized_xywh(width, height))
                    labels.append(0)
            req: dict[str, Any] = {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": int(frame_idx),
                "text": variant.get("text_prompt") or prompt_pack.category_prompt,
                "output_prob_thresh": self.confidence_threshold,
            }
            if boxes:
                req["bounding_boxes"] = boxes
                req["bounding_box_labels"] = labels
            response = self.predictor.handle_request(req)
            outputs = response["outputs"]
            out_masks = np.asarray(outputs.get("out_binary_masks", np.zeros((0, height, width), dtype=bool)), dtype=bool)
            out_scores = outputs.get("out_probs", None)
            scores = (
                np.asarray(out_scores).reshape(-1).astype(float).tolist()
                if out_scores is not None
                else [None] * len(out_masks)
            )
            candidates: list[Sam3Candidate] = []
            for idx, mask in enumerate(out_masks[:max_candidates]):
                candidates.append(
                    Sam3Candidate(
                        candidate_id=(
                            f"{video}:obj{prompt_pack.obj_id}:{frame_idx}:"
                            f"sam3_live:{variant.get('name','prompt')}:{idx}"
                        ),
                        video=video,
                        obj_id=prompt_pack.obj_id,
                        frame_idx=frame_idx,
                        mask=mask,
                        bbox=bbox_from_mask(mask),
                        score=None if idx >= len(scores) else scores[idx],
                        prompt_source=f"live:{variant.get('name','prompt')}",
                        text_prompt=req["text"],
                        visual_prompt=[
                            {"box": b, "label": l} for b, l in zip(boxes, labels)
                        ],
                        source_root=None,
                        mask_kind=f"sam3_live:{idx}",
                        audit={"request": {k: str(v) for k, v in req.items() if k != "session_id"}},
                    )
                )
            self.audit.append(
                {
                    "event": "image_propose_live",
                    "video": video,
                    "obj_id": prompt_pack.obj_id,
                    "frame_idx": frame_idx,
                    "variant": variant.get("name"),
                    "candidate_count": len(candidates),
                    "seconds": round(time.time() - started, 3),
                }
            )
            return candidates
        finally:
            self.predictor.handle_request({"type": "close_session", "session_id": session_id})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _start_live_session(self, video_dir: Path, session_id: str) -> str:
        """Start a SAM3.1 session without forwarding unsupported kwargs.

        The public predictor forwards ``offload_state_to_cpu`` into current
        SAM3.1 multiplex internals, which is unsupported in this local codebase.
        The existing SAM31 adapters use the same direct-session workaround.
        """
        init_kwargs = {
            "resource_path": str(video_dir),
            "offload_video_to_cpu": True,
        }
        if hasattr(self.predictor, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.predictor.async_loading_frames
        state = self.predictor.model.init_state(**init_kwargs)
        self.predictor._all_inference_states[session_id] = {
            "state": state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        return session_id


def event_state_for_frame(ledger: Mapping[str, Any], frame_idx: int) -> tuple[str, str]:
    for entry in ledger.get("event_story", []):
        frames = entry.get("frames")
        if isinstance(frames, list) and len(frames) == 2:
            if int(frames[0]) <= int(frame_idx) <= int(frames[1]):
                return str(entry.get("state", "")), str(entry.get("allowed_output", "review_only"))
    return "", "review_only"


def recovery_frames_from_ledger(
    ledger: Mapping[str, Any], frame_count: int, *, max_frames: int | None = None
) -> list[int]:
    wanted: list[int] = []
    for entry in ledger.get("event_story", []):
        allowed = str(entry.get("allowed_output", "review_only"))
        if allowed == "keep_current":
            continue
        frames = entry.get("frames")
        if isinstance(frames, list) and len(frames) == 2:
            wanted.extend(range(max(1, int(frames[0])), min(frame_count - 1, int(frames[1])) + 1))
    for win in ledger.get("next_review_windows", []):
        frames = win.get("frames")
        if isinstance(frames, list):
            if len(frames) == 2 and all(isinstance(x, int) for x in frames):
                wanted.extend(range(max(1, frames[0]), min(frame_count - 1, frames[1]) + 1))
            else:
                wanted.extend(int(x) for x in frames if isinstance(x, int))
    unique = sorted({idx for idx in wanted if 1 <= idx < frame_count})
    if max_frames is None or len(unique) <= max_frames:
        return unique
    picks = np.linspace(0, len(unique) - 1, max_frames).round().astype(int).tolist()
    return sorted({unique[i] for i in picks})


def iou_with_sources(
    candidate_mask: np.ndarray,
    *,
    frame_path_stem: str,
    video: str,
    obj_id: int,
    roots: Mapping[str, Path],
    exclude: Iterable[str] = (),
) -> dict[str, float]:
    from PIL import Image

    excluded = set(exclude)
    out: dict[str, float] = {}
    for name, root in roots.items():
        if name in excluded:
            continue
        p = root / video / f"{frame_path_stem}.png"
        if not p.is_file():
            continue
        arr = np.asarray(Image.open(p))
        if arr.ndim != 2:
            arr = arr[..., 0]
        if arr.shape != candidate_mask.shape:
            continue
        out[name] = mask_iou(candidate_mask, arr == int(obj_id))
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "M20Sam3Adapter",
    "PromptPack",
    "Sam3Candidate",
    "VisualPrompt",
    "box_to_mask",
    "event_state_for_frame",
    "iou_with_sources",
    "mask_summary",
    "recovery_frames_from_ledger",
    "write_json",
]

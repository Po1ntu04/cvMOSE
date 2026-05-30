#!/usr/bin/env python3
"""Run SAM2 with a Reappearance-Aware ReAnchor (RAR) first-stage controller.

This is the thin M5R entrypoint requested by the RAR plan.  It intentionally
starts with RCMS-lite + state machine + delayed memory commit, not SAM3/MLLM
retrieval.  Current-frame predictions are still yielded for inspection, but
ambiguous/recovery frames are not written into the main SAM2 non-conditioning
memory.  When recovery is detected, high-quality pre-disappearance memories are
promoted into conditioned memory for future frames.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.reanchor import (  # noqa: E402
    COMMIT_WRITE_MAIN,
    AnchorBank,
    AnchorRecord,
    CommitPolicy,
    FrameAudit,
    MaskStats,
    QualitySignals,
    Selector,
    StateMachine,
    StateMachineConfig,
)
from infer_mosev2_sam2 import (  # noqa: E402
    import_sam2,
    list_frames,
    load_first_annotation,
    logits_to_label,
    make_submission,
    save_label_png,
)


@dataclass(slots=True)
class RARConfig:
    mode: str = "state"  # rcms | state
    output_policy: str = "provisional"  # provisional | empty_in_recovery
    mask_thr: float = 0.50
    perturb: float = 0.05
    stable_quality: float = 0.60
    ambiguous_quality: float = 0.35
    rcms_quality_thr: float = 0.60
    rcms_max_anchors: int = 4
    reservoir_size: int = 12
    min_area_pixels: float = 3.0
    max_area_ratio: float = 4.0
    min_area_ratio: float = 0.20
    max_motion_floor: float = 40.0
    max_motion_scale: float = 3.0
    empty_streak_to_recovery: int = 1
    ambiguous_streak_to_recovery: int = 3
    confirm_frames: int = 2
    keep_first_cond_frame: bool = True
    remove_promoted_from_noncond: bool = True


def parse_args() -> argparse.Namespace:
    default_workspace = REPO_ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=default_workspace)
    p.add_argument("--sam2-root", type=Path, default=None)
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--pred-root", type=Path, default=None)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--offload-video-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-video-to-cpu", dest="offload_video_to_cpu", action="store_false")
    p.add_argument("--offload-state-to-cpu", action="store_true", default=True)
    p.add_argument("--no-offload-state-to-cpu", dest="offload_state_to_cpu", action="store_false")
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--overwrite-submission", action="store_true")

    p.add_argument("--rar-mode", choices=["rcms", "state"], default="state")
    p.add_argument(
        "--rar-output-policy",
        choices=["provisional", "empty_in_recovery"],
        default="provisional",
    )
    p.add_argument("--rar-stable-quality", type=float, default=0.60)
    p.add_argument("--rar-ambiguous-quality", type=float, default=0.35)
    p.add_argument("--rar-rcms-quality-thr", type=float, default=0.60)
    p.add_argument("--rar-rcms-max-anchors", type=int, default=4)
    p.add_argument("--rar-reservoir-size", type=int, default=12)
    p.add_argument("--rar-confirm-frames", type=int, default=2)
    p.add_argument("--rar-audit-json", type=Path, default=None)
    p.add_argument("--rar-audit-dir", type=Path, default=None)
    return p.parse_args()


def complete_paths(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.sam2_root = (args.sam2_root or ws / "5_19" / "sam2").resolve()
    args.checkpoint = (
        args.checkpoint or ws / "5_19" / "data" / "sam2" / "sam2.1_hiera_base_plus.pt"
    ).resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.pred_root = (args.pred_root or ws / "homework" / "pred_sam2_rar").resolve()
    args.submit_root = (args.submit_root or ws / "homework" / "submission_433_rar").resolve()
    args.zip_path = (args.zip_path or ws / "homework" / "submission_mosev2_rar.zip").resolve()
    args.rar_audit_json = (
        args.rar_audit_json or ws / "homework" / "logs" / "rar_latest.json"
    ).resolve()
    args.rar_audit_dir = (args.rar_audit_dir or ws / "homework" / "logs" / "rar_by_video").resolve()
    return args


def config_from_args(args: argparse.Namespace) -> RARConfig:
    return RARConfig(
        mode=args.rar_mode,
        output_policy=args.rar_output_policy,
        stable_quality=float(args.rar_stable_quality),
        ambiguous_quality=float(args.rar_ambiguous_quality),
        rcms_quality_thr=float(args.rar_rcms_quality_thr),
        rcms_max_anchors=int(args.rar_rcms_max_anchors),
        reservoir_size=int(args.rar_reservoir_size),
        confirm_frames=int(args.rar_confirm_frames),
    )


def _as_2d_logits(mask_logits: Any):
    logits = mask_logits.detach()
    if logits.ndim == 4:
        logits = logits[0, 0]
    elif logits.ndim == 3:
        logits = logits[0]
    elif logits.ndim != 2:
        raise ValueError(f"Expected 2D/3D/4D mask logits, got {tuple(logits.shape)}")
    return logits.float()


def _tensor_iou(a: Any, b: Any) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return float(inter / union) if union else 1.0


def _objectness(object_score_logits: Any | None) -> float | None:
    if object_score_logits is None:
        return None
    try:
        import torch

        if isinstance(object_score_logits, torch.Tensor):
            if object_score_logits.numel() == 0:
                return None
            return float(torch.sigmoid(object_score_logits.detach().float()).mean().item())
    except Exception:
        return None
    try:
        return float(object_score_logits)
    except Exception:
        return None


def _mask_stats(mask: Any, *, edge_margin_frac: float = 0.015) -> MaskStats:
    nz = mask.nonzero(as_tuple=False)
    h, w = int(mask.shape[-2]), int(mask.shape[-1])
    area = float(mask.sum().item())
    if nz.numel() == 0:
        return MaskStats(area_pixels=0.0, area_frac=0.0, bbox=None, centroid=None, edge_touch=False)
    ys = nz[:, 0].float()
    xs = nz[:, 1].float()
    bbox = [
        int(xs.min().item()),
        int(ys.min().item()),
        int(xs.max().item()) + 1,
        int(ys.max().item()) + 1,
    ]
    centroid = [float(xs.mean().item()), float(ys.mean().item())]
    mx = max(1, int(round(w * edge_margin_frac)))
    my = max(1, int(round(h * edge_margin_frac)))
    edge = bool(bbox[0] < mx or bbox[1] < my or bbox[2] >= w - mx or bbox[3] >= h - my)
    return MaskStats(
        area_pixels=area,
        area_frac=area / max(float(h * w), 1.0),
        bbox=bbox,
        centroid=centroid,
        edge_touch=edge,
    )


def _dist(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _quality_from_logits(
    mask_logits: Any,
    prev_stats: MaskStats | None,
    object_score_logits: Any | None,
    cfg: RARConfig,
) -> tuple[Any, MaskStats, QualitySignals]:
    import torch

    logits = _as_2d_logits(mask_logits)
    prob = torch.sigmoid(logits)
    mask = prob > cfg.mask_thr
    stats = _mask_stats(mask)
    mask_lo = prob > max(0.0, cfg.mask_thr - cfg.perturb)
    mask_hi = prob > min(1.0, cfg.mask_thr + cfg.perturb)
    stability = _tensor_iou(mask_lo, mask_hi)
    objectness = _objectness(object_score_logits)

    area_ratio = None
    displacement = None
    reasons: list[str] = []
    if prev_stats is not None and prev_stats.area_pixels > 0:
        area_ratio = stats.area_pixels / max(prev_stats.area_pixels, 1e-6)
        if area_ratio > cfg.max_area_ratio or area_ratio < cfg.min_area_ratio:
            reasons.append("area_ratio_jump")
        displacement = _dist(stats.centroid, prev_stats.centroid)
        max_motion = max(
            cfg.max_motion_floor,
            cfg.max_motion_scale * math.sqrt(max(prev_stats.area_pixels, 1.0)),
        )
        if displacement is not None and displacement > max_motion and not stats.edge_touch:
            reasons.append("large_motion")
    if stats.area_pixels < cfg.min_area_pixels:
        reasons.append("empty_or_tiny")
    if objectness is not None and objectness < 0.50:
        reasons.append("low_objectness")
    if stability < cfg.ambiguous_quality:
        reasons.append("unstable_mask")

    obj_component = 1.0 if objectness is None else max(0.0, min(1.0, objectness))
    area_component = 1.0
    if stats.area_pixels < cfg.min_area_pixels:
        area_component = 0.0
    elif area_ratio is not None and (
        area_ratio > cfg.max_area_ratio or area_ratio < cfg.min_area_ratio
    ):
        area_component = 0.35 if stats.edge_touch else 0.15
    motion_component = 1.0 if "large_motion" not in reasons else 0.25
    quality = float(obj_component * stability * area_component * motion_component)
    signals = QualitySignals(
        objectness=objectness,
        stability=stability,
        area_ratio=area_ratio,
        displacement_px=displacement,
        quality=quality,
        empty=stats.area_pixels < cfg.min_area_pixels,
        reasons=reasons,
    )
    return mask.detach(), stats, signals


def _clone_anchor_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # Shallow-copy the output dict.  Tensors are intentionally shared with the
    # already stored SAM2 output; they are immutable during inference and this
    # avoids duplicating large memory features.
    return dict(payload)


def install_rar_controller(predictor: Any, cfg: RARConfig, audit_by_video: dict[str, Any]) -> None:
    """Monkey-patch SAM2VideoPredictor.propagate_in_video with RAR control."""
    import torch
    from tqdm import tqdm

    predictor._rar_cfg = cfg
    predictor._rar_audit = audit_by_video
    predictor._rar_current_video = None

    @torch.inference_mode()
    def propagate_in_video_rar(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        self.propagate_in_video_preflight(inference_state)
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        if start_frame_idx is None:
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            processing_order = (
                range(start_frame_idx, end_frame_idx - 1, -1)
                if start_frame_idx > 0
                else []
            )
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        sm_cfg = StateMachineConfig(
            stable_quality=cfg.stable_quality,
            ambiguous_quality=cfg.ambiguous_quality,
            min_stable_area_pixels=cfg.min_area_pixels,
            empty_streak_to_recovery=cfg.empty_streak_to_recovery,
            ambiguous_streak_to_recovery=cfg.ambiguous_streak_to_recovery,
        )
        banks = {
            idx: AnchorBank(
                obj_id=int(obj_ids[idx]),
                max_pre_disappearance=cfg.reservoir_size,
                min_quality=cfg.rcms_quality_thr,
            )
            for idx in range(batch_size)
        }
        machines = {idx: StateMachine(sm_cfg) for idx in range(batch_size)}
        commit_policies = {
            idx: CommitPolicy(
                confirm_frames=cfg.confirm_frames,
                min_candidate_score=cfg.stable_quality,
            )
            for idx in range(batch_size)
        }
        selectors = {idx: Selector() for idx in range(batch_size)}
        prev_stats: dict[int, MaskStats | None] = {idx: None for idx in range(batch_size)}
        rcms_promoted_once: set[int] = set()
        video_name = getattr(self, "_rar_current_video", "unknown") or "unknown"

        for frame_idx in tqdm(processing_order, desc="propagate in video (rar)"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                obj_id = int(obj_ids[obj_idx])
                notes: list[str] = []
                rcms_selected: list[int] = []
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    _, stats, signals = _quality_from_logits(
                        pred_masks,
                        prev_stats[obj_idx],
                        current_out.get("object_score_logits"),
                        cfg,
                    )
                    machines[obj_idx].state = (
                        machines[obj_idx].state
                        if frame_idx != start_frame_idx
                        else "stable"
                    )
                    state = machines[obj_idx].state
                    commit_decision = COMMIT_WRITE_MAIN
                    if frame_idx == start_frame_idx and banks[obj_idx].init_anchor is None:
                        banks[obj_idx].set_init_anchor(
                            AnchorRecord(
                                frame_idx=int(frame_idx),
                                obj_id=obj_id,
                                stats=stats,
                                quality=1.0,
                                source="init",
                                payload=_clone_anchor_payload(current_out),
                                selected_as_conditioned=True,
                            )
                        )
                    prev_stats[obj_idx] = stats if stats.present else prev_stats[obj_idx]
                else:
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    _, stats, signals = _quality_from_logits(
                        pred_masks, prev_stats[obj_idx], current_out.get("object_score_logits"), cfg
                    )
                    state = machines[obj_idx].update(stats, signals)
                    if state == "stable":
                        rcms_promoted_once.discard(obj_idx)
                    commit_decision = commit_policies[obj_idx].current_decision(state, signals)
                    selector_decision = selectors[obj_idx].decide(state, signals)
                    notes.append(f"selector:{selector_decision.action}:{selector_decision.reason}")

                    if cfg.mode == "rcms":
                        # Ablation A: add RCMS anchors on disappearance but keep SAM2's
                        # normal non-conditioning memory path otherwise.
                        write_main = True
                    else:
                        write_main = commit_decision == COMMIT_WRITE_MAIN

                    if write_main:
                        obj_output_dict[storage_key][frame_idx] = current_out
                        prev_stats[obj_idx] = stats if stats.present else prev_stats[obj_idx]
                        if state == "stable" and signals.quality >= cfg.rcms_quality_thr:
                            banks[obj_idx].add_pre_disappearance(
                                AnchorRecord(
                                    frame_idx=int(frame_idx),
                                    obj_id=obj_id,
                                    stats=stats,
                                    quality=signals.quality,
                                    source="pre_disappearance",
                                    payload=_clone_anchor_payload(current_out),
                                )
                            )
                    else:
                        notes.append("main_memory_write_blocked_delayed_commit")
                        if stats.present and state == "ambiguous":
                            # Keep geometry for the next sanity check, but do not let SAM2 read
                            # this mask as memory in future frames.
                            prev_stats[obj_idx] = stats

                    if state == "recovery" and obj_idx not in rcms_promoted_once:
                        selected = banks[obj_idx].select_rcms(
                            current_frame=int(frame_idx),
                            max_anchors=cfg.rcms_max_anchors,
                            min_quality=cfg.rcms_quality_thr,
                        )
                        for anchor in selected:
                            obj_output_dict["cond_frame_outputs"][
                                anchor.frame_idx
                            ] = _clone_anchor_payload(anchor.payload)
                            if cfg.remove_promoted_from_noncond:
                                obj_output_dict["non_cond_frame_outputs"].pop(
                                    anchor.frame_idx,
                                    None,
                                )
                            rcms_selected.append(anchor.frame_idx)
                        if selected:
                            notes.append(f"rcms_promoted:{rcms_selected}")
                            rcms_promoted_once.add(obj_idx)

                    if (
                        cfg.output_policy == "empty_in_recovery"
                        and state == "recovery"
                        and signals.empty
                    ):
                        pred_masks = torch.full_like(pred_masks, -1024.0)
                        commit_decision = "output_empty_recovery"

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}
                pred_masks_per_obj[obj_idx] = pred_masks
                used_cond = sorted(int(k) for k in obj_output_dict["cond_frame_outputs"].keys())
                frame_audit = FrameAudit(
                    video=video_name,
                    obj_id=obj_id,
                    frame_idx=int(frame_idx),
                    state=state,
                    base_mask_stats=stats,
                    quality_signals=signals,
                    rcms_selected=rcms_selected,
                    candidate_count=0,
                    best_candidate_score=None,
                    commit_decision=commit_decision,
                    used_cond_frames=used_cond,
                    output_policy=cfg.output_policy,
                    notes=notes,
                )
                audit_by_video.setdefault(video_name, {}).setdefault("objects", {}).setdefault(
                    str(obj_id),
                    {"frames": []},
                )["frames"].append(frame_audit.to_json())

            all_pred_masks = (
                torch.cat(pred_masks_per_obj, dim=0)
                if len(pred_masks_per_obj) > 1
                else pred_masks_per_obj[0]
            )
            _, video_res_masks = self._get_orig_video_res_output(inference_state, all_pred_masks)
            yield frame_idx, obj_ids, video_res_masks

        audit_by_video.setdefault(video_name, {})["anchor_banks"] = {
            str(int(obj_ids[idx])): banks[idx].to_json() for idx in range(batch_size)
        }

    predictor.propagate_in_video = types.MethodType(propagate_in_video_rar, predictor)


def _empty_video_audit(video: str, obj_ids: list[int], cfg: RARConfig) -> dict[str, Any]:
    return {
        "video": video,
        "method": "rar_rcms_lite_state_machine",
        "config": asdict(cfg),
        "objects": {str(int(obj_id)): {"frames": []} for obj_id in obj_ids},
        "anchor_banks": {},
        "summary": {},
    }


def _finalize_video_audit(audit: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "frames": 0,
        "state_counts": {},
        "commit_counts": {},
        "rcms_promotions": 0,
    }
    for obj in audit.get("objects", {}).values():
        for rec in obj.get("frames", []):
            summary["frames"] += 1
            state = str(rec.get("state", "unknown"))
            commit = str(rec.get("commit_decision", "unknown"))
            summary["state_counts"][state] = summary["state_counts"].get(state, 0) + 1
            summary["commit_counts"][commit] = summary["commit_counts"].get(commit, 0) + 1
            summary["rcms_promotions"] += len(rec.get("rcms_selected", []) or [])
    summary["state_counts"] = dict(sorted(summary["state_counts"].items()))
    summary["commit_counts"] = dict(sorted(summary["commit_counts"].items()))
    audit["summary"] = summary
    return audit


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_provenance(args: argparse.Namespace) -> dict[str, Any]:
    def run_git(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                cmd,
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    return {
        "repo_root": str(REPO_ROOT),
        "git_sha": os.environ.get("CVMOSE_GIT_SHA") or run_git(["git", "rev-parse", "HEAD"]),
        "git_branch": os.environ.get("CVMOSE_GIT_BRANCH")
        or run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "script": str(Path(__file__).resolve()),
        "workspace": str(args.workspace),
        "sam2_root": str(args.sam2_root),
        "model_cfg": args.model_cfg,
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "pred_root": str(args.pred_root),
        "audit_json": str(args.rar_audit_json),
    }


def run_video(
    predictor: Any,
    args: argparse.Namespace,
    cfg: RARConfig,
    video_name: str,
) -> dict[str, Any]:
    import torch

    video_dir = args.jpeg_root / video_name
    ann_dir = args.ann_root / video_name
    out_dir = args.pred_root / video_name
    frames = list_frames(video_dir)
    ann, palette = load_first_annotation(ann_dir)
    obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
    if not obj_ids:
        raise ValueError(f"No foreground object ids in {ann_dir / '00000.png'}")

    if args.skip_existing and len(sorted(out_dir.glob("*.png"))) == len(frames):
        return {"video": video_name, "status": "skipped", "frames": len(frames), "objects": obj_ids}

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    predictor._rar_current_video = video_name
    predictor._rar_audit[video_name] = _empty_video_audit(video_name, obj_ids, cfg)

    state = predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
    )
    predictor.reset_state(state)

    with torch.inference_mode():
        for obj_id in obj_ids:
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                mask=(ann == obj_id),
            )

    seen: set[int] = set()
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_ctx:
        for frame_idx, cur_obj_ids, mask_logits in predictor.propagate_in_video(state):
            if frame_idx < 0 or frame_idx >= len(frames):
                raise IndexError(f"SAM2 returned frame_idx={frame_idx} outside 0..{len(frames)-1}")
            if frame_idx == 0:
                label = ann.copy()
            else:
                label = logits_to_label(mask_logits, [int(x) for x in cur_obj_ids])
            save_label_png(out_dir / f"{frames[frame_idx].stem}.png", label, palette)
            seen.add(int(frame_idx))

    missing = [i for i in range(len(frames)) if i not in seen]
    if missing:
        raise RuntimeError(
            f"{video_name}: missing predicted frames {missing[:20]} total={len(missing)}"
        )

    outputs = sorted(out_dir.glob("*.png"))
    expected_names = [f"{f.stem}.png" for f in frames]
    if [f.name for f in outputs] != expected_names:
        raise RuntimeError(f"{video_name}: output names mismatch")
    for sample_idx in [0, len(outputs) - 1]:
        if Image.open(outputs[sample_idx]).size != Image.open(frames[sample_idx]).size:
            raise RuntimeError(f"{video_name}: size mismatch at {outputs[sample_idx]}")

    audit = _finalize_video_audit(predictor._rar_audit[video_name])
    args.rar_audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.rar_audit_dir / f"{video_name}.json", audit)

    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result: dict[str, Any] = {
        "video": video_name,
        "status": "done",
        "frames": len(frames),
        "objects": obj_ids,
    }
    result.update({f"rar_{k}": v for k, v in audit.get("summary", {}).items()})
    return result


def aggregate_audit(
    audit_by_video: dict[str, Any],
    cfg: RARConfig,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    state_counts: dict[str, int] = {}
    commit_counts: dict[str, int] = {}
    rcms_promotions = 0
    videos = {}
    for video, audit in sorted(audit_by_video.items()):
        audit = _finalize_video_audit(audit)
        videos[video] = audit
        summary = audit.get("summary", {})
        rcms_promotions += int(summary.get("rcms_promotions", 0))
        for key, value in summary.get("state_counts", {}).items():
            state_counts[key] = state_counts.get(key, 0) + int(value)
        for key, value in summary.get("commit_counts", {}).items():
            commit_counts[key] = commit_counts.get(key, 0) + int(value)
    return {
        "method": "rar_rcms_lite_state_machine",
        "principle": (
            "SAM2 stable path plus pre-disappearance conditioned reservoir "
            "and delayed memory commit"
        ),
        "config": asdict(cfg),
        "results": results,
        "summary": {
            "videos": len(videos),
            "state_counts": dict(sorted(state_counts.items())),
            "commit_counts": dict(sorted(commit_counts.items())),
            "rcms_promotions": rcms_promotions,
        },
        "videos": videos,
    }


def main() -> None:
    args = complete_paths(parse_args())
    cfg = config_from_args(args)
    for required in [args.sam2_root, args.checkpoint, args.jpeg_root, args.ann_root]:
        if not required.exists():
            raise FileNotFoundError(required)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    build_sam2_video_predictor = import_sam2(args.sam2_root)
    print(f"workspace={args.workspace}")
    print(f"sam2_root={args.sam2_root}")
    print(f"checkpoint={args.checkpoint}")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        f"device={args.device} cuda_available={torch.cuda.is_available()} "
        f"gpus={gpu_count}"
    )
    print(f"rar_config={json.dumps(asdict(cfg), ensure_ascii=False, sort_keys=True)}", flush=True)

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    predictor = build_sam2_video_predictor(args.model_cfg, str(args.checkpoint), device=args.device)
    predictor.eval()
    audit_by_video: dict[str, Any] = {}
    install_rar_controller(predictor, cfg, audit_by_video)

    videos = args.videos or sorted(p.name for p in args.jpeg_root.iterdir() if p.is_dir())
    started = time.time()
    results: list[dict[str, Any]] = []
    print(f"videos={len(videos)} {videos}", flush=True)
    for idx, video_name in enumerate(videos, 1):
        t0 = time.time()
        print(f"[{idx}/{len(videos)}] {video_name} start", flush=True)
        result = run_video(predictor, args, cfg, video_name)
        result["seconds"] = round(time.time() - t0, 2)
        results.append(result)
        print(f"[{idx}/{len(videos)}] {video_name} {result}", flush=True)

    total_frames = sum(int(r["frames"]) for r in results if r["status"] in {"done", "skipped"})
    elapsed = time.time() - started
    aggregate = aggregate_audit(audit_by_video, cfg, results)
    aggregate["provenance"] = collect_provenance(args)
    aggregate["runtime"] = {
        "seconds": elapsed,
        "frames": total_frames,
        "fps": (total_frames / elapsed) if elapsed else None,
    }
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        aggregate["runtime"].update(
            {
                "cuda_max_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "cuda_max_memory_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            }
        )
    write_json(args.rar_audit_json, aggregate)
    print(
        f"inference_complete videos={len(results)} frames={total_frames} seconds={elapsed:.2f} "
        f"rar_summary={json.dumps(aggregate['summary'], ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )
    print(f"rar_audit_json={args.rar_audit_json}", flush=True)
    print(f"rar_audit_dir={args.rar_audit_dir}", flush=True)

    if args.make_submission:
        make_submission(args)


if __name__ == "__main__":
    main()

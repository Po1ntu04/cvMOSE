#!/usr/bin/env python3
"""Convert M9 Qwen-VL recovery boxes into SAM2 box-prompt candidate roots.

This tool is deliberately a *candidate source* builder.  It copies a fallback
prediction root (usually M11) and replaces only the requested video/object/frame
with masks produced from Qwen-proposed boxes through SAM2 ImagePredictor.  The
resulting roots can then be audited by M8 candidate-pool/DINO/MLLM fusion.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class BoxCandidate:
    video: str
    obj_id: int
    frame_idx: int
    candidate_id: str
    bbox_norm: list[float]
    confidence: float
    reason_short: str
    risk_tags: list[str]
    model_used: str | None
    panel_path: str | None


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parents[1]
    default_ws = Path("/home/yu/projects/cv/from fdu/MOSEv2") if Path("/home/yu/projects/cv/from fdu/MOSEv2").exists() else here
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=default_ws)
    p.add_argument("--proposals-json", type=Path, required=True)
    p.add_argument("--fallback-root", type=Path, required=True, help="Root to copy before replacing proposal frames, usually M11")
    p.add_argument("--pred-root-template", type=Path, required=True, help="May include {rank}; otherwise _rankN is appended")
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--sam2-root", type=Path, default=None)
    p.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ranks", nargs="*", type=int, default=[1])
    p.add_argument("--min-confidence", type=float, default=0.01)
    p.add_argument("--box-pad-frac", type=float, default=0.03)
    p.add_argument("--box-pad-px", type=int, default=0)
    p.add_argument("--clip-to-prompt-box", action="store_true", help="Intersect SAM2 output with an expanded prompt box to suppress leakage/composites")
    p.add_argument("--clip-pad-frac", type=float, default=0.50, help="Extra prompt-box padding used only for clipping")
    p.add_argument("--fallback-rectangle-if-empty", action="store_true", help="If clipped SAM2 mask is empty, use the prompt rectangle as a low-precision candidate")
    p.add_argument("--multimask-output", action="store_true", default=True)
    p.add_argument("--singlemask-output", dest="multimask_output", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Copy fallback roots and audit proposals without loading SAM2")
    return p.parse_args()


def complete_paths(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    args.workspace = ws
    args.sam2_root = (args.sam2_root or ws / "5_19" / "sam2").resolve()
    args.checkpoint = (args.checkpoint or ws / "5_19" / "data" / "sam2" / "sam2.1_hiera_base_plus.pt").resolve()
    args.fallback_root = args.fallback_root.resolve()
    args.pred_root_template = args.pred_root_template.resolve()
    args.audit_json = args.audit_json.resolve()
    return args


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(video_dir)
    return frames


def load_label(path: Path) -> tuple[np.ndarray, list[int] | None]:
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr, img.getpalette()


def save_label(path: Path, arr: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr, mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def pixel_box(bbox_norm: list[float], width: int, height: int, pad_frac: float, pad_px: int) -> list[float]:
    x1 = float(bbox_norm[0]) / 1000.0 * width
    y1 = float(bbox_norm[1]) / 1000.0 * height
    x2 = float(bbox_norm[2]) / 1000.0 * width
    y2 = float(bbox_norm[3]) / 1000.0 * height
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = max(float(pad_px), max(bw, bh) * float(pad_frac))
    x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width - 1), x2), min(float(height - 1), y2)
    if x2 <= x1 + 1:
        x2 = min(float(width - 1), x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(float(height - 1), y1 + 2)
    return [float(x1), float(y1), float(x2), float(y2)]


def parse_candidates(path: Path, min_conf: float) -> list[BoxCandidate]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[BoxCandidate] = []
    for rec in data.get("records", []):
        try:
            video = str(rec["video"]); obj = int(rec["obj_id"]); frame = int(rec["frame_idx"])
        except Exception:
            continue
        for idx, cand in enumerate(rec.get("candidates", []) or []):
            try:
                conf = float(cand.get("confidence") or 0.0)
                bbox = [float(x) for x in cand.get("bbox_norm")]
            except Exception:
                continue
            if conf < min_conf or len(bbox) != 4:
                continue
            out.append(
                BoxCandidate(
                    video=video,
                    obj_id=obj,
                    frame_idx=frame,
                    candidate_id=str(cand.get("candidate_id") or chr(ord("A") + idx)),
                    bbox_norm=bbox,
                    confidence=conf,
                    reason_short=str(cand.get("reason_short") or ""),
                    risk_tags=[str(x) for x in cand.get("risk_tags", [])] if isinstance(cand.get("risk_tags"), list) else [],
                    model_used=rec.get("model_used"),
                    panel_path=rec.get("panel_path"),
                )
            )
    return out


def pred_root_for(template: Path, rank: int) -> Path:
    s = str(template)
    if "{rank}" in s:
        return Path(s.format(rank=rank))
    return template.with_name(template.name + f"_rank{rank}")


def copy_fallback(args: argparse.Namespace, root: Path) -> None:
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for video_dir in sorted(p for p in args.fallback_root.iterdir() if p.is_dir()):
        shutil.copytree(video_dir, root / video_dir.name, dirs_exist_ok=True)


def import_sam2(args: argparse.Namespace):
    sys.path.insert(0, str(args.sam2_root))
    from sam2.build_sam import build_sam2  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

    return build_sam2, SAM2ImagePredictor


def main() -> None:
    args = complete_paths(parse_args())
    jpeg_root = args.workspace / "homework" / "JPEGImages"
    ann_root = args.workspace / "homework" / "Annotations"
    candidates = parse_candidates(args.proposals_json, args.min_confidence)
    ranks = sorted({int(r) for r in args.ranks if int(r) >= 1})
    roots = {rank: pred_root_for(args.pred_root_template, rank) for rank in ranks}
    for root in roots.values():
        copy_fallback(args, root)

    audit: dict[str, Any] = {
        "method": "m9_box_proposals_sam2",
        "workspace": str(args.workspace),
        "proposals_json": str(args.proposals_json),
        "fallback_root": str(args.fallback_root),
        "roots": {str(k): str(v) for k, v in roots.items()},
        "ranks": ranks,
        "dry_run": bool(args.dry_run),
        "sam2_root": str(args.sam2_root),
        "checkpoint": str(args.checkpoint),
        "model_cfg": args.model_cfg,
        "records": [],
    }
    if not candidates:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"audit_json": str(args.audit_json), "candidates": 0, "roots": audit["roots"]}, ensure_ascii=False, indent=2))
        return

    by_key: dict[tuple[str, int, int], list[BoxCandidate]] = defaultdict(list)
    for c in candidates:
        by_key[(c.video, c.obj_id, c.frame_idx)].append(c)
    for key in by_key:
        by_key[key].sort(key=lambda c: (-c.confidence, c.candidate_id))

    predictor = None
    autocast_ctx_factory = contextlib.nullcontext
    if not args.dry_run:
        for required in [args.sam2_root, args.checkpoint]:
            if not required.exists():
                raise FileNotFoundError(required)
        import torch

        if str(args.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        build_sam2, SAM2ImagePredictor = import_sam2(args)
        model = build_sam2(args.model_cfg, str(args.checkpoint), device=args.device)
        model.eval()
        predictor = SAM2ImagePredictor(model)
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            autocast_ctx_factory = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # type: ignore[assignment]

    frame_cache: dict[tuple[str, int], tuple[np.ndarray, Path]] = {}
    masks_cache: dict[tuple[str, int, str], tuple[np.ndarray, float, list[float]]] = {}

    def get_rgb(video: str, frame_idx: int) -> tuple[np.ndarray, Path]:
        key = (video, frame_idx)
        if key not in frame_cache:
            frames = list_frames(jpeg_root / video)
            if frame_idx < 0 or frame_idx >= len(frames):
                raise IndexError(f"{video} frame {frame_idx} outside 0..{len(frames)-1}")
            frame = frames[frame_idx]
            frame_cache[key] = (np.asarray(Image.open(frame).convert("RGB")), frame)
        return frame_cache[key]

    for (video, obj_id, frame_idx), group in sorted(by_key.items()):
        rgb, frame_path = get_rgb(video, frame_idx)
        h, w = rgb.shape[:2]
        if not args.dry_run and predictor is not None:
            predictor.set_image(rgb)
        for local_rank, cand in enumerate(group, start=1):
            if local_rank not in roots:
                continue
            pbox = pixel_box(cand.bbox_norm, w, h, args.box_pad_frac, args.box_pad_px)
            mask = np.zeros((h, w), dtype=bool)
            pred_iou = 0.0
            if not args.dry_run and predictor is not None:
                import torch

                with torch.inference_mode(), autocast_ctx_factory():
                    masks, ious, _ = predictor.predict(
                        box=np.asarray(pbox, dtype=np.float32),
                        multimask_output=args.multimask_output,
                        return_logits=False,
                        normalize_coords=False,
                    )
                if masks.ndim == 2:
                    masks = masks[None]
                best = int(np.argmax(np.asarray(ious))) if len(ious) else 0
                mask = np.asarray(masks[best]) > 0
                pred_iou = float(np.asarray(ious)[best]) if len(ious) else 0.0
                if args.clip_to_prompt_box:
                    clip_box = pixel_box(cand.bbox_norm, w, h, args.clip_pad_frac, 0)
                    x1, y1, x2, y2 = [int(round(v)) for v in clip_box]
                    clip = np.zeros_like(mask, dtype=bool)
                    clip[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)] = True
                    clipped = mask & clip
                    if int(clipped.sum()) > 0:
                        mask = clipped
                    elif args.fallback_rectangle_if_empty:
                        mask = clip
            masks_cache[(video, frame_idx, cand.candidate_id)] = (mask, pred_iou, pbox)
            root = roots[local_rank]
            out_path = root / video / f"{frame_path.stem}.png"
            label, palette = load_label(out_path)
            ann, ann_palette = load_label(ann_root / video / "00000.png")
            allowed = {int(x) for x in np.unique(ann)}
            if frame_idx == 0:
                label = ann.copy()
            else:
                label = label.copy()
                label[label == obj_id] = 0
                # Do not overwrite another object id when the box prompt creates a composite mask.
                label[(label == 0) & mask] = obj_id
            invalid = sorted({int(x) for x in np.unique(label)} - allowed)
            if invalid:
                raise RuntimeError(f"invalid labels after proposal {video}/{frame_idx}: {invalid}")
            save_label(out_path, label, palette or ann_palette)
            audit["records"].append(
                {
                    **asdict(cand),
                    "rank": local_rank,
                    "frame_stem": frame_path.stem,
                    "pixel_box": [round(float(x), 2) for x in pbox],
                    "image_size": [int(w), int(h)],
                    "sam2_pred_iou": round(float(pred_iou), 5),
                    "clip_to_prompt_box": bool(args.clip_to_prompt_box),
                    "clip_pad_frac": float(args.clip_pad_frac),
                    "mask_area": int(mask.sum()),
                    "mask_bbox": bbox_from_mask(mask),
                    "pred_root": str(root),
                    "output_path": str(out_path),
                    "dry_run": bool(args.dry_run),
                }
            )
    # Ensure frame0 is exact GT in every output root.
    for rank, root in roots.items():
        for video_dir in sorted(p for p in jpeg_root.iterdir() if p.is_dir()):
            ann, palette = load_label(ann_root / video_dir.name / "00000.png")
            first_frame = list_frames(video_dir)[0]
            save_label(root / video_dir.name / f"{first_frame.stem}.png", ann, palette)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_json": str(args.audit_json), "records": len(audit["records"]), "roots": audit["roots"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Adapt DAM4SAM/d4sm to the MOSEv2 homework first-frame-mask format.

The adapter intentionally keeps external code outside cvMOSE.  It imports the
external tracker at runtime, feeds the local first-frame GT masks, writes label
PNGs matching the existing homework contract, and records an audit JSON.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def install_minimal_vot_stub() -> None:
    """Install a tiny subset of vot-toolkit APIs used by DAM4SAM/d4sm.

    b101 currently has no internet, and the SAM2 env lacks ``vot``.  The tracker
    only needs rectangle/mask conversion and rectangle IoU for DRM decisions, so
    a local in-process stub is safer than polluting the conda env.
    """
    if "vot" in sys.modules:
        return

    vot = types.ModuleType("vot")
    region = types.ModuleType("vot.region")
    raster = types.ModuleType("vot.region.raster")
    shapes = types.ModuleType("vot.region.shapes")

    class RegionType:
        SPECIAL = "special"
        RECTANGLE = "rectangle"
        MASK = "mask"

    class Rectangle:
        type = RegionType.RECTANGLE

        def __init__(self, x: float, y: float, width: float, height: float):
            self.x = float(x); self.y = float(y); self.width = float(width); self.height = float(height)

        def is_empty(self) -> bool:
            return self.width <= 0 or self.height <= 0

        def convert(self, region_type):
            return self

    class Mask:
        type = RegionType.MASK

        def __init__(self, mask):
            self.mask = (np.asarray(mask) > 0).astype(np.uint8)

        def rasterize(self, bounds):
            return self.mask

        def convert(self, region_type):
            ys, xs = np.nonzero(self.mask)
            if len(xs) == 0:
                return Rectangle(0, 0, 0, 0)
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            return Rectangle(float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1))

    def _rect_tuple(obj):
        if isinstance(obj, Rectangle):
            return obj.x, obj.y, obj.width, obj.height
        if isinstance(obj, (tuple, list)) and len(obj) >= 4:
            return float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])
        if hasattr(obj, "convert"):
            r = obj.convert(RegionType.RECTANGLE)
            return r.x, r.y, r.width, r.height
        raise TypeError(type(obj))

    def _rect_iou(a, b) -> float:
        ax, ay, aw, ah = _rect_tuple(a); bx, by, bw, bh = _rect_tuple(b)
        if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
            return 0.0
        ax2, ay2 = ax + aw, ay + ah; bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 0 else 0.0

    def calculate_overlaps(a_list, b_list, bounds=None):
        return [_rect_iou(a, b) for a, b in zip(a_list, b_list)]

    region.RegionType = RegionType
    region.Rectangle = Rectangle
    region.Mask = Mask
    raster.calculate_overlaps = calculate_overlaps
    shapes.Rectangle = Rectangle
    shapes.Mask = Mask

    vot.region = region
    sys.modules.update({
        "vot": vot,
        "vot.region": region,
        "vot.region.raster": raster,
        "vot.region.shapes": shapes,
    })


@dataclass
class VideoAudit:
    video: str
    status: str
    frames: int
    objects: list[int]
    seconds: float
    conflicts: int = 0
    empty_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--dam4sam-root", type=Path, default=None)
    p.add_argument("--d4sm-root", type=Path, default=None)
    p.add_argument("--mode", choices=["dam4sam_single_per_obj", "d4sm_multi"], default="d4sm_multi")
    p.add_argument("--tracker-name", default="sam21pp-B", help="DAM4SAM single tracker name")
    p.add_argument("--d4sm-model-size", choices=["large", "base", "small", "tiny"], default="large")
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--overwrite-submission", action="store_true")
    p.add_argument("--device", default="cuda", help="Only cuda is supported by upstream trackers; kept for audit symmetry.")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def complete(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve(); args.workspace = ws
    args.dam4sam_root = (args.dam4sam_root or ws.parent / "external" / "DAM4SAM").resolve()
    args.d4sm_root = (args.d4sm_root or ws.parent / "external" / "d4sm").resolve()
    args.checkpoint_dir = (args.checkpoint_dir or (args.d4sm_root if args.mode == "d4sm_multi" else args.dam4sam_root) / "checkpoints").resolve()
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.pred_root = args.pred_root.resolve()
    args.submit_root = (args.submit_root or ws / "homework" / f"submission_433_{args.mode}").resolve()
    args.zip_path = (args.zip_path or ws / "homework" / f"submission_mosev2_{args.mode}.zip").resolve()
    args.audit_json = args.audit_json.resolve()
    return args


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(video_dir)
    return frames


def load_ann(path: Path) -> tuple[np.ndarray, list[int] | None]:
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr, img.getpalette()


def save_label(path: Path, label: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(label.astype(np.uint8), mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def merge_masks(masks: list[np.ndarray], obj_ids: list[int], shape: tuple[int, int]) -> tuple[np.ndarray, int, dict[str, int]]:
    label = np.zeros(shape, dtype=np.uint8)
    conflicts = 0
    empty: dict[str, int] = {str(o): 0 for o in obj_ids}
    for oid, mask in zip(obj_ids, masks):
        m = np.asarray(mask) > 0
        if m.shape != shape:
            m = np.asarray(Image.fromarray(m.astype(np.uint8)).resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0
        if not m.any():
            empty[str(oid)] += 1
            continue
        overlap = m & (label > 0)
        conflicts += int(overlap.sum())
        label[(m) & (label == 0)] = int(oid)
    return label, conflicts, empty


def reset_external_modules() -> None:
    for name in list(sys.modules):
        if name == "sam2" or name.startswith("sam2.") or name in {"dam4sam_tracker", "tracking_wrapper_mot"}:
            sys.modules.pop(name, None)


def import_d4sm(root: Path):
    reset_external_modules()
    install_minimal_vot_stub()
    sys.path.insert(0, str(root))
    try:
        os.chdir(root)
        from tracking_wrapper_mot import DAM4SAMMOT
        return DAM4SAMMOT
    finally:
        pass


def import_dam4sam(root: Path):
    reset_external_modules()
    install_minimal_vot_stub()
    sys.path.insert(0, str(root))
    os.chdir(root)
    from dam4sam_tracker import DAM4SAMTracker
    return DAM4SAMTracker


def run_d4sm_video(DAM4SAMMOT, args: argparse.Namespace, video: str) -> VideoAudit:
    import torch

    t0 = time.time()
    frames = list_frames(args.jpeg_root / video)
    ann, palette = load_ann(args.ann_root / video / "00000.png")
    obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
    out_dir = args.pred_root / video
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = DAM4SAMMOT(model_size=args.d4sm_model_size, checkpoint_dir=str(args.checkpoint_dir))
    first = Image.open(frames[0]).convert("RGB")
    init_regions = [{"obj_id": str(oid), "mask": (ann == oid).astype(np.uint8)} for oid in obj_ids]
    tracker.initialize(first, init_regions)
    save_label(out_dir / f"{frames[0].stem}.png", ann, palette)
    conflicts = 0
    empty_counts = {str(oid): 0 for oid in obj_ids}
    with torch.inference_mode():
        for frame in frames[1:]:
            image = Image.open(frame).convert("RGB")
            outputs = tracker.track(image)
            label, c, empty = merge_masks(outputs["masks"], obj_ids, ann.shape)
            conflicts += c
            for k, v in empty.items():
                empty_counts[k] += v
            save_label(out_dir / f"{frame.stem}.png", label, palette)
    return VideoAudit(video, "done", len(frames), obj_ids, round(time.time() - t0, 2), conflicts, empty_counts)


def run_single_video(DAM4SAMTracker, args: argparse.Namespace, video: str) -> VideoAudit:
    import torch

    t0 = time.time()
    frames = list_frames(args.jpeg_root / video)
    ann, palette = load_ann(args.ann_root / video / "00000.png")
    obj_ids = [int(x) for x in np.unique(ann) if int(x) != 0]
    out_dir = args.pred_root / video
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trackers = []
    first = Image.open(frames[0]).convert("RGB")
    for oid in obj_ids:
        tr = DAM4SAMTracker(args.tracker_name)
        tr.initialize(first, (ann == oid).astype(np.uint8))
        trackers.append(tr)
    save_label(out_dir / f"{frames[0].stem}.png", ann, palette)
    conflicts = 0
    empty_counts = {str(oid): 0 for oid in obj_ids}
    with torch.inference_mode():
        for frame in frames[1:]:
            image = Image.open(frame).convert("RGB")
            masks = [tr.track(image)["pred_mask"] for tr in trackers]
            label, c, empty = merge_masks(masks, obj_ids, ann.shape)
            conflicts += c
            for k, v in empty.items():
                empty_counts[k] += v
            save_label(out_dir / f"{frame.stem}.png", label, palette)
    return VideoAudit(video, "done", len(frames), obj_ids, round(time.time() - t0, 2), conflicts, empty_counts)


def make_submission(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from infer_mosev2_sam2 import make_submission as make
    ns = argparse.Namespace(
        provided_output_root=args.provided_output_root,
        pred_root=args.pred_root,
        submit_root=args.submit_root,
        zip_path=args.zip_path,
        overwrite_submission=args.overwrite_submission,
        no_zip=False,
    )
    make(ns)


def main() -> None:
    cwd = Path.cwd()
    args = complete(parse_args())
    if args.pred_root.name in {"pred", "pred_", "pred_sam2_b101"}:
        raise RuntimeError(f"refusing unsafe pred root {args.pred_root}")
    args.pred_root.mkdir(parents=True, exist_ok=True)
    videos = args.videos or sorted(p.name for p in args.jpeg_root.iterdir() if p.is_dir())
    audit: dict[str, Any] = {
        "mode": args.mode,
        "workspace": str(args.workspace),
        "dam4sam_root": str(args.dam4sam_root),
        "d4sm_root": str(args.d4sm_root),
        "checkpoint_dir": str(args.checkpoint_dir),
        "pred_root": str(args.pred_root),
        "videos": [],
    }
    try:
        if args.mode == "d4sm_multi":
            Runner = import_d4sm(args.d4sm_root)
            run_one = lambda v: run_d4sm_video(Runner, args, v)
        else:
            Runner = import_dam4sam(args.dam4sam_root)
            run_one = lambda v: run_single_video(Runner, args, v)
        for idx, video in enumerate(videos, 1):
            print(f"[{idx}/{len(videos)}] {video} {args.mode}", flush=True)
            try:
                result = run_one(video)
            except Exception as exc:
                result = VideoAudit(video, "error", 0, [], 0.0, error=f"{type(exc).__name__}: {exc}")
                print(f"error {video}: {result.error}", flush=True)
                if not args.continue_on_error:
                    audit["videos"].append(asdict(result))
                    raise
            audit["videos"].append(asdict(result))
            print(asdict(result), flush=True)
        if args.make_submission:
            os.chdir(cwd)
            make_submission(args)
    finally:
        os.chdir(cwd)
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        audit["summary"] = {
            "done": sum(1 for v in audit["videos"] if v["status"] == "done"),
            "errors": sum(1 for v in audit["videos"] if v["status"] == "error"),
            "total_conflicts": sum(int(v.get("conflicts") or 0) for v in audit["videos"]),
        }
        args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

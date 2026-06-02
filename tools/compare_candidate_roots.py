#!/usr/bin/env python3
"""Compare MOSEv2 candidate prediction roots without using hidden labels.

The output is an audit aid for conservative fusion decisions.  It checks format
invariants and reports agreement/divergence between candidate roots, the SAM2
baseline, and optional references such as M11.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class ObjectStats:
    object_id: int
    empty_frames: int = 0
    mean_area: float = 0.0
    min_area: int = 0
    max_area: int = 0


@dataclass
class RootVideoStats:
    root_name: str
    video: str
    frame_count: int
    missing_frames: int = 0
    changed_frames_vs_baseline: int | None = None
    changed_frames_vs_m11: int | None = None
    mean_binary_iou_vs_baseline: float | None = None
    mean_binary_iou_vs_m11: float | None = None
    first_frame_preserved: bool | None = None
    label_ids_valid: bool = True
    invalid_label_samples: list[str] = field(default_factory=list)
    size_errors: list[str] = field(default_factory=list)
    objects: dict[str, ObjectStats] = field(default_factory=dict)


def parse_named_root(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name.strip(), Path(path).expanduser()
    path = Path(text).expanduser()
    return path.name, path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--m11-root", type=Path, default=None)
    p.add_argument("--roots", nargs="+", required=True, help="Candidate roots as name=/path or /path")
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    return p.parse_args()


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])
    if not frames:
        raise FileNotFoundError(video_dir)
    return frames


def load_label(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    if not path.is_file():
        return None
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    if shape is not None and arr.shape != shape:
        return None
    return arr


def load_ann(ann_root: Path, video: str) -> np.ndarray:
    path = ann_root / video / "00000.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr


def binary_iou(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    aa, bb = a > 0, b > 0
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(aa, bb).sum() / union)


def root_video_stats(
    name: str,
    root: Path,
    video: str,
    frames: list[Path],
    ann: np.ndarray,
    baseline_root: Path,
    m11_root: Path | None,
) -> RootVideoStats:
    allowed_ids = {int(x) for x in np.unique(ann)}
    obj_ids = [int(x) for x in sorted(allowed_ids) if int(x) != 0]
    out = RootVideoStats(root_name=name, video=video, frame_count=len(frames))
    areas_by_obj: dict[int, list[int]] = {oid: [] for oid in obj_ids}
    changed_base = 0
    changed_m11 = 0
    iou_base: list[float] = []
    iou_m11: list[float] = []
    for frame_idx, frame in enumerate(frames):
        pred_path = root / video / f"{frame.stem}.png"
        pred = load_label(pred_path, ann.shape)
        if pred is None:
            out.missing_frames += 1
            continue
        if pred.shape != ann.shape:
            out.size_errors.append(f"{frame.stem}:{pred.shape}!={ann.shape}")
            continue
        invalid = sorted({int(x) for x in np.unique(pred)} - allowed_ids)
        if invalid:
            out.label_ids_valid = False
            if len(out.invalid_label_samples) < 10:
                out.invalid_label_samples.append(f"{frame.stem}:{invalid}")
        if frame_idx == 0:
            out.first_frame_preserved = bool(np.array_equal(pred, ann))
        for oid in obj_ids:
            areas_by_obj[oid].append(int((pred == oid).sum()))
        base = load_label(baseline_root / video / f"{frame.stem}.png", ann.shape)
        if base is not None:
            if not np.array_equal(pred, base):
                changed_base += 1
            biou = binary_iou(pred, base)
            if biou is not None:
                iou_base.append(biou)
        if m11_root is not None:
            m11 = load_label(m11_root / video / f"{frame.stem}.png", ann.shape)
            if m11 is not None:
                if not np.array_equal(pred, m11):
                    changed_m11 += 1
                miou = binary_iou(pred, m11)
                if miou is not None:
                    iou_m11.append(miou)
    out.changed_frames_vs_baseline = changed_base if out.missing_frames < len(frames) else None
    out.mean_binary_iou_vs_baseline = round(float(np.mean(iou_base)), 6) if iou_base else None
    if m11_root is not None:
        out.changed_frames_vs_m11 = changed_m11 if out.missing_frames < len(frames) else None
        out.mean_binary_iou_vs_m11 = round(float(np.mean(iou_m11)), 6) if iou_m11 else None
    for oid, areas in areas_by_obj.items():
        if not areas:
            out.objects[str(oid)] = ObjectStats(object_id=oid)
        else:
            out.objects[str(oid)] = ObjectStats(
                object_id=oid,
                empty_frames=sum(1 for a in areas if a == 0),
                mean_area=round(float(np.mean(areas)), 2),
                min_area=int(min(areas)),
                max_area=int(max(areas)),
            )
    return out


def pairwise_agreement(root_items: list[tuple[str, Path]], video: str, frames: list[Path], shape: tuple[int, int]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for i, (name_a, root_a) in enumerate(root_items):
        for name_b, root_b in root_items[i + 1 :]:
            vals: list[float] = []
            for frame in frames:
                a = load_label(root_a / video / f"{frame.stem}.png", shape)
                b = load_label(root_b / video / f"{frame.stem}.png", shape)
                val = binary_iou(a, b)
                if val is not None:
                    vals.append(val)
            out[f"{name_a}__{name_b}"] = round(float(np.mean(vals)), 6) if vals else None
    return out


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = ws / "homework" / "JPEGImages"
    ann_root = ws / "homework" / "Annotations"
    baseline_root = args.baseline_root.resolve()
    m11_root = args.m11_root.resolve() if args.m11_root else None
    roots = [(name, path.resolve()) for name, path in [parse_named_root(x) for x in args.roots]]
    videos = args.videos or sorted(p.name for p in jpeg_root.iterdir() if p.is_dir())

    aggregate: dict[str, Any] = {
        "workspace": str(ws),
        "baseline_root": str(baseline_root),
        "m11_root": str(m11_root) if m11_root else None,
        "roots": {name: str(path) for name, path in roots},
        "videos": {},
    }
    rows: list[dict[str, Any]] = []
    for video in videos:
        frames = list_frames(jpeg_root / video)
        ann = load_ann(ann_root, video)
        per_root: dict[str, Any] = {}
        for name, root in roots:
            stats = root_video_stats(name, root, video, frames, ann, baseline_root, m11_root)
            per_root[name] = asdict(stats)
            row = {
                "root": name,
                "video": video,
                "frames": stats.frame_count,
                "missing_frames": stats.missing_frames,
                "changed_vs_baseline": stats.changed_frames_vs_baseline,
                "changed_vs_m11": stats.changed_frames_vs_m11,
                "mean_iou_vs_baseline": stats.mean_binary_iou_vs_baseline,
                "mean_iou_vs_m11": stats.mean_binary_iou_vs_m11,
                "first_frame_preserved": stats.first_frame_preserved,
                "label_ids_valid": stats.label_ids_valid,
                "invalid_label_samples": ";".join(stats.invalid_label_samples),
                "size_errors": ";".join(stats.size_errors),
            }
            for oid, obj in stats.objects.items():
                row[f"obj{oid}_empty"] = obj.empty_frames
                row[f"obj{oid}_mean_area"] = obj.mean_area
                row[f"obj{oid}_min_area"] = obj.min_area
                row[f"obj{oid}_max_area"] = obj.max_area
            rows.append(row)
        aggregate["videos"][video] = {
            "frames": len(frames),
            "object_ids": [int(x) for x in np.unique(ann) if int(x) != 0],
            "roots": per_root,
            "pairwise_binary_iou": pairwise_agreement(roots, video, frames, ann.shape),
        }

    summary: dict[str, Any] = {"roots": {}}
    for name, _ in roots:
        name_rows = [r for r in rows if r["root"] == name]
        summary["roots"][name] = {
            "videos": len(name_rows),
            "missing_frames": int(sum(int(r["missing_frames"]) for r in name_rows)),
            "changed_vs_baseline": int(sum(int(r["changed_vs_baseline"] or 0) for r in name_rows)),
            "changed_vs_m11": int(sum(int(r["changed_vs_m11"] or 0) for r in name_rows)),
            "first_frame_failures": [r["video"] for r in name_rows if r["first_frame_preserved"] is False],
            "invalid_label_videos": [r["video"] for r in name_rows if not r["label_ids_valid"]],
        }
    aggregate["summary"] = summary

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({k for row in rows for k in row})
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

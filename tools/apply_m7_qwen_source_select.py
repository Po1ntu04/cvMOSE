#!/usr/bin/env python3
"""Apply Qwen-VL candidate judgments as conservative output-level source selection.

This tool is deliberately not a mask generator.  It can only copy an existing
candidate source mask for a judged video/object/frame when Qwen selected that
candidate with sufficient support.  It is meant as an ablation between pure MLLM
anchor-gating and full tracker re-propagation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class Rule:
    video: str
    obj_id: int
    frame_idx: int
    source: str
    source_root: Path
    confidence: float
    reason: str
    record: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--default-root", type=Path, required=True, help="Base prediction root, usually M11")
    p.add_argument("--judgments-json", type=Path, action="append", required=True)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--source-root", action="append", default=[], help="Override source root as name=/path; repeatable")
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--min-support-confidence", type=float, default=0.85)
    p.add_argument("--allow-support-with-veto", action="store_true")
    p.add_argument("--allow-same-class-dense", action="store_true")
    p.add_argument("--allow-stable-guard", action="store_true", help="Allow videos such as 8jsm23a7; default rejects by video guard")
    p.add_argument("--stable-guard-videos", nargs="*", default=["8jsm23a7", "jadgtmfl"])
    p.add_argument("--frame-radius", type=int, default=0, help="Optionally apply selected source to neighboring frames")
    p.add_argument("--max-changed-frames-per-video", type=int, default=12)
    p.add_argument(
        "--nonempty-area-ratio-min",
        type=float,
        default=0.80,
        help="When default and selected source are both non-empty, reject source replacements smaller than this area ratio.",
    )
    p.add_argument(
        "--nonempty-area-ratio-max",
        type=float,
        default=1.40,
        help="When default and selected source are both non-empty, reject source replacements larger than this area ratio.",
    )
    p.add_argument(
        "--allow-nonempty-area-shift",
        action="store_true",
        help="Disable the conservative area-ratio guard for replacing one non-empty mask with another.",
    )
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_named_roots(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"bad --source-root {item!r}; expected name=/path")
        name, value = item.split("=", 1)
        out[name.strip()] = Path(value).expanduser().resolve()
    return out


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
    img = Image.fromarray(arr.astype(np.uint8), mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def source_from_record(rec: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    j = rec.get("judgment", rec)
    best = str(j.get("best_candidate", "")).strip()
    if len(best) == 1 and best.isalpha():
        for cand in rec.get("candidates", []):
            if str(cand.get("candidate_id")) == best:
                return str(cand.get("source")), cand
    if best == "keep_baseline":
        return "baseline", None
    return None, None


def read_judgments(paths: list[Path], root_overrides: dict[str, Path], args: argparse.Namespace) -> tuple[list[Rule], dict[str, Path], list[dict[str, Any]]]:
    roots: dict[str, Path] = dict(root_overrides)
    raw_records: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        for name, value in data.get("pred_roots", {}).items():
            roots.setdefault(str(name), Path(value).expanduser().resolve())
        raw_records.extend(data.get("records", []))
    roots.setdefault("m11", args.default_root.resolve())
    roots.setdefault("default", args.default_root.resolve())
    rules: list[Rule] = []
    seen: set[tuple[str, int, int, str]] = set()
    for rec in raw_records:
        j = rec.get("judgment", rec)
        try:
            video = str(rec["video"]); obj_id = int(rec["obj_id"]); frame_idx = int(rec["frame_idx"])
        except Exception:
            continue
        src, _ = source_from_record(rec)
        conf = float(j.get("confidence") or 0.0)
        support = bool(j.get("should_support_anchor"))
        veto = bool(j.get("should_veto_anchor"))
        profile = rec.get("profile", {}) or {}
        target_type = str(profile.get("target_type", "unknown"))
        reason = ""
        if not support:
            reason = "no_support"
        elif conf < args.min_support_confidence:
            reason = f"low_conf:{conf:.2f}"
        elif veto and not args.allow_support_with_veto:
            reason = "support_with_veto_rejected"
        elif not src or src not in roots:
            reason = f"missing_source:{src}"
        elif target_type == "same_class_dense" and not args.allow_same_class_dense:
            reason = "same_class_dense_guard"
        elif video in set(args.stable_guard_videos) and not args.allow_stable_guard:
            reason = "stable_guard_video"
        elif src in {"m11", "default"} or roots[src].resolve() == args.default_root.resolve():
            reason = "same_as_default"
        if reason:
            rejects.append({"video": video, "obj_id": obj_id, "frame_idx": frame_idx, "source": src, "confidence": conf, "reason": reason, "judgment": j})
            continue
        key = (video, obj_id, frame_idx, src)
        if key in seen:
            continue
        seen.add(key)
        rules.append(Rule(video, obj_id, frame_idx, src, roots[src], conf, str(j.get("reason_short", "")), rec))
    return rules, roots, rejects


def make_submission(workspace: Path, pred_root: Path, submit_root: Path, zip_path: Path, overwrite: bool) -> dict[str, Any]:
    provided_root = workspace / "homework" / "output"
    if submit_root.exists() and overwrite:
        shutil.rmtree(submit_root)
    submit_root.mkdir(parents=True, exist_ok=True)
    for root in [provided_root, pred_root]:
        for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            shutil.copytree(video_dir, submit_root / video_dir.name, dirs_exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(submit_root.rglob("*.png")):
            zf.write(path, path.relative_to(submit_root).as_posix())
    return {"submit_root": str(submit_root), "zip_path": str(zip_path), "video_dirs": len([p for p in submit_root.iterdir() if p.is_dir()]), "pngs": sum(1 for _ in submit_root.rglob("*.png")), "zip_size": zip_path.stat().st_size}


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = ws / "homework" / "JPEGImages"
    ann_root = ws / "homework" / "Annotations"
    default_root = args.default_root.resolve()
    pred_root = args.pred_root.resolve()
    root_overrides = parse_named_roots(args.source_root)
    rules, roots, rejects = read_judgments(args.judgments_json, root_overrides, args)
    videos_filter = set(args.videos or [])
    rules_by_video: dict[str, list[Rule]] = {}
    for rule in rules:
        if videos_filter and rule.video not in videos_filter:
            continue
        rules_by_video.setdefault(rule.video, []).append(rule)
    if pred_root.exists() and args.overwrite:
        shutil.rmtree(pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "method": "m7_qwen_output_source_select",
        "workspace": str(ws),
        "default_root": str(default_root),
        "judgments_json": [str(p) for p in args.judgments_json],
        "source_roots": {k: str(v) for k, v in roots.items()},
        "config": {
            "min_support_confidence": args.min_support_confidence,
            "allow_support_with_veto": args.allow_support_with_veto,
            "allow_same_class_dense": args.allow_same_class_dense,
            "allow_stable_guard": args.allow_stable_guard,
            "frame_radius": args.frame_radius,
            "max_changed_frames_per_video": args.max_changed_frames_per_video,
            "nonempty_area_ratio_min": args.nonempty_area_ratio_min,
            "nonempty_area_ratio_max": args.nonempty_area_ratio_max,
            "allow_nonempty_area_shift": args.allow_nonempty_area_shift,
        },
        "rejected_records": rejects,
        "videos": {},
    }
    for video_dir in sorted(p for p in jpeg_root.iterdir() if p.is_dir()):
        video = video_dir.name
        frames = list_frames(video_dir)
        ann, palette = load_label(ann_root / video / "00000.png")
        allowed = {int(x) for x in np.unique(ann)}
        out_dir = pred_root / video
        out_dir.mkdir(parents=True, exist_ok=True)
        video_rules = rules_by_video.get(video, [])
        expanded: dict[int, list[Rule]] = {}
        for rule in video_rules:
            for f in range(max(1, rule.frame_idx - args.frame_radius), min(len(frames), rule.frame_idx + args.frame_radius + 1)):
                expanded.setdefault(f, []).append(rule)
        changed_frames = 0
        applied: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for frame_idx, frame in enumerate(frames):
            base, base_pal = load_label(default_root / video / f"{frame.stem}.png")
            label = ann.copy() if frame_idx == 0 else base.copy()
            if frame_idx > 0 and frame_idx in expanded and changed_frames < args.max_changed_frames_per_video:
                # Highest-confidence rule wins for this object/frame.
                for rule in sorted(expanded[frame_idx], key=lambda r: r.confidence, reverse=True):
                    src_path = rule.source_root / video / f"{frame.stem}.png"
                    if not src_path.is_file():
                        skipped.append({"frame_idx": frame_idx, "obj_id": rule.obj_id, "source": rule.source, "reason": "missing_src_path", "path": str(src_path)})
                        continue
                    src, _ = load_label(src_path)
                    if src.shape != label.shape:
                        skipped.append({"frame_idx": frame_idx, "obj_id": rule.obj_id, "source": rule.source, "reason": "shape_mismatch"})
                        continue
                    base_area = int((base == rule.obj_id).sum())
                    src_area = int((src == rule.obj_id).sum())
                    if (
                        not args.allow_nonempty_area_shift
                        and base_area > 0
                        and src_area > 0
                        and not (args.nonempty_area_ratio_min <= src_area / max(1, base_area) <= args.nonempty_area_ratio_max)
                    ):
                        skipped.append(
                            {
                                "frame_idx": frame_idx,
                                "obj_id": rule.obj_id,
                                "source": rule.source,
                                "reason": "nonempty_area_ratio_guard",
                                "base_area": base_area,
                                "src_area": src_area,
                                "area_ratio": src_area / max(1, base_area),
                            }
                        )
                        continue
                    before = label.copy()
                    label[label == rule.obj_id] = 0
                    label[(label == 0) & (src == rule.obj_id)] = rule.obj_id
                    if np.array_equal(label, before):
                        skipped.append({"frame_idx": frame_idx, "obj_id": rule.obj_id, "source": rule.source, "reason": "no_pixel_change"})
                    else:
                        applied.append({"frame_idx": frame_idx, "obj_id": rule.obj_id, "source": rule.source, "confidence": rule.confidence, "reason": rule.reason, "anchor_frame": rule.frame_idx})
            invalid = sorted({int(x) for x in np.unique(label)} - allowed)
            if invalid:
                raise RuntimeError(f"{video}/{frame.stem}: invalid labels {invalid}")
            if not np.array_equal(label, base):
                changed_frames += 1
            save_label(out_dir / f"{frame.stem}.png", label, palette or base_pal)
        audit["videos"][video] = {"frames": len(frames), "rules": len(video_rules), "applied": applied, "skipped": skipped[:50], "changed_vs_default": changed_frames}
    if args.make_submission:
        if args.submit_root is None or args.zip_path is None:
            raise SystemExit("--make-submission requires --submit-root and --zip-path")
        audit["submission"] = make_submission(ws, pred_root, args.submit_root.resolve(), args.zip_path.resolve(), args.overwrite)
    audit["summary"] = {"videos": len(audit["videos"]), "changed_vs_default": sum(v["changed_vs_default"] for v in audit["videos"].values()), "applied_rules": sum(len(v["applied"]) for v in audit["videos"].values()), "rejected_records": len(rejects)}
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_json": str(args.audit_json), "summary": audit["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

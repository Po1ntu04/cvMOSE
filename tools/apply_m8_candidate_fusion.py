#!/usr/bin/env python3
"""Apply M8 candidate-pool accepted records as a conservative source fusion.

This is an output-level ablation before expensive SAM2 re-anchor: it can only
copy masks from existing candidate roots recorded by build_m8_candidate_pool.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--candidate-json", type=Path, required=True)
    p.add_argument("--default-root", type=Path, required=True)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--min-score", type=float, default=0.18)
    p.add_argument("--min-margin", type=float, default=0.08)
    p.add_argument("--sameclass-min-margin", type=float, default=0.18)
    p.add_argument("--min-pos", type=float, default=0.50)
    p.add_argument("--max-frames-per-video", type=int, default=8)
    p.add_argument("--max-frames-per-object", type=int, default=6)
    p.add_argument("--only-empty-default", action="store_true", default=True)
    p.add_argument("--allow-nonempty-replacement", dest="only_empty_default", action="store_false")
    p.add_argument("--allow-stable-guard", action="store_true")
    p.add_argument("--require-qwen-for-sameclass", action="store_true", default=True)
    p.add_argument("--allow-unverified-sameclass", dest="require_qwen_for_sameclass", action="store_false")
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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


def list_frames(video_dir: Path) -> list[Path]:
    return sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")])


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


def candidate_rules(data: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    roots = {str(k): Path(v) for k, v in data.get("source_roots", {}).items()}
    same_class = {"r13u5z4y", "q0sizv6m", "msinig6m", "2smf7uq9", "8jsm23a7", "4vznweiu"}
    stable = {"8jsm23a7", "jadgtmfl"}
    rules: dict[str, list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    for video, vd in data.get("videos", {}).items():
        for obj_id, od in vd.get("objects", {}).items():
            for rec in od.get("accepted", []):
                reason = None
                root_name = str(rec.get("root_name"))
                src = str(rec.get("source"))
                margin = float(rec.get("margin") or 0.0)
                pos = float(rec.get("pos_sim") or 0.0)
                score = float(rec.get("score") or 0.0)
                if video in stable and not args.allow_stable_guard:
                    reason = "stable_guard"
                elif root_name not in roots:
                    reason = f"missing_root:{root_name}"
                elif src.startswith("default") or root_name == "default":
                    reason = "same_as_default"
                elif ":anyfg:" in src:
                    # The audit can score connected components for recall diagnostics,
                    # but this output-level fusion only has access to stored label PNGs.
                    # Copying src==obj_id from the full root would not reproduce the
                    # audited component, so component candidates must be passed to a
                    # re-anchor stage rather than copied here.
                    reason = "component_candidate_requires_reanchor"
                elif score < args.min_score:
                    reason = f"low_score:{score:.3f}"
                elif pos < args.min_pos:
                    reason = f"low_pos:{pos:.3f}"
                elif margin < (args.sameclass_min_margin if video in same_class else args.min_margin) and not rec.get("qwen_support"):
                    reason = f"low_margin:{margin:.3f}"
                elif rec.get("qwen_veto"):
                    reason = "qwen_veto"
                elif video in same_class and args.require_qwen_for_sameclass and not rec.get("qwen_support"):
                    reason = "sameclass_requires_qwen_support"
                if reason:
                    rejected.append({"reason": reason, **rec}); continue
                rule = dict(rec)
                rule["source_root"] = str(roots[root_name])
                rules.setdefault(video, []).append(rule)
    for video in list(rules):
        # Highest score per frame/object/source; cap video-level count.
        rules[video].sort(key=lambda r: (bool(r.get("qwen_support")), r.get("area_ratio_default") is None, float(r.get("score") or 0), float(r.get("margin") or 0), int(r.get("area") or 0)), reverse=True)
        per_obj: dict[int, int] = {}; kept = []
        for r in rules[video]:
            obj = int(r["obj_id"])
            if len(kept) >= args.max_frames_per_video:
                break
            if per_obj.get(obj, 0) >= args.max_frames_per_object:
                continue
            kept.append(r); per_obj[obj] = per_obj.get(obj, 0) + 1
        rules[video] = kept
    return rules, rejected


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = ws / "homework" / "JPEGImages"
    ann_root = ws / "homework" / "Annotations"
    data = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    rules, rejected = candidate_rules(data, args)
    if args.videos:
        rules = {k: v for k, v in rules.items() if k in set(args.videos)}
    pred_root = args.pred_root.resolve()
    if pred_root.exists() and args.overwrite:
        shutil.rmtree(pred_root)
    pred_root.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {"method": "m8_candidate_pool_source_fusion", "candidate_json": str(args.candidate_json), "default_root": str(args.default_root.resolve()), "config": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))}, "rejected": rejected[:200], "videos": {}}
    for video_dir in sorted(p for p in jpeg_root.iterdir() if p.is_dir()):
        video = video_dir.name
        frames = list_frames(video_dir)
        ann, palette = load_label(ann_root / video / "00000.png")
        allowed = {int(x) for x in np.unique(ann)}
        out_dir = pred_root / video; out_dir.mkdir(parents=True, exist_ok=True)
        by_frame: dict[int, list[dict[str, Any]]] = {}
        for r in rules.get(video, []):
            by_frame.setdefault(int(r["frame_idx"]), []).append(r)
        changed = 0; applied = []; skipped = []
        for idx, frame in enumerate(frames):
            base, base_pal = load_label(args.default_root / video / f"{frame.stem}.png")
            label = ann.copy() if idx == 0 else base.copy()
            if idx > 0 and idx in by_frame:
                for r in sorted(by_frame[idx], key=lambda x: float(x.get("score") or 0), reverse=True):
                    obj = int(r["obj_id"])
                    if args.only_empty_default and int((base == obj).sum()) > 0:
                        skipped.append({"frame_idx": idx, "obj_id": obj, "source": r.get("source"), "reason": "default_nonempty_guard"}); continue
                    src_path = Path(r["source_root"]) / video / f"{frame.stem}.png"
                    if not src_path.is_file():
                        skipped.append({"frame_idx": idx, "obj_id": obj, "source": r.get("source"), "reason": "missing_source_path", "path": str(src_path)}); continue
                    src, _ = load_label(src_path)
                    if src.shape != label.shape:
                        skipped.append({"frame_idx": idx, "obj_id": obj, "source": r.get("source"), "reason": "shape_mismatch"}); continue
                    mask = src == obj
                    if int(mask.sum()) <= 0:
                        skipped.append({"frame_idx": idx, "obj_id": obj, "source": r.get("source"), "reason": "source_empty"}); continue
                    before = label.copy()
                    label[label == obj] = 0
                    label[(label == 0) & mask] = obj
                    if np.array_equal(label, before):
                        skipped.append({"frame_idx": idx, "obj_id": obj, "source": r.get("source"), "reason": "no_pixel_change"})
                    else:
                        applied.append({"frame_idx": idx, "obj_id": obj, "source": r.get("source"), "root_name": r.get("root_name"), "score": r.get("score"), "margin": r.get("margin"), "reason": r.get("qwen_reason") or "m8_candidate_pool"})
            invalid = sorted({int(x) for x in np.unique(label)} - allowed)
            if invalid:
                raise RuntimeError(f"{video}/{frame.stem}: invalid labels {invalid}")
            if not np.array_equal(label, base):
                changed += 1
            save_label(out_dir / f"{frame.stem}.png", label, palette or base_pal)
        audit["videos"][video] = {"frames": len(frames), "rules": len(rules.get(video, [])), "applied": applied, "skipped": skipped[:80], "changed_vs_default": changed}
    if args.make_submission:
        if args.submit_root is None or args.zip_path is None:
            raise SystemExit("--make-submission requires --submit-root and --zip-path")
        audit["submission"] = make_submission(ws, pred_root, args.submit_root.resolve(), args.zip_path.resolve(), args.overwrite)
    audit["summary"] = {"videos": len(audit["videos"]), "rules": sum(v["rules"] for v in audit["videos"].values()), "applied": sum(len(v["applied"]) for v in audit["videos"].values()), "changed_vs_default": sum(v["changed_vs_default"] for v in audit["videos"].values()), "rejected": len(rejected)}
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_json": str(args.audit_json), "summary": audit["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

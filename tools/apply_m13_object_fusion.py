#!/usr/bin/env python3
"""Object/frame-range fusion for M13 candidate roots.

Policy-driven fusion only: default copies a baseline root and replaces selected
object ids from named candidate roots on selected frame ranges.  It is designed
for safe/balanced probes after visual review, not automatic scoring against GT.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--candidate-roots-json", type=Path, required=True, help="{name:path}")
    p.add_argument("--policy-json", type=Path, required=True)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr


def save_label(path: Path, arr: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr.astype(np.uint8), mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def frame_indices(spec: dict[str, Any], n: int) -> set[int]:
    if "frames" in spec:
        return {int(x) for x in spec["frames"] if 0 <= int(x) < n}
    start = int(spec.get("start", 1)); end = int(spec.get("end", n - 1))
    return set(range(max(1, start), min(n - 1, end) + 1))


def make_submission(workspace: Path, pred_root: Path, submit_root: Path, zip_path: Path, overwrite: bool) -> None:
    from validate_mose_submission import validate_zip

    provided = workspace / "homework" / "output"
    jpeg_root = workspace / "homework" / "JPEGImages"
    ann_root = workspace / "homework" / "Annotations"
    if submit_root.exists() and overwrite:
        shutil.rmtree(submit_root)
    submit_root.mkdir(parents=True, exist_ok=True)
    for root in [provided, pred_root]:
        for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            dst = submit_root / video_dir.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(video_dir, dst)
    if zip_path.exists() and overwrite:
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for png in sorted(submit_root.rglob("*.png")):
            zf.write(png, png.relative_to(submit_root).as_posix())
    result = validate_zip(zip_path, provided, jpeg_root, ann_root, 433, 66526, 15, False)
    if not result.get("ok"):
        raise SystemExit("submission validation failed: " + json.dumps(result, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = ws / "homework" / "JPEGImages"
    ann_root = ws / "homework" / "Annotations"
    roots = {k: Path(v).expanduser().resolve() for k, v in json.loads(args.candidate_roots_json.read_text(encoding="utf-8")).items()}
    policy = json.loads(args.policy_json.read_text(encoding="utf-8"))
    edits = policy.get("edits", policy if isinstance(policy, list) else [])
    by_video: dict[str, list[dict[str, Any]]] = {}
    for edit in edits:
        by_video.setdefault(str(edit["video"]), []).append(edit)
    if args.pred_root.exists() and args.overwrite:
        shutil.rmtree(args.pred_root)
    args.pred_root.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {"base_root": str(args.base_root), "roots": {k: str(v) for k, v in roots.items()}, "edits": [], "videos": {}}
    for video_dir in sorted(p for p in (ws / "homework" / "JPEGImages").iterdir() if p.is_dir()):
        video = video_dir.name
        frames = sorted(video_dir.glob("*.jpg")) or sorted(video_dir.glob("*.png"))
        src_base = args.base_root / video
        dst = args.pred_root / video
        dst.mkdir(parents=True, exist_ok=True)
        ann_img = Image.open(ann_root / video / "00000.png")
        palette = ann_img.getpalette()
        changed_frames: set[int] = set()
        video_edits = by_video.get(video, [])
        frame_edit_map: dict[int, list[dict[str, Any]]] = {}
        for edit in video_edits:
            for idx in frame_indices(edit, len(frames)):
                frame_edit_map.setdefault(idx, []).append(edit)
        for idx, frame in enumerate(frames):
            base = load_label(src_base / f"{frame.stem}.png").copy()
            final = base.copy()
            if idx == 0:
                final = load_label(ann_root / video / "00000.png").copy()
            else:
                for edit in frame_edit_map.get(idx, []):
                    root_name = str(edit["source"])
                    obj_id = int(edit["obj_id"])
                    cand = load_label(roots[root_name] / video / f"{frame.stem}.png")
                    final[final == obj_id] = 0
                    final[cand == obj_id] = obj_id
            if not np.array_equal(base, final):
                changed_frames.add(idx)
            save_label(dst / f"{frame.stem}.png", final, palette)
        audit["videos"][video] = {"changed_frames": sorted(changed_frames), "edit_count": len(video_edits)}
    audit["edits"] = edits
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.make_submission:
        submit_root = args.submit_root or (ws / "homework" / "submission_433_m13_object_fusion")
        zip_path = args.zip_path or (ws / "homework" / "submission_mosev2_m13_object_fusion.zip")
        make_submission(ws, args.pred_root, submit_root, zip_path, args.overwrite)
    print(json.dumps({"pred_root": str(args.pred_root), "videos": len(audit["videos"]), "changed_videos": sum(1 for v in audit["videos"].values() if v["changed_frames"]), "audit_json": str(args.audit_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

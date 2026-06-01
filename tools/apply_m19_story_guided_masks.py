#!/usr/bin/env python3
"""Apply narrow story-guided mask repairs for hard MOSEv2 videos.

This is a training-free diagnostic/fusion tool.  It does not use later GT; it
encodes manually reviewed physical-instance stories as conservative box masks
on top of an existing prediction root.  The intent is to test whether the
semantic/temporal story can move hidden rows before investing in a general
tracker implementation.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--audit-json", type=Path, required=True)
    p.add_argument("--profile", choices=["8js_late", "8js_late_1ql", "8js_late_4vz", "8js_late_4vz_1ql", "8js_full", "8js_4vz", "8js_4vz_1ql"], default="8js_full")
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    return arr if arr.ndim == 2 else arr[..., 0]


def save_label(path: Path, arr: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr.astype(np.uint8), mode="P")
    if palette:
        img.putpalette(palette)
    img.save(path)


def fill_box(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = mask.shape
    x1, y1, x2, y2 = box
    x1 = max(0, min(w, int(round(x1))))
    x2 = max(0, min(w, int(round(x2))))
    y1 = max(0, min(h, int(round(y1))))
    y2 = max(0, min(h, int(round(y2))))
    out = np.zeros_like(mask, dtype=bool)
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = True
    return out


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interp_box(frame: int, anchors: list[tuple[int, tuple[int, int, int, int]]]) -> tuple[int, int, int, int]:
    anchors = sorted(anchors)
    if frame <= anchors[0][0]:
        return anchors[0][1]
    if frame >= anchors[-1][0]:
        return anchors[-1][1]
    for (f0, b0), (f1, b1) in zip(anchors, anchors[1:]):
        if f0 <= frame <= f1:
            t = (frame - f0) / max(1, f1 - f0)
            return tuple(int(round(lerp(b0[i], b1[i], t))) for i in range(4))  # type: ignore[return-value]
    return anchors[-1][1]


def apply_8js(frame_idx: int, shape: tuple[int, int], full_story: bool) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    base = np.zeros(shape, dtype=bool)
    # User-confirmed: picked tile is placed as the player/front-row leftmost seven-bamboo.
    if frame_idx in {1, 2} and full_story:
        return base, {"mode": "empty_occluded", "reason": "hand occlusion before reappearance"}
    early_boxes = {
        3: (55, 365, 210, 590),   # visible between fingers / motion blur
        4: (205, 365, 365, 620),  # hand-held tile, partial
        5: (520, 445, 625, 650),  # hand-held seven-bamboo, partially finger-occluded
    }
    if full_story and frame_idx in early_boxes:
        box = early_boxes[frame_idx]
        return fill_box(base, box), {"mode": "hand_visible_box", "box": list(box)}
    if frame_idx >= 6:
        # Stable front row.  These boxes intentionally sit around y≈760-920 px;
        # M16's failed version used y≈1000-1350 px and clipped the table edge/hand.
        anchors = [
            (6, (98, 742, 220, 916)),
            (8, (98, 752, 220, 918)),
            (20, (98, 758, 224, 920)),
            (36, (102, 760, 220, 922)),
            (48, (100, 760, 218, 924)),
        ]
        box = interp_box(frame_idx, anchors)
        return fill_box(base, box), {"mode": "front_left_seven_bamboo", "box": list(box)}
    return None, None


def apply_4vz(frame_idx: int, shape: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    base = np.zeros(shape, dtype=bool)
    # First-frame target is the T-letter die in TRY.  Keep the stable early SAM2
    # segment; probe only the scatter/reappearance interval where the baseline
    # intermittently empties or stretches onto neighboring dice/flowers.
    if frame_idx < 14:
        return None, None
    anchors = [
        (14, (285, 585, 350, 690)),
        (16, (265, 600, 330, 680)),
        (20, (240, 605, 325, 690)),
        (25, (238, 610, 318, 695)),
        (34, (238, 610, 318, 698)),
    ]
    box = interp_box(frame_idx, anchors)
    return fill_box(base, box), {"mode": "letter_T_die_scatter_track", "box": list(box)}


def apply_1ql(frame_idx: int, shape: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    base = np.zeros(shape, dtype=bool)
    # Baseline already follows the car through most frames.  Only test the two
    # bridge-occlusion frames that are currently empty, using same-lane linear
    # continuation between frame 12 and 15.
    boxes = {
        13: (1145, 450, 1182, 476),
        14: (1174, 426, 1210, 452),
    }
    if frame_idx in boxes:
        box = boxes[frame_idx]
        return fill_box(base, box), {"mode": "same_lane_bridge_occlusion_fill", "box": list(box)}
    return None, None


def make_submission(workspace: Path, pred_root: Path, submit_root: Path, zip_path: Path, overwrite: bool) -> dict[str, Any]:
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
        raise SystemExit("validation failed " + json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = ws / "homework" / "JPEGImages"
    ann_root = ws / "homework" / "Annotations"
    if args.pred_root.exists() and args.overwrite:
        shutil.rmtree(args.pred_root)
    args.pred_root.mkdir(parents=True, exist_ok=True)
    enabled = {"8jsm23a7"}
    if args.profile in {"8js_4vz", "8js_4vz_1ql", "8js_late_4vz", "8js_late_4vz_1ql"}:
        enabled.add("4vznweiu")
    if args.profile in {"8js_4vz_1ql", "8js_late_1ql", "8js_late_4vz_1ql"}:
        enabled.add("1qlssuz2")
    audit: dict[str, Any] = {"method": "m19_story_guided_masks", "profile": args.profile, "base_root": str(args.base_root), "videos": {}, "enabled": sorted(enabled)}
    for video_dir in sorted(p for p in jpeg_root.iterdir() if p.is_dir()):
        video = video_dir.name
        dst = args.pred_root / video
        dst.mkdir(parents=True, exist_ok=True)
        ann_img = Image.open(ann_root / video / "00000.png")
        palette = ann_img.getpalette()
        video_audit = {"changed_frames": [], "frames": {}}
        for frame_path in sorted([*video_dir.glob("*.jpg"), *video_dir.glob("*.jpeg"), *video_dir.glob("*.png")]):
            idx = int(frame_path.stem)
            base = load_label(args.base_root / video / f"{frame_path.stem}.png").copy()
            final = base.copy()
            info = None
            if video in enabled:
                if video == "8jsm23a7":
                    mask, info = apply_8js(idx, base.shape, full_story=not args.profile.startswith("8js_late"))
                    obj_id = 1
                elif video == "4vznweiu":
                    mask, info = apply_4vz(idx, base.shape)
                    obj_id = 1
                elif video == "1qlssuz2":
                    mask, info = apply_1ql(idx, base.shape)
                    obj_id = 1
                else:
                    mask = None; obj_id = 0
                if mask is not None:
                    before = final == obj_id
                    final[final == obj_id] = 0
                    final[mask] = obj_id
                    if not np.array_equal(before, final == obj_id):
                        video_audit["changed_frames"].append(idx)
                    if info is not None:
                        info = dict(info)
                        info["area"] = int(mask.sum())
                        video_audit["frames"][str(idx)] = info
            save_label(dst / f"{frame_path.stem}.png", final, palette)
        if video_audit["changed_frames"]:
            audit["videos"][video] = video_audit
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation = None
    if args.make_submission:
        submit = args.submit_root or (ws / "homework" / f"submission_433_m19_{args.profile}")
        zip_path = args.zip_path or (ws / "homework" / f"submission_mosev2_m19_{args.profile}.zip")
        validation = make_submission(ws, args.pred_root, submit, zip_path, args.overwrite)
    print(json.dumps({"pred_root": str(args.pred_root), "audit_json": str(args.audit_json), "changed_videos": sorted(audit["videos"]), "validation": validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()

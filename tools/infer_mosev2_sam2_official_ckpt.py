#!/usr/bin/env python3
"""Run or extract the FudanCVL MOSEv2 official SAM2 baselines.

This is a thin, auditable wrapper around ``tools/infer_mosev2_sam2.py``.  It
keeps the homework I/O contract identical to the existing SAM2 baseline while
changing only the checkpoint/model config or, in extract mode, replacing the 15
homework videos from an official full submission zip.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


TARGET_VARIANTS = {
    "bplus": {
        "checkpoint": "sam2.1_hiera_b+_MOSEv2_mss_lvt16.pt",
        "model_cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "pred": "pred_sam2_official_bplus_mosev2",
        "submit": "submission_433_official_bplus_mosev2",
        "zip": "submission_mosev2_official_bplus_15only.zip",
    },
    "large": {
        "checkpoint": "sam2.1_hiera_l_MOSEv2_mss_lvt16.pt",
        "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "pred": "pred_sam2_official_large_mosev2",
        "submit": "submission_433_official_large_mosev2",
        "zip": "submission_mosev2_official_large_15only.zip",
    },
}


OFFICIAL_SUBMISSION_ZIPS = {
    "bplus": "sam2_b+_MOSEv2_rcms_mqf_mss_lvt_submission.zip",
    "large": "sam2_l_MOSEv2_rcms_mqf_mss_lvt_submission.zip",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["infer", "extract-submission"], default="infer")
    p.add_argument("--variant", choices=sorted(TARGET_VARIANTS), default="bplus")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--sam2-root", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--model-cfg", default=None)
    p.add_argument("--official-submission-zip", type=Path, default=None)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--ann-root", type=Path, default=None)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--pred-root", type=Path, default=None)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--make-submission", action="store_true")
    p.add_argument("--overwrite-submission", action="store_true")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--audit-json", type=Path, default=None)
    return p.parse_args()


def list_target_videos(jpeg_root: Path, videos: list[str] | None) -> list[str]:
    if videos:
        return videos
    return sorted(p.name for p in jpeg_root.iterdir() if p.is_dir())


def default_paths(args: argparse.Namespace) -> argparse.Namespace:
    ws = args.workspace.resolve()
    spec = TARGET_VARIANTS[args.variant]
    args.workspace = ws
    args.sam2_root = (args.sam2_root or ws / "5_19" / "sam2").resolve()
    args.checkpoint_dir = (args.checkpoint_dir or ws / "homework" / "external_checkpoints" / "FudanCVL_MOSEv2_baseline").resolve()
    args.checkpoint = (args.checkpoint or args.checkpoint_dir / spec["checkpoint"]).resolve()
    args.model_cfg = args.model_cfg or spec["model_cfg"]
    args.jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    args.ann_root = (args.ann_root or ws / "homework" / "Annotations").resolve()
    args.provided_output_root = (args.provided_output_root or ws / "homework" / "output").resolve()
    args.pred_root = (args.pred_root or ws / "homework" / spec["pred"]).resolve()
    args.submit_root = (args.submit_root or ws / "homework" / spec["submit"]).resolve()
    args.zip_path = (args.zip_path or ws / "homework" / spec["zip"]).resolve()
    if args.official_submission_zip is None:
        args.official_submission_zip = args.checkpoint_dir / OFFICIAL_SUBMISSION_ZIPS[args.variant]
    args.official_submission_zip = args.official_submission_zip.resolve()
    return args


def safe_clear_dir(path: Path, required_name_prefix: str) -> None:
    if path.exists():
        if not path.name.startswith(required_name_prefix):
            raise RuntimeError(f"refusing to clear unsafe directory {path}; expected name prefix {required_name_prefix!r}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_infer(args: argparse.Namespace) -> None:
    here = Path(__file__).resolve().parent
    base_script = here / "infer_mosev2_sam2.py"
    cmd = [
        sys.executable,
        str(base_script),
        "--workspace",
        str(args.workspace),
        "--sam2-root",
        str(args.sam2_root),
        "--model-cfg",
        args.model_cfg,
        "--checkpoint",
        str(args.checkpoint),
        "--jpeg-root",
        str(args.jpeg_root),
        "--ann-root",
        str(args.ann_root),
        "--provided-output-root",
        str(args.provided_output_root),
        "--pred-root",
        str(args.pred_root),
        "--submit-root",
        str(args.submit_root),
        "--zip-path",
        str(args.zip_path),
        "--device",
        args.device,
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.make_submission:
        cmd.append("--make-submission")
    if args.overwrite_submission:
        cmd.append("--overwrite-submission")
    if args.no_zip:
        cmd.append("--no-zip")
    if args.videos:
        cmd.extend(["--videos", *args.videos])
    subprocess.run(cmd, check=True)


def extract_videos_from_zip(zip_path: Path, videos: Iterable[str], pred_root: Path, ann_root: Path) -> dict[str, int]:
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    safe_clear_dir(pred_root, "pred_")
    wanted = set(videos)
    counts = {v: 0 for v in wanted}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".png"):
                continue
            parts = Path(info.filename).parts
            if len(parts) < 2:
                continue
            video = parts[0]
            if video not in wanted:
                continue
            dst = pred_root / video / parts[-1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
            counts[video] += 1
    missing = [v for v, c in sorted(counts.items()) if c == 0]
    if missing:
        raise RuntimeError(f"official submission zip lacks target videos: {missing}")

    # The public full-submission zip was produced for its own evaluation layout;
    # for this homework candidate we still enforce the local first-frame GT
    # prompt exactly, matching all SAM2/DAM/SAAS adapters and enabling fair
    # first-frame-preservation audits.
    for video in wanted:
        ann = ann_root / video / "00000.png"
        if not ann.is_file():
            raise FileNotFoundError(ann)
        dst = pred_root / video / "00000.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ann, dst)
    return counts


def build_submission(args: argparse.Namespace) -> None:
    # Reuse the existing submission builder to preserve exactly the same copy/zip behavior.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from infer_mosev2_sam2 import make_submission

    ns = SimpleNamespace(
        provided_output_root=args.provided_output_root,
        pred_root=args.pred_root,
        submit_root=args.submit_root,
        zip_path=args.zip_path,
        overwrite_submission=args.overwrite_submission,
        no_zip=args.no_zip,
    )
    make_submission(ns)


def main() -> None:
    args = default_paths(parse_args())
    if args.mode == "infer":
        run_infer(args)
        return

    videos = list_target_videos(args.jpeg_root, args.videos)
    counts = extract_videos_from_zip(args.official_submission_zip, videos, args.pred_root, args.ann_root)
    if args.make_submission:
        build_submission(args)
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(
            json.dumps(
                {
                    "mode": args.mode,
                    "variant": args.variant,
                    "official_submission_zip": str(args.official_submission_zip),
                    "pred_root": str(args.pred_root),
                    "zip_path": str(args.zip_path),
                    "video_frame_counts": counts,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

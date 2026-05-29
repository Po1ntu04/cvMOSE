#!/usr/bin/env python3
"""Build a Codabench submission from provided outputs plus predicted 15-video outputs."""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--provided-output-root", type=Path, default=None)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-zip", action="store_true")
    p.add_argument("--expected-videos", type=int, default=433)
    return p.parse_args()


def copy_tree_contents(src_root: Path, submit_root: Path) -> None:
    if not src_root.is_dir():
        raise FileNotFoundError(src_root)
    for video_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        shutil.copytree(video_dir, submit_root / video_dir.name, dirs_exist_ok=True)


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    provided = (args.provided_output_root or ws / "homework" / "output").resolve()
    submit = (args.submit_root or ws / "homework" / "submission_433").resolve()
    zip_path = (args.zip_path or ws / "homework" / "submission_mosev2.zip").resolve()
    pred = args.pred_root.resolve()

    if submit.exists() and args.overwrite:
        shutil.rmtree(submit)
    submit.mkdir(parents=True, exist_ok=True)

    copy_tree_contents(provided, submit)
    copy_tree_contents(pred, submit)

    video_dirs = sorted(p for p in submit.iterdir() if p.is_dir())
    png_count = sum(1 for _ in submit.rglob("*.png"))
    print(f"submission_dir={submit} videos={len(video_dirs)} pngs={png_count}", flush=True)
    if len(video_dirs) != args.expected_videos:
        raise RuntimeError(f"expected {args.expected_videos} video dirs, got {len(video_dirs)}")

    if args.no_zip:
        return
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(submit.rglob("*.png")):
            zf.write(path, path.relative_to(submit).as_posix())
    print(f"zip_path={zip_path} size={zip_path.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Partition MOSEv2 videos and run per-video inference workers concurrently.

This is intentionally a launcher, not a new method. It accelerates experiments
while keeping model code versioned and output roots explicit.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Allow running from an uninstalled checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cvmose.video_partition import bucket_summary, discover_videos, greedy_balance  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--jpeg-root", type=Path, default=None)
    p.add_argument("--script", type=Path, default=REPO_ROOT / "tools" / "infer_mosev2_sam2.py")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--conda-env", default=None, help="Wrap worker commands with `conda run -n ENV`.")
    p.add_argument("--gpus", default="4", help="Comma-separated visible GPU ids, e.g. 0,1 or 4.")
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--pred-root", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--extra-arg", action="append", default=[], help="Extra arg string passed to each worker.")
    p.add_argument("--make-submission-after", action="store_true")
    p.add_argument("--submit-root", type=Path, default=None)
    p.add_argument("--zip-path", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = (args.jpeg_root or ws / "homework" / "JPEGImages").resolve()
    pred_root = args.pred_root.resolve()
    log_dir = (args.log_dir or ws / "homework" / "logs" / "parallel").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    slots = len(gpu_ids) * args.jobs_per_gpu
    items = discover_videos(jpeg_root, args.videos)
    buckets = [b for b in greedy_balance(items, slots) if b]
    print(json.dumps(bucket_summary(buckets), ensure_ascii=False, indent=2), flush=True)

    procs: list[tuple[subprocess.Popen[bytes], Path, list[str]]] = []
    for slot_idx, bucket in enumerate(buckets):
        gpu = gpu_ids[slot_idx % len(gpu_ids)]
        video_names = [v.name for v in bucket]
        cmd = [args.python, str(args.script), "--workspace", str(ws), "--pred-root", str(pred_root), "--videos", *video_names]
        if args.skip_existing:
            cmd.append("--skip-existing")
        for extra in args.extra_arg:
            cmd.extend(shlex.split(extra))
        if args.conda_env:
            cmd = ["conda", "run", "-n", args.conda_env, *cmd]
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = env.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = log_dir / f"worker_{slot_idx:02d}_gpu{gpu}.log"
        print(f"[worker {slot_idx}] gpu={gpu} videos={video_names} log={log_path}", flush=True)
        print(shell_join(cmd), flush=True)
        if args.dry_run:
            continue
        fh = log_path.open("wb")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        procs.append((proc, log_path, cmd))

    failed = []
    for proc, log_path, cmd in procs:
        code = proc.wait()
        if code != 0:
            failed.append((code, log_path, cmd))
    if failed:
        for code, log_path, cmd in failed:
            print(f"FAILED code={code} log={log_path} cmd={shell_join(cmd)}", file=sys.stderr)
        raise SystemExit(1)

    if args.dry_run:
        return

    if args.make_submission_after:
        build_cmd = [
            args.python,
            str(REPO_ROOT / "tools" / "build_submission.py"),
            "--workspace",
            str(ws),
            "--pred-root",
            str(pred_root),
            "--overwrite",
        ]
        if args.submit_root:
            build_cmd.extend(["--submit-root", str(args.submit_root.resolve())])
        if args.zip_path:
            build_cmd.extend(["--zip-path", str(args.zip_path.resolve())])
        if args.conda_env:
            build_cmd = ["conda", "run", "-n", args.conda_env, *build_cmd]
        print(shell_join(build_cmd), flush=True)
        subprocess.run(build_cmd, check=True)


if __name__ == "__main__":
    main()

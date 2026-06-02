#!/usr/bin/env python3
"""Precompute frozen DINOv2 masked descriptors for MOSEv2 label roots.

This is a diagnostic/cache utility for M6.  It does not train or finetune; it
only saves per-frame/per-object descriptor vectors and metadata as compressed
NPZ/JSON for later audit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.dino_descriptors import DINO_SPECS, DinoDescriptorExtractor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--mask-root", type=Path, required=True, help="Label PNG root, e.g. baseline or M11 pred root")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--variant", choices=["dino_vitb_reg", "dino_vitl_reg"], default="dino_vitb_reg")
    p.add_argument("--dino-root", type=Path, default=None)
    p.add_argument("--dino-weights", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-side", type=int, default=700)
    return p.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root = ws / "homework" / "JPEGImages"
    dino_root = (args.dino_root or ws / "external" / "dinov2").resolve()
    weights = (args.dino_weights or ws / "homework" / "external_checkpoints" / "dinov2" / DINO_SPECS[args.variant]["default_weight_name"]).resolve()
    videos = args.videos or sorted(p.name for p in jpeg_root.iterdir() if p.is_dir())
    extractor = DinoDescriptorExtractor(args.variant, dino_root, weights, device=args.device, max_side=args.max_side)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"variant": args.variant, "weights": str(weights), "videos": {}}
    for video in videos:
        rows = []
        vecs = []
        frame_paths = sorted([* (jpeg_root / video).glob("*.jpg"), * (jpeg_root / video).glob("*.jpeg"), * (jpeg_root / video).glob("*.png")])
        for idx, frame in enumerate(frame_paths):
            label_path = args.mask_root / video / f"{frame.stem}.png"
            if not label_path.is_file():
                continue
            rgb = load_rgb(frame)
            lab = load_label(label_path)
            for obj_id in [int(x) for x in np.unique(lab) if int(x) != 0]:
                res = extractor.describe(idx, rgb, lab == obj_id)
                if res is None:
                    continue
                vecs.append(res.vector.astype(np.float32))
                rows.append({
                    "frame_idx": idx,
                    "frame": frame.stem,
                    "obj_id": obj_id,
                    "tokens_inside": res.tokens_inside,
                    "source": res.source,
                    "fallback_reason": res.fallback_reason,
                    "crop_box": list(res.crop_box) if res.crop_box else None,
                })
        out_npz = args.out_dir / f"{video}.npz"
        if vecs:
            np.savez_compressed(out_npz, vectors=np.stack(vecs, axis=0), meta=json.dumps(rows, ensure_ascii=False))
        summary["videos"][video] = {"items": len(rows), "npz": str(out_npz) if vecs else None}
        extractor.clear()
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

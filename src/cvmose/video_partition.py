from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoItem:
    name: str
    frames: int


def discover_videos(jpeg_root: Path, names: list[str] | None = None) -> list[VideoItem]:
    if names:
        dirs = [jpeg_root / name for name in names]
    else:
        dirs = sorted(p for p in jpeg_root.iterdir() if p.is_dir())
    items: list[VideoItem] = []
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(d)
        n = len([*d.glob("*.jpg"), *d.glob("*.jpeg"), *d.glob("*.png")])
        if n <= 0:
            raise FileNotFoundError(f"no frames in {d}")
        items.append(VideoItem(d.name, n))
    return items


def greedy_balance(items: list[VideoItem], bins: int) -> list[list[VideoItem]]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    buckets: list[list[VideoItem]] = [[] for _ in range(bins)]
    loads = [0 for _ in range(bins)]
    for item in sorted(items, key=lambda x: x.frames, reverse=True):
        idx = min(range(bins), key=lambda i: loads[i])
        buckets[idx].append(item)
        loads[idx] += item.frames
    return buckets


def bucket_summary(buckets: list[list[VideoItem]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for i, bucket in enumerate(buckets):
        out.append(
            {
                "bucket": i,
                "videos": [v.name for v in bucket],
                "num_videos": len(bucket),
                "frames": sum(v.frames for v in bucket),
            }
        )
    return out

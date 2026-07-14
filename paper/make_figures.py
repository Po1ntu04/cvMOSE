#!/usr/bin/env python3
"""Generate paper-ready figures from the MOSEv2 frames and prediction roots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent
FIG = PAPER / "figures"
WORKSPACE = Path("/home/yu/projects/cv/from fdu/MOSEv2")
JPEG = WORKSPACE / "homework" / "JPEGImages"
ANN = WORKSPACE / "homework" / "Annotations"
BASE = WORKSPACE / "homework" / "pred_sam2_b101"
FINAL = WORKSPACE / "homework" / "pred_m17_q0_full_box"


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_rgb(video: str, idx: int) -> Image.Image:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = JPEG / video / f"{idx:05d}{suffix}"
        if path.is_file():
            return Image.open(path).convert("RGB")
    raise FileNotFoundError((video, idx))


def load_label(root: Path, video: str, idx: int) -> np.ndarray:
    return np.asarray(Image.open(root / video / f"{idx:05d}.png"))


def overlay(image: Image.Image, mask: np.ndarray, color=(30, 220, 80), alpha=0.52) -> Image.Image:
    rgb = np.asarray(image).astype(np.float32)
    m = np.asarray(mask, dtype=bool)
    if m.shape != rgb.shape[:2]:
        raise ValueError((m.shape, rgb.shape))
    color_arr = np.asarray(color, dtype=np.float32)
    rgb[m] = rgb[m] * (1.0 - alpha) + color_arr * alpha
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def fit(image: Image.Image, size=(520, 340)) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def titled(image: Image.Image, title: str, size=(520, 390)) -> Image.Image:
    body = fit(image, (size[0], size[1] - 48))
    canvas = Image.new("RGB", size, "white")
    canvas.paste(body, (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="black", font=font(24))
    return canvas


def grid(rows: list[list[Image.Image]], path: Path, gap=10) -> None:
    width = sum(im.width for im in rows[0]) + gap * (len(rows[0]) - 1)
    height = sum(row[0].height for row in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), (225, 225, 225))
    y = 0
    for row in rows:
        x = 0
        for im in row:
            canvas.paste(im, (x, y))
            x += im.width + gap
        y += row[0].height + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=93)


def first_mask(video: str, obj_id: int) -> np.ndarray:
    return load_label(ANN, video, 0) == obj_id


def make_challenge_montage() -> None:
    cases = [
        ("r13u5z4y", 1, 0, 20, "Occlusion + same-class slices"),
        ("amfdu83t", 1, 0, 8, "Exit/reappear + distractor"),
        ("8jsm23a7", 1, 0, 20, "Pickup + identity transfer"),
        ("4vznweiu", 1, 0, 20, "Tiny rotating semantic object"),
    ]
    rows = []
    for video, obj, first_idx, later_idx, label in cases:
        first = overlay(load_rgb(video, first_idx), first_mask(video, obj))
        later = load_rgb(video, later_idx)
        rows.append(
            [
                titled(first, f"{video} / first-mask"),
                titled(later, f"frame {later_idx:05d} / {label}"),
            ]
        )
    grid(rows, FIG / "challenge_montage.jpg")


def make_success_case(video: str, obj_id: int, frames: list[int], out_name: str) -> None:
    rows = []
    for idx in frames:
        rgb = load_rgb(video, idx)
        base = load_label(BASE, video, idx) == obj_id
        final = load_label(FINAL, video, idx) == obj_id
        rows.append(
            [
                titled(rgb, f"RGB {idx:05d}"),
                titled(overlay(rgb, base, (240, 80, 60)), "SAM2 baseline"),
                titled(overlay(rgb, final, (30, 210, 80)), "final re-anchor fusion"),
            ]
        )
    grid(rows, FIG / out_name)


def make_success_timeline(video: str, obj_id: int, frames: list[int], out_name: str) -> None:
    portrait_rows = []
    for idx in frames:
        rgb = load_rgb(video, idx)
        base = load_label(BASE, video, idx) == obj_id
        final = load_label(FINAL, video, idx) == obj_id
        rgb_panel = titled(rgb, f"RGB {idx:05d}", size=(360, 270))
        base_panel = titled(overlay(rgb, base, (240, 80, 60)), "SAM2 baseline", size=(360, 270))
        final_panel = titled(overlay(rgb, final, (30, 210, 80)), "final re-anchor", size=(360, 270))
        portrait_rows.append([rgb_panel, base_panel, final_panel])
    # Keep each temporal triplet intact so the sheet is readable in portrait PDF.
    grid(portrait_rows, FIG / out_name)


def make_failure_montage() -> None:
    cases = [
        ("8jsm23a7", 1, 20, "Mahjong identity unresolved"),
        ("r13u5z4y", 1, 20, "Strawberry reappearance unresolved"),
        ("4vznweiu", 1, 20, "Tiny rotating dice ambiguity"),
    ]
    rows = []
    for video, obj, idx, label in cases:
        rgb = load_rgb(video, idx)
        base = load_label(BASE, video, idx) == obj
        final = load_label(FINAL, video, idx) == obj
        rows.append(
            [
                titled(rgb, f"{video} RGB {idx:05d}"),
                titled(overlay(rgb, base, (240, 80, 60)), "baseline prediction"),
                titled(overlay(rgb, final, (30, 210, 80)), label),
            ]
        )
    grid(rows, FIG / "remaining_failures.jpg")


def make_score_plot() -> None:
    names = ["SAM2", "Reliability", "Multi-source", "Distractor memory", "Re-anchor"]
    scores = [46.0, 46.3, 46.8, 47.4, 50.1]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(names, scores, marker="o", linewidth=2.6, color="#145DA0")
    ax.fill_between(range(len(scores)), scores, [45.5] * len(scores), color="#B1D4E0", alpha=0.35)
    for i, score in enumerate(scores):
        ax.text(i, score + 0.012, f"{score:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("J&F' (%)")
    ax.set_xlabel("Method stage")
    ax.set_ylim(45.5, 50.7)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "score_progression.pdf", bbox_inches="tight")
    fig.savefig(FIG / "score_progression.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    make_score_plot()
    make_challenge_montage()
    make_success_case("amfdu83t", 1, [0, 8, 10, 11, 20], "amfdu_success.jpg")
    make_success_timeline(
        "q0sizv6m", 2, [0, 6, 8, 13, 27, 31, 34], "q0_success_timeline.jpg"
    )
    make_failure_montage()


if __name__ == "__main__":
    main()

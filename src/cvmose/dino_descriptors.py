"""DINOv2 masked object descriptors for training-free MOSEv2 re-anchoring.

The descriptor intentionally uses frozen public DINOv2 features only.  It does
not train or finetune: every object vector is masked pooling over patch tokens,
plus a context-ring contrast and a few shape scalars.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DINO_SPECS = {
    "dino_vitb_reg": {
        "hub_name": "dinov2_vitb14_reg",
        "dim": 768,
        "default_weight_name": "dinov2_vitb14_reg4_pretrain.pth",
    },
    "dino_vitl_reg": {
        "hub_name": "dinov2_vitl14_reg",
        "dim": 1024,
        "default_weight_name": "dinov2_vitl14_reg4_pretrain.pth",
    },
}


@dataclass(slots=True)
class DinoDescriptorResult:
    vector: np.ndarray
    tokens_inside: int
    source: str
    part_tokens: np.ndarray | None = None
    fallback_reason: str | None = None
    grid_size: tuple[int, int] | None = None
    crop_box: tuple[int, int, int, int] | None = None


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    denom = float(np.linalg.norm(vec))
    return vec / denom if denom > 1e-8 else vec


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expanded_crop_box(mask: np.ndarray, min_side: int, side_mult: float) -> tuple[int, int, int, int] | None:
    box = bbox_from_mask(mask)
    if box is None:
        return None
    h, w = mask.shape
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    side = int(max(min_side, math.ceil(side_mult * max(bw, bh))))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)
    return x1, y1, x2, y2


def ring_from_patch_mask(mask: "Any") -> "Any":
    import torch
    import torch.nn.functional as F

    m = mask.float()[None, None]
    dil = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)[0, 0] > 0.5
    ring = dil & (~mask)
    if bool(ring.any()):
        return ring
    dil = F.max_pool2d(m, kernel_size=5, stride=1, padding=2)[0, 0] > 0.5
    return dil & (~mask)


class DinoDescriptorExtractor:
    """Frozen DINOv2 patch-token extractor with per-frame cache."""

    def __init__(
        self,
        variant: str,
        dino_root: Path,
        weights: Path,
        device: str = "cuda",
        max_side: int = 700,
        tiny_min_tokens: int = 3,
        tiny_crop_min_side: int = 96,
        tiny_crop_mult: float = 6.0,
        part_topk: int = 8,
    ) -> None:
        if variant not in DINO_SPECS:
            raise ValueError(f"Unsupported DINO variant: {variant}")
        self.variant = variant
        self.spec = DINO_SPECS[variant]
        self.dino_root = Path(dino_root).resolve()
        self.weights = Path(weights).resolve()
        self.device = device
        self.max_side = int(max_side)
        self.tiny_min_tokens = int(tiny_min_tokens)
        self.tiny_crop_min_side = int(tiny_crop_min_side)
        self.tiny_crop_mult = float(tiny_crop_mult)
        self.part_topk = int(part_topk)
        self.patch = 14
        self.model: Any | None = None
        self.cache: dict[int, tuple[Any, int, int, float]] = {}

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model
        if not self.dino_root.exists():
            raise FileNotFoundError(self.dino_root)
        if not self.weights.exists():
            raise FileNotFoundError(self.weights)
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        # hubconf imports the local dinov2 package; make it importable without installing.
        if str(self.dino_root) not in sys.path:
            sys.path.insert(0, str(self.dino_root))
        import torch

        model = torch.hub.load(
            str(self.dino_root),
            self.spec["hub_name"],
            source="local",
            pretrained=True,
            weights=str(self.weights),
        )
        model.eval().to(self.device)
        self.model = model
        return model

    def _preprocess(self, rgb: np.ndarray) -> tuple[Any, int, int, float]:
        import torch
        import torch.nn.functional as F

        h, w = rgb.shape[:2]
        scale = min(1.0, float(self.max_side) / max(float(h), float(w))) if self.max_side > 0 else 1.0
        new_h = max(self.patch, int(round(h * scale)))
        new_w = max(self.patch, int(round(w * scale)))
        new_h = int(math.ceil(new_h / self.patch) * self.patch)
        new_w = int(math.ceil(new_w / self.patch) * self.patch)
        img = Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BICUBIC)
        arr = np.asarray(img).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(self.device)
        return x, new_h, new_w, scale

    def _frame_tokens(self, frame_idx: int, rgb: np.ndarray) -> tuple[Any, int, int, float]:
        if frame_idx in self.cache:
            return self.cache[frame_idx]
        import torch

        model = self._load_model()
        x, new_h, new_w, scale = self._preprocess(rgb)
        with torch.inference_mode():
            out = model.forward_features(x)
            tokens = out["x_norm_patchtokens"][0].detach().float()
        self.cache[frame_idx] = (tokens, new_h, new_w, scale)
        return tokens, new_h, new_w, scale

    def _mask_to_patch_grid(self, mask: np.ndarray, new_h: int, new_w: int) -> Any:
        import torch
        import torch.nn.functional as F

        m = torch.from_numpy(mask.astype(np.float32))[None, None].to(self.device)
        m = F.interpolate(m, size=(new_h, new_w), mode="nearest")
        gh, gw = new_h // self.patch, new_w // self.patch
        pooled = F.max_pool2d(m, kernel_size=self.patch, stride=self.patch)[0, 0] > 0.5
        return pooled.reshape(gh, gw)

    def _select_part_tokens(self, fg_tokens: np.ndarray) -> np.ndarray | None:
        """Return a small diverse set of normalized foreground patch tokens.

        Mean pooling is brittle for MOSEv2 tiny/occluded targets because the
        identity signal may be a few visible parts.  A deterministic farthest
        point subset preserves those local parts without changing the frozen
        DINO model or training anything.
        """
        if fg_tokens.size == 0 or self.part_topk <= 0:
            return None
        parts = np.stack([l2_normalize(x) for x in fg_tokens.astype(np.float32)], axis=0)
        if parts.shape[0] <= self.part_topk:
            return parts
        mean = l2_normalize(parts.mean(axis=0))
        sims = parts @ mean
        selected = [int(np.argmax(sims))]
        min_dist = 1.0 - (parts @ parts[selected[0]])
        while len(selected) < self.part_topk:
            idx = int(np.argmax(min_dist))
            if idx in selected:
                break
            selected.append(idx)
            min_dist = np.minimum(min_dist, 1.0 - (parts @ parts[idx]))
        return parts[selected]

    def _describe_tokens(self, tokens: Any, patch_mask: Any, grid: tuple[int, int], shape_vec: np.ndarray, source: str, fallback: str | None, crop_box: tuple[int, int, int, int] | None) -> DinoDescriptorResult | None:
        import torch

        flat_mask = patch_mask.reshape(-1)
        tokens_inside = int(flat_mask.sum().item())
        if tokens_inside <= 0:
            return None
        obj = tokens[flat_mask].mean(dim=0)
        ring = ring_from_patch_mask(patch_mask).reshape(-1)
        if bool(ring.any()):
            ctx = tokens[ring].mean(dim=0)
            diff = obj - ctx
        else:
            ctx = torch.zeros_like(obj)
            diff = obj
        obj_np = obj.detach().float().cpu().numpy()
        diff_np = diff.detach().float().cpu().numpy()
        part_tokens = self._select_part_tokens(tokens[flat_mask].detach().float().cpu().numpy())
        vec = np.concatenate([l2_normalize(obj_np), l2_normalize(diff_np), l2_normalize(shape_vec.astype(np.float32))])
        return DinoDescriptorResult(
            vector=l2_normalize(vec),
            tokens_inside=tokens_inside,
            source=source,
            part_tokens=part_tokens,
            fallback_reason=fallback,
            grid_size=grid,
            crop_box=crop_box,
        )

    def _shape_vec(self, mask: np.ndarray, crop_box: tuple[int, int, int, int] | None = None, original_shape: tuple[int, int] | None = None) -> np.ndarray:
        h, w = original_shape or mask.shape
        box = bbox_from_mask(mask)
        if box is None:
            return np.zeros(8, dtype=np.float32)
        x1, y1, x2, y2 = box
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        crop_area = 0 if crop_box is None else max(1, (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1]))
        return np.asarray(
            [
                math.log1p(float(mask.sum())) / 12.0,
                float(mask.sum()) / max(float(h * w), 1.0),
                bw / max(float(w), 1.0),
                bh / max(float(h), 1.0),
                min(4.0, bw / max(float(bh), 1.0)) / 4.0,
                min(4.0, bh / max(float(bw), 1.0)) / 4.0,
                0.0 if crop_box is None else float(mask.sum()) / float(crop_area),
                1.0 if crop_box is not None else 0.0,
            ],
            dtype=np.float32,
        )

    def describe(self, frame_idx: int, rgb: np.ndarray, mask: np.ndarray) -> DinoDescriptorResult | None:
        if mask.sum() <= 0:
            return None
        tokens, new_h, new_w, _ = self._frame_tokens(frame_idx, rgb)
        patch_mask = self._mask_to_patch_grid(mask, new_h, new_w)
        shape_vec = self._shape_vec(mask)
        result = self._describe_tokens(
            tokens,
            patch_mask,
            (new_h // self.patch, new_w // self.patch),
            shape_vec,
            source=self.variant,
            fallback=None,
            crop_box=None,
        )
        if result is not None and result.tokens_inside >= self.tiny_min_tokens:
            return result

        # Tiny objects may vanish on the full-frame patch grid.  Re-encode a
        # local crop so the object occupies several DINO patches, but keep the
        # global shape scalars for context.
        box = expanded_crop_box(mask, self.tiny_crop_min_side, self.tiny_crop_mult)
        if box is None:
            return result
        x1, y1, x2, y2 = box
        crop_rgb = rgb[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        # Do not cache crop tokens under frame_idx; crops are candidate-specific.
        x, ch, cw, _ = self._preprocess(crop_rgb)
        model = self._load_model()
        import torch
        with torch.inference_mode():
            out = model.forward_features(x)
            crop_tokens = out["x_norm_patchtokens"][0].detach().float()
        crop_patch_mask = self._mask_to_patch_grid(crop_mask, ch, cw)
        crop_result = self._describe_tokens(
            crop_tokens,
            crop_patch_mask,
            (ch // self.patch, cw // self.patch),
            self._shape_vec(mask, crop_box=box, original_shape=mask.shape),
            source=f"{self.variant}_crop",
            fallback="full_frame_tokens_below_min" if result is not None else "full_frame_no_tokens",
            crop_box=box,
        )
        return crop_result or result

    def clear(self) -> None:
        self.cache.clear()

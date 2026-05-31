#!/usr/bin/env python3
"""Split-frame Qwen3.6 event-story harness.

This tool fixes the timeout-prone "one dense multi-frame panel -> one large
Qwen3.6 vision request" pattern used by the first M12 event-story harness.

Pipeline:
1. Render a tiny frame-level panel for each sampled frame.
2. Ask qwen3.6-plus for a short frame observation JSON.
3. Aggregate the short JSON observations with a text-only qwen3.6-plus call
   into the same event-story / teach-SAM contract.

The goal is not to produce final masks.  It produces lower-latency visual
evidence and a SAM prompt plan that downstream bounded propagation can consume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.mllm_panels import (  # noqa: E402
    BLUE,
    GREEN,
    RED,
    YELLOW,
    bbox_from_mask,
    crop_mask,
    crop_pil,
    draw_bbox,
    expand_box,
    first_annotation,
    homework_roots,
    list_frames,
    load_rgb,
    outline_mask,
)
from cvmose.qwen_vl_client import DEFAULT_MODEL, QwenVLClient  # noqa: E402


VIDEOS15 = [
    "1qlssuz2",
    "q0sizv6m",
    "lcgc29va",
    "z6dx46qr",
    "4vznweiu",
    "amfdu83t",
    "4f98052b",
    "3epdtmyr",
    "msinig6m",
    "c8lutf29",
    "2smf7uq9",
    "8jsm23a7",
    "jadgtmfl",
    "r13u5z4y",
    "pe0d85lk",
]

DEFAULT_ROOTS = {
    "m7": "pred_m7_part_balanced",
    "official_l": "pred_sam2_official_large_mosev2",
    "official_b": "pred_sam2_official_bplus_mosev2",
    "alt": "pred_sam2_m7_qwen_support_tiny_semantic",
}

DEFAULT_HUMAN_HINTS: dict[str, dict[str, str]] = {
    "8jsm23a7:1": {
        "hint": "目标是一张被拿起的麻将。第二帧附近被手遮挡，然后被拿到玩家面前牌列；直到结尾它应是面前麻将最左侧的七条。首帧背面朝上，与中间待摸牌极像，外观匹配会误导。"
    },
    "z6dx46qr:1": {"hint": "这是低对比水下难样本；可辨别线索可能主要是两个像眼睛的黑点，而不是明显物体轮廓。"},
    "q0sizv6m:2": {"hint": "首帧靠后的 obj2 仓鼠/豚鼠状动物后续走到镜头前，成为画面下方/前景的新动物；原位置附近的同类动物应视为强干扰。"},
}


@dataclass(frozen=True)
class Target:
    video: str
    obj_id: int


def font(size: int = 15) -> ImageFont.ImageFont:
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def load_label(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr


def frame_mask(workspace: Path, root_name: str, video: str, frame_stem: str, obj_id: int) -> np.ndarray | None:
    arr = load_label(workspace / "homework" / root_name / video / f"{frame_stem}.png")
    if arr is None:
        return None
    return arr == int(obj_id)


def mask_area(mask: np.ndarray | None) -> int:
    return 0 if mask is None else int(np.asarray(mask).astype(bool).sum())


def parse_targets(items: list[str] | None) -> set[tuple[str, int]] | None:
    if not items:
        return None
    out: set[tuple[str, int]] = set()
    for item in items:
        video, obj = item.split(":", 1)
        out.add((video, int(obj)))
    return out


def discover_targets(workspace: Path, videos: list[str], target_filter: set[tuple[str, int]] | None) -> list[Target]:
    _, ann_root = homework_roots(workspace)
    targets: list[Target] = []
    for video in videos:
        try:
            _, ann = first_annotation(ann_root, video)
        except Exception:
            continue
        for obj_id in [int(x) for x in np.unique(ann) if int(x) != 0]:
            if target_filter and (video, obj_id) not in target_filter:
                continue
            targets.append(Target(video, obj_id))
    return targets


def choose_frames(num_frames: int, extra: list[int] | None = None, max_frames: int = 4) -> list[int]:
    base = [0, 1, 2, num_frames - 1]
    if extra:
        base.extend(extra)
    out: list[int] = []
    for idx in base:
        idx = max(0, min(num_frames - 1, int(idx)))
        if idx not in out:
            out.append(idx)
    if len(out) <= max_frames:
        return out
    keep = [out[0], out[1], out[-1]]
    middle = [x for x in out[2:-1] if x not in keep]
    need = max_frames - len(keep)
    if need > 0 and middle:
        picks = np.linspace(0, len(middle) - 1, need).round().astype(int).tolist()
        keep[2:2] = [middle[i] for i in picks]
    return sorted(dict.fromkeys(keep))[:max_frames]


def load_json_map(path: Path | None) -> dict[str, list[int]]:
    if not path or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): [int(x) for x in v] for k, v in raw.items()}


def load_hints(path: Path | None, use_builtin: bool) -> dict[str, dict[str, str]]:
    hints = dict(DEFAULT_HUMAN_HINTS) if use_builtin else {}
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            hints[str(key)] = {"hint": str(value)} if isinstance(value, str) else {str(k): str(v) for k, v in value.items()}
    return hints


def paste_fit(canvas: Image.Image, img: Image.Image, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    img = img.convert("RGB")
    scale = min(cw / img.width, ch / img.height)
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(resized, (x1 + (cw - nw) // 2, y1 + (ch - nh) // 2))


def add_top_caption(img: Image.Image, text: str, height: int = 26) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img.convert("RGB"), (0, height))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, img.width, height], fill=(0, 0, 0))
    d.text((5, 5), text[:170], fill=(255, 215, 0), font=font(13))
    return out


def union_box(masks: list[np.ndarray | None], shape: tuple[int, int], fallback: list[int] | None) -> list[int]:
    boxes = [bbox_from_mask(m) for m in masks if m is not None]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return expand_box(fallback, shape, pad=1.2, square=True, min_side=160)
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return expand_box([x1, y1, x2, y2], shape, pad=1.1, square=True, min_side=180)


def norm_box_1000(box: list[int] | None, size: tuple[int, int]) -> list[int] | None:
    if box is None:
        return None
    w, h = size
    x1, y1, x2, y2 = box
    return [round(1000 * x1 / max(1, w)), round(1000 * y1 / max(1, h)), round(1000 * x2 / max(1, w)), round(1000 * y2 / max(1, h))]


def render_frame_panel(
    *,
    workspace: Path,
    target: Target,
    frame_idx: int,
    out_path: Path,
    roots: dict[str, str],
    compact: bool = True,
) -> dict[str, Any]:
    jpeg_root, ann_root = homework_roots(workspace)
    frames = list_frames(jpeg_root, target.video)
    _, ann = first_annotation(ann_root, target.video)
    rgb0 = load_rgb(frames[0])
    rgb = load_rgb(frames[frame_idx])
    ref_mask = ann == int(target.obj_id)
    ref_box = bbox_from_mask(ref_mask)
    ref_crop_box = expand_box(ref_box, ann.shape, pad=1.0, square=True, min_side=140)
    ref_crop = crop_pil(rgb0, ref_crop_box)
    ref_over = outline_mask(ref_crop, crop_mask(ref_mask, ref_crop_box), BLUE, width=3)

    stem = frames[frame_idx].stem
    masks = {
        name: frame_mask(workspace, root, target.video, stem, target.obj_id)
        for name, root in roots.items()
    }
    colors = {"m7": BLUE, "official_l": YELLOW, "official_b": RED, "alt": GREEN}
    wide = rgb.copy()
    for name, mask in masks.items():
        if mask is not None and mask.any():
            wide = outline_mask(wide, mask, colors.get(name, GREEN), width=3)
            wide = draw_bbox(wide, bbox_from_mask(mask), colors.get(name, GREEN), width=3, label=name)
    crop_box = union_box(list(masks.values()), (rgb.height, rgb.width), ref_box)
    cur_crop = crop_pil(rgb, crop_box)
    cur_over = cur_crop.copy()
    for name, mask in masks.items():
        if mask is not None and mask.any():
            cur_over = outline_mask(cur_over, crop_mask(mask, crop_box), colors.get(name, GREEN), width=3)

    full0 = draw_bbox(outline_mask(rgb0, ref_mask, BLUE, width=3), ref_box, YELLOW, width=3, label="REF")
    if compact:
        # Compact mode is the default because qwen3.6-plus repeatedly timed out
        # on dense multi-frame panels.  Keep only the minimum evidence needed:
        # reference crop, current wide context, and current local candidates.
        canvas = Image.new("RGB", (720, 430), "white")
        cells = [
            (add_top_caption(ref_over, "REF crop; blue outline artificial"), (0, 0, 240, 205)),
            (add_top_caption(wide, f"current f={frame_idx:05d}; candidate outlines"), (240, 0, 720, 205)),
            (add_top_caption(cur_over, "current local crop + candidates"), (0, 205, 720, 430)),
        ]
    else:
        canvas = Image.new("RGB", (900, 600), "white")
        cells = [
            (add_top_caption(full0, "frame0 context: REF target"), (0, 0, 300, 285)),
            (add_top_caption(ref_over, "REF crop; blue outline artificial"), (300, 0, 600, 285)),
            (add_top_caption(wide, f"current f={frame_idx:05d}; artificial candidate outlines"), (600, 0, 900, 285)),
            (add_top_caption(cur_crop, "current local raw crop"), (0, 285, 450, 600)),
            (add_top_caption(cur_over, "current local crop + candidate outlines"), (450, 285, 900, 600)),
        ]
    for img, box in cells:
        paste_fit(canvas, img, box)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, canvas.height - 24, canvas.width, canvas.height - 1], fill=(255, 255, 255))
    d.text((6, canvas.height - 20), "Legend: BLUE=M7, YELLOW=official_l, RED=official_b, GREEN=alt. Artificial annotations; judge physical instance.", fill=(0, 0, 0), font=font(12))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=88)
    return {
        "video": target.video,
        "obj_id": int(target.obj_id),
        "frame_idx": int(frame_idx),
        "frame_path": str(frames[frame_idx]),
        "panel_path": str(out_path),
        "ref_box_norm_1000": norm_box_1000(ref_box, rgb0.size),
        "crop_box_norm_1000": norm_box_1000(crop_box, rgb.size),
        "areas": {name: mask_area(mask) for name, mask in masks.items()},
        "boxes_norm_1000": {name: norm_box_1000(bbox_from_mask(mask), rgb.size) for name, mask in masks.items()},
    }


def render_mini_retry_panel(
    *,
    workspace: Path,
    target: Target,
    frame_idx: int,
    out_path: Path,
    roots: dict[str, str],
) -> dict[str, Any]:
    """Render an even smaller crop-only panel for qwen3.6 timeout retries."""
    jpeg_root, ann_root = homework_roots(workspace)
    frames = list_frames(jpeg_root, target.video)
    _, ann = first_annotation(ann_root, target.video)
    rgb0 = load_rgb(frames[0])
    rgb = load_rgb(frames[frame_idx])
    ref_mask = ann == int(target.obj_id)
    ref_box = bbox_from_mask(ref_mask)
    ref_crop_box = expand_box(ref_box, ann.shape, pad=1.1, square=True, min_side=140)
    ref_over = outline_mask(crop_pil(rgb0, ref_crop_box), crop_mask(ref_mask, ref_crop_box), BLUE, width=3)
    stem = frames[frame_idx].stem
    masks = {name: frame_mask(workspace, root, target.video, stem, target.obj_id) for name, root in roots.items()}
    colors = {"m7": BLUE, "official_l": YELLOW, "official_b": RED, "alt": GREEN}
    crop_box = union_box(list(masks.values()), (rgb.height, rgb.width), ref_box)
    cur_raw = crop_pil(rgb, crop_box)
    cur_over = cur_raw.copy()
    for name, mask in masks.items():
        if mask is not None and mask.any():
            cur_over = outline_mask(cur_over, crop_mask(mask, crop_box), colors.get(name, GREEN), width=3)
    canvas = Image.new("RGB", (640, 330), "white")
    cells = [
        (add_top_caption(ref_over, "REF exact target"), (0, 0, 210, 300)),
        (add_top_caption(cur_raw, f"current f={frame_idx:05d} raw"), (210, 0, 425, 300)),
        (add_top_caption(cur_over, "current + candidates"), (425, 0, 640, 300)),
    ]
    for img, box in cells:
        paste_fit(canvas, img, box)
    d = ImageDraw.Draw(canvas)
    d.text((4, 307), "Artificial outlines: BLUE=M7, YELLOW=official_l, RED=official_b, GREEN=alt.", fill=(0, 0, 0), font=font(12))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=82)
    return {
        "areas": {name: mask_area(mask) for name, mask in masks.items()},
        "boxes_norm_1000": {name: norm_box_1000(bbox_from_mask(mask), rgb.size) for name, mask in masks.items()},
        "ref_box_norm_1000": norm_box_1000(ref_box, rgb0.size),
        "crop_box_norm_1000": norm_box_1000(crop_box, rgb.size),
    }


def compact_frame_prompt(target: Target, frame_idx: int, hint: str, meta: dict[str, Any]) -> str:
    """Short retry prompt used when the richer frame observer times out."""
    return f"""
Return strict JSON only. Briefly describe this visual panel; do not solve the whole tracking task.
video={target.video}, obj_id={target.obj_id}, frame={frame_idx}.
The panel contains a REF crop and a current-frame crop with artificial colored outlines.
Return exactly:
{{"status":"ok","caption":"<=25 words visual description","target_visible":"uncertain","confidence":0.3}}
""".strip()


def frame_caption_prompt(target: Target, frame_idx: int, hint: str) -> str:
    """Minimal visual request that keeps qwen3.6-plus out of long reasoning mode."""
    return f"""
Return strict JSON only.
video={target.video}, obj_id={target.obj_id}, frame={frame_idx}.
The left crop is REF, the exact target instance. The current frame/crop may contain artificial outlines:
BLUE=M7, YELLOW=official_l, RED=official_b, GREEN=alt. Ignore colors as object appearance.
Human hint: {hint or 'none'}.

Only describe visible evidence; do not solve the whole video.
JSON:
{{"status":"ok","caption":"what is visible and where the REF-like physical instance may be","target_visible":"yes/no/partial/uncertain","candidate_hint":"which outlined region, if any, appears plausible or wrong","confidence":0.0}}
""".strip()


def frame_system_prompt() -> str:
    return """
You are a strict video-object frame observer. Compare the REF physical instance
with the current frame. Do not segment. Do not choose an object merely because
it has the same class. Artificial outline colors are not object appearance.
Return strict JSON only.
""".strip()


def frame_user_prompt(target: Target, frame_idx: int, hint: str, meta: dict[str, Any]) -> str:
    return f"""
video={target.video}, obj_id={target.obj_id}, current_frame={frame_idx}.
Human hint: {hint or 'none'}.
REF crop is the exact physical target. Current panel uses artificial outlines:
BLUE=M7, YELLOW=official_l, RED=official_b, GREEN=alt.

Judge only this frame. Return compact JSON:
{{
  "status": "ok",
  "target_visible": "yes|no|partial|uncertain",
  "observation": "one short sentence about the physical instance in this frame",
  "likely_target_location": "short location or null",
  "positive_box_norm_1000": [x1,y1,x2,y2] | null,
  "candidate_roles": {{"m7":"same_instance|hard_negative|uncertain|empty","official_l":"same_instance|hard_negative|uncertain|empty","official_b":"same_instance|hard_negative|uncertain|empty","alt":"same_instance|hard_negative|uncertain|empty"}},
  "hard_negatives": ["short descriptions"],
  "identity_cues": ["short cues"],
  "confidence": 0.0
}}
If uncertain, say uncertain; do not overthink.
""".strip()


def aggregate_system_prompt() -> str:
    return """
你是视频目标分割中的物理实例事件链分析器和 SAM 提示教师。你收到的是多个小图请求得到的逐帧视觉观察 JSON，而不是 GT。请把它们整合成事件链、hard negatives 和 teach-SAM 计划。

必须保守：MLLM 不直接输出最终 mask；它只输出 prompt/negative/propagation plan。若证据不足要标 uncertain。输出严格 JSON。
""".strip()


def aggregate_user_prompt(target: Target, hint: str, frame_observations: list[dict[str, Any]]) -> str:
    return f"""
视频={target.video}，obj_id={target.obj_id}。
人类观察提示（可辅助但不要盲从）：{hint or '无'}

逐帧视觉观察：
{json.dumps(frame_observations, ensure_ascii=False, indent=2)}

请返回严格 JSON：
{{
  "video": "{target.video}",
  "obj_id": {target.obj_id},
  "status": "ok|uncertain|insufficient_visual_evidence",
  "target_summary": "一句话说明首帧目标物理实例",
  "event_type": "static|picked_moved_placed|moves_to_foreground|exits_reappears|low_contrast_continuation|multi_object_contact|camera_motion|unknown",
  "event_chain": [
    {{"frame_range": "0-2", "description": "发生了什么", "target_visible": "yes|no|partial|uncertain", "confidence": 0.0}}
  ],
  "current_prediction_diagnosis": "mostly_correct|wrong_static_distractor|wrong_same_class|empty_when_visible|oversegmented_composite|undersegmented|uncertain",
  "identity_cues": ["可用于重识别的线索"],
  "hard_negatives": [
    {{"frame_idx": 0, "description": "外观相似但不是目标的区域/物体", "reason": "为什么是负例"}}
  ],
  "positive_prompt_plan": [
    {{"frame_idx": 0, "location_description": "应给 SAM 提示的位置", "box_norm_1000": [x1,y1,x2,y2], "prompt_type": "box|point|mask_from_candidate|empty", "confidence": 0.0, "rationale": "理由"}}
  ],
  "sam_teaching_plan": {{
    "strategy": "keep_current|empty_interval_probe|event_reanchor|official_interval_probe|mlmm_box_to_sam|needs_manual_review",
    "positive_frames": [0],
    "negative_frame_roles": ["original_position_distractor"],
    "bounded_propagation": "none|forward|backward|local_window",
    "merge_policy": "keep_m7_default|replace_object_interval|output_empty_until_reanchor|manual_probe_only"
  }},
  "recommended_action": "keep_current|empty_after_event|reanchor_at_frame|use_official_interval|generate_detector_candidates|manual_review",
  "confidence": 0.0,
  "failure_if_wrong": "若判断错了，最可能错在哪里"
}}
""".strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--videos", nargs="*", default=VIDEOS15)
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--frames-json", type=Path, default=None)
    p.add_argument("--hints-json", type=Path, default=None)
    p.add_argument("--no-builtin-hints", action="store_true")
    p.add_argument("--out-json", type=Path, default=Path("artifacts/m12_event_story_split/split_records.json"))
    p.add_argument("--out-panel-dir", type=Path, default=Path("artifacts/m12_event_story_split/panels"))
    p.add_argument("--out-doc", type=Path, default=Path("docs/m12_qwen36_split_event_story.md"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/m12_event_story_split/cache"))
    p.add_argument("--model", default=os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL))
    p.add_argument("--allow-fallback", action="store_true", help="By default split mode tests qwen3.6-plus directly with no fallback.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-calls", type=int, default=8, help="Maximum target objects.")
    p.add_argument("--max-frames", type=int, default=4)
    p.add_argument("--max-side", type=int, default=700)
    p.add_argument("--jpeg-quality", type=int, default=70)
    p.add_argument("--frame-max-tokens", type=int, default=768)
    p.add_argument("--aggregate-max-tokens", type=int, default=2048)
    p.add_argument("--frame-timeout", type=float, default=30.0, help="Short timeout so a hard visual frame falls back quickly.")
    p.add_argument("--retry-timeout", type=float, default=90.0, help="Longer timeout for ultra-simple frame retry prompts.")
    p.add_argument("--aggregate-timeout", type=float, default=180.0, help="Longer timeout for text-only qwen3.6 aggregation.")
    p.add_argument("--panel-mode", choices=["compact", "standard"], default="compact")
    p.add_argument("--frame-task", choices=["caption", "observer"], default="caption", help="caption avoids qwen3.6-plus image reasoning timeouts; observer asks richer per-frame JSON.")
    p.add_argument("--frame-retries", type=int, default=1, help="Extra qwen3.6 attempts for timed-out frame observations using a shorter prompt.")
    return p.parse_args()


def is_usable_status(status: Any) -> bool:
    return str(status) in {"ok", "extracted_first_object"}


def normalize_frame_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Accept both rich and compact frame-observer schemas."""
    if "caption" in obs and "physical_event_observation" not in obs:
        obs["physical_event_observation"] = obs.get("caption")
    if "candidate_hint" in obs and "likely_target_location" not in obs:
        obs["likely_target_location"] = obs.get("candidate_hint")
    if "observation" in obs and "physical_event_observation" not in obs:
        obs["physical_event_observation"] = obs.get("observation")
    if "hard_negatives" in obs and "hard_negative_descriptions" not in obs:
        obs["hard_negative_descriptions"] = obs.get("hard_negatives")
    if "identity_cues" in obs and "identity_cues_seen" not in obs:
        obs["identity_cues_seen"] = obs.get("identity_cues")
    return obs


def call_frame_observation(
    *,
    client: QwenVLClient,
    retry_client: QwenVLClient,
    args: argparse.Namespace,
    workspace: Path,
    target: Target,
    idx: int,
    hint: str,
    panel_path: Path,
    meta: dict[str, Any],
) -> dict[str, Any]:
    if args.frame_task == "caption":
        obs = client.call_json(
            system_prompt="You are a concise visual captioner for video object tracking. Return strict JSON only.",
            user_text=frame_caption_prompt(target, idx, hint),
            image_paths=[panel_path],
            schema_name="split_frame_caption",
            metadata={"video": target.video, "obj_id": target.obj_id, "frame_idx": idx, "hint": hint, "attempt": "caption"},
            max_tokens=min(int(args.frame_max_tokens), 256),
        )
        normalize_frame_observation(obs)
        if is_usable_status(obs.get("status")):
            return obs
        # Fall through directly to the ultra-compact retry; do not run the
        # richer observer prompt because that is the timeout-prone path.
    else:
        obs = client.call_json(
            system_prompt=frame_system_prompt(),
            user_text=frame_user_prompt(target, idx, hint, meta),
            image_paths=[panel_path],
            schema_name="split_frame_observation",
            metadata={"video": target.video, "obj_id": target.obj_id, "frame_idx": idx, "hint": hint, "attempt": "rich"},
            max_tokens=args.frame_max_tokens,
        )
        normalize_frame_observation(obs)
        if is_usable_status(obs.get("status")):
            return obs
    # Retry with a very short prompt.  The cache key differs because the prompt
    # and image differ, so this avoids being pinned by an earlier timeout cache
    # record and removes most dense context from the visual request.
    last = obs
    for attempt in range(1, max(0, int(args.frame_retries)) + 1):
        retry_panel_path = panel_path.with_name(f"{panel_path.stem}_mini_retry{attempt}.jpg")
        retry_meta = render_mini_retry_panel(workspace=workspace, target=target, frame_idx=idx, out_path=retry_panel_path, roots=DEFAULT_ROOTS)
        retry = retry_client.call_json(
            system_prompt="You are a concise visual observer. Return strict JSON only.",
            user_text=compact_frame_prompt(target, idx, hint, retry_meta),
            image_paths=[retry_panel_path],
            schema_name="split_frame_observation_retry",
            metadata={"video": target.video, "obj_id": target.obj_id, "frame_idx": idx, "hint": hint, "attempt": f"compact_retry_{attempt}"},
            max_tokens=min(int(args.frame_max_tokens), 512),
        )
        normalize_frame_observation(retry)
        retry.setdefault("previous_error_status", last.get("status"))
        if is_usable_status(retry.get("status")):
            retry["recovered_by_retry"] = True
            return retry
        last = retry
    return last


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    rows = payload.get("records", [])
    lines = [
        "# M12 qwen3.6 split-frame event-story report",
        "",
        "**Purpose:** avoid qwen3.6-plus multi-frame panel timeouts by splitting visual evidence into small frame observations, then aggregating text-only.",
        "",
        f"- model: `{payload.get('model')}`",
        f"- allow_fallback: `{payload.get('allow_fallback')}`",
        f"- dry_run: `{payload.get('dry_run')}`",
        f"- targets: `{len(rows)}`",
        "",
        "| video | obj | frame calls ok/api_error | aggregate status | event | diagnosis | action | conf |",
        "| --- | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for rec in rows:
        frame_status = rec.get("frame_status_counts", {})
        j = rec.get("judgment", {}) or {}
        ok_err = f"{frame_status.get('ok', 0)}/{frame_status.get('api_error', 0)}"
        lines.append(
            f"| `{rec.get('video')}` | {rec.get('obj_id')} | {ok_err} | {j.get('status','?')} | {j.get('event_type','?')} | {j.get('current_prediction_diagnosis','?')} | {j.get('recommended_action','?')} | {float(j.get('confidence') or 0):.2f} |"
        )
    lines += [
        "",
        "## Contract",
        "",
        "- Frame observations are short visual captions, not final masks.",
        "- Aggregation is text-only qwen3.6 reasoning over frame JSON.",
        "- Positive boxes still need SAM refinement and bounded propagation before any submission use.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    hints = load_hints(args.hints_json, not args.no_builtin_hints)
    frame_overrides = load_json_map(args.frames_json)
    target_filter = parse_targets(args.targets)
    targets = discover_targets(args.workspace, args.videos, target_filter)
    fallback_models = None if args.allow_fallback else []
    frame_client = QwenVLClient(
        model=args.model,
        fallback_models=fallback_models,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        timeout=args.frame_timeout,
    )
    retry_client = QwenVLClient(
        model=args.model,
        fallback_models=fallback_models,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        timeout=args.retry_timeout,
    )
    aggregate_client = QwenVLClient(
        model=args.model,
        fallback_models=fallback_models,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        timeout=args.aggregate_timeout,
    )
    jpeg_root, _ = homework_roots(args.workspace)
    records: list[dict[str, Any]] = []
    for target in targets[: args.max_calls]:
        frames = list_frames(jpeg_root, target.video)
        key = f"{target.video}:{target.obj_id}"
        idxs = choose_frames(len(frames), extra=frame_overrides.get(key) or frame_overrides.get(target.video), max_frames=args.max_frames)
        hint = hints.get(key, {}).get("hint", "")
        frame_records: list[dict[str, Any]] = []
        for idx in idxs:
            panel_path = args.out_panel_dir / target.video / f"obj{target.obj_id}_f{idx:05d}.jpg"
            meta = render_frame_panel(workspace=args.workspace, target=target, frame_idx=idx, out_path=panel_path, roots=DEFAULT_ROOTS, compact=args.panel_mode == "compact")
            obs = call_frame_observation(client=frame_client, retry_client=retry_client, args=args, workspace=args.workspace, target=target, idx=idx, hint=hint, panel_path=panel_path, meta=meta)
            frame_records.append({"frame_idx": idx, "panel_path": str(panel_path), "panel_meta": meta, "observation": obs})
            print(f"frame {target.video}:{target.obj_id}@{idx} {obs.get('status')} model={obs.get('model_used')} conf={obs.get('confidence')}", flush=True)
        observation_compact = [
            {
                "frame_idx": fr["frame_idx"],
                "status": fr["observation"].get("status"),
                "target_visible": fr["observation"].get("target_visible"),
                "physical_event_observation": fr["observation"].get("physical_event_observation"),
                "likely_target_location": fr["observation"].get("likely_target_location"),
                "positive_box_norm_1000": fr["observation"].get("positive_box_norm_1000"),
                "candidate_roles": fr["observation"].get("candidate_roles"),
                "hard_negative_descriptions": fr["observation"].get("hard_negative_descriptions"),
                "identity_cues_seen": fr["observation"].get("identity_cues_seen"),
                "confidence": fr["observation"].get("confidence"),
            }
            for fr in frame_records
        ]
        judgment = aggregate_client.call_json(
            system_prompt=aggregate_system_prompt(),
            user_text=aggregate_user_prompt(target, hint, observation_compact),
            image_paths=[],
            schema_name="split_event_story_aggregate",
            metadata={"video": target.video, "obj_id": target.obj_id, "frames": idxs, "hint": hint},
            max_tokens=args.aggregate_max_tokens,
        )
        status_counts: dict[str, int] = {}
        for fr in frame_records:
            status = "ok" if is_usable_status(fr["observation"].get("status")) else str(fr["observation"].get("status"))
            status_counts[status] = status_counts.get(status, 0) + 1
        rec = {
            "video": target.video,
            "obj_id": target.obj_id,
            "frames": idxs,
            "hint": hint,
            "frame_observations": frame_records,
            "frame_status_counts": status_counts,
            "judgment": judgment,
        }
        records.append(rec)
        print(f"aggregate {target.video}:{target.obj_id} {judgment.get('status')} {judgment.get('event_type')} {judgment.get('recommended_action')} conf={judgment.get('confidence')}", flush=True)
    payload = {
        "workspace": str(args.workspace),
        "model": args.model,
        "allow_fallback": bool(args.allow_fallback),
        "dry_run": bool(frame_client.cfg.dry_run or aggregate_client.cfg.dry_run),
        "frame_timeout": float(args.frame_timeout),
        "retry_timeout": float(args.retry_timeout),
        "aggregate_timeout": float(args.aggregate_timeout),
        "cache_dir": str(args.cache_dir),
        "panel_dir": str(args.out_panel_dir),
        "records": records,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_doc(args.out_doc, payload)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_doc}")


if __name__ == "__main__":
    main()

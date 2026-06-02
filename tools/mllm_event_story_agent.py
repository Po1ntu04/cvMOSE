#!/usr/bin/env python3
"""MLLM event-chain agent for MOSEv2 physical-instance tracking.

This tool upgrades the MLLM role from single-frame candidate verifier to a
sequence-level physical-instance analyst.  It renders standardized timeline
panels for every target object, asks Qwen-VL to infer the target's event story
(picked up, moved, reappeared, static distractor, low-contrast continuation,
etc.), and writes JSON/Markdown artifacts that downstream SAM prompt/reanchor
code can consume.

The output is *not* a mask and is not directly a final prediction.  It is an
audit/teaching layer: it labels likely positives, hard negatives, and prompt
plans so SAM2/SAM3/official masks can be applied in a more semantically grounded
way.
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
    add_caption,
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
from cvmose.qwen_vl_client import QwenVLClient, DEFAULT_MODEL  # noqa: E402

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
    "sam2": "pred_sam2_b101",
    "official_b": "pred_sam2_official_bplus_mosev2",
    "official_l": "pred_sam2_official_large_mosev2",
    "m7_tiny": "pred_sam2_m7_qwen_support_tiny_semantic",
    "q0only": "pred_m10_qwen36_q0only_fusion",
    "q04v": "pred_m10_qwen36_q0_4v_fusion",
}

DEFAULT_HUMAN_HINTS: dict[str, dict[str, str]] = {
    "8jsm23a7:1": (
        {
            "hint": "目标是一张被拿起的麻将。第二帧附近被手遮挡，然后被拿到玩家面前牌列；直到结尾它应是面前麻将最左侧的七条。首帧背面朝上，与中间待摸牌极像，外观匹配会误导。"
        }
    ),
    "z6dx46qr:1": (
        {
            "hint": "这是低对比水下难样本；可辨别线索可能主要是两个像眼睛的黑点，而不是明显物体轮廓。"
        }
    ),
    "q0sizv6m:2": (
        {
            "hint": "首帧靠后的 obj2 仓鼠/豚鼠状动物后续走到镜头前，成为画面下方/前景的新动物；原位置附近的同类动物应视为强干扰。"
        }
    ),
}


@dataclass(frozen=True)
class Target:
    video: str
    obj_id: int


def font(size: int = 16) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
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
    out: list[Target] = []
    for video in videos:
        try:
            _, ann = first_annotation(ann_root, video)
        except Exception:
            continue
        for obj in [int(x) for x in np.unique(ann) if int(x) != 0]:
            if target_filter and (video, obj) not in target_filter:
                continue
            out.append(Target(video, obj))
    return out


def choose_frames(num_frames: int, extra: list[int] | None = None, max_frames: int = 8) -> list[int]:
    base = [0, 1, 2, 3, 5, 8, num_frames // 3, num_frames // 2, (2 * num_frames) // 3, num_frames - 1]
    if extra:
        base.extend(extra)
    out: list[int] = []
    for x in base:
        x = max(0, min(num_frames - 1, int(x)))
        if x not in out:
            out.append(x)
    if len(out) <= max_frames:
        return out
    # Preserve first/early/last and sample the middle evenly.
    keep = [out[0], out[1], out[-1]]
    middle = [x for x in out[2:-1] if x not in keep]
    need = max_frames - len(keep)
    if need > 0 and middle:
        if need >= len(middle):
            keep[2:2] = middle
        else:
            idxs = np.linspace(0, len(middle) - 1, need).round().astype(int).tolist()
            keep[2:2] = [middle[i] for i in idxs]
    return sorted(dict.fromkeys(keep))[:max_frames]


def draw_grid(img: Image.Image, step: int = 250) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    f = font(14)
    for v in range(0, 1001, step):
        x = round(v * (w - 1) / 1000)
        y = round(v * (h - 1) / 1000)
        color = (255, 255, 255) if v % 500 else (255, 215, 0)
        d.line([(x, 0), (x, h)], fill=color, width=1)
        d.line([(0, y), (w, y)], fill=color, width=1)
        # PIL requires x1 >= x0 / y1 >= y0.  At the right/bottom edge place
        # coordinate tags inward instead of constructing an inverted rectangle.
        tx0 = min(max(1, x + 1), max(1, w - 52))
        ty0 = 1
        d.rectangle([tx0, ty0, min(w - 1, tx0 + 48), min(h - 1, ty0 + 17)], fill=(0, 0, 0))
        d.text((tx0 + 2, ty0 + 1), f"x{v}", fill=(255, 215, 0), font=f)
        tx0 = 1
        ty0 = min(max(1, y + 1), max(1, h - 20))
        d.rectangle([tx0, ty0, min(w - 1, tx0 + 49), min(h - 1, ty0 + 17)], fill=(0, 0, 0))
        d.text((tx0 + 2, ty0 + 1), f"y{v}", fill=(255, 215, 0), font=f)
    return out


def paste_fit(canvas: Image.Image, img: Image.Image, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    img = img.convert("RGB")
    scale = min(cw / img.width, ch / img.height)
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(resized, (x1 + (cw - nw) // 2, y1 + (ch - nh) // 2))


def panel_caption(img: Image.Image, text: str, height: int = 30) -> Image.Image:
    img = img.convert("RGB")
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img, (0, height))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, img.width, height], fill=(0, 0, 0))
    d.text((4, 6), text[:180], fill=(255, 215, 0), font=font(14))
    return out


def frame_mask(workspace: Path, root_name: str, video: str, frame_stem: str, obj_id: int) -> np.ndarray | None:
    path = workspace / "homework" / root_name / video / f"{frame_stem}.png"
    arr = load_label(path)
    if arr is None:
        return None
    return arr == int(obj_id)


def mask_area(mask: np.ndarray | None) -> int:
    return 0 if mask is None else int(np.asarray(mask).astype(bool).sum())


def union_box(masks: list[np.ndarray | None], shape: tuple[int, int], fallback: list[int] | None = None) -> list[int]:
    boxes = [bbox_from_mask(m) for m in masks if m is not None]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return expand_box(fallback, shape, pad=0.8, square=True, min_side=128)
    x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes); y2 = max(b[3] for b in boxes)
    return expand_box([x1, y1, x2, y2], shape, pad=0.9, square=True, min_side=160)


def load_hints(path: Path | None, use_builtin: bool) -> dict[str, dict[str, str]]:
    hints = dict(DEFAULT_HUMAN_HINTS) if use_builtin else {}
    if path and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if isinstance(value, str):
                hints[str(key)] = {"hint": value}
            elif isinstance(value, dict):
                hints[str(key)] = {str(k): str(v) for k, v in value.items()}
    return hints


def render_story_panel(
    *,
    workspace: Path,
    target: Target,
    out_path: Path,
    roots: dict[str, str],
    frames_idx: list[int],
    human_hint: dict[str, str] | None,
) -> dict[str, Any]:
    jpeg_root, ann_root = homework_roots(workspace)
    frames = list_frames(jpeg_root, target.video)
    ann_path, ann = first_annotation(ann_root, target.video)
    rgb0 = load_rgb(frames[0])
    ref_mask = ann == int(target.obj_id)
    ref_box = bbox_from_mask(ref_mask)
    ref_crop_box = expand_box(ref_box, ann.shape, pad=0.9, square=True, min_side=128)
    ref_raw = crop_pil(rgb0, ref_crop_box)
    ref_outline = outline_mask(ref_raw, crop_mask(ref_mask, ref_crop_box), BLUE, width=3)

    timeline_cells: list[Image.Image] = []
    frame_records: list[dict[str, Any]] = []
    for idx in frames_idx:
        fpath = frames[idx]
        rgb = load_rgb(fpath)
        stem = fpath.stem
        m7 = frame_mask(workspace, roots.get("m7", "pred_m7_part_balanced"), target.video, stem, target.obj_id)
        official_l = frame_mask(workspace, roots.get("official_l", "pred_sam2_official_large_mosev2"), target.video, stem, target.obj_id)
        official_b = frame_mask(workspace, roots.get("official_b", "pred_sam2_official_bplus_mosev2"), target.video, stem, target.obj_id)
        alt = frame_mask(workspace, roots.get("m7_tiny", "pred_sam2_m7_qwen_support_tiny_semantic"), target.video, stem, target.obj_id)
        # Wide frame: blue=current M7, yellow=official large, red=official b+, green=alt tiny/qwen if present.
        wide = draw_grid(rgb)
        for name, mask, color in [("M7", m7, BLUE), ("OFF_L", official_l, YELLOW), ("OFF_B", official_b, RED), ("ALT", alt, GREEN)]:
            if mask is not None and mask.any():
                wide = outline_mask(wide, mask, color, width=3)
                wide = draw_bbox(wide, bbox_from_mask(mask), color, width=3, label=name)
        crop_box = union_box([m7, official_l, official_b, alt], (rgb.height, rgb.width), ref_box)
        crop = crop_pil(rgb, crop_box)
        crop_over = crop.copy()
        for mask, color in [(m7, BLUE), (official_l, YELLOW), (official_b, RED), (alt, GREEN)]:
            if mask is not None and mask.any():
                crop_over = outline_mask(crop_over, crop_mask(mask, crop_box), color, width=3)
        pair = Image.new("RGB", (520, 250), "white")
        paste_fit(pair, wide, (0, 0, 260, 250))
        paste_fit(pair, crop_over, (260, 0, 520, 250))
        caption = f"f={idx:05d} M7={mask_area(m7)} OFF_L={mask_area(official_l)} OFF_B={mask_area(official_b)} ALT={mask_area(alt)}"
        timeline_cells.append(panel_caption(pair, caption, height=26))
        frame_records.append(
            {
                "frame_idx": idx,
                "frame_path": str(fpath),
                "areas": {"m7": mask_area(m7), "official_l": mask_area(official_l), "official_b": mask_area(official_b), "alt": mask_area(alt)},
                "boxes": {
                    "m7": bbox_from_mask(m7),
                    "official_l": bbox_from_mask(official_l),
                    "official_b": bbox_from_mask(official_b),
                    "alt": bbox_from_mask(alt),
                },
            }
        )

    ref_panel = Image.new("RGB", (1040, 330), "white")
    paste_fit(ref_panel, panel_caption(ref_raw, "REF raw crop"), (0, 0, 350, 330))
    paste_fit(ref_panel, panel_caption(ref_outline, "REF artificial blue outline; ignore color"), (350, 0, 700, 330))
    full0 = draw_bbox(outline_mask(rgb0, ref_mask, BLUE, width=3), ref_box, YELLOW, width=4, label=f"{target.video}:obj{target.obj_id}")
    paste_fit(ref_panel, panel_caption(full0, "frame0 full context"), (700, 0, 1040, 330))

    hint_text = human_hint.get("hint", "") if human_hint else ""
    legend = Image.new("RGB", (1040, 90), "white")
    d = ImageDraw.Draw(legend)
    d.text((8, 6), "Artificial outline legend: BLUE=M7/current, YELLOW=official_l, RED=official_b, GREEN=alt/qwen-tiny. Ignore outline colors as object appearance.", fill=(0, 0, 0), font=font(15))
    d.text((8, 34), f"Human hint: {hint_text[:220] if hint_text else 'none'}", fill=(180, 0, 0), font=font(15))
    d.text((8, 62), "Task: explain the same physical instance across time, identify distractors, and propose SAM prompt frames/locations.", fill=(0, 0, 0), font=font(15))

    cols = 2
    cell_w, cell_h = 1040 // cols, 276
    rows = (len(timeline_cells) + cols - 1) // cols
    timeline = Image.new("RGB", (1040, rows * cell_h), "white")
    for i, cell in enumerate(timeline_cells):
        paste_fit(timeline, cell, ((i % cols) * cell_w, (i // cols) * cell_h, (i % cols + 1) * cell_w, (i // cols + 1) * cell_h))
    sheet = Image.new("RGB", (1040, 330 + 90 + timeline.height), "white")
    sheet.paste(ref_panel, (0, 0))
    sheet.paste(legend, (0, 330))
    sheet.paste(timeline, (0, 420))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    return {
        "video": target.video,
        "obj_id": target.obj_id,
        "panel_path": str(out_path),
        "annotation_path": str(ann_path),
        "ref_bbox": ref_box,
        "ref_crop_box": ref_crop_box,
        "human_hint": human_hint or {},
        "frames": frame_records,
    }


def system_prompt() -> str:
    return """
你是视频目标分割中的“物理实例事件链分析器”和 SAM 提示教师。你的任务不是输出最终 mask，而是理解首帧 mask 指定的具体物理实例在视频中发生了什么：是否被手拿起、移动、放下、走向镜头、被遮挡、出画、重现，或被同类物体干扰。

你必须区分：同一物理实例 vs 外观相似同类对象。不要因为一个候选更清晰、更大、更居中、更像类别就选择它。若证据不足，明确 uncertain。若当前预测停在原位置但事件链显示目标已移动，应把原位置候选标为 hard negative。

图中所有蓝/黄/红/绿轮廓、框、文字都是人工标注，不能当作物体真实颜色/纹理。你可以利用它们理解候选来源，但判断身份必须依赖原图内容、手/物体运动、时序关系和语义事件。

只输出严格 JSON，不要输出解释段落。
""".strip()


def user_prompt(target: Target, frames_idx: list[int], metadata: dict[str, Any]) -> str:
    hint = metadata.get("human_hint", {}).get("hint", "")
    return f"""
视频={target.video}，obj_id={target.obj_id}。
采样帧={frames_idx}。
人类观察提示（可能有用，但你仍需看图判断，不要盲从）：{hint if hint else '无'}

请根据 REF 和时间线图，分析首帧 mask 指定的同一物理实例在后续帧中的事件链。重点回答：当前 M7/SAM2 类预测是否追错了静态同类物、是否该为空、是否需要在某个后续帧重新给 SAM 一个 box/point/mask prompt。

返回严格 JSON，schema 如下：
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
  "failure_if_wrong": "若你的判断错了，最可能错在哪里"
}}

坐标 box_norm_1000 若无法可靠给出可以填 null，但如果推荐 reanchor_at_frame 或 mlmm_box_to_sam，至少给一个大致框并降低/标明置信度。
""".strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--videos", nargs="*", default=VIDEOS15)
    p.add_argument("--targets", nargs="*", default=None, help="Optional filters like video:obj")
    p.add_argument("--frames-json", type=Path, default=None, help="Optional {video or video:obj: [frame_idx,...]} extra/override frames")
    p.add_argument("--hints-json", type=Path, default=None)
    p.add_argument("--no-builtin-hints", action="store_true")
    p.add_argument("--out-json", type=Path, default=Path("artifacts/m12_event_story/event_story_records.json"))
    p.add_argument("--out-panel-dir", type=Path, default=Path("artifacts/m12_event_story/panels"))
    p.add_argument("--out-doc", type=Path, default=Path("docs/m12_event_story_agent_report.md"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/m12_event_story/cache"))
    p.add_argument("--model", default=os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-calls", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--jpeg-quality", type=int, default=90)
    return p.parse_args()


def load_frames_json(path: Path | None) -> dict[str, list[int]]:
    if not path or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): [int(x) for x in v] for k, v in raw.items()}


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    records = payload.get("records", [])
    lines = [
        "# M12 event-story MLLM agent report",
        "",
        "**Purpose:** Use MLLM as a physical-instance event-chain analyst and SAM prompt teacher, not merely as an A/B mask verifier.",
        "",
        "## Run metadata",
        "",
        f"- model: `{payload.get('model')}`",
        f"- dry_run: `{payload.get('dry_run')}`",
        f"- total targets: `{len(records)}`",
        f"- cache_dir: `{payload.get('cache_dir')}`",
        f"- panel_dir: `{payload.get('panel_dir')}`",
        "",
        "## Summary table",
        "",
        "| video | obj | status | event_type | diagnosis | action | conf | panel |",
        "| --- | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for rec in records:
        j = rec.get("judgment", {}) or {}
        lines.append(
            f"| `{rec.get('video')}` | {rec.get('obj_id')} | {j.get('status','?')} | {j.get('event_type','?')} | {j.get('current_prediction_diagnosis','?')} | {j.get('recommended_action','?')} | {float(j.get('confidence') or 0):.2f} | `{rec.get('panel_path')}` |"
        )
    lines += [
        "",
        "## Harness contract",
        "",
        "- MLLM output is an event/story and prompt plan, not a final mask.",
        "- Positive prompt plans must still be refined by SAM2/SAM3/official masks and bounded propagation.",
        "- Hard negatives should be fed into DINO/SAM feature validation and future candidate filters.",
        "- If event story says the target moved, candidates at the original position should become negative even if visually similar.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    hints = load_hints(args.hints_json, not args.no_builtin_hints)
    frames_overrides = load_frames_json(args.frames_json)
    target_filter = parse_targets(args.targets)
    targets = discover_targets(args.workspace, args.videos, target_filter)
    client = QwenVLClient(model=args.model, cache_dir=args.cache_dir, dry_run=args.dry_run, max_side=args.max_side, jpeg_quality=args.jpeg_quality)
    records: list[dict[str, Any]] = []
    jpeg_root, _ = homework_roots(args.workspace)
    for target in targets:
        if len(records) >= args.max_calls:
            break
        frames = list_frames(jpeg_root, target.video)
        key = f"{target.video}:{target.obj_id}"
        extra = frames_overrides.get(key) or frames_overrides.get(target.video)
        idxs = choose_frames(len(frames), extra=extra, max_frames=args.max_frames)
        panel_path = args.out_panel_dir / target.video / f"obj{target.obj_id}_event_story.jpg"
        meta = render_story_panel(
            workspace=args.workspace,
            target=target,
            out_path=panel_path,
            roots=DEFAULT_ROOTS,
            frames_idx=idxs,
            human_hint=hints.get(key),
        )
        raw = client.call_json(
            system_prompt=system_prompt(),
            user_text=user_prompt(target, idxs, meta),
            image_paths=[panel_path],
            schema_name="event_story_agent",
            metadata={"video": target.video, "obj_id": target.obj_id, "frames": idxs, "human_hint": hints.get(key)},
            max_tokens=2048,
        )
        rec = {
            "video": target.video,
            "obj_id": target.obj_id,
            "panel_path": str(panel_path),
            "frames": idxs,
            "panel_meta": meta,
            "judgment": raw,
            "mllm_cache_key": raw.get("mllm_cache_key"),
            "model_used": raw.get("model_used"),
        }
        records.append(rec)
        print(f"[{len(records)}/{min(args.max_calls, len(targets))}] {target.video}:{target.obj_id} {raw.get('status')} {raw.get('event_type')} {raw.get('recommended_action')} conf={raw.get('confidence')}")
    payload = {
        "workspace": str(args.workspace),
        "model": args.model,
        "dry_run": bool(client.cfg.dry_run),
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

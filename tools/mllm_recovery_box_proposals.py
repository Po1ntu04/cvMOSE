#!/usr/bin/env python3
"""Ask Qwen-VL for high-recall reappearance boxes on hard MOSEv2 frames.

The output is *not* a mask and is never a final prediction.  It is a candidate
recall source: Qwen-VL may propose up to K normalized boxes for the same
physical instance, with risk tags and conservative confidence.  Downstream
SAM2/DINO/temporal checks must still verify every proposal before promotion.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.mllm_panels import (  # noqa: E402
    BLUE,
    YELLOW,
    add_caption,
    bbox_from_mask,
    crop_mask,
    crop_pil,
    draw_bbox,
    draw_label,
    expand_box,
    first_annotation,
    grid,
    homework_roots,
    list_frames,
    load_rgb,
    outline_mask,
    render_target_profile_panel,
)
from cvmose.qwen_vl_client import QwenVLClient  # noqa: E402


DEFAULT_FRAMES: dict[str, list[int]] = {
    "r13u5z4y": [20, 22, 24, 28, 32],
    "q0sizv6m": [16, 18, 23, 29, 35, 40],
    "msinig6m": [55, 70, 90, 110],
    "lcgc29va": [8, 9, 20, 30],
    "1qlssuz2": [10, 34],
    "2smf7uq9": [20, 30, 31],
    "4vznweiu": [9, 15, 20],
}

SAME_CLASS_DENSE = {"r13u5z4y", "q0sizv6m", "msinig6m", "2smf7uq9", "8jsm23a7", "4vznweiu"}


@dataclass(frozen=True)
class Target:
    video: str
    obj_id: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--targets", nargs="*", default=None, help="Optional video:obj filters")
    p.add_argument("--frames-json", type=Path, default=None, help="JSON mapping video -> list[int] or video:obj -> list[int]")
    p.add_argument("--target-profiles-json", type=Path, default=Path("artifacts/m7_qwen_vl/target_profiles.json"))
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-panel-dir", type=Path, required=True)
    p.add_argument("--summary-md", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/m9_qwen_boxes/cache"))
    p.add_argument("--model", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-calls", type=int, default=24)
    p.add_argument("--max-boxes", type=int, default=3)
    p.add_argument("--max-side", type=int, default=1400)
    p.add_argument("--jpeg-quality", type=int, default=90)
    return p.parse_args()


def load_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        arr = arr[..., 0]
    return arr


def parse_target_filter(items: list[str] | None) -> set[tuple[str, int]] | None:
    if not items:
        return None
    out: set[tuple[str, int]] = set()
    for item in items:
        video, obj = item.split(":", 1)
        out.add((video, int(obj)))
    return out


def profile_map(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("profiles", data if isinstance(data, list) else [])
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for rec in records:
        try:
            out[(str(rec["video"]), int(rec["obj_id"]))] = rec
        except Exception:
            continue
    return out


def custom_frames(path: Path | None) -> dict[str, list[int]]:
    if not path or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): [int(x) for x in v] for k, v in data.items()}


def target_objects(workspace: Path, videos: list[str] | None, targets_filter: set[tuple[str, int]] | None) -> list[Target]:
    jpeg_root, ann_root = homework_roots(workspace)
    use_videos = videos or sorted(p.name for p in jpeg_root.iterdir() if p.is_dir())
    out: list[Target] = []
    for video in use_videos:
        _, ann = first_annotation(ann_root, video)
        for obj_id in [int(x) for x in np.unique(ann) if int(x) != 0]:
            if targets_filter and (video, obj_id) not in targets_filter:
                continue
            out.append(Target(video, obj_id))
    return out


def draw_coord_grid(img: Image.Image) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    for v in range(0, 1001, 100):
        x = round(v * (w - 1) / 1000)
        y = round(v * (h - 1) / 1000)
        color = (255, 255, 255) if v % 200 else (255, 215, 0)
        d.line([(x, 0), (x, h)], fill=color, width=1)
        d.line([(0, y), (w, y)], fill=color, width=1)
        draw_label(d, (min(max(2, x + 2), w - 65), 2), f"x={v}", fill=(255, 215, 0), size=16)
        draw_label(d, (2, min(max(2, y + 2), h - 24)), f"y={v}", fill=(255, 215, 0), size=16)
    return out


def render_recovery_panel(
    *,
    workspace: Path,
    video: str,
    obj_id: int,
    frame_idx: int,
    ref_panel_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    jpeg_root, ann_root = homework_roots(workspace)
    frames = list_frames(jpeg_root, video)
    _, ann = first_annotation(ann_root, video)
    rgb0 = load_rgb(frames[0])
    ref_mask = ann == int(obj_id)
    ref_box = expand_box(bbox_from_mask(ref_mask), ann.shape, pad=0.75, square=True, min_side=112)
    ref_raw = crop_pil(rgb0, ref_box)
    ref_out = outline_mask(ref_raw, crop_mask(ref_mask, ref_box), BLUE, width=3)
    cur = load_rgb(frames[frame_idx])
    grid_img = draw_coord_grid(cur)
    panels = [
        add_caption(ref_raw, "REF raw crop: same physical target"),
        add_caption(ref_out, "REF with artificial blue outline; ignore outline color"),
        add_caption(grid_img, f"CURRENT frame {frame_idx} coordinate grid 0-1000"),
    ]
    # Include a copy of the pre-rendered target profile at thumbnail size if present.
    if ref_panel_path.is_file():
        prof = Image.open(ref_panel_path).convert("RGB")
        panels.append(add_caption(prof, "frame0 target profile panel"))
    sheet = grid(panels, cols=2, cell=(720, 560))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)
    grid_path = out_path.with_name(out_path.stem + "__current_grid.jpg")
    grid_img.save(grid_path, quality=92)
    return {
        "video": video,
        "obj_id": int(obj_id),
        "frame_idx": int(frame_idx),
        "panel_path": str(out_path),
        "current_grid_path": str(grid_path),
        "current_frame_path": str(frames[frame_idx]),
        "ref_panel_path": str(ref_panel_path),
        "image_size": [cur.width, cur.height],
        "ref_bbox": bbox_from_mask(ref_mask),
        "ref_crop_box": ref_box,
    }


def system_prompt() -> str:
    return (
        "你是视频目标重识别候选框召回器。任务不是分割，不输出 mask。"
        "你只在当前帧中给出与首帧 REF 指定的同一物理实例可能对应的高召回候选框。"
        "如果证据不足，可以给低置信候选并标风险；如果目标不可见或无法判断，也可以返回空。"
        "必须极度注意同类干扰、遮挡、小目标、复合区域和背景块。只输出严格 JSON。"
    )


def user_prompt(video: str, obj_id: int, frame_idx: int, image_size: list[int], profile: dict[str, Any] | None, max_boxes: int) -> str:
    prof = profile or {}
    same_class = bool(prof.get("has_same_class_distractors")) or video in SAME_CLASS_DENSE
    cues = prof.get("identity_cues") or []
    distractors = prof.get("likely_distractors") or []
    return f"""
给你两类图像：
1. REF / target profile：首帧 mask 指定的目标实例。
2. CURRENT：当前帧原图，并画了 0-1000 的坐标网格。

请在 CURRENT 原图坐标系中召回最多 {max_boxes} 个候选框，框住可能是 REF 同一物理实例的位置。
坐标必须是归一化到 0-1000 的 [x1,y1,x2,y2]，相对于 CURRENT 原图宽高，而不是拼图坐标。
当前帧原图尺寸 width,height = {image_size}。

视频={video}，obj_id={obj_id}，frame_idx={frame_idx}。
目标画像（若可信）：
- target_type: {prof.get('target_type', 'unknown')}
- semantic_description: {prof.get('semantic_description', 'unknown')}
- identity_cues: {cues}
- likely_distractors: {distractors}
- same_class_dense: {same_class}

关键规则：
- 判断同一物理实例，不是同类任意对象。
- 高召回优先：如果有多个 plausible 位置，可以都给，但必须标 confidence 和风险。
- 不要因为候选更大、更清晰、更居中就认为是目标。
- 如果候选可能是人手/人体、多个对象合并、背景块、同类干扰物，请标 risk_tags。
- 如果目标可能被完全遮挡/出画，允许 target_visible=no 或 uncertain，并返回空 candidates。
- 对 tiny/遮挡目标，如果只是猜测，confidence 不要高。

必须返回严格 JSON，格式：
{{
  "video": "{video}",
  "obj_id": {obj_id},
  "frame_idx": {frame_idx},
  "target_visible": "yes|no|uncertain",
  "should_attempt_sam2_box": true,
  "candidates": [
    {{
      "candidate_id": "A",
      "bbox_norm": [x1, y1, x2, y2],
      "confidence": 0.0,
      "reason_short": "short reason",
      "risk_tags": ["same_class_distractor|occluded|tiny_unreadable|background|composite|edge_partial|hand_or_person|uncertain"]
    }}
  ],
  "notes": "one short note"
}}
""".strip()


def normalize_proposal(raw: dict[str, Any], meta: dict[str, Any], max_boxes: int) -> dict[str, Any]:
    out = {
        "video": meta["video"],
        "obj_id": int(meta["obj_id"]),
        "frame_idx": int(meta["frame_idx"]),
        "target_visible": raw.get("target_visible", "uncertain"),
        "should_attempt_sam2_box": bool(raw.get("should_attempt_sam2_box", False)),
        "status": raw.get("status", "ok"),
        "model_used": raw.get("model_used"),
        "mllm_cache_key": raw.get("mllm_cache_key"),
        "cache_hit": raw.get("cache_hit", False),
        "notes": raw.get("notes") or raw.get("reason_short") or "",
        "raw_keys": sorted(raw.keys()),
        "candidates": [],
    }
    candidates = raw.get("candidates") if isinstance(raw.get("candidates"), list) else []
    for idx, cand in enumerate(candidates[:max_boxes]):
        if not isinstance(cand, dict):
            continue
        bbox = cand.get("bbox_norm") or cand.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            vals = [float(x) for x in bbox]
        except Exception:
            continue
        vals = [min(1000.0, max(0.0, v)) for v in vals]
        if vals[2] <= vals[0] + 2 or vals[3] <= vals[1] + 2:
            continue
        out["candidates"].append(
            {
                "candidate_id": str(cand.get("candidate_id") or chr(ord("A") + idx)),
                "bbox_norm": [round(v, 2) for v in vals],
                "confidence": float(cand.get("confidence") or 0.0),
                "reason_short": str(cand.get("reason_short") or ""),
                "risk_tags": [str(x) for x in cand.get("risk_tags", [])] if isinstance(cand.get("risk_tags"), list) else [],
            }
        )
    return out


def main() -> None:
    args = parse_args()
    ws = args.workspace.resolve()
    jpeg_root, _ = homework_roots(ws)
    profiles = profile_map(args.target_profiles_json)
    frame_cfg = custom_frames(args.frames_json)
    targets_filter = parse_target_filter(args.targets)
    targets = target_objects(ws, args.videos, targets_filter)
    client = QwenVLClient(model=args.model, cache_dir=args.cache_dir, dry_run=args.dry_run, max_side=args.max_side, jpeg_quality=args.jpeg_quality)
    records: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    calls = 0
    for target in targets:
        frames = list_frames(jpeg_root, target.video)
        key_obj = f"{target.video}:{target.obj_id}"
        frame_ids = frame_cfg.get(key_obj) or frame_cfg.get(target.video) or DEFAULT_FRAMES.get(target.video, [])
        frame_ids = [f for f in frame_ids if 0 <= int(f) < len(frames)]
        if not frame_ids:
            continue
        ref_panel = args.out_panel_dir / "reference" / f"{target.video}_obj{target.obj_id}.jpg"
        if not ref_panel.is_file():
            render_target_profile_panel(workspace=ws, video=target.video, obj_id=target.obj_id, out_path=ref_panel)
        for frame_idx in frame_ids:
            if calls >= args.max_calls:
                break
            panel_path = args.out_panel_dir / "recovery_panels" / target.video / f"obj{target.obj_id}_f{frame_idx:05d}.jpg"
            meta = render_recovery_panel(workspace=ws, video=target.video, obj_id=target.obj_id, frame_idx=int(frame_idx), ref_panel_path=ref_panel, out_path=panel_path)
            prof = profiles.get((target.video, target.obj_id), {})
            raw = client.call_json(
                system_prompt=system_prompt(),
                user_text=user_prompt(target.video, target.obj_id, int(frame_idx), meta["image_size"], prof, args.max_boxes),
                image_paths=[ref_panel, meta["current_grid_path"]],
                schema_name="recovery_box_proposal",
                metadata={"video": target.video, "obj_id": target.obj_id, "frame_idx": int(frame_idx)},
                max_tokens=1536,
            )
            prop = normalize_proposal(raw, meta, args.max_boxes)
            prop["panel_path"] = str(panel_path)
            prop["current_grid_path"] = meta["current_grid_path"]
            prop["profile_target_type"] = prof.get("target_type")
            prop["profile_anchor_policy"] = prof.get("anchor_policy")
            records.append(prop)
            panels.append(meta)
            calls += 1
        if calls >= args.max_calls:
            break
    out = {
        "method": "m9_qwen_recovery_box_proposals",
        "workspace": str(ws),
        "model_requested": client.cfg.model,
        "base_url": client.cfg.base_url,
        "dry_run": client.cfg.dry_run,
        "calls": calls,
        "records": records,
        "panels": panels,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.summary_md:
        counts = {"records": len(records), "with_candidates": sum(1 for r in records if r.get("candidates")), "dry_run": client.cfg.dry_run}
        lines = [
            "# M9 Qwen recovery box proposal summary",
            "",
            f"- Workspace: `{ws}`",
            f"- Model requested: `{client.cfg.model}`",
            f"- Calls: {calls}",
            f"- Dry run: {client.cfg.dry_run}",
            f"- Records with candidates: {counts['with_candidates']} / {counts['records']}",
            f"- JSON: `{args.out_json}`",
            f"- Panels: `{args.out_panel_dir}`",
            "",
            "| video | obj | frame | visible | candidates | model | notes |",
            "| --- | ---: | ---: | --- | ---: | --- | --- |",
        ]
        for r in records:
            lines.append(
                f"| {r['video']} | {r['obj_id']} | {r['frame_idx']} | {r.get('target_visible')} | {len(r.get('candidates', []))} | {r.get('model_used')} | {str(r.get('notes',''))[:80]} |"
            )
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "calls": calls, "records": len(records), "with_candidates": sum(1 for r in records if r.get("candidates")), "dry_run": client.cfg.dry_run}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

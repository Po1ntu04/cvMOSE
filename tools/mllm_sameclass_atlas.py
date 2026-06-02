#!/usr/bin/env python3
"""Same-class candidate atlas harness for Qwen-VL / qwen3.6-plus.

This M14 diagnostic tool is designed for crowded VOS failures where the target
is one physical instance among many visually similar objects.  It does not
produce masks.  It asks the MLLM to enumerate same-class candidates per sampled
frame, mark hard negatives, and aggregate a conservative re-anchor plan that can
be tested by SAM2 bounded propagation.
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


def parse_targets(items: list[str] | None) -> set[tuple[str, int]] | None:
    if not items:
        return None
    out: set[tuple[str, int]] = set()
    for item in items:
        video, obj = item.split(":", 1)
        out.add((video, int(obj)))
    return out


def discover_targets(workspace: Path, videos: list[str], filt: set[tuple[str, int]] | None) -> list[Target]:
    _, ann_root = homework_roots(workspace)
    out: list[Target] = []
    for video in videos:
        try:
            _, ann = first_annotation(ann_root, video)
        except Exception:
            continue
        for obj_id in [int(x) for x in np.unique(ann) if int(x) != 0]:
            if filt and (video, obj_id) not in filt:
                continue
            out.append(Target(video, obj_id))
    return out


def load_json_map(path: Path | None) -> dict[str, list[int]]:
    if not path or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): [int(x) for x in v] for k, v in raw.items()}


def load_hints(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            out[str(key)] = str(value.get("hint") or value.get("description") or "")
        else:
            out[str(key)] = str(value)
    return out


def choose_frames(num_frames: int, extras: list[int] | None, max_frames: int) -> list[int]:
    wanted: list[int] = []
    for idx in [0, 1, 2, *(extras or []), num_frames - 1]:
        idx = max(0, min(num_frames - 1, int(idx)))
        if idx not in wanted:
            wanted.append(idx)
    if len(wanted) <= max_frames:
        return sorted(wanted)
    # Preserve early identity context and late/reappearance extras over uniform sampling.
    mandatory = []
    for idx in [0, 1, 2, max(wanted)]:
        if idx not in mandatory:
            mandatory.append(idx)
    middle = [x for x in wanted if x not in mandatory]
    remaining = max(0, max_frames - len(mandatory))
    if remaining and middle:
        picks = np.linspace(0, len(middle) - 1, min(remaining, len(middle))).round().astype(int).tolist()
        mandatory.extend(middle[i] for i in picks)
    return sorted(dict.fromkeys(mandatory))[:max_frames]


def fit(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    img = img.convert("RGB")
    scale = min(cw / img.width, ch / img.height)
    nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    out = Image.new("RGB", (cw, ch), "white")
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    out.paste(resized, ((cw - nw) // 2, (ch - nh) // 2))
    return out


def add_caption(img: Image.Image, text: str, height: int = 30) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img.convert("RGB"), (0, height))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, img.width, height], fill=(0, 0, 0))
    d.text((6, 7), text[:160], fill=(255, 215, 0), font=font(13))
    return out


def draw_grid(img: Image.Image, step: int = 200) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    f = font(12)
    for x in range(0, w, step):
        d.line([(x, 0), (x, h)], fill=(255, 0, 0), width=1)
        d.text((x + 2, 2), str(x), fill=(255, 255, 0), font=f, stroke_width=1, stroke_fill=(0, 0, 0))
    for y in range(0, h, step):
        d.line([(0, y), (w, y)], fill=(255, 0, 0), width=1)
        d.text((2, y + 2), str(y), fill=(255, 255, 0), font=f, stroke_width=1, stroke_fill=(0, 0, 0))
    return out


def norm_box_1000(box: list[int] | None, size: tuple[int, int]) -> list[int] | None:
    if box is None:
        return None
    w, h = size
    x1, y1, x2, y2 = box
    return [round(1000 * x1 / max(1, w)), round(1000 * y1 / max(1, h)), round(1000 * x2 / max(1, w)), round(1000 * y2 / max(1, h))]


def paste_fit(canvas: Image.Image, img: Image.Image, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    fitted = fit(img, box)
    canvas.paste(fitted, (x1, y1))


def render_atlas_panel(workspace: Path, target: Target, frame_idx: int, out_path: Path, compact: bool = False) -> dict[str, Any]:
    jpeg_root, ann_root = homework_roots(workspace)
    frames = list_frames(jpeg_root, target.video)
    _, ann = first_annotation(ann_root, target.video)
    rgb0 = load_rgb(frames[0])
    rgb = load_rgb(frames[frame_idx])
    ref_mask = ann == int(target.obj_id)
    ref_box = bbox_from_mask(ref_mask)
    ref_crop_box = expand_box(ref_box, ann.shape, pad=1.2, square=True, min_side=160)
    ref_crop = outline_mask(crop_pil(rgb0, ref_crop_box), crop_mask(ref_mask, ref_crop_box), BLUE, width=3)
    full_ref = draw_bbox(outline_mask(rgb0, ref_mask, BLUE, width=3), ref_box, YELLOW, width=3, label="REF")
    cur_grid = draw_grid(rgb, step=200)
    # Use the frame0 x/y neighborhood as a local crop, but keep full-grid context so reappearing objects elsewhere remain visible.
    local_box = expand_box(ref_box, (rgb.height, rgb.width), pad=6.0, square=True, min_side=360)
    local = draw_grid(crop_pil(rgb, local_box), step=100)

    if compact:
        canvas = Image.new("RGB", (760, 520), "white")
        cells = [
            (add_caption(ref_crop, "REF exact target"), (0, 0, 230, 230)),
            (add_caption(full_ref, "frame0 target context"), (230, 0, 420, 230)),
            (add_caption(cur_grid, f"current {frame_idx:05d} full grid"), (420, 0, 760, 330)),
            (add_caption(local, "current local crop grid"), (0, 230, 420, 500)),
        ]
        footer_y = 500
    else:
        canvas = Image.new("RGB", (1120, 760), "white")
        cells = [
            (add_caption(ref_crop, "REF crop: exact first-frame target"), (0, 0, 300, 250)),
            (add_caption(full_ref, "frame0 full context; blue=target mask"), (300, 0, 560, 250)),
            (add_caption(cur_grid, f"current frame {frame_idx:05d}; red coordinate grid"), (560, 0, 1120, 500)),
            (add_caption(local, "current local left/REF-neighborhood crop + grid"), (0, 250, 560, 700)),
        ]
        footer_y = 720
    for im, box in cells:
        paste_fit(canvas, im, box)
    d = ImageDraw.Draw(canvas)
    d.text((8, footer_y), "Task: enumerate same/near-class candidates and hard negatives; boxes use normalized [x1,y1,x2,y2] /1000 from full current frame.", fill=(0, 0, 0), font=font(14))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=88)
    return {
        "video": target.video,
        "obj_id": int(target.obj_id),
        "frame_idx": int(frame_idx),
        "panel_path": str(out_path),
        "frame_path": str(frames[frame_idx]),
        "ref_box_norm_1000": norm_box_1000(ref_box, rgb0.size),
        "local_crop_norm_1000": norm_box_1000(local_box, rgb.size),
    }


def frame_system_prompt() -> str:
    return """
You are a same-class candidate atlas builder for video object segmentation.
Your job is to enumerate physical object candidates and hard negatives in ONE frame.
Do not output masks. Do not choose the most salient object by category. Return strict JSON only.
""".strip()


def frame_user_prompt(target: Target, frame_idx: int, hint: str, meta: dict[str, Any]) -> str:
    return f"""
video={target.video}, obj_id={target.obj_id}, frame={frame_idx}.
The REF crop/full frame identifies one exact physical instance in frame 0. The current frame has a coordinate grid.
Human observation/hypothesis, if any: {hint or 'none'}

For this frame, enumerate up to 8 same-class or visually similar candidates, including plausible target and hard negatives.
Use normalized full-frame boxes [x1,y1,x2,y2] in 0..1000 coordinates.
Be conservative: if the original target is out of frame, say so. If several objects are similar, mark them as hard_negative/uncertain rather than pretending identity is certain.

Return strict JSON:
{{
  "status":"ok|uncertain",
  "target_visible":"yes|no|partial|uncertain",
  "frame_summary":"one sentence",
  "candidates":[
    {{"candidate_id":"A", "bbox_norm_1000":[0,0,0,0], "description":"where/pose", "role":"same_instance_candidate|hard_negative|uncertain", "motion_clue":"short", "confidence":0.0}}
  ],
  "positive_box_norm_1000":[0,0,0,0] | null,
  "hard_negative_boxes":[[0,0,0,0]],
  "identity_cues_seen":["..."],
  "confidence":0.0
}}
""".strip()


def retry_user_prompt(target: Target, frame_idx: int, hint: str) -> str:
    return f"""
video={target.video}, obj_id={target.obj_id}, frame={frame_idx}.
The image is a compact candidate-atlas panel with REF and current frame/crop grids.
Human hypothesis is not ground truth: {hint or 'none'}

Return strict JSON only, with at most 4 candidates. If unsure, prefer uncertain:
{{
  "status":"ok|uncertain",
  "target_visible":"yes|no|partial|uncertain",
  "frame_summary":"short",
  "candidates":[{{"candidate_id":"A","bbox_norm_1000":[0,0,0,0],"description":"short","role":"same_instance_candidate|hard_negative|uncertain","motion_clue":"short","confidence":0.0}}],
  "positive_box_norm_1000":[0,0,0,0] | null,
  "hard_negative_boxes":[[0,0,0,0]],
  "identity_cues_seen":["..."],
  "confidence":0.0
}}
""".strip()


def aggregate_system_prompt() -> str:
    return """
你是同类多目标视频重识别的事件链/负例记忆分析器。你会收到逐帧 candidate atlas JSON。
目标是建立：1) 正例轨迹假设；2) 每帧同类 hard-negative bank；3) 可供 SAM2 bounded propagation 测试的保守 re-anchor box plan。
不要输出 mask。不要把人类提示当 GT；若证据不够，输出 uncertain/manual_review。
输出严格 JSON。
""".strip()


def aggregate_user_prompt(target: Target, hint: str, observations: list[dict[str, Any]]) -> str:
    return f"""
video={target.video}, obj_id={target.obj_id}
Human observation/hypothesis: {hint or 'none'}

Frame candidate atlases:
{json.dumps(observations, ensure_ascii=False, indent=2)}

Return strict JSON:
{{
  "video":"{target.video}",
  "obj_id":{target.obj_id},
  "status":"ok|uncertain|manual_review",
  "target_summary":"first-frame physical instance summary",
  "same_class_problem":"why same-class distractors are hard here",
  "positive_track_hypothesis":[{{"frame_idx":0,"bbox_norm_1000":[0,0,0,0],"reason":"...","confidence":0.0}}],
  "negative_memory_bank":[{{"frame_idx":0,"bbox_norm_1000":[0,0,0,0],"description":"hard negative","reason":"..."}}],
  "positive_prompt_plan":[{{"frame_idx":0,"location_description":"...","box_norm_1000":[0,0,0,0],"prompt_type":"box","confidence":0.0,"rationale":"..."}}],
  "commit_policy":{{"merge_mode":"window","merge_radius":0,"clip_mode":"nearest","needs_delayed_confirmation":true,"unsafe_frames":[0]}},
  "recommended_action":"keep_current|reanchor_at_frame|manual_review|generate_more_candidates",
  "confidence":0.0,
  "failure_if_wrong":"most likely failure mode"
}}
""".strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, default=Path("/home/yu/projects/cv/from fdu/MOSEv2"))
    p.add_argument("--videos", nargs="*", required=True)
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--frames-json", type=Path, default=None)
    p.add_argument("--hints-json", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=Path("artifacts/m14_sameclass_atlas/atlas_records.json"))
    p.add_argument("--out-panel-dir", type=Path, default=Path("artifacts/m14_sameclass_atlas/panels"))
    p.add_argument("--out-doc", type=Path, default=Path("docs/m14_sameclass_atlas.md"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/m14_sameclass_atlas/cache"))
    p.add_argument("--model", default=os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-calls", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=7)
    p.add_argument("--max-side", type=int, default=900)
    p.add_argument("--jpeg-quality", type=int, default=78)
    p.add_argument("--frame-timeout", type=float, default=45)
    p.add_argument("--retry-timeout", type=float, default=90)
    p.add_argument("--frame-retries", type=int, default=1)
    p.add_argument("--aggregate-timeout", type=float, default=180)
    p.add_argument("--frame-max-tokens", type=int, default=1536)
    p.add_argument("--aggregate-max-tokens", type=int, default=2048)
    p.add_argument("--panel-mode", choices=["compact", "standard"], default="compact")
    return p.parse_args()


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# M14 same-class candidate atlas report",
        "",
        "This tool builds per-frame same-class candidate/negative banks with Qwen-VL and aggregates a conservative re-anchor plan. It is diagnostic and must be validated by SAM2 bounded propagation plus visual review before any submission use.",
        "",
        f"- model: `{payload.get('model')}`",
        f"- dry_run: `{payload.get('dry_run')}`",
        f"- targets: `{len(payload.get('records', []))}`",
        "",
        "| video | obj | frames | aggregate status | action | conf | positives | negatives |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for rec in payload.get("records", []):
        agg = rec.get("aggregate", {}) or {}
        lines.append(
            f"| `{rec.get('video')}` | {rec.get('obj_id')} | {','.join(map(str, rec.get('frames', [])))} | {agg.get('status','?')} | {agg.get('recommended_action','?')} | {float(agg.get('confidence') or 0):.2f} | {len(agg.get('positive_prompt_plan') or [])} | {len(agg.get('negative_memory_bank') or [])} |"
        )
    lines += [
        "",
        "## Use constraints",
        "",
        "- Qwen boxes are not ground truth; boxes must be fed through SAM2 and visually audited.",
        "- Same-class scenes require hard-negative rejection and delayed commit; a single MLLM support signal must not promote an anchor alone.",
        "- For final submission, only replace frames/videos with clear visual evidence and preserve 418 provided outputs byte-identical.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    hints = load_hints(args.hints_json)
    frame_overrides = load_json_map(args.frames_json)
    targets = discover_targets(args.workspace, args.videos, parse_targets(args.targets))[: max(0, int(args.max_calls))]
    frame_client = QwenVLClient(model=args.model, fallback_models=[], cache_dir=args.cache_dir, dry_run=args.dry_run, max_side=args.max_side, jpeg_quality=args.jpeg_quality, timeout=args.frame_timeout)
    retry_client = QwenVLClient(model=args.model, fallback_models=[], cache_dir=args.cache_dir, dry_run=args.dry_run, max_side=min(args.max_side, 650), jpeg_quality=max(55, min(args.jpeg_quality, 72)), timeout=args.retry_timeout)
    aggregate_client = QwenVLClient(model=args.model, fallback_models=[], cache_dir=args.cache_dir, dry_run=args.dry_run, max_side=args.max_side, jpeg_quality=args.jpeg_quality, timeout=args.aggregate_timeout)
    jpeg_root, _ = homework_roots(args.workspace)
    records: list[dict[str, Any]] = []
    for target in targets:
        frames = list_frames(jpeg_root, target.video)
        key = f"{target.video}:{target.obj_id}"
        idxs = choose_frames(len(frames), frame_overrides.get(key), args.max_frames)
        hint = hints.get(key, "")
        observations: list[dict[str, Any]] = []
        for idx in idxs:
            panel_path = args.out_panel_dir / target.video / f"obj{target.obj_id}_f{idx:05d}.jpg"
            meta = render_atlas_panel(args.workspace, target, idx, panel_path, compact=args.panel_mode == "compact")
            obs = frame_client.call_json(
                system_prompt=frame_system_prompt(),
                user_text=frame_user_prompt(target, idx, hint, meta),
                image_paths=[panel_path],
                schema_name="sameclass_frame_atlas",
                metadata={"video": target.video, "obj_id": target.obj_id, "frame_idx": idx},
                max_tokens=args.frame_max_tokens,
            )
            if str(obs.get("status")) in {"api_error", "client_error", "parse_error"}:
                for attempt in range(1, max(0, int(args.frame_retries)) + 1):
                    retry_panel = panel_path.with_name(f"{panel_path.stem}_retry{attempt}.jpg")
                    retry_meta = render_atlas_panel(args.workspace, target, idx, retry_panel, compact=True)
                    obs = retry_client.call_json(
                        system_prompt=frame_system_prompt(),
                        user_text=retry_user_prompt(target, idx, hint),
                        image_paths=[retry_panel],
                        schema_name="sameclass_frame_atlas_retry",
                        metadata={"video": target.video, "obj_id": target.obj_id, "frame_idx": idx, "attempt": attempt},
                        max_tokens=min(args.frame_max_tokens, 768),
                    )
                    obs.setdefault("retry_panel_meta", retry_meta)
                    if str(obs.get("status")) not in {"api_error", "client_error", "parse_error"}:
                        obs["recovered_by_retry"] = True
                        break
            obs_record = {"frame_idx": idx, "panel_path": str(panel_path), "panel_meta": meta, "observation": obs}
            observations.append(obs_record)
            print(f"atlas frame {key}@{idx} {obs.get('status')} conf={obs.get('confidence')}", flush=True)
        aggregate = aggregate_client.call_json(
            system_prompt=aggregate_system_prompt(),
            user_text=aggregate_user_prompt(target, hint, observations),
            image_paths=[],
            schema_name="sameclass_atlas_aggregate",
            metadata={"video": target.video, "obj_id": target.obj_id},
            max_tokens=args.aggregate_max_tokens,
        )
        print(f"atlas aggregate {key} {aggregate.get('status')} {aggregate.get('recommended_action')} conf={aggregate.get('confidence')}", flush=True)
        records.append({"video": target.video, "obj_id": int(target.obj_id), "frames": idxs, "hint": hint, "frame_atlases": observations, "aggregate": aggregate})
    payload = {"model": args.model, "dry_run": bool(args.dry_run), "records": records}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_doc(args.out_doc, payload)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_doc}")


if __name__ == "__main__":
    main()

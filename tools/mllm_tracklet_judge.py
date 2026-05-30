#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.mllm_panels import homework_roots, list_frames, load_label, pred_label_path, render_tracklet_judge_panel, write_sheet_index  # noqa: E402
from cvmose.qwen_vl_client import DEFAULT_MODEL, QwenVLClient  # noqa: E402

SYSTEM_PROMPT = """你是视频目标分割中的多帧一致性验证器。你需要判断候选在连续几帧中是否保持为同一个物理对象，并且是否与首帧 reference target 一致。你不能输出 mask，只能判断是否允许作为 pseudo-anchor。只输出 JSON。"""

USER_TEMPLATE = """给你 REF 和候选在连续 2-3 帧中的 crop/overlay。请判断：
1. 这些候选是否是同一个物体连续出现？
2. 它们是否与 REF 是同一物理实例？
3. 是否存在切换到同类干扰物、背景、复合区域的风险？
4. 是否允许将该候选作为 SAM2 的 add_new_mask pseudo-anchor？

返回严格 JSON：
{
  "video": "<video>",
  "obj_id": <int>,
  "anchor_frame": <int>,
  "same_object_across_frames": "yes|no|uncertain",
  "same_as_reference": "yes|no|uncertain",
  "promote_anchor": true|false,
  "confidence": <0.0-1.0>,
  "reason_short": "<short reason>",
  "risk_tags": ["same_class_switch","composite","background","occlusion","tiny_unreadable","unstable_tracklet"]
}

严格规则：
- promote_anchor=true 需要 same_object_across_frames=yes 且 same_as_reference=yes 且 confidence>=0.70。
- 若 same_as_reference=uncertain，promote_anchor 必须 false。
- 若存在 same_class_switch 风险，promote_anchor 必须 false，除非证据极强。
- 若帧间候选显著跳变，promote_anchor 必须 false。

video={video}; obj_id={obj_id}; anchor_frame={anchor_frame}; candidate_source={source}; candidate_id={candidate_id}; candidate_judge={candidate_judge}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render 2-3 frame tracklet panels and optionally call Qwen-VL delayed-promotion judge.")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--target-profiles-json", type=Path, default=Path("artifacts/m7_qwen_vl/target_profiles.json"))
    p.add_argument("--candidate-judgments-json", type=Path, required=True)
    p.add_argument("--model", default=os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/m7_qwen_vl/cache"))
    p.add_argument("--out-json", type=Path, default=Path("artifacts/m7_qwen_vl/tracklet_judgments.json"))
    p.add_argument("--out-panel-dir", type=Path, default=Path("artifacts/m7_qwen_vl/tracklet_panels"))
    p.add_argument("--out-doc", type=Path, default=Path("docs/m7_tracklet_judge_summary.md"))
    p.add_argument("--max-calls", type=int, default=40)
    return p.parse_args()


def load_candidate_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    roots = {k: Path(v) for k, v in data.get("pred_roots", {}).items()}
    return list(data.get("records", [])), roots


def choose_tracklet_candidate(rec: dict[str, Any]) -> dict[str, Any] | None:
    judgment = rec.get("judgment", {})
    best = str(judgment.get("best_candidate", "")).strip()
    candidates = rec.get("candidates", [])
    if best and len(best) == 1 and best.isalpha():
        for c in candidates:
            if str(c.get("candidate_id")) == best:
                return c
    if judgment.get("should_support_anchor"):
        for c in candidates:
            if c.get("source") not in {"m11"}:
                return c
    # Dry-run / uncertain mode still renders a small audit sample, but it must not promote.
    return candidates[0] if candidates else None


def load_masks_for_tracklet(roots: dict[str, Path], workspace: Path, video: str, obj_id: int, source: str, start: int) -> dict[int, np.ndarray]:
    jpeg_root, _ = homework_roots(workspace)
    frames = list_frames(jpeg_root, video)
    root = roots.get(source)
    if root is None:
        # Try prefix before ':' for anyfg-style source names.
        root = roots.get(source.split(":", 1)[0])
    if root is None:
        return {}
    out: dict[int, np.ndarray] = {}
    for idx in range(start, min(len(frames), start + 3)):
        p = pred_label_path(root, video, frames[idx])
        if not p.is_file():
            continue
        mask = load_label(p) == int(obj_id)
        if int(mask.sum()) > 0:
            out[idx] = mask
    return out


def normalize_tracklet(raw: dict[str, Any], video: str, obj_id: int, anchor_frame: int) -> dict[str, Any]:
    if raw.get("status") in {"dry_run", "parse_error", "api_error", "client_error"}:
        return {
            "video": video,
            "obj_id": int(obj_id),
            "anchor_frame": int(anchor_frame),
            "same_object_across_frames": "uncertain",
            "same_as_reference": "uncertain",
            "promote_anchor": False,
            "confidence": 0.0,
            "reason_short": raw.get("reason_short", "dry-run/API unavailable; no pseudo-anchor promotion"),
            "risk_tags": ["dry_run", "unstable_tracklet"],
            "status": raw.get("status"),
            "mllm_cache_key": raw.get("mllm_cache_key"),
        }
    out = dict(raw)
    out.setdefault("video", video); out.setdefault("obj_id", int(obj_id)); out.setdefault("anchor_frame", int(anchor_frame))
    out.setdefault("same_object_across_frames", "uncertain"); out.setdefault("same_as_reference", "uncertain")
    out.setdefault("promote_anchor", False); out.setdefault("confidence", 0.0); out.setdefault("reason_short", ""); out.setdefault("risk_tags", [])
    if out.get("same_object_across_frames") != "yes" or out.get("same_as_reference") != "yes" or float(out.get("confidence") or 0) < 0.70:
        out["promote_anchor"] = False
    return out


def write_summary(path: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    promotes = sum(1 for r in records if r.get("tracklet_judgment", {}).get("promote_anchor"))
    lines = [
        "# M7 tracklet judge summary",
        "",
        f"- workspace: `{args.workspace}`",
        f"- model: `{args.model}`",
        f"- API key present: `{bool(os.getenv('DASHSCOPE_API_KEY'))}`",
        f"- records: `{len(records)}`",
        f"- promote_anchor=true: `{promotes}`",
        "",
        "| video | obj | anchor | source | promote | conf | same-ref | risk | panel |",
        "| --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    for rec in records:
        j = rec.get("tracklet_judgment", {})
        lines.append(f"| {rec.get('video')} | {rec.get('obj_id')} | {rec.get('anchor_frame')} | {rec.get('source')} | {j.get('promote_anchor')} | {float(j.get('confidence') or 0):.2f} | {j.get('same_as_reference')} | {','.join(j.get('risk_tags', []))[:80]} | `{rec.get('panel_path')}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate_records, roots = load_candidate_records(args.candidate_judgments_json)
    client = QwenVLClient(model=args.model, cache_dir=args.cache_dir, dry_run=args.dry_run)
    records: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    calls = 0
    for rec in candidate_records:
        if calls >= args.max_calls:
            break
        video = str(rec.get("video")); obj_id = int(rec.get("obj_id")); frame_idx = int(rec.get("frame_idx"))
        cand = choose_tracklet_candidate(rec)
        if cand is None:
            continue
        source = str(cand.get("source"))
        masks = load_masks_for_tracklet(roots, args.workspace, video, obj_id, source, frame_idx)
        if not masks:
            continue
        panel_path = args.out_panel_dir / video / f"obj{obj_id}_f{frame_idx:05d}_{source.replace(':','_')}_tracklet.jpg"
        panel_meta = render_tracklet_judge_panel(workspace=args.workspace, video=video, obj_id=obj_id, anchor_frame=frame_idx, candidate_source=source, masks_by_frame=masks, out_path=panel_path)
        raw = client.call_json(
            system_prompt=SYSTEM_PROMPT,
            user_text=(USER_TEMPLATE
                .replace("{video}", str(video))
                .replace("{obj_id}", str(obj_id))
                .replace("{anchor_frame}", str(frame_idx))
                .replace("{source}", source)
                .replace("{candidate_id}", str(cand.get("candidate_id")))
                .replace("{candidate_judge}", json.dumps(rec.get("judgment", {}), ensure_ascii=False))),
            image_paths=[panel_path],
            schema_name="tracklet_judge",
            metadata={"video": video, "obj_id": obj_id, "anchor_frame": frame_idx, "source": source, "candidate_id": cand.get("candidate_id")},
        )
        judgment = normalize_tracklet(raw, video, obj_id, frame_idx)
        out = {"video": video, "obj_id": obj_id, "anchor_frame": frame_idx, "source": source, "candidate_id": cand.get("candidate_id"), "panel_path": str(panel_path), "candidate_judgment": rec.get("judgment", {}), "tracklet_judgment": judgment, "mllm_cache_key": raw.get("mllm_cache_key")}
        records.append(out); index.append(panel_meta); calls += 1
        print(json.dumps({"video": video, "obj_id": obj_id, "anchor_frame": frame_idx, "source": source, "promote": judgment.get("promote_anchor"), "status": judgment.get("status", raw.get("status"))}, ensure_ascii=False), flush=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_sheet_index(index, args.out_panel_dir / "sheet_index.json")
    write_summary(args.out_doc, records, args)
    print(json.dumps({"out_json": str(args.out_json), "out_doc": str(args.out_doc), "records": len(records), "cache_dir": str(args.cache_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

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

from cvmose.mllm_panels import (  # noqa: E402
    CandidateVisual,
    homework_roots,
    list_frames,
    load_label,
    pred_label_path,
    render_candidate_judge_panel,
    write_sheet_index,
)
from cvmose.qwen_vl_client import DEFAULT_MODEL, QwenVLClient  # noqa: E402

SYSTEM_PROMPT = """你是一个严格的视频目标分割候选验证器。你的任务不是分割，而是在多个候选 mask 中判断哪一个最可能是首帧 reference crop 指定的同一物理实例。

你必须极度保守，尤其在多个同类目标相似、目标很小、目标被遮挡、候选是复合区域或背景块时。不要因为候选更大、更清晰、更居中、更完整就选择它。若证据不足，选择 uncertain 或 empty，而不是猜测。

你只能输出 JSON，不要输出其他文本。

重要视觉约定：所有半透明颜色、绿色/红色 overlay、黄色/蓝色框、候选字母和文字标签都是人工标注，不是物体真实颜色/纹理/文字。判断身份时必须优先看 raw crop / raw context 的真实外观；overlay 只表示 mask 覆盖区域，不能把 overlay 颜色当作目标颜色。"""

USER_TEMPLATE = """给你：
- REF：首帧目标实例 crop。
- PRE-GAP：遮挡/不确定前最后可靠 crop，若有。
- CURRENT：当前帧上下文。
- Candidates A/B/C/...：不同模型给出的候选 mask。
- EMPTY：目标不可见/不应输出的选项。

请判断当前帧目标是否可见，以及哪个候选最可能是 REF 中的同一物理实例。不要把绿色 overlay、红色 overlay 或候选字母当作目标真实颜色/文字；它们只是人工标注。

关键要求：
1. 判断同一物理实例，不是同类对象。
2. 如果候选是同类干扰物、背景块、人手/人体、多个对象合并、或者 composite region，应 veto。
3. 如果无法确认，应返回 uncertain，不要强选。
4. 如果目标可能被遮挡或出画，EMPTY 是合法选择。
5. 若目标极小，只有在 crop 中清楚可见时才支持候选。
6. 若候选来自多源但明显同错，这不是独立证据。

返回严格 JSON：
{
  "video": "<video>",
  "obj_id": <int>,
  "frame_idx": <int>,
  "target_visible": "yes|no|uncertain",
  "best_candidate": "A|B|C|none|keep_baseline|empty|uncertain",
  "should_support_anchor": true|false,
  "should_veto_anchor": true|false,
  "confidence": <0.0-1.0>,
  "reason_short": "<short reason>",
  "same_instance_evidence": ["..."],
  "distractor_risks": ["same_class_distractor|background_blob|composite|hand_or_person|tiny_unreadable|occluded|edge_partial|source_correlated_error"],
  "recommended_action": "promote_after_confirmation|output_only|keep_baseline|reject_all|keep_empty|uncertain"
}

严格规则：
- 如果 best_candidate 不是明确同一实例，should_support_anchor 必须为 false。
- 如果 confidence < 0.65，should_support_anchor 必须为 false。
- 如果 risk 包含 same_class_distractor 且没有强证据，recommended_action 应为 output_only / keep_empty / uncertain。
- 对 same_class_dense 目标，不能因为单帧视觉相似就 support anchor。
- 如果你无法区分候选和干扰物，选择 uncertain。

{route_hint}

video={video}; obj_id={obj_id}; frame_idx={frame_idx}; target_type={target_type}; candidates={candidate_summary}
"""

ROUTE_HINTS = {
    "same_class_dense": "此视频包含多个高度相似的同类对象。请不要选择“看起来最像类别”的候选。只有能判断其与 REF 是同一物理实例时才支持。否则应选择 uncertain 或 empty。",
    "semantic_dominated": "请重点比较文字、logo、衣服颜色、特殊图案、牌面/字母/局部结构等语义线索。但若候选位置或上下文明显不一致，也要保持怀疑。",
    "tiny": "目标极小。请只根据放大的 crop 判断。若 crop 中无法可靠辨认目标，应返回 uncertain，不要猜测。",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render high-risk candidate panels and optionally call Qwen-VL candidate verifier.")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--target-profiles-json", type=Path, required=True)
    p.add_argument("--m5r-audit-json", type=Path, default=Path("artifacts/m5r_reanchor/m5r_reanchor_key.json"))
    p.add_argument("--pred-roots", default="", help="Comma-separated name=path roots; defaults to discovered baseline/m11/m5r roots")
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--model", default=os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/m7_qwen_vl/cache"))
    p.add_argument("--out-json", type=Path, default=Path("artifacts/m7_qwen_vl/candidate_judgments.json"))
    p.add_argument("--out-panel-dir", type=Path, default=Path("artifacts/m7_qwen_vl/candidate_panels"))
    p.add_argument("--out-doc", type=Path, default=Path("docs/m7_candidate_judge_summary.md"))
    p.add_argument("--max-calls", type=int, default=80)
    p.add_argument("--max-candidates", type=int, default=6)
    return p.parse_args()


def parse_pred_roots(text: str, workspace: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    if text:
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"bad --pred-roots item: {part}")
            name, path = part.split("=", 1)
            roots[name.strip()] = Path(path).expanduser()
    defaults = {
        "baseline": workspace / "homework" / "pred_sam2_b101",
        "m11": workspace / "homework" / "pred_sam2_m11_cycle",
        "m5r": workspace / "homework" / "pred_sam2_m5r_reanchor",
        "dino_m5r": workspace / "homework" / "pred_sam2_m5r_dino_key",
        "official_large": workspace / "homework" / "pred_official_submission_large_15only",
        "sam2long": workspace / "homework" / "pred_sam2long_np2_15only",
        "dam": workspace / "homework" / "pred_m6_dam4sam_dam4sam_single_per_obj_smoke",
        "m2_light": workspace / "homework" / "pred_sam2_m2_light",
        "rar_state": workspace / "homework" / "pred_sam2_rar_state",
        "tiny_crop": workspace / "homework" / "pred_sam2_tiny_crop_candidate",
    }
    for k, v in defaults.items():
        roots.setdefault(k, v)
    return {k: v.resolve() for k, v in roots.items() if v.is_dir()}


def load_profiles(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("profiles", data if isinstance(data, list) else [])
    out = {}
    for r in records:
        try:
            out[(str(r["video"]), int(r["obj_id"]))] = r
        except Exception:
            continue
    return out


def load_audit(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    by_video: dict[str, Any] = {}
    # Aggregate audit lacks objects_audit; per-video files are usually beside audit_json or in artifacts.
    candidates = [path.parent / "key_by_video", path.parent / "by_video", Path("artifacts/m5r_reanchor/key_by_video")]
    for folder in candidates:
        if folder.is_dir():
            for f in folder.glob("*.json"):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    if "video" in d and "objects_audit" in d:
                        by_video[str(d["video"])] = d
                except Exception:
                    pass
    return data, by_video


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool); bb = b.astype(bool)
    inter = np.logical_and(aa, bb).sum(); union = np.logical_or(aa, bb).sum()
    return float(inter / union) if union else 1.0


def select_high_risk_frames(
    *,
    video: str,
    obj_id: int,
    frames_count: int,
    audit_by_video: dict[str, Any],
    roots: dict[str, Path],
    workspace: Path,
    max_per_object: int = 8,
) -> list[int]:
    selected: list[int] = []
    v_audit = audit_by_video.get(video, {})
    obj_audit = v_audit.get("objects_audit", {}).get(str(obj_id), {}) if isinstance(v_audit, dict) else {}
    for a in obj_audit.get("anchors", [])[:4]:
        if int(a.get("frame_idx", -1)) > 0:
            selected.append(int(a["frame_idx"]))
    for c in obj_audit.get("top_candidates", [])[:10]:
        if int(c.get("frame_idx", -1)) > 0:
            selected.append(int(c["frame_idx"]))
    for f in obj_audit.get("event_frames", [])[:10]:
        if int(f) > 0:
            selected.append(int(f))
    # disagreement fallback: baseline non-empty but M11 empty, or candidate source differs in object area.
    jpeg_root, _ = homework_roots(workspace)
    frames = list_frames(jpeg_root, video)
    labels_cache: dict[str, list[np.ndarray | None]] = {}
    shape = None
    for name, root in roots.items():
        arrs: list[np.ndarray | None] = []
        for frame in frames:
            p = pred_label_path(root, video, frame)
            if p.is_file():
                arr = load_label(p)
                shape = arr.shape
                arrs.append(arr)
            else:
                arrs.append(None)
        labels_cache[name] = arrs
    base = labels_cache.get("baseline", [])
    m11 = labels_cache.get("m11", [])
    for idx in range(1, min(frames_count, len(frames))):
        if len(selected) >= max_per_object * 2:
            break
        barea = int((base[idx] == obj_id).sum()) if idx < len(base) and base[idx] is not None else 0
        marea = int((m11[idx] == obj_id).sum()) if idx < len(m11) and m11[idx] is not None else barea
        if barea > 0 and marea == 0:
            selected.append(idx)
            continue
        areas = []
        for labels in labels_cache.values():
            if idx < len(labels) and labels[idx] is not None:
                areas.append(int((labels[idx] == obj_id).sum()))
        if len(areas) >= 2 and max(areas) > 0 and (min(areas) == 0 or max(areas) / max(1, min(a for a in areas if a > 0)) > 3.0):
            selected.append(idx)
    out: list[int] = []
    for f in selected:
        if 0 < f < frames_count and f not in out:
            out.append(f)
        if len(out) >= max_per_object:
            break
    return out


def build_candidates_for_frame(roots: dict[str, Path], workspace: Path, video: str, obj_id: int, frame_idx: int, max_candidates: int) -> list[CandidateVisual]:
    jpeg_root, _ = homework_roots(workspace)
    frames = list_frames(jpeg_root, video)
    frame = frames[frame_idx]
    cands: list[CandidateVisual] = []
    seen: list[np.ndarray] = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    priority = ["baseline", "m11", "m5r", "dino_m5r", "dam", "sam2long", "official_large", "m2_light", "rar_state", "tiny_crop"]
    items = [(k, roots[k]) for k in priority if k in roots] + [(k, v) for k, v in roots.items() if k not in priority]
    for source, root in items:
        p = pred_label_path(root, video, frame)
        if not p.is_file():
            continue
        label = load_label(p)
        mask = label == int(obj_id)
        area = int(mask.sum())
        if area <= 0:
            continue
        if any(mask_iou(mask, old) > 0.985 for old in seen):
            continue
        seen.append(mask.copy())
        cid = letters[len(cands)]
        cands.append(CandidateVisual(candidate_id=cid, source=source, frame_idx=frame_idx, obj_id=obj_id, mask=mask, metadata={"path": str(p)}))
        if len(cands) >= max_candidates:
            break
    return cands


def normalize_judgment(raw: dict[str, Any], video: str, obj_id: int, frame_idx: int) -> dict[str, Any]:
    if raw.get("status") in {"dry_run", "parse_error", "api_error", "client_error"}:
        return {
            "video": video,
            "obj_id": int(obj_id),
            "frame_idx": int(frame_idx),
            "target_visible": "uncertain",
            "best_candidate": "uncertain",
            "should_support_anchor": False,
            "should_veto_anchor": True,
            "confidence": 0.0,
            "reason_short": raw.get("reason_short", "dry-run/API unavailable; conservative veto for anchor promotion"),
            "same_instance_evidence": [],
            "distractor_risks": ["source_correlated_error", "tiny_unreadable"] if raw.get("status") == "dry_run" else ["parse_or_api_error"],
            "recommended_action": "uncertain",
            "status": raw.get("status"),
            "mllm_cache_key": raw.get("mllm_cache_key"),
        }
    out = dict(raw)
    out.setdefault("video", video); out.setdefault("obj_id", int(obj_id)); out.setdefault("frame_idx", int(frame_idx))
    out.setdefault("target_visible", "uncertain"); out.setdefault("best_candidate", "uncertain")
    out.setdefault("should_support_anchor", False); out.setdefault("should_veto_anchor", not bool(out.get("should_support_anchor")))
    out.setdefault("confidence", 0.0); out.setdefault("reason_short", "")
    out.setdefault("same_instance_evidence", []); out.setdefault("distractor_risks", [])
    out.setdefault("recommended_action", "uncertain")
    return out


def write_summary(path: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    veto = sum(1 for r in records if r.get("judgment", {}).get("should_veto_anchor"))
    support = sum(1 for r in records if r.get("judgment", {}).get("should_support_anchor"))
    uncertain = sum(1 for r in records if r.get("judgment", {}).get("best_candidate") == "uncertain")
    lines = [
        "# M7 candidate judge summary",
        "",
        f"- workspace: `{args.workspace}`",
        f"- model: `{args.model}`",
        f"- API key present: `{bool(os.getenv('DASHSCOPE_API_KEY'))}`",
        f"- records: `{len(records)}`",
        f"- support/veto/uncertain: `{support}/{veto}/{uncertain}`",
        "",
        "| video | obj | frame | candidates | best | support | veto | conf | reason | panel |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for rec in records:
        j = rec.get("judgment", {})
        lines.append(f"| {rec.get('video')} | {rec.get('obj_id')} | {rec.get('frame_idx')} | {len(rec.get('candidates', []))} | {j.get('best_candidate')} | {j.get('should_support_anchor')} | {j.get('should_veto_anchor')} | {float(j.get('confidence') or 0):.2f} | {str(j.get('reason_short',''))[:80]} | `{rec.get('panel_path')}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    profiles = load_profiles(args.target_profiles_json)
    roots = parse_pred_roots(args.pred_roots, args.workspace)
    _, audit_by_video = load_audit(args.m5r_audit_json)
    jpeg_root, ann_root = homework_roots(args.workspace)
    videos = args.videos or sorted(p.name for p in jpeg_root.iterdir() if p.is_dir())
    client = QwenVLClient(model=args.model, cache_dir=args.cache_dir, dry_run=args.dry_run)
    records: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    calls = 0
    for video in videos:
        frames = list_frames(jpeg_root, video)
        ann_files = sorted((ann_root / video).glob("*.png"))
        if not ann_files:
            continue
        ann = load_label(ann_files[0])
        obj_ids = [int(x) for x in sorted(set(ann.reshape(-1).tolist())) if int(x) != 0]
        for obj_id in obj_ids:
            if calls >= args.max_calls:
                break
            prof = profiles.get((video, obj_id), {})
            target_type = str(prof.get("target_type", "unknown"))
            route_hint = ROUTE_HINTS.get(target_type, "")
            frame_idxs = select_high_risk_frames(video=video, obj_id=obj_id, frames_count=len(frames), audit_by_video=audit_by_video, roots=roots, workspace=args.workspace)
            for frame_idx in frame_idxs:
                if calls >= args.max_calls:
                    break
                cands = build_candidates_for_frame(roots, args.workspace, video, obj_id, frame_idx, args.max_candidates)
                if not cands:
                    continue
                panel_path = args.out_panel_dir / video / f"obj{obj_id}_f{frame_idx:05d}_candidate_judge.jpg"
                panel_meta = render_candidate_judge_panel(workspace=args.workspace, video=video, obj_id=obj_id, frame_idx=frame_idx, candidates=cands, out_path=panel_path, max_candidates=args.max_candidates)
                candidate_summary = json.dumps([c.to_index() for c in cands], ensure_ascii=False)
                user_text = (USER_TEMPLATE
                    .replace("{video}", str(video))
                    .replace("{obj_id}", str(obj_id))
                    .replace("{frame_idx}", str(frame_idx))
                    .replace("{target_type}", str(target_type))
                    .replace("{candidate_summary}", candidate_summary)
                    .replace("{route_hint}", route_hint))
                raw = client.call_json(system_prompt=SYSTEM_PROMPT, user_text=user_text, image_paths=[panel_path], schema_name="candidate_judge", metadata={"video": video, "obj_id": obj_id, "frame_idx": frame_idx, "candidates": [c.to_index() for c in cands]})
                judgment = normalize_judgment(raw, video, obj_id, frame_idx)
                rec = {"video": video, "obj_id": obj_id, "frame_idx": frame_idx, "panel_path": str(panel_path), "profile": prof, "candidates": [c.to_index() for c in cands], "judgment": judgment, "mllm_cache_key": raw.get("mllm_cache_key")}
                records.append(rec); index.append(panel_meta); calls += 1
                print(json.dumps({"video": video, "obj_id": obj_id, "frame_idx": frame_idx, "best": judgment.get("best_candidate"), "support": judgment.get("should_support_anchor"), "veto": judgment.get("should_veto_anchor"), "status": judgment.get("status", raw.get("status"))}, ensure_ascii=False), flush=True)
        if calls >= args.max_calls:
            break
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"records": records, "pred_roots": {k: str(v) for k, v in roots.items()}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_sheet_index(index, args.out_panel_dir / "sheet_index.json")
    write_summary(args.out_doc, records, args)
    print(json.dumps({"out_json": str(args.out_json), "out_doc": str(args.out_doc), "records": len(records), "cache_dir": str(args.cache_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

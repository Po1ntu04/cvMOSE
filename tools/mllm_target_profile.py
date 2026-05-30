#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.mllm_panels import first_annotation, homework_roots, render_target_profile_panel  # noqa: E402
from cvmose.qwen_vl_client import DEFAULT_MODEL, QwenVLClient  # noqa: E402

SAME_CLASS_DENSE = {"r13u5z4y", "q0sizv6m", "msinig6m", "2smf7uq9", "8jsm23a7"}
SEMANTIC_DOMINATED = {"4f98052b", "4vznweiu", "8jsm23a7", "c8lutf29", "pe0d85lk"}
TINY_LIKELY = {"lcgc29va", "4vznweiu", "1qlssuz2", "2smf7uq9"}
EDGE_PARTIAL_LIKELY = {"r13u5z4y", "z6dx46qr", "lcgc29va"}

SYSTEM_PROMPT = """你是视频目标分割任务中的目标画像分析器。输入是首帧图像、首帧 mask overlay、目标 crop 和上下文 crop。任务是理解“首帧 mask 指定的具体物理实例”，而不是识别类别。你不能输出像素 mask。你必须保守判断目标类型、可见性、是否残缺、潜在干扰物，以及推荐后续验证策略。只输出 JSON，不要输出解释文本。"""

USER_TEMPLATE = """请分析图中被 mask 标出的目标实例。注意：这是视频目标分割任务，后续需要在视频中持续跟踪同一个物理实例，而不是同类任意对象。

请判断：
1. 目标是否 tiny。
2. 目标是否位于边缘或首帧已经残缺/遮挡。
3. 目标是否 semantic-dominated，即是否有文字、logo、颜色组合、服装、牌面、特殊形状等可语言描述线索。
4. 场景中是否存在同类/近同类干扰物。
5. 若后续遮挡重现，哪些视觉线索最可靠。
6. 哪些候选验证策略应启用。

必须返回严格 JSON：
{
  "video": "<video>",
  "obj_id": <int>,
  "target_type": "regular|tiny|semantic_dominated|same_class_dense|edge_partial|unknown",
  "is_tiny": true|false,
  "is_edge_or_partial": true|false,
  "is_semantic_dominated": true|false,
  "has_same_class_distractors": true|false,
  "semantic_description": "<short target description>",
  "identity_cues": ["..."],
  "likely_distractors": ["..."],
  "recommended_modules": ["m11_safety","dino_descriptor","dam4sam","sam2long","tiny_crop","mllm_candidate_judge","veto_only"],
  "anchor_policy": "allow_support|veto_only|avoid_reanchor|tiny_requires_crop|unknown",
  "confidence": <0.0-1.0>,
  "notes": "<one short sentence>"
}

规则：
- 如果目标极小或模糊，不要编造细节。
- 如果多个同类目标非常相似，anchor_policy 应偏向 veto_only 或 avoid_reanchor。
- 如果目标有文字/图案/特殊颜色，可标为 semantic_dominated。
- 若目标首帧只是残缺局部，要标注 edge_partial 或 is_edge_or_partial。

video={video}; obj_id={obj_id}; measured_area_fraction={area_fraction:.6f}; measured_bbox={bbox}; measured_tiny={is_tiny_by_area}; measured_edge_partial={is_edge_or_partial_by_bbox}.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render target profile panels and optionally call Qwen-VL for M7 routing profiles.")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument("--model", default=os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/m7_qwen_vl/cache"))
    p.add_argument("--out-json", type=Path, default=Path("artifacts/m7_qwen_vl/target_profiles.json"))
    p.add_argument("--out-panel-dir", type=Path, default=Path("artifacts/m7_qwen_vl/profile_panels"))
    p.add_argument("--out-doc", type=Path, default=Path("docs/m7_target_profile_summary.md"))
    return p.parse_args()


def list_videos(workspace: Path, requested: list[str] | None) -> list[str]:
    jpeg_root, _ = homework_roots(workspace)
    if requested:
        return requested
    return sorted(p.name for p in jpeg_root.iterdir() if p.is_dir())


def object_ids(workspace: Path, video: str) -> list[int]:
    _, ann_root = homework_roots(workspace)
    _, ann = first_annotation(ann_root, video)
    return [int(x) for x in sorted(set(ann.reshape(-1).tolist())) if int(x) != 0]


def heuristic_profile(video: str, obj_id: int, panel_meta: dict[str, Any]) -> dict[str, Any]:
    is_tiny = bool(panel_meta.get("is_tiny_by_area")) or video in TINY_LIKELY
    is_edge = bool(panel_meta.get("is_edge_or_partial_by_bbox")) or video in EDGE_PARTIAL_LIKELY
    same = video in SAME_CLASS_DENSE
    semantic = video in SEMANTIC_DOMINATED
    if same:
        target_type = "same_class_dense"
        anchor_policy = "veto_only" if not is_tiny else "avoid_reanchor"
    elif is_tiny:
        target_type = "tiny"
        anchor_policy = "tiny_requires_crop"
    elif semantic:
        target_type = "semantic_dominated"
        anchor_policy = "allow_support"
    elif is_edge:
        target_type = "edge_partial"
        anchor_policy = "veto_only"
    else:
        target_type = "regular"
        anchor_policy = "unknown"
    modules = ["m11_safety", "dino_descriptor", "mllm_candidate_judge"]
    if same:
        modules.append("veto_only")
    if is_tiny:
        modules.append("tiny_crop")
    if semantic:
        modules.append("mllm_candidate_judge")
    return {
        "video": video,
        "obj_id": int(obj_id),
        "target_type": target_type,
        "is_tiny": is_tiny,
        "is_edge_or_partial": is_edge,
        "is_semantic_dominated": semantic,
        "has_same_class_distractors": same,
        "semantic_description": "dry-run heuristic; real Qwen-VL profile pending" if os.getenv("DASHSCOPE_API_KEY") is None else "heuristic fallback",
        "identity_cues": ["first-frame mask shape", "local crop appearance", "temporal consistency"],
        "likely_distractors": ["same-class nearby objects"] if same else ["background blobs", "occluders"],
        "recommended_modules": sorted(set(modules)),
        "anchor_policy": anchor_policy,
        "confidence": 0.0,
        "notes": "Conservative route generated without trusted MLLM semantics.",
    }


def normalize_profile(raw: dict[str, Any], video: str, obj_id: int, panel_meta: dict[str, Any]) -> dict[str, Any]:
    if raw.get("status") in {"dry_run", "parse_error", "api_error", "client_error"} or not raw.get("target_type"):
        profile = heuristic_profile(video, obj_id, panel_meta)
        profile["qwen_raw_status"] = raw.get("status")
        profile["mllm_cache_key"] = raw.get("mllm_cache_key")
        profile["dry_run"] = raw.get("status") == "dry_run"
        return profile
    profile = dict(raw)
    profile.setdefault("video", video)
    profile.setdefault("obj_id", int(obj_id))
    profile.setdefault("target_type", "unknown")
    profile.setdefault("is_tiny", bool(panel_meta.get("is_tiny_by_area")))
    profile.setdefault("is_edge_or_partial", bool(panel_meta.get("is_edge_or_partial_by_bbox")))
    profile.setdefault("is_semantic_dominated", False)
    profile.setdefault("has_same_class_distractors", video in SAME_CLASS_DENSE)
    profile.setdefault("semantic_description", "")
    profile.setdefault("identity_cues", [])
    profile.setdefault("likely_distractors", [])
    profile.setdefault("recommended_modules", [])
    profile.setdefault("anchor_policy", "unknown")
    profile.setdefault("confidence", 0.0)
    profile.setdefault("notes", "")
    profile["dry_run"] = False
    return profile


def write_summary(path: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    for r in records:
        counts[str(r.get("target_type", "unknown"))] = counts.get(str(r.get("target_type", "unknown")), 0) + 1
    lines = [
        "# M7 target profile summary",
        "",
        f"- workspace: `{args.workspace}`",
        f"- model: `{args.model}`",
        f"- API key present: `{bool(os.getenv('DASHSCOPE_API_KEY'))}`",
        f"- dry_run requested: `{args.dry_run}`",
        f"- objects profiled: `{len(records)}`",
        f"- type counts: `{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`",
        "",
        "| video | obj | type | tiny | edge/partial | semantic | same-class | policy | confidence | panel |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for r in records:
        lines.append(
            f"| {r.get('video')} | {r.get('obj_id')} | {r.get('target_type')} | {r.get('is_tiny')} | {r.get('is_edge_or_partial')} | {r.get('is_semantic_dominated')} | {r.get('has_same_class_distractors')} | {r.get('anchor_policy')} | {float(r.get('confidence') or 0):.2f} | `{r.get('panel_path')}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    client = QwenVLClient(model=args.model, cache_dir=args.cache_dir, dry_run=args.dry_run)
    records: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []
    for video in list_videos(args.workspace, args.videos):
        for obj_id in object_ids(args.workspace, video):
            panel_path = args.out_panel_dir / video / f"obj{obj_id}_target_profile.jpg"
            meta = render_target_profile_panel(workspace=args.workspace, video=video, obj_id=obj_id, out_path=panel_path)
            user_text = (USER_TEMPLATE
                .replace("{video}", str(video))
                .replace("{obj_id}", str(obj_id))
                .replace("{area_fraction:.6f}", f"{float(meta.get('area_fraction', 0.0)):.6f}")
                .replace("{bbox}", str(meta.get("bbox")))
                .replace("{is_tiny_by_area}", str(meta.get("is_tiny_by_area")))
                .replace("{is_edge_or_partial_by_bbox}", str(meta.get("is_edge_or_partial_by_bbox"))))
            raw = client.call_json(
                system_prompt=SYSTEM_PROMPT,
                user_text=user_text,
                image_paths=[panel_path],
                schema_name="target_profile",
                metadata={"video": video, "obj_id": obj_id, **meta},
            )
            profile = normalize_profile(raw, video, obj_id, meta)
            profile.update({"panel_path": str(panel_path), "panel_meta": meta, "mllm_cache_key": raw.get("mllm_cache_key")})
            records.append(profile)
            panels.append(meta)
            print(json.dumps({"video": video, "obj_id": obj_id, "target_type": profile.get("target_type"), "policy": profile.get("anchor_policy"), "status": raw.get("status")}, ensure_ascii=False), flush=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"profiles": records, "panels": panels}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(args.out_doc, records, args)
    print(json.dumps({"out_json": str(args.out_json), "out_doc": str(args.out_doc), "profiles": len(records), "cache_dir": str(args.cache_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

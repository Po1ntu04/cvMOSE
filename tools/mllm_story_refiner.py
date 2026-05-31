#!/usr/bin/env python3
"""Text-level MLLM planner over visual event-story records.

The image-heavy Qwen-VL calls are useful for grounding, but qwen3.6-plus can
time out on dense multi-frame panels.  This second-stage harness keeps the
image understanding results as evidence, then asks qwen3.6-plus with *text
only* to produce a conservative teach-SAM plan: where semantic re-anchoring is
worth trying, what must become hard negatives, and which probes should be run
before touching a final submission.

It does not create masks.  It creates an auditable, model-readable plan for the
next bounded propagation / fusion step.
"""
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

from cvmose.qwen_vl_client import QwenVLClient, DEFAULT_MODEL  # noqa: E402


DEFAULT_USER_CORRECTIONS = {
    "8jsm23a7:1": "目标麻将首帧背面朝上，第二帧被遮挡/拿起，之后应是玩家面前牌列最左侧的七条；原位置/中间背面牌是强负例。",
    "z6dx46qr:1": "难点是低对比，可能只有两个类似眼睛的黑点可作为身份线索。",
    "q0sizv6m:2": "首帧靠后的动物走到镜头前，成为画面下方/前景的新动物；原位置附近同类是强负例。",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--event-json", type=Path, default=Path("artifacts/m12_event_story/all15_records.json"))
    p.add_argument("--out-json", type=Path, default=Path("artifacts/m12_event_story/q36_teach_sam_plan.json"))
    p.add_argument("--out-doc", type=Path, default=Path("docs/m12_q36_teach_sam_plan.md"))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/m12_event_story/cache"))
    p.add_argument("--model", default=os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-records", type=int, default=30)
    return p.parse_args()


def compact_record(rec: dict[str, Any]) -> dict[str, Any]:
    j = rec.get("judgment", {}) or {}
    keep = {
        "video": rec.get("video"),
        "obj_id": rec.get("obj_id"),
        "model_used_for_visual_story": j.get("model_used"),
        "status": j.get("status"),
        "target_summary": j.get("target_summary"),
        "event_type": j.get("event_type"),
        "diagnosis": j.get("current_prediction_diagnosis"),
        "recommended_action": j.get("recommended_action"),
        "confidence": j.get("confidence"),
        "identity_cues": j.get("identity_cues", [])[:5],
        "hard_negatives": j.get("hard_negatives", [])[:4],
        "positive_prompt_plan": j.get("positive_prompt_plan", [])[:4],
        "sam_teaching_plan": j.get("sam_teaching_plan", {}),
        "failure_if_wrong": j.get("failure_if_wrong"),
    }
    key = f"{rec.get('video')}:{rec.get('obj_id')}"
    if key in DEFAULT_USER_CORRECTIONS:
        keep["user_correction"] = DEFAULT_USER_CORRECTIONS[key]
    return keep


def system_prompt() -> str:
    return """
你是 MOSEv2 training-free VOS 改进的系统架构师。你收到的不是 GT，而是来自多帧视觉 MLLM harness 的事件链分析记录。你要把这些记录转化为“教 SAM”的可执行计划：正提示、负例、延迟提交、bounded propagation 和安全融合。

核心原则：
1. 不训练、不 finetune、不使用后续 GT。
2. MLLM 不直接输出最终 mask；只能提出 re-anchor / negative / fusion / probe 计划。
3. 若没有独立视觉证据，不要把同类相似对象作为 anchor。
4. 目标是从 43.30 向 44 逼近，因此要优先处理隐藏分低、改动杠杆大的事件链错位，而不是微调全局阈值。
5. 输出必须是严格 JSON。
""".strip()


def user_prompt(records: list[dict[str, Any]]) -> str:
    return f"""
下面是 15 个待测视频所有目标的事件链分析记录（已经包含人类纠正观察）。请生成一个泛化的 agent/harness 级 teach-SAM 计划。

已知最新事实：
- M7/M10 类融合只能到约 43.30；局部 q0/4v 替换没有明显涨分。
- 关键瓶颈不是阈值，而是“物理实例事件理解 -> 候选生成 -> anchor 注入 -> bounded propagation -> 延迟提交 -> 安全融合”链条没有真正闭环。
- 8jsm23a7、q0sizv6m:obj2、z6dx46qr、amfdu83t 是必须优先深挖的低分/高杠杆对象。

事件记录：
{json.dumps(records, ensure_ascii=False, indent=2)}

请返回严格 JSON：
{{
  "status": "ok|uncertain",
  "global_bottleneck": "一句话概括为什么之前 MLLM/融合只到 43.30",
  "recommended_architecture": ["分层 harness 组件，按执行顺序"],
  "ranked_targets": [
    {{
      "video": "xxx",
      "obj_id": 1,
      "priority": "P0|P1|P2|drop",
      "problem_type": "event_reanchor|same_class_switch|low_contrast|tiny_reappearance|static_ok|uncertain",
      "teach_sam_action": "mlmm_box_to_sam|official_interval_probe|empty_interval_probe|hard_negative_veto|keep_m7|manual_review_probe",
      "positive_prompt_frames": [0],
      "negative_roles": ["original_position_distractor"],
      "bounded_propagation": "forward|backward|local_window|none",
      "fusion_guard": "safe|balanced_only|aggressive_only|do_not_submit",
      "expected_score_leverage": "high|medium|low|unknown",
      "why_this_can_help": "机制理由",
      "main_risk": "最大风险",
      "validation_needed": ["必须看的图/必须跑的 probe"]
    }}
  ],
  "generalized_harness_upgrades": [
    {{
      "name": "组件名",
      "purpose": "解决什么结构性问题",
      "implementation_detail": "怎么接入当前 M5R/M7",
      "success_test": "如何证明它有用"
    }}
  ],
  "next_experiments": [
    {{
      "name": "实验名",
      "targets": ["video:obj"],
      "command_or_tool_to_add": "需要新增/调用的工具",
      "stop_condition": "何时放弃",
      "submission_candidate": "safe|balanced|aggressive|diagnostic_only"
    }}
  ]
}}
""".strip()


def write_doc(path: Path, plan: dict[str, Any], event_json: Path) -> None:
    lines = [
        "# M12 qwen3.6 teach-SAM plan",
        "",
        f"- source event JSON: `{event_json}`",
        f"- status: `{plan.get('status')}`",
        f"- global bottleneck: {plan.get('global_bottleneck')}",
        "",
        "## Ranked targets",
        "",
        "| target | priority | problem | action | fusion | leverage | risk |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in plan.get("ranked_targets", []) or []:
        lines.append(
            f"| `{t.get('video')}:{t.get('obj_id')}` | {t.get('priority')} | {t.get('problem_type')} | {t.get('teach_sam_action')} | {t.get('fusion_guard')} | {t.get('expected_score_leverage')} | {t.get('main_risk')} |"
        )
    lines += ["", "## Next experiments", ""]
    for e in plan.get("next_experiments", []) or []:
        lines += [
            f"### {e.get('name')}",
            f"- targets: `{', '.join(e.get('targets', []) or [])}`",
            f"- tool/command: `{e.get('command_or_tool_to_add')}`",
            f"- stop: {e.get('stop_condition')}",
            f"- candidate: `{e.get('submission_candidate')}`",
            "",
        ]
    lines += ["", "## Raw JSON", "", "```json", json.dumps(plan, ensure_ascii=False, indent=2), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.event_json.read_text(encoding="utf-8"))
    records = [compact_record(r) for r in payload.get("records", [])[: args.max_records]]
    client = QwenVLClient(model=args.model, fallback_models=[], cache_dir=args.cache_dir, dry_run=args.dry_run)
    plan = client.call_json(
        system_prompt=system_prompt(),
        user_text=user_prompt(records),
        image_paths=[],
        schema_name="q36_teach_sam_plan",
        metadata={"event_json": str(args.event_json), "record_count": len(records)},
        max_tokens=4096,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_doc(args.out_doc, plan, args.event_json)
    print(json.dumps({"out_json": str(args.out_json), "out_doc": str(args.out_doc), "model_used": plan.get("model_used"), "status": plan.get("status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

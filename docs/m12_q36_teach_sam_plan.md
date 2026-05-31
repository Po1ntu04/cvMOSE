# M12 qwen3.6 teach-SAM plan

- source event JSON: `artifacts/m12_event_story/all15_records.json`
- status: `ok`
- global bottleneck: 当前管线将 SAM 视为被动分割器，缺乏事件感知的负例注入、物理边界传播截断与延迟提交门控，导致同类静态干扰物引发身份漂移，且重锚定前过早融合错误掩码，使分数卡在 43.30。

## Ranked targets

| target | priority | problem | action | fusion | leverage | risk |
| --- | --- | --- | --- | --- | --- | --- |
| `8jsm23a7:1` | P0 | event_reanchor | mlmm_box_to_sam | safe | high | 若 f=48 框未精确覆盖七条牌面，SAM 可能分割到相邻牌列。 |
| `q0sizv6m:2` | P0 | same_class_switch | hard_negative_veto | safe | high | 负例点若落在目标运动路径上，会导致 SAM 短暂丢失目标。 |
| `z6dx46qr:1` | P0 | low_contrast | official_interval_probe | safe | high | 窗口过小可能导致特征断裂，需依赖双黑点相对位置稳定性。 |
| `amfdu83t:1` | P0 | event_reanchor | mlmm_box_to_sam | safe | high | 若鹿在 f=2 后完全出画，重锚定框可能误触发背景分割。 |
| `4vznweiu:1` | P1 | event_reanchor | mlmm_box_to_sam | balanced_only | medium | f=34 框若未对齐手部放置位置，可能分割到散落字母。 |
| `c8lutf29:1` | P1 | uncertain | official_interval_probe | safe | medium | 区间探测可能漏掉短暂出画后的快速重现。 |
| `pe0d85lk:2` | P1 | event_reanchor | empty_interval_probe | safe | medium | 空区间过长可能导致 SAM 完全丢失目标上下文。 |
| `lcgc29va:1` | P2 | static_ok | keep_m7 | balanced_only | low | 过度干预可能破坏 M7 已有的稳定跟踪。 |

## Next experiments

### P0_Event_Reanchor_Veto
- targets: `8jsm23a7:1, q0sizv6m:2, amfdu83t:1`
- tool/command: `注入 `NegativeVetoEngine` + `DelayedSubmissionGate`，启用 `mlmm_box_to_sam` 重锚定`
- stop: 若 3 个目标平均 IoU 提升 < 0.03 或出现新实例混淆，则回退至 `keep_m7`
- candidate: `safe`

### LowContrast_LocalWindow_Probe
- targets: `z6dx46qr:1, c8lutf29:1`
- tool/command: `切换 `BoundedPropagator` 为 `local_window`，启用 `official_interval_probe` 验证双黑点/浅色牛特征`
- stop: 若局部窗口导致特征断裂或 IoU 下降 > 0.05，则扩大窗口至 `forward` 并降低负例权重
- candidate: `balanced`

### Empty_Interval_Gate_Validation
- targets: `pe0d85lk:2, 4f98052b:1`
- tool/command: `部署 `empty_interval_probe` 与 `DelayedSubmissionGate`，验证出画/重现事件的空掩码输出`
- stop: 若空区间过长导致目标重现时 SAM 无法恢复，则缩短探测步长或引入轻量级运动先验
- candidate: `diagnostic_only`


## Raw JSON

```json
{
  "status": "ok",
  "global_bottleneck": "当前管线将 SAM 视为被动分割器，缺乏事件感知的负例注入、物理边界传播截断与延迟提交门控，导致同类静态干扰物引发身份漂移，且重锚定前过早融合错误掩码，使分数卡在 43.30。",
  "recommended_architecture": [
    "EventChainParser",
    "AnchorInjector",
    "NegativeVetoEngine",
    "BoundedPropagator",
    "DelayedSubmissionGate",
    "SafeFusionRouter"
  ],
  "ranked_targets": [
    {
      "video": "8jsm23a7",
      "obj_id": 1,
      "priority": "P0",
      "problem_type": "event_reanchor",
      "teach_sam_action": "mlmm_box_to_sam",
      "positive_prompt_frames": [
        0,
        48
      ],
      "negative_roles": [
        "original_position_distractor"
      ],
      "bounded_propagation": "forward",
      "fusion_guard": "safe",
      "expected_score_leverage": "high",
      "why_this_can_help": "明确切断牌墙静态干扰，通过 f=48 重锚定强制 SAM 接受物理位移，避免 M7 在中间帧持续输出错误背景掩码。",
      "main_risk": "若 f=48 框未精确覆盖七条牌面，SAM 可能分割到相邻牌列。",
      "validation_needed": [
        "f=0 到 f=48 的掩码连续性图",
        "f=48 重锚定后的 IoU 对比"
      ]
    },
    {
      "video": "q0sizv6m",
      "obj_id": 2,
      "priority": "P0",
      "problem_type": "same_class_switch",
      "teach_sam_action": "hard_negative_veto",
      "positive_prompt_frames": [
        0,
        2
      ],
      "negative_roles": [
        "original_position_distractor",
        "similar_appearance_distractor"
      ],
      "bounded_propagation": "forward",
      "fusion_guard": "safe",
      "expected_score_leverage": "high",
      "why_this_can_help": "利用负例门控在 f=2 前抑制原位置同类豚鼠，配合前景重锚定框，强制 SAM 切换实例身份。",
      "main_risk": "负例点若落在目标运动路径上，会导致 SAM 短暂丢失目标。",
      "validation_needed": [
        "f=0-3 的负例抑制热力图",
        "f=2 重锚定后的实例切换轨迹"
      ]
    },
    {
      "video": "z6dx46qr",
      "obj_id": 1,
      "priority": "P0",
      "problem_type": "low_contrast",
      "teach_sam_action": "official_interval_probe",
      "positive_prompt_frames": [
        0,
        2
      ],
      "negative_roles": [
        "original_position_distractor"
      ],
      "bounded_propagation": "local_window",
      "fusion_guard": "safe",
      "expected_score_leverage": "high",
      "why_this_can_help": "低对比度下全局传播极易漂移，限制为 local_window 并依赖双黑点特征进行区间探测，可避免误融合背景颗粒。",
      "main_risk": "窗口过小可能导致特征断裂，需依赖双黑点相对位置稳定性。",
      "validation_needed": [
        "f=0-5 的局部窗口掩码边界",
        "双黑点特征在低对比帧的保留率"
      ]
    },
    {
      "video": "amfdu83t",
      "obj_id": 1,
      "priority": "P0",
      "problem_type": "event_reanchor",
      "teach_sam_action": "mlmm_box_to_sam",
      "positive_prompt_frames": [
        0,
        2
      ],
      "negative_roles": [
        "original_position_distractor"
      ],
      "bounded_propagation": "forward",
      "fusion_guard": "safe",
      "expected_score_leverage": "high",
      "why_this_can_help": "鹿出画/遮挡事件导致 M7 误跟静态草地，f=2 重锚定可快速纠正轨迹，延迟提交机制可过滤 f=23 后的空预测。",
      "main_risk": "若鹿在 f=2 后完全出画，重锚定框可能误触发背景分割。",
      "validation_needed": [
        "f=0-10 的运动轨迹连续性",
        "f=23 后的空掩码输出验证"
      ]
    },
    {
      "video": "4vznweiu",
      "obj_id": 1,
      "priority": "P1",
      "problem_type": "event_reanchor",
      "teach_sam_action": "mlmm_box_to_sam",
      "positive_prompt_frames": [
        0,
        34
      ],
      "negative_roles": [
        "original_position_distractor"
      ],
      "bounded_propagation": "forward",
      "fusion_guard": "balanced_only",
      "expected_score_leverage": "medium",
      "why_this_can_help": "手部交互导致位置突变，重锚定可纠正静态字母干扰，平衡融合策略可容忍中间帧的短暂遮挡。",
      "main_risk": "f=34 框若未对齐手部放置位置，可能分割到散落字母。",
      "validation_needed": [
        "手部交互帧的掩码切换点",
        "f=34 放置位置的 IoU"
      ]
    },
    {
      "video": "c8lutf29",
      "obj_id": 1,
      "priority": "P1",
      "problem_type": "uncertain",
      "teach_sam_action": "official_interval_probe",
      "positive_prompt_frames": [
        0,
        2
      ],
      "negative_roles": [
        "original_position_distractor"
      ],
      "bounded_propagation": "local_window",
      "fusion_guard": "safe",
      "expected_score_leverage": "medium",
      "why_this_can_help": "牛出画/重现事件需区间探测确认身份，避免将 f=46 右侧新牛误融合。",
      "main_risk": "区间探测可能漏掉短暂出画后的快速重现。",
      "validation_needed": [
        "f=0-5 与 f=40-50 的掩码对比",
        "出画区间的空输出验证"
      ]
    },
    {
      "video": "pe0d85lk",
      "obj_id": 2,
      "priority": "P1",
      "problem_type": "event_reanchor",
      "teach_sam_action": "empty_interval_probe",
      "positive_prompt_frames": [
        0
      ],
      "negative_roles": [
        "original_position_distractor"
      ],
      "bounded_propagation": "forward",
      "fusion_guard": "safe",
      "expected_score_leverage": "medium",
      "why_this_can_help": "明确空区间探测策略，防止将远处相似人物误认为重现目标，延迟提交可过滤无效掩码。",
      "main_risk": "空区间过长可能导致 SAM 完全丢失目标上下文。",
      "validation_needed": [
        "f=0-60 的空掩码输出",
        "f=61 远处人物的负例抑制效果"
      ]
    },
    {
      "video": "lcgc29va",
      "obj_id": 1,
      "priority": "P2",
      "problem_type": "static_ok",
      "teach_sam_action": "keep_m7",
      "positive_prompt_frames": [
        0,
        34
      ],
      "negative_roles": [
        "original_position_distractor"
      ],
      "bounded_propagation": "forward",
      "fusion_guard": "balanced_only",
      "expected_score_leverage": "low",
      "why_this_can_help": "红色背包特征显著，M7 默认表现已较好，仅需保持当前策略并微调远处模糊帧的融合权重。",
      "main_risk": "过度干预可能破坏 M7 已有的稳定跟踪。",
      "validation_needed": [
        "f=34 远处背包的掩码清晰度",
        "全局 IoU 波动"
      ]
    }
  ],
  "generalized_harness_upgrades": [
    {
      "name": "NegativeVetoEngine",
      "purpose": "解决同类静态/相似外观干扰物导致的身份漂移问题",
      "implementation_detail": "在 `hard_negatives` 指定帧注入负点/负框，强制 SAM 在该帧输出空掩码或极低置信度，阻断错误传播链。",
      "success_test": "对比开启/关闭负例门控时，f=23(鹿)、f=41(豚鼠)、f=48(麻将) 的掩码 IoU 提升 ≥ 0.08。"
    },
    {
      "name": "BoundedPropagator",
      "purpose": "解决物理事件（遮挡、出画、低对比）导致的全局传播越界",
      "implementation_detail": "根据 `event_type` 动态设置传播窗口：`forward` 用于连续运动，`local_window` 用于低对比/遮挡，`none` 用于静态。窗口边界由 MLLM 事件链自动截断。",
      "success_test": "在 `z6dx46qr` 和 `c8lutf29` 中，掩码漂移帧数减少 ≥ 60%，且无跨实例融合。"
    },
    {
      "name": "DelayedSubmissionGate",
      "purpose": "解决重锚定前过早提交错误掩码导致的融合污染",
      "implementation_detail": "在 `reanchor_at_frame` 事件触发后，暂停掩码输出，直到新正提示框的 SAM 响应置信度 > 0.75 或 `probe` 验证通过，才恢复提交。",
      "success_test": "在 `8jsm23a7` 和 `q0sizv6m:2` 中，中间过渡帧的虚假掩码提交率降至 0%，最终融合 IoU 提升 ≥ 0.05。"
    }
  ],
  "next_experiments": [
    {
      "name": "P0_Event_Reanchor_Veto",
      "targets": [
        "8jsm23a7:1",
        "q0sizv6m:2",
        "amfdu83t:1"
      ],
      "command_or_tool_to_add": "注入 `NegativeVetoEngine` + `DelayedSubmissionGate`，启用 `mlmm_box_to_sam` 重锚定",
      "stop_condition": "若 3 个目标平均 IoU 提升 < 0.03 或出现新实例混淆，则回退至 `keep_m7`",
      "submission_candidate": "safe"
    },
    {
      "name": "LowContrast_LocalWindow_Probe",
      "targets": [
        "z6dx46qr:1",
        "c8lutf29:1"
      ],
      "command_or_tool_to_add": "切换 `BoundedPropagator` 为 `local_window`，启用 `official_interval_probe` 验证双黑点/浅色牛特征",
      "stop_condition": "若局部窗口导致特征断裂或 IoU 下降 > 0.05，则扩大窗口至 `forward` 并降低负例权重",
      "submission_candidate": "balanced"
    },
    {
      "name": "Empty_Interval_Gate_Validation",
      "targets": [
        "pe0d85lk:2",
        "4f98052b:1"
      ],
      "command_or_tool_to_add": "部署 `empty_interval_probe` 与 `DelayedSubmissionGate`，验证出画/重现事件的空掩码输出",
      "stop_condition": "若空区间过长导致目标重现时 SAM 无法恢复，则缩短探测步长或引入轻量级运动先验",
      "submission_candidate": "diagnostic_only"
    }
  ],
  "model_used": "qwen3.6-plus",
  "mllm_cache_key": "5eb4a58e29c030f995c5eae1de2bd967246faec33737daf6aa0c6fe8690004af"
}
```

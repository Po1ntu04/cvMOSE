# M2 — Reliable Memory Gate for Original SAM2

## 1. 问题归因

SAM2 的视频预测链路是 autoregressive 的：非交互帧的预测会进入 `non_cond_frame_outputs`，后续帧的 memory attention 会读取这些 mask memory / object pointer。若某一帧在遮挡、同类近邻、快速运动、小目标边缘等情形下漂移到错误实例，这个错误会被写入 memory bank 并放大为后续身份错跟踪。

本方法不把问题定义为“输出后处理是否压掉 mask”，而定义为：**当前帧预测是否足够可靠，可以作为未来跟踪的记忆证据**。

## 2. Insight

- 当前帧 mask 可以暂时输出，但不一定应该成为 tracker memory。
- 对 MOSEv2 这组小目标、遮挡、同类干扰场景，坏 memory 的代价常高于少写一帧 memory。
- 训练免费（training-free）的可靠性信号应优先来自 SAM2 已有输出：mask logits、object score、面积连续性、threshold perturbation stability、中心运动合理性。

## 3. Method

M2 在原版 SAM2 的 `propagate_in_video` 中做最小侵入 monkey patch：

1. 对 conditioning frame（首帧 GT mask）保持原始写入；
2. 对 non-conditioning frame 仍运行原版 `_run_single_frame_inference(..., run_mem_encoder=True)`；
3. 得到当前 `pred_masks` 后计算可靠性；
4. 若可靠：照常写入 `obj_output_dict["non_cond_frame_outputs"][frame_idx]`；
5. 若不可靠：当前帧 mask 仍 yield 给输出 PNG，但**不写入** `non_cond_frame_outputs`，因此后续 memory attention 不会读到该帧坏 memory。

关键点：不把 `maskmem_features=None` 存进 `non_cond_frame_outputs`，因为 SAM2 memory read path 默认会 `.to(device)`，直接保存 `None` 可能在后续读取时崩溃。M2 选择“不存该帧 output”，使 SAM2 原生 `.get(prev_frame_idx, None)` 分支自然跳过。

## 4. Reliability gate

默认配置对应当前 M2 base：

```text
obj_thr = 0.35
stability_thr = 0.60
min_area_pixels = 3
min_ratio = 0.20
max_ratio = 4.0
max_motion_px = max(40, 3.0 * sqrt(prev_area_pixels))
mask_thr = 0.50, perturb = 0.05
```

判定项：

- `obj_ok`: SAM2 `object_score_logits` sigmoid 后高于阈值；
- `area_abs_ok`: 当前 mask 面积不小于 3 pixels / mask resolution，且不异常铺满；
- `area_ratio_ok`: 当前面积相对参考 mask 不突变；
- `stable_ok`: `prob > 0.45` 与 `prob > 0.55` 的 mask IoU 足够高；
- `motion_ok`: 质心位移小于面积自适应阈值；极小 mask 可放松 motion，避免小目标中心噪声误杀。

当前默认 `reference=previous`，即严格贴近最初 pseudo-code。代码也保留 `last_reliable_on_empty` / `last_reliable` 作为后续 ablation，用于长遮挡后 reappearance 的可控实验。

## 5. 实现文件

- `tools/infer_mosev2_sam2_m2_memory_gate.py`
- `scripts/run_b101_m2_memory_gate.sh`

产物默认路径：

- predictions: `$MOSE_WORKSPACE/homework/pred_sam2_m2_memory_gate`
- submission dir: `$MOSE_WORKSPACE/homework/submission_433_m2_memory_gate`
- zip: `$MOSE_WORKSPACE/homework/submission_mosev2_m2_memory_gate.zip`
- aggregate audit: `$MOSE_WORKSPACE/homework/logs/m2_memory_gate_latest.json`
- per-video audit: `$MOSE_WORKSPACE/homework/logs/m2_memory_gate_by_video/<video>.json`

## 6. 验证计划

最小验证：

```bash
python3 -m py_compile tools/infer_mosev2_sam2_m2_memory_gate.py
bash -n scripts/run_b101_m2_memory_gate.sh
```

远端 smoke：

```bash
VIDEOS="r13u5z4y" ./scripts/run_b101_m2_memory_gate.sh
```

全量：

```bash
./scripts/run_b101_m2_memory_gate.sh
```

可选并发（每个 worker 写独立 per-video audit，最后聚合）：

```bash
M2_PARALLEL=1 GPUS=4 JOBS_PER_GPU=2 ./scripts/run_b101_m2_memory_gate.sh
```

## 7. 预期与风险

预期帮助：减少明显不稳定帧、遮挡中错漂移帧写入 memory bank，从而降低后续错误累积。

主要风险：

- gate 过严会导致 memory 过稀，长视频中 tracker 缺少近期目标证据；
- 相似近邻“稳定错跟踪”可能仍然通过面积/稳定性/运动检查；
- 目标真实快速位移可能被误判为 motion 异常；
- `previous` reference 对长遮挡后 reappearance 不一定最优，可能需要 `last_reliable_on_empty` ablation。

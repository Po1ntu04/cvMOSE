# Experiment Log


## 2026-05-30 — M2 reliable memory gate

- Commit: branch `method/m2-reliable-memory` review-fix head; use `git log -1` for exact current SHA.
- Method: original SAM2 + training-free reliability check before writing non-conditioning memory.
- Hypothesis: bad predictions should not become future tracker memory; this targets autoregressive drift without suppressing current-frame output.
- Videos: MOSEv2 15-video homework split; smoke target `r13u5z4y` because it contains severe occlusion and same-class strawberry distractors.
- Command: `VIDEOS="r13u5z4y" ./scripts/run_b101_m2_memory_gate.sh`, `M2_PARALLEL=1 GPUS=4 JOBS_PER_GPU=1 VIDEOS="r13u5z4y" ./scripts/run_b101_m2_memory_gate.sh`, then `./scripts/run_b101_m2_memory_gate.sh` for full run.
- Output: `homework/pred_sam2_m2_memory_gate`, `homework/submission_mosev2_m2_memory_gate.zip`, audit JSON under `homework/logs/m2_memory_gate_*`; local audit copy under `artifacts/m2_memory_gate/`.
- Runtime / GPU: full b101 run `230.69s`, `1004` frames, `4.35 fps`, CUDA peak allocated `970.86 MiB`, reserved `1190.0 MiB`.
- Qualitative evidence: pending visual comparison against SAM2 baseline.
- Quantitative evidence: full b101 audit `memory_skipped=710`, `memory_written=825`, `noncond_total=1535`, `skip_ratio=0.4625`; submission check `433` dirs, `66526` PNGs; zip test `bad=None`, size `71983174` bytes.
- Decision: code-review blockers fixed; M2 is experiment-ready but not yet claimed better until score/visual analysis.
- Next: inspect r13u5z4y and other high-skip videos visually; submit or evaluate zip if appropriate.

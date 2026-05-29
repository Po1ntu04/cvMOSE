# Experiment Log


## 2026-05-30 — M2 reliable memory gate

- Commit: branch `method/m2-reliable-memory` implementation commit; remote-run evidence to be appended after b101 is reachable.
- Method: original SAM2 + training-free reliability check before writing non-conditioning memory.
- Hypothesis: bad predictions should not become future tracker memory; this targets autoregressive drift without suppressing current-frame output.
- Videos: MOSEv2 15-video homework split; smoke target `r13u5z4y` because it contains severe occlusion and same-class strawberry distractors.
- Command: `VIDEOS="r13u5z4y" ./scripts/run_b101_m2_memory_gate.sh`, then `./scripts/run_b101_m2_memory_gate.sh` for full run.
- Output: `homework/pred_sam2_m2_memory_gate`, `homework/submission_mosev2_m2_memory_gate.zip`, audit JSON under `homework/logs/m2_memory_gate_*`.
- Runtime / GPU: pending remote b101 run; local CPU smoke on 3-frame `r13u5z4y` mini-workspace took 15.73s and wrote 2/2 non-conditioning memories; synthetic reliability-function smoke also passed under `cv-hw2` CPU torch.
- Qualitative evidence: pending visual comparison after outputs are available.
- Quantitative evidence: local smoke produced 3 PNGs and `m2_summary={noncond_total:2, memory_written:2, memory_skipped:0}`; full b101 submission/audit pending.
- Decision: implement first; keep M1/M1.1 separate and do not stack with M2.
- Blocker: `./scripts/sync_code_b101.sh` and direct `ssh -o ConnectTimeout=8 ... b101.guhk.cc` timed out on 2026-05-30 local session; code is pushed and ready to sync once b101 is reachable.
- Next: run smoke + full, inspect audit skip reasons, compare key frames against SAM2 baseline.

# cvMOSE

Principled experiments for the MOSEv2 homework split: first-frame instance masks -> video object segmentation outputs.

This repository versions **code + research practice docs + selected visual/audit evidence**. It does not version datasets, checkpoints, full remote prediction folders, or non-final candidate zips.

## Current stance

- Default model family for optimization: **SAM2-first**.
- SAM3.1 public prompting is treated as a failed control route for this homework interface.
- SAM3.1 GT-mask adapter remains a diagnostic/candidate source, not the primary route.
- Main research problem: **training-free re-anchor tracking: uncertainty management + proposal retrieval + identity verification + delayed commit under occlusion, tiny targets, same-class distractors, and edge disappear/reappear**.

## Layout

- `docs/experiment_index.md` — compact index of all uploaded experiment code/docs/figures and final stance.
- `docs/reanchor_tracker_direction.md` — next primary architecture after M11/M2/M3/M4 saturation.
- `tools/` — runnable inference/build scripts copied from the MOSEv2 workspace and future method code.
- `scripts/` — b101 code-only sync and experiment launch wrappers.
- `src/cvmose/` — reusable utilities.
- `docs/` — durable reasoning: problem attribution, insight, method mapping, experiment protocol.
- `configs/` — example path/env configuration only.


## Submission zip policy

Only one local workspace should retain submission zips. This workspace currently keeps only the final candidate `submission_mosev2_final_m11_cycle.zip`; experiment attempts are preserved through code, reports, audit JSON, and selected visual sheets instead.

## Remote policy

b101 is for running experiments, not for storing full local analysis artifacts. Use:

```bash
scripts/sync_code_b101.sh
```

This syncs code/config only to `${CVMOSE_CODE_ROOT:-/data1/yuzhixiang/cv_mosev2/cvMOSE}` and excludes local docs/assets, data, checkpoints, logs, contact sheets, and submission artifacts.

## Git identity

Repository commits use:

- user: `Po1ntu04`
- email: `2352190@tongji.edu.cn`

# cvMOSE

Principled experiments for the MOSEv2 homework split: first-frame instance masks -> video object segmentation outputs.

This repository intentionally versions **code + research practice docs**, not datasets, checkpoints, rendered analysis images, or submission zips.

## Current stance

- Default model family for optimization: **SAM2-first**.
- SAM3.1 public prompting is treated as a failed control route for this homework interface.
- SAM3.1 GT-mask adapter remains a diagnostic/candidate source, not the primary route.
- Main research problem: **occlusion-aware instance re-anchoring under tiny targets, same-class distractors, and edge disappear/reappear**.

## Layout

- `tools/` — runnable inference/build scripts copied from the MOSEv2 workspace and future method code.
- `scripts/` — b101 code-only sync and experiment launch wrappers.
- `src/cvmose/` — reusable utilities.
- `docs/` — durable reasoning: problem attribution, insight, method mapping, experiment protocol.
- `configs/` — example path/env configuration only.

## Remote policy

b101 is for running experiments, not for storing full local analysis artifacts. Use:

```bash
scripts/sync_code_b101.sh
```

This syncs code/config only to `${CVMOSE_CODE_ROOT:-/data1/yuzhixiang/cv_mosev2/cvMOSE}` and excludes `docs/`, data, checkpoints, logs, contact sheets, and submission artifacts.

## Git identity

Repository commits use:

- user: `Po1ntu04`
- email: `2352190@tongji.edu.cn`

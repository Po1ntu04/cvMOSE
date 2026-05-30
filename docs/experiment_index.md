# Experiment Index and Repository Map

**Best next improvement:** keep SAM2/M11 as the final-quality backbone, then build a training-free re-anchor tracker with uncertainty management, proposal retrieval, identity verification, and delayed commit.

This repository versions the runnable code, experiment reports, selected visual evidence, and smoke/full-run audits needed to understand the MOSEv2 attempts. Dataset frames, checkpoints, full remote prediction folders, and non-final candidate zips are intentionally not stored in git.

## Local zip policy

The corrected policy is: **only one local workspace should keep submission zips**. In this repo workspace, the only retained local zip is the current final candidate:

```text
submission_mosev2_final_m11_cycle.zip
```

Candidate zips were either left on b101 or not generated when smoke evidence was negative. This avoids mixing stale zip artifacts across local workspaces while still committing code, reports, audits, and visual sheets.

## Experiment summary

| method | core idea | uploaded evidence | strengths | weaknesses / decision |
|---|---|---|---|---|
| SAM2 baseline | Use first-frame GT mask with SAM2.1 video predictor. | `tools/infer_mosev2_sam2.py`, `scripts/run_b101_single_sam2.sh`, `scripts/run_b101_parallel_sam2.sh`, `docs/mose15_observations.md` | Best interface match: first-frame mask defines a specific instance; stable enough to be the reference. | Autoregressive memory can drift after occlusion, same-class contact, tiny/edge exits. |
| SAM3/SAM3.1 adapter | Try SAM3-style detector/tracker ideas and GT-mask/exemplar adapters. | `tools/infer_mosev2_sam31.py`, `tools/sam31_gt_mask_adapter.py`, `tools/infer_mosev2_sam31_public_boxes.py`, `docs/paper_insights_sam2_sam3.md`, `docs/research_program.md` | Useful conceptual source: presence, confirmation delay, positive/negative exemplars. | Public/simple route mismatches this task: it tends to solve concept discovery, not first-mask instance identity. Not final. |
| M1 visibility gate | Suppress suspicious visible masks using confidence/geometry continuity. | Planned in `docs/research_program.md`; prior visual lessons folded into M11/M3 docs. | Clear problem framing: wrong mask can be worse than empty. | Too easy to become threshold-only suppression; over-empty risk dominated. Not retained as main code path here. |
| M11 cycle gate | Tracklet-level identity rejection using cycle/consistency evidence. | `docs/assets/m11_cycle/*.jpg`, `docs/m11_m2_visual_comparisons.md`, `docs/m3_initial_audit.md`, final local zip. | Best current final: targets post-gap wrong tracklets while preserving most baseline behavior; prior result gave the most reliable gain. | Still mostly rejects bad continuations; it does not recover all true reappearances. |
| M2 reliable memory gate | Current frame may output, but only reliable frames write into SAM2 non-conditioning memory. | `tools/infer_mosev2_sam2_m2_memory_gate.py`, `scripts/run_b101_m2_memory_gate.sh`, `docs/m2_reliable_memory_gate.md`, `docs/m2_visual_analysis.md`, `docs/assets/m2_memory_gate/` | Scientifically clean: attacks autoregressive error accumulation at the memory write point; low VRAM and auditable. | Stable wrong same-class tracks can still pass geometry/stability; aggressive gating hurts tiny/fast targets. Diagnostic, not final. |
| M2-light | Softer memory gate to provide tri-state evidence for M3. | `scripts/run_b101_m2_light.sh`, `docs/m2_light_ablation.md`, M2-light audit consumed by M3. | Better as evidence source than final method: exposes likely-absent / output-only / memory-write states. | Still cannot solve identity after long occlusion; not independently sufficient. |
| M3 state selector | Candidate table over baseline/M2/M2-light/M11/SAM3 with explicit presence/identity state and negative banks. | `tools/apply_m3_state_select.py`, `scripts/run_b101_m3_state.sh`, `tools/validate_mose_submission.py`, `tools/make_m3_compare_sheets.py`, `docs/m3_experiment_report.md`, `docs/assets/m3_state/` | Most auditable framework: decisions are visible per source/object/frame; validates 418 provided outputs and 15 predicted videos. | Conservative/balanced import too many weak alternatives or output too many empties; surgical is safer but cannot recover targets. Not final. |
| M4 tiny crop | Rerun frozen SAM2 on per-object crop videos for small targets; use as M3 candidate source only. | `tools/infer_mosev2_sam2_tiny_crop.py`, `scripts/run_b101_tiny_crop.sh`, `scripts/run_b101_m4_tiny_state.sh`, `docs/m4_tiny_crop_experiment.md`, `docs/assets/m4_tiny_crop/safe_smoke/` | Helps diagnose feature-resolution limits and can tighten small backpack/person masks. | Decisive failure on `r13u5z4y`: crop improves resolution but not identity; after occlusion it follows correlated wrong evidence. Not final. |
| M5R/RAR scaffold | Reappearance-aware SAM2 controller: RCMS-lite pre-disappearance reservoir, stable/ambiguous/recovery state machine, delayed main-memory commit. | `src/cvmose/reanchor.py`, `tools/infer_mosev2_sam2_rar.py`, `scripts/run_b101_rar.sh`, `docs/m5r_rar_plan.md`, `docs/m5r_rar_smoke.md`, `docs/m5r_rar_review_and_comparison.md`, `docs/assets/m5r_rar/key_compare/` | Cleanest instrumentation for state/anchor/memory governance; review fixes made audit and launcher safer. | Phase A/B does not recover identity; visual comparison rejects it as final and points to retrieval anchors as mandatory. |

## Next primary direction

The next route is documented in `docs/reanchor_tracker_direction.md`, with external method support in `docs/external_method_study.md`: M11/M2/M3/M4 are useful constraints and proposal sources, but the missing layer is a re-acquisition loop that can verify and commit a recovered identity after occlusion.

## Why the current best route is not “just run M4/SAM3”

The failure that matters most is not mask sharpness; it is **which instance is being segmented after the target is hidden and later reappears near similar objects**. Crop reruns and SAM3-style detections can supply proposals, but they become harmful if they are treated as independent proof. The next meaningful method should therefore be proposal + verification:

1. propose reappearance candidates from SAM2/M11, tiny crop, or SAM3 adapter;
2. verify against the first-frame target and hard negatives;
3. only then allow anchor update, memory write, or final replacement.

## Repository path tree

```text
cvMOSE/
├── README.md
├── configs/
│   └── example.env
├── docs/
│   ├── experiment_index.md                  # this overview and path map
│   ├── experiment_log.md                    # chronological experiment log
│   ├── experiment_protocol.md               # required evidence/validation protocol
│   ├── mose15_observations.md               # semantic observations for the 15 target videos
│   ├── paper_insights_sam2_sam3.md          # SAM2/SAM3 paper-based task reasoning
│   ├── research_program.md                  # method roadmap M1-M6/M5R
│   ├── reanchor_tracker_direction.md        # next primary re-acquisition architecture
│   ├── external_method_study.md             # source-backed external method study for M5R
│   ├── m5r_rar_plan.md                      # accepted RAR mainline and ablation protocol
│   ├── m5r_rar_smoke.md                     # RAR b101 subset/full-15 smoke evidence
│   ├── m5r_rar_review_and_comparison.md     # RAR code review + real visual comparison verdict
│   ├── m2_reliable_memory_gate.md           # M2 mechanism and risks
│   ├── m2_visual_analysis.md                # M2 visual diagnosis
│   ├── m2_light_ablation.md                 # M2-light ablation record
│   ├── m2_code_review.md                    # M2 review/fix evidence
│   ├── m11_m2_visual_comparisons.md         # M11/M2 figure index
│   ├── m3_initial_audit.md                  # baseline/M2/M11 validation audit
│   ├── m3_experiment_report.md              # M3 full report and final selection rationale
│   ├── m4_tiny_crop_experiment.md           # M4 smoke report and rejection rationale
│   └── assets/
│       ├── m11_cycle/                       # selected M11 cycle-gate comparison figures
│       ├── m2_memory_gate/
│       │   ├── full/                        # SAM2 vs M2 full-frame sheets
│       │   ├── zooms/                       # zoomed M2 failure/success sheets
│       │   └── summary_table.jpg
│       ├── m3_state/
│       │   ├── conservative/{full,zooms}/   # all 15 video M3 conservative sheets
│       │   └── balanced/{full,zooms}/       # all 15 video M3 balanced sheets
│       ├── m4_tiny_crop/safe_smoke/
│       │   ├── audits/                      # tiny-crop and M4 smoke JSON audits
│       │   ├── full/                        # 3-video M4 full-frame sheets
│       │   ├── zooms/                       # 3-video M4 zoom sheets
│       │   └── sheet_index.json
│       └── m5r_rar/
│           ├── smoke_compare/{full,zoom}/   # 2-video RAR smoke sheets
│           └── key_compare/{full,zoom}/     # 8 key-video RAR visual comparison sheets
├── scripts/
│   ├── sync_code_b101.sh                    # code-only sync to b101
│   ├── run_b101_single_sam2.sh              # baseline single-run launcher
│   ├── run_b101_parallel_sam2.sh            # baseline parallel launcher
│   ├── run_b101_m2_memory_gate.sh           # M2 full/smoke launcher
│   ├── run_b101_m2_light.sh                 # M2-light ablation launcher
│   ├── run_b101_m3_state.sh                 # M3 selector launcher
│   ├── run_b101_tiny_crop.sh                # M4 candidate-source launcher, no submission
│   ├── run_b101_m4_tiny_state.sh            # M4 wrapper: requires tiny source by default
│   └── run_b101_rar.sh                      # M5R/RAR RCMS-lite and state-machine launcher
├── tools/
│   ├── infer_mosev2_sam2.py                 # baseline SAM2 inference
│   ├── infer_mosev2_sam2_m2_memory_gate.py  # M2 memory-write gate
│   ├── infer_mosev2_sam2_tiny_crop.py       # M4 tiny-crop candidate generator
│   ├── infer_mosev2_sam2_rar.py             # M5R/RAR SAM2 entrypoint
│   ├── infer_mosev2_sam31.py                # SAM3.1 adapter route
│   ├── infer_mosev2_sam31_public_boxes.py   # SAM3.1 public/simple control route
│   ├── sam31_gt_mask_adapter.py             # GT-mask adapter helpers
│   ├── apply_m3_state_select.py             # M3/M4 candidate selector
│   ├── make_m3_compare_sheets.py            # visual comparison sheet generator
│   ├── validate_mose_submission.py          # hard submission invariant checker
│   ├── build_submission.py                  # submission packaging helper
│   └── launch_parallel_infer.py             # parallel inference helper
├── src/cvmose/reanchor.py                   # RAR state/anchor/commit primitives
└── submission_mosev2_final_m11_cycle.zip    # final zip retained in this local workspace only
```

## Final/current stance

- Keep `submission_mosev2_final_m11_cycle.zip` as the current final local submission artifact.
- Keep M2/M3/M4 code and visual evidence as analysis infrastructure.
- Next experiment should not be another raw threshold sweep; it should add independent identity verification for reappearance proposals.

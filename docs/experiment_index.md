# Experiment Index and Repository Map

**Best next improvement:** keep SAM2/M11 as the final-quality backbone, then build a training-free re-anchor tracker with uncertainty management, proposal retrieval, identity verification, and delayed commit.

This repository versions the runnable code, experiment reports, selected visual evidence, and smoke/full-run audits needed to understand the MOSEv2 attempts. Dataset frames, checkpoints, full remote prediction folders, and candidate submission zips are intentionally not stored in git.

## Local zip policy

The corrected policy is: **submission zips live in one data workspace, not scattered across repo checkouts**.

In the git repo workspace, the only retained legacy zip is:

```text
submission_mosev2_final_m11_cycle.zip
```

Current M6 candidate zips are kept outside git in:

```text
/home/yu/projects/cv/from fdu/MOSEv2/homework/
├── submission_mosev2_m6_safe.zip
├── submission_mosev2_m6_balanced.zip
└── submission_mosev2_m6_aggressive.zip
```

They are documented and validated in `docs/m6_fusion_report.md`, but not committed. This preserves the "one local data workspace for zip artifacts" rule while still committing code, reports, and small visual sheets.

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
| M5R-C retrieval re-anchor | High-recall candidate pool + SAM2 image/RGB object descriptors + hard negative margin + verified `add_new_mask` repropagation. | `tools/infer_mosev2_sam2_reanchor.py`, `scripts/run_b101_m5r_reanchor.sh`, `scripts/make_m5r_reanchor_compare.py`, `docs/m5r_reanchor_experiment.md`, `docs/assets/m5r_reanchor/key_compare/` | First implementation that truly re-drives SAM2 from later candidate anchors; audits expose candidate/positive/negative decisions. | Current descriptors/proposals are not reliable enough in same-class reappearance; key visual smoke rejects it as final. |
| M6 Phase 1 official ceiling | Test public FudanCVL MOSEv2-finetuned SAM2 checkpoints/submission zips as ceiling/reference. | `tools/infer_mosev2_sam2_official_ckpt.py`, `scripts/run_b101_official_ckpt.sh`, `docs/m6_official_ceiling.md`, `docs/assets/m6_official/compare/` | Establishes score ceiling and confirms public MOSEv2 tricks can change 15-video behavior. | Rule-dependent: MOSEv2-finetuned checkpoints/submissions may be disallowed as final; still fails some identity cases. |
| M6 Phase 2 DAM4SAM/d4sm | Use external distractor-aware trackers as off-the-shelf candidate roots. | `tools/infer_mosev2_dam4sam_adapter.py`, `scripts/run_b101_dam4sam_adapter.sh`, `docs/m6_dam4sam_smoke.md`, `docs/assets/m6_dam4sam/smoke/` | Good diagnostic for whether external tracker memory avoids distractors. | Smoke rejected: q0sizv6m/msinig6m still produce large wrong same-class/composite masks; not used in fusion. |
| M6 Phase 3 SAM2Long | Test memory-tree/branching SAM2Long as training-free alternative to hand-written branch logic. | `tools/infer_mosev2_sam2long_adapter.py`, `scripts/run_b101_sam2long_adapter.sh`, `docs/m6_sam2long_smoke.md`, `docs/assets/m6_sam2long/smoke/` | Directly targets greedy-memory error accumulation. | Smoke rejected: branch diversity did not solve identity and sometimes amplified same-class drift. |
| M6 Phase 4 SAAS | Prepare SAAS adapter for multi-shot robustness/scene-change candidate source. | `tools/infer_mosev2_saas_adapter.py`, `scripts/run_b101_saas_adapter.sh`, `docs/m6_saas_smoke.md` | Non-vendored harness is ready if a reproducible checkpoint URL is provided. | No smoke promoted: public checkpoint was not obtained scriptably; no source used in fusion. |
| M6 Phase 5 DINO-reanchor | Replace weak descriptors with frozen DINOv2 masked object descriptor + hard-negative margin + delayed promotion/rollback. | `src/cvmose/dino_descriptors.py`, `tools/precompute_dino_features.py`, updated `tools/infer_mosev2_sam2_reanchor.py`, `docs/m6_dino_reanchor_report.md`, `docs/assets/m6_dino/smoke/` | Best training-free identity verifier so far; prevents accepting obvious wrong anchors and plausibly improves `2smf7uq9`. | Candidate generation is still the bottleneck; not a standalone final root. |
| M6 Phase 6 safe fusion | Per-video conservative policy over M11, official ceiling, and DINO-reanchor candidates. | `tools/apply_m6_safe_fusion.py`, `docs/m6_fusion_report.md`, `docs/assets/m6_fusion/summary/` | Produces three validated candidate zips while preserving all 418 provided outputs unchanged. | Safe remains M11-dominated; balanced only accepts DINO for two videos; aggressive is rule-dependent. |
| M7 Qwen-VL verifier | Use Qwen-VL / Aliyun Bailian as cached target profiler, candidate verifier, and delayed-promotion judge for M5R-C. | `src/cvmose/qwen_vl_client.py`, `src/cvmose/mllm_panels.py`, `tools/mllm_target_profile.py`, `tools/mllm_candidate_judge.py`, `tools/mllm_tracklet_judge.py`, `docs/m7_qwen_vl_verifier.md`, `docs/m7_qwen_fusion_report.md` | Real `qwen3.5-plus` works; 16 target profiles + 24 merged key-video candidate judgments; strict veto is safe/no-change, and `1qlssuz2` support is visually low-risk. | Safe fusion is M11-equivalent; balanced fusion replaces only `1qlssuz2` and validates. Expected gain is small; M7 is a verifier, not the missing high-recall reappearance generator. |
| M8 retrieval re-anchor | High-recall candidate-pool audit + conservative source fusion + DINO/M5R re-anchor with M11 fallback safety. | `src/cvmose/candidate_pool.py`, `tools/build_m8_candidate_pool.py`, `tools/apply_m8_candidate_fusion.py`, updated `tools/infer_mosev2_sam2_reanchor.py`, `docs/m8_retrieval_reanchor_report.md`, `docs/assets/m8_reanchor/` | Fixes a key structural issue: failed re-anchor now fails closed to M11 via `--fallback-root`; produces safe/balanced/aggressive validated M8 zips. | Still does not solve true reappearance recall on r13/q0/msi; balanced/aggressive are score probes, not safe finals. |

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
│   ├── m5r_reanchor_experiment.md           # M5R-C retrieval-anchor implementation and rejection verdict
│   ├── m6_initial_ceiling_and_plan.md        # M6 branch baseline, constraints, and run plan
│   ├── m6_official_ceiling.md                # FudanCVL checkpoint/submission ceiling study
│   ├── m6_dam4sam_smoke.md                  # DAM4SAM/d4sm smoke verdict
│   ├── m6_sam2long_smoke.md                 # SAM2Long memory-tree smoke verdict
│   ├── m6_saas_smoke.md                     # SAAS adapter/checkpoint status
│   ├── m6_dino_reanchor_report.md           # DINOv2 descriptor M5R-C upgrade
│   ├── m6_fusion_report.md                  # safe/balanced/aggressive fusion candidates
│   ├── m7_qwen_vl_verifier.md               # Qwen-VL verifier design and real API evidence
│   ├── m7_qwen_fusion_report.md             # validated M7 safe/balanced fusion candidates
│   ├── m8_retrieval_reanchor_report.md      # M8 candidate-pool, fallback-safe reanchor, and fusion report
│   ├── m7_qwen_vl_setup.md                  # Bailian/OpenAI-compatible smoke setup result
│   ├── m7_target_profile_summary.md         # target profiler dry-run route table
│   ├── m7_candidate_judge_summary.md        # candidate judge dry-run panel summary
│   ├── m7_tracklet_judge_summary.md         # tracklet judge dry-run panel summary
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
│       ├── m5r_rar/
│       │   ├── smoke_compare/{full,zoom}/   # 2-video RAR smoke sheets
│       │   └── key_compare/{full,zoom}/     # 8 key-video RAR visual comparison sheets
│       ├── m5r_reanchor/key_compare/        # 8 key-video M5R-C visual comparison sheets
│       ├── m6_official/compare/             # official checkpoint/submission comparison sheets
│       ├── m6_dam4sam/smoke/                # DAM4SAM/d4sm smoke sheets
│       ├── m6_sam2long/smoke/               # SAM2Long smoke sheets
│       ├── m6_dino/smoke/                   # DINO-reanchor smoke sheets
│       └── m6_fusion/summary/               # final per-video fusion sheets
├── scripts/
│   ├── sync_code_b101.sh                    # code-only sync to b101
│   ├── run_b101_single_sam2.sh              # baseline single-run launcher
│   ├── run_b101_parallel_sam2.sh            # baseline parallel launcher
│   ├── run_b101_m2_memory_gate.sh           # M2 full/smoke launcher
│   ├── run_b101_m2_light.sh                 # M2-light ablation launcher
│   ├── run_b101_m3_state.sh                 # M3 selector launcher
│   ├── run_b101_tiny_crop.sh                # M4 candidate-source launcher, no submission
│   ├── run_b101_m4_tiny_state.sh            # M4 wrapper: requires tiny source by default
│   ├── run_b101_rar.sh                      # M5R/RAR RCMS-lite and state-machine launcher
│   ├── run_b101_m5r_reanchor.sh             # M5R-C candidate retrieval re-anchor launcher
│   ├── make_m7_qwen_compare.py              # M7 Qwen verifier visual sheet generator
│   ├── make_m6_compare.py                   # reusable M6/M8 multi-root visual sheet generator
│   ├── run_b101_official_ckpt.sh            # FudanCVL MOSEv2 checkpoint launcher
│   ├── run_b101_dam4sam_adapter.sh          # external DAM4SAM/d4sm adapter launcher
│   ├── run_b101_sam2long_adapter.sh         # external SAM2Long adapter launcher
│   ├── run_b101_saas_adapter.sh             # external SAAS adapter launcher
│   └── make_m5r_reanchor_compare.py         # M5R-C visual sheet generator
├── tools/
│   ├── infer_mosev2_sam2.py                 # baseline SAM2 inference
│   ├── compare_candidate_roots.py           # multi-root submission invariant/candidate comparison
│   ├── infer_mosev2_sam2_m2_memory_gate.py  # M2 memory-write gate
│   ├── infer_mosev2_sam2_tiny_crop.py       # M4 tiny-crop candidate generator
│   ├── infer_mosev2_sam2_rar.py             # M5R/RAR SAM2 entrypoint
│   ├── infer_mosev2_sam2_reanchor.py        # M5R-C/M8 candidate retrieval + add_new_mask repropagation + M7 MLLM gate + fallback root
│   ├── build_m8_candidate_pool.py           # M8 high-recall candidate-pool audit
│   ├── apply_m8_candidate_fusion.py         # M8 conservative source-fusion builder
│   ├── test_qwen_vl_bailian.py              # Qwen-VL / Bailian OpenAI-compatible smoke test
│   ├── mllm_target_profile.py               # M7 target profile panel + JSON generation
│   ├── mllm_candidate_judge.py              # M7 candidate judge panel + JSON generation
│   ├── mllm_tracklet_judge.py               # M7 tracklet judge panel + JSON generation
│   ├── infer_mosev2_sam2_official_ckpt.py   # FudanCVL MOSEv2 checkpoint wrapper
│   ├── infer_mosev2_dam4sam_adapter.py      # DAM4SAM/d4sm wrapper
│   ├── infer_mosev2_sam2long_adapter.py     # SAM2Long wrapper
│   ├── infer_mosev2_saas_adapter.py         # SAAS wrapper
│   ├── apply_m6_safe_fusion.py              # M6 per-video fusion builder
│   ├── precompute_dino_features.py          # DINO descriptor cache utility
│   ├── infer_mosev2_sam31.py                # SAM3.1 adapter route
│   ├── infer_mosev2_sam31_public_boxes.py   # SAM3.1 public/simple control route
│   ├── sam31_gt_mask_adapter.py             # GT-mask adapter helpers
│   ├── apply_m3_state_select.py             # M3/M4 candidate selector
│   ├── make_m3_compare_sheets.py            # visual comparison sheet generator
│   ├── validate_mose_submission.py          # hard submission invariant checker
│   ├── build_submission.py                  # submission packaging helper
│   └── launch_parallel_infer.py             # parallel inference helper
├── src/cvmose/
│   ├── reanchor.py                          # RAR state/anchor/commit primitives
│   ├── dino_descriptors.py                  # frozen DINOv2 masked object descriptors
│   ├── qwen_vl_client.py                    # cached Bailian/OpenAI-compatible Qwen-VL JSON client
│   ├── candidate_pool.py                    # M8 training-free candidate-pool utilities
│   └── mllm_panels.py                       # standardized panels for target/candidate/tracklet verification
└── submission_mosev2_final_m11_cycle.zip    # final zip retained in this local workspace only
```

## Final/current stance

- Safest current submission: `submission_mosev2_m8_candidate_pool_key7_safe_v2.zip` in the MOSEv2 homework workspace; it is M11-dominated, validates 418 provided outputs unchanged, and only imports the Qwen-supported `lcgc29va` tiny recovery.
- Best training-free probe: `submission_mosev2_m8_balanced_reanchor_probe.zip`; it keeps M11 by default, keeps the safe tiny recovery, and probes DINO-reanchor only on visually non-catastrophic videos.
- Aggressive/rule-dependent probe: `submission_mosev2_m8_aggressive_dino_probe.zip`; it additionally tests the later-window `lcgc29va` DINO anchor and should be treated as score probing only.
- Next experiment should still avoid raw threshold sweep; the real missing capability is higher-recall global later-frame proposal generation plus DINO/SAM identity verification, MLLM semantic veto/support, and delayed prompt injection.


## M7 overlay/source-select follow-up

- `docs/m7_overlay_source_select_report.md` records the filled-overlay hallucination bug, the outline-panel fix, guarded source-select ablation, validated zips, and visual evidence paths.

## M9 — Qwen-box recovery smoke

- Report: `docs/m9_qwen_box_recovery_report.md`
- Proposal summary: `docs/m9_qwen_box_proposals_smoke.md`
- Visuals: `docs/assets/m9_qwen_boxes/`
- Core result: Qwen is useful for absence/veto reasoning, but direct coordinate-box output plus SAM2 box prompt is not final-quality; clipping prevents leakage but does not solve identity.

## M10 — Qwen3.6 verifier-to-anchor coupling fix

- Report: `docs/m10_qwen36_reanchor_report.md`
- Setup: `docs/m9_qwen36_setup.md`
- Summaries: `docs/m10_qwen36_target_profile_summary.md`, `docs/m10_qwen36_candidate_judge_key_summary.md`, `docs/m10_qwen36_tracklet_judge_key_full_summary.md`
- Visuals: `docs/assets/m10_qwen36/confirm_compare/`
- Core result: fixed the main implementation bottleneck where Qwen-judged frames were not prioritized in candidate generation and verified tracklets were still rejected by duplicate delayed-confirm logic. This enables real anchor insertion for `q0sizv6m:obj2`, `msinig6m:obj1`, and `4vznweiu:obj1`; after bounded rollback, only `q0sizv6m:obj2` remains a visually plausible probe over `m7part-balanced`.
- Validated zips in local MOSEv2 workspace:
  - submit first: `submission_mosev2_m7_part_balanced.zip` / repo copy `submission_mosev2_m7_part_balanced_submit.zip`
  - probe: `submission_mosev2_m10_qwen36_q0only.zip`
  - aggressive probe: `submission_mosev2_m10_qwen36_q0_4v.zip`

## M14 — same-class kangaroo re-ID probe

- Report: `docs/m14_amfdu83t_kangaroo_reid.md`
- Qwen split note: `docs/m14_amfdu83t_qwen36_split.md`
- New tool: `tools/mllm_sameclass_atlas.py`
- SAM2 teach-box update: `tools/infer_mosev2_sam2_teach_boxes.py --clip-mode nearest`
- Visuals: `docs/assets/m14_amfdu83t_sources_a/`, `docs/assets/m14_amfdu83t_sources_b/`, `docs/assets/m14_amfdu83t_box_probes/`, `docs/assets/m14_amfdu83t_tight_probes_a/`, `docs/assets/m14_amfdu83t_tight_probes_b/`, `docs/assets/m14_amfdu83t_temporal_8_11_large/`
- Validated probe zip: `submission_mosev2_m14_zofficial_amfdu_temporal811.zip` and MOSE workspace copy `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m14_zofficial_amfdu_temporal811.zip`.

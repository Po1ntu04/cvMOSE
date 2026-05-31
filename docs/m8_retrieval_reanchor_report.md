# M8 Retrieval Re-anchor Implementation Report

**One-line conclusion:** the most meaningful path is still not a broader threshold sweep; it is a high-recall candidate pool guarded by hard negatives / DINO / MLLM and committed only on top of the M11 safety baseline.

## Objective

M8 implements the next layer after M7: instead of asking the verifier to choose among a narrow set of masks, first build a broader candidate pool from existing roots/components, then only promote/copy candidates when identity evidence is strong enough.  The goal is to maximize true reappearance recovery without losing M11's safe rejection behavior.

Hard invariants:

- no training / no finetune / no future GT;
- MLLM and DINO are verifiers, not final mask generators;
- 418 provided-output videos remain unchanged in every zip;
- large prediction roots and zips stay in the MOSEv2 homework workspace, not git.

## Implemented code

| file | role |
|---|---|
| `src/cvmose/candidate_pool.py` | shared mask stats, connected components, cheap descriptor, source parsing, candidate records. |
| `tools/build_m8_candidate_pool.py` | high-recall pool audit over baseline/M11/M5R/M7/official/DAM/SAM2Long/etc. roots plus cached Qwen judgments. |
| `tools/apply_m8_candidate_fusion.py` | conservative output-level source fusion from accepted M8 records, with same-class and empty-default guards. |
| `tools/infer_mosev2_sam2_reanchor.py` | upgraded with `--fallback-root`, so DINO/M5R re-anchor can use M11 as the no-anchor/rollback base instead of silently falling back to baseline. |
| `scripts/run_b101_m5r_reanchor.sh` | allowlisted `--fallback-root` for remote b101 experiments. |

The important engineering fix is `--fallback-root`: previous M5R-C/DINO smoke outputs copied baseline whenever no anchor was accepted.  That made a failed recovery lose M11's hard-won safety.  With `--fallback-root pred_sam2_m11_cycle`, re-anchor experiments can fail closed.

## Candidate-pool audit: key 7 videos

Command class:

```bash
python tools/build_m8_candidate_pool.py \
  --workspace "$WS" \
  --default-root artifacts/m5r_reanchor/source_preds/m11 \
  --source-root baseline="$WS/homework/pred_sam2_b101" \
  --source-root official_large="$WS/homework/pred_official_submission_large_15only" \
  --source-root official_bplus="$WS/homework/pred_official_submission_bplus_15only" \
  --source-root rar_state="artifacts/m5r_reanchor/source_preds/rar_state" \
  --source-root m5r="artifacts/m5r_reanchor/key_pred" \
  --source-root dino_m5r="$WS/homework/pred_sam2_m5r_dino_key" \
  --source-root m7_balanced_plus="$WS/homework/pred_m7_qwen_balanced_plus_outline" \
  --source-root m6_balanced="$WS/homework/pred_m6_balanced" \
  --source-root dam="$WS/homework/pred_m6_dam4sam_dam4sam_single_per_obj_smoke" \
  --source-root sam2long_np2="$WS/homework/pred_m6_sam2long_np2_iou03_u1_smoke" \
  --judgments-json artifacts/m7_qwen_vl/candidate_judgments_outline_4v.json \
  --judgments-json artifacts/m7_qwen_vl/candidate_judgments_outline_lcgc.json \
  --judgments-json artifacts/m7_qwen_vl/candidate_judgments_real_merged.json \
  --judgments-json artifacts/m7_qwen_vl/candidate_judgments_real_tiny_semantic.json
```

Audit output:

```text
/home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m8_candidate_pool/m8_candidate_pool_key7.json
/home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m8_candidate_pool/m8_candidate_pool_key7.csv
```

Summary: 7 videos / 10 objects / 300 top CSV rows / 51 accepted records before final fusion guards.

| video | accepted before final guards | interpretation |
|---|---:|---|
| `r13u5z4y` | 0 | good: the pool did not find a trustworthy post-gap strawberry anchor.  This avoids repeating the earlier wrong-strawberry failure. |
| `q0sizv6m` | obj1 0, obj2 4 | raw accepted candidates existed, but visual zoom showed same-class animal risk; final safe policy rejects unverified same-class replacements. |
| `msinig6m` | 0 | good: hard negatives prevent person/koala composites from being promoted. |
| `lcgc29va` | 17 | useful tiny-target candidates; only Qwen-supported, M11-empty frames 8/9 survive safe fusion. |
| `4vznweiu` | 0 | same-class/tiny letter-object ambiguity remains too high without stronger evidence. |
| `1qlssuz2` | 30 | many visually plausible car candidates; useful for balanced probes. |
| `2smf7uq9` | 0 | candidate pool remained conservative; DINO-reanchor separately provides a non-catastrophic probe. |

Key visual evidence:

```text
docs/assets/m8_candidate_pool/key7_fusion/q0sizv6m_obj2_f5_zoom.jpg
docs/assets/m8_candidate_pool/key7_fusion/lcgc29va_m6_compare.jpg
docs/assets/m8_candidate_pool/key7_fusion/1qlssuz2_m6_compare.jpg
```

## M8 safe source fusion

Candidate zip:

```text
/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m8_candidate_pool_key7_safe_v2.zip
```

Validation:

```text
ok=true, video_dirs=433, pngs=66526, provided_changed_count=0, predicted_error_count=0
```

Policy:

- default root: M11;
- only copy candidates when M11 is empty by default;
- same-class dense videos require Qwen support;
- stable guard videos are not replaced;
- applied changes: `lcgc29va` object 1, frames 8 and 9 from `rar_state`, with Qwen support.

This is safe but small; it essentially preserves M11 and only imports the previously validated tiny-target recovery.

## Remote DINO-reanchor smoke

First smoke, baseline fallback:

```text
pred:  /home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m8_dino_reanchor_key7
audit: /home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m8_reanchor/m8_dino_reanchor_key7.json
```

Summary:

| video | accepted anchors | changed vs SAM2 baseline | visual / audit verdict |
|---|---:|---:|---|
| `r13u5z4y` | 0 | 0 | identity verifier safely refuses wrong strawberry anchors, but no recovery. |
| `q0sizv6m` | 0 | 0 | safely refuses risky animal anchors. |
| `msinig6m` | 0 | 0 | safely refuses composite person/koala anchors. |
| `lcgc29va` | 1 | 15 | later-window tiny anchor changes too many frames; not safe without score probe. |
| `4vznweiu` | 1 | 7 | visual difference is small and not clearly beneficial. |
| `1qlssuz2` | 1 | 13 | plausible low-risk car re-anchor; score-probe candidate. |
| `2smf7uq9` | 1 | 13 | plausible but same-class flamingo ambiguity remains; score-probe only. |

Visual evidence:

```text
docs/assets/m8_reanchor/key7_smoke/
docs/assets/m8_reanchor/final_candidates/
```

Important lesson: when no anchor was accepted, this first smoke copied baseline, so it lost M11 safety on r13/msi in root-level comparisons.  The code is now fixed with `--fallback-root`; the second b101 M11-base smoke completed with the same 4 accepted anchors but `changed_vs_output_base=48` and no safety loss on no-anchor videos.

M11-base smoke output:

```text
pred:  /home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m8_dino_reanchor_m11base_key7
audit: /home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m8_reanchor/m8_dino_reanchor_m11base_key7.json
summary: videos=7, frames=374, accepted_anchor_count=4, changed_vs_baseline=96, changed_vs_output_base=48
VRAM: about 7.2 GiB during the smoke, observed by nvidia-smi.
```

## Generated M8 candidate zips

All zips are stored in the MOSEv2 homework workspace and pass `tools/validate_mose_submission.py`.

| candidate | path | changed vs M11 on 15 target videos | validation | recommendation |
|---|---|---:|---|---|
| safe | `submission_mosev2_m8_candidate_pool_key7_safe_v2.zip` | 2 frames | 433 dirs / 66526 png / 418 unchanged | safest M8; equivalent to M7 tiny-source select behavior. |
| balanced probe | `submission_mosev2_m8_balanced_reanchor_probe.zip` | 35 frames | 433 dirs / 66526 png / 418 unchanged | submit as score probe if a second submission is allowed; M11 default + lcgc safe + DINO on 1ql/2sm/4v. |
| aggressive probe | `submission_mosev2_m8_aggressive_dino_probe.zip` | 48 frames | 433 dirs / 66526 png / 418 unchanged | not safe recommendation; tests whether later-window lcgc DINO anchor helps or hurts. |

Validation JSONs:

```text
/home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m8_candidate_pool/validate_m8_candidate_pool_key7_safe_v2.json
/home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m8_reanchor/validate_m8_balanced_reanchor_probe.json
/home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m8_reanchor/validate_m8_aggressive_dino_probe.json
```

## Per-video qualitative verdict

| video | verdict | reason |
|---|---|---|
| `r13u5z4y` | unresolved / safe empty favored | DINO and M8 pool correctly reject wrong strawberry but still lack a true reappearance candidate. |
| `q0sizv6m` | unresolved / safe M11 favored | high recall finds animal-like masks, but same-class identity is not reliable; Qwen/DINO guards should veto rather than promote. |
| `msinig6m` | safe M11 favored | DINO margin correctly collapses on composite person/koala masks; no recovery anchor. |
| `lcgc29va` | small safe improvement | frame 8/9 tiny candidate from Qwen-supported pool is the only safe import; DINO later-window anchor is aggressive. |
| `4vznweiu` | neutral | DINO changes are tiny and not clearly better; keep only in probe. |
| `1qlssuz2` | plausible balanced probe | DINO/SAM3-like candidate remains on the same car trajectory; likely low-risk but not visually decisive. |
| `2smf7uq9` | plausible but ambiguous probe | DINO re-anchor stays near flamingo instance; same-class ambiguity means probe only. |

## Why this still may not reach 44

M8 improves the system architecture but exposes the real remaining bottleneck: **candidate recall for the true object after hard occlusion is still missing**.  On `r13u5z4y`, `q0sizv6m`, and `msinig6m`, the safe verifier is doing the right thing by refusing wrong anchors.  The score ceiling will not move much until a global later-frame candidate generator reliably includes the true reappearing instance, not merely plausible same-class objects.

The next substantive move should therefore be one of:

1. a real detector/proposal source with higher recall in recovery windows, then DINO/Qwen hard-negative validation;
2. reverse propagation from later high-confidence frames into the occlusion boundary;
3. a small manually auditable per-video policy only if allowed by the assignment, with explicit visual proof.

M8's practical contribution is that these stronger sources can now be plugged into an auditable candidate-pool / fallback-safe reanchor pipeline without losing M11 safety.

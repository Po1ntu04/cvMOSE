# M3 State/Re-anchor Experiment Report

Date: 2026-05-30  
Branch: `method/m3-state-reanchor`  
Goal: training-free improvement for MOSEv2 homework 15 videos under strict 418-provided-output invariance.

## Goal and constraints

M3 targets the real task: first-frame-mask-specified **instance identity preservation** under small targets, occlusion/reappearance, same-class distractors, edge truncation, and multi-object conflicts. It does not train/finetune, does not use hidden later labels, and does not hardcode per-video winning sources.

Hard validation gates:

- candidate zip has 433 video dirs / 66526 PNGs;
- `zipfile.testzip() == None`;
- 418 provided-output videos unchanged;
- 15 predicted videos match frame names, counts, image sizes, and first-frame label id set.

## Implemented stages

### Stage 1 — validator and initial audit

Implemented `tools/validate_mose_submission.py` and wrote `docs/m3_initial_audit.md`.

Validated existing artifacts:

| artifact | validation |
|---|---|
| SAM2 baseline | ok; 433 dirs / 66526 PNGs / provided_changed=0 / pred_errors=0 |
| M2 memory gate | ok; 433 dirs / 66526 PNGs / provided_changed=0 / pred_errors=0 |
| M11 cycle gate | ok; 433 dirs / 66526 PNGs / provided_changed=0 / pred_errors=0 |

### Stage 2 — M2-light standalone ablation

Implemented `scripts/run_b101_m2_light.sh` and tri-state audit in `tools/infer_mosev2_sam2_m2_memory_gate.py`.

See `docs/m2_light_ablation.md`.

### Stage 3 — M3-state candidate table and rule selector

Implemented `tools/apply_m3_state_select.py` and `scripts/run_b101_m3_state.sh`.

Candidate sources:

- `baseline`: original SAM2.1-B+ prediction;
- `m2`: M2 reliable-memory-gate prediction;
- `m2_light`: M2-light prediction and tri-state audit;
- `m11`: M11 cycle-gate prediction and suppression audit;
- `sam31`: optional SAM3.1 adapter prediction if present.

For each frame/object/source, the audit records:

- present / area / area fraction / bbox / centroid / edge touch;
- area ratio to first-frame target and last confirmed target;
- displacement from last confirmed target;
- source agreement IoUs;
- appearance to identity bank;
- appearance to hard negative / other-object bank;
- M2-light likely-absent and M11 cycle suppression evidence;
- candidate flags and final candidate class.

Decision is intentionally rule-tiered rather than a weighted `final_score`.

## Deviations from V2 plan

This section is mandatory. Every deviation must explain why, risk, and validation.

1. **M11 source import used artifact/audit path first, not `src/cvmose/cycle_verify.py`.**
   - Why: M11 branch artifacts and audit are already validated; copying only the full verifier would add dependency/code-surface risk before the M3-state selector is validated.
   - Risk: M3-state cannot recompute new cycle evidence for candidate roots that lack M11 outputs; it only consumes existing M11 suppression evidence.
   - Validation: M11 zip was validated with 433 dirs / 66526 PNGs / provided_changed=0; M3 audit records whether `m11-root` and `m11-audit-json` are actually present.

2. **M3-state first implementation does not rerun SAM2 pseudo-anchor re-prompt (M3R).**
   - Why: V2 plan says M3-state must first produce `accepted_anchor_candidate`; rerunning SAM2 with anchors is only safe after the state selector is inspectable.
   - Risk: M3-state can suppress or select candidate masks but cannot repair cases where all candidate sources miss the correct reappearance.
   - Validation: baseline and 418 invariants are preserved by output-level selection; accepted-anchor frames are recorded for later M3R eligibility.

3. **Tiny crop is not directly activated in the first M3-state run.**
   - Why: V2 plan explicitly says tiny crop should be a candidate source after state/candidate evidence; adding it before selector validation would conflate mechanisms.
   - Risk: extremely small targets (`amfdu83t`, `lcgc29va`, `1qlssuz2`) may remain unrecovered if all existing candidates are empty/wrong.
   - Validation: M3 audit records empty outputs and tiny/edge flags so a tiny-crop follow-up can target only structurally justified frames.

4. **Appearance uses simple masked RGB histograms rather than learned DINO/SAM features.**
   - Why: training-free and dependency-minimal; RGB histograms are robustly available on b101 and sufficient to instantiate identity/negative banks audibly.
   - Risk: color histograms are weak for same-color distractors, illumination change, and partial masks.
   - Validation: decisions never rely on appearance alone; they require source agreement / M2-light absence / M11 suppression tiers, and the report/visual sheets inspect high-risk disagreements.


5. **Added `m3_state_surgical` after visual review of conservative/balanced.**
   - Why: visual evidence showed the first M3-state variants were too willing to import alternative source masks and too often over-empty, so a baseline-preserving rejector was needed to isolate the safest part of M3.
   - Risk: surgical can still over-suppress baseline frames and cannot recover missed targets.
   - Validation: surgical passed the same strong validator; visual/statistical review showed it is still not strong enough to replace M11/SAM2 final.

6. **Skipped M3R in this goal run.**
   - Why: V2 requires M3R only after trustworthy accepted anchors. Conservative/balanced accepted anchors are not visually trustworthy; surgical intentionally produces no pseudo-anchors.
   - Risk: current branch does not solve cases where all existing candidate sources miss true reappearance.
   - Validation: baseline, M11, M2-light, and all M3 zips passed invariants; final selection avoids unvalidated M3R changes.

7. **M4 tiny-crop was attempted only as a smoke-tested candidate source, not as a full final submission.**
   - Why: smoke visual evidence showed a decisive negative case on `r13u5z4y`: crop rerun followed correlated wrong evidence after occlusion rather than recovering identity.
   - Risk: no full M4 zip means the branch does not claim global improvement; however this avoids wasting a full submission on a mechanism already shown unsafe.
   - Validation: `--require-tiny-crop` verified exact 3-video frame coverage; tiny-crop audit recorded runtime/VRAM; comparison sheets and audits are committed under `docs/assets/m4_tiny_crop/safe_smoke/`.


## Upstream evidence notes

Evidence gathered from official/upstream sources and used only as mechanism context, not as hidden supervision:

1. **SAM2 fact:** SAM2 unifies image/video promptable segmentation and video processing through a streaming memory design (SAM2 paper / official repository / Meta blog).
2. **SAM2 fact:** SAM2 video prediction is not independent per-frame segmentation; its memory encoder, memory bank, and memory attention use previous frames/interactions, which directly motivates memory-quality gates.
3. **SAM2 fact:** SAM2 includes an occlusion/object-presence head for temporarily invisible objects, supporting the M3 choice to model visibility as a latent state rather than treating every non-empty mask as valid.
4. **SAM2.1 fact + task inference:** official SAM2.1 notes mention improved robustness for visually similar objects and occlusions; for this homework, the remaining failures concentrate exactly in same-class distractor and long-occlusion regimes.
5. **SAM3 fact:** SAM3 concept prompts can include text and image exemplars and returns masks/unique identities for matching instances; this supports borrowing positive/negative exemplar logic without making SAM3 public-simple the main route.
6. **SAM3 fact:** SAM3 supports positive/negative clicks and exemplar-conditioned detection/presence; M3 mirrors this at the system level through identity and hard-negative banks.
7. **Task inference:** neither SAM2 nor SAM3 official public interfaces give an explicit motion-model module for this setting, so M3 treats motion/area/appearance as training-free external evidence around the frozen predictor.

Sources: SAM2 paper/arXiv and official repo (`facebookresearch/sam2`), Meta SAM2 blog; SAM3 paper/arXiv, official SAM3 page, and `facebookresearch/sam3` repository.

## Experiment matrix

Planned/attempted variants:

| ID | method | status | purpose |
|---|---|---|---|
| A0 | SAM2 baseline | existing | reference |
| A1 | M2-current | existing | memory hygiene diagnostic |
| A2 | M2-light | done | lighter memory-state audit source |
| A3 | M11 cycle | existing | post-gap rejection evidence |
| B1 | M3-state conservative | done, not final | safest identity-preserving selector; over-empty visually |
| B2 | M3-state balanced | done, not final | more permissive but source-switch risk high |
| B3 | M3-state aggressive | skipped | balanced already showed excessive source-switch risk |
| C1 | M3R pseudo-anchor | skipped for now | M3 accepted anchors not trustworthy enough for rerun |
| D1 | tiny crop candidate | smoke-tested in M4, not final | high-resolution crop source helped locally but failed on post-occlusion identity in `r13u5z4y` |

## Results

All M3-state variants were run on b101 and passed the strong submission validator (`pred-root + submit-root + zip`).

| variant | zip validation | changed vs SAM2 | selected sources / behavior | interpretation |
|---|---|---:|---|---|
| `m3_state_conservative` | ok: 433 dirs, 66526 PNGs, testzip None, provided_changed=0 | 502 frames | baseline 231, M2 110, M2-light 165, SAM3 192, empty 837 object-frames | Mechanistically useful but too aggressive: imports many non-baseline masks and over-suppresses visible small objects. |
| `m3_state_balanced` | ok: 433 dirs, 66526 PNGs, testzip None, provided_changed=0 | 641 frames | baseline 225, M2 223, M2-light 246, SAM3 207, empty 634 object-frames | More permissive but visually riskier: larger number of source switches and accepted anchors, likely more identity switches. |
| `m3_state_surgical` | ok: 433 dirs, 66526 PNGs, testzip None, provided_changed=0 | 186 frames | baseline 543, empty 992 object-frames, no alternative mask import | Safest M3 rejector in full diagnostic run but still over-empty; a later post-gap reconfirmation fix made it stricter, so it remains non-final. |

M2-light rerun after code-review fix also passed the strong validator:

```text
pred_root=/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_m2_light
submission=/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/submission_433_m2_light
zip=/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/submission_mosev2_m2_light.zip
validation ok=True, provided_changed=0, predicted_error_count=0, zip testzip=None
runtime=189.76s, CUDA peak allocated/reserved ~= 970.9 MiB / 1190.0 MiB
```

The current M3-state implementation therefore satisfies engineering validity, but the visual evidence does **not** support submitting it as the best final mask set.

## Visual analysis

Committed comparison sheets:

- Conservative full/zoom sheets: `docs/assets/m3_state/conservative/{full,zooms}/`
- Balanced full/zoom sheets: `docs/assets/m3_state/balanced/{full,zooms}/`

Vision review on conservative M3 over all 15 videos:

| video | visual verdict vs SAM2 | reason |
|---|---|---|
| `1qlssuz2` | roughly tie / slight improvement | suppresses some risky tiny-car frames but also misses visible small car. |
| `2smf7uq9` | worse | flamingo/bird thin structures and small target become over-empty. |
| `3epdtmyr` | uncertain-worse | near frames okay, distant lions often emptied while SAM3/SAM2 had plausible masks. |
| `4f98052b` | improved on false positives | rejects repeated red/black chair/logo distractors, but can miss people. |
| `4vznweiu` | roughly tie / slight improvement | small dice/letter block mostly stable; small boundary differences. |
| `8jsm23a7` | worse | white tile/card target becomes empty after occlusion despite SAM2 continuity. |
| `amfdu83t` | worse | tiny/edge kangaroo mostly empty. |
| `c8lutf29` | slight improvement but unstable | rejects some wrong far-field masks; semantic drift remains. |
| `jadgtmfl` | tie / slight improvement | fish masks close to SAM2, with some risky empty frames. |
| `lcgc29va` | slight improvement | tiny pedestrian/backpack cases benefit from conservative rejection of crowd clutter. |
| `msinig6m` | uncertain-worse | dense koala/person multi-id scene still switches and over-empties. |
| `pe0d85lk` | slight improvement | rejects some reflection/remote errors; still misses later persons. |
| `q0sizv6m` | slight improvement but unstable | reduces multi-guinea-pig spread but often outputs empty. |
| `r13u5z4y` | clearly improved as rejector | avoids SAM2's post-occlusion wrong strawberry slice, at cost of missing true reappearance. |
| `z6dx46qr` | uncertain-worse | low-contrast underwater target alternates between wrong blobs and empty. |

Conclusion: M3-state is a useful **diagnostic/rejector** for same-class post-occlusion failures (`r13u5z4y`, `4f98052b`), but it is not a robust final method because it turns many ambiguous visible small targets into empty masks. This is especially harmful for thin/small/edge targets (`2smf7uq9`, `8jsm23a7`, `amfdu83t`, parts of `3epdtmyr`).

## Code review

Initial independent code review verdict: **REQUEST CHANGES**. Findings and fixes:

| severity | finding | fix status |
|---|---|---|
| HIGH | M3 remote runner did not invoke the strong validator. | Fixed: full-run M3 now calls `tools/validate_mose_submission.py` for pred-root + submit-root + zip. |
| HIGH | M3 submission builder copied all `out_pred_root` dirs without a 15-video whitelist. | Fixed: `make_submission()` now asserts exact predicted-video set, disjointness from 418 provided videos, expected pred PNG count, 433 dirs, 66526 PNGs, and zip `testzip()`. |
| HIGH | Remote destructive path guard used raw string prefix checks. | Fixed: M2/M3 runners canonicalize with `realpath -m`, reject roots, and then enforce prefix/under-directory constraints. |
| MEDIUM | M2-light tri-state `likely_absent` used fixed objectness cutoff 0.5 despite `obj_thr=0.25`. | Fixed: `memory_audit_state()` now respects the configured check result (`obj_ok`). M2-light audit was rerun after this change. |
| MEDIUM | M3 audit summary did not count frame-0 GT anchors. | Fixed: frame 0 now contributes `first_frame_gt` / `gt_anchor` / `CONFIRMED_VISIBLE` stats. |
| MEDIUM | M2/M2-light full runner had weaker final submission checks. | Fixed: M2/M2-light runner now invokes the same strong validator on full submissions. |
| LOW/WATCH | Strict JSON/safe rounding and histogram weakness. | Fixed non-standard NaN/Inf JSON output and set M3 audit dumps to `allow_nan=False`; histogram limitation remains documented and inspected visually. |

### Second code-review pass

Second review confirmed the prior HIGH/MEDIUM engineering fixes were in place. It flagged one remaining MEDIUM issue in `surgical`: after a post-gap frame it could reconfirm baseline too quickly. This was fixed by keeping post-gap baseline in `REAPPEARING_CANDIDATE` until `confirm_delay` consecutive non-suspicious frames, and by avoiding identity update before that confirmation. A smoke rerun on `r13u5z4y` passed and showed the variant became stricter, reinforcing the decision not to use M3-state as final.


## M4 tiny-crop continuation

After the initial M3 conclusion, the next high-quality step was executed on branch `method/m4-tinycrop-candidate`: a training-free SAM2 tiny-crop candidate source plus required-source validation in M3.

Smoke evidence (`1qlssuz2`, `r13u5z4y`, `lcgc29va`):

- tiny-crop source runtime: 31.815s, CUDA peak allocated/reserved ~= 934.96 / 1146.0 MiB;
- M3+tiny safe balanced smoke runtime: 58.929s;
- exact tiny-crop frame coverage verified for all 3 selected videos;
- source usage: tiny_crop 46 object-frames, baseline 30, empty 25, M2-light 10, M2 3, SAM3 3.

Visual result: mixed and **not final**. `lcgc29va` shows some tighter backpack/person masks, `1qlssuz2` is ambiguous, and `r13u5z4y` remains the key failure: after hand occlusion, crop centering follows correlated wrong evidence/velocity and segments a wrong board/strawberry-adjacent region. Thus crop rerun improves local resolution but does not solve identity.

Artifacts:

- report: `docs/m4_tiny_crop_experiment.md`;
- sheets: `docs/assets/m4_tiny_crop/safe_smoke/{full,zooms}/`;
- audits: `docs/assets/m4_tiny_crop/safe_smoke/audits/`.

Conclusion: M4 is a useful diagnostic candidate source but is unsafe as final output without a stronger identity verifier.

## Final selection

**Do not submit M3-state as final at this stage.**

Reasoning:

1. M3-state achieved engineering validity and auditability, but image review shows excessive empty-mask behavior.
2. Balanced/conservative variants import many alternative masks (M2/M2-light/SAM3), increasing identity-switch risk.
3. Surgical variant preserves baseline more in the full diagnostic run, but still suppressed 186 baseline frames; after code-review tightening of post-gap reconfirmation it became even more conservative on smoke (`r13u5z4y`: 23 changed frames / 39 empty object-frames), so it remains too empty to justify replacing the known stronger result.
4. The best currently supported final remains the previously validated **M11 cycle-gate** submission: it is much closer to baseline, is known from prior discussion to have a small positive score delta (~+0.02), and directly targets post-gap wrong tracklets without the broad over-empty behavior of M3-state.

Chosen final local zip:

```text
/home/yu/projects/cv/cvMOSE/submission_mosev2_final_m11_cycle.zip
size=71,380,305 bytes
validation ok=True; 433 dirs / 66,526 PNGs / testzip=None / provided_changed=0 / predicted_error_count=0
```

M3 should continue as a research branch, not as the final submission. The next scientifically justified follow-up is not more threshold tuning, but a stronger identity feature/crop candidate source for tiny targets and a stricter rule that M3 may reject high-risk post-gap baseline masks without replacing them by weak SAM3/M2 alternatives.

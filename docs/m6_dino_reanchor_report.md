# M6 Phase 5 — DINOv2 descriptor upgrade for M5R-C

## Goal
Upgrade M5R-C from RGB/SAM2-pooled descriptors to frozen DINOv2 object descriptors so candidate acceptance depends on object-level identity margin against hard negatives, not mask quality alone.

## Implementation

Code:
- `src/cvmose/dino_descriptors.py`
- `tools/precompute_dino_features.py`
- updated `tools/infer_mosev2_sam2_reanchor.py`
- updated `scripts/run_b101_m5r_reanchor.sh`

Key changes:
1. `--descriptor dino_vitb_reg | dino_vitl_reg | dino_sam2_fusion`.
2. Frozen DINOv2 masked descriptor:
   - full-frame patch tokens;
   - masked object mean;
   - local context-ring contrast `object_vec - ring_vec`;
   - bbox/area/aspect shape scalars;
   - tiny fallback to bbox-expanded crop when full-frame patch tokens `< 3`.
   - DINO modes fail closed if no DINO descriptor can be extracted; RGB/HOG remains only a weak auxiliary cue, not a standalone fallback.
3. Positive bank remains first-frame GT + early stable/pre-disappearance masks.
4. Negative bank remains first-frame context ring + M11-rejected masks + other IDs + nearby any-FG components.
5. Candidate score remains margin-based: `max positive cosine - max hard-negative cosine`.
6. Added delayed promotion:
   - `--anchor-confirm-mode delayed2_or_cycle`;
   - single candidate becomes provisional unless confirmed in nearby future frames or margin is very strong.
7. Added rollback guard during `anchor_window` merge when repropagation changes too much without high confidence.

## External source and checkpoint

- DINOv2 repo: `https://github.com/facebookresearch/dinov2`
- Local clone: `external/dinov2`
- Remote clone: `/data1/yuzhixiang/cv_mosev2/external/dinov2`
- Commit tested: `7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c`
- Weight: `dinov2_vitb14_reg4_pretrain.pth`
- Local path: `/home/yu/projects/cv/from fdu/MOSEv2/homework/external_checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth`
- Remote path: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/external_checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth`
- SHA256: `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71`
- Rule note: public frozen pretrained checkpoint, no training/fine-tuning. Likely more acceptable than MOSEv2-finetuned official checkpoints, but final permission still depends on assignment rules for external public checkpoints.

## Smoke command

```bash
SYNC_DINO=0 GPU=4 \
VIDEOS='r13u5z4y q0sizv6m msinig6m 1qlssuz2 2smf7uq9 8jsm23a7 lcgc29va 4vznweiu' MAKE_SUBMISSION=0 \
PRED_ROOT='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_m5r_dino_key' \
AUDIT_JSON='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m6_dino/m5r_dino_key.json' \
AUDIT_DIR='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m6_dino/m5r_dino_key_by_video' \
M5R_EXTRA_ARGS='--descriptor dino_vitb_reg --dino-root /data1/yuzhixiang/cv_mosev2/external/dinov2 --dino-weights /data1/yuzhixiang/cv_mosev2/MOSEv2/homework/external_checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth --merge-policy anchor_window --max-anchors-per-object 1 --anchor-confirm-mode delayed2_or_cycle --identity-margin 0.12 --sameclass-margin 0.18 --sam2-auto-mask-candidates --auto-mask-max-frames-per-object 3 --auto-mask-points-per-side 16 --auto-mask-max-count 20 --rollback-max-change-frac 0.50' \
scripts/run_b101_m5r_reanchor.sh
```

## Artifacts

- pred root: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_m5r_dino_key`
- local mirror: `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_m5r_dino_key`
- audit: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m6_dino/m5r_dino_key.json`
- per-video audits: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m6_dino/m5r_dino_key_by_video/`
- visual sheets: `docs/assets/m6_dino/smoke/`
- compact committed audit summary: `docs/assets/m6_dino/smoke/audit_summary.json`
- compare JSON: `artifacts/m6_phase5/compare_m5r_dino_key.json`
- compare CSV: `artifacts/m6_phase5/compare_m5r_dino_key.csv`

## Smoke summary

Runtime: 8 videos / 423 frames / 5 accepted anchors / 84 changed frames vs SAM2 baseline.

| video | anchors | changed vs SAM2 | empty counts | qualitative verdict |
| --- | ---: | ---: | --- | --- |
| `r13u5z4y` | 0 | 0 | `{1:16}` | DINO safely accepts no anchor, but output falls back to SAM2 baseline, not M11; do **not** use in final. |
| `q0sizv6m` | 0 | 0 | `{1:28,2:10}` | Avoids worse same-class anchors; no recovery. Keep M11/default. |
| `msinig6m` | 1 | 0 after rollback | `{1:85,2:95,3:94}` | Accepted object-3 anchor was mostly rolled back; avoids DAM/SAM2Long composite disaster, but no gain. |
| `1qlssuz2` | 1 | 17 | `{1:5}` | Plausible small-object continuity changes; visually similar/slightly cleaner. Candidate for balanced but not high-confidence. |
| `2smf7uq9` | 1 | 24 | `{1:11}` | Best DINO case: it reintroduces flamingo masks in frames where baseline/M11 go empty. Same-class ambiguity remains but visual evidence is plausible. |
| `8jsm23a7` | 1 | 17 | `{1:1}` | Stable guard video; changes are unnecessary and slightly shrink/perturb mask. Do not replace. |
| `lcgc29va` | 0 | 0 | `{1:8}` | No tiny-object improvement. |
| `4vznweiu` | 1 | 26 | `{1:15}` | Regression: anchor/reprop empties many later frames. Reject. |

## Interpretation

DINOv2 materially improves **rejection safety**: the known harmful cases (`r13u5z4y`, `q0sizv6m`, `msinig6m`) no longer accept the obviously wrong anchors that earlier M5R-C/RAR variants risked. However, it still rarely supplies a verified new anchor because the candidate generator remains too weak/high-noise. The only visually plausible recovery is `2smf7uq9`; `1qlssuz2` is at most a mild balanced candidate.

This confirms the structural diagnosis: strong descriptors help, but score gains require high-recall candidate generation that actually contains the true reappearing object.

## Decision

- Safe final: do not directly use the DINO root.
- Balanced: allow `2smf7uq9` and possibly `1qlssuz2` from DINO after visual review.
- Aggressive: DINO is useful only for those selected videos; do not use on `r13u5z4y`, `q0sizv6m`, `msinig6m`, `8jsm23a7`, `lcgc29va`, or `4vznweiu`.

## Not run

`dino_vitl_reg` was not run: the ViT-B/reg smoke already showed the bottleneck is candidate recall/promotion, not descriptor capacity alone, and ViT-L would add large download/VRAM cost without addressing missing true proposals.

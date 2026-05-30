# M7 — DINO part matching + short-tracklet verification

## Goal

This experiment implements the third-priority improvement from the 44-point analysis: stop treating DINO identity as only a masked mean vector, and add local patch/part matching plus short tracklet aggregation before accepting a re-anchor.

The target is not another threshold sweep. The question is whether stronger object-level verification can reject same-class/composite anchors while preserving the useful M6 DINO recovery on `2smf7uq9`.

## Implementation

Code changes:

- `src/cvmose/dino_descriptors.py`
  - adds foreground DINO patch-token subsets (`part_tokens`) using deterministic diverse/farthest-point selection;
  - keeps frozen DINOv2 only, no training/fine-tuning;
  - tiny-object crop fallback also returns part tokens.
- `tools/infer_mosev2_sam2_reanchor.py`
  - adds `--dino-part-matching`, `--dino-part-topk`, `--dino-part-weight`, `--dino-part-min-tokens`;
  - adds symmetric local-part similarity blended with descriptor cosine;
  - adds `--tracklet-aggregate future|local`, `--tracklet-window`, `--tracklet-min-count`, `--tracklet-weight`;
  - aggregates margin over nearby source/motion-consistent candidates before thresholding and delayed anchor commit;
  - audits `raw_margin`, part positive/negative scores, part count, tracklet count, and tracklet reason.
- `scripts/run_b101_m5r_reanchor.sh`
  - allowlists M7 DINO-part and tracklet flags for remote smoke runs.

## Smoke command

```bash
SYNC_DINO=0 GPU=4 \
VIDEOS='r13u5z4y q0sizv6m msinig6m 1qlssuz2 2smf7uq9 8jsm23a7 lcgc29va 4vznweiu' MAKE_SUBMISSION=0 \
PRED_ROOT='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_m7_part_tracklet_key' \
AUDIT_JSON='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m7_part_tracklet/m7_part_tracklet_key.json' \
AUDIT_DIR='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m7_part_tracklet/m7_part_tracklet_key_by_video' \
M5R_EXTRA_ARGS='--descriptor dino_vitb_reg --dino-root /data1/yuzhixiang/cv_mosev2/external/dinov2 --dino-weights /data1/yuzhixiang/cv_mosev2/MOSEv2/homework/external_checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth --dino-part-matching --dino-part-topk 8 --dino-part-weight 0.45 --dino-part-min-tokens 2 --tracklet-aggregate future --tracklet-window 3 --tracklet-min-count 2 --tracklet-weight 0.45 --merge-policy anchor_window --max-anchors-per-object 1 --anchor-confirm-mode delayed2_or_cycle --identity-margin 0.12 --sameclass-margin 0.18 --sam2-auto-mask-candidates --auto-mask-max-frames-per-object 3 --auto-mask-points-per-side 16 --auto-mask-max-count 20 --rollback-max-change-frac 0.50' \
scripts/run_b101_m5r_reanchor.sh
```

## Artifacts

- pred root: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_m7_part_tracklet_key`
- local mirror: `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_m7_part_tracklet_key`
- audit JSON: `/home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m7_part_tracklet/m7_part_tracklet_key.json`
- per-video audits: `/home/yu/projects/cv/from fdu/MOSEv2/homework/logs/m7_part_tracklet/m7_part_tracklet_key_by_video/`
- committed compact audit: `docs/assets/m7_part_tracklet/smoke/audit_summary.json`
- visual sheets: `docs/assets/m7_part_tracklet/smoke/`
- comparison CSV/JSON: `artifacts/m7_part_tracklet/compare_m7_key.{csv,json}`

## Smoke result

Runtime summary: 8 videos / 423 frames / 3 accepted anchors / 52 changed frames vs SAM2 baseline.

| video | M6 DINO anchors | M7 anchors | M7 changed vs baseline | verdict |
| --- | ---: | ---: | ---: | --- |
| `r13u5z4y` | 0 | 0 | 0 | Still no true strawberry candidate after occlusion. Part matching keeps safe rejection; does not solve recall. |
| `q0sizv6m` | 0 | 0 | 0 | Same-class animal anchors stay rejected. Good safety, no recovery. |
| `msinig6m` | 1 then rollback | 0 | 0 | Improvement over M6 mechanics: part/tracklet suppresses the weak object-3 anchor instead of relying on rollback. |
| `1qlssuz2` | 1 | 0 | 0 | M7 rejects M6's mild car-mask tweak. This is safer but may lose the small balanced gain. |
| `2smf7uq9` | 1 | 1 | 24 | Keeps the only strong M6 recovery; anchor now has part/tracklet evidence (`margin≈0.334`, `tracklet_count=3`). |
| `8jsm23a7` | 1 | 1 | 17 | Still accepts an unnecessary early stable-video anchor. Do not use for fusion. |
| `lcgc29va` | 0 | 0 | 0 | Tiny target still has no reliable candidate; descriptor strength is not enough. |
| `4vznweiu` | 1 bad/regressive | 1 with rollback | 11 | M7 is much safer than M6: fewer changed frames and 15 rollback skips; still not a clear improvement over M11. |

## Candidate zips

Both keep all 418 provided outputs unchanged and passed `tools/validate_mose_submission.py`.

| zip | path | size | policy | recommendation |
| --- | --- | ---: | --- | --- |
| M7 conservative | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m7_part_conservative.zip` | 71,349,795 | M11 default + M7 part-tracklet only on `2smf7uq9` | safe probe if you want to isolate whether `2smf7uq9` was the real M6 gain |
| M7 balanced | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m7_part_balanced.zip` | 71,349,832 | M11 default + M6 on `1qlssuz2` + M7 part-tracklet on `2smf7uq9` | best M7 probe; should be compared against previous M6 balanced score 43.29 |

Validation:

- `artifacts/m7_part_tracklet/m7_part_conservative_validation.json`: OK, 433 dirs / 66526 PNGs / 418 unchanged.
- `artifacts/m7_part_tracklet/m7_part_balanced_validation.json`: OK, 433 dirs / 66526 PNGs / 418 unchanged.

## Interpretation

M7 improves the **verification layer** but not the **candidate recall layer**.

What improved:

- M7 rejects weak/unstable anchors that M6 accepted and then had to rollback (`msinig6m`).
- M7 keeps the useful `2smf7uq9` anchor with stronger part/tracklet evidence.
- M7 makes the `4vznweiu` failure less damaging, shrinking changed frames from 26 to 11 and rolling back most risky reprop frames.

What did not improve:

- `r13u5z4y` remains impossible for this pipeline because the correct post-occlusion strawberry slice is not in the candidate pool.
- `lcgc29va` remains a proposal-recall/tiny-resolution failure.
- `q0sizv6m` remains safe but unrecovered.

This supports the bottleneck diagnosis: DINO part matching helps precision, but 44 requires high-recall later-frame proposal mining and reverse/cycle tracklet validation.

## Decision

- Do not replace M6 balanced blindly.
- Submit/score `submission_mosev2_m7_part_balanced.zip` as a focused probe if budget allows.
- If M7 balanced does not exceed 43.29, keep M6 balanced as the best current probe and move to M7-B: high-recall global candidate bank + reverse propagation.

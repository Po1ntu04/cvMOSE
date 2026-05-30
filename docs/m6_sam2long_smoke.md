# M6 Phase 3 — SAM2Long memory-tree smoke

## Purpose
Test whether SAM2Long's training-free memory tree provides a safer branch/hypothesis mechanism than our hand-written M5R-C/RAR path on the MOSEv2 homework failure modes: occlusion/reappearance, same-class distractors, tiny objects, and baseline-stable videos.

## External source
- Repository: `https://github.com/Mark12Ding/SAM2Long`
- Local clone: `external/SAM2Long`
- Remote clone: `/data1/yuzhixiang/cv_mosev2/external/SAM2Long`
- Commit tested: `7193b77fa0c8827e0520ab281acd2cf394ab898e`
- License/rule note: public code + public SAM2.1-B+ checkpoint. This is training-free and potentially final-allowable if the assignment permits public inference code changes; no MOSEv2-finetuned checkpoint was used in this smoke.

## Adapter
- Code: `tools/infer_mosev2_sam2long_adapter.py`
- Remote runner: `scripts/run_b101_sam2long_adapter.sh`
- Design:
  - invokes external `tools/vos_inference.py` rather than vendoring SAM2Long;
  - uses first-frame MOSEv2 GT masks as prompts;
  - overwrites frame `00000.png` with exact local GT after inference;
  - writes normal single-channel label PNGs;
  - optional 433-video submission build keeps the 418 provided outputs unchanged.

## Smoke commands

```bash
NUM_PATHWAY=2 IOU_THRE=0.3 UNCERTAINTY=1 RUN_NAME=np2_iou03_u1_smoke GPU=4 MAKE_SUBMISSION=0 \
VIDEOS='r13u5z4y q0sizv6m msinig6m 1qlssuz2 8jsm23a7 lcgc29va' \
scripts/run_b101_sam2long_adapter.sh

NUM_PATHWAY=3 IOU_THRE=0.3 UNCERTAINTY=1 RUN_NAME=np3_iou03_u1_smoke GPU=4 MAKE_SUBMISSION=0 SYNC_EXTERNAL=0 \
VIDEOS='r13u5z4y q0sizv6m msinig6m 1qlssuz2 8jsm23a7 lcgc29va' \
scripts/run_b101_sam2long_adapter.sh

NUM_PATHWAY=2 IOU_THRE=0.1 UNCERTAINTY=2 RUN_NAME=np2_iou01_u2_smoke GPU=4 MAKE_SUBMISSION=0 SYNC_EXTERNAL=0 \
VIDEOS='r13u5z4y q0sizv6m msinig6m 1qlssuz2 8jsm23a7 lcgc29va' \
scripts/run_b101_sam2long_adapter.sh
```

## Artifacts

| run | pred_root | audit_json | visual sheets |
| --- | --- | --- | --- |
| np2 iou0.3 u1 | `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_m6_sam2long_np2_iou03_u1_smoke` | `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m6_sam2long/np2_iou03_u1_smoke.json` | `docs/assets/m6_sam2long/smoke/` |
| np3 iou0.3 u1 | `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_m6_sam2long_np3_iou03_u1_smoke` | `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m6_sam2long/np3_iou03_u1_smoke.json` | `docs/assets/m6_sam2long/smoke/` |
| np2 iou0.1 u2 | `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_m6_sam2long_np2_iou01_u2_smoke` | `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m6_sam2long/np2_iou01_u2_smoke.json` | `docs/assets/m6_sam2long/smoke/` |

Local comparison summary:
- JSON: `artifacts/m6_phase3/compare_sam2long_smoke.json`
- CSV: `artifacts/m6_phase3/compare_sam2long_smoke.csv`
- All smoke outputs preserved frame 0 and had valid label IDs.

## Quantitative audit vs SAM2/M11 (hidden GT not used)

The three parameterizations were almost identical, so the table below emphasizes the representative `np2_iou03_u1` run.

| video | changed vs SAM2 | IoU vs SAM2 | changed vs M11 | IoU vs M11 | empty counts | verdict |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `r13u5z4y` | 28 | 0.4886 | 5 | 0.9997 | `{1:39}` | essentially M11-safe empty after occlusion; no strawberry reappearance recovery |
| `q0sizv6m` | 41 | 0.2047 | 41 | 0.2047 | `{1:13,2:2}` | severe same-class animal drift/composite; reject |
| `msinig6m` | 87 | 0.6529 | 67 | 0.8088 | `{1:104,2:87,3:98}` | suppresses some regions but still creates wrong person/koala fragments; reject |
| `1qlssuz2` | 33 | 0.9824 | 33 | 0.9824 | `{1:5}` | close to baseline; no visible recovery reason |
| `8jsm23a7` | 47 | 0.9915 | 47 | 0.9915 | `{1:1}` | mostly stable but no improvement; stable guard says do not replace |
| `lcgc29va` | 27 | 0.7390 | 27 | 0.7390 | `{1:12}` | no reliable tiny-target gain; loose run raises IoU to 0.802 but visual still not decisive |

## Qualitative findings

- `r13u5z4y`: SAM2Long behaves like M11/official safe rejection after the hand occlusion. It avoids tracking the wrong strawberry, but it also cannot reacquire the heavily occluded rear slice.
- `q0sizv6m`: the memory-tree branches do not solve identity. They amplify alternate guinea-pig regions and generate large same-class blobs; this violates the safe/balanced fusion rule.
- `msinig6m`: still confuses koala/person clusters. Branch diversity does not become object-level re-identification.
- `lcgc29va`: no robust tiny recall improvement. The method sometimes keeps small blobs, but not enough to justify replacing SAM2/M11.
- `8jsm23a7`: no meaningful improvement over baseline and unnecessary per-frame changes.

## Decision

Do **not** promote SAM2Long to full-15/fusion at this stage. It is useful evidence that memory-tree branching alone is not sufficient for MOSEv2's same-class identity cases. The missing component remains a stronger object-level descriptor/retrieval gate, so Phase 5 DINOv2 re-anchor remains higher priority.

## Recommendation for fusion

- Safe: exclude SAM2Long.
- Balanced: exclude SAM2Long, except possibly revisit `r13u5z4y` only if a later policy prefers safe empty over another non-empty wrong strawberry.
- Aggressive probe: not recommended; q0sizv6m/msinig6m regressions are exactly the benchmark's key distractor failure mode.

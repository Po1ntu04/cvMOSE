# M6 Phase 2 — DAM4SAM / d4sm Smoke

Date: 2026-05-30  
Branch: `method/m6_external_reid_ensemble`

## External repos

| repo | local path | remote path | commit | role |
|---|---|---|---|---|
| DAM4SAM | `external/DAM4SAM` | `/data1/yuzhixiang/cv_mosev2/external/DAM4SAM` | `9c954504b39ebca4c412f207be0787c26bfac85a` | single-object DAM4SAM per object |
| d4sm | `external/d4sm` | `/data1/yuzhixiang/cv_mosev2/external/d4sm` | `67d5dfe86d8ae7b557193231c6058a46f5c0c490` | multi-object DAM4SAM |

Sources:

- <https://github.com/jovanavidenovic/DAM4SAM>
- <https://github.com/alanlukezic/d4sm>
- DAM4SAM paper: <https://arxiv.org/abs/2411.17576>

## Adapter notes

Implemented:

- `tools/infer_mosev2_dam4sam_adapter.py`
- `scripts/run_b101_dam4sam_adapter.sh`

The adapter keeps frame 0 as local GT, supports:

- `--mode d4sm_multi`: one d4sm tracker for all first-frame objects;
- `--mode dam4sam_single_per_obj`: independent DAM4SAM tracker per object, then label merge.

b101 has no internet and the existing `mose_sam2` env lacks `vot-toolkit`, so the adapter installs an in-process minimal `vot` stub for the exact rectangle/mask IoU APIs used by DAM4SAM/d4sm DRM. This avoids polluting the conda env.

Checkpoint caveat:

- `dam4sam_single_per_obj` used the existing public SAM2.1-B+ checkpoint from `MOSEv2/5_19/data/sam2/sam2.1_hiera_base_plus.pt`.
- `d4sm_multi` expects `sam2.1_hiera_large.pt`. Because b101 cannot download and the standard large checkpoint was not available locally in time, the smoke used the already-synced Fudan MOSEv2 large checkpoint as a **diagnostic fallback**. This makes d4sm smoke **not final-eligible under strict no-finetuned-external rules**.

## Commands

```bash
MODE=d4sm_multi \
VIDEOS='r13u5z4y q0sizv6m msinig6m 8jsm23a7 lcgc29va 1qlssuz2' \
MAKE_SUBMISSION=0 GPU=4 scripts/run_b101_dam4sam_adapter.sh

MODE=dam4sam_single_per_obj \
VIDEOS='r13u5z4y q0sizv6m msinig6m 8jsm23a7 lcgc29va 1qlssuz2' \
MAKE_SUBMISSION=0 GPU=4 SYNC_EXTERNAL=0 scripts/run_b101_dam4sam_adapter.sh
```

Outputs:

| method | pred_root | audit_json | validation |
|---|---|---|---|
| d4sm smoke | `/data1/.../homework/pred_m6_dam4sam_d4sm_multi_smoke` | `/data1/.../homework/logs/m6_dam4sam/d4sm_multi_smoke.json` | pred-root compare ok, first frames preserved |
| DAM4SAM single smoke | `/data1/.../homework/pred_m6_dam4sam_dam4sam_single_per_obj_smoke` | `/data1/.../homework/logs/m6_dam4sam/dam4sam_single_per_obj_smoke.json` | pred-root compare ok, first frames preserved |

Visual sheets:

```text
docs/assets/m6_dam4sam/smoke/{video}_m6_compare.jpg
```

Audit summaries:

| method | videos | pngs | total conflicts | notes |
|---|---:|---:|---:|---|
| d4sm_multi | 6 | 345 | 2,518,529 | huge multi-object overlap on `msinig6m` |
| dam4sam_single_per_obj | 6 | 345 | 103,689 | lower conflicts but still severe same-class drift |

## Quantitative agreement on smoke set

| video | d4sm IoU vs SAM2 / empty | DAM-single IoU vs SAM2 / empty | visual verdict |
|---|---|---|---|
| `r13u5z4y` | 0.486 / `{1:39}` | 0.488 / `{1:39}` | same as M11/official: safe empty after occlusion, no true strawberry recovery. |
| `q0sizv6m` | 0.619 / `{1:5,2:6}` | 0.205 / `{1:5,2:2}` | rejects neither same-class risk; DAM-single creates large wrong blobs on neighboring animals. |
| `msinig6m` | 0.455 / `{1:83,2:64,3:73}` | 0.515 / `{1:85,2:68,3:82}` | both produce person/koala composite or wrong-object masks; not safe. |
| `8jsm23a7` | 0.916 / `{1:2}` | 0.862 / `{1:2}` | mostly stable but no improvement over SAM2/M11; stable-guard means do not replace. |
| `lcgc29va` | 0.330 / `{1:28}` | 0.747 / `{1:13}` | DAM-single is the only potentially useful signal; it keeps more tiny-person frames but still loses/reappears inconsistently. |
| `1qlssuz2` | 0.413 / `{1:2}` | 0.951 / `{1:5}` | DAM-single mostly close to baseline; no visible reason to replace. |

## Decision

Do **not** run full-15 DAM4SAM/d4sm in this round.

Reason: the smoke criterion was not met. There is not at least two clear improvements without severe regression. The most relevant hard cases (`q0sizv6m`, `msinig6m`) regress through same-class/composite masks, and `r13u5z4y` remains an empty-safe failure rather than a recovery. `lcgc29va` from DAM-single is a useful tiny-target diagnostic/candidate source for future fusion, but not enough to justify global 15-video submission.

Recommended use in M6 fusion:

- keep `dam_single` only as a possible **single-video tiny candidate** for `lcgc29va`, requiring visual proof;
- exclude d4sm and DAM-single from safe/balanced on `q0sizv6m` and `msinig6m`;
- never replace stable `8jsm23a7` with DAM outputs.

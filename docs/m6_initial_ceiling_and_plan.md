# M6 Initial Ceiling, Invariants, and Execution Plan

Date: 2026-05-30
Branch: `method/m6_external_reid_ensemble`
Base commit: `1e3489335843eaf3d8a2d5c1f4e696d180d9b90a`

## Current fixed facts

| item | status |
|---|---|
| Current retained final zip | `submission_mosev2_final_m11_cycle.zip` |
| SAM2 baseline score | about `43.24` |
| Provided-output-only/no-new-15 score | about `41.25` |
| Current safest final | M11 cycle gate, small positive gain over baseline by prior Codabench feedback |
| M2 / M2-light / RAR A/B / M3 / M4 / M5R-C | diagnostic only; not final |
| M5R-C status | structurally correct re-anchor scaffold, but descriptor/proposal not strong enough |

## Hard submission invariant

The 418 provided outputs are immutable. Every candidate zip must have:

- 433 video directories;
- 66,526 PNGs;
- byte-identical provided outputs unless explicitly using array-equal fallback for diagnostics only;
- 15 predicted video outputs with valid frame names, image sizes, and label ids drawn from first-frame GT ids;
- successful `tools/validate_mose_submission.py` JSON.

Validation already rerun in Phase 0:

```text
artifacts/m6_phase0/final_m11_validation.json      ok=true
artifacts/m6_phase0/baseline_zip_validation.json   ok=true
```

Both current local zips preserve the 418 provided outputs and validate successfully.

## 15 target videos

| video | frames | object ids | first-frame areas |
|---|---:|---|---|
| `1qlssuz2` | 40 | `[1]` | `{1: 1304}` |
| `2smf7uq9` | 43 | `[1]` | `{1: 16083}` |
| `3epdtmyr` | 190 | `[1]` | `{1: 70947}` |
| `4f98052b` | 179 | `[1, 2]` | `{1: 92965, 2: 57596}` |
| `4vznweiu` | 35 | `[1]` | `{1: 2465}` |
| `8jsm23a7` | 49 | `[1]` | `{1: 2761}` |
| `amfdu83t` | 24 | `[1]` | `{1: 1077}` |
| `c8lutf29` | 47 | `[1]` | `{1: 40971}` |
| `jadgtmfl` | 41 | `[1]` | `{1: 19821}` |
| `lcgc29va` | 35 | `[1]` | `{1: 947}` |
| `msinig6m` | 134 | `[1, 2, 3]` | `{1: 56869, 2: 54739, 3: 37036}` |
| `pe0d85lk` | 62 | `[1, 2]` | `{1: 485350, 2: 589869}` |
| `q0sizv6m` | 42 | `[1, 2]` | `{1: 14581, 2: 7787}` |
| `r13u5z4y` | 45 | `[1]` | `{1: 33250}` |
| `z6dx46qr` | 38 | `[1]` | `{1: 7670}` |

Raw metadata is stored in `artifacts/m6_phase0/video_meta_15.json`.

## Phase-0 tools added

### `tools/compare_candidate_roots.py`

Compares prediction roots without hidden labels. It reports:

- changed frames vs SAM2 baseline;
- changed frames vs M11;
- per-object empty counts and area stats;
- mean binary IoU agreement vs baseline/M11;
- pairwise source agreement IoU;
- first-frame preservation;
- label id validity;
- missing/size errors.

Initial audit:

```text
artifacts/m6_phase0/compare_baseline_m11_m5rc.json
artifacts/m6_phase0/compare_baseline_m11_m5rc.csv
```

Summary: baseline changed 0 frames vs baseline; M11 changed 60 frames vs baseline; M5R-C key smoke changed 163 frames on the 8 videos it covers and is missing 581 frames because it was not a full-15 candidate.

### `scripts/make_m6_compare.py`

Creates all-video visual sheets with columns:

```text
RGB | SAM2 baseline | M11 | candidate A/B/C | delta last candidate vs SAM2 | zoom crop
```

Initial sheets:

```text
docs/assets/m6_phase0/baseline_m11_m5rc/{video}_m6_compare.jpg
docs/assets/m6_phase0/baseline_m11_m5rc/sheet_index.json
```

Deep zoom is enabled for the critical videos listed in the user plan.

## Current failure model to carry forward

- `r13u5z4y`: current M5R-C safely rejects anchors but cannot recover the strawberry slice; descriptor/proposal recall is not enough.
- `q0sizv6m`: current M5R-C accepts plausible but unreliable animal regions; same-class identity remains the key risk.
- `msinig6m`: SAM2 automatic masks improve recall but create person/koala composites.
- `lcgc29va`: no reliable tiny anchor found.
- Stable videos such as `8jsm23a7` must not be harmed by high-variance external methods.

## M6 execution plan

1. **Official MOSEv2 baseline ceiling:** download/list FudanCVL checkpoints and submission zips; run 15-video checkpoint inference and optionally 15-video extracted official zip substitution. Mark final eligibility as rule-dependent.
2. **DAM4SAM/d4sm:** use `initialize(image, init_regions)` + `track(image)` if installable; prioritize mask init and multi-object d4sm; judge identity rather than non-empty masks.
3. **SAM2Long:** run training-free memory-tree/pathway inference using first-frame GT masks; only small parameter sweep.
4. **SAAS:** attempt available base+ weights; treat as cross-view/camera-motion candidate, not a guaranteed same-class re-id solution.
5. **DINOv2 M5R-C upgrade:** replace RGB/SAM2-pooled descriptor with DINOv2 masked/crop descriptors, hard negatives, and multi-frame delayed promotion.
6. **Safe fusion:** default M11; per-video/object replacement only with visual evidence and validator success. Produce safe, balanced, and aggressive zips.

## Stop / reject rules

- Any candidate that fails the 418 provided-output invariant is stopped.
- Any method producing large wrong same-class blobs on `q0sizv6m`/`msinig6m` is excluded from safe/balanced fusion.
- Any method damaging stable baseline videos such as `8jsm23a7` is excluded from safe/balanced fusion.
- External public checkpoints are allowed as diagnostics; final-submission eligibility must be explicitly marked and depends on homework rules.

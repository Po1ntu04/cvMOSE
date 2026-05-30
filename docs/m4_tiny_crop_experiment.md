# M4 Tiny-Crop Candidate Experiment

Date: 2026-05-30
Branch: `method/m4-tinycrop-candidate`
Status: **smoke-tested, not promoted to final**

## Question

Can a training-free SAM2 crop rerun rescue tiny / edge / post-occlusion targets that full-frame SAM2 and M3-state either lose or confuse?

The core hypothesis was narrower than “crop improves masks”: if the first-frame target is very small, a larger relative crop should give SAM2 more effective resolution around the object. The crop result must remain a **candidate source**; it is not an independent identity oracle.

## Implementation

Added `tools/infer_mosev2_sam2_tiny_crop.py`:

- builds one crop video per eligible object;
- eligibility default: first-frame mask area <= 2% of the frame;
- crop center comes from M11 cycle-gated SAM2 when plausible, otherwise damped velocity / last position;
- frozen SAM2.1-B+ is run on the crop video;
- crop masks are mapped back to full-frame coordinates;
- non-eligible objects are intentionally empty, so this output is **candidate-only**.

Added `scripts/run_b101_tiny_crop.sh`:

- defaults to `MAKE_SUBMISSION=0`;
- refuses `MAKE_SUBMISSION=1` because candidate-only outputs would drop non-eligible objects;
- destructive prediction/work paths are limited to `pred_sam2_tiny_crop_candidate*` and `tiny_crop_work*` roots;
- audit paths are limited to `homework/logs/tiny_crop_candidate*` so M2/M11 evidence logs cannot be deleted by typo;
- records git dirty count, diff hash, and `outputs.out_pred_root` in the remote audit.

Added `scripts/run_b101_m4_tiny_state.sh`:

- wraps the M3 selector with `REQUIRE_TINY_CROP=1` by default;
- fixes M4-specific pred/submission/audit names so an intended M4 run cannot silently become ordinary M3.

Extended `tools/apply_m3_state_select.py`:

- optional `tiny_crop` source;
- `--require-tiny-crop` validates exact selected video/frame coverage before using it;
- `--tiny-crop-audit-json` links crop config/runtime into M3 provenance and must match the current tiny-crop prediction root when required;
- missing crop source/frame cannot silently masquerade as an intended M4 run;
- expected `obj_id` missing from SAM2 crop propagation becomes an empty candidate, never `k=0` fallback.

## Smoke experiment

Smoke videos: `1qlssuz2`, `r13u5z4y`, `lcgc29va`.

Remote commands used the b101 `mose_sam2` env and GPU 4:

1. Tiny-crop candidate source:
   - pred root: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_tiny_crop_candidate`
   - audit: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/tiny_crop_candidate_latest.json`
2. M3 + required tiny source, balanced selector:
   - pred root: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_m4_tiny_safe_balanced_smoke`
   - audit: `/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m4_tiny_safe_balanced_smoke.json`

## Evidence

### Tiny-crop source audit

```text
videos=3, frames=120, eligible_objects=3, conflict_frames=0
runtime=31.815s
CUDA peak allocated/reserved ~= 934.96 MiB / 1146.0 MiB
```

Per-video eligibility:

- `1qlssuz2`: eligible; 23/40 non-empty crop frames.
- `r13u5z4y`: eligible after threshold raised to 2%; 30/45 non-empty crop frames.
- `lcgc29va`: eligible; 26/35 non-empty crop frames.

### M3 + tiny safe smoke audit

```text
videos=3, frames=120, objects=3
changed_vs_baseline=62
source_counts={first_frame_gt:3, tiny_crop:46, sam31:3, baseline:30, empty:25, m2_light:10, m2:3}
class_counts={gt_anchor:3, accept_anchor_candidate:87, output_only:1, empty:22, accept_visible:4, reject:3}
state_counts={CONFIRMED_VISIBLE:94, UNCERTAIN:1, LIKELY_OCCLUDED:25}
conflict_frames=0
runtime=58.929s
```

`--require-tiny-crop` provenance confirmed exact frame coverage for the 3 selected smoke videos.

Visual sheets committed under:

- `docs/assets/m4_tiny_crop/safe_smoke/full/`
- `docs/assets/m4_tiny_crop/safe_smoke/zooms/`
- audits copied to `docs/assets/m4_tiny_crop/safe_smoke/audits/`

## Visual findings

### `1qlssuz2`

Tiny-crop is close to SAM2/M11 while the car is still in the crop track. It does not solve far-distance ambiguity. In late frames the selector still moves to larger vehicle masks from other sources, so M4 does not clearly improve identity preservation.

### `r13u5z4y`

This is the decisive negative case. Tiny-crop follows the early strawberry mask, but after the hand occlusion and relocation it relies mostly on velocity / weak guide context. In late frames it outputs a wrong board/background-adjacent blob that overlaps the already wrong SAM2-style evidence. Because it has source agreement with a wrong source, even the safe single-source guard cannot reject it.

Mechanism: crop rerun increases local resolution, but it does **not** add identity information. When the crop window is centered by a corrupted guide or velocity after long occlusion, SAM2 segments the most salient strawberry-colored/edge-like region in the crop, not necessarily the original instance.

### `lcgc29va`

Tiny-crop sometimes improves localization of the backpack/person mask and can be slightly tighter than full-frame SAM2. However it also inherits crowd/occlusion ambiguity and cannot reliably decide whether a distant tiny target is absent or merely unresolved.

## Conclusion

M4 tiny-crop is useful as a **diagnostic candidate source**, but it is not safe to promote to final or to full-run submission yet.

Reasons:

1. It is not independent identity evidence; it amplifies the guide source used to center the crop.
2. In `r13u5z4y`, it reproduces the core post-occlusion failure mode rather than fixing it.
3. M3 selector agreement is insufficient when two sources agree on the same wrong distractor/region.
4. The best evidence remains local and mixed: one video mildly positive (`lcgc29va`), one ambiguous (`1qlssuz2`), one clearly negative (`r13u5z4y`).

Therefore no full M4 submission zip was produced or downloaded. The local final remains `submission_mosev2_final_m11_cycle.zip`.

## Next research direction

If continuing beyond M4, the crop branch needs a stronger identity verifier before it can be used as final output. Candidate directions:

- use crop rerun only to propose masks, then verify with cycle-to-first-frame or exemplar similarity;
- require disagreement-aware evidence: if tiny-crop agrees with a known wrong post-gap tracklet, that should count as correlated error, not independent support;
- use target-vs-distractor competition with negative exemplars from nearby same-class instances;
- keep tiny-crop as `output_only` evidence unless it passes a stronger identity check.

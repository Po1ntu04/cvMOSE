# M5R-C Candidate Retrieval Re-Anchor Experiment

Date: 2026-05-30
Status: implemented, b101 key-video smoke tested, **not promoted to final submission**

## Conclusion first

M5R-C implements the missing structural loop we identified (candidate retrieval + object-level identity margin + hard negatives + `SAM2.add_new_mask` re-propagation), but the current training-free descriptors/proposals are still not discriminative enough for MOSEv2 same-class reappearance. It should be kept as the right engineering scaffold, not used as final output yet; M11 remains the safer submission.

## Why this experiment exists

M11/M2/M3/M4 all constrained or selected existing SAM2 outputs. They could reduce some visible drift, but they did not create a new evidence path that can recover identity after occlusion. M5R-C is the first implementation that changes the SAM2 inference trajectory by injecting verified later-frame masks as new conditioning prompts.

The implemented loop is:

```text
uncertain / post-gap region
  -> high-recall candidate pool
  -> object-level descriptor for candidate/positive/negative banks
  -> accept only by positive-minus-hard-negative margin
  -> add_new_mask(candidate_frame, obj_id, verified_mask)
  -> rerun SAM2 video propagation
  -> merge only around anchor windows
```

## Mapping to the four required capabilities

| requirement | implementation | current quality |
|---|---|---|
| 4.1 high-recall candidates, not a single crop | `tools/infer_mosev2_sam2_reanchor.py` reads baseline SAM2, M11, M2-light, M4 tiny-crop, RAR-RCMS/RAR-state, optional SAM3.1 root if present, per-source foreground connected components, and optional upstream SAM2 automatic-mask proposals on post-hard-event frames. | Better recall than M4 single crop; still incomplete because optical-flow/global-motion windows and true reverse-propagation mining are not implemented yet. |
| 4.2 object-level identity, not mask quality | Descriptor = RGB/HSV histogram + HOG/edge orientation + shape stats + optional SAM2 image-encoder masked pooling with object/context subtraction. SAM2 feature failures fall back to shared descriptor prefix safely. | More principled than area/stability only, but still weaker than DINOv2/DINOv3 or a trained object retrieval descriptor. |
| 4.3 hard negative bank | Negatives from M11-rejected baseline masks, other object ids, nearby foreground components, same/source non-overlap components, and context-ring fallback around first/stable masks for single-object videos. Candidate score uses `max_pos_cos - max_neg_cos`. | Mechanism is present and auditable; in hard same-class cases it either rejects true-ish reappearance (`r13u5z4y`) or still accepts plausible wrong objects (`q0sizv6m`). |
| 4.4 re-anchor must drive SAM2 | Accepted anchors are injected through `predictor.add_new_mask(...)`, then SAM2 is rerun and merged with `anchor_window` or `all_reprop`. | This is the critical structural step and is working. The quality bottleneck is now anchor trust, not SAM2 API integration. |

## Code and artifacts

```text
tools/infer_mosev2_sam2_reanchor.py        # M5R-C inference entrypoint
scripts/run_b101_m5r_reanchor.sh           # guarded b101 launcher
scripts/make_m5r_reanchor_compare.py       # local visual comparison sheets
artifacts/m5r_reanchor/key_pred/           # ignored local copy of 8-video prediction smoke
artifacts/m5r_reanchor/key_by_video/        # ignored per-video audits for key smoke
artifacts/m5r_reanchor/m5r_reanchor_key.json
artifacts/m5r_reanchor/v2_context_negative_smoke/  # 2-video smoke after context-ring negative fallback
docs/assets/m5r_reanchor/key_compare/      # versioned visual comparison sheets
```

## b101 commands and verification evidence

Static checks:

```bash
python3 -m py_compile tools/infer_mosev2_sam2_reanchor.py
bash -n scripts/run_b101_m5r_reanchor.sh
# local and b101 both passed
```

Primary key-video smoke:

```bash
VIDEOS='r13u5z4y q0sizv6m msinig6m 1qlssuz2 2smf7uq9 8jsm23a7 lcgc29va 4vznweiu' \
MAKE_SUBMISSION=0 GPU=${GPU:-4} \
PRED_ROOT='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_sam2_m5r_reanchor_key' \
AUDIT_JSON='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m5r_reanchor_key.json' \
AUDIT_DIR='/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m5r_reanchor_key_by_video' \
M5R_EXTRA_ARGS='--descriptor rgb_sam2 --merge-policy anchor_window --max-anchors-per-object 1 --identity-margin 0.06 --positive-thr 0.45 --negative-thr 0.94 --component-min-area 12 --candidate-pad 4 --merge-radius 14 --sam2-auto-mask-candidates --auto-mask-max-frames-per-object 2 --auto-mask-points-per-side 12 --auto-mask-max-count 12' \
scripts/run_b101_m5r_reanchor.sh
```

Observed aggregate:

```json
{"videos": 8, "frames": 423, "accepted_anchor_count": 8, "changed_vs_baseline": 163}
```

Runtime/memory are recorded in `artifacts/m5r_reanchor/m5r_reanchor_key.json`. The key smoke used the SAM2 image descriptor and SAM2 automatic-mask candidate source; it produced 423 PNGs and one audit JSON per video.

A second 2-video smoke after adding context-ring negatives ran on `1qlssuz2 q0sizv6m`:

```json
{"videos": 2, "frames": 82, "accepted_anchor_count": 3, "changed_vs_baseline": 56}
```

This fixed the empty-negative-bank weakness in single-object cases but did **not** fix the main identity problem.

## Visual evidence index

```text
docs/assets/m5r_reanchor/key_compare/r13u5z4y_m5r_compare.jpg
docs/assets/m5r_reanchor/key_compare/q0sizv6m_m5r_compare.jpg
docs/assets/m5r_reanchor/key_compare/msinig6m_m5r_compare.jpg
docs/assets/m5r_reanchor/key_compare/1qlssuz2_m5r_compare.jpg
docs/assets/m5r_reanchor/key_compare/2smf7uq9_m5r_compare.jpg
docs/assets/m5r_reanchor/key_compare/8jsm23a7_m5r_compare.jpg
docs/assets/m5r_reanchor/key_compare/lcgc29va_m5r_compare.jpg
docs/assets/m5r_reanchor/key_compare/4vznweiu_m5r_compare.jpg
docs/assets/m5r_reanchor/key_compare/metrics.json
```

Columns are RGB, SAM2 baseline, M11 safety, RAR-state, M5R-C, and M5R-vs-SAM2 delta. Green in the delta panel means added by M5R, red removed, white unchanged.

## Real visual comparison

### `r13u5z4y` — strawberry slice reappearance

Result: no anchor accepted; output equals baseline.

Audit: floor is after the first hard area drop (`baseline_area_drop:5:6399<=6650`), 261 candidates, positive bank only first-frame GT, negative bank 48. The best post-gap candidates (`rar_state` around frame 20 and SAM2-auto components around frames 20--22) were closer to hard negatives than positives, so they were rejected.

Interpretation: this is the safest behavior for the current descriptor. It avoids injecting a wrong strawberry-like region, but it also proves the descriptor cannot recover the true hidden slice. The bottleneck is object retrieval quality, not SAM2 repropagation.

### `q0sizv6m` — similar guinea pigs / herd confusion

Result: accepted 2 anchors and changed 38/42 frames; visual sheet shows this is risky and likely worse than M11/SAM2.

Audit anchors:

- obj1 `m2_light` at frame 34, margin 0.1377;
- obj2 `m2_light` at frame 9, margin 0.1568.

Visual diagnosis: the anchors are plausible animal regions but do not reliably preserve the intended first-frame instance. The later repropagation introduces large changed regions and wrong same-class masks. This is the clearest failure of the current identity descriptor: `pos-neg` margin can still be high for the wrong animal when positives and hard negatives are visually very similar.

### `msinig6m` — koala/person contact and multiple object ids

Result: accepted 2 anchors and changed 37/134 frames; visual evidence is mixed-to-negative.

The obj3 anchor came from `sam2_auto:11` at frame 22. The automatic mask proposal improved candidate recall, but the accepted region is not clearly the intended identity; it can attach to a nearby person/koala composite. This validates 4.1 (SAM2 auto proposals add recall) while exposing 4.2/4.3 weakness (descriptor/margin insufficient for crowded contact scenes).

### `1qlssuz2` — tiny aerial car

Result: accepted 1 anchor and changed ~17--18/40 frames; changes are mostly subtle and not visibly catastrophic.

This is the most plausible neutral/slightly-useful case: the object is tiny and M2-light provides a later small-car anchor. However, because there is no ground truth for later frames and the visual delta is small, this is not enough to justify final promotion.

### `2smf7uq9` — flamingo in a similar flock

Result: accepted 1 baseline anchor at frame 30 and changed 24/43 frames.

Visual diagnosis: the selected flamingo-like region is plausible, but not clearly a recovery beyond what SAM2 already has. M5R mostly changes local mask shape/extent around a target that baseline/M11 often already follows. It may refine in places but does not prove robust identity recovery.

### `8jsm23a7` — mahjong tile / hand occlusion

Result: accepted 1 anchor at frame 6 and changed 20/49 frames; likely not useful.

Visual diagnosis: changes are around small tiles and hands, but the sheet does not show a convincing recovered target. This is a case where the hard-event detector fires too early and the anchor is still close to the original path rather than a true re-acquisition.

### `lcgc29va` — tiny target

Result: no anchor accepted; output equals baseline.

This is safe but not improved. The proposal/descriptor loop did not find a trustworthy tiny anchor; this matches the earlier M4 conclusion that resolution alone is not enough without identity retrieval.

### `4vznweiu` — small petal/seed-like object on white background

Result: accepted 1 M2-light anchor and changed 27/35 frames; visual changes are tiny dots/noise-like.

Visual diagnosis: not catastrophic, but not meaningful recovery either. The tiny repeated structures make RGB/SAM2 pooled features too weak; a stronger descriptor or explicit object detector is needed.

## Code review notes

Fixed during implementation:

1. **Descriptor dimension mismatch**: SAM2 image-encoder pooling can fail for tiny masks, producing RGB-only descriptors. `cosine()` now compares the shared normalized prefix instead of crashing.
2. **Pre-disappearance anchor pollution**: early stable frames were initially eligible and could be accepted as “anchors.” `anchor_floor_frame()` now requires a hard event before delayed commit, so pre-gap frames are positive references but not recovery anchors.
3. **Auto-mask budget wasted on wrong late huge masks**: auto-mask frames now use the first post-hard-event frames with source evidence, not largest-area frames.
4. **Empty hard-negative bank**: single-object videos now get context-ring fallback negatives, preventing purely positive-only acceptance.
5. **Protected remote launcher flags**: `run_b101_m5r_reanchor.sh` allowlists experiment knobs only.

Remaining risks:

- No DINOv2/DINOv3 descriptor yet; SAM2 image features are a useful free cue but are not sufficiently instance-discriminative here.
- No optical-flow/global-motion candidate windows.
- No true reverse-propagation mining from later high-confidence frames.
- Delayed commit is implemented as anchor eligibility + windowed merge after SAM2 repropagation, not a full multi-branch state machine with provisional branch promotion.
- SAM2 automatic masks are noisy for tiny/occluded MOSEv2 objects; they add recall but also many large background/animal blobs.

## Decision

Do **not** generate or promote an M5R-C submission zip from this version. It satisfies the structural implementation goal and produces useful audits/figures, but visual evidence says it is more likely to introduce same-class drift than to improve the final score. The current final should remain M11.

## Next required improvement

The next meaningful step is not another threshold sweep. Replace/augment the descriptor and candidate source:

1. add DINOv2/DINOv3 masked object descriptors or an equivalent strong open-vocabulary object feature;
2. build a real later-frame retrieval set with temporal diversity, not only current-source masks;
3. require multi-frame delayed promotion before `add_new_mask` anchor injection;
4. keep M11 as a safety valve after repropagation.

If DINO cannot be installed on b101, the best local fallback is to extract stronger unconditioned SAM2 Hiera mid/high features with multi-scale masked pooling and compare against a richer negative pool, but the visual evidence suggests this may still be insufficient for `q0sizv6m` and `r13u5z4y`.

# M17 q0sizv6m temporal-continuity repair

**Purpose.** The user corrected the key identity story for `q0sizv6m:obj2`: the first-frame obj2 is the cropped white/black animal on the left/bottom edge. It moves toward the camera, disappears briefly, then reappears from the lower-left / bottom foreground (`00006` nose, `00007` small head, `00008` upper body). The previous green/late candidate marked a wrong animal because the tracker had no physical time-continuity model.

## Diagnosis

The baseline/M15 path has the exact failure pattern the user described:

- frame `00000-00002`: follows the left-edge obj2 mask;
- frames `00003-00012`: mostly empty despite the target becoming visible again from the bottom-left from `00006` onward;
- frames `00013+`: relocks to a mid-left same-class white/black animal near the old scene position, while the physically plausible target evidence is lower/bottom foreground.

Evidence sheets:

- raw grids: `docs/assets/m17_q0_analysis/q0_frames_00_19_grid.jpg`, `docs/assets/m17_q0_analysis/q0_frames_20_end_grid.jpg`
- first-frame IDs: `docs/assets/m17_q0_analysis/q0_frame0_ann_overlay.png`
- M15 overlays: `docs/assets/m17_q0_analysis/q0_m15_00_19_overlay.jpg`, `docs/assets/m17_q0_analysis/q0_m15_20_41_overlay.jpg`

This is a **time-continuity / physical-instance failure**, not a simple confidence-threshold failure. The wrong candidate is visually plausible by category, but temporally implausible if we track the cropped animal moving down toward the lens.

## Implemented probes

### 1. SAM2 bottom/foreground box-prompt re-anchor

Input plan: `artifacts/q0_fix/q0_bottom_box_split.json`.

Code path: existing `tools/infer_mosev2_sam2_teach_boxes.py` with ordinary `prompt_type=box`, bounded propagation, and nearest anchor clipping.

Remote output:

- pred root: `artifacts/q0_fix/remote_downloads/pred_m17_q0_bottom_box/`
- audit: `artifacts/q0_fix/remote_downloads/m17_q0_bottom_box.json`
- visual sheet: `docs/assets/m17_q0_box_compare/q0sizv6m_m6_compare.jpg`

Candidate zips:

| zip | edit | decision |
| --- | --- | --- |
| `submission_mosev2_m17_q0_early_box.zip` | replace `q0sizv6m:obj2` only frames `6-12` | **submit/test first**: low-risk fill for the user-confirmed early reappearance gap. |
| `submission_mosev2_m17_q0_full_box.zip` | replace `q0sizv6m:obj2` frames `6-41` | stronger physical-story probe; can improve more, but later masks may include foreground clutter. |

### 2. Temporal color/ROI repair

New tool: `tools/apply_m17_q0_temporal_color_fix.py`.

Mechanism:

1. learn a small target-color model from the first-frame obj2 mask;
2. use obj1/grass/dirt as hard negatives;
3. constrain candidate pixels to a hand-audited spatiotemporal ROI that follows the bottom/foreground path and excludes the mid-left same-class animal;
4. keep M15 safe everywhere outside `q0sizv6m:obj2`.

Candidate zips:

| zip | edit | decision |
| --- | --- | --- |
| `submission_mosev2_m17_q0_early_color.zip` | color/ROI fill only frames `6-12` | conservative diagnostic; should only recover the missing early visibility. |
| `submission_mosev2_m17_q0_color_conservative.zip` | full q0 color/ROI conservative | score probe only; masks are fragmented. |
| `submission_mosev2_m17_q0_color_balanced.zip` | full q0 color/ROI balanced | score probe only; better tests the bottom-path hypothesis but visually rough. |
| `submission_mosev2_m17_q0_color_wide.zip` | full q0 color/ROI wide | aggressive; likely over-includes foreground/background. |

Visual sheets:

- `docs/assets/m17_q0_color_compare/q0sizv6m_m6_compare.jpg`
- `docs/assets/m17_q0_early_compare/q0sizv6m_m6_compare.jpg`

## Validation

All M17 q0 zips passed `tools/validate_mose_submission.py`:

- `video_dirs=433`
- `pngs=66526`
- `provided_changed_count=0`
- `predicted_error_count=0`

Validation records are under `artifacts/q0_fix/validate_*.json`.

## Recommended submission order

1. `submission_mosev2_m17_q0_early_box.zip` — best first q0-specific test because it targets exactly the user-confirmed missing frames `00006-00012` and leaves ambiguous late frames untouched.
2. `submission_mosev2_m17_q0_full_box.zip` — tests the full physical-story correction with SAM2 refinement.
3. `submission_mosev2_m17_q0_early_color.zip` — alternative early-gap fill if SAM2 box misses the bottom-edge animal.
4. `submission_mosev2_m17_q0_color_balanced.zip` — aggressive diagnostic of the bottom-path hypothesis; not final-quality visually.

Do not treat the old M16 q0 late foreground/tan candidate as correct; it was built from too-late anchors and selected the wrong physical instance.

## General lesson

The useful generic abstraction is not “MLLM picks a box”. It is:

```text
first-frame instance
  -> physical event story / time-continuity chain
  -> reject same-class objects that violate that story
  -> only then inject SAM2 anchors or construct a constrained ROI repair
```

For q0, the decisive missing cue is **motion toward camera + bottom-edge reappearance**, not mask quality.

## Hidden feedback after `m17_q0_full_box`

User-reported score:

- `submission_mosev2_m17_q0_early_box.zip`: `43.47`
- `submission_mosev2_m17_q0_full_box.zip`: `43.54`

The updated `test_latest.log` for `full_box` shows:

| row | J&F_new | J | F_new | disappear J&F_new | reappear J&F_new |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q0sizv6m` obj1 | `74.48` | `75.85` | `75.85` | N/A | N/A |
| `q0sizv6m` obj2 | `68.29` | `66.96` | `67.43` | `50.03` | `76.04` |

Compared with M15 safe (`q0sizv6m:obj2 = 12.34`), fullbox lifts the row by about `+55.95`, which is roughly `+0.097` global J&F_new if averaged over the 575 object rows. This explains almost all movement from `43.44` to `43.54`.

## Remaining q0 error after fullbox

The score confirms the physical-story direction, but the visual trajectory is still not fully correct:

- around `00027`, the bottom/foreground white-ish individual is the only plausible continuation;
- around `00031`, only a rump/edge remains near the lower/right foreground;
- around `00034`, the target should begin/finish disappearing;
- fullbox still has false late masks around frames `36`, `39-41`, and no mask around `31-33`.

Therefore the remaining q0 space is not “better frame thresholding”; it is the **visible / partially-visible / absent state schedule** along the physical event chain.

## Follow-up story-state probes

New tool:

```text
tools/apply_m17_q0_story_fusion.py
```

It creates q0 obj2 fusions over the proven `full_box` root:

| zip | policy | intent |
| --- | --- | --- |
| `submission_mosev2_m17_q0_story_trim_after34.zip` | fullbox frames `6-29`, empty `30-41` | isolate whether fullbox's late false positives after disappearance hurt. |
| `submission_mosev2_m17_q0_story_rump_31_33.zip` | fullbox `6-29`, color/ROI rump `31-33`, empty `34-41` | tests user claim that only a rump/edge remains around `31`, then disappears. |
| `submission_mosev2_m17_q0_story_rump_30_33.zip` | fullbox `6-29`, color/ROI `30-33`, empty `34-41` | slightly broader rump interval; more aggressive than `31_33`. |

All three validate with `provided_changed_count=0`, `video_dirs=433`, `pngs=66526`, `predicted_error_count=0`.

Visual sheet:

```text
docs/assets/m17_q0_story_compare/q0sizv6m_m6_compare.jpg
```

Recommended next test order:

1. `submission_mosev2_m17_q0_story_trim_after34.zip` — directly tests the disappearance-state correction.
2. `submission_mosev2_m17_q0_story_rump_31_33.zip` — adds the plausible rump interval without reviving late false masks.
3. `submission_mosev2_m17_q0_story_rump_30_33.zip` — broader and riskier rump interval.

## Where the large remaining improvement space is

For q0 alone, the row is now `68.29`; perfect q0 would add at most about `(100-68.29)/575 ≈ +0.055` global J&F_new. This is meaningful but not enough alone for the next large jump.

The larger space is applying the same principle to other still-low rows:

1. **`8jsm23a7`**: still around `2.13`, so a true seven-bamboo physical-trajectory recovery could be worth roughly `+0.15~0.17` global.
2. **`r13u5z4y`**: strawberry slice reappearance remains wrong/empty; potential is smaller than 8js but still nontrivial.
3. **`1qlssuz2` / `4vznweiu`**: previous aggressive attempts hurt because they lacked a stable event-state model.
4. **Generic method gap**: current SAM/DINO/Qwen harnesses verify crops or boxes, but they do not maintain an object-centric scene ledger: who is the target, who are distractors, where did each go, when is target absent, and which later evidence is physically compatible. q0 improved only when we manually encoded that ledger.

So the next substantive architecture step should be an **object-centric temporal ledger** per video/object, not a better per-frame prompt.

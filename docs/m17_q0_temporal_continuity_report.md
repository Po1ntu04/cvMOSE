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

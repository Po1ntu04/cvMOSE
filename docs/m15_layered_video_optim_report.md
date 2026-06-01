# M15 layered per-video optimization report

## One-line result

The most defensible new candidate is **M15 safe**: keep current `zofficial-balanced` everywhere and add only the user/visual-reviewed `amfdu83t:obj1` same-class kangaroo reappearance continuation; `8jsm23a7` and tiny/semantic probes are implemented but remain score-probing/aggressive because visual identity is still ambiguous.

## Objective

Systematically revisit the three failure classes identified from MOSE15, then produce candidate submissions without touching the 418 provided outputs:

1. **Event-chain / physical-instance reasoning**: `8jsm23a7`, `q0sizv6m:obj2`, `amfdu83t:obj1`.
2. **Same-class reappearance / hard-negative scenes**: `q0sizv6m:obj2`, `r13u5z4y`, `4f98052b:obj2`, `amfdu83t:obj1`.
3. **Tiny / semantic-object scenes**: `4vznweiu`, `1qlssuz2`, `lcgc29va`.

The invariant for every generated zip is: 433 video dirs, 66526 PNGs, 418 provided outputs unchanged, first-frame labels preserved by construction.

## Code changes

- `tools/mllm_sameclass_atlas.py`
  - Added compact atlas panel mode to avoid qwen3.6-plus timeout on large multi-frame panels.
  - Added per-frame retry path with smaller panels and shorter JSON prompt.
  - Kept the same contract: Qwen only proposes candidates/hard negatives; it does not output masks or promote anchors.

Validation:

```bash
conda run -n cv-hw2 python -m py_compile \
  tools/mllm_sameclass_atlas.py \
  tools/mllm_event_story_split.py \
  tools/infer_mosev2_sam2_teach_boxes.py
```

## Qwen3.6 split-event analysis

Artifacts:

- Combined event JSON: `artifacts/m15_layered/event_split_m15_combined.json`
- Tiny/semantic output doc: `docs/m15_layered_event_split_tiny_semantic.md`
- Initial visual compare sheets: `docs/assets/m15_initial_compare/`

| video/object | class | Qwen3.6 event | diagnosis | action | confidence | decision |
| --- | --- | --- | --- | --- | ---: | --- |
| `8jsm23a7:1` | event-chain + same-class tile | `picked_moved_placed` | says mostly-correct, but warns tile swap risk | `reanchor_at_frame` | 0.80 | Implemented probe, not recommended safe. |
| `q0sizv6m:2` | same-class animal + foreground motion | `moves_to_foreground` | `wrong_same_class` | `reanchor_at_frame` | 0.45 | Rejected: no reliable positive late box; high same-class/composite risk. |
| `amfdu83t:1` | same-class animal + reappearance | `exits_reappears` | uncertain | `reanchor_at_frame` | 0.65 | Use prior M14 post-anchor continuation as safe/balanced probe. |
| `r13u5z4y:1` | same-class strawberry + occlusion | `multi_object_contact` | uncertain | `reanchor_at_frame` | 0.55 | Rejected: Qwen only produced frame-0 box, no later recovery candidate. |
| `4vznweiu:1` | tiny semantic bead | `static` | uncertain | `generate_detector_candidates` | 0.65 | Probe implemented; visual result risky. |
| `1qlssuz2:1` | tiny vehicle + camera motion | `camera_motion` | uncertain | `reanchor_at_frame` | 0.65 | Probe implemented; visual result overlarge/risky. |
| `lcgc29va:1` | tiny object | unknown | uncertain | manual review | 0.30 | Rejected: no precise localization. |
| `4f98052b:2` | repeated chair/person-like scene | static | uncertain | `reanchor_at_frame` | 0.65 | Rejected for safe/balanced: repeated identical chairs, no reliable semantic distinction. |

## Implemented probes and visual verdicts

### `amfdu83t:obj1` — safe candidate source

Source: existing M14 no-clip continuation.

- Candidate root: `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m14_zofficial_amfdu_from8_noclip`
- Fusion frames: `8-23`
- Visual: `docs/assets/m15_layered_fusion_compare/amfdu83t_m6_compare.jpg`
- Verdict: **best new safe probe**. It fills an otherwise empty post-reappearance interval with a single plausible kangaroo track. The risk is still identity mismatch after frame 11, but it is much more mechanism-aligned than the previous radius-0 truncation.

### `8jsm23a7:obj1` — balanced/aggressive only

New Qwen3.6 split harness no longer hard-asserts “leftmost seven bamboo”; it asks to verify the event chain and treat all similar tiles as hard negatives. Qwen proposed a frame-20 post-placement box rather than the previous frame-48 endpoint.

Remote SAM2 runs:

- `pred_m15_8js_frame20_window`: frame-20 + frame-2 bounded window probe.
- `pred_m15_8js_frame20_fromanchor`: frame-20 from-anchor propagation used in M15 balanced/aggressive.

Visuals:

- `docs/assets/m15_8js_frame20_window_compare/8jsm23a7_m6_compare.jpg`
- `docs/assets/m15_8js_frame20_fromanchor_compare/8jsm23a7_m6_compare.jpg`
- `docs/assets/m15_layered_fusion_compare/8jsm23a7_m6_compare.jpg`

Verdict: **not safe**. It moves the mask into the player's front-row tile sequence, which is mechanism-aligned, but the exact tile identity remains ambiguous. This should be hidden-score probed only after the safe candidate.

### `1qlssuz2:obj1` — aggressive only

Remote SAM2 run:

- `pred_m15_1ql_window`

Visual:

- `docs/assets/m15_1ql_window_compare/1qlssuz2_m6_compare.jpg`

Verdict: **rejected for safe/balanced**. The probe often expands the tiny car mask substantially. It may help if hidden GT rewards coarse object presence, but visually it is less precise than the current track and thus only appears in aggressive.

### `4vznweiu:obj1` — aggressive only

Remote SAM2 run:

- `pred_m15_4v_window`

Visual:

- `docs/assets/m15_4v_window_compare/4vznweiu_m6_compare.jpg`

Verdict: **rejected for safe/balanced**. The probe sometimes shifts from the intended `T`-like bead toward adjacent letter beads. The right next step is an explicit high-resolution candidate-letter atlas, not more SAM2 box propagation.

### Rejected without propagation

- `q0sizv6m:obj2`: Qwen confirmed the conceptual issue but did not provide a reliable late positive. Same-class animal identity remains too risky; existing SAM2Long/DAM/official candidates are large/composite or wrong-position.
- `r13u5z4y:obj1`: no reliable late strawberry-slice candidate; all current roots remain empty after the occlusion.
- `lcgc29va:obj1`: Qwen confidence only 0.30 and no precise box.
- `4f98052b:obj2`: Qwen identifies repeated chair-like distractors; same-class dense static scene is unsafe without a true object-level candidate atlas.

## Submission candidates

All zips pass validator with `provided_changed_count=0` and `predicted_error_count=0`.

| candidate | path | changed target videos | recommendation |
| --- | --- | --- | --- |
| M15 safe | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m15_layered_safe.zip` | `amfdu83t` | Submit first if testing the new M15/M14 kangaroo improvement. |
| M15 balanced | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m15_layered_balanced.zip` | `amfdu83t`, `8jsm23a7` | Probe after safe; contains risky Mahjong front-row reanchor. |
| M15 aggressive | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m15_layered_aggressive.zip` | `amfdu83t`, `8jsm23a7`, `1qlssuz2`, `4vznweiu` | Score-probing only; not recommended as final unless hidden feedback surprises. |

Repo hardlinks:

- `submission_mosev2_m15_layered_safe.zip`
- `submission_mosev2_m15_layered_balanced.zip`
- `submission_mosev2_m15_layered_aggressive.zip`

Validation artifacts:

- `artifacts/m15_layered/validate_safe.json`
- `artifacts/m15_layered/validate_balanced.json`
- `artifacts/m15_layered/validate_aggressive.json`

Fusion policies:

- `artifacts/m15_layered/policy_safe.json`
- `artifacts/m15_layered/policy_balanced.json`
- `artifacts/m15_layered/policy_aggressive.json`

## What this round proves

1. **Event-chain MLLM is useful for diagnosis/routing, but still not a reliable coordinate generator.** It explains `8js`, `q0`, `amfdu`, but only `amfdu` currently has a visually defensible track after SAM2 propagation.
2. **The main bottleneck remains high-recall, instance-separated candidate generation.** For `q0`, `r13`, `4v`, and `4f`, MLLM can describe the failure but cannot invent a reliable mask/box without a candidate atlas/detector.
3. **Safe fusion must remain conservative.** The videos with biggest theoretical upside (`8js`, `q0`) are also the easiest to make worse by selecting a plausible but wrong same-class instance.

## Next high-value step

Build a true **candidate atlas + tracker bridge** for same-class/tiny cases:

- enumerate all similar objects per key frame with a detector/proposal source;
- assign stable candidate IDs over time;
- ask Qwen3.6 only to judge identity/negative roles among numbered candidates;
- inject only verified candidate boxes/masks into SAM2 with delayed commit.

This is specifically needed for `8jsm23a7`, `q0sizv6m:obj2`, `r13u5z4y`, and `4vznweiu`.

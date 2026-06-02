# M7 overlay-panel fix and Qwen source-select ablation

**One-line verdict:** Filled mask overlays were leaking artificial color into Qwen identity reasoning; switching MLLM panels to raw-crop plus thin artificial outline fixed the worst semantic hallucination and enabled one conservative output-level recovery candidate on `lcgc29va` frames 8-9.

## Why this follow-up was needed

The first tiny/semantic Qwen pass supported several candidates, but visual review found a critical prompt/panel failure:

- `4vznweiu` frame 16: Qwen selected an M5R candidate on the wrong `T/O` die.  Its language treated the green mask overlay as target appearance ("green T die"), even though green was only an artificial mask color.
- `lcgc29va`: Qwen target/profile language also over-weighted overlay color and occasionally confused backpack/person/clothing cues.

This is not a model-score issue; it is an evidence-presentation bug.  A verifier cannot be trusted if the UI panel makes annotation color look like object color.

## Code changes

| file | change |
| --- | --- |
| `src/cvmose/mllm_panels.py` | Added outline-only mask rendering for MLLM target/candidate/tracklet panels.  Candidate crop cells now show raw crop plus artificial outline, not filled green semantic-looking overlay. |
| `tools/mllm_target_profile.py` | Added explicit prompt warning that overlay colors/boxes/letters are artificial. |
| `tools/mllm_candidate_judge.py` | Same warning for candidate verification; the model is instructed to rely on raw crop/context appearance. |
| `tools/mllm_tracklet_judge.py` | Same warning for multi-frame promotion decisions. |
| `tools/apply_m7_qwen_source_select.py` | New ablation tool: copy an existing source mask only when Qwen explicitly supports that candidate.  It is output-level source selection, not mask generation.  Added a non-empty area-ratio guard so Qwen cannot replace a valid non-empty M11 mask with a much smaller/larger source unless explicitly allowed. |

## Real Qwen evidence after outline panels

### `4vznweiu`

Outline panels changed the key bad decision:

- old filled-overlay pass: frame 16 selected `m5r`, visually wrong same-class die;
- outline pass: frame 16 became `uncertain` + veto (`confidence=0.30`), correctly refusing the wrong `T/O`-like distractor.

This is a qualitative verifier improvement: it prevents a bad semantic support signal.

### `lcgc29va`

Outline panels still supported two post-empty candidates from `rar_state`:

| frame | selected source | confidence | default/M11 | visual reading |
| ---: | --- | ---: | --- | --- |
| 8 | `rar_state` | 0.90 | empty | likely re-detects the small target person/backpack near the turnstile context |
| 9 | `rar_state` | 0.85 | empty | same short recovery continuation; no replacement of an existing non-empty M11 mask |

A third potential replacement at frame 3 was rejected by the area-ratio guard: `official_large` was only 0.768x M11 area, while M11 was already non-empty and visually not clearly worse.

## Candidate zips

### Source-select guarded ablation

- zip: `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m7_qwen_source_select_outline_guarded_v2.zip`
- pred root: `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m7_qwen_source_select_outline_guarded_v2`
- changed vs M11: 2 frames, both `lcgc29va` (`00008`, `00009`)
- validation: pass; 433 video dirs, 66526 PNGs, 418 provided outputs unchanged
- recommendation: useful diagnostic / low-risk probe, but expected score impact is tiny because only two frames change.

### Balanced-plus fusion

- zip: `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m7_qwen_balanced_plus_outline.zip`
- pred root: `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m7_qwen_balanced_plus_outline`
- policy: M11 default + prior Qwen-supported `1qlssuz2` M5R repropagation + guarded `lcgc29va` frame 8-9 source-select.
- compare audit: 95 changed frames vs SAM2 baseline, 35 changed frames vs M11; no first-frame failures, no missing frames.
- validation: pass; 433 video dirs, 66526 PNGs, 418 provided outputs unchanged.
- recommendation: best current M7 probe after M11/M6 balanced.  It is still not expected to jump to 44 because it recovers only small local cases, not the hard same-class reappearance videos.

## Visual evidence

| evidence | path |
| --- | --- |
| First bad source-select smoke, showing why filled overlay was unsafe | `docs/assets/m7_qwen_vl/source_select_smoke/` |
| Outline source-select sheets | `docs/assets/m7_qwen_vl/source_select_outline_guarded/` |
| `lcgc29va` focused guarded zoom | `docs/assets/m7_qwen_vl/source_select_outline_guarded/lcgc29va_guarded_zoom.jpg` |
| Balanced-plus final sheets | `docs/assets/m7_qwen_vl/balanced_plus_outline/` |

## Interpretation

M7 is now a better verifier than before: it can veto panel-induced semantic false positives and can safely accept a tiny number of visually plausible recovery frames.  However, the experiment also confirms the larger bottleneck: without a high-recall external proposal that contains the correct post-occlusion instance, Qwen cannot create substantial recovery by itself.  The next high-value work remains candidate generation / object retrieval, with Qwen kept as a conservative semantic verifier.

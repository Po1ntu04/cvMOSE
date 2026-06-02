# M13 Round 2 key3 teach-SAM smoke

## Setup

- Source split JSON: `artifacts/m12_event_story_split/key3_split_q36_fixed_v2.json`
- Tool: `tools/infer_mosev2_sam2_teach_boxes.py`
- Remote smoke pred roots downloaded locally:
  - unclipped: `artifacts/m13_teach_boxes_smoke/pred_m13_teach_boxes_key3_smoke`
  - clipped: `artifacts/m13_teach_boxes_clip/pred_m13_teach_boxes_key3_clip`
- Visual sheets:
  - unclipped: `docs/assets/m13_round2_key3_smoke/`
  - clipped: `docs/assets/m13_round2_key3_clip/`
- Conservative fusion zip: `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m13_key3_safe_fusion.zip`

## Findings

| target | action | local visual/proxy verdict |
| --- | --- | --- |
| `8jsm23a7:1` | final-frame box prompt on front-left 七条 | Useful as a targeted one-frame probe. It creates a non-empty mask at the event endpoint where M7 was tracking a static/middle tile path. Risk: box coordinates are coarse; should not propagate broadly without review. |
| `q0sizv6m:2` | final-frame foreground animal box prompt | Potentially useful for testing the user-corrected hypothesis that obj2 becomes the foreground/bottom animal. Risk remains high because same-class animals overlap and hidden score previously rejected plausible q0 variants. |
| `z6dx46qr:1` | early low-contrast box prompt | Unsafe. Even with clip-to-anchor-box, SAM2 box prompting creates large blob masks in early frames and should not enter safe/balanced fusion. Better route is official/local-window interval or a two-dot detector, not direct MLLM box prompt. |

## Implemented safeguards after smoke

- Added `--clip-to-anchor-box` and `--clip-pad-frac` to `tools/infer_mosev2_sam2_teach_boxes.py` after z6 produced large blobs.
- Added `tools/apply_m13_object_fusion.py` for policy-controlled object/frame fusion rather than whole-root replacement.
- Created a safe diagnostic fusion that changes only:
  - `8jsm23a7:1` frame 48
  - `q0sizv6m:2` frame 41
- Validator result: passed, 433 videos / 66526 PNGs, `provided_changed_count=0`.

## Decision

Do not promote direct teach-box propagation as a global method. Use it only as a tightly bounded probe for event endpoints. For low-contrast and same-class dense cases, MLLM should supply event roles and hard negatives; SAM box prompting must be constrained by independent candidate/source evidence.

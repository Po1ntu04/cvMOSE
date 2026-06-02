# M13 Round 2B reverse re-anchor probe

## Why this was necessary

The first teach-SAM smoke used a late Qwen event anchor but only propagated SAM2 forward.  That made endpoint anchors almost score-invisible: a final-frame anchor can change only the last frame, while most of the hidden score impact is in the post-reappearance interval.  SAM2's video predictor supports `reverse=True`, so M13 adds a bounded reverse-propagation path to test whether a verified late anchor can repair the preceding reappearance segment.

## Implementation

- Tool changed: `tools/infer_mosev2_sam2_teach_boxes.py`
- New arguments:
  - `--propagate-direction forward|reverse|both`
  - `--reverse-max-frames`
- Smoke command shape:

```bash
LOCAL_SPLIT_JSON=artifacts/m12_event_story_split/key3_split_q36_fixed_v2.json \
PRED_ROOT=/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/pred_m13_teach_boxes_key3_revwin \
AUDIT_JSON=/data1/yuzhixiang/cv_mosev2/MOSEv2/homework/logs/m13_teach_boxes_key3_revwin.json \
VIDEOS="8jsm23a7 q0sizv6m" \
TARGETS="8jsm23a7:1 q0sizv6m:2" \
M13_EXTRA_ARGS="--merge-mode window --merge-radius 12 --propagate-direction both --min-confidence 0.80 --max-actions-per-object 1" \
DOWNLOAD_ROOT=artifacts/m13_teach_boxes_revwin \
MAKE_SUBMISSION=0 GPU=4 SYNC_CODE=1 scripts/run_b101_m13_teach_boxes.sh
```

Artifacts:

- Audit: `artifacts/m13_teach_boxes_revwin/m13_teach_boxes_key3_revwin.json`
- Compare CSV: `artifacts/m13_teach_boxes_revwin/compare_key3_revwin.csv`
- Visual sheets: `docs/assets/m13_round2_key3_revwin/`

## Result by target

| target | changed frames | visual verdict | decision |
| --- | ---: | --- | --- |
| `8jsm23a7:1` | 13 (`36-48`) | Positive local proxy. Reverse propagation moves the mask from the stale middle-pile/old tile path to the front-left seven-bamboo tile segment identified by the event story. This directly addresses the user-observed failure mode: the tile is picked, moved, and placed in front. | Keep as a bounded fusion candidate. |
| `q0sizv6m:2` | 13 (`29-41`) | Negative. The reverse anchor produces large foreground/composite guinea-pig masks and corrupts multiple same-class frames. | Reject from safe/balanced fusion; keep only as diagnostic evidence. |

## Key lesson

The structural issue was not only semantic identity. It was also propagation direction. Late MLLM anchors must drive **bounded backward propagation** to affect the interval where the tracker was wrong.  However, this helps only when the anchor is geometrically clean; same-class animal scenes still need a stricter candidate verifier or external tracker before SAM2 injection.

## Independent visual review

A separate vision review of `docs/assets/m13_round2_key3_revwin/` agreed with the fusion decision:

- `8jsm23a7:1`: **better**. `m13rev` follows the target to the front-left seven-bamboo tile after it is picked and placed. It may be slightly over-expanded, but it matches the event chain better than SAM2/M11/m13clip.
- `q0sizv6m:2`: **worse**. `m13rev` merges multiple foreground guinea pigs and creates very large composite masks, so it must stay out of safe/balanced submissions.

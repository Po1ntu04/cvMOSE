# M19 story-guided hard-video probes

## Goal

Use the user's confirmed physical-instance stories to test whether direct story-grounded re-anchoring can move the remaining low rows, especially `8jsm23a7`.  This is a diagnostic/probe layer on top of the current best known local base root:

- base root: `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m17_q0_full_box`
- base hidden score memory: `43.54` from `submission_mosev2_m17_q0_full_box.zip`

No later GT is used.  The edits are manual visual/story box masks, not trained predictions.

## Key correction versus M16

M16's `8jsm23a7` mask-box probe used normalized y boxes around `520..700`, which map to pixel y ≈ `998..1344` in a 1920 px tall frame.  Visual review shows this targets the table/hand/edge region below the front row, not the moved Mahjong tile.

M19 corrects the event geometry:

- target after placement: **front/player-side row, leftmost seven-bamboo tile**.
- pixel y should be around `742..924`, i.e. normalized y ≈ `383..486`.
- far/upper row two-bamboo and old-position tiles are hard negatives.

## Implemented tool

- `tools/apply_m19_story_guided_masks.py`

Profiles:

| profile | edits | purpose |
| --- | --- | --- |
| `8js_late` | `8jsm23a7:obj1` frames 6-48 only | safest M19 probe; fixes the known M16 coordinate error without touching hand-held frames. |
| `8js_full` | `8jsm23a7:obj1` frames 2-48, including coarse hand-held boxes at frames 3-5 | higher-upside but coarse early-frame risk. |
| `8js_late_1ql` | `8js_late` + `1qlssuz2:obj1` frames 13-14 same-lane bridge fill | tests the user's car-lane story with only two added frames. |
| `8js_late_4vz` | `8js_late` + `4vznweiu:obj1` frames 14-34 T-letter die boxes | tests the letter-die story; risky because boxes are larger than current SAM masks. |
| `8js_late_4vz_1ql` | combined late 8js + 4vz + 1ql | balanced/aggressive probe after isolated feedback. |
| `8js_4vz`, `8js_4vz_1ql` | full early 8js + 4vz(+1ql) | aggressive only. |

## Generated candidate zips

All passed `tools/validate_mose_submission.py` with 433 video dirs, 66526 PNGs, 15 predicted videos, and `provided_changed_count=0`.

| recommended order | zip | changed target frames | verdict |
| ---: | --- | --- | --- |
| 1 | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m19_8js_late.zip` | `8jsm23a7:1` frames 6-48 | Submit first. It isolates the corrected Mahjong front-row y-coordinate and has the cleanest score signal. |
| 2 | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m19_8js_late_1ql.zip` | plus `1qlssuz2:1` frames 13-14 | Low extra risk: only fills two bridge-occlusion frames along the same lane. |
| 3 | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m19_8js_late_4vz.zip` | plus `4vznweiu:1` frames 14-34 | Higher risk: box masks may overcover the T-letter die and hurt if the target identity hypothesis is wrong. |
| 4 | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m19_8js_late_4vz_1ql.zip` | combined late 8js + 4vz + 1ql | Use only after isolated feedback. |
| 5 | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m19_8js_full.zip` | `8jsm23a7:1` frames 2-48 | Tests whether hand-held frames matter, but early coarse boxes can hurt. |
| 6 | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m19_8js_4vz.zip` | full 8js + 4vz | Aggressive. |
| 7 | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m19_8js_4vz_1ql.zip` | full 8js + 4vz + 1ql | Most aggressive. |

## Visual evidence

- `docs/assets/m19_story_guided_compare/8jsm23a7_compare.jpg`
- `docs/assets/m19_story_guided_compare/4vznweiu_compare.jpg`
- `docs/assets/m19_story_guided_compare/1qlssuz2_compare.jpg`

Additional RGB/grid sheets are local artifacts under `artifacts/m19_story_guided/` and are not meant for large-scale submission.

## Interpretation before hidden feedback

- `8js_late` is the clearest correction.  It directly fixes the earlier coordinate-level mistake and replaces the old/far-row Mahjong track with the front-row leftmost tile after frame 6.
- `8js_full` may improve if hidden GT annotates the hand-held tile in frames 3-5, but its boxes are coarse and may include hand/background.  It is not the first test.
- `4vznweiu` remains uncertain.  The user's story says the target is the T-letter die, but the current box implementation is still rectangular and may overcover neighboring dice/shadow.  It should be isolated.
- `1qlssuz2` is intentionally minimal: only frames 13-14, where the car should continue under/near the bridge.  If it helps, future work can build a proper lane-constrained vehicle tracker.

## Verification

```bash
conda run -n cv-hw2 python -m py_compile tools/apply_m19_story_guided_masks.py
conda run -n cv-hw2 python tools/validate_mose_submission.py --workspace "/home/yu/projects/cv/from fdu/MOSEv2" --zip-path <candidate.zip> --output-json artifacts/m19_story_guided/validate_<profile>.json
```

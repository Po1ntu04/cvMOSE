# M13 long-run implementation report

## Objective

Turn the fixed Qwen3.6 split harness into a sustained improvement loop for MOSEv2: use MLLM event understanding to create high-information anchors, inject them into SAM2 only under bounded policies, compare visually, and produce submission candidates without touching the 418 provided outputs.

## Round summary

| round | work completed | evidence | decision |
| --- | --- | --- | --- |
| R1 split harness | Ran qwen3.6-plus split event-story on all 15 videos / 20 objects; fixed frame selection so explicit late/reappearance frames are not dropped. | `docs/m13_round1_split_all15.md`, `docs/m13_round1_proxy_report.md`, `docs/m13_round1_fixed_key5_split.md`, `docs/m13_round1_fixed_key5_proxy_report.md` | MLLM is useful for event roles but often returns `uncertain`; use as routing/evidence, not as a standalone anchor promoter. |
| R2 teach-SAM smoke | Implemented SAM2 box-prompt injection from split JSON and ran key probes on b101. | `docs/m13_round2_key3_teach_boxes_smoke.md`, `docs/assets/m13_round2_key3_smoke/`, `docs/assets/m13_round2_key3_clip/` | Direct forward-only injection is too weak for late anchors; z6 direct box is unsafe. |
| R2B reverse re-anchor | Added reverse propagation (`--propagate-direction both`) so a late event anchor can repair the preceding window. | `docs/m13_round2_reverse_reanchor.md`, `docs/assets/m13_round2_key3_revwin/` | Clear local proxy improvement on `8jsm23a7`; clear rejection on `q0sizv6m`. |
| R3 safe fusion | Implemented object/frame-level fusion and generated isolated/safe/balanced candidates. | `docs/m13_candidate_fusions.md`, `docs/assets/m13_revwin_fusions/`, validation logs | Produce zips for hidden-score feedback; never whole-root replace an unsafe candidate. |
| R4 TEP review | Read the 1st-place TEP report and mapped our remaining gaps. | `docs/m13_round4_tep_review.md` | Next true breakthrough likely needs external image-prompt tracker or stronger detector source, not more threshold sweeps. |

## Current candidate submissions

| candidate | path | validator | local intent |
| --- | --- | --- | --- |
| `m13_8rev_safe` | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m13_8rev_safe.zip` | pass; 433 videos; 66526 PNGs; provided unchanged | Isolate Qwen event reverse-anchor effect for `8jsm23a7:1` frames 36-48. |
| `m13_8rev_zofficial_balanced` | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m13_8rev_zofficial_balanced.zip` | pass; 433 videos; 66526 PNGs; provided unchanged | Combine `8jsm23a7` reverse-anchor with `z6dx46qr` official-large interval. |
| `m13_key3_safe_fusion` | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m13_key3_safe_fusion.zip` | pass | Earlier endpoint-only probe, likely lower impact than reverse-anchor. |
| `m13_key3_balanced_zofficial` | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m13_key3_balanced_zofficial.zip` | pass | Earlier endpoint + z official probe, likely lower impact than reverse-anchor. |

## Main technical findings

1. **Propagation direction was a real implementation bottleneck.** Late MLLM anchors need reverse propagation; otherwise a correct final anchor changes only one frame and hidden score barely moves.
2. **Qwen semantic event reasoning is valuable but geometry is coarse.** It correctly explains `8jsm23a7` as picked/moved/placed and gives a usable endpoint prior, but same-class animal scenes remain high risk.
3. **Same-class dense scenes need hard-negative candidate sources before injection.** `q0sizv6m` reverse propagation produced large composite masks and was rejected from new fusions.
4. **Low-contrast tiny targets need external tracker/detector support.** Direct SAM2 box prompts oversegment `z6dx46qr`; official-large interval is safer than MLLM box injection.

## Recommended hidden-test order

1. Submit `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m13_8rev_zofficial_balanced.zip` if you want the strongest current probe.
2. Submit `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m13_8rev_safe.zip` if you want to isolate the new MLLM reverse-anchor contribution.
3. Keep the older M13 endpoint-only zips only as ablations.

## Next implementation if score remains flat

Do not sweep thresholds. Implement an external image-prompt tracking candidate source (SUTrack/T-Rex2/DINO tracking-style bbox proposals) and feed those bboxes through the existing M13 reverse-injection + object fusion pipeline. This is the closest remaining gap to TEP and should address tiny/semantic cases where Qwen understands the event but cannot provide a precise enough bbox.

## Post-review hardening

A code-review pass found one blocking risk: the teach-box runner could create an incomplete zip if `MAKE_SUBMISSION=1` was used with a subset `VIDEOS`. This is now fixed:

- `tools/infer_mosev2_sam2_teach_boxes.py` fills missing 15-video predictions from the baseline root before zipping and validates the final zip fail-closed.
- `tools/apply_m13_object_fusion.py` validates generated zips fail-closed.
- `scripts/run_b101_m13_teach_boxes.sh` now defaults M13 probes to `--propagate-direction both` with a bounded window, matching the reverse-anchor conclusion.
- A local temp subset-submission test passed after automatically filling missing predicted videos: `provided_changed_count=0`, `predicted_error_count=0`, `ok=true`.

## Hidden feedback update: zofficial-balanced = 43.34

The latest full-test log for `submission_mosev2_m13_8rev_zofficial_balanced.zip` reports `J&F_new=43.34`, above the prior 43.30 family. Row-level inspection shows the improvement is concentrated in `z6dx46qr`:

- `z6dx46qr`: `40.47 -> 65.14`, explaining about `+0.043` global score over 575 object rows.
- `8jsm23a7`: remains `2.13`, so the visually better reverse-anchor did not translate to hidden-metric gain.
- `q0sizv6m`: unchanged, as expected because the unsafe reverse-anchor was excluded.

Interpretation: this is a valid incremental improvement and confirms the fusion infrastructure is useful, but it is not a structural MLLM breakthrough. The next high-value step is to understand why `8jsm23a7` remains at 2.13 despite the plausible visual event fix, then pursue stronger bbox/identity sources rather than more Qwen box geometry alone.

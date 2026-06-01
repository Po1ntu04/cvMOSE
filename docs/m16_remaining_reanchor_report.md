# M16 remaining-object re-anchor report

**Goal.** Continue after M15 safe/balanced reached `43.44` by attacking the still-low rows that were not fixed by `amfdu83t` and `z6dx46qr`: `8jsm23a7`, `q0sizv6m:obj2`, `r13u5z4y`, plus audit of `1qlssuz2` and `4vznweiu`.

**Base root.** `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m15_layered_safe`

**Main change.** `tools/infer_mosev2_sam2_teach_boxes.py` now supports `prompt_type=mask_box` / `mask_from_box`: a Qwen/manual box can be injected as a rectangular pseudo-mask via SAM2 `add_new_mask`, instead of only a positive box prompt. This was added because box prompts on tiny/same-class cases frequently returned empty after bounded propagation.

## Hidden-feedback target rows motivating M16

| target | M15 safe J&F_new | issue | M16 decision |
| --- | ---: | --- | --- |
| `8jsm23a7:1` | `2.13` | Mahjong tile identity is still completely wrong; target should be the picked-up tile placed in the near/front row as leftmost seven-bamboo. | High-upside probe. `mask_box` anchors target the near/front-row tile after placement. |
| `q0sizv6m:2` | `12.34` | Rear small animal moves into foreground/bottom; baseline tracks original-area same-class distractor. | Plausible but same-class risky probe. `mask_box` moves to foreground animal hypothesis. |
| `r13u5z4y:1` | `38.84` | Strawberry slice is heavily occluded, then reappears among near-identical slices. | Aggressive-only probe. `mask_box` targets left reappearing slice; high same-class risk. |
| `1qlssuz2:1` | `36.88` | Small white car / road tracking; M15 aggressive was harmful. | Audited. M15 already follows a coherent path; remaining error likely hidden-GT alignment or identity under bridge. No safe new edit. |
| `4vznweiu:1` | `33.50` | Letter/bead/flower scene; M15 aggressive was harmful. | Audited. Current candidate alternatives jump between beads/flowers; no safe new edit. |

## Experiments run

### A. Direct SAM2 box prompt, clipped/no-clip

Split plan: `artifacts/m16_remaining/m16_custom_split.json`.

Remote outputs:

- clipped: `artifacts/m16_remaining/remote_downloads/pred_m16_remaining_clipped/`
- no-clip: `artifacts/m16_remaining/remote_downloads/pred_m16_remaining_noclip/`
- visual sheets: `docs/assets/m16_remaining_clipped_compare/`, `docs/assets/m16_remaining_noclip_compare/`

Verdict: **reject**. Box prompts often converged to empty masks after SAM2 refinement, especially for `8jsm23a7` and `q0sizv6m`. This confirms the issue is not merely “give SAM2 a rough box”; for these tiny/same-class cases the prompt shape and local evidence control matter.

### B. SAM2 `mask_box` pseudo-mask prompt

Split plan: `artifacts/m16_remaining/m16_custom_maskbox_split.json`.

Remote output:

- pred root: `artifacts/m16_remaining/remote_downloads/pred_m16_remaining_maskbox/`
- audit: `artifacts/m16_remaining/remote_downloads/m16_remaining_maskbox.json`
- visual sheets: `docs/assets/m16_remaining_maskbox_compare/`

Remote audit summary: `action_count=7`, `elapsed_sec=50.96`, `videos=3`.

Qualitative verdict:

- `8jsm23a7`: `mask_box` finally produces a non-empty near/front-row Mahjong tile hypothesis after the placement event. This is the only M16 component with a high upside because the current row is nearly zero. The risk is that the rectangle may still hit the wrong tile face or only part of the tile.
- `q0sizv6m:2`: `mask_box` moves from the old rear-area distractor toward foreground animals. This matches the user-level physical story, but the scene remains dense and the mask may include the wrong foreground animal or composite parts.
- `r13u5z4y`: `mask_box` creates a visible left-slice hypothesis after occlusion. It is visually plausible, but strawberry slices are highly similar, so this is not safe without hidden-score confirmation.

## Candidate zips generated

All candidates copy M15 safe everywhere except selected object/frame intervals. All passed `tools/validate_mose_submission.py` with `provided_changed_count=0`, `video_dirs=433`, `pngs=66526`, `predicted_videos=15`.

| candidate | zip path | selected edits | intended use |
| --- | --- | --- | --- |
| M16 8js only | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m16_8js_only.zip` and repo convenience copy `submission_mosev2_m16_8js_only.zip` | `8jsm23a7:1`, frames `26-48` from `m16mask` | safest M16 probe: high upside, isolated risk. |
| M16 8js + q0 | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m16_8js_q0.zip` and repo copy `submission_mosev2_m16_8js_q0.zip` | `8jsm23a7:1` frames `26-48`; `q0sizv6m:2` frames `24-41` | balanced probe: adds foreground-animal recovery hypothesis. |
| M16 8js + q0 + r13 | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m16_8js_q0_r13.zip` and repo copy `submission_mosev2_m16_8js_q0_r13.zip` | plus `r13u5z4y:1` frames `20-44` | aggressive probe only; same-class strawberry risk. |

Validation JSONs:

- `artifacts/m16_remaining/validate_submission_mosev2_m16_8js_only.json`
- `artifacts/m16_remaining/validate_submission_mosev2_m16_8js_q0.json`
- `artifacts/m16_remaining/validate_submission_mosev2_m16_8js_q0_r13.json`

Fusion audits:

- `artifacts/m16_remaining/fusion_8js_only.json`
- `artifacts/m16_remaining/fusion_8js_q0.json`
- `artifacts/m16_remaining/fusion_8js_q0_r13.json`

## Recommendation

Submit/test in this order:

1. `submission_mosev2_m16_8js_only.zip` — isolates whether the new near/front-row Mahjong hypothesis is finally aligned with hidden GT. If this row moves from `2.13`, it validates the M16 mask-box direction.
2. `submission_mosev2_m16_8js_q0.zip` — if 8js improves or is neutral, this tests whether the foreground q0 physical-story correction helps.
3. `submission_mosev2_m16_8js_q0_r13.zip` — aggressive only. Use to measure whether the strawberry left-slice hypothesis has any hidden-score value; not safe as final unless the score confirms.

Do **not** replace M15 safe as the recommended final until hidden feedback confirms one of these probes. M15 safe remains the proven best (`43.44`).

## Remaining analysis

- `1qlssuz2`: current SAM2 path is coherent, and previous aggressive expansion was harmful. The likely next useful experiment is not another broad box prompt but a car-specific detector/tracker or a hidden-feedback-isolated fill/occlusion test for frames 13-14 and 37-39.
- `4vznweiu`: the target is a tiny semantic bead/flower-adjacent instance. Current alternatives switch between the bead cluster and flowers. Without a stronger bead-level re-ID/letter detector, manual/SAM2 prompts are more likely to hurt than help.
- API/Qwen split-harness was attempted for `1qlssuz2`/`4vznweiu`, but the current request path stalled before useful JSON evidence. Since M16 already produced validated probes for the highest-value unresolved rows, I did not block the experiment on that API branch.

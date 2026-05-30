# M7 Qwen-VL fusion report

**One-line verdict:** Use M7 as an architectural verifier/safety layer; the only currently recommended positive replacement is `1qlssuz2` from Qwen-supported M5R repropagation, while same-class dense cases stay on M11/veto.

## Inputs

| item | path / value |
| --- | --- |
| model | `qwen3.5-plus` through Bailian OpenAI-compatible API |
| default prediction root | `artifacts/m5r_reanchor/source_preds/m11` |
| Qwen support root | `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_m7_qwen_support_1ql_real` |
| Qwen veto root | `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_sam2_m7_qwen_veto_real_key4` |
| real candidate summary | `docs/m7_candidate_judge_real_merged_summary.md` |
| visual sheets | `docs/assets/m7_qwen_vl/real_key4_compare/` |

## Real Qwen evidence used

Merged real candidate judgments covered 24 high-risk records:

| video | records | support | veto / uncertain reading |
| --- | ---: | ---: | --- |
| `1qlssuz2` | 6 | 3 | Qwen supported green/white car candidates when the target vehicle was clear; it vetoed the white-car distractor at frame 9. |
| `q0sizv6m` | 8 | 0 | Qwen repeatedly refused dense same-class guinea-pig re-ID and vetoed wrong-neighbor anchors. |
| `msinig6m` | 5 | 0 | Qwen stayed uncertain/veto on person/koala composite risk. |
| `r13u5z4y` | 5 | 0 | Qwen stayed conservative on the strawberry reappearance case and did not certify a correct recovery. |

The strict `veto_only + require_tracklet` b101 run over `r13u5z4y/q0sizv6m/msinig6m/1qlssuz2` accepted zero anchors and changed zero frames.  This proves safety/no pollution, but not recovery.

The targeted `support_and_veto` b101 run for `1qlssuz2` accepted 2 anchors and changed 33 frames versus M11/SAM2 within the bounded repropagation window.

## Visual decision

| video | observation | decision |
| --- | --- | --- |
| `1qlssuz2` | Zoom sheet shows Qwen-support masks remain on the target car; changes are mostly small contour/coverage refinements, including the right-edge frame. No visible drift into lane markings, pole, bridge, or adjacent cars. | **Use in balanced fusion** as low-risk small improvement. |
| `r13u5z4y` | Veto path is effectively unchanged/safe. Support deltas occur on ambiguous strawberry edges or adjacent slice regions; no reliable same-instance recovery after occlusion. | Keep M11/veto; do **not** promote support. |
| `q0sizv6m` | Veto path is unchanged/safe. Support/delta regions in the dense guinea-pig scene are plausible but identity-unsafe. | Keep M11/veto; do **not** promote support. |
| `msinig6m` | Veto path is unchanged/safe. Support deltas risk person/koala composite or other koala regions. | Keep M11/veto; do **not** promote support. |

Visual evidence:

- `docs/assets/m7_qwen_vl/real_key4_compare/1qlssuz2_m7_qwen_compare.jpg`
- `docs/assets/m7_qwen_vl/real_key4_compare/1qlssuz2_zoom.jpg`
- `docs/assets/m7_qwen_vl/real_key4_compare/r13u5z4y_m7_qwen_compare.jpg`
- `docs/assets/m7_qwen_vl/real_key4_compare/q0sizv6m_m7_qwen_compare.jpg`
- `docs/assets/m7_qwen_vl/real_key4_compare/msinig6m_m7_qwen_compare.jpg`

## Candidate zips

### Safe

- zip: `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m7_qwen_safe_fusion.zip`
- pred root: `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m7_qwen_safe_fusion`
- policy: M11 default, no positive replacement.
- validation: pass; 433 video dirs, 66526 PNGs, 418 provided outputs unchanged.
- recommendation: equivalent to M11 safety; not expected to improve score.

### Balanced

- zip: `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m7_qwen_balanced_fusion.zip`
- pred root: `/home/yu/projects/cv/from fdu/MOSEv2/homework/pred_m7_qwen_balanced_fusion`
- policy: M11 default + replace `1qlssuz2` with Qwen-supported M5R repropagation.
- changed frames: `1qlssuz2` changes 33 frames versus M11; other 14 target videos remain default/M11.
- validation: pass; 433 video dirs, 66526 PNGs, 418 provided outputs unchanged.
- recommendation: submit only as a low-risk probe after M11; expected gain is small because the visible improvement is contour-level, not a recovered hard identity case.

## What M7 proves and does not prove

- Proves: MLLM verification can reduce anchor pollution by refusing same-class/composite anchors; the integration path is safe and cached; Qwen support can be fused conservatively when visual evidence is clean.
- Does not prove: Qwen can recover `r13u5z4y/q0sizv6m/msinig6m` by itself. It is a verifier, not a high-recall candidate generator.

## Next architectural step

M7 should remain in the pipeline as a high-risk verifier. To move toward 44, the next useful step is not more Qwen prompting but stronger candidate generation before Qwen: object retrieval / detector proposals / reverse propagation anchors that actually contain the correct reappearing object. Qwen can then veto or support those candidates.

## Additional tiny/semantic verifier pass

After the first balanced fusion, I ran a focused real Qwen pass on `4vznweiu`, `lcgc29va`, `8jsm23a7`, and `2smf7uq9` with 16 calls (`docs/m7_candidate_judge_real_tiny_semantic_summary.md`).  Qwen produced many semantic/tiny support judgments, especially for `4vznweiu` and `lcgc29va`, but M5R's descriptor/reprop gate accepted only one anchor:

| video | Qwen support signal | M5R accepted anchors | changed frames | visual verdict | fusion decision |
| --- | --- | ---: | ---: | --- | --- |
| `4vznweiu` | strong support for the green letter/tile candidate in several frames | 0 | 0 | no output change; Qwen alone was not allowed to promote | no replacement |
| `lcgc29va` | support mixed with uncertainty/veto for tiny backpack-like target | 0 | 0 | no output change; candidate evidence too weak for M5R | no replacement |
| `8jsm23a7` | one supported tile candidate, one uncertain frame | 1 | 19 | changes occur in an already stable guarded video; no clear visual improvement over baseline/M11 | do **not** include in balanced/safe |

This reinforces the main M7 conclusion: Qwen can recognize semantic cues, but without a stronger high-recall candidate/retrieval path it mostly acts as a verifier.  It should not override M5R/SAM2 evidence.

Additional visual sheets:

- `docs/assets/m7_qwen_vl/tiny_semantic_compare/4vznweiu_m7_qwen_compare.jpg`
- `docs/assets/m7_qwen_vl/tiny_semantic_compare/lcgc29va_m7_qwen_compare.jpg`
- `docs/assets/m7_qwen_vl/tiny_semantic_compare/8jsm23a7_m7_qwen_compare.jpg`

## Overlay-panel fix and guarded source-select update

A follow-up review found that filled green/red MLLM overlays leaked artificial annotation color into Qwen identity reasoning.  I changed MLLM panels to raw-crop + thin artificial outline and added explicit prompt warnings.  This fixed the clearest bad support case: `4vznweiu` frame 16 changed from supporting a wrong same-class `T/O` die to `uncertain` + veto.

The same pass introduced `tools/apply_m7_qwen_source_select.py`, an output-level ablation that can only copy an existing source mask when Qwen explicitly supports it.  A conservative non-empty area-ratio guard prevents replacing a valid non-empty M11 mask with a substantially smaller/larger source.

Validated new probes:

| candidate | zip | policy | validation | recommendation |
| --- | --- | --- | --- | --- |
| M7 source-select guarded | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m7_qwen_source_select_outline_guarded_v2.zip` | M11 + `lcgc29va` frames 8-9 from `rar_state` only | pass; 418 provided unchanged | diagnostic/low-risk, very small expected effect |
| M7 balanced-plus outline | `/home/yu/projects/cv/from fdu/MOSEv2/homework/submission_mosev2_m7_qwen_balanced_plus_outline.zip` | M11 + prior `1qlssuz2` Qwen-supported reprop + guarded `lcgc29va` source-select | pass; 418 provided unchanged | best current M7 probe after M11/M6 balanced |

Detailed report: `docs/m7_overlay_source_select_report.md`.

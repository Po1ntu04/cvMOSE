# M18 same-class candidate/distractor atlas report

This split harness builds per-frame same-class candidate/negative banks with Qwen-VL and aggregates an M18 target/distractor memory plan. It is diagnostic and must be validated by descriptor checks, SAM2 bounded propagation, and visual review before any submission use.

- model: `qwen3.6-plus`
- vision_model: `qwen-vl-plus`
- aggregate_model: `qwen3.6-plus`
- dry_run: `False`
- targets: `4`

| video | obj | frames | aggregate status | action | final allowable | conf | targets | distractors | negatives |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `8jsm23a7` | 1 | 0,1,2,20,26,36,48 | manual_review | manual_review | False | 0.65 | 7 | 7 | 7 |
| `r13u5z4y` | 1 | 0,1,2,7,20,34,44 | manual_review | manual_review | False | 0.00 | 0 | 3 | 16 |
| `4vznweiu` | 1 | 0,1,2,10,20,34 | uncertain | manual_review | False | 0.65 | 6 | 5 | 11 |
| `1qlssuz2` | 1 | 0,1,2,13,14,37,39 | manual_review | manual_review | False | 0.75 | 7 | 7 | 7 |

## Use constraints

- Qwen boxes are not ground truth; boxes must be fed through SAM2 and visually audited.
- Same-class scenes require hard-negative rejection and delayed commit; a single MLLM support signal must not promote an anchor alone.
- For final submission, only replace frames/videos with clear visual evidence and preserve 418 provided outputs byte-identical.

## Gate B result interpretation

MLLM routing used fast `qwen-vl-plus` for per-frame visual captions and `qwen3.6-plus` for text-only aggregation. This avoided the repeated qwen3.6 vision timeout observed in the first Gate-B smoke while keeping qwen3.6 for event-chain reasoning.

| object | MLLM action | conf | ledger additions | retrieval effect | verdict |
| --- | --- | ---: | --- | --- | --- |
| `8jsm23a7:obj1` | manual_review | 0.65 | +8 candidates / +28 distractors | official_large:object_id output_only margin=0.12145; bank +1/-29 | Atlas useful as hard-negative memory; no safe promotion yet because official/current candidates remain below same-class margin. |
| `r13u5z4y:obj1` | manual_review | 0.00 | +0 candidates / +46 distractors | m16mask:anyfg:1 needs_more_evidence margin=0.19784; bank +1/-50 | Confirms no reliable strawberry reanchor; adds many hard negatives and preserves empty-safe policy. |
| `4vznweiu:obj1` | manual_review | 0.65 | +7 candidates / +33 distractors | official_large:object_id output_only margin=0.19241; bank +1/-34 | Finds letter/cube hypotheses but retrieval rejects/keeps output-only due strong hard-negative similarity; needs higher-res detector proof. |
| `1qlssuz2:obj1` | manual_review | 0.75 | +16 candidates / +27 distractors | ledger_anchor:m18_atlas_target_f39_E1 reject margin=-0.02425; bank +1/-28 | MLLM proposed car anchors, but retrieval rejects late atlas boxes as closer to negatives; no broad replacement. |

### Gate B safety conclusions

- `8jsm23a7`: do **not** promote the MLLM leftmost-tile boxes yet. The atlas is valuable mainly because it names old-position/static tile hard negatives and flags frame-36 spatial jump risk.
- `r13u5z4y`: MLLM explicitly failed to identify a positive strawberry reanchor; this is a useful negative result. Keep M11/M15 empty-safe unless a stronger detector provides source-independent evidence.
- `4vznweiu`: Qwen sees a small T/letter/cube track, but retrieval says many boxes are close to hard negatives. Treat as review material, not final.
- `1qlssuz2`: atlas boxes are rejected by the current positive-vs-negative descriptor gate; do not revive previous aggressive broad replacement.

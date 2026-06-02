# M13 Round 1 split-harness proxy report

- split_json: `artifacts/m13_split_all15/all15_split_q36.json`
- records: `20`
- aggregate_success_rate: `25.00%`

## Ranked action table

| priority | target | event | diagnosis | action | conf | proxy | key reasons |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| P0 | `8jsm23a7:1` | picked_moved_placed | uncertain | reanchor_at_frame | 0.65 | 14 | manual_P0, high_leverage_video, event:picked_moved_placed, diagnosis:uncertain |
| P0 | `z6dx46qr:1` | low_contrast_continuation | uncertain | reanchor_at_frame | 0.75 | 14 | manual_P0, high_leverage_video, event:low_contrast_continuation, diagnosis:uncertain |
| P0 | `q0sizv6m:2` | moves_to_foreground | uncertain | reanchor_at_frame | 0.65 | 13 | manual_P0, high_leverage_video, event:moves_to_foreground, diagnosis:uncertain |
| P0 | `amfdu83t:1` | unknown | uncertain | manual_review | 0.65 | 9 | manual_P0, high_leverage_video, diagnosis:uncertain |
| P1 | `1qlssuz2:1` | camera_motion | wrong_same_class | reanchor_at_frame | 0.90 | 8 | high_leverage_video, diagnosis:wrong_same_class, action:reanchor_at_frame, high_mllm_conf |
| P1 | `4vznweiu:1` | static | uncertain | reanchor_at_frame | 0.60 | 7 | high_leverage_video, diagnosis:uncertain, action:reanchor_at_frame, source_disagrees:official_l |
| P1 | `jadgtmfl:1` | static | uncertain | reanchor_at_frame | 0.80 | 6 | diagnosis:uncertain, action:reanchor_at_frame, high_mllm_conf, source_disagrees:official_b |
| P2 | `2smf7uq9:1` | static | uncertain | generate_detector_candidates | 0.30 | 5 | diagnosis:uncertain, action:generate_detector_candidates, source_disagrees:official_b |
| P2 | `3epdtmyr:1` | unknown | wrong_same_class | reanchor_at_frame | 0.85 | 5 | diagnosis:wrong_same_class, action:reanchor_at_frame, high_mllm_conf |
| P2 | `4f98052b:2` | exits_reappears | uncertain | manual_review | 0.65 | 5 | event:exits_reappears, diagnosis:uncertain, source_disagrees:official_l |
| P2 | `msinig6m:3` | multi_object_contact | uncertain | generate_detector_candidates | 0.35 | 5 | diagnosis:uncertain, action:generate_detector_candidates, source_disagrees:official_l |
| P2 | `q0sizv6m:1` | unknown | uncertain | manual_review | 0.45 | 5 | high_leverage_video, diagnosis:uncertain, source_disagrees:official_l |
| P2 | `pe0d85lk:2` | exits_reappears | wrong_same_class | manual_review | 0.75 | 4 | event:exits_reappears, diagnosis:wrong_same_class |
| P2 | `c8lutf29:1` | unknown | uncertain | manual_review | 0.40 | 3 | diagnosis:uncertain, source_disagrees:official_l |
| P2 | `msinig6m:1` | static | uncertain | manual_review | 0.30 | 3 | diagnosis:uncertain, source_disagrees:official_l |
| P2 | `msinig6m:2` | static | uncertain | manual_review | 0.60 | 3 | diagnosis:uncertain, source_disagrees:official_l |
| P2 | `r13u5z4y:1` | multi_object_contact | mostly_correct | keep_current | 0.85 | 3 | high_leverage_video, high_mllm_conf |
| drop | `4f98052b:1` | unknown | uncertain | manual_review | 0.35 | 2 | diagnosis:uncertain |
| drop | `lcgc29va:1` | unknown | mostly_correct | keep_current | 0.88 | 2 | high_mllm_conf, source_disagrees:official_l |
| drop | `pe0d85lk:1` | unknown | uncertain | manual_review | 0.70 | 2 | diagnosis:uncertain |

## Interpretation

- P0 means event-chain intervention is likely worth a bounded SAM/proxy experiment.
- This report is not hidden-GT evaluation; it is a local routing/audit layer for deciding which probes deserve zip candidates.

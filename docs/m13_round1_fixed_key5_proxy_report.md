# M13 Round 1 split-harness proxy report

- split_json: `artifacts/m13_split_all15/fixed_key5_split_q36.json`
- records: `5`
- aggregate_success_rate: `20.00%`

## Ranked action table

| priority | target | event | diagnosis | action | conf | proxy | key reasons |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| P0 | `8jsm23a7:1` | picked_moved_placed | uncertain | reanchor_at_frame | 0.65 | 14 | manual_P0, high_leverage_video, event:picked_moved_placed, diagnosis:uncertain |
| P0 | `q0sizv6m:2` | moves_to_foreground | wrong_same_class | reanchor_at_frame | 0.75 | 13 | manual_P0, high_leverage_video, event:moves_to_foreground, diagnosis:wrong_same_class |
| P0 | `z6dx46qr:1` | low_contrast_continuation | uncertain | manual_review | 0.65 | 12 | manual_P0, high_leverage_video, event:low_contrast_continuation, diagnosis:uncertain |
| P0 | `amfdu83t:1` | exits_reappears | uncertain | manual_review | 0.65 | 11 | manual_P0, high_leverage_video, event:exits_reappears, diagnosis:uncertain |
| P2 | `1qlssuz2:1` | unknown | uncertain | manual_review | 0.65 | 5 | high_leverage_video, diagnosis:uncertain, source_disagrees:official_l |

## Interpretation

- P0 means event-chain intervention is likely worth a bounded SAM/proxy experiment.
- This report is not hidden-GT evaluation; it is a local routing/audit layer for deciding which probes deserve zip candidates.

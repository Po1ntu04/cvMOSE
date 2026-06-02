# M13 qwen3.6 split-frame event-story report

**Purpose:** avoid qwen3.6-plus multi-frame panel timeouts by splitting visual evidence into small frame observations, then aggregating text-only.

- model: `qwen3.6-plus`
- allow_fallback: `False`
- dry_run: `False`
- targets: `20`

| video | obj | frame calls ok/api_error | aggregate status | event | diagnosis | action | conf |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| `1qlssuz2` | 1 | 4/0 | ok | camera_motion | wrong_same_class | reanchor_at_frame | 0.90 |
| `q0sizv6m` | 1 | 4/0 | uncertain | unknown | uncertain | manual_review | 0.45 |
| `q0sizv6m` | 2 | 4/0 | uncertain | moves_to_foreground | uncertain | reanchor_at_frame | 0.65 |
| `lcgc29va` | 1 | 4/0 | ok | unknown | mostly_correct | keep_current | 0.88 |
| `z6dx46qr` | 1 | 4/0 | uncertain | low_contrast_continuation | uncertain | reanchor_at_frame | 0.75 |
| `4vznweiu` | 1 | 4/0 | uncertain | static | uncertain | reanchor_at_frame | 0.60 |
| `amfdu83t` | 1 | 4/0 | uncertain | unknown | uncertain | manual_review | 0.65 |
| `4f98052b` | 1 | 4/0 | uncertain | unknown | uncertain | manual_review | 0.35 |
| `4f98052b` | 2 | 4/0 | uncertain | exits_reappears | uncertain | manual_review | 0.65 |
| `3epdtmyr` | 1 | 4/0 | ok | unknown | wrong_same_class | reanchor_at_frame | 0.85 |
| `msinig6m` | 1 | 4/0 | insufficient_visual_evidence | static | uncertain | manual_review | 0.30 |
| `msinig6m` | 2 | 4/0 | uncertain | static | uncertain | manual_review | 0.60 |
| `msinig6m` | 3 | 4/0 | uncertain | multi_object_contact | uncertain | generate_detector_candidates | 0.35 |
| `c8lutf29` | 1 | 4/0 | uncertain | unknown | uncertain | manual_review | 0.40 |
| `2smf7uq9` | 1 | 4/0 | uncertain | static | uncertain | generate_detector_candidates | 0.30 |
| `8jsm23a7` | 1 | 4/0 | uncertain | picked_moved_placed | uncertain | reanchor_at_frame | 0.65 |
| `jadgtmfl` | 1 | 4/0 | ok | static | uncertain | reanchor_at_frame | 0.80 |
| `r13u5z4y` | 1 | 4/0 | ok | multi_object_contact | mostly_correct | keep_current | 0.85 |
| `pe0d85lk` | 1 | 4/0 | uncertain | unknown | uncertain | manual_review | 0.70 |
| `pe0d85lk` | 2 | 4/0 | uncertain | exits_reappears | wrong_same_class | manual_review | 0.75 |

## Contract

- Frame observations are short visual captions, not final masks.
- Aggregation is text-only qwen3.6 reasoning over frame JSON.
- Positive boxes still need SAM refinement and bounded propagation before any submission use.

# M13 qwen3.6 split-frame event-story report

**Purpose:** avoid qwen3.6-plus multi-frame panel timeouts by splitting visual evidence into small frame observations, then aggregating text-only.

- model: `qwen3.6-plus`
- allow_fallback: `False`
- dry_run: `False`
- targets: `5`

| video | obj | frame calls ok/api_error | aggregate status | event | diagnosis | action | conf |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| `1qlssuz2` | 1 | 4/0 | uncertain | unknown | uncertain | manual_review | 0.65 |
| `q0sizv6m` | 2 | 4/0 | uncertain | moves_to_foreground | wrong_same_class | reanchor_at_frame | 0.75 |
| `z6dx46qr` | 1 | 4/0 | ok | low_contrast_continuation | uncertain | manual_review | 0.65 |
| `amfdu83t` | 1 | 4/0 | uncertain | exits_reappears | uncertain | manual_review | 0.65 |
| `8jsm23a7` | 1 | 4/0 | uncertain | picked_moved_placed | uncertain | reanchor_at_frame | 0.65 |

## Contract

- Frame observations are short visual captions, not final masks.
- Aggregation is text-only qwen3.6 reasoning over frame JSON.
- Positive boxes still need SAM refinement and bounded propagation before any submission use.

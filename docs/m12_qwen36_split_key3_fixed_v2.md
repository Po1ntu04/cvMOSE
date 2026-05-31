# M12 qwen3.6 split-frame event-story report

**Purpose:** avoid qwen3.6-plus multi-frame panel timeouts by splitting visual evidence into small frame observations, then aggregating text-only.

- model: `qwen3.6-plus`
- allow_fallback: `False`
- dry_run: `False`
- targets: `3`

| video | obj | frame calls ok/api_error | aggregate status | event | diagnosis | action | conf |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| `q0sizv6m` | 2 | 4/0 | ok | moves_to_foreground | mostly_correct | reanchor_at_frame | 0.88 |
| `z6dx46qr` | 1 | 4/0 | uncertain | low_contrast_continuation | mostly_correct | reanchor_at_frame | 0.65 |
| `8jsm23a7` | 1 | 4/0 | ok | picked_moved_placed | mostly_correct | reanchor_at_frame | 0.85 |

## Contract

- Frame observations are short visual captions, not final masks.
- Aggregation is text-only qwen3.6 reasoning over frame JSON.
- Positive boxes still need SAM refinement and bounded propagation before any submission use.

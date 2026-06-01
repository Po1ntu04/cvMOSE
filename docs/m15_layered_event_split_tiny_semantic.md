# M13 qwen3.6 split-frame event-story report

**Purpose:** avoid qwen3.6-plus multi-frame panel timeouts by splitting visual evidence into small frame observations, then aggregating text-only.

- model: `qwen3.6-plus`
- allow_fallback: `False`
- dry_run: `False`
- targets: `4`

| video | obj | frame calls ok/api_error | aggregate status | event | diagnosis | action | conf |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| `4vznweiu` | 1 | 5/0 | uncertain | static | uncertain | generate_detector_candidates | 0.65 |
| `1qlssuz2` | 1 | 3/2 | uncertain | camera_motion | uncertain | reanchor_at_frame | 0.65 |
| `lcgc29va` | 1 | 5/0 | uncertain | unknown | uncertain | manual_review | 0.30 |
| `4f98052b` | 2 | 3/2 | uncertain | static | uncertain | reanchor_at_frame | 0.65 |

## Contract

- Frame observations are short visual captions, not final masks.
- Aggregation is text-only qwen3.6 reasoning over frame JSON.
- Positive boxes still need SAM refinement and bounded propagation before any submission use.

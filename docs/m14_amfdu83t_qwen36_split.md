# M14 amfdu83t qwen3.6 split-frame event-story report

**Purpose:** use the split harness on `amfdu83t:obj1` to test whether qwen3.6-plus understands the same-class kangaroo disappearance/reappearance event without relying on a dense multi-frame panel.

- model: `qwen3.6-plus`
- allow_fallback: `False`
- dry_run: `False`
- targets: `1`

| video | obj | frame calls ok/api_error | aggregate status | event | diagnosis | action | conf |
| --- | ---: | --- | --- | --- | --- | --- | ---: |
| `amfdu83t` | 1 | 7/0 | ok | exits_reappears | wrong_same_class | reanchor_at_frame | 0.65 |

## Contract

- Frame observations are short visual captions, not final masks.
- Aggregation is text-only qwen3.6 reasoning over frame JSON.
- Positive boxes still need SAM refinement and bounded propagation before any submission use.

## Key interpretation

Qwen3.6 captured the high-level `exits_reappears` event and diagnosed `wrong_same_class`, but its direct coordinate boxes were broad. They are therefore used as semantic/event evidence only; the submitted probe uses tighter manually audited SAM2 box anchors.

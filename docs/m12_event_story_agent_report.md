# M12 event-story MLLM agent report

**Purpose:** Use MLLM as a physical-instance event-chain analyst and SAM prompt teacher, not merely as an A/B mask verifier.

## Run metadata

- model: `qwen3.6-plus`
- dry_run: `False`
- total targets: `20`
- cache_dir: `artifacts/m12_event_story/cache`
- panel_dir: `artifacts/m12_event_story/all15_panels`

## Summary table

| video | obj | status | event_type | diagnosis | action | conf | panel |
| --- | ---: | --- | --- | --- | --- | ---: | --- |
| `1qlssuz2` | 1 | ok | moves_to_foreground | mostly_correct | reanchor_at_frame | 0.90 | `artifacts/m12_event_story/all15_panels/1qlssuz2/obj1_event_story.jpg` |
| `q0sizv6m` | 1 | ok | picked_moved_placed | wrong_static_distractor | reanchor_at_frame | 0.85 | `artifacts/m12_event_story/all15_panels/q0sizv6m/obj1_event_story.jpg` |
| `q0sizv6m` | 2 | ok | moves_to_foreground | wrong_same_class | reanchor_at_frame | 0.90 | `artifacts/m12_event_story/all15_panels/q0sizv6m/obj2_event_story.jpg` |
| `lcgc29va` | 1 | ok | picked_moved_placed | mostly_correct | keep_current | 0.90 | `artifacts/m12_event_story/all15_panels/lcgc29va/obj1_event_story.jpg` |
| `z6dx46qr` | 1 | uncertain | low_contrast_continuation | uncertain | reanchor_at_frame | 0.50 | `artifacts/m12_event_story/all15_panels/z6dx46qr/obj1_event_story.jpg` |
| `4vznweiu` | 1 | ok | picked_moved_placed | wrong_static_distractor | reanchor_at_frame | 0.85 | `artifacts/m12_event_story/all15_panels/4vznweiu/obj1_event_story.jpg` |
| `amfdu83t` | 1 | ok | picked_moved_placed | wrong_static_distractor | reanchor_at_frame | 0.90 | `artifacts/m12_event_story/all15_panels/amfdu83t/obj1_event_story.jpg` |
| `4f98052b` | 1 | ok | picked_moved_placed | wrong_static_distractor | reanchor_at_frame | 0.90 | `artifacts/m12_event_story/all15_panels/4f98052b/obj1_event_story.jpg` |
| `4f98052b` | 2 | ok | static | mostly_correct | keep_current | 0.95 | `artifacts/m12_event_story/all15_panels/4f98052b/obj2_event_story.jpg` |
| `3epdtmyr` | 1 | ok | moves_to_foreground | mostly_correct | reanchor_at_frame | 0.95 | `artifacts/m12_event_story/all15_panels/3epdtmyr/obj1_event_story.jpg` |
| `msinig6m` | 1 | ok | picked_moved_placed | wrong_same_class | reanchor_at_frame | 0.80 | `artifacts/m12_event_story/all15_panels/msinig6m/obj1_event_story.jpg` |
| `msinig6m` | 2 | ok | picked_moved_placed | wrong_static_distractor | reanchor_at_frame | 0.90 | `artifacts/m12_event_story/all15_panels/msinig6m/obj2_event_story.jpg` |
| `msinig6m` | 3 | ok | picked_moved_placed | mostly_correct | keep_current | 0.97 | `artifacts/m12_event_story/all15_panels/msinig6m/obj3_event_story.jpg` |
| `c8lutf29` | 1 | ok | exits_reappears | uncertain | reanchor_at_frame | 0.75 | `artifacts/m12_event_story/all15_panels/c8lutf29/obj1_event_story.jpg` |
| `2smf7uq9` | 1 | ok | moves_to_foreground | mostly_correct | reanchor_at_frame | 0.85 | `artifacts/m12_event_story/all15_panels/2smf7uq9/obj1_event_story.jpg` |
| `8jsm23a7` | 1 | ok | picked_moved_placed | wrong_static_distractor | reanchor_at_frame | 0.85 | `artifacts/m12_event_story/all15_panels/8jsm23a7/obj1_event_story.jpg` |
| `jadgtmfl` | 1 | ok | moves_to_foreground | mostly_correct | reanchor_at_frame | 0.95 | `artifacts/m12_event_story/all15_panels/jadgtmfl/obj1_event_story.jpg` |
| `r13u5z4y` | 1 | ok | picked_moved_placed | wrong_static_distractor | reanchor_at_frame | 0.90 | `artifacts/m12_event_story/all15_panels/r13u5z4y/obj1_event_story.jpg` |
| `pe0d85lk` | 1 | ok | exits_reappears | wrong_static_distractor | reanchor_at_frame | 0.80 | `artifacts/m12_event_story/all15_panels/pe0d85lk/obj1_event_story.jpg` |
| `pe0d85lk` | 2 | ok | exits_reappears | empty_when_visible | empty_after_event | 0.90 | `artifacts/m12_event_story/all15_panels/pe0d85lk/obj2_event_story.jpg` |

## Harness contract

- MLLM output is an event/story and prompt plan, not a final mask.
- Positive prompt plans must still be refined by SAM2/SAM3/official masks and bounded propagation.
- Hard negatives should be fed into DINO/SAM feature validation and future candidate filters.
- If event story says the target moved, candidates at the original position should become negative even if visually similar.

# M12 qwen3.6-plus multi-frame timeout solution

## Problem

The first event-story harness sent one dense multi-frame panel plus a long event-chain JSON schema to `qwen3.6-plus`. This often timed out in DashScope/OpenAI-compatible calls even though the same API key and model could answer simple image requests.

Observed cache pattern before the fix:

- Simple image smoke: `qwen3.6-plus` succeeds.
- Compact `8jsm23a7` panel: `qwen3.6-plus` succeeds.
- Many dense event-story panels: `qwen3.6-plus: APITimeoutError` and fallback to `qwen-vl-max` / `qwen-vl-plus`.

So the failure was not authentication or base URL. It was the coupling of image-heavy multi-frame evidence with long reasoning/output in one request.

## Implemented fix

New tool: `tools/mllm_event_story_split.py`.

It splits the task into three stages:

1. **Small frame caption calls**
   - one small compact panel per frame;
   - short qwen3.6 visual caption JSON;
   - short timeout (`--frame-timeout`, default 30s) so hard frames fail fast.
2. **Ultra-simple retry**
   - crop-only mini panel;
   - very short caption prompt;
   - longer retry timeout (`--retry-timeout`, default 90s).
3. **Text-only event aggregation**
   - qwen3.6 aggregates frame captions into event story / hard negatives / teach-SAM plan;
   - longer timeout (`--aggregate-timeout`, default 180s), no image payload.

Default mode is `--frame-task caption`, not rich frame reasoning. This intentionally keeps qwen3.6 out of the slow vision+long-reasoning path.

## Verification command

```bash
QWEN_VL_MODEL=qwen3.6-plus PYTHONUNBUFFERED=1 \
/home/yu/miniconda3/envs/cv-hw2/bin/python tools/mllm_event_story_split.py \
  --workspace "/home/yu/projects/cv/from fdu/MOSEv2" \
  --targets q0sizv6m:2 8jsm23a7:1 z6dx46qr:1 \
  --frames-json artifacts/m12_event_story_split/key_frames.json \
  --out-json artifacts/m12_event_story_split/key3_split_q36_fixed_v2.json \
  --out-panel-dir artifacts/m12_event_story_split/key3_panels_fixed_v2 \
  --out-doc docs/m12_qwen36_split_key3_fixed_v2.md \
  --cache-dir artifacts/m12_event_story_split/cache \
  --max-calls 3 --max-frames 4 --max-side 450 --jpeg-quality 55 \
  --frame-task caption --frame-timeout 30 --retry-timeout 90 --aggregate-timeout 180 --frame-retries 1
```

## Verification result

Output JSON: `artifacts/m12_event_story_split/key3_split_q36_fixed_v2.json`.
Report: `docs/m12_qwen36_split_key3_fixed_v2.md`.

| target | frame calls | frame model | aggregate model | aggregate result |
| --- | ---: | --- | --- | --- |
| `q0sizv6m:2` | 4/4 ok | `qwen3.6-plus` | `qwen3.6-plus` | `moves_to_foreground`, `reanchor_at_frame`, confidence 0.88 |
| `z6dx46qr:1` | 4/4 ok | `qwen3.6-plus` | `qwen3.6-plus` | `low_contrast_continuation`, `reanchor_at_frame`, confidence 0.65 |
| `8jsm23a7:1` | 4/4 ok | `qwen3.6-plus` | `qwen3.6-plus` | `picked_moved_placed`, `reanchor_at_frame`, confidence 0.85 |

This directly fixes the old failure mode: all tested visual frame calls and all aggregation calls used `qwen3.6-plus` without fallback.

## Practical guidance

- Use split harness for qwen3.6-plus when the task needs event reasoning.
- Do not send dense multi-frame panels to qwen3.6-plus unless only one or two frames are needed.
- Keep visual calls short and let qwen3.6 do deep reasoning in text-only aggregation.
- Failed/uncertain frame captions are acceptable as missing evidence; aggregate stage should remain conservative.

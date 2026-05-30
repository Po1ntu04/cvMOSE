# M7 Qwen-VL / Bailian setup

- base_url: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- requested model: `qwen3.5-plus`
- fallback models: `qwen-vl-max, qwen2.5-vl-72b-instruct, qwen-vl-plus`
- API key present: `True`
- dry_run: `False`
- elapsed_sec: `0.0`
- success: `True`
- model_used: `qwen3.5-plus`
- cache_key: `28efe0c79ca7e010d1c1cff348432b84e4ce67b2409e7da5d65c6da3828f29c2`

## Raw parsed sample

```json
{
  "image_contains": "text 'Qwen-VL JSON smoke test', a yellow rectangular border, and a green circle inside the rectangle",
  "status": "ok",
  "model_used": "qwen3.5-plus",
  "mllm_cache_key": "28efe0c79ca7e010d1c1cff348432b84e4ce67b2409e7da5d65c6da3828f29c2",
  "cache_hit": true
}
```

## Notes

The client uses the OpenAI-compatible Chat Completions path and reads `DASHSCOPE_API_KEY`; if the key is missing it generates conservative dry-run JSON so panel generation and downstream policies remain reproducible.

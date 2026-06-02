# M7 Qwen-VL / Bailian setup

- base_url: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- requested model: `qwen3.6-plus`
- fallback models: `qwen-vl-max, qwen2.5-vl-72b-instruct, qwen-vl-plus`
- API key present: `True`
- dry_run: `False`
- elapsed_sec: `4.57`
- success: `True`
- model_used: `qwen3.6-plus`
- cache_key: `ad3b3aa7b12300ff9693559e29d63a854161f02dd6fa1d0ab922694a2148da42`

## Raw parsed sample

```json
{
  "image_contains": "a yellow rectangle with a green circle inside and text 'Qwen-VL JSON smoke test' at the top",
  "status": "ok",
  "model_used": "qwen3.6-plus",
  "mllm_cache_key": "ad3b3aa7b12300ff9693559e29d63a854161f02dd6fa1d0ab922694a2148da42"
}
```

## Notes

The client uses the OpenAI-compatible Chat Completions path and reads `DASHSCOPE_API_KEY`; if the key is missing it generates conservative dry-run JSON so panel generation and downstream policies remain reproducible.

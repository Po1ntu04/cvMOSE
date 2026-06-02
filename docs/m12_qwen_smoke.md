# M7 Qwen-VL / Bailian setup

- base_url: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- requested model: `qwen3.6-plus`
- fallback models: `qwen3.5-plus, qwen-vl-max-latest, qwen-vl-max, qwen2.5-vl-72b-instruct, qwen-vl-plus`
- API key present: `True`
- dry_run: `False`
- elapsed_sec: `4.65`
- success: `True`
- model_used: `qwen3.6-plus`
- cache_key: `97a6f535dbe5f74687d1160d2c34ee370735cb64cbe1e6513e132b533d3a8adb`

## Raw parsed sample

```json
{
  "image_contains": "Text 'Qwen-VL JSON smoke test' at the top, a yellow rectangular border, and a green circle centered inside the rectangle on a white background",
  "status": "ok",
  "model_used": "qwen3.6-plus",
  "mllm_cache_key": "97a6f535dbe5f74687d1160d2c34ee370735cb64cbe1e6513e132b533d3a8adb"
}
```

## Notes

The client uses the OpenAI-compatible Chat Completions path and reads `DASHSCOPE_API_KEY`; if the key is missing it generates conservative dry-run JSON so panel generation and downstream policies remain reproducible.

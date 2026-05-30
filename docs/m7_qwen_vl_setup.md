# M7 Qwen-VL / Bailian setup

- base_url: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- requested model: `qwen-vl-max-latest`
- fallback models: `qwen-vl-max, qwen2.5-vl-72b-instruct, qwen-vl-plus`
- API key present: `True`
- dry_run: `True`
- elapsed_sec: `0.01`
- success: `True`
- model_used: `None`
- cache_key: `2622a19af6ca2486cb27f0d2662597a106e2d85eea373ef1ec8e59d9baed566f`

## Raw parsed sample

```json
{
  "status": "dry_run",
  "decision": "uncertain",
  "target_visible": "uncertain",
  "best_candidate": "uncertain",
  "should_support_anchor": false,
  "should_veto_anchor": true,
  "confidence": 0.0,
  "reason_short": "DASHSCOPE_API_KEY missing or dry_run enabled",
  "recommended_action": "uncertain",
  "risk_tags": [
    "dry_run"
  ],
  "video": null,
  "obj_id": null,
  "frame_idx": null,
  "mllm_cache_key": "2622a19af6ca2486cb27f0d2662597a106e2d85eea373ef1ec8e59d9baed566f",
  "cache_hit": true
}
```

## Notes

The client uses the OpenAI-compatible Chat Completions path and reads `DASHSCOPE_API_KEY`; if the key is missing it generates conservative dry-run JSON so panel generation and downstream policies remain reproducible.

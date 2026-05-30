#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cvmose.qwen_vl_client import DEFAULT_BASE_URL, DEFAULT_FALLBACKS, DEFAULT_MODEL, QwenVLClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test DashScope/Bailian Qwen-VL OpenAI-compatible JSON calls.")
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/m7_qwen_vl/cache"))
    p.add_argument("--out-doc", type=Path, default=Path("docs/m7_qwen_vl_setup.md"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def make_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (512, 320), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([90, 70, 420, 250], outline=(255, 180, 0), width=8)
    d.ellipse([210, 120, 300, 210], fill=(30, 180, 90))
    d.text((112, 25), "Qwen-VL JSON smoke test", fill=(0, 0, 0))
    img.save(path)


def main() -> None:
    args = parse_args()
    image_path = Path("artifacts/m7_qwen_vl/smoke/qwen_vl_test.jpg")
    make_test_image(image_path)
    base_url = os.getenv("BAILIAN_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL)
    fallbacks = [x.strip() for x in os.getenv("QWEN_VL_FALLBACK_MODELS", DEFAULT_FALLBACKS).split(",") if x.strip()]
    client = QwenVLClient(model=model, base_url=base_url, fallback_models=fallbacks, cache_dir=args.cache_dir, dry_run=args.dry_run)
    started = time.time()
    result = client.call_json(
        system_prompt="You are a vision JSON smoke tester. Return strict JSON only.",
        user_text='Describe the test image and return exactly JSON: {"image_contains":"...","status":"ok"}',
        image_paths=[image_path],
        schema_name="smoke_test",
        metadata={"tool": "test_qwen_vl_bailian"},
        max_tokens=512,
    )
    elapsed = round(time.time() - started, 2)
    ok = result.get("status") in {"ok", "dry_run"} and (result.get("image_contains") or result.get("decision") == "uncertain" or result.get("status") == "dry_run")
    args.out_doc.parent.mkdir(parents=True, exist_ok=True)
    args.out_doc.write_text(
        "# M7 Qwen-VL / Bailian setup\n\n"
        f"- base_url: `{base_url}`\n"
        f"- requested model: `{model}`\n"
        f"- fallback models: `{', '.join(fallbacks)}`\n"
        f"- API key present: `{bool(os.getenv('DASHSCOPE_API_KEY'))}`\n"
        f"- dry_run: `{client.cfg.dry_run}`\n"
        f"- elapsed_sec: `{elapsed}`\n"
        f"- success: `{ok}`\n"
        f"- model_used: `{result.get('model_used', result.get('model'))}`\n"
        f"- cache_key: `{result.get('mllm_cache_key')}`\n\n"
        "## Raw parsed sample\n\n"
        "```json\n" + json.dumps(result, ensure_ascii=False, indent=2)[:3000] + "\n```\n\n"
        "## Notes\n\n"
        "The client uses the OpenAI-compatible Chat Completions path and reads `DASHSCOPE_API_KEY`; if the key is missing it generates conservative dry-run JSON so panel generation and downstream policies remain reproducible.\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": ok, "elapsed": elapsed, "result": result, "doc": str(args.out_doc)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

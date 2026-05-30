"""Qwen-VL / DashScope OpenAI-compatible client with cache and dry-run fallback.

This module intentionally has no required network side effects at import time.
It is safe to use without ``DASHSCOPE_API_KEY``: calls return conservative
JSON stubs and still write cache records so downstream MLLM tooling remains
reproducible until a real API key is provided.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-max-latest"
DEFAULT_FALLBACKS = "qwen-vl-max,qwen2.5-vl-72b-instruct,qwen-vl-plus"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_json_object(text: str) -> tuple[dict[str, Any], str | None]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None
        return {
            "status": "parse_error",
            "decision": "uncertain",
            "should_veto_anchor": True,
            "raw_text": text[:4000],
            "parse_error": f"expected JSON object, got {type(parsed).__name__}",
        }, "parse_error"
    except Exception as first_error:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed, "extracted_first_object"
            except Exception:
                pass
        return {
            "status": "parse_error",
            "decision": "uncertain",
            "should_veto_anchor": True,
            "raw_text": text[:4000],
            "parse_error": str(first_error),
        }, "parse_error"


@dataclass(slots=True)
class QwenVLConfig:
    model: str = field(default_factory=lambda: os.getenv("QWEN_VL_MODEL", DEFAULT_MODEL))
    base_url: str = field(default_factory=lambda: os.getenv("BAILIAN_BASE_URL", DEFAULT_BASE_URL))
    api_key: str | None = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY"))
    fallback_models: list[str] = field(default_factory=lambda: [x.strip() for x in os.getenv("QWEN_VL_FALLBACK_MODELS", DEFAULT_FALLBACKS).split(",") if x.strip()])
    cache_dir: Path = Path("artifacts/m7_qwen_vl/cache")
    dry_run: bool = False
    max_side: int = 1600
    jpeg_quality: int = 90


class QwenVLClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        fallback_models: list[str] | None = None,
        cache_dir: str | Path | None = None,
        dry_run: bool = False,
        max_side: int = 1600,
        jpeg_quality: int = 90,
    ) -> None:
        cfg = QwenVLConfig()
        if model:
            cfg.model = model
        if base_url:
            cfg.base_url = base_url
        if api_key is not None:
            cfg.api_key = api_key
        if fallback_models is not None:
            cfg.fallback_models = fallback_models
        if cache_dir is not None:
            cfg.cache_dir = Path(cache_dir)
        cfg.dry_run = dry_run or not bool(cfg.api_key)
        cfg.max_side = int(max_side)
        cfg.jpeg_quality = int(jpeg_quality)
        self.cfg = cfg
        self.cache_dir = cfg.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def models_to_try(self) -> list[str]:
        out: list[str] = []
        for model in [self.cfg.model, *self.cfg.fallback_models]:
            if model and model not in out:
                out.append(model)
        return out

    def _normalized_image_bytes(self, path: Path) -> bytes:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if max(w, h) > self.cfg.max_side:
            scale = self.cfg.max_side / float(max(w, h))
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        # Add a thin border to avoid all-white panels disappearing in some viewers.
        if img.width > 8 and img.height > 8:
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, img.width - 1, img.height - 1], outline=(0, 0, 0), width=1)
        import io

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.cfg.jpeg_quality, optimize=True)
        return buf.getvalue()

    def encode_image_as_data_url(self, path: str | Path) -> tuple[str, str, int]:
        data = self._normalized_image_bytes(Path(path))
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"), sha256_bytes(data), len(data)

    def _cache_key(self, model: str, system_prompt: str, user_text: str, image_hashes: list[str], schema_name: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "schema_name": schema_name,
                "system_prompt": system_prompt,
                "user_text": user_text,
                "image_hashes": image_hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return sha256_bytes(payload)

    def _dry_stub(self, schema_name: str, metadata: dict[str, Any], reason: str) -> dict[str, Any]:
        base = {
            "status": "dry_run",
            "decision": "uncertain",
            "target_visible": "uncertain",
            "best_candidate": "uncertain",
            "should_support_anchor": False,
            "should_veto_anchor": True,
            "confidence": 0.0,
            "reason_short": reason,
            "recommended_action": "uncertain",
            "risk_tags": ["dry_run"],
        }
        if schema_name == "target_profile":
            base.update(
                {
                    "video": metadata.get("video"),
                    "obj_id": metadata.get("obj_id"),
                    "target_type": "unknown",
                    "is_tiny": bool(metadata.get("is_tiny", False)),
                    "is_edge_or_partial": bool(metadata.get("is_edge_or_partial", False)),
                    "is_semantic_dominated": False,
                    "has_same_class_distractors": bool(metadata.get("has_same_class_distractors", False)),
                    "semantic_description": "dry-run: not analyzed by Qwen-VL",
                    "identity_cues": [],
                    "likely_distractors": [],
                    "recommended_modules": ["m11_safety", "dino_descriptor", "veto_only"],
                    "anchor_policy": "unknown",
                    "notes": reason,
                }
            )
        elif schema_name == "tracklet_judge":
            base.update(
                {
                    "video": metadata.get("video"),
                    "obj_id": metadata.get("obj_id"),
                    "anchor_frame": metadata.get("anchor_frame"),
                    "same_object_across_frames": "uncertain",
                    "same_as_reference": "uncertain",
                    "promote_anchor": False,
                    "risk_tags": ["dry_run", "uncertain"],
                }
            )
        else:
            base.update({"video": metadata.get("video"), "obj_id": metadata.get("obj_id"), "frame_idx": metadata.get("frame_idx")})
        return base

    def call_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_paths: list[str | Path],
        schema_name: str,
        metadata: dict[str, Any] | None = None,
        max_tokens: int = 1536,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        encoded: list[tuple[str, str, int, str]] = []
        for path in image_paths[:6]:
            data_url, digest, size = self.encode_image_as_data_url(path)
            encoded.append((str(path), digest, size, data_url))
        image_hashes = [x[1] for x in encoded]
        model = self.cfg.model
        cache_key = self._cache_key(model, system_prompt, user_text, image_hashes, schema_name)
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.is_file():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            parsed = record.get("parsed", {})
            if isinstance(parsed, dict):
                parsed.setdefault("mllm_cache_key", cache_key)
                parsed.setdefault("cache_hit", True)
                return parsed

        if self.cfg.dry_run:
            parsed = self._dry_stub(schema_name, metadata, "DASHSCOPE_API_KEY missing or dry_run enabled")
            record = {
                "cache_key": cache_key,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "dry_run": True,
                "base_url": self.cfg.base_url,
                "model": model,
                "schema_name": schema_name,
                "metadata": metadata,
                "image_hashes": image_hashes,
                "image_sizes": [x[2] for x in encoded],
                "raw_response": None,
                "parsed": parsed,
            }
            cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            parsed["mllm_cache_key"] = cache_key
            return parsed

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            parsed = self._dry_stub(schema_name, metadata, f"openai package unavailable: {exc}")
            parsed["status"] = "client_error"
            parsed["mllm_cache_key"] = cache_key
            return parsed

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for _, _, _, data_url in encoded:
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        errors: list[str] = []
        raw_text = ""
        model_used = None
        for model_try in self.models_to_try:
            client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)
            try:
                kwargs: dict[str, Any] = {
                    "model": model_try,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                }
                try:
                    response = client.chat.completions.create(**kwargs, top_p=0.01)
                except TypeError:
                    response = client.chat.completions.create(**kwargs)
                raw_text = response.choices[0].message.content or ""
                model_used = model_try
                break
            except Exception as exc:
                errors.append(f"{model_try}: {type(exc).__name__}: {exc}")
        if model_used is None:
            parsed = self._dry_stub(schema_name, metadata, "all models failed")
            parsed.update({"status": "api_error", "errors": errors})
            raw_text = ""
        else:
            parsed, parse_status = parse_json_object(raw_text)
            parsed.setdefault("status", "ok" if parse_status is None else parse_status)
            parsed["model_used"] = model_used
            if errors:
                parsed["fallback_errors"] = errors
        record = {
            "cache_key": cache_key,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "dry_run": False,
            "base_url": self.cfg.base_url,
            "model": model_used or model,
            "requested_model": model,
            "schema_name": schema_name,
            "metadata": metadata,
            "image_hashes": image_hashes,
            "image_sizes": [x[2] for x in encoded],
            "raw_response": raw_text,
            "parsed": parsed,
        }
        cache_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        parsed["mllm_cache_key"] = cache_key
        return parsed

"""Google Gemini client adapter.

Implements the ``LLMClient`` protocol using the ``google-genai`` Python SDK so
that Gemini models can be used as evaluators alongside or instead of Anthropic
and OpenAI models.

Content block mapping
---------------------
TextBlock       → Part.from_text(text)
ImageBlock      → base64 source : Part.from_bytes(data, mime_type)
                  url source    : Part.from_uri(uri, mime_type)
                  (``media_type`` is used when present; defaults to
                  ``image/jpeg`` for URL sources without an explicit type)
DocumentBlock   → text source  : Part.from_text(data)
                  base64 source : Part.from_bytes(data, mime_type)
                  (PDF and other binary formats are sent as raw bytes;
                  Gemini 1.5+ natively supports PDF inputs)

Role mapping
------------
The Gemini API uses ``"user"`` and ``"model"`` roles.  The ``"assistant"``
role from LLMMessage is mapped to ``"model"`` automatically.

Install
-------
    pip install 'quality-framework[gemini]'
"""
from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .llm_client import ContentBlock, DocumentBlock, ImageBlock, LLMClientConfig, LLMMessage, ModelUsage, TextBlock, _TokenRateLimiter, _store_usage


# Gemini role names differ from OpenAI / Anthropic convention
_ROLE_MAP = {"assistant": "model", "user": "user", "model": "model"}


class GeminiClientAdapter:
    """Google Gemini SDK adapter satisfying the ``LLMClient`` protocol.

    Parameters
    ----------
    api_key:
        Gemini API key.  Falls back to the ``GEMINI_API_KEY`` environment
        variable, then ``GOOGLE_API_KEY``, if not provided.
    model:
        Default model to use (e.g. ``"gemini-2.5-pro"``, ``"gemini-3.1-pro"``).
        Overridden by ``LLMClientConfig.model`` at call time.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-pro",
    ) -> None:
        try:
            from google import genai as _genai
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required for GeminiClientAdapter. "
                "Install with: pip install 'quality-framework[gemini]'"
            ) from exc

        effective_api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        # Gemini uses a blocking (non-streaming) generate_content call whose entire
        # server-side processing time counts as a single "read".  Reasoning models can
        # take several minutes, so the ceiling is generous (600_000 ms = 10 min) — but
        # it MUST be finite: timeout=None lets a stalled connection hang forever (the
        # retry logic in complete() never fires because the read never raises), which
        # froze whole all-or-nothing batches for ~50 min. A finite ceiling lets a stall
        # raise -> retry. (local_patch, re-applied 2026-07-22 after the 07-20 reconcile
        # regressed it back to None.)
        _http_opts = _genai.types.HttpOptions(timeout=600_000)
        self._client = _genai.Client(api_key=effective_api_key, http_options=_http_opts)
        self._default_model = model
        self._limiter = _TokenRateLimiter(tokens_per_minute=2_000_000)

    def complete(self, messages: list[LLMMessage], config: LLMClientConfig) -> str:
        """Call the Gemini API and return the text response."""
        from .context_limiter import truncate_if_needed
        from google.genai import types as _types

        # Guard before truncate_if_needed: it calls get_context_limit(config.model)
        # → model.lower(), which raises an opaque AttributeError when model is None.
        if not config.model:
            raise ValueError(
                "LLMClientConfig.model is not set. The model must be specified in the "
                "spec's 'models:' front-matter — no default model is provided."
            )

        messages, _ = truncate_if_needed(messages, config)
        # TruncationInfo stored in thread-local by truncate_if_needed.

        model = config.model

        contents = []
        for msg in messages:
            role = _ROLE_MAP.get(msg.role, msg.role)
            if isinstance(msg.content, str):
                parts = [_types.Part.from_text(text=msg.content)]
            else:
                parts = [self._block_to_part(b) for b in msg.content]
            contents.append(_types.Content(role=role, parts=parts))

        _THINKING_LEVEL_MAP = {
            "low": "LOW",
            "medium": "MEDIUM",
            "high": "HIGH",
            "xhigh": "HIGH",
        }
        thinking_level_str = _THINKING_LEVEL_MAP.get((config.reasoning_effort or "").lower())

        gen_config = _types.GenerateContentConfig(
            max_output_tokens=config.max_tokens,
        )
        if thinking_level_str:
            gen_config.thinking_config = _types.ThinkingConfig(
                thinking_level=_types.ThinkingLevel(thinking_level_str)
            )
        else:
            gen_config.temperature = config.temperature
        if config.system:
            gen_config.system_instruction = config.system

        _input_chars = sum(
            sum(len(str(p)) for p in c.parts)
            for c in contents
        )
        # Reserve input + the requested output budget so the per-minute limit
        # reflects the full call, not just the prompt.
        _estimated_tokens = max(1, _input_chars // 4) + max(0, config.max_tokens or 0)
        self._limiter.acquire(_estimated_tokens)

        _MAX_ATTEMPTS = 2
        last_exc: Exception | None = None
        for _attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gen_config,
                )
                # Record token usage immediately after the API call.
                try:
                    _um = response.usage_metadata
                    _cached = getattr(_um, "cached_content_token_count", 0) or 0
                    _store_usage([ModelUsage.from_tokens(
                        model=model,
                        input_tokens=getattr(_um, "prompt_token_count", 0) - _cached,
                        output_tokens=getattr(_um, "candidates_token_count", 0),
                        cache_read_tokens=_cached,
                    )])
                except Exception:
                    _store_usage([])
                return response.text or ""
            except Exception as exc:
                import httpcore as _httpcore
                import httpx as _httpx
                if isinstance(exc, (_httpcore.RemoteProtocolError, _httpx.RemoteProtocolError)):
                    last_exc = exc
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Gemini request failed without a captured exception.")

    @staticmethod
    def _block_to_part(block: ContentBlock):
        from google.genai import types as _types

        if isinstance(block, TextBlock):
            return _types.Part.from_text(text=block.text)

        elif isinstance(block, ImageBlock):
            src = block.source
            if src.get("type") == "base64":
                data = base64.b64decode(src["data"])
                mt = src.get("media_type", "image/png")
                return _types.Part.from_bytes(data=data, mime_type=mt)
            elif src.get("type") == "url":
                mt = src.get("media_type", "image/jpeg")
                return _types.Part.from_uri(uri=src["url"], mime_type=mt)
            raise ValueError(f"Unsupported ImageBlock source type: {src.get('type')}")

        elif isinstance(block, DocumentBlock):
            src = block.source
            if src.get("type") == "text":
                return _types.Part.from_text(text=src.get("data", ""))
            elif src.get("type") == "base64":
                data = base64.b64decode(src["data"])
                mt = src.get("media_type", "application/pdf")
                return _types.Part.from_bytes(data=data, mime_type=mt)
            raise ValueError(f"Unsupported DocumentBlock source type: {src.get('type')}")

        raise TypeError(f"Unknown ContentBlock type: {type(block)}")

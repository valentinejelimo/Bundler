"""OpenAI client adapter.

Implements the ``LLMClient`` protocol using the OpenAI Python SDK so that
GPT models (including reasoning models with ``reasoning_effort``) can be used
as evaluators alongside or instead of Anthropic models.

Content block mapping
---------------------
TextBlock       → {"type": "text", "text": "..."}
ImageBlock      → {"type": "image_url", "image_url": {"url": "data:<mt>;base64,..."}}
                  or {"type": "image_url", "image_url": {"url": "<url>"}}
DocumentBlock   → text source  : inlined as {"type": "text", "text": "..."}
                  base64 source : {"type": "file", "file": {"file_data": "data:<mt>;base64,...",
                                   "filename": "document.<ext>"}}
                  (PDF file blocks require a model that supports the file input type,
                  e.g. gpt-4o and later.  Older models will raise an API error.)

Install
-------
    pip install 'quality-framework[openai]'
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .llm_client import ContentBlock, DocumentBlock, ImageBlock, LLMClientConfig, LLMMessage, ModelUsage, TextBlock, _TokenRateLimiter, _store_usage


_MEDIA_TYPE_TO_EXT: dict[str, str] = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/html": "html",
    "text/csv": "csv",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}


def build_request_body(
    messages: list[LLMMessage],
    config: LLMClientConfig,
    *,
    default_reasoning_effort: str | None = None,
) -> dict:
    """Build the OpenAI Chat Completions request body for one call.

    Shared by the synchronous ``OpenAIClientAdapter.complete()`` path and the
    batch JSONL builder (``openai_batch.py``), so a batched request is
    constructed identically to a live one — no drift between the two paths.
    """
    from .context_limiter import truncate_if_needed

    # Guard before truncate_if_needed: it calls get_context_limit(config.model)
    # → model.lower(), which raises an opaque AttributeError when model is None.
    if not config.model:
        raise ValueError(
            "LLMClientConfig.model is not set. The model must be specified in the "
            "spec's 'models:' front-matter — no default model is provided."
        )

    messages, _ = truncate_if_needed(messages, config)
    # TruncationInfo stored in thread-local by truncate_if_needed.

    reasoning_effort = config.reasoning_effort or default_reasoning_effort

    oai_messages: list[dict] = []

    # System prompt as a system message
    if config.system:
        oai_messages.append({"role": "system", "content": config.system})

    for msg in messages:
        if isinstance(msg.content, str):
            oai_messages.append({"role": msg.role, "content": msg.content})
        else:
            parts = [OpenAIClientAdapter._block_to_api(b) for b in msg.content]
            oai_messages.append({"role": msg.role, "content": parts})

    body: dict = {
        "model": config.model,
        "max_completion_tokens": config.max_tokens,
        "messages": oai_messages,
    }
    if reasoning_effort is not None:
        # Reasoning models (o-series) fix temperature at 1; don't pass temperature.
        body["reasoning_effort"] = reasoning_effort
    else:
        body["temperature"] = config.temperature
    return body


class OpenAIClientAdapter:
    """OpenAI SDK adapter satisfying the ``LLMClient`` protocol.

    Parameters
    ----------
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable if not provided.
    model:
        Default model to use (e.g. ``"gpt-4o"``, ``"gpt-5.4"``).
        Overridden by ``LLMClientConfig.model`` at call time.
    reasoning_effort:
        Default reasoning effort for models that support it
        (``"low"``, ``"medium"``, ``"high"``, ``"xhigh"``).
        Overridden by ``LLMClientConfig.reasoning_effort`` at call time.
    base_url:
        Optional base URL override (useful for proxies or Azure endpoints).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        reasoning_effort: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIClientAdapter. "
                "Install with: pip install 'quality-framework[openai]'"
            ) from exc
        import httpx as _httpx
        import os
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        # base_url: explicit arg wins, then OPENAI_BASE_URL env var, then SDK default.
        # Set OPENAI_BASE_URL=https://us.api.openai.com/v1 if you get a regional
        # hostname error (401 "incorrect_hostname").
        effective_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if effective_base_url:
            kwargs["base_url"] = effective_base_url
        # read=300s idle timeout: fires only on genuine stalls (no bytes for 5min).
        kwargs["timeout"] = _httpx.Timeout(connect=30.0, read=300.0, write=None, pool=None)
        self._client = _openai.OpenAI(**kwargs)
        self._default_model = model
        self._default_reasoning_effort = reasoning_effort
        self._limiter = _TokenRateLimiter(tokens_per_minute=20_000_000)

    def complete(self, messages: list[LLMMessage], config: LLMClientConfig) -> str:
        """Call the OpenAI Chat Completions API and return the text response."""
        kwargs = build_request_body(
            messages, config,
            default_reasoning_effort=self._default_reasoning_effort,
        )
        model = kwargs["model"]

        _input_chars = sum(
            len(m["content"]) if isinstance(m["content"], str)
            else sum(len(str(p)) for p in m["content"])
            for m in kwargs["messages"]
        )
        # Reserve input + the requested output budget so the per-minute limit
        # reflects the full call, not just the prompt.
        _estimated_tokens = max(1, _input_chars // 4) + max(0, config.max_tokens or 0)
        self._limiter.acquire(_estimated_tokens)

        _MAX_ATTEMPTS = 2
        last_exc: Exception | None = None
        for _attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._client.chat.completions.create(**kwargs)
                # Record token usage immediately after the API call.
                try:
                    _u = response.usage
                    _cached = getattr(getattr(_u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                    _store_usage([ModelUsage.from_tokens(
                        model=model,
                        input_tokens=getattr(_u, "prompt_tokens", 0) - _cached,
                        output_tokens=getattr(_u, "completion_tokens", 0),
                        cache_read_tokens=_cached,
                    )])
                except Exception:
                    _store_usage([])
                choice = response.choices[0]
                content = choice.message.content
                if not content:
                    finish_reason = choice.finish_reason
                    if finish_reason == "length":
                        raise RuntimeError(
                            f"Model returned empty content with finish_reason='length'. "
                            f"When reasoning_effort is set, reasoning tokens count against "
                            f"max_completion_tokens ({kwargs.get('max_completion_tokens')}), "
                            f"leaving no budget for the output. Increase max_tokens in the "
                            f"dimension's eval_config (recommended: 8192+ for reasoning models)."
                        )
                    raise RuntimeError(
                        f"Model returned empty content (finish_reason={finish_reason!r}). "
                        f"This may indicate a content policy refusal or an API error."
                    )
                return content
            except Exception as exc:
                import httpcore as _httpcore
                import httpx as _httpx
                if isinstance(exc, (
                    _httpcore.RemoteProtocolError, _httpx.RemoteProtocolError,
                    _httpx.ReadTimeout, _httpcore.ReadTimeout,
                )):
                    last_exc = exc
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OpenAI completion failed after retries without a captured retryable exception.")

    @staticmethod
    def _block_to_api(block: ContentBlock) -> dict:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}

        elif isinstance(block, ImageBlock):
            src = block.source
            if src.get("type") == "url":
                return {"type": "image_url", "image_url": {"url": src["url"]}}
            elif src.get("type") == "base64":
                mt = src.get("media_type", "image/png")
                data = src["data"]
                return {"type": "image_url", "image_url": {"url": f"data:{mt};base64,{data}"}}
            raise ValueError(f"Unsupported ImageBlock source type: {src.get('type')}")

        elif isinstance(block, DocumentBlock):
            src = block.source
            if src.get("type") == "text":
                # Plain-text document — inline as text content
                return {"type": "text", "text": src.get("data", "")}
            elif src.get("type") == "base64":
                mt = src.get("media_type", "application/pdf")
                data = src["data"]
                ext = _MEDIA_TYPE_TO_EXT.get(mt, "bin")
                # OpenAI file input block (supported by gpt-4o and later)
                return {
                    "type": "file",
                    "file": {
                        "file_data": f"data:{mt};base64,{data}",
                        "filename": f"document.{ext}",
                    },
                }
            raise ValueError(f"Unsupported DocumentBlock source type: {src.get('type')}")

        raise TypeError(f"Unknown ContentBlock type: {type(block)}")

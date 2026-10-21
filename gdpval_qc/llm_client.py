"""LLM client abstraction layer.

All LLM interaction goes through the ``LLMClient`` protocol. Three concrete
adapters implement it: ``OpenAIClientAdapter`` and ``GeminiClientAdapter`` (the
default judges, defined in ``openai_client.py`` / ``gemini_client.py``) and
``AnthropicClientAdapter`` below, which is the only place in the codebase that
imports the Anthropic SDK. No module imports a provider SDK directly outside its
adapter.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable


# ── Token usage and cost tracking ─────────────────────────────────────────────

# Pricing table: USD per 1 million tokens.
# Format: (input, output, cache_read, cache_write)
# Update when providers change pricing.
# Models are matched by checking if the key is a substring of the model name (lower-case).
_PRICING: dict[str, tuple[float, float, float, float]] = {
    # Anthropic — Claude Opus (most specific first; 4.6/4.5 are cheaper than 4.1/4.0)
    "claude-opus-4-6":   ( 5.00,  25.00, 0.50,  6.25),
    "claude-opus-4-5":   ( 5.00,  25.00, 0.50,  6.25),
    "claude-opus-4":     (15.00,  75.00, 1.50, 18.75),  # catches 4.1, 4.0
    "claude-opus-3":     (15.00,  75.00, 1.50, 18.75),
    # Anthropic — Claude Sonnet (all 4.x same price; 3.7 same price too)
    "claude-sonnet-4":   ( 3.00,  15.00, 0.30,  3.75),
    "claude-sonnet-3-7": ( 3.00,  15.00, 0.30,  3.75),
    # Anthropic — Claude Haiku (most specific first to avoid 3.5 matching "haiku-3")
    "claude-haiku-4-5":  ( 1.00,   5.00, 0.10,  1.25),
    "claude-haiku-4":    ( 1.00,   5.00, 0.10,  1.25),  # future haiku 4.x fallback
    "claude-haiku-3-5":  ( 0.80,   4.00, 0.08,  1.00),
    "claude-haiku-3":    ( 0.25,   1.25, 0.03,  0.30),
    # OpenAI — GPT-5.4 family (most specific first to avoid substring false-matches)
    "gpt-5.4-pro":      (30.00, 180.00, 0.00,   0.00),
    "gpt-5.4-mini":     ( 0.75,   4.50, 0.075,  0.00),
    "gpt-5.4-nano":     ( 0.20,   1.25, 0.02,   0.00),
    "gpt-5.4":          ( 2.50,  15.00, 0.25,   0.00),
    # OpenAI — GPT-5.2 family
    "gpt-5.2-pro":      (21.00, 168.00, 0.00,   0.00),
    "gpt-5.2":          ( 1.75,  14.00, 0.175,  0.00),
    # OpenAI — GPT-5.1
    "gpt-5.1":          ( 1.25,  10.00, 0.125,  0.00),
    # OpenAI — GPT-5 family (after 5.x variants so "gpt-5" doesn't match e.g. "gpt-5.4")
    "gpt-5-pro":        (15.00, 120.00, 0.00,   0.00),
    "gpt-5-mini":       ( 0.25,   2.00, 0.025,  0.00),
    "gpt-5-nano":       ( 0.05,   0.40, 0.005,  0.00),
    "gpt-5":            ( 1.25,  10.00, 0.125,  0.00),
    # OpenAI — GPT-4.1 family
    "gpt-4.1-mini":     ( 0.40,   1.60, 0.10,   0.00),
    "gpt-4.1-nano":     ( 0.10,   0.40, 0.025,  0.00),
    "gpt-4.1":          ( 2.00,   8.00, 0.50,   0.00),
    # OpenAI — GPT-4o family
    "gpt-4o-mini":      ( 0.15,   0.60, 0.075,  0.00),
    "gpt-4o":           ( 2.50,  10.00, 1.25,   0.00),
    # Gemini
    # Gemini — 3.1 series (most specific first; >200k ctx tier; live/image-gen models omitted)
    "gemini-3.1-pro-preview":        ( 4.00, 18.00, 0.40,  0.00),  # also matches -customtools
    "gemini-3.1-flash-lite-preview": ( 0.25,  1.50, 0.025, 0.00),
    # Gemini — 3.0 series
    "gemini-3-flash-preview":        ( 0.50,  3.00, 0.05,  0.00),
    # Gemini — 2.5 series
    "gemini-2.5-pro":                ( 1.25, 10.00, 0.00,  0.00),
    "gemini-2.5-flash":              ( 0.15,  0.60, 0.00,  0.00),
    # Unknown models: pricing is logged as 0. Token counts are still recorded.
}


def _lookup_pricing(model: str) -> tuple[float, float, float, float]:
    """Return (input, output, cache_read, cache_write) per 1M tokens for a model.
    Matches by checking if a pricing key is a substring of the model name.
    Returns (0, 0, 0, 0) for unknown models — token counts are still recorded.
    """
    m = model.lower()
    for key, prices in _PRICING.items():
        if key in m:
            return prices
    return (0.0, 0.0, 0.0, 0.0)


@dataclass
class ModelUsage:
    """Token usage and dollar cost for one LLM call to one model."""
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0    # Anthropic prompt-cache read hits
    cache_creation_tokens: int = 0  # Anthropic prompt-cache writes
    cost_usd: float = 0.0
    pricing_known: bool = True    # False when no pricing entry exists for this model

    @classmethod
    def from_tokens(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        batch_discount: float = 1.0,
    ) -> "ModelUsage":
        """Build usage/cost for one call.

        ``batch_discount`` multiplies the computed cost — pass 0.5 for
        responses returned via the OpenAI Batch API so accumulated cost
        reporting reflects the real discount instead of pricing the call as
        if it had been made synchronously.
        """
        inp, out, cr, cw = _lookup_pricing(model)
        known = any(k in model.lower() for k in _PRICING)
        cost = batch_discount * (
            input_tokens * inp / 1_000_000
            + output_tokens * out / 1_000_000
            + cache_read_tokens * cr / 1_000_000
            + cache_creation_tokens * cw / 1_000_000
        )
        return cls(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cost_usd=round(cost, 8),
            pricing_known=known,
        )

    def to_dict(self) -> dict:
        d = {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }
        if self.cache_read_tokens:
            d["cache_read_tokens"] = self.cache_read_tokens
        if self.cache_creation_tokens:
            d["cache_creation_tokens"] = self.cache_creation_tokens
        if not self.pricing_known:
            d["pricing_known"] = False
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelUsage":
        return cls(
            model=d["model"],
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            cache_read_tokens=d.get("cache_read_tokens", 0),
            cache_creation_tokens=d.get("cache_creation_tokens", 0),
            cost_usd=d.get("cost_usd", 0.0),
            pricing_known=d.get("pricing_known", True),
        )


# Thread-local storage for usage, following the same pattern as TruncationInfo.
_tl_usage = threading.local()

# Process-wide accumulator across ALL threads. complete() calls run on
# ThreadPoolExecutor workers (see v18_engine), so a thread-local "last usage"
# only ever sees the main thread's final call. The accumulator aggregates every
# call so cost reporting reflects the whole run, not one thread.
_accumulated_usage_lock = threading.Lock()
_accumulated_usage: list[ModelUsage] = []


def _store_usage(usage: list[ModelUsage]) -> None:
    _tl_usage.last_usage = usage
    if usage:
        with _accumulated_usage_lock:
            _accumulated_usage.extend(usage)


def get_last_usage() -> list[ModelUsage]:
    """Return usage from the most recent complete() call on this thread.
    Single client: list of length 1.  MultiLLMClient: one entry per model.
    Returns [] if no complete() has been called yet on this thread.
    """
    return getattr(_tl_usage, "last_usage", [])


def get_accumulated_usage() -> list[ModelUsage]:
    """Return every ModelUsage recorded since the last reset, across all threads.

    Thread-safe; returns a snapshot copy. Use this for run-wide cost totals
    rather than ``get_last_usage()`` (which is per-thread and only holds the
    most recent call).
    """
    with _accumulated_usage_lock:
        return list(_accumulated_usage)


def reset_accumulated_usage() -> None:
    """Clear the process-wide usage accumulator (call before a fresh run)."""
    with _accumulated_usage_lock:
        _accumulated_usage.clear()


# ── Content block types ────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = field(default="text", init=False)


@dataclass
class ImageBlock:
    source: dict  # {"type": "base64", "media_type": "image/png", "data": "..."}
                  # or {"type": "url", "url": "..."}
    type: Literal["image"] = field(default="image", init=False)


@dataclass
class DocumentBlock:
    source: dict  # {"type": "base64", "media_type": "application/pdf", "data": "..."}
    type: Literal["document"] = field(default="document", init=False)


ContentBlock = TextBlock | ImageBlock | DocumentBlock


# ── Message and config types ───────────────────────────────────────────────────

@dataclass
class LLMMessage:
    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]  # str is a text-only shorthand


@dataclass
class LLMClientConfig:
    model: str | None = None  # Must be set explicitly — no default model.
    max_tokens: int = 4096
    system: str | None = None
    # Reasoning effort level for models that support extended thinking.
    # Anthropic (claude-opus-4-6): maps to output_config={"effort": ...} + thinking={"type":"adaptive"}.
    #   Accepted values: "low" | "medium" | "high" | "max".
    # OpenAI o-series / GPT reasoning models: passed directly as reasoning_effort.
    #   Accepted values: "low" | "medium" | "high" | "xhigh".
    reasoning_effort: str | None = None
    # Sampling temperature in [0, 1].  0 = most deterministic (greedy).
    # Ignored when reasoning_effort is set — thinking/reasoning models fix temperature
    # internally and reject the parameter.
    temperature: float = 0.0

    def to_dict(self) -> dict:
        d = {"model": self.model, "max_tokens": self.max_tokens, "system": self.system,
             "temperature": self.temperature}
        if self.reasoning_effort is not None:
            d["reasoning_effort"] = self.reasoning_effort
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LLMClientConfig":
        return cls(
            model=d.get("model"),  # None if not specified — caller must set from spec
            max_tokens=d.get("max_tokens", 4096),
            system=d.get("system"),
            reasoning_effort=d.get("reasoning_effort"),
            temperature=d.get("temperature", 0.0),
        )


# ── Protocol ───────────────────────────────────────────────────────────────────

@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface required by the framework and generated Assessors."""

    def complete(self, messages: list[LLMMessage], config: LLMClientConfig) -> str:
        """Send messages and return the model's text response."""
        ...


# ── Per-key token rate limiter ─────────────────────────────────────────────────

class _TokenRateLimiter:
    """Sliding-window token rate limiter.

    Tracks (timestamp, tokens) usage over a 60-second window.  ``acquire()``
    blocks until the requested tokens can be consumed without exceeding
    ``tokens_per_minute``.  Thread-safe under CPython's GIL.
    """

    def __init__(self, tokens_per_minute: int = 1_000_000) -> None:
        import collections, threading
        self._limit = tokens_per_minute
        self._usage: collections.deque = collections.deque()  # (monotonic_ts, tokens)
        self._lock = threading.Lock()

    def acquire(self, tokens: int) -> None:
        import time
        while True:
            with self._lock:
                now = time.monotonic()
                while self._usage and self._usage[0][0] < now - 60.0:
                    self._usage.popleft()
                if not self._usage or sum(t for _, t in self._usage) + tokens <= self._limit:
                    self._usage.append((now, tokens))
                    return
                sleep_until = self._usage[0][0] + 60.0
            time.sleep(max(0.05, sleep_until - time.monotonic()))


# ── Anthropic adapter ──────────────────────────────────────────────────────────

class AnthropicClientAdapter:
    """The only place in the framework that imports ``anthropic``.

    Wraps ``anthropic.Anthropic`` to satisfy ``LLMClient``.

    Accepts one or more API keys.  When multiple keys are supplied the adapter
    rotates through them in round-robin order on every ``complete()`` call,
    spreading token usage evenly across organisations / projects.

    ``api_key`` may be:
    - ``None``              — read ``ANTHROPIC_API_KEY`` from the environment
    - a single ``str``      — use that key for every call
    - a ``list[str]``       — rotate through the list
    """

    def __init__(self, api_key: str | list[str] | None = None) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required. Install it with: pip install anthropic"
            ) from exc
        import httpx as _httpx
        # read=300s: if the server sends no bytes for 5 minutes the stream is stalled.
        # LLM tokens arrive continuously during active generation, so this fires only
        # on genuine stalls (dead connection) — not on slow-but-live generation.
        # connect=30s: fail fast on dead hosts.
        _timeout = _httpx.Timeout(connect=30.0, read=300.0, write=None, pool=None)
        keys: list[str | None] = (
            api_key if isinstance(api_key, list) else [api_key]
        )
        self._clients = [
            _anthropic.Anthropic(api_key=k, timeout=_timeout) if k
            else _anthropic.Anthropic(timeout=_timeout)
            for k in keys
        ]
        self._limiters = [_TokenRateLimiter() for _ in keys]
        self._idx = 0
        # Optional callback fired during streaming to report progress.
        # Signature: (message: str) -> None — receives a human-readable status
        # string already containing tokens/elapsed-time information and phase
        # context (thinking / module name being generated).
        # Set before a long generation call; cleared automatically after.
        self.on_generation_progress: "Callable[[str], None] | None" = None

    def _next_client(self, tokens: int = 0):
        """Return the next client in round-robin order, acquiring the token budget first."""
        idx = self._idx % len(self._clients)
        self._idx += 1
        self._limiters[idx].acquire(tokens)
        return self._clients[idx]

    def complete(self, messages: list[LLMMessage], config: LLMClientConfig) -> str:
        """Call the Anthropic Messages API and return the text response."""
        if not config.model:
            raise ValueError(
                "LLMClientConfig.model is not set. The model must be specified in the "
                "spec's 'models:' front-matter — no default model is provided."
            )
        import anthropic as _anthropic
        from .context_limiter import truncate_if_needed
        messages, _ = truncate_if_needed(messages, config)
        # TruncationInfo stored in thread-local by truncate_if_needed;
        # readable via get_last_truncation_info() on this thread after complete() returns.

        api_messages = []
        for msg in messages:
            if isinstance(msg.content, str):
                api_content: list | str = msg.content
            else:
                api_content = [self._block_to_api(b) for b in msg.content]
            api_messages.append({"role": msg.role, "content": api_content})

        kwargs: dict = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": api_messages,
        }
        if config.system:
            kwargs["system"] = config.system
        if config.reasoning_effort:
            # Extended thinking fixes temperature at 1; don't pass temperature parameter.
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": config.reasoning_effort}
        else:
            kwargs["temperature"] = config.temperature

        # Use streaming to avoid the SDK's non-streaming timeout for large max_tokens.
        # Retry once on connection drops (RemoteProtocolError / incomplete chunked read),
        # which can happen when generation runs for several minutes.
        # Rough input token estimate for rate limiting (characters / 4).
        _input_chars = sum(
            len(c) if isinstance(c, str) else sum(
                len(b.text) if hasattr(b, "text") else len(str(getattr(b, "source", "")))
                for b in c
            )
            for m in api_messages for c in [m["content"]]
        )
        # Reserve input + the requested output budget so the per-minute limit
        # reflects the full call, not just the prompt.
        _estimated_input_tokens = max(1, _input_chars // 4) + max(0, config.max_tokens or 0)

        # Time-based progress heartbeat. Every call gets a status update at least
        # this often (in seconds), so the UI never goes silent for longer than
        # this interval during a long synchronous build. Also fires opportunistic
        # phase-change events (thinking→text, module boundaries) ahead of schedule.
        _PROGRESS_HEARTBEAT_S = 20.0
        # Module markers the assembler instructs Claude to emit around each
        # generated module. Detecting them in the text stream gives semantic
        # progress ("generating prompts.py..." vs just a token count).
        _MODULE_MARKERS = (
            ("<init_py>",        "__init__.py"),
            ("<checks_py>",      "checks.py"),
            ("<prompts_py>",     "prompts.py"),
            ("<aggregation_py>", "aggregation.py"),
        )
        import time as _time
        _MAX_ATTEMPTS = 2
        last_exc: Exception | None = None
        for _attempt in range(_MAX_ATTEMPTS):
            try:
                with self._next_client(_estimated_input_tokens).messages.stream(**kwargs) as stream:
                    chunks: list[str] = []
                    thinking_chars = 0
                    text_chars = 0
                    text_seen = False
                    current_module: str | None = None
                    # Rolling buffer of the most-recent text so module markers
                    # that straddle chunk boundaries are still caught. Modules
                    # are tracked in `seen_modules` so each marker fires at most
                    # once even though its substring keeps appearing in later
                    # buffer windows.
                    recent_text_tail = ""
                    seen_modules: set = set()
                    started_t = _time.monotonic()
                    last_progress_t = started_t - _PROGRESS_HEARTBEAT_S  # force first

                    def _fire_progress(force: bool = False,
                                       override_msg: "str | None" = None) -> None:
                        """Emit a status string via the registered callback.

                        Rate-limited to once per _PROGRESS_HEARTBEAT_S unless
                        `force` is True (phase change events). `override_msg`
                        lets callers supply a phase-specific message such as
                        the thinking→text transition.
                        """
                        nonlocal last_progress_t
                        cb = self.on_generation_progress
                        if cb is None:
                            return
                        now = _time.monotonic()
                        if not force and (now - last_progress_t) < _PROGRESS_HEARTBEAT_S:
                            return
                        elapsed = int(now - started_t)
                        if override_msg is not None:
                            msg = override_msg
                        elif not text_seen:
                            msg = (f"thinking… ~{thinking_chars // 4:,} thinking tokens, "
                                   f"{elapsed}s elapsed")
                        else:
                            total_tok = (thinking_chars + text_chars) // 4
                            if current_module:
                                msg = (f"generating {current_module}… "
                                       f"~{total_tok:,} tokens, {elapsed}s elapsed")
                            else:
                                msg = (f"generating output… ~{total_tok:,} tokens, "
                                       f"{elapsed}s elapsed")
                        try:
                            cb(msg)
                        except Exception:
                            # Never let a buggy progress callback poison the
                            # generation flow.
                            pass
                        last_progress_t = now

                    # Immediate "we've started" notification so the UI never
                    # sits silent at t=0 waiting for the heartbeat to fire.
                    _fire_progress(force=True,
                                    override_msg="streaming started; awaiting first event…")

                    # Iterate the raw event stream so we see both thinking and
                    # text deltas. The Anthropic SDK exposes these as objects
                    # with `.type` and `.delta` fields. Falls back gracefully
                    # for unexpected event shapes.
                    for event in stream:
                        etype = getattr(event, "type", "")
                        if etype == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            dtype = getattr(delta, "type", "") if delta is not None else ""
                            if dtype == "thinking_delta":
                                thinking_chars += len(getattr(delta, "thinking", "") or "")
                            elif dtype == "text_delta":
                                text_piece = getattr(delta, "text", "") or ""
                                if not text_seen:
                                    # Announce the transition with thinking-phase
                                    # totals, then flip the phase flag.
                                    _elapsed = int(_time.monotonic() - started_t)
                                    _fire_progress(force=True, override_msg=(
                                        f"thinking complete (~{thinking_chars // 4:,} "
                                        f"thinking tokens, {_elapsed}s); generating output…"
                                    ))
                                    text_seen = True
                                text_chars += len(text_piece)
                                chunks.append(text_piece)
                                # Detect module-boundary markers in a rolling
                                # window so a marker split across two chunks
                                # is still caught.
                                recent_text_tail = (recent_text_tail + text_piece)[-160:]
                                # Detect each module marker AT MOST ONCE per call.
                                # The rolling buffer holds the last 160 chars; if
                                # a marker has already fired, ignore subsequent
                                # appearances of its substring in later windows.
                                for marker, module_name in _MODULE_MARKERS:
                                    if module_name in seen_modules:
                                        continue
                                    if marker in recent_text_tail:
                                        seen_modules.add(module_name)
                                        current_module = module_name
                                        _fire_progress(force=True)
                                        break
                        _fire_progress()   # heartbeat (rate-limited)
                    # Final wrap-up message — total tokens + elapsed time.
                    _final_elapsed = int(_time.monotonic() - started_t)
                    _final_total_tokens = (thinking_chars + text_chars) // 4
                    _fire_progress(force=True, override_msg=(
                        f"streaming complete — ~{_final_total_tokens:,} output tokens "
                        f"({thinking_chars // 4:,} thinking + {text_chars // 4:,} text) "
                        f"in {_final_elapsed}s"
                    ))
                    # Capture token usage from the completed stream.
                    try:
                        _final = stream.get_final_message()
                        _u = _final.usage
                        _store_usage([ModelUsage.from_tokens(
                            model=config.model,
                            input_tokens=getattr(_u, "input_tokens", 0),
                            output_tokens=getattr(_u, "output_tokens", 0),
                            cache_read_tokens=getattr(_u, "cache_read_input_tokens", 0),
                            cache_creation_tokens=getattr(_u, "cache_creation_input_tokens", 0),
                        )])
                    except Exception:
                        _store_usage([])
                return "".join(chunks)
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
        raise RuntimeError("Retry loop exhausted without a captured retryable exception")

    @staticmethod
    def _block_to_api(block: ContentBlock) -> dict:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text}
        elif isinstance(block, ImageBlock):
            return {"type": "image", "source": block.source}
        elif isinstance(block, DocumentBlock):
            return {"type": "document", "source": block.source}
        raise TypeError(f"Unknown ContentBlock type: {type(block)}")

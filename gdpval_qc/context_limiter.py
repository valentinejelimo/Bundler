"""Input context limiter for LLM evaluation calls.

When document content is very large, evaluation prompts can exceed a model's
context window.  This module estimates token usage and, when needed, falls back
to text-only extraction for binary documents (PDFs) so that instructions,
rubrics, and CSV/structured-text blocks are never modified.

Design notes
------------
* Token estimation uses a 4 chars/token approximation for text (conservative for
  English; slightly generous for code / structured data).  Binary content (base64
  PDFs, images) is estimated from decoded byte size.  Exact tokenisation is not
  needed — the goal is to avoid hard API errors, not to fill every last token.

* PDF token estimation uses 30 decoded bytes / token.  This is intentionally
  conservative: Anthropic processes PDFs by rendering each page as image tiles
  (~1 500–3 000 tokens/page) regardless of file size, so a 5-page PDF can cost
  10 000–15 000 tokens even if it is only a few hundred KB on disk.

* Three-stage overflow strategy (only triggered when estimated tokens exceed limit):

  1. Convert binary PDF DocumentBlocks to plain-text extracts via ``pypdf``.
     This trades rich layout rendering for a much smaller token footprint.
     Text blocks (instructions, rubrics, CSV, original plain-text docs) are
     never modified at this stage.

  2. If the prompt is still over-limit after PDF-to-text conversion, apply
     middle-out truncation to the converted PDF text blocks only.  Original
     TextBlocks and non-PDF DocumentBlocks remain untouched.

  3. If the prompt is still over-limit even after truncating PDF text (e.g.
     images alone overflow the window), emit a warning and send as-is.

* Converted PDF blocks are tagged internally with ``_converted_from_pdf: True``
  in their source dict so the truncation step can identify them.  This marker
  is stripped before the messages are returned so the API never sees it.

* If ``pypdf`` is not installed, or text extraction yields nothing (scanned /
  image-only PDFs), the original binary block is kept unchanged.

* A fixed response headroom (default 4 096 tokens) is always reserved so the
  model has room to write its verdict and reasoning.

* An unknown model name falls back to a conservative 128 000-token limit.
"""
from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING


# ── Thread-local truncation signal ─────────────────────────────────────────────

_tls = threading.local()


@dataclass
class TruncationInfo:
    """Counts of context-reduction actions taken for a single LLM call.

    pdfs_converted:   Number of base64 PDF DocumentBlocks replaced with
                      plain-text extracts (Stage 1 — lossy: images, layout,
                      and charts are discarded).
    blocks_truncated: Number of converted-PDF text blocks where middle-out
                      truncation removed content to fit the context window
                      (Stage 2).
    """

    pdfs_converted: int = 0
    blocks_truncated: int = 0

    @property
    def input_truncated(self) -> bool:
        """True if any input was silently reduced to fit the model context window."""
        return self.pdfs_converted > 0 or self.blocks_truncated > 0


def _store_truncation_info(info: TruncationInfo) -> None:
    """Store *info* in thread-local storage for the current thread."""
    _tls.last_info = info


def get_last_truncation_info() -> TruncationInfo:
    """Return the TruncationInfo from the most recent ``truncate_if_needed()``
    call on this thread.  Returns a zeroed ``TruncationInfo`` if no call has
    been made yet on this thread."""
    return getattr(_tls, "last_info", TruncationInfo())


class ContextLimitError(ValueError):
    """Raised when a prompt cannot be reduced to fit within the model's context window.

    Attributes:
        estimated_tokens:    Estimated input token count after all reduction attempts.
        limit:               Model's context window limit (input tokens).
        response_budget:     Tokens reserved for the model's response.
        overflow:            Tokens over limit (estimated_tokens + response_budget - limit).
        model:               Model name.
    """

    def __init__(
        self,
        estimated_tokens: int,
        limit: int,
        model: str,
        response_token_budget: int = 4_096,
    ) -> None:
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        self.response_budget = response_token_budget
        self.overflow = estimated_tokens + response_token_budget - limit
        self.model = model
        super().__init__(
            f"Context limit exceeded for model '{model}': "
            f"~{estimated_tokens:,} input tokens + {response_token_budget:,} response budget "
            f"= ~{estimated_tokens + response_token_budget:,} total "
            f"(limit {limit:,}; ~{self.overflow:,} tokens over) "
            f"even after PDF text extraction and truncation. "
            f"Reduce the number or size of input/output files for this sample."
        )

if TYPE_CHECKING:
    from .llm_client import LLMClientConfig, LLMMessage


# ── Context window limits (input tokens) ───────────────────────────────────────

_MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Anthropic
    # claude-opus-4-6, claude-sonnet-4-6: 1M token context window
    # claude-haiku-4-5, claude-opus-4-x, claude-sonnet-4-x (older): 200K
    # Source: https://docs.anthropic.com/en/docs/about-claude/models
    "claude-opus-4-6":    1_000_000,
    "claude-sonnet-4-6":  1_000_000,
    "claude-haiku-4-5":     200_000,
    "claude-opus-4-5":      200_000,
    "claude-opus-4-1":      200_000,
    "claude-opus-4":        200_000,
    "claude-sonnet-4-5":    200_000,  # 1M available via beta header; use 200K as default
    "claude-sonnet-4":      200_000,  # 1M available via beta header; use 200K as default

    # OpenAI GPT-4o family — 128K
    # Source: https://platform.openai.com/docs/models/gpt-4o
    "gpt-4o":             128_000,
    "gpt-4o-mini":        128_000,

    # OpenAI GPT-4.1 family — 1M
    # Source: https://platform.openai.com/docs/models/gpt-4.1
    "gpt-4.1":          1_000_000,
    "gpt-4.1-mini":     1_000_000,
    "gpt-4.1-nano":     1_000_000,

    # OpenAI GPT-5 family
    # gpt-5.4 standard context window is 272K; 1.05M requires explicit experimental config
    # Source: https://developers.openai.com/api/docs/models/gpt-5.4
    "gpt-5.5":            272_000,
    "gpt-5.4":            272_000,
    "gpt-5":              272_000,

    # OpenAI o-series reasoning models
    # o1, o1-pro, o3, o3-mini, o4-mini: 200K
    # o1-mini: 128K (smaller variant)
    # Source: https://platform.openai.com/docs/models/o3
    "o1":                 200_000,
    "o1-mini":            128_000,
    "o1-pro":             200_000,
    "o3":                 200_000,
    "o3-mini":            200_000,
    "o4-mini":            200_000,

    # Google Gemini
    # gemini-1.5-pro: 2M; all 1.5-flash and 2.x models: 1M
    # Source: https://ai.google.dev/gemini-api/docs/models
    "gemini-1.5-pro":   2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-2.5-pro":         1_000_000,
    "gemini-2.5-flash":       1_000_000,
    "gemini-3.1-pro-preview": 1_000_000,
}

# Conservative fallback for unrecognised models
_DEFAULT_CONTEXT_LIMIT = 128_000

# Chars per token (rough approximation for English/structured text)
_CHARS_PER_TOKEN = 4

# Chars per token for text extracted FROM a converted PDF (via pypdf, Stage 1
# of the overflow strategy below). This text is characteristically dense and
# whitespace-heavy — cell-alignment padding, repeated column headers per
# rendered page, numeric-only cells — and tokenizes far less efficiently than
# prose. Measured against a real API rejection on task_322 (two xlsx workbooks
# whose LibreOffice conversion exploded to 1,610 and 2,363 pages from a
# degenerate print-area/column-width setup): our original 4-chars/token
# estimate said ~676K-892K tokens fit under Gemini's 1,048,576-token ceiling,
# but the live API still 400'd. An interim 2.5 chars/token guess was STILL
# 1.40-1.41x too generous — verified directly against Gemini's own
# `models.count_tokens` (free, exact, no judge spend) across 4 dims/PDF
# combinations on task_322, all converging on the same ~1.41x factor
# (2.5 / 1.41 ≈ 1.77). Using 1.75 chars/token for a little extra headroom.
# Only text-type DocumentBlocks are affected — in this codebase those are
# produced exclusively by `_convert_pdfs_to_text` below, never by the
# TextBlock path (prose keeps the 4-chars/token estimate).
_PDF_EXTRACTED_TEXT_CHARS_PER_TOKEN = 1.75

# Rough bytes-per-token for binary documents (PDFs processed by the model API).
#
# Anthropic converts each PDF page to image tiles for processing.  A standard
# letter-size page costs roughly 1 500–3 000 tokens.  A typical 5-page text PDF
# encoded in base64 is ~500 KB (decoded ~375 KB), giving roughly 7 500–15 000
# real tokens — about 30 bytes per token decoded.
# Using 30 bytes/token rather than 750 avoids a 25× under-estimation that would
# otherwise let large PDFs silently overflow the context window.
_PDF_BYTES_PER_TOKEN = 30

# Tokens reserved for message-framing overhead (role markers, JSON structure, etc.)
_FRAMING_OVERHEAD_TOKENS = 500

# Minimum chars to keep in any single PDF text block after truncation.
_MIN_BLOCK_CHARS = 200

# Headroom (tokens) subtracted from the Stage-2 PDF-text budget so truncation
# targets *below* the hard ceiling rather than right at it. Absorbs unbudgeted
# overhead that lands after the budget is computed: the per-block "[... N
# characters omitted ...]" markers (~22 tokens each) and _CHARS_PER_TOKEN
# rounding drift. A flat 2,000-token floor was tuned against a ~272K window
# (~0.7%) but proved far too thin on a 1M-token window: task_322's two
# LibreOffice-exploded xlsx->PDF conversions (1,610 and 2,363 pages, from
# degenerate print-area/column-width setup) extract to ~5.5M chars of dense,
# whitespace-padded tabular text. Our 4-chars/token estimate under-counts
# real tokenizer output for this kind of content enough that a request
# "truncated to fit" by our own math still overflowed Gemini's real 1M-token
# ceiling with a hard 400. Scaling the margin with the limit (10%, floored at
# the old 2,000) gives real headroom against that estimator error instead of
# a constant that shrinks to irrelevance as the window grows.
_TRUNCATION_SAFETY_MARGIN_FRACTION = 0.10
_TRUNCATION_SAFETY_MARGIN_TOKENS_FLOOR = 2_000


def _truncation_safety_margin_tokens(limit: int) -> int:
    return max(_TRUNCATION_SAFETY_MARGIN_TOKENS_FLOOR,
               int(limit * _TRUNCATION_SAFETY_MARGIN_FRACTION))


def get_context_limit(model: str) -> int:
    """Return the input context limit in tokens for a model name.

    Performs an exact match first, then a prefix match so that versioned names
    like ``"gpt-4o-2024-05-13"`` resolve to ``"gpt-4o"``'s limit.  Falls back
    to ``_DEFAULT_CONTEXT_LIMIT`` for unknown models.
    """
    if model in _MODEL_CONTEXT_LIMITS:
        return _MODEL_CONTEXT_LIMITS[model]
    ml = model.lower().strip()
    # Prefix match — sort by key length descending so longer (more specific)
    # prefixes win over shorter ones.
    for key in sorted(_MODEL_CONTEXT_LIMITS, key=len, reverse=True):
        if ml.startswith(key.lower()):
            return _MODEL_CONTEXT_LIMITS[key]
    return _DEFAULT_CONTEXT_LIMIT


# ── Token estimation ────────────────────────────────────────────────────────────

def _estimate_text_tokens(text: str, chars_per_token: float = _CHARS_PER_TOKEN) -> int:
    return max(1, int(len(text) / chars_per_token))


def _estimate_block_tokens(block) -> int:
    from .llm_client import DocumentBlock, ImageBlock, TextBlock

    if isinstance(block, TextBlock):
        return _estimate_text_tokens(block.text)

    if isinstance(block, DocumentBlock):
        src = block.source
        if src.get("type") == "text":
            # Text-type DocumentBlocks are produced exclusively by
            # _convert_pdfs_to_text below (PDF-extracted, not prose) — use
            # the stricter ratio calibrated against that content.
            return _estimate_text_tokens(
                src.get("data", ""), _PDF_EXTRACTED_TEXT_CHARS_PER_TOKEN
            )
        if src.get("type") == "base64":
            # Decode length: base64 chars × 3/4 ≈ bytes
            decoded_bytes = len(src.get("data", "")) * 3 // 4
            return max(1, decoded_bytes // _PDF_BYTES_PER_TOKEN)

    if isinstance(block, ImageBlock):
        src = block.source
        if src.get("type") == "base64":
            # Image tokens depend on resolution; use byte size as proxy.
            # Anthropic/OpenAI vision: ~85–1 568 tokens per image tile.
            # Rough: 1 token per 30 decoded bytes (images are dense).
            decoded_bytes = len(src.get("data", "")) * 3 // 4
            return max(85, decoded_bytes // 30)
        return 512  # URL image: assume moderate cost

    return 0


def _estimate_messages_tokens(
    messages: "list[LLMMessage]",
    system: str | None,
) -> int:
    total = _FRAMING_OVERHEAD_TOKENS
    if system:
        total += _estimate_text_tokens(system)
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            total += _estimate_text_tokens(content)
        else:
            for block in content:
                total += _estimate_block_tokens(block)
    return total


# ── Truncation helpers (PDF text only) ─────────────────────────────────────────

def _truncate_middle(text: str, max_chars: int) -> str:
    """Keep the first and last halves of ``text``, inserting a gap marker."""
    if len(text) <= max_chars:
        return text
    keep = max(max_chars, _MIN_BLOCK_CHARS)
    half = keep // 2
    removed = len(text) - keep
    return (
        text[:half]
        + f"\n\n[... {removed:,} characters omitted — PDF text truncated to fit"
        f" model context window ...]\n\n"
        + text[len(text) - half :]
    )


# ── PDF text extraction ─────────────────────────────────────────────────────────

def _extract_pdf_text(base64_data: str) -> str | None:
    """Extract plain text from a base64-encoded PDF using ``pypdf``.

    Returns the concatenated text of all pages, or ``None`` if extraction
    fails (pypdf not installed, scanned/image-only PDF, corrupt file, etc.).
    """
    try:
        import base64 as _base64
        import io

        pdf_bytes = _base64.b64decode(base64_data)
        try:
            import pypdf
        except ImportError:
            return None

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)
        return "\n\n".join(pages) if pages else None
    except Exception:
        return None


def _convert_pdfs_to_text(
    messages: "list[LLMMessage]",
) -> "tuple[list[LLMMessage], int]":
    """Replace base64 PDF DocumentBlocks with text-only blocks where possible.

    Returns ``(converted_messages, n_converted)`` where ``n_converted`` is the
    number of PDF blocks that were successfully converted.  Messages are
    deep-copied before modification so the originals are untouched.
    """
    import copy
    from .llm_client import DocumentBlock

    messages = copy.deepcopy(messages)
    n_converted = 0

    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            continue
        for block in content:
            if not isinstance(block, DocumentBlock):
                continue
            src = block.source
            if src.get("type") != "base64":
                continue
            media_type = src.get("media_type", "")
            if "pdf" not in media_type.lower():
                continue
            # Use pre-extracted text if the file cache populated it (avoids re-running pypdf).
            extracted = src.get("_pdf_text_cache") or _extract_pdf_text(src.get("data", ""))
            if extracted:
                block.source = {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": f"[PDF text extract]\n{extracted}",
                    # Internal marker — identifies this block as truncation-eligible.
                    # Stripped by _strip_pdf_markers() before messages reach the API.
                    "_converted_from_pdf": True,
                }
                n_converted += 1

    return messages, n_converted


def _truncate_pdf_text_blocks(
    messages: "list[LLMMessage]",
    token_budget: int,
) -> "tuple[list[LLMMessage], int]":
    """Proportionally truncate converted-PDF text blocks to fit *token_budget*.

    Only blocks tagged with ``_converted_from_pdf: True`` are eligible.
    Original TextBlocks and non-PDF DocumentBlocks are never touched.

    Returns ``(messages, n_truncated)`` where ``n_truncated`` is the number of
    blocks where content was actually removed.
    """
    from .llm_client import DocumentBlock

    # Collect eligible (block, current_text, token_estimate) tuples
    items: list[tuple[object, str, int]] = []
    for msg in messages:
        if isinstance(msg.content, str):
            continue
        for block in msg.content:
            if not isinstance(block, DocumentBlock):
                continue
            src = block.source
            if src.get("type") != "text" or not src.get("_converted_from_pdf"):
                continue
            text = src.get("data", "")
            items.append((block, text,
                          _estimate_text_tokens(text, _PDF_EXTRACTED_TEXT_CHARS_PER_TOKEN)))

    if not items:
        return messages, 0

    total_tokens = sum(tok for _, _, tok in items)
    if total_tokens <= token_budget:
        return messages, 0  # already fits within the allocated budget

    frac_to_keep = max(0.0, token_budget / total_tokens)
    n_truncated = 0
    for block, text, _ in items:
        max_chars = max(_MIN_BLOCK_CHARS, int(len(text) * frac_to_keep))
        if len(text) > max_chars:
            n_truncated += 1
        block.source = dict(block.source, data=_truncate_middle(text, max_chars))

    return messages, n_truncated


def _strip_pdf_markers(messages: "list[LLMMessage]") -> "list[LLMMessage]":
    """Remove internal markers from all DocumentBlock source dicts.

    Strips ``_converted_from_pdf`` (added by context truncation) and
    ``_pdf_text_cache`` (added by _build_file_cache in generated prompts.py).
    Must be called before messages are sent to any LLM API.
    """
    from .llm_client import DocumentBlock

    for msg in messages:
        if isinstance(msg.content, str):
            continue
        for block in msg.content:
            if isinstance(block, DocumentBlock):
                block.source.pop("_converted_from_pdf", None)
                block.source.pop("_pdf_text_cache", None)

    return messages


# ── Public entry point ──────────────────────────────────────────────────────────

def truncate_if_needed(
    messages: "list[LLMMessage]",
    config: "LLMClientConfig",
    response_token_budget: int | None = None,
) -> "tuple[list[LLMMessage], TruncationInfo]":
    """Return ``(messages, TruncationInfo)`` that fit within the model's context limit.

    Always stores the ``TruncationInfo`` in thread-local storage before returning
    so that callers can retrieve it via ``get_last_truncation_info()`` after the
    LLM call completes on the same thread.

    Strategy
    --------
    1. If the estimated token count already fits — return messages unchanged,
       ``TruncationInfo(pdfs_converted=0, blocks_truncated=0)``.
    2. If it does not fit — convert base64 PDF DocumentBlocks to plain-text
       extracts via ``pypdf``.  Original TextBlocks and non-PDF DocumentBlocks
       are never modified.
    3. If still over-limit — apply middle-out truncation to the converted PDF
       text blocks only (proportional across all of them).
    4. If still over-limit after that — raise ``ContextLimitError``.

    Parameters
    ----------
    messages:
        The messages to be sent to the model.
    config:
        The LLMClientConfig for this call (used for model name and system prompt).
    response_token_budget:
        Tokens to reserve for the model's response.  When ``None`` (the
        default) the request's own ``config.max_tokens`` is reserved, so the
        fit check matches what will actually be sent (the spec asks judges for
        ``max_tokens`` far larger than the old fixed 4 096). Falls back to
        4 096 only when ``config.max_tokens`` is unset.
    """
    if response_token_budget is None:
        response_token_budget = config.max_tokens or 4_096
    limit = get_context_limit(config.model)
    estimated = _estimate_messages_tokens(messages, config.system)

    if estimated + response_token_budget <= limit:
        _strip_pdf_markers(messages)
        info = TruncationInfo()
        _store_truncation_info(info)
        return messages, info

    # ── Stage 1: convert binary PDFs to extracted text ──────────────────────
    converted, n_converted = _convert_pdfs_to_text(messages)

    if n_converted == 0:
        raise ContextLimitError(estimated, limit, config.model, response_token_budget)

    after_convert = _estimate_messages_tokens(converted, config.system)

    if after_convert + response_token_budget <= limit:
        warnings.warn(
            f"Prompt for model '{config.model}' estimated at ~{estimated:,} tokens "
            f"(limit {limit:,}). Converted {n_converted} PDF(s) to plain-text "
            f"extract (new estimate ~{after_convert:,} tokens).",
            RuntimeWarning,
            stacklevel=4,
        )
        _strip_pdf_markers(converted)
        info = TruncationInfo(pdfs_converted=n_converted)
        _store_truncation_info(info)
        return converted, info

    # ── Stage 2: truncate converted-PDF text blocks to reclaim more budget ──
    # Token budget available exclusively for the converted PDF text blocks:
    # total limit minus everything that cannot be shrunk (response headroom,
    # system prompt, framing, and all non-PDF-text content).
    non_pdf_tokens = after_convert - sum(
        _estimate_text_tokens(block.source.get("data", ""), _PDF_EXTRACTED_TEXT_CHARS_PER_TOKEN)
        for msg in converted
        if not isinstance(msg.content, str)
        for block in msg.content
        if hasattr(block, "source")
        and block.source.get("type") == "text"
        and block.source.get("_converted_from_pdf")
    )
    pdf_text_budget = max(
        int(_MIN_BLOCK_CHARS / _PDF_EXTRACTED_TEXT_CHARS_PER_TOKEN),
        limit - response_token_budget - non_pdf_tokens
        - _truncation_safety_margin_tokens(limit),
    )
    converted, n_truncated = _truncate_pdf_text_blocks(converted, pdf_text_budget)

    after_truncate = _estimate_messages_tokens(converted, config.system)
    still_over = after_truncate + response_token_budget - limit

    if still_over > 0:
        raise ContextLimitError(after_truncate, limit, config.model, response_token_budget)

    warnings.warn(
        f"Prompt for model '{config.model}' estimated at ~{estimated:,} tokens "
        f"(limit {limit:,}). Converted {n_converted} PDF(s) to text and truncated "
        f"extracted content (new estimate ~{after_truncate:,} tokens).",
        RuntimeWarning,
        stacklevel=4,
    )
    _strip_pdf_markers(converted)
    info = TruncationInfo(pdfs_converted=n_converted, blocks_truncated=n_truncated)
    _store_truncation_info(info)
    return converted, info

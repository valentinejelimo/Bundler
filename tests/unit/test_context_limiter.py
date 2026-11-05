"""Unit tests for gdpval_qc.context_limiter — token estimation and truncation.

Tests get_context_limit, TruncationInfo, get_last_truncation_info, and
truncate_if_needed with message lists. Does NOT test the lazy pypdf path.
"""
from __future__ import annotations

import warnings

import pytest

from gdpval_qc.context_limiter import (
    ContextLimitError, TruncationInfo, get_context_limit,
    get_last_truncation_info, truncate_if_needed,
    _estimate_text_tokens, _estimate_messages_tokens,
    _truncate_middle, _CHARS_PER_TOKEN, _FRAMING_OVERHEAD_TOKENS,
)
from gdpval_qc.llm_client import (
    DocumentBlock, ImageBlock, LLMClientConfig, LLMMessage, TextBlock,
)


# ── TruncationInfo ────────────────────────────────────────────────────────────

class TestTruncationInfo:
    def test_defaults(self):
        ti = TruncationInfo()
        assert ti.pdfs_converted == 0
        assert ti.blocks_truncated == 0

    def test_input_truncated_false_when_zero(self):
        assert TruncationInfo().input_truncated is False

    def test_input_truncated_true_pdfs(self):
        assert TruncationInfo(pdfs_converted=1).input_truncated is True

    def test_input_truncated_true_blocks(self):
        assert TruncationInfo(blocks_truncated=1).input_truncated is True


# ── get_last_truncation_info ─────────────────────────────────────────────────

class TestGetLastTruncationInfo:
    def test_default_when_no_call(self, monkeypatch):
        # Reset thread-local state: must delete the attribute, not set to None,
        # because get_last_truncation_info uses getattr with a default that
        # only fires when the attribute is absent.
        import gdpval_qc.context_limiter as cl
        if hasattr(cl._tls, "last_info"):
            monkeypatch.delattr(cl._tls, "last_info")
        info = get_last_truncation_info()
        assert info.pdfs_converted == 0
        assert info.blocks_truncated == 0
        assert info.input_truncated is False


# ── get_context_limit ────────────────────────────────────────────────────────

class TestGetContextLimit:
    def test_exact_match(self):
        assert get_context_limit("gpt-4o") == 128_000

    def test_exact_match_anthropic(self):
        assert get_context_limit("claude-opus-4-6") == 1_000_000

    def test_prefix_match_versioned(self):
        # gpt-4o-2024-05-13 should resolve to gpt-4o's limit (128K)
        assert get_context_limit("gpt-4o-2024-05-13") == 128_000

    def test_prefix_match_gemini(self):
        assert get_context_limit("gemini-1.5-pro-002") == 2_000_000

    def test_unknown_model_falls_back(self):
        assert get_context_limit("some-unknown-model") == 128_000

    def test_case_insensitive_prefix(self):
        # The prefix match lowercases both
        assert get_context_limit("GPT-4o") == 128_000

    def test_o1(self):
        assert get_context_limit("o1") == 200_000

    def test_o1_mini(self):
        assert get_context_limit("o1-mini") == 128_000


# ── Token estimation ─────────────────────────────────────────────────────────

class TestEstimateTextTokens:
    def test_empty_string(self):
        # max(1, 0/4) = 1
        assert _estimate_text_tokens("") == 1

    def test_short_string(self):
        # max(1, 4/4) = 1
        assert _estimate_text_tokens("abcd") == 1

    def test_longer_string(self):
        # max(1, 40/4) = 10
        assert _estimate_text_tokens("a" * 40) == 10

    def test_custom_chars_per_token(self):
        assert _estimate_text_tokens("a" * 10, chars_per_token=2) == 5


class TestEstimateMessagesTokens:
    def test_overhead_present(self):
        msgs = [LLMMessage(role="user", content="")]
        # Empty string still costs 1 token: max(1, 0/4) = 1
        assert _estimate_messages_tokens(msgs, None) == _FRAMING_OVERHEAD_TOKENS + 1

    def test_system_prompt_counted(self):
        msgs = [LLMMessage(role="user", content="")]
        # 500 framing + 1 (empty content) + 1000 (4000-char system / 4)
        toks = _estimate_messages_tokens(msgs, "a" * 4000)
        assert toks == _FRAMING_OVERHEAD_TOKENS + 1 + 1000

    def test_text_content_counted(self):
        msgs = [LLMMessage(role="user", content="a" * 4000)]
        toks = _estimate_messages_tokens(msgs, None)
        assert toks == _FRAMING_OVERHEAD_TOKENS + 1000

    def test_block_content_counted(self):
        block = TextBlock(text="a" * 4000)
        msgs = [LLMMessage(role="user", content=[block])]
        toks = _estimate_messages_tokens(msgs, None)
        assert toks == _FRAMING_OVERHEAD_TOKENS + 1000


# ── _truncate_middle ─────────────────────────────────────────────────────────

class TestTruncateMiddle:
    def test_no_truncation_when_short(self):
        text = "short"
        assert _truncate_middle(text, 100) == text

    def test_truncation_inserts_marker(self):
        text = "a" * 1000
        result = _truncate_middle(text, 100)
        assert "characters omitted" in result
        assert len(result) < len(text)

    def test_truncation_keeps_head_and_tail(self):
        text = "HEAD" + "x" * 1000 + "TAIL"
        result = _truncate_middle(text, 100)
        assert result.startswith("HEAD")
        assert result.endswith("TAIL")


# ── truncate_if_needed ───────────────────────────────────────────────────────

class TestTruncateIfNeeded:
    def test_fits_unchanged(self):
        msgs = [LLMMessage(role="user", content="hello world")]
        config = LLMClientConfig(model="gpt-4o", max_tokens=4096, system=None)
        out, info = truncate_if_needed(msgs, config)
        assert out is msgs  # returned as-is (stripped but same objects)
        assert info.pdfs_converted == 0
        assert info.blocks_truncated == 0
        assert info.input_truncated is False

    def test_stores_truncation_info_in_thread_local(self):
        msgs = [LLMMessage(role="user", content="hello")]
        config = LLMClientConfig(model="gpt-4o", max_tokens=4096)
        truncate_if_needed(msgs, config)
        info = get_last_truncation_info()
        assert isinstance(info, TruncationInfo)
        assert info.pdfs_converted == 0

    def test_overflow_no_pdfs_raises_context_limit_error(self):
        # Very large text with no PDFs to convert -> must raise
        big_text = "a" * (500_000 * 4)  # ~500K tokens, over gpt-4o's 128K
        msgs = [LLMMessage(role="user", content=big_text)]
        config = LLMClientConfig(model="gpt-4o", max_tokens=4096)
        with pytest.raises(ContextLimitError) as exc_info:
            truncate_if_needed(msgs, config)
        assert exc_info.value.model == "gpt-4o"
        assert exc_info.value.limit == 128_000

    def test_response_budget_from_config_max_tokens(self):
        # When response_token_budget is None, config.max_tokens is used
        msgs = [LLMMessage(role="user", content="a" * (130_000 * 4))]
        config = LLMClientConfig(model="gpt-4o", max_tokens=4096)
        with pytest.raises(ContextLimitError):
            truncate_if_needed(msgs, config)

    def test_explicit_response_budget(self):
        msgs = [LLMMessage(role="user", content="a" * (120_000 * 4))]
        config = LLMClientConfig(model="gpt-4o", max_tokens=4096)
        with pytest.raises(ContextLimitError):
            truncate_if_needed(msgs, config, response_token_budget=100_000)

    def test_context_limit_error_attributes(self):
        err = ContextLimitError(
            estimated_tokens=200_000, limit=128_000, model="gpt-4o",
            response_token_budget=4096,
        )
        assert err.estimated_tokens == 200_000
        assert err.limit == 128_000
        assert err.response_budget == 4096
        assert err.overflow == 200_000 + 4096 - 128_000
        assert err.model == "gpt-4o"

    def test_warns_on_overflow_with_pdfs(self):
        # A base64 PDF that we can't extract (no pypdf) should NOT warn/convert
        # but if it fits, no warning. We test the fits path to avoid pypdf.
        msgs = [LLMMessage(role="user", content="small")]
        config = LLMClientConfig(model="gpt-4o", max_tokens=4096)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out, info = truncate_if_needed(msgs, config)
        assert info.input_truncated is False

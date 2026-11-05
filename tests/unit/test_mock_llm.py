"""Unit tests for gdpval_qc.mock_llm — MockLLMClient and helpers.

Tests canned-response resolution order (model,dim)->(model,)->responder->default,
call recording, RecordedCall.dim_name_from_user, make_attempt_json,
make_canned_response, calls_for_model, calls_for_dim.
"""
from __future__ import annotations

import json

import pytest

from gdpval_qc.llm_client import LLMClientConfig, LLMMessage, TextBlock
from gdpval_qc.mock_llm import (
    MockLLMClient, RecordedCall, make_attempt_json, make_canned_response,
)


def _config(model="gpt-5") -> LLMClientConfig:
    return LLMClientConfig(model=model, max_tokens=4096, system="sys")


def _user_msg(text: str) -> LLMMessage:
    return LLMMessage(role="user", content=text)


# ── make_attempt_json / make_canned_response ──────────────────────────────────

class TestMakeAttemptJson:
    def test_defaults(self):
        s = make_attempt_json()
        d = json.loads(s)
        assert d["severity"] == 0
        assert d["confidence"] == "medium"
        assert d["applicable"] is True
        assert d["evidence_quote"] is None
        assert d["headline"] == "no issue"
        assert d["justification"] == "All criteria met."
        assert d["fixability"] == "n/a"

    def test_custom(self):
        s = make_attempt_json(severity=2, confidence="high", applicable=True,
                              evidence_quote="ev", headline="hl",
                              justification="just", fixability="structural")
        d = json.loads(s)
        assert d["severity"] == 2
        assert d["confidence"] == "high"
        assert d["evidence_quote"] == "ev"
        assert d["headline"] == "hl"
        assert d["fixability"] == "structural"


class TestMakeCannedResponse:
    def test_is_alias(self):
        s = make_canned_response(severity=3, confidence="high")
        d = json.loads(s)
        assert d["severity"] == 3
        assert d["confidence"] == "high"

    def test_applicable_param(self):
        s = make_canned_response(applicable=False)
        d = json.loads(s)
        assert d["applicable"] is False


# ── RecordedCall ──────────────────────────────────────────────────────────────

class TestRecordedCall:
    def test_dim_name_from_user(self):
        rc = RecordedCall(model="m", system=None,
                          user="<dimension>input_prompt_leakage</dimension>")
        assert rc.dim_name_from_user == "input_prompt_leakage"

    def test_dim_name_from_user_with_whitespace(self):
        rc = RecordedCall(model="m", system=None,
                          user="<dimension>  spaced  </dimension>")
        assert rc.dim_name_from_user == "spaced"

    def test_dim_name_from_user_none(self):
        rc = RecordedCall(model="m", system=None, user="no dim tag")
        assert rc.dim_name_from_user is None


# ── MockLLMClient resolution order ─────────────────────────────────────────────

class TestMockLLMClientResolution:
    def test_model_dim_key_wins(self):
        client = MockLLMClient(canned={
            ("gpt-5", "dim_a"): "specific",
            ("gpt-5",): "model_only",
        })
        msg = _user_msg("<dimension>dim_a</dimension>")
        result = client.complete([msg], _config("gpt-5"))
        assert result == "specific"

    def test_model_only_key_next(self):
        client = MockLLMClient(canned={
            ("gpt-5", "dim_a"): "specific",
            ("gpt-5",): "model_only",
        })
        msg = _user_msg("<dimension>dim_b</dimension>")
        result = client.complete([msg], _config("gpt-5"))
        assert result == "model_only"

    def test_responder_next(self):
        client = MockLLMClient(
            responder=lambda call: f"responder:{call.model}")
        msg = _user_msg("<dimension>dim_x</dimension>")
        result = client.complete([msg], _config("gpt-5"))
        assert result == "responder:gpt-5"

    def test_default_last(self):
        client = MockLLMClient(default="fallback")
        msg = _user_msg("<dimension>dim_x</dimension>")
        result = client.complete([msg], _config("gpt-5"))
        assert result == "fallback"

    def test_raises_when_nothing_matches(self):
        client = MockLLMClient()
        msg = _user_msg("<dimension>dim_x</dimension>")
        with pytest.raises(RuntimeError, match="no canned response"):
            client.complete([msg], _config("gpt-5"))

    def test_str_key_normalized_to_tuple(self):
        client = MockLLMClient(canned={"gpt-5": "model_str"})
        msg = _user_msg("<dimension>dim_a</dimension>")
        result = client.complete([msg], _config("gpt-5"))
        assert result == "model_str"

    def test_dim_without_model_does_not_match_model_dim(self):
        # When config.model is None, (model, dim) lookup is skipped
        client = MockLLMClient(canned={("gpt-5", "dim_a"): "specific"})
        msg = _user_msg("<dimension>dim_a</dimension>")
        config = LLMClientConfig(model=None, max_tokens=4096)
        with pytest.raises(RuntimeError):
            client.complete([msg], config)

    def test_responder_overrides_default(self):
        client = MockLLMClient(
            responder=lambda call: "from_responder", default="fallback")
        msg = _user_msg("<dimension>dim_x</dimension>")
        result = client.complete([msg], _config("gpt-5"))
        assert result == "from_responder"


# ── Call recording ─────────────────────────────────────────────────────────────

class TestCallRecording:
    def test_calls_recorded(self):
        client = MockLLMClient(default="d")
        client.complete([_user_msg("<dimension>dim_a</dimension>")],
                        _config("gpt-5"))
        client.complete([_user_msg("<dimension>dim_b</dimension>")],
                        _config("gemini-2.5-pro"))
        assert len(client.calls) == 2
        assert client.calls[0].model == "gpt-5"
        assert client.calls[0].dim_name_from_user == "dim_a"
        assert client.calls[1].model == "gemini-2.5-pro"
        assert client.calls[1].dim_name_from_user == "dim_b"

    def test_calls_for_model(self):
        client = MockLLMClient(default="d")
        client.complete([_user_msg("a")], _config("gpt-5"))
        client.complete([_user_msg("b")], _config("gemini"))
        client.complete([_user_msg("c")], _config("gpt-5"))
        gpt_calls = client.calls_for_model("gpt-5")
        assert len(gpt_calls) == 2

    def test_calls_for_dim(self):
        client = MockLLMClient(default="d")
        client.complete([_user_msg("<dimension>dim_a</dimension>")],
                        _config("gpt-5"))
        client.complete([_user_msg("<dimension>dim_b</dimension>")],
                        _config("gpt-5"))
        client.complete([_user_msg("<dimension>dim_a</dimension>")],
                        _config("gpt-5"))
        dim_a_calls = client.calls_for_dim("dim_a")
        assert len(dim_a_calls) == 2


# ── raise_on_call ──────────────────────────────────────────────────────────────

class TestRaiseOnCall:
    def test_raises_on_call(self):
        client = MockLLMClient(raise_on_call=ValueError("API down"))
        with pytest.raises(ValueError, match="API down"):
            client.complete([_user_msg("x")], _config("gpt-5"))


# ── Content blocks ────────────────────────────────────────────────────────────

class TestBlockContent:
    def test_text_block_concatenated(self):
        client = MockLLMClient(default="d")
        block = TextBlock(text="<dimension>dim_block</dimension>")
        client.complete([LLMMessage(role="user", content=[block])],
                        _config("gpt-5"))
        assert client.calls[0].dim_name_from_user == "dim_block"


# ── Invalid canned key ─────────────────────────────────────────────────────────

class TestInvalidCannedKey:
    def test_invalid_key_type_raises(self):
        with pytest.raises(TypeError, match="must be str or tuple"):
            MockLLMClient(canned={123: "v"})  # type: ignore[dict-item]

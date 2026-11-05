"""Unit tests for gdpval_qc.cli pure helpers.

Tests _is_openai_model, _is_gemini_model, _is_anthropic_model (model-name
prefix matching), _select_task_ids, and judge_error_exit_code.

NOTE: importing gdpval_qc.cli calls load_dotenv() at import time — that's
fine, it's a no-op if .env is absent.
"""
from __future__ import annotations

import argparse

import pytest

from gdpval_qc.cli import (
    _is_anthropic_model, _is_gemini_model, _is_openai_model, _select_task_ids,
)


# ── _is_openai_model ─────────────────────────────────────────────────────────

class TestIsOpenaiModel:
    def test_gpt_prefix(self):
        assert _is_openai_model("gpt-4") is True

    def test_gpt_4o(self):
        assert _is_openai_model("gpt-4o") is True

    def test_gpt_5(self):
        assert _is_openai_model("gpt-5") is True

    def test_gpt_5_4(self):
        assert _is_openai_model("gpt-5.4") is True

    def test_o1_prefix(self):
        assert _is_openai_model("o1") is True

    def test_o1_mini(self):
        assert _is_openai_model("o1-mini") is True

    def test_o3_prefix(self):
        assert _is_openai_model("o3") is True

    def test_o3_mini(self):
        assert _is_openai_model("o3-mini") is True

    def test_openai_slash_prefix(self):
        assert _is_openai_model("openai/gpt-5") is True

    def test_gemini_not_openai(self):
        assert _is_openai_model("gemini-1.5-pro") is False

    def test_claude_not_openai(self):
        assert _is_openai_model("claude-3-opus") is False

    def test_case_insensitive(self):
        assert _is_openai_model("GPT-4") is True
        assert _is_openai_model("O1") is True

    def test_empty_string(self):
        assert _is_openai_model("") is False


# ── _is_gemini_model ──────────────────────────────────────────────────────────

class TestIsGeminiModel:
    def test_gemini_prefix(self):
        assert _is_gemini_model("gemini-1.5-pro") is True

    def test_gemini_2(self):
        assert _is_gemini_model("gemini-2.0-flash") is True

    def test_gemini_3(self):
        assert _is_gemini_model("gemini-3.1-pro-preview") is True

    def test_gpt_not_gemini(self):
        assert _is_gemini_model("gpt-4") is False

    def test_claude_not_gemini(self):
        assert _is_gemini_model("claude-3-opus") is False

    def test_case_insensitive(self):
        assert _is_gemini_model("GEMINI-1.5-pro") is True

    def test_empty_string(self):
        assert _is_gemini_model("") is False


# ── _is_anthropic_model ───────────────────────────────────────────────────────

class TestIsAnthropicModel:
    def test_claude_in_name(self):
        assert _is_anthropic_model("claude-3-opus") is True

    def test_claude_opus_4(self):
        assert _is_anthropic_model("claude-opus-4-6") is True

    def test_claude_sonnet(self):
        assert _is_anthropic_model("claude-sonnet-4-5") is True

    def test_claude_haiku(self):
        assert _is_anthropic_model("claude-haiku-4-5") is True

    def test_gpt_not_anthropic(self):
        assert _is_anthropic_model("gpt-4") is False

    def test_gemini_not_anthropic(self):
        assert _is_anthropic_model("gemini-1.5-pro") is False

    def test_case_insensitive(self):
        assert _is_anthropic_model("CLAUDE-3-opus") is True

    def test_empty_string(self):
        assert _is_anthropic_model("") is False

    def test_claude_anywhere(self):
        # "claude" substring match anywhere
        assert _is_anthropic_model("my-claude-model") is True


# ── _select_task_ids ──────────────────────────────────────────────────────────

class TestSelectTaskIds:
    def _args(self, **kw) -> argparse.Namespace:
        defaults = dict(tasks=None, limit=None)
        defaults.update(kw)
        return argparse.Namespace(**defaults)

    def test_no_filter_returns_all(self):
        all_ids = ["task_a", "task_b", "task_c"]
        result = _select_task_ids(all_ids, self._args())
        assert result == all_ids

    def test_limit(self):
        all_ids = ["task_a", "task_b", "task_c"]
        result = _select_task_ids(all_ids, self._args(limit=2))
        assert result == ["task_a", "task_b"]

    def test_tasks_filter(self):
        all_ids = ["task_a", "task_b", "task_c"]
        result = _select_task_ids(all_ids, self._args(tasks="task_a,task_c"))
        assert result == ["task_a", "task_c"]

    def test_tasks_strips_whitespace(self):
        all_ids = ["task_a", "task_b"]
        result = _select_task_ids(all_ids, self._args(tasks=" task_a , task_b "))
        assert result == ["task_a", "task_b"]

    def test_tasks_missing_exits(self):
        all_ids = ["task_a"]
        with pytest.raises(SystemExit) as exc_info:
            _select_task_ids(all_ids, self._args(tasks="task_a,task_x"))
        assert exc_info.value.code == 2

    def test_limit_zero(self):
        all_ids = ["task_a", "task_b"]
        result = _select_task_ids(all_ids, self._args(limit=0))
        assert result == []

    def test_tasks_overrides_limit(self):
        all_ids = ["task_a", "task_b", "task_c"]
        result = _select_task_ids(all_ids, self._args(tasks="task_c", limit=1))
        assert result == ["task_c"]

    def test_empty_tasks_string(self):
        all_ids = ["task_a", "task_b"]
        # An empty string for tasks -> falsy -> no filter applied
        result = _select_task_ids(all_ids, self._args(tasks=""))
        assert result == all_ids


# ── judge_error_exit_code ─────────────────────────────────────────────────────

class TestJudgeErrorExitCode:
    def test_no_errors_returns_0(self):
        from gdpval_qc.cli import judge_error_exit_code
        from gdpval_qc.review import BatchReview
        batch = BatchReview(spec_name="s", spec_version="18.0")
        # collect_judge_errors returns total_errors=0 for empty batch
        assert judge_error_exit_code(batch, 0.5) == 0

    def test_threshold_over_1_disables(self):
        from gdpval_qc.cli import judge_error_exit_code
        from gdpval_qc.review import BatchReview
        batch = BatchReview(spec_name="s", spec_version="18.0")
        # threshold > 1.0 disables the check -> always 0
        assert judge_error_exit_code(batch, 2.0) == 0

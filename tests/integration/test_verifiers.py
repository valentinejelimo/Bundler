"""Integration tests for verifier_lib/verifiers.py verify_deterministic /
verify_rubric in isolation.

Build minimal VerifierDefinition / Source / Assertion pydantic objects and call
the verify functions directly with an injected SourceRegistry / completion_fn.

Scope: area 2 — verify functions in isolation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifier_lib.models import (
    Assertion, Config, DeterministicConfig, FileCommand, RubricConfig,
    Source, VerifierDefinition, VerifierMetadata,
)
from verifier_lib.sources.registry import SourceRegistry
from verifier_lib.verifiers import (
    verify_deterministic, verify_rubric, verify_definition, result_entry,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _metadata(weight: float | None = None) -> VerifierMetadata:
    return VerifierMetadata(
        how_justification="h", why_justification="w", weight=weight,
    )


def _json_source(path: str) -> Source:
    return Source(
        type="file",
        file=FileCommand(type="json", command="read_file",
                         arguments={"path": path}),
    )


def _text_source(path: str) -> Source:
    return Source(
        type="file",
        file=FileCommand(type="text", command="extract_text",
                         arguments={"path": path}),
    )


def _deterministic_def(name: str, source: Source, path: str,
                       comparison, expected, weight: float | None = None) -> VerifierDefinition:
    return VerifierDefinition(
        name=name,
        metadata=_metadata(weight),
        source=source,
        assertion=Assertion(
            type="deterministic",
            expected=expected,
            deterministic=DeterministicConfig(path=path, comparison=comparison),
        ),
    )


def _rubric_def(name: str, source: Source, prompt: str,
                expected: bool, weight: float | None = None) -> VerifierDefinition:
    return VerifierDefinition(
        name=name,
        metadata=_metadata(weight),
        source=source,
        assertion=Assertion(
            type="rubric",
            expected=expected,
            rubric=RubricConfig(prompt=prompt),
        ),
    )


def _registry(tmp_path: Path, workspace: Path) -> SourceRegistry:
    return SourceRegistry(workspace, agent_logs_dir=tmp_path / "logs",
                          max_content_chars=90000)


def _fake_completion(verdict: bool):
    def _completion(**kwargs):
        content = json.dumps({"verdict": verdict, "justification": "fake"})
        return {"choices": [{"message": {"content": content}}]}
    return _completion


# ── verify_deterministic ───────────────────────────────────────────────────

def test_verify_deterministic_pass(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "out.json").write_text(json.dumps({"count": 42}),
                                         encoding="utf-8")
    reg = _registry(tmp_path, workspace)
    d = _deterministic_def("v", _json_source("out.json"), "$.count",
                           "equals", 42)
    entry = verify_deterministic(d, reg, effective_weight=1.0)
    assert entry["name"] == "v"
    assert entry["result"]["success"] is True
    assert entry["result"]["expected"] == 42
    assert entry["result"]["actual"] == 42
    assert entry["result"]["effective_weight"] == 1.0
    assert entry["result"]["error"] is None
    assert entry["result"]["reason"] is None
    assert entry["result"]["evaluation"]["path"] == "$.count"
    assert entry["result"]["evaluation"]["comparison"] == "equals"
    assert entry["result"]["evaluation"]["matched_values"] == [42]


def test_verify_deterministic_fail(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "out.json").write_text(json.dumps({"count": 7}),
                                        encoding="utf-8")
    reg = _registry(tmp_path, workspace)
    d = _deterministic_def("v", _json_source("out.json"), "$.count",
                           "equals", 42)
    entry = verify_deterministic(d, reg, effective_weight=1.0)
    assert entry["result"]["success"] is False
    assert entry["result"]["actual"] == 7
    assert entry["result"]["reason"] == "assertion failed"


def test_verify_deterministic_missing_jsonpath(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "out.json").write_text(json.dumps({"other": 1}),
                                        encoding="utf-8")
    reg = _registry(tmp_path, workspace)
    d = _deterministic_def("v", _json_source("out.json"), "$.count",
                           "equals", 42)
    entry = verify_deterministic(d, reg, effective_weight=1.0)
    assert entry["result"]["success"] is False
    assert entry["result"]["actual"] is None
    assert "required key missing" in entry["result"]["reason"]


def test_verify_deterministic_missing_file(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    reg = _registry(tmp_path, workspace)
    d = _deterministic_def("v", _json_source("missing.json"), "$.count",
                           "equals", 42)
    entry = verify_deterministic(d, reg, effective_weight=1.0)
    assert entry["result"]["success"] is False
    assert entry["result"]["actual"] is None
    assert "missing" in entry["result"]["reason"]


def test_verify_deterministic_weight_handling(tmp_path: Path):
    """effective_weight is the weight passed in, regardless of authored weight."""
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "out.json").write_text(json.dumps({"count": 42}),
                                         encoding="utf-8")
    reg = _registry(tmp_path, workspace)
    d = _deterministic_def("v", _json_source("out.json"), "$.count",
                           "equals", 42, weight=0.25)
    entry = verify_deterministic(d, reg, effective_weight=0.25)
    assert entry["result"]["effective_weight"] == 0.25


def test_verify_deterministic_comparison_contains(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "answer.txt").write_text("hello world", encoding="utf-8")
    reg = _registry(tmp_path, workspace)
    d = _deterministic_def("v", _text_source("answer.txt"), "$.text",
                           "contains", "world")
    entry = verify_deterministic(d, reg, effective_weight=1.0)
    assert entry["result"]["success"] is True
    assert entry["result"]["actual"] == "hello world"


# ── verify_rubric ───────────────────────────────────────────────────────────

def test_verify_rubric_pass(tmp_path: Path):
    """Rubric pass case. BLOCKED without tenacity (undeclared dep)."""
    pytest.importorskip("tenacity", reason=(
        "BLOCKED: tenacity is not installed (undeclared dependency of "
        "verifier_lib/judge.py). Rubric judging cannot run without it."
    ))
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "answer.txt").write_text("yes", encoding="utf-8")
    reg = _registry(tmp_path, workspace)
    config = Config(models=["openai/gpt-5.4"], temperature=0.0)
    d = _rubric_def("r", _text_source("answer.txt"), "say yes?", True)
    entry = verify_rubric(d, reg, effective_weight=1.0, config=config,
                          completion_fn=_fake_completion(True))
    assert entry["result"]["success"] is True
    assert entry["result"]["actual"] is True
    assert entry["result"]["error"] is None
    assert entry["result"]["reason"] is None
    assert entry["result"]["evaluation"]["true_count"] == 1
    assert entry["result"]["evaluation"]["false_count"] == 0
    assert entry["result"]["evaluation"]["judge_models"] == ["openai/gpt-5.4"]


def test_verify_rubric_fail(tmp_path: Path):
    """Rubric fail case. BLOCKED without tenacity (undeclared dep)."""
    pytest.importorskip("tenacity", reason=(
        "BLOCKED: tenacity is not installed (undeclared dependency of "
        "verifier_lib/judge.py). Rubric judging cannot run without it."
    ))
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "answer.txt").write_text("no", encoding="utf-8")
    reg = _registry(tmp_path, workspace)
    config = Config(models=["openai/gpt-5.4"], temperature=0.0)
    d = _rubric_def("r", _text_source("answer.txt"), "say yes?", True)
    entry = verify_rubric(d, reg, effective_weight=1.0, config=config,
                          completion_fn=_fake_completion(False))
    assert entry["result"]["success"] is False
    assert entry["result"]["actual"] is False
    assert entry["result"]["reason"] == "assertion failed"


def test_verify_rubric_missing_config_raises(tmp_path: Path):
    """verify_definition with a rubric and config=None raises RuntimeError."""
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "answer.txt").write_text("yes", encoding="utf-8")
    reg = _registry(tmp_path, workspace)
    d = _rubric_def("r", _text_source("answer.txt"), "say yes?", True)
    with pytest.raises(RuntimeError, match="rubric assertion requires top-level config"):
        verify_definition(d, reg, effective_weight=1.0, config=None,
                          completion_fn=_fake_completion(True))


def test_verify_definition_deterministic_dispatch(tmp_path: Path):
    """verify_definition routes deterministic assertions to verify_deterministic."""
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "out.json").write_text(json.dumps({"count": 42}),
                                         encoding="utf-8")
    reg = _registry(tmp_path, workspace)
    d = _deterministic_def("v", _json_source("out.json"), "$.count",
                           "equals", 42)
    entry = verify_definition(d, reg, effective_weight=1.0, config=None)
    assert entry["result"]["success"] is True


def test_verify_definition_unknown_assertion_raises(tmp_path: Path):
    """An unhandled assertion type raises AssertionError.

    We build the model by bypassing the Assertion validator (construct the
    dict directly) — but the real Assertion model enforces the type literal,
    so a true 'unknown' can't be built via the pydantic model. Instead we
    patch the assertion type on a constructed object to test the dispatch
    guard in verify_definition.
    """
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "out.json").write_text(json.dumps({"count": 42}),
                                         encoding="utf-8")
    reg = _registry(tmp_path, workspace)
    d = _deterministic_def("v", _json_source("out.json"), "$.count",
                           "equals", 42)
    # Force the assertion type to an unrecognized value to exercise the guard
    d.assertion.__dict__["type"] = "mystery"
    with pytest.raises(AssertionError, match="unhandled assertion type"):
        verify_definition(d, reg, effective_weight=1.0, config=None)


# ── result_entry ────────────────────────────────────────────────────────────

def test_result_entry_shape_pass(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    d = _deterministic_def("v", _json_source("x.json"), "$.a", "equals", 1)
    entry = result_entry(
        d, success=True, expected=1, actual=1, effective_weight=0.5,
        error=None, reason=None, evaluation={"path": "$.a"},
        source_data={"a": 1},
    )
    assert entry["name"] == "v"
    assert entry["result"]["success"] is True
    assert entry["result"]["effective_weight"] == 0.5
    assert "source_output_preview" not in entry["result"]["evaluation"]


def test_result_entry_failure_includes_preview(tmp_path: Path):
    """A failing result_entry adds a source_output_preview to evaluation."""
    workspace = tmp_path / "ws"; workspace.mkdir()
    d = _deterministic_def("v", _json_source("x.json"), "$.a", "equals", 1)
    entry = result_entry(
        d, success=False, expected=1, actual=None, effective_weight=0.5,
        error=None, reason="assertion failed",
        evaluation={"path": "$.a"}, source_data={"a": 1},
    )
    assert entry["result"]["success"] is False
    assert "source_output_preview" in entry["result"]["evaluation"]

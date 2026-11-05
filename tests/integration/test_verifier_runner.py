"""Integration tests for templates/harbor/verifier_lib/runner.py run_verifier.

Wires the real run_verifier end-to-end against a temp verifier.json spec and a
temp workspace, injecting a fake llm_completion so no network is used.

Scope: area 1 — verifier runner end-to-end.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifier_lib.runner import run_verifier, load_spec
from verifier_lib.models import VerifierSpec


# ── helpers ─────────────────────────────────────────────────────────────────

def _write_csv(workspace: Path, name: str, rows: list[list[str]]) -> None:
    (workspace / name).write_text(
        "\n".join(",".join(r) for r in rows) + "\n", encoding="utf-8"
    )


def _write_json(workspace: Path, name: str, value: dict) -> None:
    (workspace / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_text(workspace: Path, name: str, text: str) -> None:
    (workspace / name).write_text(text, encoding="utf-8")


def _deterministic_spec(task_id: str = "task-123") -> dict:
    """A spec with two deterministic verifiers reading a JSON + a CSV file."""
    return {
        "task_id": task_id,
        "verifiers": [
            {
                "name": "row_count_is_three",
                "metadata": {
                    "how_justification": "checks row count",
                    "why_justification": "must have 3 rows",
                    "weight": 0.5,
                },
                "source": {
                    "type": "file",
                    "file": {"type": "json", "command": "read_file",
                             "arguments": {"path": "out.json"}},
                },
                "assertion": {
                    "type": "deterministic",
                    "expected": 3,
                    "deterministic": {
                        "path": "$.count",
                        "comparison": "equals",
                    },
                },
            },
            {
                "name": "csv_contains_value",
                "metadata": {
                    "how_justification": "checks csv content",
                    "why_justification": "csv must contain hello",
                    "weight": 0.5,
                },
                "source": {
                    "type": "file",
                    "file": {"type": "csv", "command": "extract_text",
                             "arguments": {"path": "out.csv"}},
                },
                "assertion": {
                    "type": "deterministic",
                    "expected": "hello",
                    "deterministic": {
                        "path": "$.text",
                        "comparison": "contains",
                    },
                },
            },
        ],
    }


def _rubric_spec(task_id: str = "task-rubric") -> dict:
    """A spec with one rubric verifier. Requires top-level config."""
    return {
        "task_id": task_id,
        "config": {
            "models": ["openai/gpt-5.4"],
            "temperature": 0.0,
        },
        "verifiers": [
            {
                "name": "rubric_passes",
                "metadata": {
                    "how_justification": "rubric checks prose",
                    "why_justification": "must be valid",
                },
                "source": {
                    "type": "file",
                    "file": {"type": "text", "command": "extract_text",
                             "arguments": {"path": "answer.txt"}},
                },
                "assertion": {
                    "type": "rubric",
                    "expected": True,
                    "rubric": {"prompt": "Does the answer say 'yes'?"},
                },
            },
        ],
    }


def _fake_verdict_completion(verdict: bool):
    """Build a LiteLLM-compatible completion_fn returning a JSON verdict."""
    def _completion(**kwargs):
        content = json.dumps({
            "verdict": verdict,
            "justification": "fake judge verdict",
        })
        return {"choices": [{"message": {"content": content}}]}

    return _completion


# ── tests ───────────────────────────────────────────────────────────────────

def test_run_verifier_passing_deterministic(tmp_path: Path):
    """A passing deterministic run writes reward.json + verifier-report.json
    with reward == 1.0 and summary counts matching expectations."""
    spec_path = tmp_path / "verifier.json"
    spec_path.write_text(json.dumps(_deterministic_spec()), encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_json(workspace, "out.json", {"count": 3, "items": ["a", "b", "c"]})
    _write_csv(workspace, "out.csv", [["greeting", "body"], ["hello", "world"]])

    verifier_dir = tmp_path / "verifier_out"
    payload = run_verifier(
        spec_path,
        workspace_dir=workspace,
        verifier_dir=verifier_dir,
        agent_logs_dir=tmp_path / "logs",
    )

    # Payload structure
    assert payload["task_id"] == "task-123"
    assert set(payload.keys()) >= {"task_id", "reward", "summary", "assertion_results"}
    # Both pass → reward == 1.0
    assert payload["reward"] == 1.0
    assert payload["summary"]["total_verifiers"] == 2
    assert payload["summary"]["passed"] == 2
    assert payload["summary"]["failed"] == 0
    assert payload["summary"]["errored"] == 0
    assert len(payload["assertion_results"]) == 2

    # Files were written
    report = verifier_dir / "verifier-report.json"
    reward_json = verifier_dir / "reward.json"
    reward_txt = verifier_dir / "reward.txt"
    assert report.is_file()
    assert reward_json.is_file()
    assert reward_txt.is_file()
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["reward"] == 1.0
    assert json.loads(reward_json.read_text(encoding="utf-8")) == {"reward": 1.0}
    assert reward_txt.read_text(encoding="utf-8") == "1.0"

    # Each result entry shape
    for r in payload["assertion_results"]:
        assert "name" in r and "definition" in r and "result" in r
        assert r["result"]["success"] is True
        assert r["result"]["effective_weight"] in (0.5,)


def test_run_verifier_failing_deterministic(tmp_path: Path):
    """A failing deterministic assertion yields reward 0.0 and a fail summary."""
    spec_path = tmp_path / "verifier.json"
    spec_path.write_text(json.dumps(_deterministic_spec()), encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Wrong count → first verifier fails; csv missing 'hello' → second fails too
    _write_json(workspace, "out.json", {"count": 5})
    _write_csv(workspace, "out.csv", [["greeting"], ["bye"]])

    verifier_dir = tmp_path / "verifier_out"
    payload = run_verifier(
        spec_path,
        workspace_dir=workspace,
        verifier_dir=verifier_dir,
        agent_logs_dir=tmp_path / "logs",
    )

    assert payload["reward"] == 0.0
    assert payload["summary"]["passed"] == 0
    assert payload["summary"]["failed"] == 2
    assert all(not r["result"]["success"] for r in payload["assertion_results"])
    # Failing results get a reason and a source_output_preview
    for r in payload["assertion_results"]:
        assert r["result"]["reason"] is not None
        assert "source_output_preview" in r["result"]["evaluation"]


def test_run_verifier_missing_jsonpath(tmp_path: Path):
    """When a jsonpath matches no values, the verifier fails with a
    'required key missing' / 'json path matched no values' reason."""
    spec_path = tmp_path / "verifier.json"
    spec_path.write_text(json.dumps(_deterministic_spec()), encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_json(workspace, "out.json", {"different_key": 3})
    _write_csv(workspace, "out.csv", [["greeting"], ["hello"]])

    verifier_dir = tmp_path / "verifier_out"
    payload = run_verifier(
        spec_path,
        workspace_dir=workspace,
        verifier_dir=verifier_dir,
        agent_logs_dir=tmp_path / "logs",
    )

    row_count_result = next(
        r for r in payload["assertion_results"] if r["name"] == "row_count_is_three"
    )
    assert row_count_result["result"]["success"] is False
    assert "missing" in row_count_result["result"]["reason"]


def test_run_verifier_missing_source_file(tmp_path: Path):
    """A missing source file fails the verifier (not the whole run) with a
    'required file missing' reason."""
    spec_path = tmp_path / "verifier.json"
    spec_path.write_text(json.dumps(_deterministic_spec()), encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # out.json intentionally absent
    _write_csv(workspace, "out.csv", [["greeting"], ["hello"]])

    verifier_dir = tmp_path / "verifier_out"
    payload = run_verifier(
        spec_path,
        workspace_dir=workspace,
        verifier_dir=verifier_dir,
        agent_logs_dir=tmp_path / "logs",
    )

    row_count_result = next(
        r for r in payload["assertion_results"] if r["name"] == "row_count_is_three"
    )
    assert row_count_result["result"]["success"] is False
    assert "missing" in row_count_result["result"]["reason"]


def test_run_verifier_rubric_pass(tmp_path: Path):
    """A rubric verifier with an injected fake completion returns a rubric
    verdict and passes when the judge returns True.

    BLOCKED: verifier_lib/judge.py.evaluate_once_with_retry imports tenacity
    lazily; tenacity is not installed and is not declared in pyproject.toml.
    See FINDINGS in the integration report.
    """
    pytest.importorskip("tenacity", reason=(
        "BLOCKED: tenacity is not installed (undeclared dependency of "
        "verifier_lib/judge.py). Rubric judging cannot run without it."
    ))
    spec_path = tmp_path / "verifier.json"
    spec_path.write_text(json.dumps(_rubric_spec()), encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_text(workspace, "answer.txt", "yes, this is correct")

    verifier_dir = tmp_path / "verifier_out"
    payload = run_verifier(
        spec_path,
        workspace_dir=workspace,
        verifier_dir=verifier_dir,
        agent_logs_dir=tmp_path / "logs",
        llm_completion=_fake_verdict_completion(True),
    )

    assert payload["reward"] == 1.0
    assert payload["summary"]["passed"] == 1
    r = payload["assertion_results"][0]
    assert r["result"]["success"] is True
    assert r["result"]["actual"] is True
    # evaluation carries judge metadata
    assert "judge_models" in r["result"]["evaluation"]
    assert r["result"]["evaluation"]["true_count"] == 1
    assert r["result"]["evaluation"]["false_count"] == 0


def test_run_verifier_rubric_fail(tmp_path: Path):
    """Rubric verifier fails when the judge returns False.

    BLOCKED: same tenacity dependency as test_run_verifier_rubric_pass.
    """
    pytest.importorskip("tenacity", reason=(
        "BLOCKED: tenacity is not installed (undeclared dependency of "
        "verifier_lib/judge.py). Rubric judging cannot run without it."
    ))
    spec_path = tmp_path / "verifier.json"
    spec_path.write_text(json.dumps(_rubric_spec()), encoding="utf-8")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_text(workspace, "answer.txt", "no comment")

    verifier_dir = tmp_path / "verifier_out"
    payload = run_verifier(
        spec_path,
        workspace_dir=workspace,
        verifier_dir=verifier_dir,
        agent_logs_dir=tmp_path / "logs",
        llm_completion=_fake_verdict_completion(False),
    )

    assert payload["reward"] == 0.0
    r = payload["assertion_results"][0]
    assert r["result"]["success"] is False
    assert r["result"]["actual"] is False
    assert r["result"]["evaluation"]["true_count"] == 0
    assert r["result"]["evaluation"]["false_count"] == 1


def test_load_spec_parses(tmp_path: Path):
    """load_spec returns a VerifierSpec model from a valid JSON file."""
    spec_path = tmp_path / "verifier.json"
    spec_path.write_text(json.dumps(_deterministic_spec()), encoding="utf-8")
    spec = load_spec(spec_path)
    assert isinstance(spec, VerifierSpec)
    assert spec.task_id == "task-123"
    assert len(spec.verifiers) == 2


def test_load_spec_invalid_raises(tmp_path: Path):
    """An invalid spec raises ValueError."""
    spec_path = tmp_path / "verifier.json"
    spec_path.write_text(json.dumps({"task_id": "x", "verifiers": []}),
                         encoding="utf-8")
    with pytest.raises(ValueError):
        load_spec(spec_path)

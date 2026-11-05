"""E2E tests for `gdpval_qc.cli --dry-run` — the no-API, no-network engine seam.

`--dry-run` validates the spec + loads samples without any LLM calls and exits
0. We build a minimal corpus matching the svc-noeval spec's field bindings
(solution/golden/, tests/rubric.json, environment/data/) and invoke the engine
as a subprocess via `python -m gdpval_qc.cli --dry-run`.

Per the task brief, this is the main E2E seam for the engine: it needs NO API
keys and NO network. The svc-noeval spec binds solution_files -> solution/golden
(non-optional) and verifier_spec_json -> tests/rubric.json (optional), so the
corpus is shaped accordingly.
"""
from __future__ import annotations

import json

import pytest


def _make_corpus(root, *, task_id: str = "task_1", category: str = "finance",
                  rubric: dict | None = None):
    """Build a minimal corpus matching the svc-noeval field bindings."""
    t = root / task_id
    (t / "environment" / "data").mkdir(parents=True)
    (t / "solution" / "golden").mkdir(parents=True)
    (t / "tests").mkdir(parents=True)
    (t / "instruction.md").write_text(
        f"Analyze the input and produce {task_id} output. Anchor value 2186512.",
        encoding="utf-8",
    )
    (t / "environment" / "data" / "input.txt").write_text(
        "Input anchor 2186512 Q1 revenue.", encoding="utf-8")
    (t / "solution" / "golden" / "output.txt").write_text(
        "Golden output anchor 2186512.", encoding="utf-8")
    (t / "tests" / "rubric.json").write_text(
        json.dumps(rubric or {"criteria": [{"id": "c1", "expected_value": "2186512"}]}),
        encoding="utf-8")
    (t / "task.toml").write_text(
        'version = "1.0"\n[metadata]\n'
        f'task_id = "{task_id}"\ncategory = "{category}"\ndifficulty = "easy"\ntags = []\n',
        encoding="utf-8")
    return root


@pytest.fixture
def dry_corpus(tmp_path):
    return _make_corpus(tmp_path / "corpus")


@pytest.fixture
def dry_out(tmp_path):
    o = tmp_path / "out"
    o.mkdir()
    return o


def test_dry_run_loads_spec_and_samples_exit0(dry_corpus, dry_out, run_cli_module, svc_noeval_spec):
    """--dry-run prints the spec header, the loaded sample summary, and exits 0."""
    res = run_cli_module([
        "--spec", str(svc_noeval_spec),
        "--corpus", str(dry_corpus),
        "--output-dir", str(dry_out),
        "--dry-run",
        "--no-conversion",
    ])
    assert res.returncode == 0, \
        f"dry-run exited {res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # Spec load banner.
    assert "Loaded spec: gdpval_svc_noeval_qc" in res.stdout, f"missing spec banner:\n{res.stdout}"
    assert "v18.6" in res.stdout  # spec version
    # Document conversion disabled banner (because --no-conversion).
    assert "Document conversion: disabled" in res.stdout

    # Corpus + selection lines.
    assert "Corpus has 1 task(s)" in res.stdout
    assert "Selected: 1 task(s)" in res.stdout
    assert "task_1" in res.stdout

    # DRY-RUN sample summary block.
    assert "DRY-RUN: loading samples without any LLM calls" in res.stdout
    assert "task_1:" in res.stdout
    # Per-sample fields printed (cli.py lines 333-338).
    assert "category:        'finance'" in res.stdout
    assert "instruction len:" in res.stdout
    assert "input_files:     1" in res.stdout
    assert "solution_files:  1" in res.stdout
    assert "verifier_spec:   present" in res.stdout

    # Closing line.
    assert "Dry-run complete. Re-run without --dry-run to actually grade." in res.stdout


def test_dry_run_makes_no_llm_calls_and_writes_no_artifacts(dry_corpus, dry_out, run_cli_module, svc_noeval_spec):
    """--dry-run must not write batch artifacts (it returns before the engine run)."""
    res = run_cli_module([
        "--spec", str(svc_noeval_spec),
        "--corpus", str(dry_corpus),
        "--output-dir", str(dry_out),
        "--dry-run",
        "--no-conversion",
    ])
    assert res.returncode == 0
    # No batch_review.json / reviewer_queue.csv written in dry-run.
    assert not (dry_out / "batch_review.json").is_file()
    assert not (dry_out / "reviewer_queue.csv").is_file()


def test_dry_run_multiple_tasks(tmp_path, dry_out, run_cli_module, svc_noeval_spec):
    """A multi-task corpus is fully enumerated by the loader in dry-run."""
    corpus = tmp_path / "corpus"
    _make_corpus(corpus, task_id="task_1")
    _make_corpus(corpus, task_id="task_2", category="audit")
    res = run_cli_module([
        "--spec", str(svc_noeval_spec),
        "--corpus", str(corpus),
        "--output-dir", str(dry_out),
        "--dry-run",
        "--no-conversion",
    ])
    assert res.returncode == 0
    assert "Corpus has 2 task(s)" in res.stdout
    assert "task_1:" in res.stdout and "task_2:" in res.stdout


def test_dry_run_missing_required_directory_fails(tmp_path, dry_out, run_cli_module, svc_noeval_spec):
    """If the corpus task is missing the non-optional solution/golden directory,
    the loader raises FileNotFoundError and dry-run exits non-zero (uncaught on
    the dry-run path)."""
    # Build an EKW-shaped corpus (golden_outputs, not golden) -> svc-noeval
    # requires solution/golden (non-optional) so this must fail to load.
    t = tmp_path / "corpus" / "task_1"
    (t / "environment" / "data").mkdir(parents=True)
    (t / "solution" / "golden_outputs").mkdir(parents=True)  # WRONG name for svc-noeval
    (t / "tests").mkdir(parents=True)
    (t / "instruction.md").write_text("p", encoding="utf-8")
    (t / "environment" / "data" / "input.txt").write_text("x", encoding="utf-8")
    (t / "solution" / "golden_outputs" / "output.txt").write_text("g", encoding="utf-8")
    (t / "tests" / "rubric.json").write_text("{}", encoding="utf-8")
    (t / "task.toml").write_text('version="1.0"\n[metadata]\ncategory="x"\n', encoding="utf-8")

    res = run_cli_module([
        "--spec", str(svc_noeval_spec),
        "--corpus", str(tmp_path / "corpus"),
        "--output-dir", str(dry_out),
        "--dry-run",
        "--no-conversion",
    ])
    # The loader raises FileNotFoundError for the required solution/golden dir.
    assert res.returncode != 0, "expected non-zero exit when required dir missing"
    assert "Required directory missing for field 'solution_files'" in res.stderr or \
           "Required directory missing for field 'solution_files'" in res.stdout


def test_dry_run_no_tasks_selected_exits_2(tmp_path, dry_out, run_cli_module, svc_noeval_spec):
    """An empty corpus -> 0 tasks selected -> cli prints ERROR and exits 2."""
    empty = tmp_path / "empty"
    empty.mkdir()
    res = run_cli_module([
        "--spec", str(svc_noeval_spec),
        "--corpus", str(empty),
        "--output-dir", str(dry_out),
        "--dry-run",
        "--no-conversion",
    ])
    assert res.returncode == 2, f"expected exit 2 for empty corpus, got {res.returncode}\n{res.stdout}\n{res.stderr}"
    assert "no tasks selected" in res.stderr or "no tasks selected" in res.stdout

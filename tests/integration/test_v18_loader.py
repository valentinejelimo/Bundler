"""Integration tests for gdpval_qc/v18_loader.py TaskCorpusLoader.

Builds a temp corpus dir mimicking the bundled task structure and exercises
task_ids / load_sample / has. Uses only .txt/.csv/.json/.md fixtures so no
heavy deps are needed.

Scope: area 4 — TaskCorpusLoader.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gdpval_qc.v18_spec import (
    FieldBinding, V18Spec, JudgeModelConfig, SynthesisConfig, CalibrationConfig,
    ConfidenceDemotion, GateConfig, V18Dimension,
)
from gdpval_qc.v18_loader import TaskCorpusLoader, Sample, FileContent


# ── minimal spec helper ────────────────────────────────────────────────────

def _minimal_spec(field_bindings: list[FieldBinding] | None = None) -> V18Spec:
    """A hand-built V18Spec with the canonical verifier/solution bindings."""
    return V18Spec(
        name="test",
        version="18.0",
        cache_bust=None,
        models=[JudgeModelConfig(model="gpt-5.5")],
        field_types={"solution_files": "document", "verifier_spec_json": "text"},
        field_bindings=field_bindings or _canonical_bindings(),
        synthesis=SynthesisConfig(),
        calibration=CalibrationConfig(),
        severity_scale={0: "none", 1: "minor", 2: "major", 3: "critical"},
        require_evidence_quote=True,
        confidence_demotion=ConfidenceDemotion(),
        verdict_tiers=["reject_structural", "reject_fixable", "needs_review",
                       "accept_with_fixes", "accept"],
        dimensions=[],
    )


def _canonical_bindings() -> list[FieldBinding]:
    # solution_files is optional here so default-field tests can build minimal
    # tasks without a solution dir; requiredness is tested separately by
    # test_required_directory_missing_raises with a non-optional binding.
    return [
        FieldBinding(field_name="solution_files", source="directory",
                     path="solution/golden_outputs", exclude=["Reasoning.docx"],
                     optional=True),
        FieldBinding(field_name="solution_reasoning", source="file",
                     path="solution/golden_outputs/Reasoning.docx", optional=True),
        FieldBinding(field_name="verifier_spec_json", source="file",
                     path="tests/verifier.json", optional=True),
    ]


def _make_task(root: Path, task_id: str, *,
               category: str = "finance",
               level: str | None = None,
               with_verifier: bool = True,
               with_solution: bool = True,
               instruction: str = "Do the task.",
               verifier_json: dict | None = None,
               solution_files: dict[str, str] | None = None) -> Path:
    """Create a minimal task directory mirroring the bundled layout."""
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")

    meta = {"category": category, "difficulty": "hard", "tags": ["t1"],
            "annotatorID": "annotator-1"}
    if level:
        meta["level"] = level
    toml_text = (
        '[metadata]\n'
        f'category = "{meta["category"]}"\n'
        f'difficulty = "{meta["difficulty"]}"\n'
        f'annotatorID = "{meta["annotatorID"]}"\n'
        'tags = ["t1"]\n'
    )
    if level:
        toml_text += f'level = "{level}"\n'
    (task_dir / "task.toml").write_text(toml_text, encoding="utf-8")

    env_dir = task_dir / "environment"; env_dir.mkdir()
    (env_dir / "input.txt").write_text("input data\n", encoding="utf-8")

    if with_solution:
        sol_dir = task_dir / "solution" / "golden_outputs"; sol_dir.mkdir(parents=True)
        files = solution_files or {"answer.txt": "the answer is 42\n"}
        for name, content in files.items():
            (sol_dir / name).write_text(content, encoding="utf-8")

    if with_verifier:
        tests_dir = task_dir / "tests"; tests_dir.mkdir()
        vj = verifier_json or {
            "task_id": task_id,
            "verifiers": [
                {"name": "v1", "metadata": {"how_justification": "h",
                                            "why_justification": "w",
                                            "classification": "core"},
                 "source": {"type": "file",
                            "file": {"type": "text", "command": "extract_text",
                                     "arguments": {"path": "answer.txt"}}},
                 "assertion": {"type": "deterministic", "expected": "42",
                               "deterministic": {"path": "$.text",
                                                 "comparison": "contains"}}}
            ],
        }
        (tests_dir / "verifier.json").write_text(json.dumps(vj), encoding="utf-8")

    return task_dir


# ── task_ids / has ─────────────────────────────────────────────────────────

def test_task_ids_sorted(tmp_path: Path):
    root = tmp_path / "corpus"
    _make_task(root, "task_b")
    _make_task(root, "task_a")
    (root / ".hidden").mkdir()  # should be excluded
    (root / "not_a_dir.txt").write_text("x", encoding="utf-8")
    loader = TaskCorpusLoader(root, _minimal_spec())
    assert loader.task_ids() == ["task_a", "task_b"]


def test_has_detects_present_fields(tmp_path: Path):
    root = tmp_path / "corpus"
    _make_task(root, "task_1")
    loader = TaskCorpusLoader(root, _minimal_spec())
    sample = loader.load_sample("task_1")
    assert sample.has("task_id") is True
    assert sample.has("instruction") is True
    assert sample.has("category") is True
    assert sample.has("verifier_spec_json") is True
    assert sample.has("solution_files") is True


def test_has_treats_absent_as_false(tmp_path: Path):
    root = tmp_path / "corpus"
    _make_task(root, "task_1", with_verifier=False)
    loader = TaskCorpusLoader(root, _minimal_spec())
    sample = loader.load_sample("task_1")
    assert sample.has("verifier_spec_json") is False


def test_load_sample_missing_dir_raises(tmp_path: Path):
    root = tmp_path / "corpus"
    _make_task(root, "task_1")
    loader = TaskCorpusLoader(root, _minimal_spec())
    with pytest.raises(FileNotFoundError):
        loader.load_sample("task_missing")


def test_loader_root_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        TaskCorpusLoader(tmp_path / "nope", _minimal_spec())


# ── default fields ─────────────────────────────────────────────────────────

def test_default_fields_populated(tmp_path: Path):
    root = tmp_path / "corpus"
    _make_task(root, "task_1", category="banking")
    loader = TaskCorpusLoader(root, _minimal_spec())
    s = loader.load_sample("task_1")
    assert s.task_id == "task_1"
    assert s.fields["task_id"] == "task_1"
    assert s.fields["category"] == "banking"
    assert s.fields["difficulty"] == "hard"
    assert s.fields["tags"] == ["t1"]
    assert s.fields["annotatorID"] == "annotator-1"
    assert s.fields["instruction"] == "Do the task."
    assert isinstance(s.fields["input_files"], list)
    assert len(s.fields["input_files"]) == 1
    fc = s.fields["input_files"][0]
    assert isinstance(fc, FileContent)
    assert fc.name == "input.txt"
    assert "input data" in fc.text


def test_category_falls_back_to_domain(tmp_path: Path):
    root = tmp_path / "corpus"
    task_dir = root / "task_1"; task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("x", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        '[metadata]\ndomain = "legacy"\n', encoding="utf-8")
    (task_dir / "environment").mkdir()
    loader = TaskCorpusLoader(root, _minimal_spec())
    s = loader.load_sample("task_1")
    assert s.fields["category"] == "legacy"


def test_empty_category(tmp_path: Path):
    root = tmp_path / "corpus"
    task_dir = root / "task_1"; task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("x", encoding="utf-8")
    (task_dir / "task.toml").write_text("[metadata]\n", encoding="utf-8")
    (task_dir / "environment").mkdir()
    loader = TaskCorpusLoader(root, _minimal_spec())
    s = loader.load_sample("task_1")
    assert s.fields["category"] == ""
    assert s.has("category") is False


# ── field bindings ─────────────────────────────────────────────────────────

def test_solution_files_directory_binding(tmp_path: Path):
    root = tmp_path / "corpus"
    _make_task(root, "task_1", solution_files={
        "answer.txt": "the answer is 42\n", "notes.md": "# notes\n",
    })
    loader = TaskCorpusLoader(root, _minimal_spec())
    s = loader.load_sample("task_1")
    sol = s.fields["solution_files"]
    assert isinstance(sol, list)
    assert len(sol) == 2
    names = sorted(fc.name for fc in sol)
    assert names == ["answer.txt", "notes.md"]
    assert any("42" in fc.text for fc in sol)


def test_solution_files_excludes_reasoning_docx(tmp_path: Path):
    root = tmp_path / "corpus"
    _make_task(root, "task_1", solution_files={
        "answer.txt": "x\n", "Reasoning.docx": "[fake docx]",
    })
    loader = TaskCorpusLoader(root, _minimal_spec())
    s = loader.load_sample("task_1")
    sol = s.fields["solution_files"]
    names = [fc.name for fc in sol]
    assert "Reasoning.docx" not in names
    assert "answer.txt" in names


def test_verifier_spec_json_binding(tmp_path: Path):
    root = tmp_path / "corpus"
    vj = {"task_id": "task_1", "verifiers": []}
    _make_task(root, "task_1", verifier_json=vj)
    loader = TaskCorpusLoader(root, _minimal_spec())
    s = loader.load_sample("task_1")
    assert isinstance(s.fields["verifier_spec_json"], str)
    assert json.loads(s.fields["verifier_spec_json"]) == vj


def test_optional_verifier_missing_returns_none(tmp_path: Path):
    root = tmp_path / "corpus"
    _make_task(root, "task_1", with_verifier=False)
    loader = TaskCorpusLoader(root, _minimal_spec())
    s = loader.load_sample("task_1")
    assert s.fields["verifier_spec_json"] is None


def test_optional_solution_reasoning_missing_returns_none(tmp_path: Path):
    root = tmp_path / "corpus"
    _make_task(root, "task_1", solution_files={"answer.txt": "x\n"})
    loader = TaskCorpusLoader(root, _minimal_spec())
    s = loader.load_sample("task_1")
    assert s.fields["solution_reasoning"] is None


def test_required_directory_missing_raises(tmp_path: Path):
    root = tmp_path / "corpus"
    spec = V18Spec(
        name="t", version="18.0", cache_bust=None,
        models=[JudgeModelConfig(model="gpt-5.5")],
        field_types={},
        field_bindings=[FieldBinding(field_name="solution_files",
                                     source="directory",
                                     path="solution/golden_outputs")],
        synthesis=SynthesisConfig(), calibration=CalibrationConfig(),
        severity_scale={0: "n", 1: "m", 2: "M", 3: "c"},
        require_evidence_quote=True, confidence_demotion=ConfidenceDemotion(),
        verdict_tiers=["accept"], dimensions=[],
    )
    task_dir = root / "task_1"; task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("x", encoding="utf-8")
    (task_dir / "task.toml").write_text("[metadata]\n", encoding="utf-8")
    (task_dir / "environment").mkdir()
    loader = TaskCorpusLoader(root, spec)
    with pytest.raises(FileNotFoundError, match="Required directory missing"):
        loader.load_sample("task_1")


# ── environment excludes ──────────────────────────────────────────────────

def test_environment_excludes_infrastructure_files(tmp_path: Path):
    root = tmp_path / "corpus"
    task_dir = root / "task_1"; task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("x", encoding="utf-8")
    (task_dir / "task.toml").write_text("[metadata]\n", encoding="utf-8")
    env = task_dir / "environment"; env.mkdir()
    (env / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (env / "Dockerfile").write_text("FROM x\n", encoding="utf-8")
    (env / "docker-compose.yaml").write_text("x: 1\n", encoding="utf-8")
    (env / "Makefile").write_text("all:\n", encoding="utf-8")
    loader = TaskCorpusLoader(root, _minimal_spec())
    s = loader.load_sample("task_1")
    names = [fc.name for fc in s.fields["input_files"]]
    assert "data.csv" in names
    assert "Dockerfile" not in names
    assert "docker-compose.yaml" not in names
    assert "Makefile" not in names


# ── load_all ───────────────────────────────────────────────────────────────

def test_load_all(tmp_path: Path):
    root = tmp_path / "corpus"
    _make_task(root, "task_a")
    _make_task(root, "task_b")
    loader = TaskCorpusLoader(root, _minimal_spec())
    samples = loader.load_all()
    assert len(samples) == 2
    assert {s.task_id for s in samples} == {"task_a", "task_b"}

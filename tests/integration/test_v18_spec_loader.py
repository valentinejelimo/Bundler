"""Integration tests for v18_spec.py + v18_loader.py interaction.

Parses the real specs/overlays/svc-noeval.md (which extends svc.md which
extends the base spec), then loads a temp corpus with the loader using that
spec. Asserts the spec's dimensions match what the loader expects.

Scope: area 8 — spec-overlay + loader integration.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gdpval_qc.v18_spec import V18Spec, FieldBinding
from gdpval_qc.v18_loader import TaskCorpusLoader, Sample, FileContent


# ── spec parsing ───────────────────────────────────────────────────────────

def test_svc_noeval_spec_parses():
    """The real svc-noeval.md overlay parses successfully."""
    spec = V18Spec.from_markdown("specs/overlays/svc-noeval.md")
    assert spec.name == "gdpval_svc_noeval_qc"
    assert spec.version.startswith("18")
    assert len(spec.dimensions) > 0
    assert len(spec.models) >= 1
    # verdict tiers present
    assert "reject_structural" in spec.verdict_tiers
    assert "accept" in spec.verdict_tiers


def test_svc_noeval_spec_extends_resolves_dimensions():
    """svc-noeval inherits dimensions from the base spec via svc.md."""
    spec = V18Spec.from_markdown("specs/overlays/svc-noeval.md")
    names = [d.name for d in spec.dimensions]
    # base-spec dimensions inherited
    assert "overall_gdpval_fit" in names
    assert "instruction_input_consistency" in names
    assert "metadata_completeness" in names
    assert "synthetic_marker_scan" in names
    # svc-noeval-specific overrides present (re-gated to task_has_eval_runs)
    construct = spec.get_dimension("verifier_rubric_construct_validity")
    assert construct is not None
    assert construct.applies_when is not None
    assert construct.applies_when.builtin == "task_has_eval_runs"
    coverage = spec.get_dimension("verifier_outcome_coverage")
    assert coverage is not None
    assert coverage.applies_when.builtin == "task_has_eval_runs"


def test_svc_noeval_spec_field_bindings():
    """The overlay inherits the canonical field bindings."""
    spec = V18Spec.from_markdown("specs/overlays/svc-noeval.md")
    binding_names = [fb.field_name for fb in spec.field_bindings]
    assert "solution_files" in binding_names
    assert "verifier_spec_json" in binding_names
    assert "verifier_assertion_summary" in binding_names
    assert "solution_reasoning" in binding_names
    # solution_files is a directory binding. The SVC overlay (svc.md, which
    # svc-noeval.md extends) intentionally remaps this to solution/golden
    # (documented at svc.md:4,17: "golden=solution/golden ... was
    # solution/golden_outputs"). The base spec uses solution/golden_outputs.
    sol = next(fb for fb in spec.field_bindings if fb.field_name == "solution_files")
    assert sol.source == "directory"
    assert sol.path == "solution/golden"


def test_svc_noeval_spec_dimensions_unique():
    """All dimension names are unique after extends-resolution."""
    spec = V18Spec.from_markdown("specs/overlays/svc-noeval.md")
    names = [d.name for d in spec.dimensions]
    assert len(names) == len(set(names))


def test_base_spec_parses():
    spec = V18Spec.from_markdown("specs/gdpval_task_quality_v18.md")
    assert spec.name == "gdpval_task_quality"
    assert len(spec.dimensions) > 20


def test_svc_overlay_parses():
    spec = V18Spec.from_markdown("specs/overlays/svc.md")
    assert spec.name == "gdpval_svc_qc"
    assert len(spec.dimensions) > 0


# ── spec + loader integration ──────────────────────────────────────────────

def _make_task(root: Path, task_id: str, **kwargs) -> Path:
    """Build a task directory matching the SVC overlay's path remapping:
    environment/data, solution/golden, tests/rubric.json."""
    task_dir = root / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(
        kwargs.get("instruction", "Do the task."), encoding="utf-8")
    meta_lines = ["[metadata]"]
    meta_lines.append('category = "finance"')
    meta_lines.append('difficulty = "hard"')
    meta_lines.append('annotatorID = "a1"')
    meta_lines.append('tags = ["t"]')
    if kwargs.get("level"):
        meta_lines.append(f'level = "{kwargs["level"]}"')
    (task_dir / "task.toml").write_text("\n".join(meta_lines) + "\n",
                                         encoding="utf-8")
    # SVC overlay remaps input_files to environment/data
    env_data = task_dir / "environment" / "data"; env_data.mkdir(parents=True)
    (env_data / "input.txt").write_text("input\n", encoding="utf-8")
    # SVC overlay remaps solution_files to solution/golden
    sol = task_dir / "solution" / "golden"; sol.mkdir(parents=True)
    for name, content in kwargs.get("solution_files",
                                     {"answer.txt": "answer 42\n"}).items():
        (sol / name).write_text(content, encoding="utf-8")
    # SVC overlay remaps verifier_spec_json to tests/rubric.json
    tests = task_dir / "tests"; tests.mkdir()
    vj = kwargs.get("verifier_json", {"task_id": task_id, "verifiers": [
        {"name": "v1", "metadata": {"how_justification": "h",
                                    "why_justification": "w",
                                    "classification": "core"},
         "source": {"type": "file",
                    "file": {"type": "text", "command": "extract_text",
                             "arguments": {"path": "answer.txt"}}},
         "assertion": {"type": "deterministic", "expected": "42",
                       "deterministic": {"path": "$.text",
                                         "comparison": "contains"}}}
    ]})
    (tests / "rubric.json").write_text(json.dumps(vj), encoding="utf-8")
    return task_dir


def test_loader_with_real_svc_noeval_spec(tmp_path: Path):
    """A temp corpus loads under the real svc-noeval spec; the spec's
    dimensions' inputs reference field names the loader actually populates."""
    spec = V18Spec.from_markdown("specs/overlays/svc-noeval.md")
    root = tmp_path / "corpus"
    _make_task(root, "task_1")
    loader = TaskCorpusLoader(root, spec)
    assert loader.task_ids() == ["task_1"]
    s = loader.load_sample("task_1")
    assert s.task_id == "task_1"
    # Every dimension's declared inputs must be a field the loader knows about
    # (default fields or spec field_bindings). This is the spec+loader seam.
    known_fields = set(s.fields.keys())
    missing_inputs = []
    for d in spec.dimensions:
        for inp in d.inputs:
            if inp not in known_fields:
                missing_inputs.append((d.name, inp))
    # All dimension inputs should resolve to a populated field (possibly None
    # for optional bindings). If any input references an unknown field name,
    # the spec and loader are out of sync.
    assert not missing_inputs, (
        f"spec dimensions reference unknown fields: {missing_inputs}"
    )


def test_loader_default_fields_present_under_real_spec(tmp_path: Path):
    spec = V18Spec.from_markdown("specs/overlays/svc-noeval.md")
    root = tmp_path / "corpus"
    _make_task(root, "task_1")
    loader = TaskCorpusLoader(root, spec)
    s = loader.load_sample("task_1")
    for f in ("task_id", "category", "difficulty", "tags", "annotatorID",
              "instruction", "input_files"):
        assert f in s.fields, f"missing default field {f}"


def test_programmatic_dims_have_callbacks_registered(tmp_path: Path):
    """Every programmatic dim in the real spec has its callback registered.

    Programmatic dimension callbacks (check_*) are registered on the engine
    via BUILTIN_PROGRAMMATIC_CALLBACKS (v18_engine.py), not on the loader.
    The loader's own .callbacks dict only holds the build_* data builders
    (v18_loader.py:140-147). Assert against the engine registry.
    """
    from gdpval_qc.v18_engine import BUILTIN_PROGRAMMATIC_CALLBACKS

    spec = V18Spec.from_markdown("specs/overlays/svc-noeval.md")
    for d in spec.dimensions:
        if d.check_kind == "programmatic":
            assert d.callback is not None
            assert d.callback in BUILTIN_PROGRAMMATIC_CALLBACKS, (
                f"callback {d.callback} not registered for dim {d.name}"
            )


def test_l1_task_skips_substantive_dims(tmp_path: Path):
    """A task with level=L1 is gated out of task_is_substantive dims."""
    spec = V18Spec.from_markdown("specs/overlays/svc-noeval.md")
    root = tmp_path / "corpus"
    _make_task(root, "task_l1", level="L1")
    loader = TaskCorpusLoader(root, spec)
    s = loader.load_sample("task_l1")
    from gdpval_qc.precondition import task_is_substantive
    assert task_is_substantive(s) is False
    # task_1 (no level) is substantive
    _make_task(root, "task_l3", level="L3")
    s3 = loader.load_sample("task_l3")
    assert task_is_substantive(s3) is True


def test_spec_hash_stable():
    """Parsing the same spec twice yields identical spec_hash."""
    s1 = V18Spec.from_markdown("specs/overlays/svc-noeval.md")
    s2 = V18Spec.from_markdown("specs/overlays/svc-noeval.md")
    assert s1.spec_hash() == s2.spec_hash()

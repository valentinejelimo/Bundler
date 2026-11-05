"""Integration tests for gdpval_qc/v18_callbacks.py programmatic callbacks.

Builds a fake Sample (constructed manually) and V18Dimension and calls each
deterministic callback, asserting the returned DimensionFinding shape and
outcome.

Scope: area 6 — deterministic callbacks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gdpval_qc.v18_callbacks import (
    check_metadata_completeness,
    check_verifier_parseable,
    check_verifier_count_in_range,
    check_verifier_outcome_grader_present,
    check_synthetic_markers,
)
from gdpval_qc.v18_loader import Sample, FileContent
from gdpval_qc.v18_spec import GateConfig, V18Dimension
from gdpval_qc.finding import DimensionFinding


# ── helpers ────────────────────────────────────────────────────────────────

def _dim(name: str = "verifier_spec_parseable", *,
         gate_role: str = "blocking",
         reject_at_severity: int = 3,
         config: dict | None = None) -> V18Dimension:
    return V18Dimension(
        name=name,
        dimension_name=name.replace("_", " ").title(),
        check_kind="programmatic",
        inputs=[],
        gate=GateConfig(role=gate_role, reject_at_severity=reject_at_severity,
                        on_disagreement="review"),
        prose="",
        callback=name,
        config=config or {},
    )


def _sample(task_id: str = "task_1", fields: dict | None = None) -> Sample:
    return Sample(task_id=task_id, task_dir=Path("/tmp/fake"),
                  fields=fields or {})


def _filecontent(name: str, text: str) -> FileContent:
    return FileContent(name=name, path=Path(f"/tmp/{name}"), text=text)


# ── check_metadata_completeness ────────────────────────────────────────────

def test_metadata_completeness_present():
    s = _sample(fields={"category": "finance"})
    d = _dim("metadata_completeness")
    f = check_metadata_completeness(s, d)
    assert isinstance(f, DimensionFinding)
    assert f.dimension == "metadata_completeness"
    assert f.applicable is True
    assert f.severity == 0
    assert f.confidence == "high"
    assert f.judges_agreed is True


def test_metadata_completeness_empty():
    s = _sample(fields={"category": ""})
    d = _dim("metadata_completeness")
    f = check_metadata_completeness(s, d)
    assert f.applicable is True
    assert f.severity == 2
    assert "category is empty" in f.headline
    assert f.fixability == "trivial"


def test_metadata_completeness_missing_key():
    s = _sample(fields={})
    d = _dim("metadata_completeness")
    f = check_metadata_completeness(s, d)
    assert f.severity == 2
    assert "category is empty" in f.headline


# ── check_verifier_parseable ───────────────────────────────────────────────

def test_verifier_parseable_valid():
    vj = {"task_id": "x", "verifiers": [{"name": "v"}]}
    s = _sample(fields={"verifier_spec_json": json.dumps(vj)})
    d = _dim("verifier_spec_parseable")
    f = check_verifier_parseable(s, d)
    assert f.severity == 0


def test_verifier_parseable_malformed_json():
    s = _sample(fields={"verifier_spec_json": "{not json}"})
    d = _dim("verifier_spec_parseable")
    f = check_verifier_parseable(s, d)
    assert f.severity == 2
    assert "verifier.json fails to parse" in f.headline


def test_verifier_parseable_root_not_object():
    s = _sample(fields={"verifier_spec_json": "[1,2,3]"})
    d = _dim("verifier_spec_parseable")
    f = check_verifier_parseable(s, d)
    assert f.severity == 2
    assert "root is not an object" in f.headline


def test_verifier_parseable_missing_verifiers_list():
    s = _sample(fields={"verifier_spec_json": json.dumps({"task_id": "x"})})
    d = _dim("verifier_spec_parseable")
    f = check_verifier_parseable(s, d)
    assert f.severity == 2
    assert "verifiers" in f.headline


def test_verifier_parseable_missing_spec():
    s = _sample(fields={})
    d = _dim("verifier_spec_parseable")
    f = check_verifier_parseable(s, d)
    assert f.severity == 2
    assert "missing" in f.headline


# ── check_verifier_count_in_range ─────────────────────────────────────────

def test_verifier_count_in_range_above_min():
    verifiers = [{"name": f"v{i}"} for i in range(10)]
    vj = {"task_id": "x", "verifiers": verifiers}
    s = _sample(fields={"verifier_spec_json": json.dumps(vj)})
    d = _dim("verifier_count_in_range", config={"min_verifiers": 6})
    f = check_verifier_count_in_range(s, d)
    assert f.severity == 0
    assert "10 verifiers" in f.headline


def test_verifier_count_below_min():
    verifiers = [{"name": f"v{i}"} for i in range(2)]
    vj = {"task_id": "x", "verifiers": verifiers}
    s = _sample(fields={"verifier_spec_json": json.dumps(vj)})
    d = _dim("verifier_count_in_range", config={"min_verifiers": 6})
    f = check_verifier_count_in_range(s, d)
    assert f.severity == 2  # short by 4 → sev 2
    assert "below the minimum" in f.detail


def test_verifier_count_short_by_one_or_two_sev1():
    verifiers = [{"name": f"v{i}"} for i in range(5)]
    vj = {"task_id": "x", "verifiers": verifiers}
    s = _sample(fields={"verifier_spec_json": json.dumps(vj)})
    d = _dim("verifier_count_in_range", config={"min_verifiers": 6})
    f = check_verifier_count_in_range(s, d)
    assert f.severity == 1  # short by 1 → sev 1


def test_verifier_count_missing_spec():
    s = _sample(fields={})
    d = _dim("verifier_count_in_range", config={"min_verifiers": 6})
    f = check_verifier_count_in_range(s, d)
    assert f.severity == 2


def test_verifier_count_unparseable():
    s = _sample(fields={"verifier_spec_json": "not json"})
    d = _dim("verifier_count_in_range", config={"min_verifiers": 6})
    f = check_verifier_count_in_range(s, d)
    assert f.severity == 2
    assert "unparseable" in f.headline


# ── check_verifier_outcome_grader_present ─────────────────────────────────

def test_outcome_grader_present_core_classification():
    vj = {"task_id": "x", "verifiers": [
        {"name": "v1", "metadata": {"classification": "core"}},
        {"name": "v2", "metadata": {}},
    ]}
    s = _sample(fields={"verifier_spec_json": json.dumps(vj)})
    d = _dim("verifier_outcome_grader_present")
    f = check_verifier_outcome_grader_present(s, d)
    assert f.severity == 0
    assert "1 outcome-grading verifier" in f.headline


def test_outcome_grader_present_grading_role_outcome():
    vj = {"task_id": "x", "verifiers": [
        {"name": "v1", "metadata": {"grading_role": "outcome"}},
    ]}
    s = _sample(fields={"verifier_spec_json": json.dumps(vj)})
    d = _dim("verifier_outcome_grader_present")
    f = check_verifier_outcome_grader_present(s, d)
    assert f.severity == 0


def test_outcome_grader_absent():
    vj = {"task_id": "x", "verifiers": [
        {"name": "v1", "metadata": {"classification": "process"}},
    ]}
    s = _sample(fields={"verifier_spec_json": json.dumps(vj)})
    d = _dim("verifier_outcome_grader_present")
    f = check_verifier_outcome_grader_present(s, d)
    assert f.severity == 2
    assert "no outcome-grading verifiers" in f.headline


def test_outcome_grader_missing_spec():
    s = _sample(fields={})
    d = _dim("verifier_outcome_grader_present")
    f = check_verifier_outcome_grader_present(s, d)
    assert f.severity == 2


# ── check_synthetic_markers ───────────────────────────────────────────────

def test_synthetic_markers_clean():
    s = _sample(fields={"input_files": [_filecontent("a.txt", "normal text")]})
    d = _dim("synthetic_marker_scan")
    f = check_synthetic_markers(s, d)
    assert f.severity == 0


def test_synthetic_markers_finds_bracket_tag():
    s = _sample(fields={"input_files": [
        _filecontent("a.txt", "This is a [SYNTHETIC] document."),
    ]})
    d = _dim("synthetic_marker_scan")
    f = check_synthetic_markers(s, d)
    assert f.severity == 2
    assert "synthetic pipeline marker" in f.headline
    assert "a.txt" in f.detail


def test_synthetic_markers_finds_prose_disclosure():
    s = _sample(fields={"input_files": [
        _filecontent("a.txt", "This document is a FABRICATED artifact "
                              "created solely for a financial-analysis "
                              "training case study."),
    ]})
    d = _dim("synthetic_marker_scan")
    f = check_synthetic_markers(s, d)
    assert f.severity == 2


def test_synthetic_markers_no_input_files():
    s = _sample(fields={})
    d = _dim("synthetic_marker_scan")
    f = check_synthetic_markers(s, d)
    assert f.severity == 0
    assert "no input files to scan" in f.headline


def test_synthetic_markers_finding_shape():
    s = _sample(fields={"input_files": [
        _filecontent("a.txt", "[AI-GENERATED] content"),
    ]})
    d = _dim("synthetic_marker_scan")
    f = check_synthetic_markers(s, d)
    assert f.dimension == "synthetic_marker_scan"
    assert f.applicable is True
    assert f.fixability == "fixable"
    assert f.gate_role == "blocking"
    assert f.confidence == "high"


# ── gate_role propagation ──────────────────────────────────────────────────

def test_finding_carries_gate_role():
    """Every callback result carries the dim's gate_role."""
    s = _sample(fields={"category": "x"})
    d = _dim("metadata_completeness", gate_role="informational")
    f = check_metadata_completeness(s, d)
    assert f.gate_role == "informational"

"""Unit tests for gdpval_qc.tools.check_verifier — deterministic verifier checks.

Tests check_parseable, check_count_in_range, check_outcome_grader_present, run.
Feeds JSON strings; asserts finding dicts. 100% stdlib.
"""
from __future__ import annotations

import json

from gdpval_qc.tools.check_verifier import (
    check_count_in_range, check_outcome_grader_present, check_parseable, run,
    GATES,
)


# ── check_parseable ───────────────────────────────────────────────────────────

class TestCheckParseable:
    def test_valid_json(self):
        spec_json = json.dumps({"task_id": "t1", "verifiers": [
            {"name": "v1", "metadata": {}, "source": {}, "assertion": {}}
        ]})
        data, finding = check_parseable(spec_json)
        assert data is not None
        assert data["task_id"] == "t1"
        assert finding["dimension"] == "verifier_spec_parseable"
        assert finding["severity"] == 0
        assert finding["role"] == "blocking"
        assert finding["reject_at_severity"] == 2

    def test_empty_string(self):
        data, finding = check_parseable("")
        assert data is None
        assert finding["severity"] == 2
        assert "missing" in finding["headline"].lower()

    def test_whitespace_only(self):
        data, finding = check_parseable("   ")
        assert data is None
        assert finding["severity"] == 2

    def test_none_input(self):
        data, finding = check_parseable(None)  # type: ignore[arg-type]
        assert data is None
        assert finding["severity"] == 2

    def test_unparseable_json(self):
        data, finding = check_parseable("{not valid json")
        assert data is None
        assert finding["severity"] == 2
        assert "parse" in finding["headline"].lower()

    def test_root_not_object(self):
        data, finding = check_parseable(json.dumps([1, 2, 3]))
        assert data is None
        assert finding["severity"] == 2
        assert "not an object" in finding["headline"].lower()

    def test_missing_verifiers_list(self):
        data, finding = check_parseable(json.dumps({"task_id": "t1"}))
        assert data is None
        assert finding["severity"] == 2
        assert "verifiers" in finding["headline"].lower()

    def test_verifiers_not_list(self):
        data, finding = check_parseable(
            json.dumps({"verifiers": "not a list"}))
        assert data is None
        assert finding["severity"] == 2


# ── check_count_in_range ──────────────────────────────────────────────────────

class TestCheckCountInRange:
    def _data(self, n: int) -> dict:
        return {"verifiers": [{"name": f"v{i}"} for i in range(n)]}

    def test_in_range(self):
        f = check_count_in_range(self._data(10), 6, 20)
        assert f["dimension"] == "verifier_count_in_range"
        assert f["severity"] == 0
        assert f["role"] == "advisory"

    def test_at_min_boundary(self):
        f = check_count_in_range(self._data(6), 6, 20)
        assert f["severity"] == 0

    def test_at_max_boundary(self):
        f = check_count_in_range(self._data(20), 6, 20)
        assert f["severity"] == 0

    def test_below_by_1(self):
        f = check_count_in_range(self._data(5), 6, 20)
        # distance = 1 -> sev 1
        assert f["severity"] == 1

    def test_below_by_3(self):
        f = check_count_in_range(self._data(3), 6, 20)
        # distance = 3 -> sev 2
        assert f["severity"] == 2

    def test_above_by_1(self):
        f = check_count_in_range(self._data(21), 6, 20)
        # distance = 1 -> sev 1
        assert f["severity"] == 1

    def test_above_by_5(self):
        f = check_count_in_range(self._data(25), 6, 20)
        # distance = 5 -> sev 2
        assert f["severity"] == 2

    def test_empty_verifiers(self):
        f = check_count_in_range({"verifiers": []}, 6, 20)
        assert f["severity"] == 2  # distance = 6 -> sev 2

    def test_no_verifiers_key(self):
        f = check_count_in_range({}, 6, 20)
        assert f["severity"] == 2  # count = 0


# ── check_outcome_grader_present ──────────────────────────────────────────────

class TestCheckOutcomeGrader:
    def test_present_legacy_core(self):
        data = {"verifiers": [
            {"metadata": {"classification": "core"}},
            {"metadata": {"classification": "supporting"}},
        ]}
        f = check_outcome_grader_present(data)
        assert f["severity"] == 0
        assert "1 outcome" in f["headline"]

    def test_present_omega_grading_role(self):
        data = {"verifiers": [
            {"metadata": {"grading_role": "outcome"}},
        ]}
        f = check_outcome_grader_present(data)
        assert f["severity"] == 0

    def test_absent(self):
        data = {"verifiers": [
            {"metadata": {"classification": "supporting"}},
            {"metadata": {"grading_role": "process"}},
        ]}
        f = check_outcome_grader_present(data)
        assert f["severity"] == 2
        assert "no outcome" in f["headline"].lower()

    def test_empty_verifiers(self):
        f = check_outcome_grader_present({"verifiers": []})
        assert f["severity"] == 2

    def test_missing_metadata(self):
        data = {"verifiers": [{"name": "v1"}]}
        f = check_outcome_grader_present(data)
        assert f["severity"] == 2

    def test_both_markers_present(self):
        data = {"verifiers": [
            {"metadata": {"classification": "core", "grading_role": "outcome"}},
        ]}
        f = check_outcome_grader_present(data)
        assert f["severity"] == 0


# ── run ───────────────────────────────────────────────────────────────────────

class TestRun:
    def test_valid_spec(self):
        spec_json = json.dumps({
            "task_id": "t1",
            "verifiers": [
                {"name": f"v{i}", "metadata": {"classification": "core"}}
                for i in range(8)
            ],
        })
        findings = run(spec_json, 6, 20)
        assert len(findings) == 3
        assert all(f["severity"] == 0 for f in findings)

    def test_unparseable_returns_three_findings(self):
        findings = run("not json", 6, 20)
        assert len(findings) == 3
        # parseable fails at sev 2, others report unrunnable at sev 2
        assert findings[0]["dimension"] == "verifier_spec_parseable"
        assert findings[0]["severity"] == 2
        assert findings[1]["severity"] == 2  # count
        assert findings[2]["severity"] == 2  # outcome grader

    def test_empty_string(self):
        findings = run("", 6, 20)
        assert len(findings) == 3
        assert findings[0]["severity"] == 2

    def test_finding_shape(self):
        spec_json = json.dumps({
            "task_id": "t1",
            "verifiers": [{"name": "v1", "metadata": {"classification": "core"}}],
        })
        findings = run(spec_json, 6, 20)
        for f in findings:
            assert "dimension" in f
            assert "role" in f
            assert "reject_at_severity" in f
            assert "severity" in f
            assert "confidence" in f
            assert "applicable" in f
            assert "evidence_quote" in f
            assert "headline" in f
            assert "justification" in f
            assert "fixability" in f

    def test_gates_constant(self):
        assert GATES["verifier_spec_parseable"]["role"] == "blocking"
        assert GATES["verifier_count_in_range"]["role"] == "advisory"
        assert GATES["verifier_outcome_grader_present"]["role"] == "advisory"

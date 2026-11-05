"""Unit tests for gdpval_qc.tools.check_rubric — structural rubric checks.

Tests run, check_category_weights_sum, check_gate_present_and_sane,
check_vague_language, infer_level, check_criteria_not_blank, and more.
Tests flat and weighted-category schemas; asserts finding dicts for weight
mismatches, blank criteria, vague language, missing gate. 100% stdlib.
"""
from __future__ import annotations

import json

from gdpval_qc.tools.check_rubric import (
    GATES, LEVEL_MIN, _is_flat_schema, _is_must_pass, _points, _expected_text,
    check_category_weights_sum, check_count_for_level,
    check_criteria_category_valid, check_criteria_not_blank,
    check_gate_present_and_sane, check_parseable,
    check_points_penalty_coherent, check_vague_language, infer_level, run,
)


# ── Schema helpers ─────────────────────────────────────────────────────────────

class TestSchemaHelpers:
    def test_is_flat_schema_no_categories(self):
        assert _is_flat_schema({"criteria": []}) is True

    def test_is_flat_schema_empty_categories(self):
        assert _is_flat_schema({"criteria": [], "categories": []}) is True

    def test_is_flat_schema_with_categories(self):
        assert _is_flat_schema({"criteria": [], "categories": [{"name": "x"}]}) is False

    def test_is_must_pass_bool_true(self):
        assert _is_must_pass({"must_pass": True}) is True

    def test_is_must_pass_bool_false(self):
        assert _is_must_pass({"must_pass": False}) is False

    def test_is_must_pass_string_yes(self):
        assert _is_must_pass({"must_pass": "Yes"}) is True

    def test_is_must_pass_string_no(self):
        assert _is_must_pass({"must_pass": "No"}) is False

    def test_is_must_pass_missing(self):
        assert _is_must_pass({}) is False

    def test_points_number(self):
        assert _points({"points": 10}) == 10.0

    def test_points_string(self):
        assert _points({"points": "8"}) == 8.0

    def test_points_missing(self):
        assert _points({}) == 0.0

    def test_points_negative(self):
        assert _points({"points": -5}) == -5.0

    def test_expected_text_expected_value(self):
        assert _expected_text({"expected_value": "abc"}) == "abc"

    def test_expected_text_expected(self):
        assert _expected_text({"expected": "xyz"}) == "xyz"

    def test_expected_text_missing(self):
        assert _expected_text({}) == ""


# ── infer_level ────────────────────────────────────────────────────────────────

class TestInferLevel:
    def test_l1(self):
        assert infer_level({"task": "svc-something-l1-suffix"}) == "L1"

    def test_l2(self):
        assert infer_level({"task": "svc-something-l2-suffix"}) == "L2"

    def test_l3(self):
        assert infer_level({"task": "svc-something-l3-suffix"}) == "L3"

    def test_unknown(self):
        assert infer_level({"task": "svc-something"}) == "?"

    def test_empty_task(self):
        assert infer_level({"task": ""}) == "?"


# ── check_parseable ───────────────────────────────────────────────────────────

class TestCheckParseable:
    def test_valid(self):
        spec_json = json.dumps({"task": "t-l1", "criteria": []})
        data, f = check_parseable(spec_json)
        assert data is not None
        assert f["severity"] == 0

    def test_empty(self):
        data, f = check_parseable("")
        assert data is None
        assert f["severity"] == 2

    def test_unparseable(self):
        data, f = check_parseable("{bad")
        assert data is None
        assert f["severity"] == 2

    def test_missing_criteria(self):
        data, f = check_parseable(json.dumps({"task": "t"}))
        assert data is None
        assert f["severity"] == 2
        assert "criteria" in f["headline"].lower()


# ── check_category_weights_sum ──────────────────────────────────────────────

class TestCategoryWeightsSum:
    def test_flat_schema_not_applicable(self):
        data = {"criteria": [], "categories": []}
        f = check_category_weights_sum(data)
        assert f["severity"] == 0
        assert "not applicable" in f["headline"].lower()

    def test_weights_sum_to_100(self):
        data = {"categories": [{"name": "a", "weight": 60},
                               {"name": "b", "weight": 40}]}
        f = check_category_weights_sum(data)
        assert f["severity"] == 0

    def test_weights_do_not_sum_to_100(self):
        data = {"categories": [{"name": "a", "weight": 50},
                               {"name": "b", "weight": 30}]}
        f = check_category_weights_sum(data)
        assert f["severity"] == 2
        assert "80" in f["headline"]


# ── check_gate_present_and_sane ───────────────────────────────────────────────

class TestCheckGatePresentAndSane:
    def test_no_gate(self):
        data = {"criteria": [{"id": "c1", "must_pass": False}]}
        f = check_gate_present_and_sane(data, "L2")
        assert f["severity"] == 2
        assert "no must_pass" in f["headline"].lower()

    def test_has_gate(self):
        data = {"criteria": [{"id": "c1", "must_pass": True},
                            {"id": "c2", "must_pass": False}]}
        f = check_gate_present_and_sane(data, "L2")
        assert f["severity"] == 0

    def test_over_gated(self):
        # 4+ criteria, >50% gates -> sev 1
        data = {"criteria": [
            {"id": f"c{i}", "must_pass": True} for i in range(3)
        ] + [{"id": "c3", "must_pass": False}]}
        # 3 gates out of 4 = 75% > 50% -> over-gated
        f = check_gate_present_and_sane(data, "L2")
        assert f["severity"] == 1
        assert "over-gated" in f["headline"].lower()

    def test_l1_single_gate_ok(self):
        data = {"criteria": [{"id": "c1", "must_pass": True}]}
        f = check_gate_present_and_sane(data, "L1")
        assert f["severity"] == 0


# ── check_count_for_level ─────────────────────────────────────────────────────

class TestCheckCountForLevel:
    def test_l1_one_criterion_ok(self):
        data = {"criteria": [{"id": "c1"}]}
        f = check_count_for_level(data, "L1")
        assert f["severity"] == 0

    def test_l1_zero_criteria(self):
        data = {"criteria": []}
        f = check_count_for_level(data, "L1")
        assert f["severity"] == 2

    def test_l2_meets_minimum(self):
        data = {"criteria": [{"id": f"c{i}"} for i in range(6)]}
        f = check_count_for_level(data, "L2")
        assert f["severity"] == 0

    def test_l2_below_minimum_by_1(self):
        data = {"criteria": [{"id": f"c{i}"} for i in range(5)]}
        f = check_count_for_level(data, "L2")
        assert f["severity"] == 1  # dist = 1

    def test_l2_below_minimum_by_3(self):
        data = {"criteria": [{"id": f"c{i}"} for i in range(3)]}
        f = check_count_for_level(data, "L2")
        assert f["severity"] == 2  # dist = 3

    def test_l3_meets_minimum(self):
        data = {"criteria": [{"id": f"c{i}"} for i in range(30)]}
        f = check_count_for_level(data, "L3")
        assert f["severity"] == 0

    def test_unknown_level(self):
        data = {"criteria": [{"id": "c1"}]}
        f = check_count_for_level(data, "?")
        assert f["severity"] == 1
        assert "unknown" in f["headline"].lower()

    def test_level_min_constant(self):
        assert LEVEL_MIN == {"L1": 1, "L2": 6, "L3": 30}


# ── check_criteria_not_blank ──────────────────────────────────────────────────

class TestCheckCriteriaNotBlank:
    def test_all_filled(self):
        data = {"criteria": [
            {"id": "c1", "description": "desc", "expected_value": "ev"},
        ]}
        f = check_criteria_not_blank(data)
        assert f["severity"] == 0

    def test_blank_description(self):
        data = {"criteria": [
            {"id": "c1", "description": "", "expected_value": "ev"},
        ]}
        f = check_criteria_not_blank(data)
        assert f["severity"] == 2

    def test_blank_expected_value(self):
        data = {"criteria": [
            {"id": "c1", "description": "desc", "expected_value": ""},
        ]}
        f = check_criteria_not_blank(data)
        assert f["severity"] == 2

    def test_blank_expected_new_schema(self):
        data = {"criteria": [
            {"id": "c1", "description": "desc", "expected": ""},
        ]}
        f = check_criteria_not_blank(data)
        assert f["severity"] == 2

    def test_whitespace_only_description(self):
        data = {"criteria": [
            {"id": "c1", "description": "   ", "expected_value": "ev"},
        ]}
        f = check_criteria_not_blank(data)
        assert f["severity"] == 2


# ── check_vague_language ──────────────────────────────────────────────────────

class TestCheckVagueLanguage:
    def test_no_vague_terms(self):
        data = {"criteria": [
            {"id": "c1", "description": "Check that the file exists.",
             "expected_value": "True"},
        ]}
        f = check_vague_language(data)
        assert f["severity"] == 0

    def test_vague_in_description(self):
        data = {"criteria": [
            {"id": "c1", "description": "The deck is well-organized.",
             "expected_value": "True"},
        ]}
        f = check_vague_language(data)
        assert f["severity"] == 1
        assert "1 criterion" in f["headline"]

    def test_vague_in_expected_value(self):
        data = {"criteria": [
            {"id": "c1", "description": "Quality check",
             "expected_value": "Professional and polished"},
        ]}
        f = check_vague_language(data)
        assert f["severity"] == 1

    def test_vague_in_category(self):
        data = {"criteria": [
            {"id": "c1", "description": "Check X",
             "category": "branding", "expected_value": "True"},
        ]}
        f = check_vague_language(data)
        assert f["severity"] == 1


# ── check_points_penalty_coherent ──────────────────────────────────────────────

class TestCheckPointsPenaltyCoherent:
    def test_coherent(self):
        data = {"total_points": 100, "criteria": [
            {"id": "c1", "points": 10},
            {"id": "c2", "points": -5, "penalty": True},
        ]}
        f = check_points_penalty_coherent(data)
        assert f["severity"] == 0

    def test_total_not_100(self):
        data = {"total_points": 80, "criteria": [
            {"id": "c1", "points": 10},
        ]}
        f = check_points_penalty_coherent(data)
        assert f["severity"] == 1
        assert "80" in f["headline"]

    def test_penalty_without_negative_points(self):
        data = {"total_points": 100, "criteria": [
            {"id": "c1", "points": 5, "penalty": True},
        ]}
        f = check_points_penalty_coherent(data)
        assert f["severity"] == 1
        assert "penalty" in f["headline"].lower()

    def test_total_none_ok(self):
        data = {"criteria": [{"id": "c1", "points": 10}]}
        f = check_points_penalty_coherent(data)
        assert f["severity"] == 0


# ── check_criteria_category_valid ─────────────────────────────────────────────

class TestCheckCriteriaCategoryValid:
    def test_flat_schema_not_applicable(self):
        data = {"criteria": [{"id": "c1", "category": "anything"}]}
        f = check_criteria_category_valid(data)
        assert f["severity"] == 0

    def test_all_categories_valid(self):
        data = {
            "categories": [{"name": "a"}, {"name": "b"}],
            "criteria": [{"id": "c1", "category": "a"},
                         {"id": "c2", "category": "b"}],
        }
        f = check_criteria_category_valid(data)
        assert f["severity"] == 0

    def test_orphan_category(self):
        data = {
            "categories": [{"name": "a"}],
            "criteria": [{"id": "c1", "category": "a"},
                         {"id": "c2", "category": "xyz"}],
        }
        f = check_criteria_category_valid(data)
        assert f["severity"] == 1
        assert "1 criterion category" in f["headline"]


# ── run ───────────────────────────────────────────────────────────────────────

class TestRun:
    def test_valid_flat_spec(self):
        spec_json = json.dumps({
            "task": "svc-task-l1",
            "criteria": [
                {"id": "c1", "description": "desc", "expected": "True",
                 "must_pass": True, "points": "10"},
            ],
        })
        findings = run(spec_json, None)
        assert len(findings) == 8
        # parseable should pass
        assert findings[0]["dimension"] == "rubric_spec_parseable"
        assert findings[0]["severity"] == 0

    def test_unparseable(self):
        findings = run("not json", None)
        assert len(findings) == 8
        assert all(f["severity"] == 2 for f in findings)

    def test_empty_string(self):
        findings = run("", None)
        assert len(findings) == 8
        assert findings[0]["severity"] == 2

    def test_level_override(self):
        spec_json = json.dumps({
            "task": "svc-task",  # no -lN so level = "?"
            "criteria": [
                {"id": "c1", "description": "d", "expected": "True",
                 "must_pass": True, "points": "10"},
            ],
        })
        findings_inferred = run(spec_json, None)
        findings_l1 = run(spec_json, "L1")
        # With override, count_for_level should pass for L1
        count_inferred = next(f for f in findings_inferred
                              if f["dimension"] == "rubric_count_for_level")
        count_l1 = next(f for f in findings_l1
                        if f["dimension"] == "rubric_count_for_level")
        # Inferred level "?" -> sev 1; L1 -> sev 0
        assert count_inferred["severity"] == 1
        assert count_l1["severity"] == 0

    def test_gates_constant(self):
        assert GATES["rubric_spec_parseable"]["role"] == "blocking"
        assert GATES["rubric_gate_present_and_sane"]["role"] == "blocking"
        assert GATES["rubric_criteria_not_blank"]["role"] == "blocking"
        assert GATES["rubric_vague_language_scan"]["role"] == "advisory"

    def test_finding_shape(self):
        spec_json = json.dumps({
            "task": "svc-task-l2",
            "criteria": [
                {"id": "c1", "description": "d", "expected": "True",
                 "must_pass": True, "points": "10"},
            ],
        })
        findings = run(spec_json, None)
        for f in findings:
            assert "dimension" in f
            assert "role" in f
            assert "reject_at_severity" in f
            assert "severity" in f
            assert "confidence" in f
            assert "applicable" in f
            assert "headline" in f
            assert "justification" in f
            assert "fixability" in f

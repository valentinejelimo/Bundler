"""Unit tests for gdpval_qc.finding — pure dataclasses, zero deps.

Tests to_dict(), is_informational_only(), is_finding(), field defaults,
and Literal types for JudgeAttempt, DimensionFinding, and Issue.
"""
from __future__ import annotations

import dataclasses

from gdpval_qc.finding import JudgeAttempt, DimensionFinding, Issue


# ── JudgeAttempt ─────────────────────────────────────────────────────────────

class TestJudgeAttempt:
    def test_required_fields(self):
        a = JudgeAttempt(
            model="gpt-5",
            severity=2,
            confidence="high",
            applicable=True,
            headline="bad thing",
            justification="because reasons",
            fixability="fixable",
        )
        assert a.model == "gpt-5"
        assert a.severity == 2
        assert a.confidence == "high"
        assert a.applicable is True
        assert a.headline == "bad thing"
        assert a.justification == "because reasons"
        assert a.fixability == "fixable"

    def test_optional_defaults(self):
        a = JudgeAttempt(
            model="m", severity=0, confidence="low",
            applicable=False, headline="x", justification="y", fixability="n/a",
        )
        assert a.evidence_quote is None
        assert a.raw_response == ""
        assert a.error is None

    def test_to_dict_contains_all_fields(self):
        a = JudgeAttempt(
            model="m", severity=1, confidence="medium", applicable=True,
            headline="h", justification="j", fixability="fixable",
            evidence_quote="ev", raw_response="raw", error=None,
        )
        d = a.to_dict()
        assert set(d.keys()) == {
            "model", "severity", "confidence", "applicable",
            "headline", "justification", "fixability",
            "evidence_quote", "raw_response", "error",
        }
        assert d["model"] == "m"
        assert d["severity"] == 1

    def test_to_dict_is_plain_dict(self):
        a = JudgeAttempt(
            model="m", severity=0, confidence="low", applicable=False,
            headline="h", justification="j", fixability="n/a",
        )
        d = a.to_dict()
        assert isinstance(d, dict)
        assert not isinstance(d, JudgeAttempt)


# ── DimensionFinding ──────────────────────────────────────────────────────────

def _make_finding(**kw) -> DimensionFinding:
    defaults = dict(
        dimension="dim_a", dimension_name="Dim A", gate_role="blocking",
        applicable=True, severity=2, fixability="fixable", confidence="high",
        headline="headline", detail="detail", evidence_quote="quote",
        judges_agreed=True,
    )
    defaults.update(kw)
    return DimensionFinding(**defaults)


class TestDimensionFinding:
    def test_default_factory_fields(self):
        f = _make_finding()
        assert f.raw_attempts == []
        assert f.demoted_because == []
        assert f.skipped_because is None
        assert f.needs_review_signal is False

    def test_default_factory_is_per_instance(self):
        f1 = _make_finding()
        f2 = _make_finding()
        f1.raw_attempts.append("x")
        assert f2.raw_attempts == []

    def test_is_informational_only_true(self):
        f = _make_finding(gate_role="informational")
        assert f.is_informational_only() is True

    def test_is_informational_only_false_blocking(self):
        f = _make_finding(gate_role="blocking")
        assert f.is_informational_only() is False

    def test_is_informational_only_false_advisory(self):
        f = _make_finding(gate_role="advisory")
        assert f.is_informational_only() is False

    def test_is_finding_true_sev1(self):
        f = _make_finding(applicable=True, severity=1)
        assert f.is_finding() is True

    def test_is_finding_true_sev3(self):
        f = _make_finding(applicable=True, severity=3)
        assert f.is_finding() is True

    def test_is_finding_false_sev0(self):
        f = _make_finding(applicable=True, severity=0)
        assert f.is_finding() is False

    def test_is_finding_false_not_applicable(self):
        f = _make_finding(applicable=False, severity=2)
        assert f.is_finding() is False

    def test_to_dict_raw_attempts_serialized(self):
        a = JudgeAttempt(
            model="m", severity=1, confidence="high", applicable=True,
            headline="h", justification="j", fixability="fixable",
        )
        f = _make_finding(raw_attempts=[a])
        d = f.to_dict()
        assert isinstance(d["raw_attempts"], list)
        assert len(d["raw_attempts"]) == 1
        assert d["raw_attempts"][0]["model"] == "m"

    def test_to_dict_contains_all_fields(self):
        f = _make_finding()
        d = f.to_dict()
        expected = {
            "dimension", "dimension_name", "gate_role", "applicable",
            "severity", "fixability", "confidence", "headline", "detail",
            "evidence_quote", "judges_agreed", "raw_attempts",
            "demoted_because", "skipped_because", "needs_review_signal",
        }
        assert set(d.keys()) == expected


# ── Issue ─────────────────────────────────────────────────────────────────────

class TestIssue:
    def test_required_fields(self):
        i = Issue(
            rank=1, dimension="dim", gate_role="blocking", severity=2,
            fixability="fixable", headline="h", action="do X",
            evidence_quote="eq", judges_agreed=True,
        )
        assert i.rank == 1
        assert i.dimension == "dim"
        assert i.action == "do X"

    def test_optional_defaults(self):
        i = Issue(
            rank=1, dimension="dim", gate_role="advisory", severity=1,
            fixability="trivial", headline="h", action="a",
            evidence_quote=None, judges_agreed=False,
        )
        assert i.promoted_above_pre_rank is None
        assert i.promotion_reason is None
        assert i.blocking is True
        assert i.classification_reason == ""
        assert i.forced_blocking is False

    def test_to_dict_contains_all_fields(self):
        i = Issue(
            rank=1, dimension="dim", gate_role="blocking", severity=2,
            fixability="structural", headline="h", action="a",
            evidence_quote="eq", judges_agreed=True,
        )
        d = i.to_dict()
        expected = {
            "rank", "dimension", "gate_role", "severity", "fixability",
            "headline", "action", "evidence_quote", "judges_agreed",
            "promoted_above_pre_rank", "promotion_reason",
            "blocking", "classification_reason", "forced_blocking",
        }
        assert set(d.keys()) == expected


# ── Field-count sanity ────────────────────────────────────────────────────────

class TestFieldCounts:
    def test_judge_attempt_field_count(self):
        fields = dataclasses.fields(JudgeAttempt)
        assert len(fields) == 10

    def test_dimension_finding_field_count(self):
        fields = dataclasses.fields(DimensionFinding)
        assert len(fields) == 15

    def test_issue_field_count(self):
        fields = dataclasses.fields(Issue)
        assert len(fields) == 14

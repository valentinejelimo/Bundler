"""Unit tests for gdpval_qc.v18_aggregate — JudgeAttempts → DimensionFinding.

Tests aggregate_attempts, enforce_evidence_quote, make_skipped_finding.
Builds JudgeAttempt lists and asserts aggregation rules: majority
applicability, confidence demotion, anti-overcall (evidence-quote enforcement).
"""
from __future__ import annotations

from gdpval_qc.finding import JudgeAttempt
from gdpval_qc.v18_aggregate import (
    aggregate_attempts, enforce_evidence_quote, make_skipped_finding,
)
from gdpval_qc.v18_spec import (
    AppliesWhen, CalibrationConfig, ConfidenceDemotion, FieldBinding,
    GateConfig, JudgeModelConfig, SynthesisConfig, V18Dimension, V18Spec,
)


# ── Spec / dimension builders ──────────────────────────────────────────────────

def _dim(name="dim_a", role="blocking", reject_at_severity=3) -> V18Dimension:
    return V18Dimension(
        name=name, dimension_name=name, check_kind="llm", inputs=[],
        gate=GateConfig(role=role, reject_at_severity=reject_at_severity),
        prose="",
    )


def _spec(require_evidence_quote=True,
          low_at_sev1="demote_to_0",
          low_at_sev2_or_higher="route_to_review") -> V18Spec:
    return V18Spec(
        name="test_spec", version="18.0", cache_bust=None,
        models=[JudgeModelConfig(model="gpt-5")],
        field_types={}, field_bindings=[],
        synthesis=SynthesisConfig(), calibration=CalibrationConfig(),
        severity_scale={0: "none", 1: "minor", 2: "major", 3: "critical"},
        require_evidence_quote=require_evidence_quote,
        confidence_demotion=ConfidenceDemotion(
            low_at_sev1=low_at_sev1,
            low_at_sev2_or_higher=low_at_sev2_or_higher,
        ),
        verdict_tiers=["reject_structural", "reject_fixable",
                       "needs_review", "accept_with_fixes", "accept"],
        dimensions=[_dim()],
    )


def _attempt(model="m1", severity=0, confidence="high", applicable=True,
             headline="h", justification="j", fixability="n/a",
             evidence_quote=None, error=None) -> JudgeAttempt:
    return JudgeAttempt(
        model=model, severity=severity, confidence=confidence,
        applicable=applicable, headline=headline, justification=justification,
        fixability=fixability, evidence_quote=evidence_quote,
        raw_response="", error=error,
    )


# ── enforce_evidence_quote ────────────────────────────────────────────────────

class TestEnforceEvidenceQuote:
    def test_no_enforcement_when_disabled(self):
        a = _attempt(severity=2, evidence_quote=None)
        out, demoted = enforce_evidence_quote([a], require_evidence_quote=False)
        assert out == [a]
        assert demoted == []

    def test_demotes_sev1_without_quote(self):
        a = _attempt(severity=1, evidence_quote=None)
        out, demoted = enforce_evidence_quote([a], require_evidence_quote=True)
        assert out[0].severity == 0
        assert out[0].fixability == "n/a"
        assert out[0].evidence_quote is None
        assert len(demoted) == 1
        assert "demoted to 0" in demoted[0]

    def test_keeps_sev1_with_quote(self):
        a = _attempt(severity=1, evidence_quote="evidence here")
        out, demoted = enforce_evidence_quote([a], require_evidence_quote=True)
        assert out[0].severity == 1
        assert demoted == []

    def test_demotes_sev2_without_quote(self):
        a = _attempt(severity=2, evidence_quote=None)
        out, demoted = enforce_evidence_quote([a], require_evidence_quote=True)
        assert out[0].severity == 0
        assert len(demoted) == 1

    def test_keeps_sev0_without_quote(self):
        a = _attempt(severity=0, evidence_quote=None)
        out, demoted = enforce_evidence_quote([a], require_evidence_quote=True)
        assert out[0].severity == 0
        assert demoted == []

    def test_does_not_demote_not_applicable(self):
        a = _attempt(severity=2, applicable=False, evidence_quote=None)
        out, demoted = enforce_evidence_quote([a], require_evidence_quote=True)
        assert out[0].severity == 2
        assert demoted == []

    def test_empty_quote_string_treated_as_missing(self):
        a = _attempt(severity=1, evidence_quote="   ")
        out, demoted = enforce_evidence_quote([a], require_evidence_quote=True)
        assert out[0].severity == 0
        assert len(demoted) == 1

    def test_preserves_headline_and_justification(self):
        a = _attempt(severity=2, headline="my headline",
                     justification="my justification", evidence_quote=None)
        out, _ = enforce_evidence_quote([a], require_evidence_quote=True)
        assert out[0].headline == "my headline"
        assert out[0].justification == "my justification"

    def test_input_not_mutated(self):
        a = _attempt(severity=1, evidence_quote=None)
        out, _ = enforce_evidence_quote([a], require_evidence_quote=True)
        assert a.severity == 1  # original unchanged
        assert out[0] is not a  # new object returned


# ── aggregate_attempts: applicability ────────────────────────────────────────

class TestAggregateApplicability:
    def test_majority_not_applicable(self):
        # 2 of 3 say not applicable -> dim not applicable
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", applicable=True),
            _attempt(model="b", applicable=False),
            _attempt(model="c", applicable=False),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.applicable is False
        assert f.severity == 0
        assert f.skipped_because == "not_applicable"

    def test_majority_applicable(self):
        # 3 attempts: 2 yes, 1 no -> 2/3 > 1.5 -> majority applicable
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", applicable=True, severity=1,
                     evidence_quote="ev"),
            _attempt(model="b", applicable=True, severity=0,
                     evidence_quote="ev"),
            _attempt(model="c", applicable=False),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.applicable is True

    def test_all_errors_not_applicable(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", error="timeout"),
            _attempt(model="b", error="rate limit"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.applicable is False

    def test_errors_excluded_from_vote(self):
        # 1 applicable, 1 not applicable, 1 error -> vote is 1 yes / 1 no
        # (error excluded) -> not majority yes -> not applicable
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", applicable=True),
            _attempt(model="b", applicable=False),
            _attempt(model="c", error="err"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.applicable is False


# ── aggregate_attempts: severity and confidence ──────────────────────────────

class TestAggregateSeverityConfidence:
    def test_severity_is_max(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=1, evidence_quote="ev1",
                     confidence="high"),
            _attempt(model="b", severity=3, evidence_quote="ev3",
                     confidence="medium"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.severity == 3

    def test_confidence_is_min(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=0, confidence="high"),
            _attempt(model="b", severity=0, confidence="low"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.confidence == "low"

    def test_headline_from_highest_severity(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=1, headline="low",
                     evidence_quote="ev"),
            _attempt(model="b", severity=3, headline="high",
                     evidence_quote="ev"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.headline == "high"

    def test_evidence_quote_from_highest_severity(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=1, evidence_quote="ev1"),
            _attempt(model="b", severity=2, evidence_quote="ev2"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.evidence_quote == "ev2"

    def test_fixability_from_highest_severity(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=1, fixability="trivial",
                     evidence_quote="ev"),
            _attempt(model="b", severity=3, fixability="structural",
                     evidence_quote="ev"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.fixability == "structural"

    def test_fixability_na_when_sev0(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [_attempt(model="a", severity=0, fixability="fixable")]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.fixability == "n/a"


# ── aggregate_attempts: judges_agreed ─────────────────────────────────────────

class TestJudgesAgreed:
    def test_agreed_within_1(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=1, evidence_quote="ev"),
            _attempt(model="b", severity=2, evidence_quote="ev"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.judges_agreed is True

    def test_disagreed_beyond_1(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=0, evidence_quote=None),
            _attempt(model="b", severity=3, evidence_quote="ev"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.judges_agreed is False

    def test_disagreement_noted_in_detail(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=0, evidence_quote=None),
            _attempt(model="b", severity=3, evidence_quote="ev",
                     justification="base detail"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert "disagreed" in f.detail.lower()


# ── aggregate_attempts: confidence demotion ───────────────────────────────────

class TestConfidenceDemotion:
    def test_low_confidence_sev1_demoted_to_0(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=1, confidence="low",
                     evidence_quote="ev"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.severity == 0
        assert f.headline == "no issue"
        assert f.evidence_quote is None
        assert any("demoted to 0" in m for m in f.demoted_because)

    def test_low_confidence_sev2_review_signal(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=2, confidence="low",
                     evidence_quote="ev"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.severity == 2  # not demoted
        assert f.needs_review_signal is True
        assert any("review signal" in m for m in f.demoted_because)

    def test_high_confidence_sev1_not_demoted(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=1, confidence="high",
                     evidence_quote="ev"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.severity == 1
        assert f.needs_review_signal is False

    def test_low_confidence_sev1_not_demoted_when_disabled(self):
        spec = _spec(low_at_sev1="keep")
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=1, confidence="low",
                     evidence_quote="ev"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        assert f.severity == 1


# ── aggregate_attempts: evidence enforcement integration ─────────────────────

class TestEvidenceEnforcementIntegration:
    def test_sev2_without_quote_demoted_then_sev0(self):
        spec = _spec()
        dim = spec.dimensions[0]
        attempts = [
            _attempt(model="a", severity=2, evidence_quote=None,
                     confidence="high"),
        ]
        f = aggregate_attempts(dim, attempts, spec)
        # Evidence enforcement demotes to sev 0, so aggregate result is sev 0
        assert f.severity == 0
        assert any("no evidence quote" in m for m in f.demoted_because)


# ── make_skipped_finding ───────────────────────────────────────────────────────

class TestMakeSkippedFinding:
    def test_skipped_finding(self):
        dim = _dim(name="skipped_dim", role="advisory")
        f = make_skipped_finding(dim, "precondition not met")
        assert f.applicable is False
        assert f.severity == 0
        assert f.skipped_because == "precondition not met"
        assert "precondition not met" in f.detail
        assert f.raw_attempts == []
        assert f.demoted_because == []
        assert f.needs_review_signal is False

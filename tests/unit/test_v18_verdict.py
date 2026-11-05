"""Unit tests for gdpval_qc.v18_verdict — deterministic verdict computation.

Tests compute_verdict, pre_rank_findings, is_must_fix, split_must_fix,
reject_at_severity_for. Builds DimensionFinding lists and asserts the verdict
decision tree: needs_review -> reject_structural -> reject_fixable -> needs_review
-> accept_with_fixes -> accept.
"""
from __future__ import annotations

from gdpval_qc.finding import DimensionFinding
from gdpval_qc.v18_spec import (
    CalibrationConfig, ConfidenceDemotion, GateConfig, JudgeModelConfig,
    SynthesisConfig, V18Dimension, V18Spec,
)
from gdpval_qc.v18_verdict import (
    compute_verdict, is_must_fix, pre_rank_findings, reject_at_severity_for,
    split_must_fix,
)


# ── Builders ──────────────────────────────────────────────────────────────────

def _dim(name="dim_a", role="blocking", reject_at_severity=3) -> V18Dimension:
    return V18Dimension(
        name=name, dimension_name=name, check_kind="llm", inputs=[],
        gate=GateConfig(role=role, reject_at_severity=reject_at_severity),
        prose="",
    )


def _spec() -> V18Spec:
    return V18Spec(
        name="test", version="18.0", cache_bust=None,
        models=[JudgeModelConfig(model="gpt-5")],
        field_types={}, field_bindings=[],
        synthesis=SynthesisConfig(), calibration=CalibrationConfig(),
        severity_scale={0: "none", 1: "minor", 2: "major", 3: "critical"},
        require_evidence_quote=True,
        confidence_demotion=ConfidenceDemotion(),
        verdict_tiers=["reject_structural", "reject_fixable",
                        "needs_review", "accept_with_fixes", "accept"],
        dimensions=[
            _dim("dim_a", "blocking", 3),
            _dim("dim_b", "blocking", 3),
            _dim("dim_c", "advisory", 3),
            _dim("dim_d", "informational", 3),
        ],
    )


def _finding(dim="dim_a", role="blocking", severity=0, applicable=True,
             fixability="fixable", judges_agreed=True,
             needs_review_signal=False) -> DimensionFinding:
    return DimensionFinding(
        dimension=dim, dimension_name=dim, gate_role=role,
        applicable=applicable, severity=severity, fixability=fixability,
        confidence="high", headline="h", detail="d", evidence_quote=None,
        judges_agreed=judges_agreed, needs_review_signal=needs_review_signal,
    )


# ── reject_at_severity_for ────────────────────────────────────────────────────

class TestRejectAtSeverityFor:
    def test_known_dim(self):
        spec = _spec()
        assert reject_at_severity_for("dim_a", spec) == 3

    def test_unknown_dim_defaults_to_3(self):
        spec = _spec()
        assert reject_at_severity_for("nonexistent", spec) == 3


# ── compute_verdict: decision tree ─────────────────────────────────────────────

class TestComputeVerdictAccept:
    def test_no_findings_accept(self):
        spec = _spec()
        verdict, reason = compute_verdict([], spec)
        assert verdict == "accept"
        assert reason is None

    def test_only_informational_accept(self):
        spec = _spec()
        findings = [_finding(dim="dim_d", role="informational", severity=2)]
        verdict, _ = compute_verdict(findings, spec)
        assert verdict == "accept"

    def test_only_sev0_accept(self):
        spec = _spec()
        findings = [_finding(severity=0)]
        verdict, _ = compute_verdict(findings, spec)
        assert verdict == "accept"

    def test_not_applicable_ignored(self):
        spec = _spec()
        findings = [_finding(applicable=False, severity=3)]
        verdict, _ = compute_verdict(findings, spec)
        assert verdict == "accept"


class TestComputeVerdictNeedsReviewDisagreement:
    def test_disagreement_on_blocking(self):
        spec = _spec()
        findings = [_finding(dim="dim_a", role="blocking",
                             severity=2, judges_agreed=False)]
        verdict, reason = compute_verdict(findings, spec)
        assert verdict == "needs_review"
        assert "disagreement" in reason

    def test_disagreement_priority_over_reject(self):
        # Even if severity is at reject threshold, disagreement -> needs_review
        spec = _spec()
        findings = [_finding(dim="dim_a", role="blocking",
                             severity=3, judges_agreed=False,
                             fixability="structural")]
        verdict, reason = compute_verdict(findings, spec)
        assert verdict == "needs_review"

    def test_disagreement_on_advisory_not_needs_review(self):
        # Disagreement on advisory dim should NOT trigger needs_review
        spec = _spec()
        findings = [_finding(dim="dim_c", role="advisory",
                             severity=2, judges_agreed=False)]
        verdict, _ = compute_verdict(findings, spec)
        # No blocking disagreement, no review signal, no blocking above threshold,
        # advisory sev >= 1 -> accept_with_fixes
        assert verdict == "accept_with_fixes"


class TestComputeVerdictNeedsReviewSignal:
    def test_review_signal_low_confidence(self):
        spec = _spec()
        findings = [_finding(dim="dim_a", role="blocking",
                             severity=2, needs_review_signal=True)]
        verdict, reason = compute_verdict(findings, spec)
        assert verdict == "needs_review"
        assert "low-confidence" in reason

    def test_review_signal_priority_over_reject(self):
        spec = _spec()
        findings = [_finding(dim="dim_a", role="blocking",
                             severity=3, needs_review_signal=True,
                             fixability="structural")]
        verdict, _ = compute_verdict(findings, spec)
        assert verdict == "needs_review"


class TestComputeVerdictReject:
    def test_reject_structural(self):
        spec = _spec()
        findings = [_finding(dim="dim_a", role="blocking",
                             severity=3, fixability="structural",
                             judges_agreed=True)]
        verdict, reason = compute_verdict(findings, spec)
        assert verdict == "reject_structural"
        assert reason is None

    def test_reject_fixable(self):
        spec = _spec()
        findings = [_finding(dim="dim_a", role="blocking",
                             severity=3, fixability="fixable",
                             judges_agreed=True)]
        verdict, reason = compute_verdict(findings, spec)
        assert verdict == "reject_fixable"
        assert reason is None

    def test_reject_structural_wins_over_fixable(self):
        # When both a structural and fixable finding at threshold exist,
        # structural wins
        spec = _spec()
        findings = [
            _finding(dim="dim_a", role="blocking", severity=3,
                     fixability="fixable"),
            _finding(dim="dim_b", role="blocking", severity=3,
                     fixability="structural"),
        ]
        verdict, _ = compute_verdict(findings, spec)
        assert verdict == "reject_structural"

    def test_severity_below_threshold_not_reject(self):
        # reject_at_severity = 3, severity 2 -> not reject
        spec = _spec()
        findings = [_finding(dim="dim_a", role="blocking",
                             severity=2, fixability="structural",
                             judges_agreed=True)]
        verdict, _ = compute_verdict(findings, spec)
        # severity 2 = threshold - 1 = borderline -> needs_review
        assert verdict == "needs_review"


class TestComputeVerdictBorderline:
    def test_borderline_blocking_sev_threshold_minus_1(self):
        # reject_at_severity=3, severity=2 -> borderline -> needs_review
        spec = _spec()
        findings = [_finding(dim="dim_a", role="blocking",
                             severity=2, judges_agreed=True)]
        verdict, reason = compute_verdict(findings, spec)
        assert verdict == "needs_review"
        assert "borderline" in reason

    def test_borderline_sev0_not_borderline(self):
        # severity 0 is not borderline (must be >= 1)
        spec = _spec()
        findings = [_finding(dim="dim_a", role="blocking",
                             severity=0, judges_agreed=True)]
        verdict, _ = compute_verdict(findings, spec)
        assert verdict == "accept"


class TestComputeVerdictAcceptWithFixes:
    def test_blocking_sev1_accept_with_fixes(self):
        spec = _spec()
        findings = [_finding(dim="dim_a", role="blocking",
                             severity=1, judges_agreed=True)]
        verdict, _ = compute_verdict(findings, spec)
        assert verdict == "accept_with_fixes"

    def test_advisory_sev1_accept_with_fixes(self):
        spec = _spec()
        findings = [_finding(dim="dim_c", role="advisory",
                             severity=1, judges_agreed=True)]
        verdict, _ = compute_verdict(findings, spec)
        assert verdict == "accept_with_fixes"

    def test_advisory_sev2_accept_with_fixes(self):
        # Advisory at severity 2: below blocking reject threshold, so
        # accept_with_fixes (not reject)
        spec = _spec()
        findings = [_finding(dim="dim_c", role="advisory",
                             severity=2, judges_agreed=True)]
        verdict, _ = compute_verdict(findings, spec)
        assert verdict == "accept_with_fixes"


# ── pre_rank_findings ─────────────────────────────────────────────────────────

class TestPreRankFindings:
    def test_excludes_informational(self):
        findings = [
            _finding(dim="dim_d", role="informational", severity=2),
            _finding(dim="dim_a", role="blocking", severity=1),
        ]
        ranked = pre_rank_findings(findings)
        assert len(ranked) == 1
        assert ranked[0].dimension == "dim_a"

    def test_excludes_sev0(self):
        findings = [
            _finding(severity=0),
            _finding(dim="dim_b", severity=2),
        ]
        ranked = pre_rank_findings(findings)
        assert len(ranked) == 1

    def test_excludes_not_applicable(self):
        findings = [
            _finding(applicable=False, severity=3),
            _finding(dim="dim_a", severity=1),
        ]
        ranked = pre_rank_findings(findings)
        assert len(ranked) == 1

    def test_blocking_before_advisory(self):
        findings = [
            _finding(dim="dim_c", role="advisory", severity=3),
            _finding(dim="dim_a", role="blocking", severity=1),
        ]
        ranked = pre_rank_findings(findings)
        assert ranked[0].dimension == "dim_a"
        assert ranked[1].dimension == "dim_c"

    def test_higher_severity_first(self):
        findings = [
            _finding(dim="dim_a", severity=1),
            _finding(dim="dim_b", severity=3),
        ]
        ranked = pre_rank_findings(findings)
        assert ranked[0].severity == 3
        assert ranked[1].severity == 1

    def test_structural_before_fixable_same_severity(self):
        findings = [
            _finding(dim="dim_a", severity=2, fixability="fixable"),
            _finding(dim="dim_b", severity=2, fixability="structural"),
        ]
        ranked = pre_rank_findings(findings)
        assert ranked[0].fixability == "structural"
        assert ranked[1].fixability == "fixable"


# ── is_must_fix ────────────────────────────────────────────────────────────────

class TestIsMustFix:
    def test_sev2_blocking_must_fix(self):
        assert is_must_fix(_finding(severity=2, role="blocking")) is True

    def test_sev2_advisory_must_fix(self):
        assert is_must_fix(_finding(severity=2, role="advisory")) is True

    def test_sev1_blocking_must_fix(self):
        assert is_must_fix(_finding(severity=1, role="blocking")) is True

    def test_sev1_advisory_not_must_fix(self):
        assert is_must_fix(_finding(severity=1, role="advisory")) is False

    def test_sev0_not_must_fix(self):
        assert is_must_fix(_finding(severity=0, role="blocking")) is False

    def test_informational_not_must_fix(self):
        assert is_must_fix(_finding(severity=3, role="informational")) is False

    def test_not_applicable_not_must_fix(self):
        assert is_must_fix(
            _finding(severity=3, applicable=False)) is False

    def test_sev3_advisory_must_fix(self):
        assert is_must_fix(_finding(severity=3, role="advisory")) is True


# ── split_must_fix ────────────────────────────────────────────────────────────

class TestSplitMustFix:
    def test_splits_correctly(self):
        findings = [
            _finding(dim="a", role="blocking", severity=2),  # must_fix
            _finding(dim="b", role="advisory", severity=1),  # minor
            _finding(dim="c", role="blocking", severity=3),  # must_fix
            _finding(dim="d", role="informational", severity=2),  # excluded
        ]
        must_fix, minor = split_must_fix(findings)
        assert len(must_fix) == 2
        assert len(minor) == 1
        assert {f.dimension for f in must_fix} == {"a", "c"}
        assert minor[0].dimension == "b"

    def test_empty(self):
        must_fix, minor = split_must_fix([])
        assert must_fix == []
        assert minor == []

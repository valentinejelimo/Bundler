"""Unit tests for gdpval_qc.review — per-sample and per-batch review types.

Tests to_dict(), blocking_findings(), samples_by_verdict(), verdict ordering,
and field defaults for SampleReview, BatchSummary, CalibrationReport,
BatchReview, Verdict, and VERDICT_ORDER.
"""
from __future__ import annotations

import dataclasses

from gdpval_qc.finding import DimensionFinding, Issue
from gdpval_qc.review import (
    BatchReview, BatchSummary, CalibrationReport, SampleReview,
    VERDICT_ORDER, Verdict,
)


# ── VERDICT_ORDER ─────────────────────────────────────────────────────────────

class TestVerdictOrder:
    def test_order_contents(self):
        assert VERDICT_ORDER == [
            "reject_structural", "reject_fixable", "needs_review",
            "accept_with_fixes", "accept",
        ]

    def test_order_has_five_tiers(self):
        assert len(VERDICT_ORDER) == 5

    def test_reject_structural_most_severe(self):
        # reject_structural ranks highest (first)
        assert VERDICT_ORDER[0] == "reject_structural"

    def test_accept_least_severe(self):
        assert VERDICT_ORDER[-1] == "accept"


# ── SampleReview ──────────────────────────────────────────────────────────────

def _make_finding(dim="d", role="blocking", sev=0, applicable=True) -> DimensionFinding:
    return DimensionFinding(
        dimension=dim, dimension_name=dim, gate_role=role,
        applicable=applicable, severity=sev, fixability="fixable",
        confidence="high", headline="h", detail="d", evidence_quote=None,
        judges_agreed=True,
    )


def _make_issue(**kw) -> Issue:
    defaults = dict(
        rank=1, dimension="d", gate_role="blocking", severity=2,
        fixability="fixable", headline="h", action="a",
        evidence_quote=None, judges_agreed=True,
    )
    defaults.update(kw)
    return Issue(**defaults)


class TestSampleReview:
    def test_defaults(self):
        sr = SampleReview(sample_id="s1", verdict="accept", one_line="ok")
        assert sr.sample_id == "s1"
        assert sr.verdict == "accept"
        assert sr.one_line == "ok"
        assert sr.top_issues == []
        assert sr.suggested_fixes == []
        assert sr.review_required_because is None
        assert sr.all_findings == []
        assert sr.synthesizer_fell_back is False
        assert sr.raw_trace_path is None
        assert sr.filtered_issues == []
        assert sr.data_quality_warnings == []

    def test_default_factory_per_instance(self):
        sr1 = SampleReview(sample_id="s1", verdict="accept", one_line="ok")
        sr2 = SampleReview(sample_id="s2", verdict="accept", one_line="ok")
        sr1.top_issues.append(_make_issue())
        assert sr2.top_issues == []

    def test_blocking_findings_returns_only_blocking_sev_ge_1(self):
        f_blocking_sev2 = _make_finding(dim="b2", role="blocking", sev=2)
        f_blocking_sev0 = _make_finding(dim="b0", role="blocking", sev=0)
        f_advisory_sev2 = _make_finding(dim="a2", role="advisory", sev=2)
        f_info_sev2 = _make_finding(dim="i2", role="informational", sev=2)
        f_not_applicable = _make_finding(dim="na", role="blocking", sev=2,
                                         applicable=False)
        sr = SampleReview(
            sample_id="s1", verdict="accept", one_line="ok",
            all_findings=[
                f_blocking_sev2, f_blocking_sev0, f_advisory_sev2,
                f_info_sev2, f_not_applicable,
            ],
        )
        blocking = sr.blocking_findings()
        assert len(blocking) == 1
        assert blocking[0].dimension == "b2"

    def test_blocking_findings_empty_when_none(self):
        sr = SampleReview(sample_id="s1", verdict="accept", one_line="ok")
        assert sr.blocking_findings() == []

    def test_to_dict_serializes_nested(self):
        sr = SampleReview(
            sample_id="s1", verdict="accept_with_fixes", one_line="ok",
            top_issues=[_make_issue(rank=1)],
            filtered_issues=[_make_issue(rank=2)],
            all_findings=[_make_finding(dim="d1", sev=1)],
        )
        d = sr.to_dict()
        assert d["sample_id"] == "s1"
        assert isinstance(d["top_issues"], list)
        assert d["top_issues"][0]["rank"] == 1
        assert isinstance(d["filtered_issues"], list)
        assert d["filtered_issues"][0]["rank"] == 2
        assert isinstance(d["all_findings"], list)
        assert d["all_findings"][0]["dimension"] == "d1"


# ── BatchSummary ───────────────────────────────────────────────────────────────

class TestBatchSummary:
    def test_defaults(self):
        bs = BatchSummary()
        assert bs.verdict_counts == {}
        assert bs.verdict_percentages == {}
        assert bs.failure_mode_histogram == {}
        assert bs.judge_agreement_rate == 1.0
        assert bs.sample_count == 0
        assert bs.samples_with_findings == 0

    def test_to_dict(self):
        bs = BatchSummary(sample_count=5, samples_with_findings=2,
                          judge_agreement_rate=0.8)
        d = bs.to_dict()
        assert d["sample_count"] == 5
        assert d["samples_with_findings"] == 2
        assert d["judge_agreement_rate"] == 0.8


# ── CalibrationReport ──────────────────────────────────────────────────────────

class TestCalibrationReport:
    def test_defaults(self):
        cr = CalibrationReport()
        assert cr.enabled is True
        assert cr.goldens_count == 0
        assert cr.drift == []
        assert cr.warning is None

    def test_to_dict(self):
        cr = CalibrationReport(enabled=False, goldens_count=3,
                               drift=[{"sample_id": "s1"}],
                               warning="drift detected")
        d = cr.to_dict()
        assert d["enabled"] is False
        assert d["goldens_count"] == 3
        assert d["drift"] == [{"sample_id": "s1"}]
        assert d["warning"] == "drift detected"


# ── BatchReview ───────────────────────────────────────────────────────────────

class TestBatchReview:
    def test_defaults(self):
        br = BatchReview(spec_name="spec", spec_version="18.6")
        assert br.spec_name == "spec"
        assert br.spec_version == "18.6"
        assert br.samples == []
        assert isinstance(br.batch_summary, BatchSummary)
        assert br.calibration is None

    def test_samples_by_verdict(self):
        s1 = SampleReview(sample_id="s1", verdict="accept", one_line="ok")
        s2 = SampleReview(sample_id="s2", verdict="reject_fixable", one_line="bad")
        s3 = SampleReview(sample_id="s3", verdict="accept", one_line="ok")
        br = BatchReview(spec_name="spec", spec_version="18.6",
                         samples=[s1, s2, s3])
        accepted = br.samples_by_verdict("accept")
        assert len(accepted) == 2
        assert {s.sample_id for s in accepted} == {"s1", "s3"}
        rejected = br.samples_by_verdict("reject_fixable")
        assert len(rejected) == 1
        assert rejected[0].sample_id == "s2"
        empty = br.samples_by_verdict("reject_structural")
        assert empty == []

    def test_to_dict_serializes(self):
        s1 = SampleReview(sample_id="s1", verdict="accept", one_line="ok")
        cr = CalibrationReport(enabled=True, goldens_count=2)
        br = BatchReview(spec_name="spec", spec_version="18.6",
                         samples=[s1], calibration=cr)
        d = br.to_dict()
        assert d["spec_name"] == "spec"
        assert d["spec_version"] == "18.6"
        assert isinstance(d["samples"], list)
        assert d["samples"][0]["sample_id"] == "s1"
        assert isinstance(d["batch_summary"], dict)
        assert d["calibration"]["goldens_count"] == 2

    def test_to_dict_calibration_none(self):
        br = BatchReview(spec_name="spec", spec_version="18.6")
        d = br.to_dict()
        assert d["calibration"] is None


# ── Field count sanity ─────────────────────────────────────────────────────────

class TestFieldCounts:
    def test_sample_review_field_count(self):
        assert len(dataclasses.fields(SampleReview)) == 11

    def test_batch_summary_field_count(self):
        assert len(dataclasses.fields(BatchSummary)) == 6

    def test_calibration_report_field_count(self):
        assert len(dataclasses.fields(CalibrationReport)) == 4

    def test_batch_review_field_count(self):
        assert len(dataclasses.fields(BatchReview)) == 5

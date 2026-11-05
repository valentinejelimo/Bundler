"""Integration tests for gdpval_qc/v18_emitter.py.

Builds minimal BatchReview / SampleReview / DimensionFinding / Issue objects by
hand and serializes them via to_json, to_csv, to_markdown_summary, and
write_batch_artifacts.

Scope: area 7 — emitters.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from gdpval_qc.v18_emitter import (
    to_json, to_csv, to_markdown_summary, write_batch_artifacts,
    collect_judge_errors,
)
from gdpval_qc.finding import DimensionFinding, Issue, JudgeAttempt
from gdpval_qc.review import BatchReview, BatchSummary, SampleReview


# ── builders ──────────────────────────────────────────────────────────────

def _finding(dimension: str = "metadata_completeness", *,
             severity: int = 2, applicable: bool = True,
             gate_role: str = "blocking", fixability: str = "fixable",
             headline: str = "issue", detail: str = "detail",
             evidence_quote: str | None = "evidence",
             judges_agreed: bool = True,
             raw_attempts: list[JudgeAttempt] | None = None) -> DimensionFinding:
    return DimensionFinding(
        dimension=dimension,
        dimension_name=dimension.replace("_", " ").title(),
        gate_role=gate_role,
        applicable=applicable,
        severity=severity,
        fixability=fixability,
        confidence="high",
        headline=headline,
        detail=detail,
        evidence_quote=evidence_quote,
        judges_agreed=judges_agreed,
        raw_attempts=raw_attempts or [],
    )


def _attempt(model: str = "gpt-5.5", *, severity: int = 2,
              error: str | None = None, applicable: bool = True) -> JudgeAttempt:
    return JudgeAttempt(
        model=model, severity=severity, confidence="high",
        applicable=applicable, headline="h", justification="j",
        fixability="fixable", evidence_quote="e",
        raw_response="raw", error=error,
    )


def _issue(dimension: str = "metadata_completeness", *,
           rank: int = 1, severity: int = 2, fixability: str = "fixable",
           headline: str = "issue", action: str = "fix it",
           evidence_quote: str | None = "ev",
           judges_agreed: bool = True, blocking: bool = True,
           classification_reason: str = "reason") -> Issue:
    return Issue(
        rank=rank, dimension=dimension, gate_role="blocking",
        severity=severity, fixability=fixability, headline=headline,
        action=action, evidence_quote=evidence_quote,
        judges_agreed=judges_agreed, blocking=blocking,
        classification_reason=classification_reason,
    )


def _sample_review(sample_id: str = "task_1", *,
                   verdict: str = "reject_fixable",
                   one_line: str = "has issues",
                   top_issues: list[Issue] | None = None,
                   filtered_issues: list[Issue] | None = None,
                   all_findings: list[DimensionFinding] | None = None,
                   review_required_because: str | None = None,
                   raw_trace_path: str | None = "raw/sample_task_1.json",
                   data_quality_warnings: list[str] | None = None) -> SampleReview:
    return SampleReview(
        sample_id=sample_id,
        verdict=verdict,
        one_line=one_line,
        top_issues=top_issues or [],
        filtered_issues=filtered_issues or [],
        suggested_fixes=[],
        review_required_because=review_required_because,
        all_findings=all_findings or [],
        synthesizer_fell_back=False,
        raw_trace_path=raw_trace_path,
        data_quality_warnings=data_quality_warnings or [],
    )


def _batch_review(samples: list[SampleReview]) -> BatchReview:
    return BatchReview(
        spec_name="test_spec",
        spec_version="18.0",
        samples=samples,
        batch_summary=BatchSummary(
            verdict_counts={"reject_fixable": 1},
            verdict_percentages={"reject_fixable": 100.0},
            failure_mode_histogram={"metadata_completeness": 1},
            judge_agreement_rate=1.0,
            sample_count=1,
            samples_with_findings=1,
        ),
        calibration=None,
    )


# ── to_json ────────────────────────────────────────────────────────────────

def test_to_json_structure():
    f = _finding()
    r = _sample_review(all_findings=[f], top_issues=[_issue()])
    br = _batch_review([r])
    d = to_json(br)
    assert d["spec_name"] == "test_spec"
    assert d["spec_version"] == "18.0"
    assert len(d["samples"]) == 1
    assert d["samples"][0]["sample_id"] == "task_1"
    assert d["samples"][0]["verdict"] == "reject_fixable"
    assert d["batch_summary"]["sample_count"] == 1
    assert d["calibration"] is None
    # JSON-serializable
    json.dumps(d)


def test_to_json_finding_serialized():
    f = _finding(raw_attempts=[_attempt()])
    r = _sample_review(all_findings=[f])
    br = _batch_review([r])
    d = to_json(br)
    raw = d["samples"][0]["all_findings"][0]["raw_attempts"]
    assert len(raw) == 1
    assert raw[0]["model"] == "gpt-5.5"


# ── to_csv ─────────────────────────────────────────────────────────────────

def test_to_csv_header_and_rows():
    f = _finding()
    r = _sample_review(all_findings=[f], top_issues=[_issue()])
    br = _batch_review([r])
    out = to_csv(br)
    reader = csv.reader(io.StringIO(out))
    rows = list(reader)
    header = rows[0]
    assert "sample_id" in header
    assert "verdict" in header
    assert "dimension" in header
    assert "severity" in header
    assert "data_quality_warnings" in header
    # one data row for the blocking issue
    assert rows[1][0] == "task_1"
    assert rows[1][1] == "reject_fixable"
    assert "metadata_completeness" in out


def test_to_csv_no_issue_sample_has_blank_row():
    r = _sample_review(verdict="accept", one_line="no issues",
                       top_issues=[], all_findings=[])
    br = BatchReview(
        spec_name="t", spec_version="18.0", samples=[r],
        batch_summary=BatchSummary(
            verdict_counts={"accept": 1},
            verdict_percentages={"accept": 100.0},
            failure_mode_histogram={},
            judge_agreement_rate=1.0,
            sample_count=1, samples_with_findings=0,
        ),
    )
    out = to_csv(br)
    reader = csv.reader(io.StringIO(out))
    rows = list(reader)
    # header + 1 row
    assert len(rows) == 2
    assert rows[1][0] == "task_1"


def test_to_csv_orders_by_verdict_urgency():
    """needs_review (0) sorts before accept (4)."""
    r_accept = _sample_review("accept_id", verdict="accept", one_line="",
                              top_issues=[], all_findings=[])
    r_review = _sample_review("review_id", verdict="needs_review",
                              one_line="", top_issues=[], all_findings=[])
    br = BatchReview(
        spec_name="t", spec_version="18.0",
        samples=[r_accept, r_review],
        batch_summary=BatchSummary(
            verdict_counts={"accept": 1, "needs_review": 1},
            verdict_percentages={"accept": 50.0, "needs_review": 50.0},
            failure_mode_histogram={}, judge_agreement_rate=1.0,
            sample_count=2, samples_with_findings=0,
        ),
    )
    out = to_csv(br)
    reader = csv.reader(io.StringIO(out))
    rows = list(reader)
    # header + 2 rows; review_id first
    assert rows[1][0] == "review_id"
    assert rows[2][0] == "accept_id"


def test_to_csv_data_quality_warnings_column():
    r = _sample_review("task_1", data_quality_warnings=["file x failed"],
                       top_issues=[], all_findings=[])
    br = BatchReview(
        spec_name="t", spec_version="18.0", samples=[r],
        batch_summary=BatchSummary(
            verdict_counts={"accept": 1},
            verdict_percentages={"accept": 100.0},
            failure_mode_histogram={}, judge_agreement_rate=1.0,
            sample_count=1, samples_with_findings=0,
        ),
    )
    out = to_csv(br)
    assert "file x failed" in out


# ── to_markdown_summary ────────────────────────────────────────────────────

def test_to_markdown_summary_basic():
    f = _finding()
    r = _sample_review(all_findings=[f], top_issues=[_issue()])
    br = _batch_review([r])
    md = to_markdown_summary(br)
    assert "# Batch summary" in md
    assert "test_spec" in md
    assert "reject_fixable" in md
    assert "Verdict distribution" in md


def test_to_markdown_summary_includes_judge_errors():
    """When a JudgeAttempt has error set, the markdown surfaces the error."""
    f = _finding(raw_attempts=[_attempt(error="boom")])
    r = _sample_review(all_findings=[f])
    br = _batch_review([r])
    md = to_markdown_summary(br)
    assert "Judge-call errors" in md
    assert "boom" in md


def test_to_markdown_summary_no_judge_errors_section():
    f = _finding(raw_attempts=[_attempt()])
    r = _sample_review(all_findings=[f])
    br = _batch_review([r])
    md = to_markdown_summary(br)
    assert "Judge-call errors" not in md


def test_to_markdown_summary_data_quality_warnings():
    r = _sample_review("task_1", data_quality_warnings=["conv failed"],
                       top_issues=[], all_findings=[])
    br = BatchReview(
        spec_name="t", spec_version="18.0", samples=[r],
        batch_summary=BatchSummary(
            verdict_counts={"accept": 1},
            verdict_percentages={"accept": 100.0},
            failure_mode_histogram={}, judge_agreement_rate=1.0,
            sample_count=1, samples_with_findings=0,
        ),
    )
    md = to_markdown_summary(br)
    assert "Data-quality warnings" in md
    assert "conv failed" in md


# ── collect_judge_errors ───────────────────────────────────────────────────

def test_collect_judge_errors_no_errors():
    f = _finding(raw_attempts=[_attempt(), _attempt()])
    r = _sample_review(all_findings=[f])
    br = _batch_review([r])
    errs = collect_judge_errors(br)
    assert errs["total_attempts"] == 2
    assert errs["total_errors"] == 0
    assert errs["by_sample"] == {}
    assert errs["by_reason"] == {}


def test_collect_judge_errors_with_errors():
    f = _finding(raw_attempts=[_attempt(), _attempt(error="HTTP 500"),
                               _attempt(model="gemini", error="timeout")])
    r = _sample_review(all_findings=[f])
    br = _batch_review([r])
    errs = collect_judge_errors(br)
    assert errs["total_attempts"] == 3
    assert errs["total_errors"] == 2
    assert "task_1" in errs["by_sample"]
    assert errs["by_sample"]["task_1"]["errors"] == 2
    assert errs["by_sample"]["task_1"]["attempts"] == 3
    assert errs["by_reason"]["HTTP 500"] == 1
    assert errs["by_reason"]["timeout"] == 1


# ── write_batch_artifacts ─────────────────────────────────────────────────

def test_write_batch_artifacts_writes_files(tmp_path: Path):
    f = _finding()
    r = _sample_review(all_findings=[f], top_issues=[_issue()])
    br = _batch_review([r])
    out_dir = tmp_path / "out"
    paths = write_batch_artifacts(br, out_dir, top_issues_cap=3)
    assert paths["batch_review.json"].is_file()
    assert paths["reviewer_queue.csv"].is_file()
    assert paths["batch_summary.md"].is_file()
    # raw per-sample transcript
    raw = out_dir / "raw" / "sample_task_1.json"
    assert raw.is_file()
    written = json.loads(paths["batch_review.json"].read_text(encoding="utf-8"))
    assert written["spec_name"] == "test_spec"
    raw_payload = json.loads(raw.read_text(encoding="utf-8"))
    assert raw_payload["sample_id"] == "task_1"
    assert raw_payload["verdict"] == "reject_fixable"
    assert len(raw_payload["findings"]) == 1


def test_write_batch_artifacts_creates_dir(tmp_path: Path):
    r = _sample_review(verdict="accept", one_line="ok",
                       top_issues=[], all_findings=[])
    br = BatchReview(
        spec_name="t", spec_version="18.0", samples=[r],
        batch_summary=BatchSummary(
            verdict_counts={"accept": 1},
            verdict_percentages={"accept": 100.0},
            failure_mode_histogram={}, judge_agreement_rate=1.0,
            sample_count=1, samples_with_findings=0,
        ),
    )
    out_dir = tmp_path / "nested" / "out"
    write_batch_artifacts(br, out_dir)
    assert out_dir.is_dir()
    assert (out_dir / "batch_review.json").is_file()

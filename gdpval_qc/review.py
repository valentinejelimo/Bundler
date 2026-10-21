"""Per-sample and per-batch review types for the v18 quality framework.

A `SampleReview` is what the synthesizer produces per sample: the verdict
(computed deterministically before the synthesizer runs), a one-line summary,
the top-K ranked issues, suggested fixes, and the full set of findings for
audit drill-down.

A `BatchReview` is the framework's overall output: the list of `SampleReview`s
plus a `BatchSummary` of aggregate metrics and (optionally) a calibration drift
report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

from .finding import DimensionFinding, Issue

# ── Verdict tier ────────────────────────────────────────────────────────────

Verdict = Literal[
    "accept",
    "accept_with_fixes",
    "needs_review",
    "reject_fixable",
    "reject_structural",
]

VERDICT_ORDER: list[Verdict] = [
    "reject_structural",
    "reject_fixable",
    "needs_review",
    "accept_with_fixes",
    "accept",
]


# ── Per-sample review ──────────────────────────────────────────────────────

@dataclass
class SampleReview:
    """The framework's per-sample output, intended for the reviewer."""

    sample_id:              str
    verdict:                Verdict
    one_line:               str
    top_issues:             list[Issue] = field(default_factory=list)   # v18.6: ALL blocking issues (no numeric cap)
    suggested_fixes:        list[str]   = field(default_factory=list)   # retained for back-compat; empty under v18.6
    review_required_because: str | None = None
    all_findings:           list[DimensionFinding] = field(default_factory=list)
    synthesizer_fell_back:  bool = False    # True when LLM output was rejected and we used deterministic prose
    raw_trace_path:         str | None = None  # location of raw per-judge transcripts for this sample
    # v18.6: findings the synthesizer judged cosmetic / judge-overcall and filtered out of top_issues
    # (recorded, not dropped — every applicable sev>=1 finding is in top_issues XOR filtered_issues).
    filtered_issues:        list[Issue] = field(default_factory=list)
    # input/solution files whose conversion failed → judged from degraded text
    # (set by v18_engine._collect_conversion_warnings).
    data_quality_warnings:  list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["top_issues"]      = [i.to_dict() for i in self.top_issues]
        d["filtered_issues"] = [i.to_dict() for i in self.filtered_issues]
        d["all_findings"]    = [f.to_dict() for f in self.all_findings]
        return d

    def blocking_findings(self) -> list[DimensionFinding]:
        return [f for f in self.all_findings
                if f.applicable and f.gate_role == "blocking" and f.severity >= 1]


# ── Batch-level aggregates ─────────────────────────────────────────────────

@dataclass
class BatchSummary:
    """Aggregate metrics across one batch."""

    verdict_counts:        dict[Verdict, int] = field(default_factory=dict)
    verdict_percentages:   dict[Verdict, float] = field(default_factory=dict)
    failure_mode_histogram: dict[str, int] = field(default_factory=dict)   # issue dim -> count
    judge_agreement_rate:  float = 1.0                                     # 0..1 across (sample × dim)
    sample_count:          int = 0
    samples_with_findings: int = 0    # samples with any non-informational sev >= 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationReport:
    """Result of running the spec's calibration goldens against current judges.

    `drift` lists goldens whose verdict changed relative to the recorded
    expected verdict; an empty list means no drift detected. `enabled=False`
    indicates the spec disabled calibration (or `goldens_path` is empty).
    """

    enabled:        bool = True
    goldens_count:  int = 0
    drift:          list[dict] = field(default_factory=list)   # [{sample_id, expected, observed}]
    warning:        str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Batch-level review ─────────────────────────────────────────────────────

@dataclass
class BatchReview:
    """The framework's overall output for one `assess_batch` call."""

    spec_name:     str
    spec_version:  str
    samples:       list[SampleReview] = field(default_factory=list)
    batch_summary: BatchSummary       = field(default_factory=BatchSummary)
    calibration:   CalibrationReport | None = None

    def to_dict(self) -> dict:
        return {
            "spec_name":     self.spec_name,
            "spec_version":  self.spec_version,
            "samples":       [s.to_dict() for s in self.samples],
            "batch_summary": self.batch_summary.to_dict(),
            "calibration":   self.calibration.to_dict() if self.calibration else None,
        }

    def samples_by_verdict(self, verdict: Verdict) -> list[SampleReview]:
        return [s for s in self.samples if s.verdict == verdict]

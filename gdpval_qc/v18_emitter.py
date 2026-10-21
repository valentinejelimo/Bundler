"""Pure-function output emitters for a BatchReview.

Three emitters, all deterministic given the input BatchReview:

  to_json(batch_review)        → JSON-serializable dict (the source of truth)
  to_csv(batch_review, cap)    → CSV string with fixed-width issue slot columns
  to_markdown_summary(batch_review) → batch-level markdown summary

`write_batch_artifacts(batch_review, out_dir)` writes all three plus the raw
per-sample judge transcripts under `out_dir/raw/sample_<id>.json`.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .finding import DimensionFinding
from .review import BatchReview, SampleReview, Verdict


# ── JSON ───────────────────────────────────────────────────────────────────

def to_json(batch_review: BatchReview) -> dict:
    """Return the BatchReview as a JSON-serializable dict."""
    return batch_review.to_dict()


# ── CSV (reviewer queue) ───────────────────────────────────────────────────

# Sort order so reviewers see the most urgent samples first
_VERDICT_SORT_RANK: dict[Verdict, int] = {
    "needs_review":      0,
    "reject_structural": 1,
    "reject_fixable":    2,
    "accept_with_fixes": 3,
    "accept":            4,
}


#: Row-per-issue CSV columns (v18.6). Each surfaced or filtered issue is one
#: row; the numeric top-N slot layout was retired when the synthesizer moved
#: from a fixed cap to a qualitative blocking-vs-cosmetic gate. evidence_quote
#: and detail are sourced from the underlying DimensionFinding.
_CSV_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "verdict",
    "one_line",
    "review_reason",
    "issue_rank",
    "classification",        # blocking | cosmetic | "" (no-issue sample)
    "dimension",
    "severity",
    "fixability",
    "headline",
    "action",                # synthesizer's fix step (blocking issues)
    "classification_reason", # synthesizer's one-clause justification
    "evidence_quote",
    "detail",
    "judges_agreed",
    "judge_trace_path",
    "data_quality_warnings",  # files judged from degraded text (conversion failed)
)


def to_csv(batch_review: BatchReview, top_issues_cap: int | None = None) -> str:
    """Produce a flat CSV with ONE ROW PER ISSUE (v18.6).

    Each blocking issue (from `top_issues`) and each filtered/cosmetic issue
    (from `filtered_issues`) becomes its own row; blocking rows come before
    cosmetic rows within a sample. A sample with no issues (e.g. `accept`)
    emits a single row with empty issue columns so it still appears in the
    queue. Samples are ordered most-urgent-first by verdict.

    `top_issues_cap` is accepted for backward-compatible call sites and
    ignored — there is no longer a numeric cap.
    """
    sorted_samples = sorted(
        batch_review.samples,
        key=lambda s: (_VERDICT_SORT_RANK.get(s.verdict, 99), s.sample_id),
    )

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(list(_CSV_COLUMNS))
    for s in sorted_samples:
        finding_by_dim = {f.dimension: f for f in s.all_findings}
        base = [s.sample_id, s.verdict, s.one_line, s.review_required_because or ""]
        dq = "; ".join(s.data_quality_warnings)

        issue_rows = (
            [("blocking", i) for i in s.top_issues]
            + [("cosmetic", i) for i in s.filtered_issues]
        )
        if not issue_rows:
            # 11 empty issue columns (issue_rank … judges_agreed) + judge_trace_path + dq
            w.writerow(base + [""] * 11 + [s.raw_trace_path or "", dq])
            continue

        for classification, issue in issue_rows:
            finding = finding_by_dim.get(issue.dimension)
            evidence = (finding.evidence_quote if finding else issue.evidence_quote) or ""
            detail   = (finding.detail if finding else "") or ""
            agreed   = "Y" if issue.judges_agreed else "N"
            w.writerow(base + [
                issue.rank,
                classification,
                issue.dimension,
                issue.severity,
                issue.fixability,
                issue.headline,
                issue.action,
                issue.classification_reason,
                evidence,
                detail,
                agreed,
                s.raw_trace_path or "",
                dq,
            ])
    return buf.getvalue()


# ── Markdown summary ───────────────────────────────────────────────────────

def collect_judge_errors(batch_review: BatchReview) -> dict:
    """Walk every JudgeAttempt and tabulate ones whose call failed.

    Returns::

        {
            "total_attempts": int,       # across all samples / dims / judges
            "total_errors": int,         # attempts where .error is not None
            "by_sample": {sid: {"errors": n, "attempts": n}},
            "by_reason": {reason_first_line: count},
        }

    A judge attempt with ``error is not None`` indicates the LLM call itself
    raised (HTTP error, missing dependency, parse failure, etc.) and the
    framework substituted a default-applicable=False attempt. These attempts
    must be visible in the surface artifacts — they look like clean
    "not applicable" findings in the verdict otherwise.
    """
    by_sample: dict[str, dict[str, int]] = {}
    by_reason: dict[str, int] = {}
    total_attempts = 0
    total_errors = 0
    for s in batch_review.samples:
        sid = s.sample_id
        s_attempts = 0
        s_errors = 0
        for f in s.all_findings:
            for a in f.raw_attempts:
                s_attempts += 1
                if a.error:
                    s_errors += 1
                    # Use first line of the error as a category key, truncated.
                    reason = (a.error.splitlines()[0] if a.error else "")[:120]
                    by_reason[reason] = by_reason.get(reason, 0) + 1
        total_attempts += s_attempts
        total_errors += s_errors
        if s_errors:
            by_sample[sid] = {"errors": s_errors, "attempts": s_attempts}
    return {
        "total_attempts": total_attempts,
        "total_errors": total_errors,
        "by_sample": by_sample,
        "by_reason": by_reason,
    }


def to_markdown_summary(batch_review: BatchReview) -> str:
    bs = batch_review.batch_summary
    lines: list[str] = []
    lines.append(f"# Batch summary — {batch_review.spec_name} v{batch_review.spec_version}")
    lines.append("")
    lines.append(f"Total samples evaluated: **{bs.sample_count}**  ")
    lines.append(f"Samples with at least one finding: **{bs.samples_with_findings}**  ")
    lines.append(f"Judge agreement rate (within 1 severity level): **{bs.judge_agreement_rate:.3f}**")
    lines.append("")

    # ── Errors section ───────────────────────────────────────────────────
    # A run with many failed judge calls can otherwise look clean (every
    # failed attempt aggregates to applicable=False, every dim becomes
    # "not applicable", verdicts default to accept). Surface this loudly.
    errs = collect_judge_errors(batch_review)
    if errs["total_errors"] > 0:
        rate = errs["total_errors"] / errs["total_attempts"] if errs["total_attempts"] else 0.0
        lines.append("## ⚠ Judge-call errors")
        lines.append("")
        lines.append(
            f"**{errs['total_errors']}** of **{errs['total_attempts']}** judge calls "
            f"failed ({rate:.1%}). When judge calls fail, the affected dimensions "
            f"silently become `applicable=False` and contribute no findings to the "
            f"verdict — review the per-sample raw transcripts before trusting the "
            f"verdict distribution above."
        )
        lines.append("")
        if errs["by_sample"]:
            lines.append("### Per-sample error counts")
            lines.append("")
            lines.append("| sample_id | judge errors | total attempts | error rate |")
            lines.append("|---|---|---|---|")
            for sid, counts in sorted(errs["by_sample"].items()):
                e, t = counts["errors"], counts["attempts"]
                lines.append(f"| {sid} | {e} | {t} | {e/t:.1%} |")
            lines.append("")
        if errs["by_reason"]:
            lines.append("### Error reasons (first line, truncated)")
            lines.append("")
            lines.append("| count | reason |")
            lines.append("|---|---|")
            for reason, n in sorted(errs["by_reason"].items(),
                                     key=lambda kv: (-kv[1], kv[0])):
                lines.append(f"| {n} | `{reason}` |")
            lines.append("")

    lines.append("## Verdict distribution")
    lines.append("")
    lines.append("| verdict | count | % |")
    lines.append("|---|---|---|")
    # Render in priority order
    for v in ("reject_structural", "reject_fixable", "needs_review",
              "accept_with_fixes", "accept"):
        n = bs.verdict_counts.get(v, 0)
        pct = bs.verdict_percentages.get(v, 0.0)
        lines.append(f"| {v} | {n} | {pct:.1f}% |")
    lines.append("")

    if bs.failure_mode_histogram:
        lines.append("## Failure-mode histogram (top issues by dimension)")
        lines.append("")
        lines.append("| dimension | count |")
        lines.append("|---|---|")
        for dim, n in sorted(bs.failure_mode_histogram.items(),
                              key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| {dim} | {n} |")
        lines.append("")

    if batch_review.calibration is not None:
        c = batch_review.calibration
        lines.append("## Calibration")
        lines.append("")
        if not c.enabled:
            lines.append("Calibration disabled in spec.")
        elif c.goldens_count == 0:
            lines.append(c.warning or "No calibration goldens registered.")
        else:
            lines.append(f"Goldens evaluated: **{c.goldens_count}**  ")
            if c.drift:
                lines.append(f"Drift detected on **{len(c.drift)}** golden(s):")
                for d in c.drift:
                    lines.append(
                        f"  - `{d['sample_id']}`: expected `{d['expected']}` "
                        f"→ observed `{d['observed']}`"
                    )
            else:
                lines.append("No drift detected.")
        lines.append("")

    # ── Data-quality warnings ────────────────────────────────────────────
    # Files whose conversion failed are judged from degraded text-only
    # fallback. Surface this so a clean-looking verdict isn't trusted over
    # content the judges never actually saw rendered.
    dq = [(s.sample_id, w) for s in batch_review.samples
          for w in s.data_quality_warnings]
    if dq:
        affected = sorted({sid for sid, _ in dq})
        lines.append("## ⚠ Data-quality warnings")
        lines.append("")
        lines.append(
            f"**{len(dq)}** file conversion failure(s) across "
            f"**{len(affected)}** sample(s). The affected files were judged "
            f"from degraded text-only fallback, so layout/format/visual "
            f"assessments for them are unreliable."
        )
        lines.append("")
        lines.append("| sample_id | warning |")
        lines.append("|---|---|")
        for sid, w in sorted(dq):
            lines.append(f"| {sid} | {w} |")
        lines.append("")

    return "\n".join(lines)


# ── Composite writer ───────────────────────────────────────────────────────

def write_batch_artifacts(
    batch_review: BatchReview,
    out_dir: Path,
    top_issues_cap: int = 3,
) -> dict[str, Path]:
    """Write batch_review.json, reviewer_queue.csv, batch_summary.md and the
    per-sample raw transcripts under out_dir/. Returns a dict mapping
    artifact name → final path."""
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    paths: dict[str, Path] = {}

    paths["batch_review.json"] = out_dir / "batch_review.json"
    paths["batch_review.json"].write_text(
        json.dumps(to_json(batch_review), indent=2, default=str),
        encoding="utf-8",
    )

    paths["reviewer_queue.csv"] = out_dir / "reviewer_queue.csv"
    paths["reviewer_queue.csv"].write_text(
        to_csv(batch_review, top_issues_cap), encoding="utf-8",
    )

    paths["batch_summary.md"] = out_dir / "batch_summary.md"
    paths["batch_summary.md"].write_text(
        to_markdown_summary(batch_review), encoding="utf-8",
    )

    # Per-sample raw transcripts
    for s in batch_review.samples:
        raw_path = raw_dir / f"sample_{s.sample_id}.json"
        raw_path.write_text(
            json.dumps(_raw_payload_for(s), indent=2, default=str),
            encoding="utf-8",
        )

    return paths


def _raw_payload_for(s: SampleReview) -> dict:
    return {
        "sample_id":   s.sample_id,
        "verdict":     s.verdict,
        "findings": [
            {
                "dimension":   f.dimension,
                "severity":    f.severity,
                "confidence":  f.confidence,
                "applicable":  f.applicable,
                "evidence_quote":  f.evidence_quote,
                "headline":    f.headline,
                "detail":      f.detail,
                "judges_agreed": f.judges_agreed,
                "demoted_because": f.demoted_because,
                "raw_attempts": [a.to_dict() for a in f.raw_attempts],
            }
            for f in s.all_findings
        ],
    }

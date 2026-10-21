"""Deterministic verdict computation for v18.

The verdict for a sample is a function of (DimensionFinding list, V18Spec)
with no LLM in the loop. This module implements the entire decision tree in
~30 lines of code; everything that follows in the synthesizer is prose work
that cannot change the verdict.

Verdict tiers, in priority order (first match wins):

  1. REJECT_STRUCTURAL — any blocking dim at sev ≥ reject_at_severity
                          with fixability=structural
  2. REJECT_FIXABLE    — any blocking dim at sev ≥ reject_at_severity
                          with fixability ≠ structural
  3. NEEDS_REVIEW      — judge disagreement on any blocking dim,
                          OR a confidence-flagged needs_review_signal,
                          OR a blocking dim at sev == reject_at_severity − 1
                            (borderline)
  4. ACCEPT_WITH_FIXES — any non-informational dim at severity ≥ 1
                          (after the higher tiers ruled out)
  5. ACCEPT            — no non-informational findings ≥ severity 1

The pre_ranking() function produces the ordered list of finding candidates
the synthesizer will read. Ordering rule:
  (gate_role_priority, -severity, -fixability_priority)
where blocking dims rank above advisory, advisory above informational,
informational excluded entirely from the pre-rank. Within the same role,
higher severity ranks above lower, and within the same severity, more
structural (less fixable) ranks above more trivially fixable.
"""

from __future__ import annotations

from typing import Iterable

from .finding import DimensionFinding, Fixability, GateRole
from .review import Verdict
from .v18_spec import V18Spec

# ── Priority maps ──────────────────────────────────────────────────────────

_ROLE_RANK: dict[GateRole, int] = {
    "blocking":       0,
    "advisory":       1,
    "informational":  2,
}

_FIX_RANK: dict[Fixability, int] = {
    "structural":  3,
    "fixable":     2,
    "trivial":     1,
    "n/a":         0,
}


# ── Public functions ───────────────────────────────────────────────────────

def reject_at_severity_for(dim_name: str, spec: V18Spec) -> int:
    """Look up the spec's reject threshold for a named blocking dim."""
    d = spec.get_dimension(dim_name)
    if d is None:
        return 3
    return d.gate.reject_at_severity


def compute_verdict(
    findings: list[DimensionFinding],
    spec: V18Spec,
) -> tuple[Verdict, str | None]:
    """Apply the verdict rules to a sample's findings.

    Returns (verdict, review_required_because). The latter is a non-None
    string only when the verdict is NEEDS_REVIEW; it explains why review is
    needed.

    Priority order:

      1. NEEDS_REVIEW takes priority over REJECT when judges disagree on a
         blocking dim, or when confidence is low on a sev-≥-2 finding. The
         logic is that uncertainty about whether to reject should defer to
         human review rather than auto-rejecting on a contested signal.
      2. REJECT_STRUCTURAL / REJECT_FIXABLE: blocking dim at sev ≥
         reject_at_severity with consensus across judges.
      3. NEEDS_REVIEW for borderline blocking findings (sev = threshold - 1).
      4. ACCEPT_WITH_FIXES for any non-informational sev ≥ 1.
      5. ACCEPT.
    """
    active = [f for f in findings if f.applicable]

    # Tier 1 (priority): NEEDS_REVIEW for contested blocking findings.
    # When judges DISAGREED on a blocking dim, we don't trust the
    # aggregated severity enough to auto-reject — defer to human.
    disagreed_blocking = next(
        (f for f in active
         if f.gate_role == "blocking" and not f.judges_agreed),
        None,
    )
    if disagreed_blocking:
        return (
            "needs_review",
            f"judge disagreement on blocking dim '{disagreed_blocking.dimension}'",
        )

    # Tier 2 (priority): low-confidence finding at sev >= 2 also routes to review
    review_signal = next((f for f in active if f.needs_review_signal), None)
    if review_signal:
        return (
            "needs_review",
            f"low-confidence finding at severity {review_signal.severity} "
            f"on dim '{review_signal.dimension}'",
        )

    # Tier 3 & 4: REJECT when judges agree the finding is reject-worthy
    blocking_above_threshold = [
        f for f in active
        if f.gate_role == "blocking"
        and f.severity >= reject_at_severity_for(f.dimension, spec)
    ]
    if blocking_above_threshold:
        if any(f.fixability == "structural" for f in blocking_above_threshold):
            return "reject_structural", None
        return "reject_fixable", None

    # Tier 5: borderline blocking finding
    borderline = next(
        (f for f in active
         if f.gate_role == "blocking"
         and f.severity == reject_at_severity_for(f.dimension, spec) - 1
         and f.severity >= 1),
        None,
    )
    if borderline:
        return (
            "needs_review",
            f"borderline blocking finding at severity {borderline.severity} "
            f"on dim '{borderline.dimension}' "
            f"(reject threshold = {reject_at_severity_for(borderline.dimension, spec)})",
        )

    # Tier 6: ACCEPT_WITH_FIXES
    advisory_or_blocking_sev1 = any(
        f.gate_role in ("blocking", "advisory") and f.severity >= 1
        for f in active
    )
    if advisory_or_blocking_sev1:
        return "accept_with_fixes", None

    # Tier 7: ACCEPT
    return "accept", None


# ── Pre-ranking for the synthesizer ────────────────────────────────────────

def pre_rank_findings(
    findings: Iterable[DimensionFinding],
) -> list[DimensionFinding]:
    """Order non-informational findings with severity ≥ 1 by:
        (role_rank ASC, -severity, -fixability_rank).

    Informational dims and severity-0 findings are excluded — they're not
    candidates for the reviewer-facing top_issues list.
    """
    candidates = [
        f for f in findings
        if f.applicable and f.severity >= 1 and f.gate_role != "informational"
    ]
    candidates.sort(
        key=lambda f: (
            _ROLE_RANK.get(f.gate_role, 99),
            -f.severity,
            -_FIX_RANK.get(f.fixability, 0),
        )
    )
    return candidates


def is_must_fix(f: DimensionFinding) -> bool:
    """A finding is 'must-fix' (and therefore a candidate for the
    must-fix-only top_issues list) when:

      - severity ≥ 2 on any non-informational dim, OR
      - severity 1 on a blocking dim (one notch from the reject threshold).

    A severity-1 finding on an advisory dim is a minor/nice-to-have and
    does NOT qualify as must-fix — it belongs in suggested_fixes, not
    top_issues. This filtering prevents minor formatting nits from
    becoming severe blockers in the reviewer-facing artifact.
    """
    if not f.applicable or f.gate_role == "informational":
        return False
    if f.severity >= 2:
        return True
    if f.severity == 1 and f.gate_role == "blocking":
        return True
    return False


def split_must_fix(
    findings: Iterable[DimensionFinding],
) -> tuple[list[DimensionFinding], list[DimensionFinding]]:
    """Partition pre-ranked findings into (must_fix, minor) lists.

    Both lists preserve the input ordering (which is already
    severity-descending). Used by the synthesizer to draw top_issues
    from `must_fix` and to populate suggested_fixes from
    `minor` + must-fix overflow beyond the cap.
    """
    ranked = pre_rank_findings(findings)
    must_fix: list[DimensionFinding] = []
    minor:    list[DimensionFinding] = []
    for f in ranked:
        (must_fix if is_must_fix(f) else minor).append(f)
    return must_fix, minor

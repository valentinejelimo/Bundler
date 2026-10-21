"""Per-dimension aggregation: JudgeAttempts → DimensionFinding.

All aggregation logic is here as pure functions over input data. No I/O.
This module is the only place where the anti-overcall guards described in
the v18 spec frontmatter are applied:

  - Evidence-quote enforcement (`require_evidence_quote: true`): any
    attempt with severity ≥ 1 but no evidence_quote is demoted to
    severity 0 for that attempt, with `demoted_because` recording why.

  - Confidence demotion (`confidence_demotion`): after the per-dim
    severity is aggregated, low-confidence findings at severity 1 are
    further demoted to 0, and low-confidence findings at severity ≥ 2 are
    flagged with `needs_review_signal=True` (consumed by `verdict.py`).

  - Applicability fold-in: if the majority of attempts say not applicable,
    the whole finding becomes applicable=False with severity 0.

Aggregation rules:
  - severity = max(attempt_severities) over applicable attempts
  - confidence = min(attempt_confidences) over applicable attempts
  - fixability = most-severe attempt's fixability
  - judges_agreed = (max - min) severity within 1 across applicable attempts
  - headline, evidence_quote, detail = picked from the highest-severity
    applicable attempt; detail extended with a disagreement note when
    judges disagreed
"""

from __future__ import annotations

from typing import Iterable

from .finding import (
    Confidence, DimensionFinding, Fixability, JudgeAttempt,
)
from .v18_spec import V18Dimension, V18Spec


# ── Confidence helpers ─────────────────────────────────────────────────────

_CONFIDENCE_ORDER: dict[Confidence, int] = {"low": 0, "medium": 1, "high": 2}


def _min_confidence(attempts: Iterable[JudgeAttempt]) -> Confidence:
    items = [(_CONFIDENCE_ORDER.get(a.confidence, 1), a.confidence) for a in attempts]
    if not items:
        return "low"
    return min(items)[1]


# ── Anti-overcall enforcement (per-attempt) ────────────────────────────────

def enforce_evidence_quote(
    attempts: list[JudgeAttempt],
    require_evidence_quote: bool,
) -> tuple[list[JudgeAttempt], list[str]]:
    """If evidence quotes are required, demote any sev>=1 attempt that has no
    quote. Returns (new_attempts, demoted_messages).

    The input list is not mutated; new JudgeAttempt objects are returned with
    severity=0 and the original headline/justification preserved (the
    evidence is still useful audit context).
    """
    if not require_evidence_quote:
        return list(attempts), []

    demoted: list[str] = []
    out: list[JudgeAttempt] = []
    for a in attempts:
        if a.applicable and a.severity >= 1 and not (a.evidence_quote and a.evidence_quote.strip()):
            new_a = JudgeAttempt(
                model=a.model,
                severity=0,
                confidence=a.confidence,
                applicable=a.applicable,
                headline=a.headline,
                justification=a.justification,
                fixability="n/a",
                evidence_quote=None,
                raw_response=a.raw_response,
                error=a.error,
            )
            out.append(new_a)
            demoted.append(
                f"{a.model}: severity {a.severity} demoted to 0 (no evidence quote)"
            )
        else:
            out.append(a)
    return out, demoted


# ── Applicability ──────────────────────────────────────────────────────────

def _resolve_applicability(attempts: list[JudgeAttempt]) -> tuple[bool, list[JudgeAttempt]]:
    """Decide whether the dim is applicable for this sample.

    Rule: applicable = True iff a majority of attempts say applicable=True
    (and at least one attempt exists). Failed attempts (error set) are
    excluded from the vote — they neither vote nor count toward the total.

    Returns (is_applicable, attempts_used_for_aggregation).
    """
    voting = [a for a in attempts if a.error is None]
    if not voting:
        return False, []
    yes = sum(1 for a in voting if a.applicable)
    is_applicable = yes > len(voting) / 2
    if not is_applicable:
        return False, voting
    return True, [a for a in voting if a.applicable]


# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate_attempts(
    dim: V18Dimension,
    attempts: list[JudgeAttempt],
    spec: V18Spec,
) -> DimensionFinding:
    """Aggregate the ensemble's attempts into one DimensionFinding.

    Workflow:
      1. Enforce evidence-quote rule per attempt.
      2. Resolve applicability via majority vote.
      3. If not applicable → DimensionFinding(applicable=False, sev=0).
      4. Else compute severity (max), confidence (min), fixability (from
         highest-severity attempt), judges_agreed (within 1 severity).
      5. Apply confidence demotion (sev 1 → 0, sev ≥ 2 → needs_review_signal).
    """
    # Step 1: evidence enforcement
    enforced, evidence_demotions = enforce_evidence_quote(
        attempts, spec.require_evidence_quote
    )

    # Step 2: applicability
    is_applicable, voting = _resolve_applicability(enforced)
    if not is_applicable:
        return DimensionFinding(
            dimension=dim.name,
            dimension_name=dim.dimension_name,
            gate_role=dim.gate.role,
            applicable=False,
            severity=0,
            fixability="n/a",
            confidence="medium",
            headline="not applicable",
            detail="The dimension's preconditions or judge majority indicated "
                   "this criterion does not apply to this sample.",
            evidence_quote=None,
            judges_agreed=True,
            raw_attempts=attempts,
            demoted_because=evidence_demotions,
            skipped_because="not_applicable",
            needs_review_signal=False,
        )

    # Step 3-4: severity, confidence, judges_agreed, headline, evidence
    severities = [a.severity for a in voting]
    max_sev = max(severities)
    min_sev = min(severities)
    judges_agreed = (max_sev - min_sev) <= 1
    aggregated_confidence = _min_confidence(voting)

    # Pick the attempt that drove the max severity as the canonical voice
    canonical = next(a for a in voting if a.severity == max_sev)
    fixability: Fixability = canonical.fixability if max_sev >= 1 else "n/a"
    headline = canonical.headline
    evidence_quote = canonical.evidence_quote
    detail = canonical.justification

    # Note judge disagreement explicitly in the detail
    if not judges_agreed:
        other_severities = sorted({a.severity for a in voting if a is not canonical})
        detail = (
            f"{detail} (Judge ensemble disagreed: this judge called severity "
            f"{max_sev}; other judges reported {other_severities}.)"
        )

    # Step 5: confidence demotion
    demoted_because = list(evidence_demotions)
    needs_review_signal = False
    final_severity = max_sev

    if aggregated_confidence == "low" and final_severity == 1:
        # low_at_sev1 → demote_to_0
        if spec.confidence_demotion.low_at_sev1 == "demote_to_0":
            demoted_because.append(
                "aggregated low confidence at severity 1 → demoted to 0"
            )
            final_severity = 0
            fixability = "n/a"
            headline = "no issue"
            evidence_quote = None
    elif aggregated_confidence == "low" and final_severity >= 2:
        # low_at_sev2_or_higher → route_to_review
        if spec.confidence_demotion.low_at_sev2_or_higher == "route_to_review":
            needs_review_signal = True
            demoted_because.append(
                f"aggregated low confidence at severity {final_severity} → review signal"
            )

    return DimensionFinding(
        dimension=dim.name,
        dimension_name=dim.dimension_name,
        gate_role=dim.gate.role,
        applicable=True,
        severity=final_severity,
        fixability=fixability,
        confidence=aggregated_confidence,
        headline=headline if final_severity >= 1 else "no issue",
        detail=detail,
        evidence_quote=evidence_quote if final_severity >= 1 else None,
        judges_agreed=judges_agreed,
        raw_attempts=attempts,
        demoted_because=demoted_because,
        skipped_because=None,
        needs_review_signal=needs_review_signal,
    )


def make_skipped_finding(
    dim: V18Dimension,
    reason: str,
) -> DimensionFinding:
    """Build a DimensionFinding for a dim that was skipped before any judge
    call was made (e.g., applies_when=False)."""
    return DimensionFinding(
        dimension=dim.name,
        dimension_name=dim.dimension_name,
        gate_role=dim.gate.role,
        applicable=False,
        severity=0,
        fixability="n/a",
        confidence="medium",
        headline="not applicable",
        detail=f"Dimension skipped: {reason}.",
        evidence_quote=None,
        judges_agreed=True,
        raw_attempts=[],
        demoted_because=[],
        skipped_because=reason,
        needs_review_signal=False,
    )

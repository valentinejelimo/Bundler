"""Per-sample synthesizer (v18.6 — qualitative blocking-vs-cosmetic triage).

After every dim has been graded and the verdict has been computed
deterministically (`v18_verdict.compute_verdict`), the synthesizer makes one
LLM call to triage the findings for the reviewer:

  - a one-line summary of the sample's state
  - `top_issues`: EVERY finding the synthesizer judges genuinely blocking
    (must-fix), with no numeric cap
  - `filtered_issues`: findings the synthesizer judges cosmetic / a judge
    over-call, filtered out of the must-fix list but recorded (not dropped)

This replaces the old fixed `top_issues_cap` (top-3) design, which was shown
to silence real signal: on the calibration set, 11 issues the post-hoc 20% QC
re-flagged were already found by the first pass but hidden by the cap. The
synthesizer now decides *which* findings matter rather than truncating to a
count — aware that the judge ensemble over-calls (severity is the MAX across
judges, so one dissenting judge inflates a finding).

Invariants preserved:
  - The LLM cannot change the verdict (computed deterministically).
  - Every applicable, severity>=1, non-informational finding appears in
    `top_issues` XOR `filtered_issues` — nothing is silently dropped.
  - A verdict-determinative finding (severity >= reject_at_severity on a
    blocking dim) is FORCE-kept blocking; the synthesizer cannot filter it.
  - On any LLM failure the synthesizer falls back to surfacing ALL candidate
    findings as blocking (conservative — never filters without reasoning).
"""

from __future__ import annotations

import json
import time
from typing import Any

from .finding import DimensionFinding, Issue
from .llm_client import LLMClient, LLMClientConfig, LLMMessage
from .review import SampleReview, Verdict
from .v18_judge import _JSON_FENCE_RE, _BARE_OBJECT_RE
from .v18_spec import V18Spec
from .v18_verdict import compute_verdict, pre_rank_findings, reject_at_severity_for


# ── Retry policy for the synthesis LLM call ─────────────────────────────────
# Covers transient client-level failures (auth/rate-limit/timeout/transport)
# on the synthesis call specifically. Malformed-output errors do not retry
# (see the except clause below) since a shape problem won't fix itself.
_MAX_SYNTH_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0


# ── Public API ─────────────────────────────────────────────────────────────

def synthesize_sample_review(
    sample_id: str,
    findings: list[DimensionFinding],
    spec: V18Spec,
    client: LLMClient | None,
    raw_trace_path: str | None = None,
) -> SampleReview:
    """Produce the SampleReview for one sample.

    Steps:
      1. Compute verdict deterministically.
      2. Collect candidate findings (applicable, sev>=1, non-informational).
      3. If no candidates / synthesis disabled / client is None: deterministic
         review (all candidates surfaced blocking).
      4. Otherwise: call the synthesizer LLM to classify each candidate
         blocking vs cosmetic, validate, fall back if invalid.
    """
    verdict, review_reason = compute_verdict(findings, spec)
    candidates = pre_rank_findings(findings)

    if not candidates or not spec.synthesis.enabled or client is None:
        return _deterministic_sample_review(
            sample_id, verdict, review_reason, candidates, findings, spec,
            raw_trace_path=raw_trace_path,
        )

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_SYNTH_ATTEMPTS + 1):
        try:
            return _synthesize_via_llm(
                sample_id, verdict, review_reason, candidates, findings, spec,
                client, raw_trace_path=raw_trace_path,
            )
        except _SynthesizerConstraintError as exc:
            # Malformed LLM output — retrying the same prompt won't fix a
            # shape problem. Fall back immediately.
            last_exc = exc
            break
        except Exception as exc:  # noqa: BLE001 - the LLM client call itself
            # (auth/rate-limit/timeout/transport). These have been observed
            # to be transient on a key that is valid for the rest of the run
            # (e.g. a one-off OpenAI 401 invalid_organization) — retry a
            # couple of times before giving up on this one sample.
            last_exc = exc
            if attempt < _MAX_SYNTH_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S * attempt)
                continue
            break

    # Every attempt failed: degrade this ONE sample to the conservative
    # deterministic review (never raise past here — a single sample's
    # synthesis failure must never take down the rest of the batch).
    review = _deterministic_sample_review(
        sample_id, verdict, review_reason, candidates, findings, spec,
        raw_trace_path=raw_trace_path,
    )
    review.synthesizer_fell_back = True
    if review.review_required_because is None:
        review.review_required_because = f"synthesizer fell back: {last_exc}"
    return review


# ── Internals ──────────────────────────────────────────────────────────────

class _SynthesizerConstraintError(ValueError):
    """Raised when the LLM's synthesis output violates a hard constraint."""


_SYSTEM_PROMPT = """\
You are the lead quality-control reviewer for a benchmark of long-horizon expert knowledge-work tasks. An ensemble of LLM judges has already graded one task-under-review across many quality dimensions and produced FINDINGS. Your job is to triage those findings into exactly what a human reviewer must act on before the task ships — surfacing every genuine defect while filtering nits and judge over-calls.

You receive, for one task:
- `verdict`: the programmatic verdict (already decided — context only; you cannot change it).
- `findings`: a numbered list. Each finding has `dimension`, `dimension_name`, `gate_role` (`blocking` or `advisory`), `severity` (1-3), `judges_agreed`, `confidence`, `per_judge_severities` (each judge's individual score), `demoted_because`, an `evidence_quote` from the task materials, `headline`, and `detail`.

Classify EVERY finding as exactly one of **blocking** or **cosmetic**.

## Blocking

A genuine defect a domain expert would require fixed before the task is put in front of a client:
- the task is unsolvable, self-contradictory, missing required inputs, or the instruction leaks the answer;
- the source documents are mutually inconsistent on a value the task depends on;
- an input/source document substantially mirrors or duplicates the task instruction — a near-duplicate "prompt artifact" that contaminates the task and does not function as authentic source data — even if it does not reveal the numeric answer (`input_prompt_leakage`);
- the VERIFIER is invalid — it would grade a solution wrongly.

### Verifier-quality findings (read carefully — do NOT under-call these)

A benchmark verifier must grade objectively and reproducibly: concrete pass/fail anchors, so two fair reviewers reach the same verdict on the same solution. A rubric that fails this is broken regardless of `gate_role` — judges often file these as `advisory`; ignore that and judge by content.

Classify **blocking** any verifier finding that makes a SPECIFIC, GROUNDED argument (it names one or more rubrics/checks, or quotes an instruction/rubric passage) that:
- a grading criterion is unanchored/subjective — vague qualifiers like "appropriate", "credible", "reasonable", "strong", "defensible", "sufficient", "material" used as a pass/fail condition (`verifier_rubric_clarity`, `verifier_rubric_specificity`); or
- a rubric grades a text/output proxy instead of the real artifact, or asserts a check it cannot perform (`verifier_rubric_construct_validity`); or
- a required outcome the task demands is graded by no rubric, or the deliverable could be wrong (fabricated/incorrect values, missing required content) and still pass (`verifier_missing_critical_dimensions`, `verifier_outcome_coverage`, `verifier_discriminative_power`).

Crucial: the defect is the non-reproducibility / coverage gap itself. You do NOT need the finding to exhibit a specific solution that gets mis-graded. "Two fair reviewers could disagree because the threshold is subjective" and "a fabricated value could still pass" ARE blocking conditions. Do NOT downgrade on the grounds that it is "not tied to a concrete pass/fail error", "not shown to make the rubric misgrade a specific output", or "a minor subjective edge in otherwise concrete rubrics". A grounded verifier-quality finding is blocking even when the judges split, provided the argument names something specific (a general "it's all covered / it's fine" rebuttal from the dissenting judge does not make it cosmetic).

When genuinely unsure about a verifier-quality finding, classify **blocking**.

## Cosmetic

A nit or an over-call an expert would NOT require fixed. Classify **cosmetic** when any of these clearly holds:
- **Pure polish** — formatting, style, layout, wording, or an optional nice-to-have with no effect on solvability or grading correctness.
- **Incidental overlap** (`input_prompt_leakage`) — a legitimate source document shares only a few sentences or business-rule phrases with the instruction (minor, incidental overlap) without exposing the analytical answer. Cosmetic ONLY when the overlap is limited; when an input document substantially mirrors the instruction it is blocking contamination (see above).
- **No grading impact** (verifier findings) — the finding is only about redundant/double-counted rubric weighting, cosmetic rubric wording, or a purely stylistic rubric concern that does not let a wrong solution pass or a right one fail.
- **Lone-judge over-call** — for a NON-verifier finding, the severity rests on a single dissenting judge (`per_judge_severities` shows one high outlier while the other(s) are 0 and `judges_agreed` is false) AND the evidence is weak: no `evidence_quote`, or `evidence_quote`/`detail` that argues only in the abstract ("could", "may", "risks") without naming a specific phrase, file, value, or rubric.

Do NOT filter a finding merely because it lacks a concrete failing-output example. Do NOT apply the lone-judge rule to verifier-quality findings that name something specific.

## Rules
- No limit on the number of blocking issues. Surface ALL genuine ones; never suppress one to shorten the list. Never pad the list with cosmetic items.
- For each **blocking** issue write an `action`: one concrete sentence naming the specific rubric / phrase / file / value from the finding's `detail` or `evidence_quote`.
- For every finding write a one-clause `reason` justifying the classification.

Return STRICT JSON only, no prose outside it:

{{
  "one_line": "<=25-word summary of the task's overall state>",
  "issues": [
    {{
      "index": <int — the finding's index>,
      "classification": "blocking" | "cosmetic",
      "reason": "<one clause justifying the call>",
      "action": "<concrete reviewer step; required when blocking, else empty string>"
    }}
  ]
}}

Classify every finding exactly once, keyed by its `index`.
"""


def _candidate_view(candidates: list[DimensionFinding]) -> list[dict]:
    return [
        {
            "index":            i,
            "dimension":        f.dimension,
            "dimension_name":   f.dimension_name,
            "gate_role":        f.gate_role,
            "severity":         f.severity,
            "judges_agreed":    f.judges_agreed,
            "confidence":       f.confidence,
            "per_judge_severities": [
                {"model": a.model, "severity": a.severity, "applicable": a.applicable}
                for a in f.raw_attempts
            ],
            "demoted_because":  f.demoted_because,
            "evidence_quote":   f.evidence_quote,
            "headline":         f.headline,
            "detail":           f.detail,
        }
        for i, f in enumerate(candidates)
    ]


def _synthesize_via_llm(
    sample_id: str,
    verdict: Verdict,
    review_reason: str | None,
    candidates: list[DimensionFinding],
    all_findings: list[DimensionFinding],
    spec: V18Spec,
    client: LLMClient,
    raw_trace_path: str | None = None,
) -> SampleReview:
    view = _candidate_view(candidates)
    user_msg = (
        f"<verdict>\n{verdict}\n</verdict>\n\n"
        f"<review_required_because>\n{review_reason or 'n/a'}\n</review_required_because>\n\n"
        f"<findings>\n{json.dumps(view, indent=2)}\n</findings>\n\n"
        "Classify every finding by its index. Return only the JSON object specified in the system prompt."
    )
    config = LLMClientConfig(
        model=spec.synthesis.judge_model,
        max_tokens=spec.synthesis.judge_max_tokens,
        reasoning_effort=spec.synthesis.judge_reasoning_effort,
        system=_SYSTEM_PROMPT,
    )
    response = client.complete([LLMMessage(role="user", content=user_msg)], config)
    data = _extract_json(response)
    decisions = _validate_synth_response(data, candidates)

    one_line = data["one_line"]
    top_issues: list[Issue] = []
    filtered_issues: list[Issue] = []
    for i, f in enumerate(candidates):
        d = decisions[i]
        cls = d["classification"]
        # Hard guard: a verdict-determinative finding can never be filtered.
        forced = False
        if f.gate_role == "blocking" and f.severity >= reject_at_severity_for(f.dimension, spec):
            if cls != "blocking":
                cls, forced = "blocking", True
        action = d.get("action") or ""
        reason = d.get("reason") or ""
        if cls == "blocking" and not action.strip():
            action = _default_action_for(f)
        issue = Issue(
            rank=0,  # assigned below per list
            dimension=f.dimension,
            gate_role=f.gate_role,
            severity=f.severity,
            fixability=f.fixability,
            headline=f.headline,
            action=action if cls == "blocking" else "",
            evidence_quote=f.evidence_quote,
            judges_agreed=f.judges_agreed,
            blocking=(cls == "blocking"),
            classification_reason=("verdict-determinative (hard guard)" if forced else reason),
            forced_blocking=forced,
        )
        (top_issues if cls == "blocking" else filtered_issues).append(issue)

    for r, issue in enumerate(top_issues, start=1):
        issue.rank = r
    for r, issue in enumerate(filtered_issues, start=1):
        issue.rank = r

    return SampleReview(
        sample_id=sample_id,
        verdict=verdict,
        one_line=one_line,
        top_issues=top_issues,
        suggested_fixes=[],
        review_required_because=review_reason,
        all_findings=all_findings,
        synthesizer_fell_back=False,
        raw_trace_path=raw_trace_path,
        filtered_issues=filtered_issues,
    )


def _extract_json(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise _SynthesizerConstraintError("empty response from synthesizer")
    m = _JSON_FENCE_RE.search(text)
    payload = m.group(1).strip() if m else None
    if payload is None or not payload.startswith("{"):
        m = _BARE_OBJECT_RE.search(text)
        payload = m.group(0) if m else text.strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise _SynthesizerConstraintError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _SynthesizerConstraintError(f"JSON root must be object, got {type(data).__name__}")
    return data


def _validate_synth_response(
    data: dict[str, Any],
    candidates: list[DimensionFinding],
) -> dict[int, dict]:
    """Validate the LLM response and return index -> decision dict.

    Requires: a non-empty one_line; an `issues` list; every candidate index
    classified exactly once with a valid classification. Raises
    `_SynthesizerConstraintError` (→ deterministic fallback) otherwise.
    """
    if not isinstance(data.get("one_line"), str) or not data["one_line"].strip():
        raise _SynthesizerConstraintError("one_line missing or empty")
    issues = data.get("issues")
    if not isinstance(issues, list):
        raise _SynthesizerConstraintError("issues must be a list")

    decisions: dict[int, dict] = {}
    for entry in issues:
        if not isinstance(entry, dict):
            raise _SynthesizerConstraintError("issues[] entry is not an object")
        idx = entry.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(candidates)):
            raise _SynthesizerConstraintError(f"issues[] index {idx!r} out of range")
        if idx in decisions:
            raise _SynthesizerConstraintError(f"issues[] index {idx} classified more than once")
        cls = entry.get("classification")
        if cls not in ("blocking", "cosmetic"):
            raise _SynthesizerConstraintError(f"issues[{idx}].classification {cls!r} invalid")
        decisions[idx] = entry

    missing = [i for i in range(len(candidates)) if i not in decisions]
    if missing:
        raise _SynthesizerConstraintError(f"findings not classified: {missing}")
    return decisions


# ── Deterministic fallback prose ───────────────────────────────────────────

_VERDICT_PROSE = {
    "accept":             "No findings of concern across the dimensions evaluated.",
    "accept_with_fixes":  "Acceptable with minor fixes required.",
    "needs_review":       "Borderline or contested findings — manual review recommended.",
    "reject_fixable":     "Rejectable due to one or more material findings.",
    "reject_structural":  "Rejectable due to one or more structural findings.",
}


def _deterministic_sample_review(
    sample_id: str,
    verdict: Verdict,
    review_reason: str | None,
    candidates: list[DimensionFinding],
    all_findings: list[DimensionFinding],
    spec: V18Spec,
    raw_trace_path: str | None = None,
) -> SampleReview:
    """Conservative fallback: surface EVERY candidate finding as blocking.

    Used when synthesis is disabled, no client is available, or the LLM output
    was rejected. We never filter without the LLM's reasoning, so nothing is
    silently masked.
    """
    one_line = _VERDICT_PROSE[verdict]
    if candidates:
        one_line = f"{one_line.rstrip('.')}: {candidates[0].headline}."

    top_issues = [
        Issue(
            rank=i + 1,
            dimension=f.dimension,
            gate_role=f.gate_role,
            severity=f.severity,
            fixability=f.fixability,
            headline=f.headline,
            action=_default_action_for(f),
            evidence_quote=f.evidence_quote,
            judges_agreed=f.judges_agreed,
            blocking=True,
            classification_reason="surfaced conservatively (synthesizer unavailable)",
            forced_blocking=False,
        )
        for i, f in enumerate(candidates)
    ]

    return SampleReview(
        sample_id=sample_id,
        verdict=verdict,
        one_line=one_line,
        top_issues=top_issues,
        suggested_fixes=[],
        review_required_because=review_reason,
        all_findings=all_findings,
        synthesizer_fell_back=False,
        raw_trace_path=raw_trace_path,
        filtered_issues=[],
    )


def _default_action_for(f: DimensionFinding) -> str:
    if f.fixability == "structural":
        return f"Review and revise the underlying sample to address: {f.headline}"
    if f.fixability == "fixable":
        return f"Rewrite the relevant section of the deliverable to address: {f.headline}"
    if f.fixability == "trivial":
        return f"Apply a small edit to fix: {f.headline}"
    return f"See finding detail for: {f.headline}"

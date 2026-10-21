#!/usr/bin/env python3
"""Deterministic structural checks for an SVC (Cirrus) `rubric.json`.

This is the SVC replacement for the EKW `check_verifier.py`. The EKW checker
reads `verifiers[]` (flat, atomic, weights→1.0) and emits a *false* blocking
sev-2 auto-reject on every SVC task, because SVC ships a different rubric object:

    { task, total_points, scoring, task_instruction, grading_guidance,
      categories: [ {name, weight}, ... ],           # weights sum to 100
      criteria:   [ {id, category, tier, description, expected_value,
                     points, must_pass, penalty?}, ... ] }

  tiers: objective | judgment | structural
  gates: criteria with must_pass=true   (any gate fails ⇒ reward 0)
  penalties: criteria with points<0     (subtracted; may carry a `penalty` field)

The checks below grade the rubric as a *scoring instrument* — the mechanics EKW
had no analog for (category weights, gates, penalties) plus per-level count
sanity that replaces EKW's fixed [6,20] band (SVC counts legitimately span
1 at an L1 leaf to 100 at the L3 apex), plus a blank-field check
(`rubric_criteria_not_blank`) added after a real delivered task shipped a
criterion with an empty `description`/`expected_value`.

Findings use the SAME shape as the EKW skill so they fold into the identical
verdict computation (see the EKW SKILL.md). Stdlib only; no network.

    python check_rubric.py <rubric.json>            # level inferred from `task`
    python check_rubric.py <rubric.json> --level L2  # force the level
"""
from __future__ import annotations

import argparse
import json
import re
import sys

# Gate metadata — role + reject threshold per dimension, folded into the verdict.
GATES = {
    "rubric_spec_parseable":            {"role": "blocking", "reject_at_severity": 2},
    "rubric_category_weights_sum":      {"role": "blocking", "reject_at_severity": 2},
    "rubric_criteria_category_valid":   {"role": "advisory", "reject_at_severity": 3},
    "rubric_gate_present_and_sane":     {"role": "blocking", "reject_at_severity": 2},
    "rubric_count_for_level":           {"role": "advisory", "reject_at_severity": 3},
    "rubric_points_penalty_coherent":   {"role": "advisory", "reject_at_severity": 3},
    "rubric_criteria_not_blank":        {"role": "blocking", "reject_at_severity": 2},
    "rubric_vague_language_scan":       {"role": "advisory", "reject_at_severity": 3},
}

# Vague/subjective quality-language a criterion should not lean on as its
# pass condition without a concrete anchor elsewhere in the same criterion.
# Curated from a random-sample calibration run against real delivered
# rubrics: every one of 15 real verifier_rubric_clarity/specificity findings
# missed by an earlier, narrower version of this script used one of these
# words. Deliberately high-recall, not high-precision — a keyword hit here
# is a candidate for a human to look at (see `rubric-structure-check`'s
# SKILL.md), not an automatic verdict. Some of these words are fine when an
# `expected_value`/`expected` anchor backs them; this check does not try to
# detect that nuance, because that nuance is exactly what caused the misses
# it's meant to close.
_VAGUE_TERMS = [
    "board-ready", "board ready", "well-organized", "well organized",
    "professionalism", "on-brand", "on brand", "branding",
    "narrative-quality", "narrative quality", "visual-quality",
    "visual quality", "chart-quality", "chart quality", "customer-voice",
    "customer voice", "deck-source", "deck source", "proportionate",
    "appropriate", "reasonable", "sufficient", "material", "credible",
    "defensible", "compelling", "polished", "plausible", "non-defensive",
    "controls pass", "tone",
]
_VAGUE_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _VAGUE_TERMS) + r")\b",
    re.IGNORECASE,
)

# Per-level criterion-count expectations. L1 leaves are intentionally atomic
# (exactly 1 criterion is correct — NOT a coverage defect); L2/L3 need breadth.
LEVEL_MIN = {"L1": 1, "L2": 6, "L3": 30}


def _finding(dim, severity, headline, justification, fixability):
    g = GATES[dim]
    return {
        "dimension": dim,
        "role": g["role"],
        "reject_at_severity": g["reject_at_severity"],
        "severity": severity,
        "confidence": "high",
        "applicable": True,
        "evidence_quote": None,
        "headline": headline,
        "justification": justification,
        "fixability": fixability,
    }


def infer_level(data) -> str:
    """L1 / L2 / L3 from the `task` name (…-l1-…, -l2-…, -l3)."""
    task = str(data.get("task", ""))
    m = re.search(r"-l([123])\b", task) or re.search(r"-l([123])-", task)
    return f"L{m.group(1)}" if m else "?"


# ── schema compatibility ────────────────────────────────────────────────────
# Two rubric.json shapes have shipped: an older one with a top-level
# `categories: [{name, weight}, ...]` array (weights sum to 100), and the
# current one (100% of the 2026-07-31 delivery, 0/204 files on the older
# shape) with no `categories` array at all — `category` is just a free-text
# label per criterion, `points` and `must_pass` are strings ("8", "Yes"/"No")
# rather than a number/bool. The checks below detect which shape they're
# looking at and adapt rather than force a check that only makes sense for
# the other shape — inventing a fake "must sum to 100" rule for a schema that
# has no such invariant would just be a new false positive.

def _is_flat_schema(data) -> bool:
    """True for the current (categories-less) rubric shape."""
    return not (isinstance(data.get("categories"), list) and data.get("categories"))


def _is_must_pass(c: dict) -> bool:
    """`must_pass` is a bool in the old shape, "Yes"/"No" (any case) in the new."""
    v = (c or {}).get("must_pass")
    if isinstance(v, bool):
        return v
    return str(v or "").strip().lower() in ("yes", "true", "1")


def _points(c: dict) -> float:
    """`points` is a number in the old shape, a numeric string in the new."""
    try:
        return float((c or {}).get("points", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _expected_text(c: dict) -> str:
    """The pass-condition field is `expected_value` (old) or `expected` (new)."""
    return str((c or {}).get("expected_value") or (c or {}).get("expected") or "")


# ── checks ───────────────────────────────────────────────────────────────────

def check_parseable(spec_json: str):
    """Sev 0 if it parses to an object with a `criteria` list.

    A top-level `categories` list is no longer required — it's present in the
    older weighted-categories rubric shape but absent from 100% of the current
    delivery's rubric.json files (they use a flat per-criterion `category`
    label instead). Its absence is a schema variant, not a parse failure.
    """
    if not isinstance(spec_json, str) or not spec_json.strip():
        return None, _finding("rubric_spec_parseable", 2, "rubric.json is missing",
                               "No rubric spec was provided.", "structural")
    try:
        data = json.loads(spec_json)
    except json.JSONDecodeError as exc:
        return None, _finding("rubric_spec_parseable", 2,
                              f"rubric.json fails to parse ({type(exc).__name__})",
                              f"JSON parse error: {exc}.", "fixable")
    if not isinstance(data, dict):
        return None, _finding("rubric_spec_parseable", 2,
                              "rubric.json root is not an object",
                              f"got {type(data).__name__}", "fixable")
    if not isinstance(data.get("criteria"), list):
        return None, _finding("rubric_spec_parseable", 2,
                              "rubric.json missing `criteria` list",
                              "Top-level `criteria` must be a list.", "fixable")
    return data, _finding("rubric_spec_parseable", 0, "rubric.json parses cleanly",
                          "Programmatic check passed.", "n/a")


def check_category_weights_sum(data):
    """Sev 0 if category weights sum to 100 (±0.01) — only for the older
    weighted-categories shape. The current flat shape has no weight concept
    to check (`points` are per-criterion importance values that do not sum
    to any fixed total — 16, 14, 85, 100 all appear as legitimate totals
    across real delivered rubrics), so this check reports not-applicable
    rather than inventing a threshold the schema was never designed to meet.
    """
    if _is_flat_schema(data):
        return _finding("rubric_category_weights_sum", 0,
                        "not applicable — this rubric has no categories[] weight "
                        "schema (flat per-criterion category label + points instead)",
                        "Programmatic check passed (not applicable to this schema).",
                        "n/a")
    cats = data.get("categories") or []
    wsum = sum((c or {}).get("weight", 0) for c in cats)
    if abs(wsum - 100) <= 0.01:
        return _finding("rubric_category_weights_sum", 0,
                        f"{len(cats)} category weights sum to 100",
                        "Programmatic check passed.", "n/a")
    return _finding("rubric_category_weights_sum", 2,
                    f"category weights sum to {wsum:g}, not 100",
                    "SVC scoring is category-weight = weight × (passed/available); "
                    "weights must sum to 100 or the reward is mis-scaled.", "fixable")


def check_criteria_category_valid(data):
    """Every criterion.category must name a declared category — only checkable
    in the older shape, which declares `categories[].name`. The current flat
    shape has no declared category list to validate against (category is just
    a free-text label a creator writes on each criterion), so there is
    nothing to be an "orphan" of.
    """
    if _is_flat_schema(data):
        return _finding("rubric_criteria_category_valid", 0,
                        "not applicable — no declared categories[] to validate "
                        "against in this schema",
                        "Programmatic check passed (not applicable to this schema).",
                        "n/a")
    names = {(c or {}).get("name") for c in (data.get("categories") or [])}
    orphan = sorted({(c or {}).get("category") for c in (data.get("criteria") or [])
                     if (c or {}).get("category") not in names})
    if not orphan:
        return _finding("rubric_criteria_category_valid", 0,
                        "every criterion maps to a declared category",
                        "Programmatic check passed.", "n/a")
    return _finding("rubric_criteria_category_valid", 1,
                    f"{len(orphan)} criterion category name(s) not in categories[]",
                    f"Orphan category name(s): {orphan}. Points in an undeclared "
                    "category carry no weight.", "fixable")


def check_gate_present_and_sane(data, level: str):
    """>=1 must_pass gate; flag over-gating on multi-criterion rubrics.

    Any gate failing zeros the whole reward, so gates must be few and essential.
    An L1 leaf legitimately has its single criterion as the gate.
    """
    crit = data.get("criteria") or []
    gates = [c for c in crit if _is_must_pass(c)]
    n, g = len(crit), len(gates)
    if g == 0:
        return _finding("rubric_gate_present_and_sane", 2, "no must_pass gate",
                        "No criterion is must_pass; nothing can force reward→0, so a "
                        "fatally wrong solution can still score high.", "fixable")
    # Over-gating: on a multi-criterion rubric, >50% gates makes reward brittle.
    if n >= 4 and g / n > 0.5:
        return _finding("rubric_gate_present_and_sane", 1,
                        f"{g}/{n} criteria are must_pass gates (over-gated)",
                        "More than half the criteria zero the reward on failure; "
                        "judge noise on any one gate sinks valid solutions. Reserve "
                        "gates for non-negotiable requirements.", "fixable")
    return _finding("rubric_gate_present_and_sane", 0,
                    f"{g} must_pass gate(s) across {n} criteria",
                    "Programmatic check passed.", "n/a")


def check_count_for_level(data, level: str):
    """Per-level minimum criterion count (replaces EKW's fixed [6,20])."""
    n = len(data.get("criteria") or [])
    lo = LEVEL_MIN.get(level)
    if lo is None:
        return _finding("rubric_count_for_level", 1,
                        f"{n} criteria; level unknown",
                        "Could not infer L1/L2/L3 from `task`; count not gated "
                        "against a level expectation.", "fixable")
    if level == "L1":
        # Atomic leaf: exactly 1 is correct; more is fine, 0 is broken.
        if n >= 1:
            return _finding("rubric_count_for_level", 0,
                            f"L1 leaf: {n} criterion(a)",
                            "Programmatic check passed.", "n/a")
        return _finding("rubric_count_for_level", 2, "L1 leaf has 0 criteria",
                        "An atomic leaf must grade its single fact.", "fixable")
    if n >= lo:
        return _finding("rubric_count_for_level", 0,
                        f"{level}: {n} criteria (>= {lo})",
                        "Programmatic check passed.", "n/a")
    dist = lo - n
    sev = 1 if dist <= 2 else 2
    return _finding("rubric_count_for_level", sev,
                    f"{level}: {n} criteria (expected >= {lo})",
                    f"A {level} node with {n} criteria is thin by {dist}; coverage "
                    "of its analytical scope is likely incomplete.", "fixable")


def check_points_penalty_coherent(data):
    """total_points sanity + penalty criteria are negative-pointed.

    `points` is a number in the older shape and a numeric string ("8") in the
    current one — comparing a raw string to 0 with `<`/`>=` raises a
    TypeError in Python 3, so this always goes through `_points()` first.
    """
    crit = data.get("criteria") or []
    total = data.get("total_points")
    pens = [c for c in crit if _points(c) < 0 or (c or {}).get("penalty")]
    bad_pen = [c.get("id") for c in pens if _points(c) >= 0]
    msgs = []
    if total not in (100, None):
        msgs.append(f"total_points={total} (SVC convention is 100)")
    if bad_pen:
        msgs.append(f"penalty-flagged criteria with non-negative points: {bad_pen}")
    if msgs:
        return _finding("rubric_points_penalty_coherent", 1,
                        "; ".join(msgs),
                        "Penalty criteria must carry negative points; total_points "
                        "should be 100. NOTE: conditional penalty logic (e.g. 'C08c "
                        "scored only if C08b does not fire') is not machine-checkable "
                        "here — route it to a human.", "fixable")
    note = " (incl. conditional-penalty logic — verify by hand)" if pens else ""
    return _finding("rubric_points_penalty_coherent", 0,
                    f"points/penalties coherent{note}",
                    "Programmatic check passed.", "n/a")


def check_criteria_not_blank(data):
    """Flag any criterion with an empty `description` or expected-value field.

    The expected-value field is named `expected_value` in the older schema
    and `expected` in the current one — `_expected_text()` reads either. A
    criterion with a blank description or expected value has no observable
    pass condition — it can't be graded, and it can't be checked for clarity
    or specificity by any downstream stage. This is a raw authoring gap
    (a row left unfinished), not a judgment call, so it's checked here
    deterministically rather than left to an LLM dimension to notice.
    """
    crit = data.get("criteria") or []
    blank = [
        c.get("id") for c in crit
        if not str((c or {}).get("description") or "").strip()
        or not _expected_text(c).strip()
    ]
    if not blank:
        return _finding("rubric_criteria_not_blank", 0,
                        "every criterion has a description and expected_value",
                        "Programmatic check passed.", "n/a")
    return _finding("rubric_criteria_not_blank", 2,
                    f"{len(blank)} criterion/criteria have a blank description "
                    f"or expected_value: {blank}",
                    "A blank description or expected_value gives a judge and a "
                    "grader nothing to check against — the criterion cannot be "
                    "graded as written.", "fixable")


def check_vague_language(data):
    """Flag criteria whose description uses known vague/subjective language.

    High-recall by design (see `_VAGUE_TERMS` above) — this will over-flag
    relative to a human judge who can tell an anchored "professional" from
    an unanchored one. Treat every hit as a candidate, not a verdict.
    """
    crit = data.get("criteria") or []
    hits = []
    for c in crit:
        # The vague word is often the pass condition itself, e.g.
        # `"expected": "Well-organized, review-ready"` — a real miss this
        # check's first version had, scanning only description/category.
        text = (f"{(c or {}).get('description','')} {(c or {}).get('category','')} "
                f"{_expected_text(c)}")
        m = _VAGUE_RE.search(text)
        if m:
            hits.append((c.get("id"), m.group(0)))
    if not hits:
        return _finding("rubric_vague_language_scan", 0,
                        "no known vague/subjective terms found in any criterion",
                        "Programmatic check passed.", "n/a")
    shown = hits[:8]
    more = f" (+{len(hits)-8} more)" if len(hits) > 8 else ""
    return _finding("rubric_vague_language_scan", 1,
                    f"{len(hits)} criterion/criteria use known vague/subjective "
                    f"language: {[f'{cid} ({term!r})' for cid, term in shown]}{more}",
                    "High-recall keyword match, not a verdict — a hit is fine if "
                    "the same criterion's expected_value/expected field supplies "
                    "a concrete anchor. Confirm by hand before editing; see "
                    "verifier_rubric_clarity in QC_DIMENSIONS_FOR_CREATORS.md.",
                    "fixable")


def run(spec_json: str, level_override: str | None):
    data, parseable = check_parseable(spec_json)
    if data is None:
        unrun = "Cannot run — rubric.json did not parse (see rubric_spec_parseable)."
        return [parseable] + [
            _finding(d, 2, f"rubric.json unparseable — {d} not checked", unrun, "fixable")
            for d in ("rubric_category_weights_sum", "rubric_criteria_category_valid",
                      "rubric_gate_present_and_sane", "rubric_count_for_level",
                      "rubric_points_penalty_coherent", "rubric_criteria_not_blank",
                      "rubric_vague_language_scan")
        ]
    level = level_override or infer_level(data)
    return [
        parseable,
        check_category_weights_sum(data),
        check_criteria_category_valid(data),
        check_gate_present_and_sane(data, level),
        check_count_for_level(data, level),
        check_points_penalty_coherent(data),
        check_criteria_not_blank(data),
        check_vague_language(data),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Structural checks for an SVC rubric.json")
    ap.add_argument("rubric_json", help="Path to the SVC rubric.json")
    ap.add_argument("--level", choices=["L1", "L2", "L3"], default=None,
                    help="Force the DAG level (else inferred from the `task` name)")
    args = ap.parse_args(argv)
    try:
        with open(args.rubric_json, encoding="utf-8") as fh:
            spec_json = fh.read()
    except OSError as exc:
        print(f"error: cannot read {args.rubric_json!r}: {exc}", file=sys.stderr)
        return 2

    findings = run(spec_json, args.level)
    print(f"SVC structural rubric checks — {args.rubric_json}")
    print("=" * 64)
    for f in findings:
        flag = "PASS" if f["severity"] == 0 else f"sev {f['severity']}"
        print(f"  [{flag:>5}] {f['dimension']:<32} {f['headline']}")
        if f["severity"] > 0:
            print(f"          ↳ {f['justification']}")
    print("\nFold these findings into the verdict (JSON):")
    print(json.dumps(findings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

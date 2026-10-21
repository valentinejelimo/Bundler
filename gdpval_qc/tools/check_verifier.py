#!/usr/bin/env python3
"""Run the three deterministic (no-LLM) verifier checks for Verifier QC.

These are the *programmatic* dimensions of the quality spec — they are
graded by code, not by reading rubric prose, so the agent runs this script
instead of eyeballing the JSON:

  verifier_spec_parseable          (blocking, reject_at_severity 2)
  verifier_count_in_range          (advisory, reject_at_severity 3)
  verifier_outcome_grader_present  (advisory, reject_at_severity 3)

The expert pastes the verifier (the task's "Rubric") into the conversation;
save it to a file and point this script at it:

    python scripts/check_verifier.py <verifier.json>
    python scripts/check_verifier.py <verifier.json> --min 6 --max 20

Input shape (the same object stored in the tracker's "Rubric" column and in a
Harbor task's tests/verifier.json):

    {"task_id": ..., "config": {...}, "verifiers": [ {"name","metadata","assertion"}, ... ]}

Stdlib only — no third-party deps, no network. The agent folds the emitted
findings into the same verdict computation as the LLM dimensions (see
SKILL.md). It prints a human-readable summary followed by a JSON block.
"""

from __future__ import annotations

import argparse
import json
import sys

# Gate metadata, copied verbatim from the spec's per-dimension `gate:` blocks
# so the agent knows how to fold each finding into the verdict.
GATES = {
    "verifier_spec_parseable":         {"role": "blocking", "reject_at_severity": 2},
    "verifier_count_in_range":         {"role": "advisory", "reject_at_severity": 3},
    "verifier_outcome_grader_present": {"role": "advisory", "reject_at_severity": 3},
}


def _finding(dimension, severity, headline, justification, fixability):
    """Build a finding in the skill's finding shape (see SKILL.md).

    Programmatic findings are always high-confidence and carry no evidence
    quote (the evidence is the structural fact the code checked).
    """
    g = GATES[dimension]
    return {
        "dimension": dimension,
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


# ── The three checks (faithful port of gdpval_qc/v18_callbacks.py) ───────────

def check_parseable(spec_json: str):
    """Sev 0 if the verifier parses to {verifiers: [...]}; sev 2 otherwise.

    Mirrors the lightweight structural validation the framework runtime does
    before handing the spec to its Pydantic `VerifierSpec` model. (Full
    Pydantic validation lives upstream; this reproduces the json.loads + shape
    checks, which is what catches the truncated / malformed cells in practice.)
    """
    if not isinstance(spec_json, str) or not spec_json.strip():
        return None, _finding(
            "verifier_spec_parseable", 2,
            "verifier.json is missing",
            "No verifier spec was provided for this task.",
            "structural")
    try:
        data = json.loads(spec_json)
    except json.JSONDecodeError as exc:
        return None, _finding(
            "verifier_spec_parseable", 2,
            f"verifier.json fails to parse ({type(exc).__name__})",
            f"JSON parse error: {exc}. A truncated paste is the usual cause — "
            "re-copy the full Rubric cell.",
            "fixable")
    if not isinstance(data, dict):
        return None, _finding(
            "verifier_spec_parseable", 2,
            "verifier.json root is not an object",
            f"got {type(data).__name__}", "fixable")
    if not isinstance(data.get("verifiers"), list):
        return None, _finding(
            "verifier_spec_parseable", 2,
            "verifier.json missing `verifiers` list",
            "The top-level field `verifiers` must be a list.", "fixable")
    return data, _finding(
        "verifier_spec_parseable", 0,
        "verifier.json parses cleanly", "Programmatic check passed.", "n/a")


def check_count_in_range(data, min_n: int, max_n: int):
    """Sev 0 inside [min, max]; sev 1 if outside by 1-2; sev 2 otherwise."""
    count = len(data.get("verifiers") or [])
    if min_n <= count <= max_n:
        return _finding(
            "verifier_count_in_range", 0,
            f"{count} verifiers in [{min_n}, {max_n}]",
            "Programmatic check passed.", "n/a")
    distance = min(abs(count - min_n), abs(count - max_n))
    sev = 1 if distance <= 2 else 2
    qualifier = "just" if sev == 1 else "materially"
    return _finding(
        "verifier_count_in_range", sev,
        f"{count} verifiers (target: [{min_n}, {max_n}])",
        f"Verifier count {count} is {qualifier} outside the recommended range "
        f"by {distance}; consider adding/removing rubrics.",
        "fixable")


def check_outcome_grader_present(data):
    """Sev 0 if >=1 verifier is an outcome-grader; sev 2 otherwise.

    Handles both schemas: legacy GDPVal-216 `metadata.classification == "core"`
    and Omega `metadata.grading_role == "outcome"`. A rubric matches if EITHER
    marker is set.
    """
    outcome = 0
    for v in (data.get("verifiers") or []):
        meta = (v or {}).get("metadata") or {}
        if meta.get("classification") == "core" or meta.get("grading_role") == "outcome":
            outcome += 1
    if outcome >= 1:
        return _finding(
            "verifier_outcome_grader_present", 0,
            f"{outcome} outcome-grading verifier(s) present",
            "Programmatic check passed.", "n/a")
    return _finding(
        "verifier_outcome_grader_present", 2,
        "no outcome-grading verifiers",
        "No verifier carries metadata.classification == 'core' (legacy schema) "
        "or metadata.grading_role == 'outcome' (omega schema). A verifier "
        "without an outcome-grading rubric checks only process/format and "
        "cannot drive a meaningful reward.",
        "fixable")


def run(spec_json: str, min_n: int, max_n: int):
    data, parseable = check_parseable(spec_json)
    if data is None:
        # Unparseable: the blocking parseable check carries the real failure;
        # the count/outcome checks cannot run, so report them as such (they
        # do not add new information beyond the parse error).
        unrunnable = (
            "Cannot run — verifier.json did not parse (see "
            "verifier_spec_parseable).")
        return [
            parseable,
            _finding("verifier_count_in_range", 2,
                     "verifier.json unparseable — count not checked",
                     unrunnable, "fixable"),
            _finding("verifier_outcome_grader_present", 2,
                     "verifier.json unparseable — graders not checked",
                     unrunnable, "fixable"),
        ]
    return [
        parseable,
        check_count_in_range(data, min_n, max_n),
        check_outcome_grader_present(data),
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the 3 deterministic verifier checks for Verifier QC.")
    ap.add_argument("verifier_json",
                    help="Path to the verifier JSON (the pasted Rubric saved to a file).")
    ap.add_argument("--min", type=int, default=6,
                    help="Minimum recommended verifier count (spec default: 6).")
    ap.add_argument("--max", type=int, default=20,
                    help="Maximum recommended verifier count (spec default: 20).")
    args = ap.parse_args(argv)

    try:
        with open(args.verifier_json, "r", encoding="utf-8") as fh:
            spec_json = fh.read()
    except OSError as exc:
        print(f"error: cannot read {args.verifier_json!r}: {exc}", file=sys.stderr)
        return 2

    findings = run(spec_json, args.min, args.max)

    print(f"Programmatic verifier checks — {args.verifier_json}")
    print("=" * 64)
    for f in findings:
        flag = "PASS" if f["severity"] == 0 else f"sev {f['severity']}"
        print(f"  [{flag:>5}] {f['dimension']:<32} {f['headline']}")
        if f["severity"] > 0:
            print(f"          ↳ {f['justification']}")
    print()
    print("Fold these findings into the verdict (JSON):")
    print(json.dumps(findings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

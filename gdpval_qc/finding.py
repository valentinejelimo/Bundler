"""Per-dimension finding types for the v18 quality framework.

A `JudgeAttempt` is one model's response for one rubric on one sample. A
`DimensionFinding` is the aggregated result of an ensemble of `JudgeAttempt`s
for one dimension on one sample, after anti-overcall enforcement.

All types here are pure data — no behaviour. Aggregation lives in
`quality_framework.aggregate` (M3). Verdict computation lives in
`quality_framework.verdict` (M4). The synthesizer's per-issue output type
`Issue` is here as well because it is also a pure data record.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

# ── Common literal alias types ─────────────────────────────────────────────

Confidence    = Literal["low", "medium", "high"]
GateRole      = Literal["blocking", "advisory", "informational"]
Fixability    = Literal["trivial", "fixable", "structural", "n/a"]
OnDisagree    = Literal["review", "majority", "most_severe"]


# ── Per-judge attempt ──────────────────────────────────────────────────────

@dataclass
class JudgeAttempt:
    """The structured response from a single judge call for one dimension.

    The full LLM response (`raw_response`) is preserved verbatim for audit;
    everything else is the framework's parsed view of the structured output.
    """

    model:           str
    severity:        int                  # 0..3
    confidence:      Confidence
    applicable:      bool
    headline:        str
    justification:   str
    fixability:      Fixability
    evidence_quote:  str | None = None    # required by policy when severity >= 1
    raw_response:    str = ""             # full LLM response text, for audit
    error:           str | None = None    # populated only if the judge call failed

    def to_dict(self) -> dict:
        return asdict(self)


# ── Per-dim aggregated finding ─────────────────────────────────────────────

@dataclass
class DimensionFinding:
    """The aggregated outcome of one dimension's judge ensemble on one sample.

    Aggregation is deterministic from `raw_attempts` via the rules described
    in the v18 design. The fields below carry the post-aggregation state plus
    enough audit trail to reconstruct it.
    """

    dimension:        str                              # python identifier from spec
    dimension_name:   str                              # display name from spec
    gate_role:        GateRole
    applicable:       bool                             # if False, skipped (no LLM call made)
    severity:         int                              # 0..3, post-aggregation
    fixability:       Fixability
    confidence:       Confidence
    headline:         str                              # picked from the highest-severity applicable attempt
    detail:           str                              # 1-3 sentence summary (may merge attempts when they disagree)
    evidence_quote:   str | None
    judges_agreed:    bool                             # all applicable attempts within 1 severity level
    raw_attempts:     list[JudgeAttempt] = field(default_factory=list)
    demoted_because:  list[str] = field(default_factory=list)
    skipped_because:  str | None = None                # only set when applicable=False
    needs_review_signal: bool = False                  # set on low-confidence sev>=2; consumed by verdict.py

    def is_informational_only(self) -> bool:
        return self.gate_role == "informational"

    def is_finding(self) -> bool:
        """True when this dimension produced a non-zero severity finding."""
        return self.applicable and self.severity >= 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["raw_attempts"] = [a.to_dict() for a in self.raw_attempts]
        return d


# ── Synthesised top-K issue (output of the Synthesizer) ────────────────────

@dataclass
class Issue:
    """One synthesised issue in a sample's top_issues list.

    Synthesised by the Synthesizer LLM call from one or more `DimensionFinding`s.
    `action` is written by the synthesizer; everything else carries through
    from the underlying finding(s).
    """

    rank:             int
    dimension:        str
    gate_role:        GateRole
    severity:         int
    fixability:       Fixability
    headline:         str
    action:           str
    evidence_quote:   str | None
    judges_agreed:    bool
    promoted_above_pre_rank: int | None = None     # set when synthesizer reordered
    promotion_reason:        str | None = None
    # ── v18.6 qualitative-triage fields (synthesizer classifies every finding) ──
    blocking:               bool = True            # True → surfaced as must-fix; False → filtered cosmetic/overcall
    classification_reason:  str = ""               # synthesizer's one-clause justification for blocking/cosmetic
    forced_blocking:        bool = False           # kept blocking by the hard guard (verdict-determinative), not LLM choice

    def to_dict(self) -> dict:
        return asdict(self)

"""Spec parser for v18 of the GDPVal quality framework.

A v18 spec is a markdown file with YAML frontmatter and per-dimension sections.
The frontmatter declares spec-wide configuration; each per-dimension section
declares the criterion, the inputs the judge sees, the gate (how the finding
becomes part of the sample's verdict), and the rubric prose.

See `specs/gdpval-type-tasks/gdpval_task_quality_v18.md` for the canonical
example.

This parser is intentionally strict — it rejects any unknown frontmatter key
or per-dim block field. Older v17 specs will not load; callers get a clear
error message pointing at the version mismatch.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import yaml

from .finding import GateRole, OnDisagree

# ── Constants ──────────────────────────────────────────────────────────────

MIN_SPEC_VERSION_MAJOR = 18

# Known frontmatter keys — anything else raises.
KNOWN_FRONTMATTER_KEYS = {
    "name", "version", "cache_bust",
    "models", "field_types", "field_bindings",
    "synthesis", "calibration",
    "severity_scale", "require_evidence_quote", "confidence_demotion",
    "verdict_tiers",
}

# Overlay-only frontmatter directives. Consumed entirely by extends-resolution
# (see `_resolve_extends_text`) — never seen by `_build_spec`, never valid in
# a spec that isn't an overlay.
OVERLAY_DIRECTIVE_KEYS = {"extends", "remove_field_types", "remove_dimensions"}

# Known per-dimension yaml block keys.
KNOWN_DIM_KEYS = {
    "dimension_name", "check_kind", "callback",
    "inputs", "gate", "applies_when", "config",
    "eval_config",
}

# Known gate keys.
KNOWN_GATE_KEYS = {"role", "reject_at_severity", "on_disagreement"}

# Known applies_when keys.
KNOWN_APPLIES_WHEN_KEYS = {"builtin", "callback"}

VALID_GATE_ROLES: set[str] = {"blocking", "advisory", "informational"}
VALID_ON_DISAGREE: set[str] = {"review", "majority", "most_severe"}
VALID_CHECK_KINDS: set[str] = {"llm", "programmatic"}


# ── Validation issue / error types ─────────────────────────────────────────

@dataclass
class SpecIssue:
    severity: Literal["error", "warning"]
    location: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.location}: {self.message}"


class SpecValidationError(Exception):
    """Raised on any hard error during v18 spec parsing."""

    def __init__(self, spec_path: str | Path, issues: list[SpecIssue]):
        self.spec_path = str(spec_path)
        self.issues = issues
        msg = f"Spec validation failed for {self.spec_path}:\n"
        msg += "\n".join(f"  {it}" for it in issues)
        super().__init__(msg)


# ── Sub-types ──────────────────────────────────────────────────────────────

@dataclass
class JudgeModelConfig:
    """One judge model in the ensemble."""

    model: str
    reasoning_effort: str | None = None
    max_tokens: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class FieldBinding:
    """How a per-sample field is loaded from the task directory."""

    field_name: str
    source:     Literal["file", "directory", "callback", "text"]
    path:       str | None = None
    exclude:    list[str] = field(default_factory=list)
    callback:   str | None = None
    optional:   bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SynthesisConfig:
    """Synthesizer LLM call configuration."""

    enabled:                bool = True
    top_issues_cap:         int  = 3   # DEPRECATED (v18.6): the synthesizer no longer
                                        # caps surfaced issues to a count — it classifies
                                        # each finding blocking vs cosmetic and surfaces all
                                        # blocking ones. Parsed for back-compat; unused.
    judge_model:            str  = "gpt-5.5"
    judge_reasoning_effort: str  = "high"
    judge_max_tokens:       int  = 16000

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationConfig:
    enabled:       bool = True
    goldens_path:  str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConfidenceDemotion:
    low_at_sev1:           str = "demote_to_0"
    low_at_sev2_or_higher: str = "route_to_review"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateConfig:
    role:                GateRole
    reject_at_severity:  int
    on_disagreement:     OnDisagree = "review"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AppliesWhen:
    """Precondition expression for whether a dimension applies to a sample."""

    builtin:  str | None = None
    callback: str | None = None

    def is_unconditional(self) -> bool:
        return self.builtin is None and self.callback is None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class V18Dimension:
    """One per-sample dimension under v18."""

    name:           str                          # python identifier
    dimension_name: str                          # display label
    check_kind:     Literal["llm", "programmatic"]
    inputs:         list[str]
    gate:           GateConfig
    prose:          str                          # the markdown body of the dim section (rubric + hard-fails + anti-overcall)
    applies_when:   AppliesWhen | None = None
    callback:       str | None = None            # required for check_kind=programmatic
    config:         dict = field(default_factory=dict)
    eval_config:    dict | None = None           # rarely used per-dim model override

    def is_conditional(self) -> bool:
        return self.applies_when is not None and not self.applies_when.is_unconditional()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gate"] = self.gate.to_dict()
        if self.applies_when:
            d["applies_when"] = self.applies_when.to_dict()
        return d


@dataclass
class V18Spec:
    """A v18 quality spec, parsed and validated."""

    name:           str
    version:        str
    cache_bust:     str | None
    models:         list[JudgeModelConfig]
    field_types:    dict[str, str]
    field_bindings: list[FieldBinding]
    synthesis:      SynthesisConfig
    calibration:    CalibrationConfig
    severity_scale: dict[int, str]
    require_evidence_quote: bool
    confidence_demotion:    ConfidenceDemotion
    verdict_tiers:  list[str]
    dimensions:     list[V18Dimension]
    source_path:    Path | None = None

    @classmethod
    def from_markdown(cls, path: str | Path) -> "V18Spec":
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"v18 spec not found: {p}")

        text = _resolve_extends_text(p)
        frontmatter, body = _split_frontmatter(text, p)

        issues: list[SpecIssue] = []
        spec = _build_spec(frontmatter, body, p, issues)
        spec.source_path = p

        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise SpecValidationError(p, errors)

        return spec

    def get_dimension(self, name: str) -> V18Dimension | None:
        for d in self.dimensions:
            if d.name == name:
                return d
        return None

    def spec_hash(self) -> str:
        """SHA-256 over the canonicalised spec content.

        The hash is used by the framework cache to decide when the generated
        assessor package must be regenerated. It MUST cover every input that
        affects judge behaviour: all frontmatter (severity scale, evidence
        policy, confidence policy, verdict tiers, synthesis config, judge
        models, gating defaults) and the full prose + gate of every dimension.
        """
        h = hashlib.sha256()
        h.update(self.name.encode("utf-8"))
        h.update(b"|")
        h.update(self.version.encode("utf-8"))
        h.update(b"|")
        h.update((self.cache_bust or "").encode("utf-8"))
        h.update(b"|")
        h.update(repr(sorted([m.to_dict().items() for m in self.models],
                              key=lambda x: list(x))).encode("utf-8"))
        h.update(b"|")
        h.update(repr(sorted(self.field_types.items())).encode("utf-8"))
        h.update(b"|")
        h.update(repr([fb.to_dict() for fb in self.field_bindings]).encode("utf-8"))
        h.update(b"|")
        h.update(repr(self.synthesis.to_dict()).encode("utf-8"))
        h.update(b"|")
        h.update(repr(self.calibration.to_dict()).encode("utf-8"))
        h.update(b"|")
        h.update(repr(sorted(self.severity_scale.items())).encode("utf-8"))
        h.update(b"|")
        h.update(str(self.require_evidence_quote).encode("utf-8"))
        h.update(b"|")
        h.update(repr(self.confidence_demotion.to_dict()).encode("utf-8"))
        h.update(b"|")
        h.update(repr(self.verdict_tiers).encode("utf-8"))
        h.update(b"|")
        for d in self.dimensions:
            h.update(d.name.encode("utf-8"))
            h.update(b"|")
            h.update(repr(d.to_dict()).encode("utf-8"))
            h.update(b"|")
            h.update(d.prose.encode("utf-8"))
            h.update(b"||")
        return h.hexdigest()


# ── Parsing internals ──────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _split_frontmatter(text: str, path: Path) -> tuple[dict, str]:
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise SpecValidationError(path, [SpecIssue(
            "error", "frontmatter",
            "Spec must begin with `---` YAML frontmatter followed by `---`.",
        )])
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        raise SpecValidationError(path, [SpecIssue(
            "error", "frontmatter",
            f"Invalid YAML in frontmatter: {exc}",
        )]) from exc
    if not isinstance(fm, dict):
        raise SpecValidationError(path, [SpecIssue(
            "error", "frontmatter",
            "Frontmatter must be a YAML mapping.",
        )])
    return fm, m.group(2)


# ── Spec composition (`extends`) ────────────────────────────────────────────
#
# A front-specific spec (e.g. specs/overlays/svc.md) may declare
# `extends: <relative-path>` in its frontmatter instead of duplicating the
# whole base spec. Only its real deltas need to be present: overridden/added
# `field_types` and `field_bindings` entries, `remove_field_types` /
# `remove_dimensions` lists for what the front genuinely drops, and full
# `### dim_name` replacement sections for any dimension it changes. Anything
# the overlay doesn't mention is inherited byte-identical from the base, so a
# base-spec fix propagates to every front automatically. Resolution happens
# once, at raw-text level, before the normal strict parser ever runs — the
# parser never sees `extends` or the two removal directives.

def _resolve_extends_text(path: Path, _seen: tuple[Path, ...] = ()) -> str:
    """Return `path`'s spec text with any `extends` chain fully flattened."""
    p = path.expanduser().resolve()
    if p in _seen:
        chain = " -> ".join(str(x) for x in _seen + (p,))
        raise SpecValidationError(p, [SpecIssue(
            "error", "extends", f"Circular `extends` chain: {chain}",
        )])

    text = p.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(text, p)
    extends = fm.get("extends")
    if not extends:
        return text

    base_path = (p.parent / extends).resolve()
    if not base_path.is_file():
        raise SpecValidationError(p, [SpecIssue(
            "error", "extends", f"`extends: {extends}` not found (resolved to {base_path}).",
        )])
    base_text = _resolve_extends_text(base_path, _seen + (p,))
    base_fm, base_body = _split_frontmatter(base_text, base_path)

    merged_fm = _merge_frontmatter(base_fm, fm)
    merged_body = _merge_dimension_sections(
        base_body, body, remove_names=set(fm.get("remove_dimensions") or []),
    )
    return "---\n" + yaml.safe_dump(merged_fm, sort_keys=False) + "---\n" + merged_body


def _merge_frontmatter(base_fm: dict, overlay_fm: dict) -> dict:
    """Deep-merge an overlay's frontmatter onto its base's.

    `field_types`/`field_bindings` merge key-by-key (overlay entry wins for a
    shared key, both dicts' other keys survive); every other key is a
    wholesale override when the overlay sets it. `remove_field_types` (an
    overlay-only directive) drops entries from both dicts after merging.
    """
    merged = dict(base_fm)
    for key, value in overlay_fm.items():
        if key in OVERLAY_DIRECTIVE_KEYS:
            continue
        if key in ("field_types", "field_bindings"):
            combined = dict(merged.get(key) or {})
            combined.update(value or {})
            merged[key] = combined
        else:
            merged[key] = value

    for name in (overlay_fm.get("remove_field_types") or []):
        merged.get("field_types", {}).pop(name, None)
        merged.get("field_bindings", {}).pop(name, None)

    return merged


def _split_dim_sections(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a spec body into (prefix before first `### dim`, [(name, section text), ...]).

    Returns a list, not a dict — duplicate `### name` headers within one body
    are preserved as separate entries so `_validate_dimension_uniqueness` can
    still catch them; only the extends-merge path collapses by name. Each
    section's text spans from its own `### name` header up to (but not
    including) the next `### ` header or an intervening `## ` family heading.
    """
    headers = list(_DIM_HEADER_RE.finditer(body))
    if not headers:
        return body, []
    prefix = body[:headers[0].start()]
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        next_h2 = re.search(r"^##\s+\w", body[m.end():end], re.MULTILINE)
        if next_h2:
            end = min(end, m.end() + next_h2.start())
        sections.append((m.group(1), body[start:end]))
    return prefix, sections


def _merge_dimension_sections(base_body: str, overlay_body: str, remove_names: set[str]) -> str:
    prefix, base_list = _split_dim_sections(base_body)
    _, overlay_list = _split_dim_sections(overlay_body)
    base_sections = dict(base_list)      # extends-merge assumes each side is well-formed
    overlay_sections = dict(overlay_list)  # (no internal dupes) — that's checked at parse time

    merged = dict(base_sections)
    merged.update(overlay_sections)  # overlay replaces same-name sections, adds new ones
    for name in remove_names:
        merged.pop(name, None)

    # Preserve base ordering for inherited/overridden dims; append overlay-only
    # dims (new front-specific checks with no base analog) at the end.
    order = [n for n in base_sections if n in merged]
    order += [n for n in overlay_sections if n not in base_sections and n in merged]

    return prefix + "".join(merged[n] for n in order)


def _build_spec(fm: dict, body: str, path: Path, issues: list[SpecIssue]) -> V18Spec:
    # Reject unknown frontmatter keys (strict mode)
    unknown = set(fm.keys()) - KNOWN_FRONTMATTER_KEYS
    if unknown:
        issues.append(SpecIssue(
            "error", "frontmatter",
            f"Unknown frontmatter key(s): {sorted(unknown)}. "
            f"Known keys: {sorted(KNOWN_FRONTMATTER_KEYS)}.",
        ))

    # Version gate — refuse anything older than v18.
    version = str(fm.get("version", ""))
    if not version:
        issues.append(SpecIssue("error", "frontmatter", "`version` is required."))
    else:
        major = _parse_major_version(version)
        if major is None:
            issues.append(SpecIssue(
                "error", "frontmatter",
                f"Cannot parse major version from `version: {version!r}`.",
            ))
        elif major < MIN_SPEC_VERSION_MAJOR:
            issues.append(SpecIssue(
                "error", "frontmatter",
                f"This framework requires spec version {MIN_SPEC_VERSION_MAJOR}+. "
                f"Got version {version}. (v17 specs are not supported.)",
            ))

    # Required fields
    name = str(fm.get("name", "")).strip()
    if not name:
        issues.append(SpecIssue("error", "frontmatter", "`name` is required."))

    # Models
    models_raw = fm.get("models", [])
    models = _parse_models(models_raw, issues)

    # field_types
    field_types = fm.get("field_types", {}) or {}
    if not isinstance(field_types, dict):
        issues.append(SpecIssue("error", "field_types", "`field_types` must be a mapping."))
        field_types = {}

    # field_bindings
    field_bindings = _parse_field_bindings(fm.get("field_bindings", {}) or {}, issues)

    # synthesis
    synthesis = _parse_synthesis(fm.get("synthesis", {}) or {}, issues)

    # calibration
    calibration = _parse_calibration(fm.get("calibration", {}) or {}, issues)

    # severity_scale
    severity_scale = _parse_severity_scale(fm.get("severity_scale", {}) or {}, issues)

    # require_evidence_quote
    rev = fm.get("require_evidence_quote", True)
    if not isinstance(rev, bool):
        issues.append(SpecIssue("error", "require_evidence_quote",
                                 "`require_evidence_quote` must be a boolean."))
        rev = True

    # confidence_demotion
    cd = _parse_confidence_demotion(fm.get("confidence_demotion", {}) or {}, issues)

    # verdict_tiers
    verdict_tiers = fm.get("verdict_tiers", [
        "reject_structural", "reject_fixable", "needs_review",
        "accept_with_fixes", "accept",
    ])
    if not isinstance(verdict_tiers, list):
        issues.append(SpecIssue("error", "verdict_tiers",
                                 "`verdict_tiers` must be a list."))
        verdict_tiers = []

    # Dimensions
    dims = _parse_dimensions(body, issues)
    _validate_dimension_uniqueness(dims, issues)

    return V18Spec(
        name=name,
        version=version,
        cache_bust=fm.get("cache_bust"),
        models=models,
        field_types=field_types,
        field_bindings=field_bindings,
        synthesis=synthesis,
        calibration=calibration,
        severity_scale=severity_scale,
        require_evidence_quote=rev,
        confidence_demotion=cd,
        verdict_tiers=list(verdict_tiers),
        dimensions=dims,
    )


def _parse_major_version(version: str) -> int | None:
    m = re.match(r"^(\d+)", version.strip())
    return int(m.group(1)) if m else None


def _parse_models(raw, issues: list[SpecIssue]) -> list[JudgeModelConfig]:
    if not isinstance(raw, list) or not raw:
        issues.append(SpecIssue("error", "models",
                                 "`models` must be a non-empty list."))
        return []
    out: list[JudgeModelConfig] = []
    for i, m in enumerate(raw):
        if not isinstance(m, dict) or "model" not in m:
            issues.append(SpecIssue("error", f"models[{i}]",
                                     "Each entry must be a mapping with at least a `model` field."))
            continue
        out.append(JudgeModelConfig(
            model=str(m["model"]),
            reasoning_effort=m.get("reasoning_effort"),
            max_tokens=m.get("max_tokens"),
        ))
    return out


def _parse_field_bindings(raw: dict, issues: list[SpecIssue]) -> list[FieldBinding]:
    out: list[FieldBinding] = []
    if not isinstance(raw, dict):
        issues.append(SpecIssue("error", "field_bindings",
                                 "`field_bindings` must be a mapping."))
        return out
    for field_name, binding in raw.items():
        if not isinstance(binding, dict):
            issues.append(SpecIssue("error", f"field_bindings.{field_name}",
                                     "Each binding must be a mapping."))
            continue
        source = binding.get("source")
        if source not in ("file", "directory", "callback", "text"):
            issues.append(SpecIssue("error", f"field_bindings.{field_name}.source",
                                     f"`source` must be one of file/directory/callback/text "
                                     f"(got {source!r})."))
            continue
        out.append(FieldBinding(
            field_name=field_name,
            source=source,
            path=binding.get("path"),
            exclude=list(binding.get("exclude") or []),
            callback=binding.get("callback"),
            optional=bool(binding.get("optional", False)),
        ))
    return out


def _parse_synthesis(raw: dict, issues: list[SpecIssue]) -> SynthesisConfig:
    if not isinstance(raw, dict):
        issues.append(SpecIssue("error", "synthesis",
                                 "`synthesis` must be a mapping."))
        return SynthesisConfig()
    cap = raw.get("top_issues_cap", 3)
    if not isinstance(cap, int) or cap <= 0:
        issues.append(SpecIssue("error", "synthesis.top_issues_cap",
                                 "`top_issues_cap` must be a positive integer."))
        cap = 3
    return SynthesisConfig(
        enabled=bool(raw.get("enabled", True)),
        top_issues_cap=cap,
        judge_model=str(raw.get("judge_model", "gpt-5.5")),
        judge_reasoning_effort=str(raw.get("judge_reasoning_effort", "high")),
        judge_max_tokens=int(raw.get("judge_max_tokens", 16000)),
    )


def _parse_calibration(raw: dict, issues: list[SpecIssue]) -> CalibrationConfig:
    if not isinstance(raw, dict):
        issues.append(SpecIssue("error", "calibration",
                                 "`calibration` must be a mapping."))
        return CalibrationConfig()
    return CalibrationConfig(
        enabled=bool(raw.get("enabled", True)),
        goldens_path=raw.get("goldens_path"),
    )


def _parse_severity_scale(raw: dict, issues: list[SpecIssue]) -> dict[int, str]:
    if not isinstance(raw, dict):
        issues.append(SpecIssue("error", "severity_scale",
                                 "`severity_scale` must be a mapping of int → str."))
        return {}
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            ki = int(k)
        except (TypeError, ValueError):
            issues.append(SpecIssue("error", "severity_scale",
                                     f"Severity key {k!r} must be an integer."))
            continue
        if ki < 0 or ki > 4:
            issues.append(SpecIssue("error", "severity_scale",
                                     f"Severity {ki} out of range 0-4."))
            continue
        out[ki] = str(v)
    # Require at least levels 0-3
    for lvl in (0, 1, 2, 3):
        if lvl not in out:
            issues.append(SpecIssue("error", "severity_scale",
                                     f"Severity level {lvl} missing from `severity_scale`."))
    return out


def _parse_confidence_demotion(raw: dict, issues: list[SpecIssue]) -> ConfidenceDemotion:
    if not isinstance(raw, dict):
        issues.append(SpecIssue("error", "confidence_demotion",
                                 "`confidence_demotion` must be a mapping."))
        return ConfidenceDemotion()
    return ConfidenceDemotion(
        low_at_sev1=str(raw.get("low_at_sev1", "demote_to_0")),
        low_at_sev2_or_higher=str(raw.get("low_at_sev2_or_higher", "route_to_review")),
    )


# ── Dimension parsing ──────────────────────────────────────────────────────

# Per-dim sections start with `### <name>` and end at the next `### ` or `## ` heading
# (or end of file). Top-level family headings (`## Task Structure`, etc.) are ignored
# for dim extraction.
#
# Inside each dim section we expect (in any order):
#   - one ```yaml block holding the structured config
#   - prose markdown describing the criterion, severity rubric, hard fails, anti-overcall

_DIM_HEADER_RE   = re.compile(r"^###\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", re.MULTILINE)
_BLOCK_BOUNDARY_RE = re.compile(r"^(###\s+\w|##\s+\w)", re.MULTILINE)


def _parse_dimensions(body: str, issues: list[SpecIssue]) -> list[V18Dimension]:
    _, sections = _split_dim_sections(body)
    if not sections:
        issues.append(SpecIssue("error", "dimensions",
                                 "No `### <dimension_name>` sections found in body."))
        return []

    dims: list[V18Dimension] = []
    for name, full_section in sections:
        # Strip the `### name` header line itself — _parse_dim_section wants
        # only what follows it (the yaml config block + prose). Keep the
        # leading newline so this matches the original body[m.end():...] slice.
        nl = full_section.find("\n")
        section = full_section[nl:] if nl != -1 else ""
        dim = _parse_dim_section(name, section, issues)
        if dim is not None:
            dims.append(dim)
    return dims


def _parse_dim_section(name: str, section: str, issues: list[SpecIssue]) -> V18Dimension | None:
    # Pull out the ```yaml block. v18 requires exactly one per dim.
    yaml_blocks = re.findall(r"```yaml\s*\n(.*?)\n```", section, re.DOTALL)
    if not yaml_blocks:
        issues.append(SpecIssue("error", f"dim:{name}",
                                 "Missing ```yaml config block."))
        return None
    if len(yaml_blocks) > 1:
        issues.append(SpecIssue("error", f"dim:{name}",
                                 f"Expected exactly one ```yaml block, got {len(yaml_blocks)}."))
        return None
    try:
        cfg = yaml.safe_load(yaml_blocks[0]) or {}
    except yaml.YAMLError as exc:
        issues.append(SpecIssue("error", f"dim:{name}",
                                 f"Invalid YAML config: {exc}"))
        return None
    if not isinstance(cfg, dict):
        issues.append(SpecIssue("error", f"dim:{name}",
                                 "YAML config must be a mapping."))
        return None

    # Reject unknown keys
    unknown = set(cfg.keys()) - KNOWN_DIM_KEYS
    if unknown:
        issues.append(SpecIssue("error", f"dim:{name}",
                                 f"Unknown config key(s): {sorted(unknown)}. "
                                 f"Known: {sorted(KNOWN_DIM_KEYS)}."))

    # Required: dimension_name, check_kind, inputs, gate
    dimension_name = cfg.get("dimension_name")
    if not dimension_name:
        issues.append(SpecIssue("error", f"dim:{name}.dimension_name",
                                 "Required."))
    check_kind = cfg.get("check_kind", "llm")
    if check_kind not in VALID_CHECK_KINDS:
        issues.append(SpecIssue("error", f"dim:{name}.check_kind",
                                 f"Must be one of {sorted(VALID_CHECK_KINDS)} (got {check_kind!r})."))
        return None
    inputs = cfg.get("inputs", [])
    if not isinstance(inputs, list):
        issues.append(SpecIssue("error", f"dim:{name}.inputs",
                                 "Must be a list of field names."))
        inputs = []

    # gate
    gate_raw = cfg.get("gate", {})
    gate = _parse_gate(gate_raw, f"dim:{name}.gate", issues)
    if gate is None:
        return None

    # applies_when (optional)
    aw_raw = cfg.get("applies_when")
    applies_when = None
    if aw_raw is not None:
        if not isinstance(aw_raw, dict):
            issues.append(SpecIssue("error", f"dim:{name}.applies_when",
                                     "Must be a mapping."))
        else:
            unknown_aw = set(aw_raw.keys()) - KNOWN_APPLIES_WHEN_KEYS
            if unknown_aw:
                issues.append(SpecIssue("error", f"dim:{name}.applies_when",
                                         f"Unknown key(s): {sorted(unknown_aw)}."))
            applies_when = AppliesWhen(
                builtin=aw_raw.get("builtin"),
                callback=aw_raw.get("callback"),
            )

    # callback (required if check_kind=programmatic)
    callback = cfg.get("callback")
    if check_kind == "programmatic" and not callback:
        issues.append(SpecIssue("error", f"dim:{name}",
                                 "Programmatic dims require a `callback`."))

    # Extract prose (everything after the yaml block, trimmed)
    prose_start = section.find("```", section.find("```yaml") + len("```yaml")) + len("```")
    prose = section[prose_start:].strip()

    return V18Dimension(
        name=name,
        dimension_name=str(dimension_name) if dimension_name else name,
        check_kind=check_kind,
        inputs=list(inputs),
        gate=gate,
        prose=prose,
        applies_when=applies_when,
        callback=callback,
        config=dict(cfg.get("config") or {}),
        eval_config=cfg.get("eval_config"),
    )


def _parse_gate(raw, loc: str, issues: list[SpecIssue]) -> GateConfig | None:
    if not isinstance(raw, dict):
        issues.append(SpecIssue("error", loc, "Must be a mapping."))
        return None
    unknown = set(raw.keys()) - KNOWN_GATE_KEYS
    if unknown:
        issues.append(SpecIssue("error", loc,
                                 f"Unknown gate key(s): {sorted(unknown)}."))
    role = raw.get("role")
    if role not in VALID_GATE_ROLES:
        issues.append(SpecIssue("error", f"{loc}.role",
                                 f"Must be one of {sorted(VALID_GATE_ROLES)} (got {role!r})."))
        return None
    sev = raw.get("reject_at_severity", 3)
    if not isinstance(sev, int) or sev < 1 or sev > 4:
        issues.append(SpecIssue("error", f"{loc}.reject_at_severity",
                                 "Must be an integer in 1..4."))
        return None
    on_dis = raw.get("on_disagreement", "review")
    if on_dis not in VALID_ON_DISAGREE:
        issues.append(SpecIssue("error", f"{loc}.on_disagreement",
                                 f"Must be one of {sorted(VALID_ON_DISAGREE)} (got {on_dis!r})."))
        return None
    return GateConfig(role=role, reject_at_severity=sev, on_disagreement=on_dis)


def _validate_dimension_uniqueness(dims: list[V18Dimension], issues: list[SpecIssue]) -> None:
    seen: set[str] = set()
    for d in dims:
        if d.name in seen:
            issues.append(SpecIssue("error", f"dim:{d.name}",
                                     f"Duplicate dimension name."))
        seen.add(d.name)

"""Built-in programmatic callbacks for the v18 spec's deterministic dims.

Four callbacks map to the four programmatic dims in v18:

  check_metadata_completeness         → metadata_completeness
  check_verifier_parseable            → verifier_spec_parseable
  check_verifier_count_in_range       → verifier_count_in_range
  check_verifier_outcome_grader_present → verifier_outcome_grader_present

Each callback inspects sample fields without any LLM call and emits a
DimensionFinding with a deterministic severity, fixability, and headline.
"""

from __future__ import annotations

import json

from .finding import DimensionFinding
from .v18_loader import Sample
from .v18_spec import V18Dimension


# ── Helpers ────────────────────────────────────────────────────────────────

def _ok(dim: V18Dimension, headline: str = "no issue") -> DimensionFinding:
    return DimensionFinding(
        dimension=dim.name,
        dimension_name=dim.dimension_name,
        gate_role=dim.gate.role,
        applicable=True,
        severity=0,
        fixability="n/a",
        confidence="high",
        headline=headline,
        detail="Programmatic check passed.",
        evidence_quote=None,
        judges_agreed=True,
    )


def _fail(
    dim: V18Dimension,
    severity: int,
    headline: str,
    detail: str,
    fixability: str = "fixable",
) -> DimensionFinding:
    return DimensionFinding(
        dimension=dim.name,
        dimension_name=dim.dimension_name,
        gate_role=dim.gate.role,
        applicable=True,
        severity=severity,
        fixability=fixability,           # type: ignore
        confidence="high",
        headline=headline,
        detail=detail,
        evidence_quote=None,
        judges_agreed=True,
    )


# ── Callbacks ──────────────────────────────────────────────────────────────

def check_metadata_completeness(sample: Sample, dim: V18Dimension) -> DimensionFinding:
    """Severity 2 when `category` is empty; otherwise 0."""
    category = sample.fields.get("category")
    if not category or not str(category).strip():
        return _fail(
            dim, severity=2,
            headline="category is empty",
            detail="task.toml [metadata].category (or .domain) is empty or missing.",
            fixability="trivial",
        )
    return _ok(dim)


def check_verifier_parseable(sample: Sample, dim: V18Dimension) -> DimensionFinding:
    """Severity 2 when verifier.json is malformed; severity 0 when it parses."""
    spec_json = sample.fields.get("verifier_spec_json")
    if not isinstance(spec_json, str) or not spec_json.strip():
        # Should not reach here when applies_when=task_has_verifier is active,
        # but handle defensively.
        return _fail(dim, severity=2,
                     headline="verifier.json is missing",
                     detail="No verifier spec was provided for this task.",
                     fixability="structural")
    try:
        data = json.loads(spec_json)
    except json.JSONDecodeError as exc:
        return _fail(
            dim, severity=2,
            headline=f"verifier.json fails to parse ({type(exc).__name__})",
            detail=f"JSON parse error: {exc}",
            fixability="fixable",
        )
    if not isinstance(data, dict):
        return _fail(dim, severity=2,
                     headline="verifier.json root is not an object",
                     detail=f"got {type(data).__name__}",
                     fixability="fixable")
    if not isinstance(data.get("verifiers"), list):
        return _fail(dim, severity=2,
                     headline="verifier.json missing `verifiers` list",
                     detail="The top-level field `verifiers` must be a list.",
                     fixability="fixable")
    return _ok(dim)


def check_verifier_count_in_range(sample: Sample, dim: V18Dimension) -> DimensionFinding:
    """Minimum-only gate: severity 0 at/above min; 1 if short by 1-2; 2 if short by more.

    There is NO upper limit on verifier count — a thorough rubric with many checks
    is not a defect. `max_verifiers: null` (or absent) means unbounded above; a
    numeric `max_verifiers` is still honoured if a spec ever sets one."""
    min_n = int(dim.config.get("min_verifiers", 6))
    max_raw = dim.config.get("max_verifiers", None)
    max_n = int(max_raw) if max_raw not in (None, "", "null") else None
    spec_json = sample.fields.get("verifier_spec_json")
    if not isinstance(spec_json, str):
        return _fail(dim, severity=2,
                     headline="verifier.json missing",
                     detail="Cannot count verifiers without a spec.")
    try:
        data = json.loads(spec_json)
        count = len(data.get("verifiers") or [])
    except Exception as exc:
        return _fail(dim, severity=2,
                     headline="verifier.json unparseable",
                     detail=str(exc))

    target = f"[{min_n}, {'∞' if max_n is None else max_n}]"
    within = count >= min_n and (max_n is None or count <= max_n)
    if within:
        return _ok(dim, headline=f"{count} verifiers (min {min_n}, no upper limit)"
                   if max_n is None else f"{count} verifiers in {target}")
    # only reachable when below min, or above a (rare) explicit max
    distance = (min_n - count) if count < min_n else (count - max_n)
    sev = 1 if distance <= 2 else 2
    return _fail(
        dim, severity=sev,
        headline=f"{count} verifiers (minimum {min_n})",
        detail=f"Verifier count {count} is below the minimum of {min_n} by {distance}; "
               f"add rubrics to improve coverage." if count < min_n else
               f"Verifier count {count} exceeds the configured maximum of {max_n} by {distance}.",
        fixability="fixable",
    )


def check_verifier_outcome_grader_present(
    sample: Sample, dim: V18Dimension,
) -> DimensionFinding:
    """Severity 0 if at least one verifier is an outcome-grader; else 2.

    Handles two verifier-spec schemas:
      - Legacy GDPVal-216: `metadata.classification == "core"` (where "core"
        implied outcome-checking).
      - Omega: `metadata.grading_role == "outcome"` (decoupled from
        importance).
    A rubric matches if EITHER schema's outcome marker is set.
    """
    spec_json = sample.fields.get("verifier_spec_json")
    if not isinstance(spec_json, str):
        return _fail(dim, severity=2,
                     headline="verifier.json missing",
                     detail="Cannot check classifications without a spec.")
    try:
        data = json.loads(spec_json)
    except Exception as exc:
        return _fail(dim, severity=2,
                     headline="verifier.json unparseable",
                     detail=str(exc))

    outcome_count = 0
    for v in (data.get("verifiers") or []):
        meta = (v or {}).get("metadata") or {}
        if meta.get("classification") == "core":
            outcome_count += 1
        elif meta.get("grading_role") == "outcome":
            outcome_count += 1

    if outcome_count >= 1:
        return _ok(dim, headline=f"{outcome_count} outcome-grading verifier(s) present")
    return _fail(
        dim, severity=2,
        headline="no outcome-grading verifiers",
        detail=("No verifier in the spec carries either "
                "metadata.classification == 'core' (legacy schema) or "
                "metadata.grading_role == 'outcome' (omega schema). "
                "A verifier without any outcome-grading rubric checks only "
                "process / format and cannot drive a meaningful reward."),
        fixability="fixable",
    )


# ── synthetic_marker_scan ────────────────────────────────────────────────────

import re as _re

# Maintained list of known synthetic pipeline markers.
# Add new patterns here as they are discovered in production.
_SYNTHETIC_MARKER_PATTERNS: list[_re.Pattern] = [
    _re.compile(r'\[TIE\]', _re.IGNORECASE),
    _re.compile(r'\[SYNTHETIC\]', _re.IGNORECASE),
    _re.compile(r'\[AI-GENERATED\]', _re.IGNORECASE),
    _re.compile(r'\[AI_GENERATED\]', _re.IGNORECASE),
    _re.compile(r'\[LLM-GENERATED\]', _re.IGNORECASE),
    _re.compile(r'\[LLM_GENERATED\]', _re.IGNORECASE),
    _re.compile(r'\[FABRICATED\]', _re.IGNORECASE),
    _re.compile(r'\[GENERATED\]', _re.IGNORECASE),
    _re.compile(r'\[OMEGA\]', _re.IGNORECASE),
    _re.compile(r'\[OMEGA-REBORN\]', _re.IGNORECASE),
    # Bracket-style metadata tags with optional content: [TAG: value]
    _re.compile(r'\[(?:TIE|SYNTHETIC|AI|LLM|FABRICATED|GENERATED|OMEGA)[^\]]*\]', _re.IGNORECASE),
    # Prose fourth-wall break: a document stating outright it's a fabricated/
    # synthetic/AI-generated artifact made for training/eval/benchmark purposes.
    # Added 2026-07-27 after a real miss: task_272's source PDF read "This
    # document is a FABRICATED artifact created solely for a financial-analysis
    # training case study" — no bracket tag, so the patterns above never fired.
    # Deliberately narrow (see consistency_ledger.py's docstring for the skills'
    # copy of this pattern and why a broader "meta-commentary" regex was
    # rejected as unsafe): real business prose legitimately uses "model",
    # "training", etc. all the time, so this only matches the unambiguous
    # self-disclosure phrasing, not any directive-style sentence.
    _re.compile(
        r'\b(fabricat\w*|synthetic(?:ally)?|ai[- ]generated|llm[- ]generated)\b'
        r'[^.]{0,60}\b(training|case study|benchmark|eval(?:uation)?|exercise)\b',
        _re.IGNORECASE),
    _re.compile(
        r'\b(training|case study|benchmark|eval(?:uation)?)\b[^.]{0,60}'
        r'\b(fabricat\w*|synthetic(?:ally)?|ai[- ]generated|llm[- ]generated)\b',
        _re.IGNORECASE),
]


def check_synthetic_markers(sample: Sample, dim: V18Dimension) -> DimensionFinding:
    """Sev 2 if any input file contains a known synthetic pipeline marker."""
    input_files = sample.fields.get("input_files")
    if not input_files:
        return _ok(dim, headline="no input files to scan")

    hits: list[tuple[str, str, str]] = []  # (filename, pattern, matched_text)

    for file_obj in input_files:
        filename = getattr(file_obj, 'name', getattr(file_obj, 'filename', 'unknown'))
        text = getattr(file_obj, 'text', getattr(file_obj, 'content', ''))
        if not text:
            continue
        for pattern in _SYNTHETIC_MARKER_PATTERNS:
            match = pattern.search(text)
            if match:
                hits.append((filename, pattern.pattern, match.group(0)))
                break  # one hit per file is enough

    if not hits:
        return _ok(dim)

    hit_count = len(hits)
    files_affected = ', '.join(h[0] for h in hits)
    first_match = hits[0][2]

    return _fail(
        dim,
        severity=2,
        headline=f"synthetic pipeline marker found in {hit_count} input file(s)",
        detail=(
            f"Found synthetic generation marker(s) in: {files_affected}. "
            f"First match: '{first_match}'. These bracket-style tags are artifacts "
            f"of the task creation pipeline and must not appear in source documents "
            f"presented to the agent. Remove or scrub the markers from the affected files."
        ),
        fixability="fixable",
    )


# ── Consistency-ledger checks (ported 2026-07-27 from the QC-skills'
#    consistency_ledger.py, where they were built and validated against real
#    task files) ─────────────────────────────────────────────────────────────
#
# These four checks close specific, confirmed recall gaps found in a 12-task
# audit: round-1 QC missed real, verifiable-at-the-time defects that these
# deterministic checks catch mechanically. Each aggregates every candidate it
# finds across input_files + solution_files (+ solution_reasoning where noted)
# into ONE DimensionFinding; see the skills' consistency_ledger.py docstring
# for the full detector rationale (near-miss tolerance, sum-check block
# boundaries, ratio-invariant column pairing, citation ground truth).

_NUM_RE = _re.compile(
    r"\(?\s*\$?\s*-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
    r"\(?\s*\$?\s*-?\d+\.\d+|"
    r"\(?\s*\$?\s*-?\d+",
    _re.ASCII)
_SCALE = {"k": 1e3, "m": 1e6, "mm": 1e6, "bn": 1e9, "b": 1e9}
_GENERIC_LABELS = {"total", "amount", "value", "sum", "subtotal", "check", "note",
                   "notes", "date", "name", "id", "no", "n/a", ""}
_TOTAL_RE = _re.compile(r"\b(total|subtotal|grand total|sum)\b", _re.I)
_NOT_A_SUM_RE = _re.compile(r"\b(average|avg|weighted|mean|median|per\b|margin|ratio|rate|yield)\b|%", _re.I)
_DOC_CITE_RE = _re.compile(r"\b(?:doc|document)\.?\s*#?\s*0*(\d{1,3})\b", _re.I)
_FILENAME_LEADING_NUM_RE = _re.compile(r"^(\d{1,3})[_\-\s]+(.+)$")
_CITATION_GENERIC_WORDS = {
    "the", "and", "for", "doc", "document", "documents", "file", "files",
    "report", "reports", "note", "notes", "of", "statement", "statements",
    "excerpt", "excerpts", "summary", "package", "schedule", "reference",
    "data", "pack", "sheet", "sheets", "pdf", "docx", "xlsx",
}


class _Num:
    __slots__ = ("value", "raw", "is_pct", "frac", "is_year")

    def __init__(self, value, raw, is_pct, frac, is_year):
        self.value = value; self.raw = raw; self.is_pct = is_pct
        self.frac = frac; self.is_year = is_year


def _parse_number(tok: str, trailing: str = ""):
    raw = tok.strip()
    neg = raw.startswith("(")
    had_dollar = "$" in raw
    had_comma = "," in raw
    is_pct = raw.rstrip().endswith("%") or trailing.strip().startswith("%")
    s = raw.replace("(", "").replace(")", "").replace("$", "").replace(",", "").replace("%", "").strip()
    if not s or s in ("-", "."):
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    scale = _SCALE.get(trailing.strip().lower()) if trailing else None
    if scale:
        val *= scale
    if neg:
        val = -abs(val)
    frac = abs(val - round(val)) > 1e-9
    is_year = (not frac and not is_pct and not had_dollar and not had_comma
               and not scale and 1900 <= val <= 2099)
    return _Num(val, raw, is_pct, frac, is_year)


def _numbers_in(text: str) -> list:
    out = []
    for m in _NUM_RE.finditer(text):
        tail = text[m.end():m.end() + 3]
        mt = _re.match(r"\s*(mm|bn|[kmb])\b", tail, _re.I)
        n = _parse_number(m.group(0), mt.group(1) if mt else "")
        if n is not None:
            out.append(n)
    return out


def _norm_label(label: str) -> str:
    lab = _re.sub(r"\s+", " ", label.strip().lower())
    lab = lab.strip(" :–-•*").strip()
    lab = _re.sub(r"\s*\((?:\$?mm|\$?m|\$?k|\$?bn|usd|%|000s?)\)\s*$", "", lab)
    return lab


def _decimals(raw: str) -> int:
    s = raw.replace("(", "").replace(")", "").replace("$", "").replace("%", "").strip()
    return len(s.split(".", 1)[1]) if "." in s else 0


def _rounding_match(a: float, ra: str, b: float, rb: str) -> bool:
    d = min(_decimals(ra), _decimals(rb))
    return round(a, d) == round(b, d)


def _looks_like_id(label_key: str) -> bool:
    return " " not in label_key and any(c.isdigit() for c in label_key) and len(label_key) <= 20


def _distinctive(label_key: str) -> bool:
    if len(label_key) < 4:
        return False
    words = [w for w in _re.split(r"[^a-z]+", label_key) if w]
    return any(w not in _GENERIC_LABELS and len(w) >= 3 for w in words)


def _distinctive_prose(label_key: str) -> bool:
    if not _distinctive(label_key):
        return False
    words = [w for w in _re.split(r"[^a-z]+", label_key) if w]
    non_generic = [w for w in words if w not in _GENERIC_LABELS and len(w) >= 3]
    return len(non_generic) >= 2


class _Cell:
    __slots__ = ("label", "nums", "section", "file", "lineno", "raw")

    def __init__(self, label, nums, section, file, lineno, raw):
        self.label = label; self.nums = nums; self.section = section
        self.file = file; self.lineno = lineno; self.raw = raw


def _is_tabular(file_rel: str) -> bool:
    return file_rel.lower().endswith((".xlsx", ".csv", ".tsv"))


def _rows_from_text(text: str, file_rel: str) -> list:
    rows = []
    section = ""
    for i, line in enumerate(text.splitlines(), 1):
        s = line.rstrip()
        if not s.strip():
            continue
        m = _re.match(r"=== Sheet: (.*) ===", s)
        if m:
            section = m.group(1).strip()
            continue
        if _re.match(r"^\s*\|.*\|\s*$", s) and not _re.match(r"^\s*\|[\s:\-|]+\|\s*$", s):
            # markdown table row ('| Commercial Auto (CA) | 71,610 | **81,489** |...'),
            # excluding the header-separator row ('|---|---|'). Strip markdown bold
            # so a wrapped numeric cell ('**81,489**') is still recognized as numeric.
            cells = [c.strip().strip("*") for c in s.strip().strip("|").split("|")]
        elif "\t" in s:
            cells = s.split("\t")
        else:
            mm = _re.split(r":\s+|\s{2,}", s, maxsplit=1)
            cells = mm if len(mm) == 2 else [s]
        label_parts = [c for c in cells if not _NUM_RE.fullmatch(c.strip().replace(",", ""))
                       and c.strip() and not _re.fullmatch(r"[\$%\s.,()-]+", c.strip())]
        label = label_parts[0].strip() if label_parts else ""
        nums = _numbers_in(s)
        rows.append(_Cell(label, nums, section, file_rel, i, s.strip()[:120]))
    return rows


def _all_rows(sample: Sample) -> list:
    """Rows from input_files + solution_files + solution_reasoning, in the
    shared '=== Sheet: X ===' / tab-separated format v18_loader.py's
    _xlsx_to_text emits (and plain text/prose files pass through as-is)."""
    rows = []
    for field_name in ("input_files", "solution_files"):
        for fo in (sample.fields.get(field_name) or []):
            text = getattr(fo, "text", "") or ""
            name = getattr(fo, "name", "unknown")
            if text:
                rows.extend(_rows_from_text(text, name))
    reasoning = sample.fields.get("solution_reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        rows.extend(_rows_from_text(reasoning, "Reasoning.docx"))
    return rows


def _all_grids(sample: Sample) -> list:
    """(file_rel, sheet, grid) tuples reconstructed from the same flattened
    text (no direct openpyxl access to Sample files — only extracted text is
    available), by re-splitting each '=== Sheet: X ===' block's tab-separated
    lines back into a 2D grid."""
    grids = []
    for field_name in ("input_files", "solution_files"):
        for fo in (sample.fields.get(field_name) or []):
            text = getattr(fo, "text", "") or ""
            name = getattr(fo, "name", "unknown")
            if not name.lower().endswith((".xlsx", ".csv", ".tsv")) or not text:
                continue
            sheet, grid = "", []
            for line in text.splitlines():
                m = _re.match(r"=== Sheet: (.*) ===", line)
                if m:
                    if grid:
                        grids.append((name, sheet, grid))
                    sheet, grid = m.group(1).strip(), []
                    continue
                if "\t" in line or line.strip():
                    grid.append(line.split("\t") if "\t" in line else [line])
            if grid:
                grids.append((name, sheet, grid))
    return grids


def _coerce_cell(cell):
    if cell is None or cell == "":
        return None
    s = str(cell).strip()
    if not s:
        return None
    t = s.replace("$", "").replace(",", "").replace("%", "").strip()
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        v = float(t)
        return -v if neg else v
    except ValueError:
        return s


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _detect_near_miss(rows: list, near_tol: float = 0.01, min_mag: float = 1000.0):
    by_label = {}
    for r in rows:
        is_tab = _is_tabular(r.file)
        key = _norm_label(r.label)
        distinctive_ok = _distinctive(key) if is_tab else _distinctive_prose(key)
        if not distinctive_ok or _looks_like_id(key):
            continue
        for n in r.nums:
            if n.is_pct or n.is_year:
                continue
            loc = (r.file, r.section)
            rv = round(n.value, 4)
            by_label.setdefault(key, {}).setdefault(rv, (r, n.raw, loc))
    cands = []
    for key, uniq in by_label.items():
        vals = sorted(uniq)
        flagged = set()
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a, b = vals[i], vals[j]
                hi = max(abs(a), abs(b))
                if hi < min_mag or a == b:
                    continue
                if uniq[a][2] == uniq[b][2]:
                    continue
                if _rounding_match(a, uniq[a][1], b, uniq[b][1]):
                    continue
                if abs(a - b) / hi <= near_tol:
                    flagged.update([a, b])
        if flagged and len(flagged) <= 4:
            seen = sorted(flagged)
            values = [{"raw": uniq[v][1], "file": uniq[v][0].file,
                       "location": uniq[v][0].section or f"line {uniq[v][0].lineno}"} for v in seen]
            cands.append({"label": key, "values": values,
                          "note": f"same label carries {len(seen)} near-but-unequal values "
                                  f"(within {near_tol:.0%}) at different locations"})
    return cands


def _check_sum_block(block, file_rel, sheet, tol_frac, cands):
    if len(block) < 3:
        return
    totals = [k for k, (_i, lab, _n) in enumerate(block) if lab and _TOTAL_RE.search(lab)]
    if len(totals) != 1 or totals[0] != len(block) - 1:
        return
    _ti, tlabel, tnums = block[-1]
    if tlabel and _NOT_A_SUM_RE.search(tlabel):
        return
    addends = block[:-1]
    for col, tval in tnums.items():
        if abs(tval) < 1:
            continue
        col_vals = [nums[col] for (_i, _l, nums) in addends if col in nums]
        if len(col_vals) < 2:
            continue
        if max(abs(v) for v in col_vals) >= 0.98 * abs(tval):
            continue
        s = sum(col_vals)
        ratio = abs(s) / abs(tval) if tval else 0
        if ratio >= 1.9 and abs(ratio - round(ratio)) < 0.02:
            continue
        if abs(tval - s) > max(1.0, tol_frac * abs(tval)):
            cands.append({"label": _norm_label(tlabel), "file": file_rel,
                          "location": sheet or f"row {block[-1][0]}",
                          "note": f"stated total {round(tval, 2)} != sum of parts {round(s, 2)} "
                                  f"(Δ {round(tval - s, 2)})"})


def _detect_sum_check(grids: list, tol_frac: float = 0.005):
    cands = []
    for file_rel, sheet, grid in grids:
        block = []
        for i, row in enumerate(grid, 1):
            label, nums = "", {}
            for c_idx, cell in enumerate(row):
                v = _coerce_cell(cell)
                if _is_num(v):
                    nums[c_idx] = v
                elif isinstance(v, str) and v.strip() and not label:
                    label = v.strip()
            if not nums:
                _check_sum_block(block, file_rel, sheet, tol_frac, cands)
                block = []
                continue
            block.append((i, label, nums))
        _check_sum_block(block, file_rel, sheet, tol_frac, cands)
    return cands


def _norm_header_token(h) -> str:
    return _re.sub(r"[^a-z0-9]+", "_", str(h).lower()).strip("_")


def _find_ratio_column_pairs(headers):
    pairs = []
    norm = [_norm_header_token(h) for h in headers]
    for j, nh in enumerate(norm):
        m = _re.search(r"(.+?)_per_(.+)", nh)
        if not m:
            continue
        denom_hint = m.group(2)
        for i, nh2 in enumerate(norm):
            if i == j:
                continue
            if denom_hint in nh2 or denom_hint.rstrip("s") in nh2:
                pairs.append((i, j, headers[i], headers[j]))
                break
    return pairs


def _detect_ratio_check(grids: list):
    cands = []
    for file_rel, sheet, grid in grids:
        if not grid or not grid[0]:
            continue
        headers = grid[0]
        pairs = _find_ratio_column_pairs(headers)
        if not pairs:
            continue
        for denom_i, ratio_j, denom_h, ratio_h in pairs:
            hits = []
            for r_i, row in enumerate(grid[1:], 2):
                if denom_i >= len(row) or ratio_j >= len(row):
                    continue
                denom_v, ratio_v = _coerce_cell(row[denom_i]), _coerce_cell(row[ratio_j])
                if (_is_num(denom_v) and denom_v == 0
                        and _is_num(ratio_v) and ratio_v != 0):
                    row_label = next((str(c) for c in row if isinstance(c, str) and c.strip()), "")
                    hits.append((r_i, row_label, ratio_v))
            if hits:
                cands.append({"label": f"{ratio_h} vs {denom_h}", "file": file_rel,
                              "location": sheet, "hits": hits,
                              "note": f"{len(hits)} row(s) where {denom_h}=0 but {ratio_h} is "
                                      f"nonzero — a per-unit metric over a zero-unit base"})
    return cands


def _title_tokens(name: str) -> set:
    stem = _re.sub(r"\.\w+$", "", name)
    stem = _re.sub(r"^\d{1,3}[_\-\s]+", "", stem)
    words = _re.split(r"[^a-z0-9]+", stem.lower())
    return {w for w in words if w and w not in _CITATION_GENERIC_WORDS and len(w) >= 4}


def _doc_ground_truth(filenames: list) -> dict:
    gt = {}
    for name in filenames:
        m = _FILENAME_LEADING_NUM_RE.match(name)
        if m:
            gt[int(m.group(1))] = _title_tokens(name)
    return gt


def _declared_aliases(rows: list) -> dict:
    aliases = {}
    for r in rows:
        nums = {int(m.group(1)) for m in _DOC_CITE_RE.finditer(r.raw)}
        if len(nums) >= 2:
            aliases.setdefault(r.file, set()).update(nums)
    return aliases


def _detect_citation_check(rows: list, filenames: list):
    gt = _doc_ground_truth(filenames)
    if len(gt) < 2:
        return []
    aliases = _declared_aliases(rows)
    cands = []
    for r in rows:
        for m in _DOC_CITE_RE.finditer(r.raw):
            claimed = int(m.group(1))
            if claimed not in gt or claimed in aliases.get(r.file, ()):
                continue
            label = r.raw[:m.start()].strip(" \t-:|") or r.label
            lab_tok = _title_tokens(label)
            if not lab_tok:
                continue
            scores = {n: len(lab_tok & toks) for n, toks in gt.items() if toks}
            if not scores:
                continue
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            best_n, best_s = ranked[0]
            runner_s = ranked[1][1] if len(ranked) > 1 else 0
            if best_s < 1 or best_s - runner_s < 1:
                continue
            if best_n != claimed:
                cands.append({
                    "label": label[:80], "file": r.file,
                    "note": f"citation says Doc {claimed}, but the cited title's distinctive "
                            f"words ({', '.join(sorted(lab_tok & gt[best_n]))}) match Doc {best_n}"})
    return cands


def check_near_miss(sample: Sample, dim: V18Dimension) -> DimensionFinding:
    """Same distinctive label carrying two near-but-unequal values at different
    file/sheet locations — the 'same fact rendered slightly differently'
    signature. Advisory (severity 1): flags a real cross-file/within-file
    reconciliation gap, but the two values' true relationship (typo vs. two
    legitimately close-but-different figures) needs a human/LLM read."""
    rows = _all_rows(sample)
    cands = _detect_near_miss(rows)
    if not cands:
        return _ok(dim)
    c = cands[0]
    return _fail(
        dim, severity=1,
        headline=f'"{c["label"]}" carries {len(c["values"])} near-but-unequal values across files',
        detail=(f"{c['note']}. " + "; ".join(
            f"{v['raw']} ({v['file']} · {v['location']})" for v in c["values"])
            + (f" ({len(cands) - 1} more label(s) also flagged)" if len(cands) > 1 else "")),
        fixability="fixable",
    )


def check_sum_mismatch(sample: Sample, dim: V18Dimension) -> DimensionFinding:
    """A total/subtotal row whose value != the column sum of its addend rows,
    computed on the real spreadsheet grid. Blocking (severity 2): this is a
    pure arithmetic check with a validated low false-positive rate."""
    grids = _all_grids(sample)
    cands = _detect_sum_check(grids)
    if not cands:
        return _ok(dim)
    c = cands[0]
    return _fail(
        dim, severity=2,
        headline=f'"{c["label"]}" total does not equal the sum of its parts',
        detail=f"{c['note']} ({c['file']} · {c['location']})"
               + (f" ({len(cands) - 1} more mismatch(es) also flagged)" if len(cands) > 1 else ""),
        fixability="fixable",
    )


def check_ratio_invariant(sample: Sample, dim: V18Dimension) -> DimensionFinding:
    """A '<X>_per_<Y>' column is nonzero on a row where its own <Y> denominator
    is zero — a per-unit metric is undefined, not nonzero, over a zero-unit
    base. Pure arithmetic on the column's own header-declared definition, no
    domain knowledge required. Blocking (severity 2)."""
    grids = _all_grids(sample)
    cands = _detect_ratio_check(grids)
    if not cands:
        return _ok(dim)
    c = cands[0]
    total_hits = sum(len(x["hits"]) for x in cands)
    return _fail(
        dim, severity=2,
        headline=f'"{c["label"]}" nonzero over a zero-unit base ({total_hits} row(s) across '
                 f"{len(cands)} column pair(s))",
        detail=c["note"] + f" ({c['file']} · {c['location']})",
        fixability="fixable",
    )


def check_doc_citation(sample: Sample, dim: V18Dimension) -> DimensionFinding:
    """A 'Doc NN' citation whose title's distinctive words clearly match a
    DIFFERENT doc number than the one claimed, checked against real on-disk
    leading-numbered filenames. A file citing 2+ distinct doc numbers in one
    line is treated as declaring its own numbering legend and exempted for
    those numbers (see _declared_aliases). Blocking (severity 2)."""
    rows = _all_rows(sample)
    filenames = [getattr(fo, "name", "") for fo in (sample.fields.get("input_files") or [])]
    filenames += [getattr(fo, "name", "") for fo in (sample.fields.get("solution_files") or [])]
    cands = _detect_citation_check(rows, filenames)
    if not cands:
        return _ok(dim)
    c = cands[0]
    return _fail(
        dim, severity=2,
        headline=f'citation mismatch: "{c["label"]}"',
        detail=f"{c['note']} ({c['file']})"
               + (f" ({len(cands) - 1} more mismatch(es) also flagged)" if len(cands) > 1 else ""),
        fixability="fixable",
    )


# ── deliverable_manifest_check ───────────────────────────────────────────────
#
# Required deliverables are named in the instruction as a line-item list, e.g.
# "Reshoot_Funding_Plan.xlsx — the revised post budget..." or a bulleted,
# backtick-quoted variant — always following a "Produce N deliverables in
# /workspace/output/:" (or equivalent) anchor phrase. An earlier version tried
# to match filenames anywhere near trigger words like "produce"/"save as" and
# had two real bugs found during validation: (1) it could match a substring
# spanning the trigger word into an unrelated nearby filename (e.g. "produce"
# bleeding into "Producer_Memo.docx" and capturing "r_Memo.docx"), and (2) it
# had no length/section scoping, so on tasks that list SOURCE files in a
# similar bulleted format earlier in the prompt, it false-matched those too.
# This version anchors on the "Produce ... deliverables in /workspace/output/"
# phrase first, then scans only the text AFTER it (a bounded window) for
# filenames — source files mentioned earlier in the prompt are out of scope
# by construction, and the anchor's own trailing colon means the window never
# overlaps the trigger phrase itself.
_DELIVERABLES_ANCHOR_RE = _re.compile(
    r'produce\s+(?:the\s+following\s+|one\s+|two\s+|three\s+|four\s+|five\s+|\d+\s+)?'
    r'deliverables?\s+in\s+`?/workspace/output/`?\s*:?',
    _re.IGNORECASE)
_DELIVERABLE_FILENAME_RE = _re.compile(r'([A-Za-z0-9_\-]+\.(?:xlsx|docx|pdf|pptx|csv))', _re.IGNORECASE)
_DELIVERABLES_WINDOW_CHARS = 2000


def check_deliverable_manifest(sample: Sample, dim: V18Dimension) -> DimensionFinding:
    """Extracts the literal required deliverable filenames named right after
    the instruction's "Produce ... deliverables in /workspace/output/" anchor
    phrase, and diffs them against the actual solution_files listing. Flags:
    a required filename entirely absent, present with a different
    extension/case, packaged inside an archive instead of standalone, or an
    extra file present beyond what was named. Pure file-listing comparison —
    no LLM judgment needed. Blocking (severity 2). `Reasoning.docx` is
    excluded from solution_files by this spec's own field binding (surfaced
    separately as solution_reasoning), so its presence never counts as an
    extra; other reasoning-style files (e.g. reasoning.md, under a different
    name) are NOT excluded and will correctly flag as an extra if not named
    as a deliverable."""
    instruction = sample.fields.get("instruction") or ""
    anchor = _DELIVERABLES_ANCHOR_RE.search(instruction)
    if not anchor:
        return _ok(dim, headline="no 'Produce ... deliverables in /workspace/output/' anchor found")
    window = instruction[anchor.end(): anchor.end() + _DELIVERABLES_WINDOW_CHARS]
    required = list(dict.fromkeys(_DELIVERABLE_FILENAME_RE.findall(window)))  # de-dup, keep order
    if not required:
        return _ok(dim, headline="anchor found but no deliverable filenames matched after it")

    solution_files = sample.fields.get("solution_files") or []
    actual_names = {getattr(fo, "name", "") for fo in solution_files}
    actual_lower = {n.lower(): n for n in actual_names}
    required_lower = {r.lower() for r in required}

    missing, wrong_case, archived = [], [], []
    for req in required:
        if req in actual_names:
            continue
        if req.lower() in actual_lower:
            wrong_case.append((req, actual_lower[req.lower()]))
            continue
        stem = req.rsplit(".", 1)[0].lower()
        if any(n.lower().endswith(".zip") and stem in n.lower() for n in actual_names):
            archived.append(req)
            continue
        missing.append(req)
    extra = sorted(n for n in actual_names if n.lower() not in required_lower
                   and not n.lower().endswith(".zip"))

    if not (missing or wrong_case or archived or extra):
        return _ok(dim, headline=f"all {len(required)} named deliverable(s) present as specified, "
                                  f"no unrequested extras")

    parts = []
    if missing:
        parts.append(f"missing entirely: {', '.join(sorted(missing))}")
    if wrong_case:
        parts.append("; ".join(f"'{req}' required but found as '{actual}'" for req, actual in wrong_case))
    if archived:
        parts.append(f"packaged inside an archive instead of standalone: {', '.join(sorted(archived))}")
    if extra:
        parts.append(f"unrequested extra file(s) present: {', '.join(extra)}")

    return _fail(
        dim, severity=2,
        headline=f"{len(missing) + len(wrong_case) + len(archived) + len(extra)} deliverable "
                 f"mismatch(es) against the {len(required)} named in the instruction",
        detail="; ".join(parts) + f". Actual solution files: {', '.join(sorted(actual_names)) or '(none)'}.",
        fixability="fixable",
    )


# ── cross_artifact_reconciliation_check (LLM, via a programmatic callback) ──
#
# The 5 remaining confirmed-real misses that the deterministic checks above
# can't catch: a deck's table disagreeing with a workbook, a memo's aggregate
# claim not matching two files' worth of source data, a prompt claiming data
# is fully provided when it isn't, and a source document that substantially
# restates the task's own instructions. All 5 need semantic comparison across
# multiple documents — not regex-able.
#
# A single LLM call asked to "extract claims, then judge consistency" reliably
# runs the instruction *nominally* without it changing the verdict — this is
# the exact failure mode already diagnosed and fixed once, in the QC-skills'
# equivalent check (see consistency_ledger.py / SKILL.md there). The fix is
# not a better prompt on the existing single-call dimensions
# (solution_internal_consistency, instruction_input_consistency,
# input_prompt_leakage) — it's splitting extraction from judgment into two
# separate calls, where the second call never sees the raw documents, so it
# can't be swayed by how plausible everything looks in context. This is a
# `check_kind: programmatic` dimension that internally makes its own LLM
# calls (via the 3-arg callback convention added in v18_engine.py's
# `_run_programmatic` — see that file's comment) rather than a new engine
# override mechanism; it contributes to the verdict as its own independent
# dimension, the same way `synthetic_marker_scan` always has.

def _gather_reconciliation_corpus(sample: Sample) -> tuple[str, str]:
    """Returns (corpus_text, instruction_text). corpus_text is every input +
    solution file's extracted text, each capped to keep the extractor call's
    prompt a reasonable size — this dimension only needs to find a claim
    mentioned in 2+ places, not read every word of every file."""
    parts = []
    for field_name in ("input_files", "solution_files"):
        for fo in (sample.fields.get(field_name) or []):
            text = getattr(fo, "text", "") or ""
            name = getattr(fo, "name", "unknown")
            if text:
                parts.append(f"=== FILE: {name} ===\n{text[:8000]}")
    reasoning = sample.fields.get("solution_reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        parts.append(f"=== FILE: Reasoning.docx ===\n{reasoning[:8000]}")
    return "\n\n".join(parts), (sample.fields.get("instruction") or "")


_EXTRACTOR_PROMPT = """You are extracting facts for a cross-document consistency check. Do NOT judge whether anything is consistent or produce a verdict — your only job is enumeration.

## Task instruction (for context only — do not extract claims from this section itself)

{instruction}

## Files

{corpus}

## Task

Do TWO separate things:

1. List every entity, date, amount, status, or quantity that is mentioned in TWO OR MORE of the files above (the same real-world fact referenced more than once, possibly worded differently). For each one, quote every file's mention verbatim. Be exhaustive — a deck's summary table and a workbook's detail sheet reporting the same metric is exactly the kind of pair to list, even if the values look similar. Do not pre-judge agreement — list it and let the next step compare.

2. Separately, check whether any FILE's text substantially restates or closely paraphrases the task instruction's own content (its option list, its required analysis steps, its workplan) rather than reading as independent, in-universe business content. List any such file with a verbatim quote.

## Required output

Return exactly one JSON object, nothing else:
{{
  "documents_scanned": <int>,
  "claims": [
    {{"entity": "<short label>", "mentions": [{{"file": "<filename>", "quote": "<verbatim>", "value": "<the stated number/date/status>"}}]}}
  ],
  "prompt_duplication": [
    {{"file": "<filename>", "quote": "<verbatim>", "why": "<one phrase>"}}
  ]
}}
If you find nothing for either array, return it empty — do not skip either field."""

_VERDICT_PROMPT = """You are checking a pre-extracted claims table for cross-document disagreement. You do not have access to the original documents — work only from the table below.

## Claims table (from an independent extraction pass)

{claims_json}

## Task

For EVERY entity row, try to find a disagreement between its documents — do not conclude "consistent" for a row until you have explicitly compared every listed value/quote for that entity and stated why they agree. A same-fact disagreement across documents (e.g. one document says 94% on-time, another says 36% for the same metric) is a real defect even if each document individually reads as plausible.

"disagree" means a genuine same-fact, same-event conflict. Two values that differ only because they're as-of different dates, or because one is a sub-total feeding into the other, are NOT a disagreement — mark those "agree" and say why. Reserve "disagree" for two documents stating different values for what is unambiguously the SAME single fact at the SAME point in time.

## Required output

Return exactly one JSON object, nothing else:
{{"rows_checked": <int>, "results": [{{"entity": "<from the table>", "verdict": "agree"|"disagree", "reason": "<one sentence>"}}]}}"""


def _llm_json_call(client, model: str, prompt: str) -> dict | None:
    from .llm_client import LLMMessage, LLMClientConfig
    from .v18_judge import _extract_json_object
    # No explicit temperature: gpt-5.5 is a reasoning model and rejects
    # anything but the default (1.0) — matches run_one_judge's convention of
    # leaving it unset rather than forcing 0.0 for determinism.
    config = LLMClientConfig(model=model, max_tokens=8000, reasoning_effort="high")
    try:
        text = client.complete([LLMMessage(role="user", content=prompt)], config)
        return json.loads(_extract_json_object(text))
    except Exception:
        return None


def check_cross_artifact_reconciliation(
    sample: Sample, dim: V18Dimension, judge_client_factory,
) -> DimensionFinding:
    corpus, instruction = _gather_reconciliation_corpus(sample)
    if not corpus.strip():
        return _ok(dim, headline="no files to reconcile")

    client = judge_client_factory("gpt-5.5")

    call_a = _llm_json_call(
        client, "gpt-5.5",
        _EXTRACTOR_PROMPT.format(instruction=instruction[:4000], corpus=corpus[:60000]))
    if call_a is None:
        return _ok(dim, headline="extractor call failed", )

    dup = call_a.get("prompt_duplication") or []
    claims = call_a.get("claims") or []

    disagreements = []
    if claims:
        call_b = _llm_json_call(
            client, "gpt-5.5",
            _VERDICT_PROMPT.format(claims_json=json.dumps({"claims": claims})))
        if call_b:
            for r in (call_b.get("results") or []):
                if r.get("verdict") == "disagree":
                    disagreements.append(r)

    if not disagreements and not dup:
        return _ok(dim, headline=f"no cross-document disagreement found "
                                  f"({len(claims)} claim(s) checked)")

    parts = []
    if disagreements:
        d = disagreements[0]
        parts.append(f'cross-document disagreement on "{d.get("entity")}": {d.get("reason")}')
        if len(disagreements) > 1:
            parts.append(f"({len(disagreements) - 1} more disagreement(s) also flagged)")
    if dup:
        d = dup[0]
        parts.append(f'"{d.get("file")}" substantially restates the task instruction '
                      f'({d.get("why")}): "{d.get("quote")}"')

    return _fail(
        dim, severity=2,
        headline=(f"cross-document disagreement: {disagreements[0].get('entity')}" if disagreements
                  else f"instruction restated in {dup[0].get('file')}"),
        detail="; ".join(parts),
        fixability="fixable",
    )


# ── Registry exposed for V18Engine ─────────────────────────────────────────

BUILTIN_CALLBACKS = {
    "check_metadata_completeness":         check_metadata_completeness,
    "check_verifier_parseable":            check_verifier_parseable,
    "check_verifier_count_in_range":       check_verifier_count_in_range,
    "check_verifier_outcome_grader_present": check_verifier_outcome_grader_present,
    "check_synthetic_markers":             check_synthetic_markers,
    "check_near_miss":                     check_near_miss,
    "check_sum_mismatch":                  check_sum_mismatch,
    "check_ratio_invariant":               check_ratio_invariant,
    "check_doc_citation":                  check_doc_citation,
    "check_deliverable_manifest":          check_deliverable_manifest,
    "check_cross_artifact_reconciliation": check_cross_artifact_reconciliation,
}

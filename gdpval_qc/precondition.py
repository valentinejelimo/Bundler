"""Built-in preconditions for v18 dimension `applies_when:` clauses.

Each precondition is a `Callable[[Sample], bool]` registered under a short
name. The framework's dimension dispatcher consults this registry once per
sample × conditional-dim: when a precondition returns False, the dim emits a
DimensionFinding with `applicable=False` and no LLM call is made.

Custom preconditions can be passed alongside built-ins by callers that
construct a framework instance with their own callback registry; the
`get_precondition` helper looks up built-ins first and falls back to the
caller-supplied map.

Falsibility direction: every built-in returns False conservatively when the
input it inspects is missing or malformed. The exception is `solution_has_charts`
which returns True on inspection failure — better to run a possibly-redundant
chart-quality judge than to silently skip a real chart issue.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Callable, Iterable

from .v18_loader import FileContent, Sample

# Public type alias
Precondition = Callable[[Sample], bool]


# ── Built-in preconditions ─────────────────────────────────────────────────

def solution_present(sample: Sample) -> bool:
    """True iff `solution_files` is bound and contains at least one file."""
    files = sample.fields.get("solution_files")
    return isinstance(files, list) and len(files) > 0


def task_has_verifier(sample: Sample) -> bool:
    """True iff `verifier_spec_json` is bound and non-empty."""
    spec = sample.fields.get("verifier_spec_json")
    return isinstance(spec, str) and spec.strip() != ""


def task_has_rubric_verifier(sample: Sample) -> bool:
    """True iff at least one verifier in the spec has assertion.type='rubric'."""
    return _any_assertion_type(sample, "rubric")


def task_has_deterministic_verifier(sample: Sample) -> bool:
    """True iff at least one verifier in the spec has assertion.type='deterministic'."""
    return _any_assertion_type(sample, "deterministic")


def task_has_eval_runs(sample: Sample) -> bool:
    """True iff the task carries eval-run data — i.e. the run digest field
    `verifier_assertion_summary` was populated (only happens when
    `evals/<model>/trial_*/` reports exist). Gates the run-based dims."""
    summary = sample.fields.get("verifier_assertion_summary")
    return isinstance(summary, str) and summary.strip() != ""


def task_has_full_trajectories(sample: Sample) -> bool:
    """True iff >=2 models under evals/ each have a parseable agent trajectory.

    Gates `intermodel_trajectory_consistency` (E): that dim compares HOW models
    worked, so it needs at least two models' `agent/trajectory.json` present.
    Cheapest reliable signal is the digest field itself — the callback returns
    None (→ empty field) unless >=2 models have a trajectory. Conservative:
    False on anything falsy."""
    digest = sample.fields.get("intermodel_trajectory_digest")
    return isinstance(digest, str) and digest.strip() != ""


def task_is_substantive(sample: Sample) -> bool:
    """False only for atomic L1 decomposition leaves; True otherwise.

    SVC tasks carry ``[metadata].level = L1|L2|L3`` in task.toml. L1 leaves are
    intentionally atomic single-fact extraction nodes fed a prior-run golden, so
    the "standalone substantive professional engagement" dimensions
    (overall_gdpval_fit, professional_complexity, input_prompt_leakage)
    false-positive on them. Gating those dims with this precondition skips L1 and
    keeps them at L2/L3. Non-SVC tasks (no level metadata) return True, so
    behaviour on EKW corpora is unchanged.
    """
    return _task_level(sample) != "L1"


def _task_level(sample: Sample) -> str | None:
    """Read ``[metadata].level`` from the sample's task.toml (None if absent)."""
    try:
        import tomllib
        with open(sample.task_dir / "task.toml", "rb") as fh:
            data = tomllib.load(fh)
        lvl = (data.get("metadata") or {}).get("level")
        return str(lvl).upper().strip() if lvl else None
    except Exception:
        return None


def solution_has_charts(sample: Sample) -> bool:
    """True iff any solution file appears to contain chart-like content.

    Conservative default: returns True when chart detection is inconclusive
    (so the chart-quality dim runs and the LLM can return applicable=False
    if needed). Returns False only when the solution exists AND every file
    is confidently chart-free.
    """
    files = sample.fields.get("solution_files")
    if not isinstance(files, list) or not files:
        return False
    for fc in files:
        if not isinstance(fc, FileContent):
            continue
        if _file_has_charts(fc.path):
            return True
    return False


BUILTINS: dict[str, Precondition] = {
    "solution_present":                 solution_present,
    "task_has_verifier":                task_has_verifier,
    "task_has_rubric_verifier":         task_has_rubric_verifier,
    "task_has_deterministic_verifier":  task_has_deterministic_verifier,
    "task_has_eval_runs":               task_has_eval_runs,
    "task_has_full_trajectories":       task_has_full_trajectories,
    "task_is_substantive":              task_is_substantive,
    "solution_has_charts":              solution_has_charts,
}


# ── Lookup helper ──────────────────────────────────────────────────────────

def get_precondition(
    name: str,
    extra: dict[str, Precondition] | None = None,
) -> Precondition:
    """Return a precondition by name from built-ins or the supplied extras.

    Raises KeyError with a useful list of known names if the precondition is
    unknown.
    """
    if name in BUILTINS:
        return BUILTINS[name]
    if extra and name in extra:
        return extra[name]
    known = sorted(BUILTINS.keys())
    if extra:
        known = sorted(known + list(extra.keys()))
    raise KeyError(
        f"Unknown precondition: {name!r}. Known: {known}. "
        f"Either use a built-in or pass a custom precondition via the "
        f"framework's precondition registry."
    )


# ── Internals ──────────────────────────────────────────────────────────────

def _any_assertion_type(sample: Sample, kind: str) -> bool:
    spec = sample.fields.get("verifier_spec_json")
    if not isinstance(spec, str) or not spec.strip():
        return False
    try:
        d = json.loads(spec)
    except Exception:
        return False
    for v in d.get("verifiers", []) or []:
        a = (v or {}).get("assertion") or {}
        if a.get("type") == kind:
            return True
    return False


def _file_has_charts(path: Path) -> bool:
    """Detect chart-like content for one file.

    Returns True conservatively when detection can't run (e.g., openpyxl/pypdf
    missing, file unreadable). False only when the file is confidently
    chart-free.
    """
    if not path.is_file():
        return False
    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        return _xlsx_has_charts(path)
    if suffix == ".docx":
        return _docx_has_charts(path)
    if suffix == ".pdf":
        return _pdf_has_charts(path)
    if suffix in (".png", ".jpg", ".jpeg", ".gif",
                  ".tif", ".tiff", ".bmp", ".webp", ".svg"):
        # The file itself is a graphic and is part of the solution — count
        # it as a chart for the purpose of this precondition.
        return True

    return False


def _xlsx_has_charts(path: Path) -> bool:
    """True if any sheet contains chart objects.

    read_only=False is required: chart objects are not loaded in read-only mode.
    """
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
        has = any(getattr(wb[s], "_charts", None) for s in wb.sheetnames)
        wb.close()
        return has
    except Exception:
        return True   # conservative fallback


def _docx_has_charts(path: Path) -> bool:
    """True if the docx contains inline shapes (images, charts) or embedded chart parts.

    python-docx exposes inline_shapes (images, including chart-as-image). For
    true embedded chart objects (Word's "Insert > Chart") the chart part lives
    under word/charts/chart*.xml; we detect those by inspecting the package.
    """
    try:
        # First-pass: inline shapes via python-docx
        from docx import Document
        doc = Document(str(path))
        if len(doc.inline_shapes) > 0:
            return True
        # Second-pass: peek into the OOXML zip for chart parts
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        for n in names:
            if n.startswith("word/charts/") and n.endswith(".xml"):
                return True
            if n.startswith("word/media/") and any(
                n.lower().endswith(ext)
                for ext in (".png", ".jpg", ".jpeg", ".gif", ".emf", ".wmf")
            ):
                return True
        return False
    except Exception:
        return True  # conservative fallback


def _pdf_has_charts(path: Path) -> bool:
    """Heuristic: True if any PDF page contains image XObjects.

    pypdf exposes page.images for each page. Empty across all pages → False.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for page in reader.pages:
            try:
                if list(page.images):
                    return True
            except Exception:
                # If image enumeration fails for one page, conservatively assume
                # it might have charts and return True
                return True
        return False
    except Exception:
        return True  # conservative fallback

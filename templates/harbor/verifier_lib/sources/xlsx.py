import re
import warnings
from pathlib import Path

from pydantic import field_validator

from ._content_limits import finalize_accumulated
from ..models import StrictModel
from ..source_types import SourceAuthoringError, SourceCommand, SourceContext


class XlsxExtractTextInput(StrictModel):
    """Input for xlsx.extract_text.

    Attributes:
        path: Workspace-relative XLSX or XLSM path.
    """

    path: str

    @field_validator("path")
    @classmethod
    def require_workbook_extension(cls, value: str) -> str:
        """Requires the path to target an XLSX/XLSM workbook.

        Args:
            value: Workspace-relative path from verifier.json.

        Returns:
            The unchanged path when it has a supported workbook extension.

        Raises:
            ValueError: If the path does not end in ``.xlsx`` or ``.xlsm``.
        """
        if Path(value).suffix.lower() not in {".xlsx", ".xlsm"}:
            raise ValueError("xlsx.extract_text requires a .xlsx or .xlsm file")
        return value


class XlsxExtractTextOutput(StrictModel):
    """Output from xlsx.extract_text.

    Attributes:
        text: Extracted sheet names and non-empty cell rows.
    """

    text: str


class ExtractText(SourceCommand):
    """Extracts readable text from workbook sheets."""

    name = "extract_text"
    input_model = XlsxExtractTextInput
    output_model = XlsxExtractTextOutput

    def run(self, source_input: XlsxExtractTextInput, context: SourceContext) -> XlsxExtractTextOutput:
        """Runs XLSX text extraction.

        Args:
            source_input: Validated workbook extraction input.
            context: Source runtime context.

        Returns:
            Extracted workbook text.

        """
        resolved = context.resolve_path(source_input.path)
        return XlsxExtractTextOutput(text=extract_xlsx(resolved, context.max_content_chars))


# ---------------------------------------------------------------------------
# Uncached-formula recalculation. ``extract_xlsx`` reads cached values (data_only=True). A workbook
# written by openpyxl/xlsxwriter carries formulas with NO cached value, so every formula cell reads
# back blank and a judge sees an empty model -- confirmed on real runs (2026-09-16 expert review:
# two task_31 runs lost 0.61 and 0.72 of weight to blank extraction of correct formulas; task_24
# runs had 1,281/1,656 formula cells with no cache). Whether a solver's file caches values is a
# property of the library it used, not of its work, and it is model-dependent (openpyxl writers
# lose; Excel/LibreOffice savers win). Fix: if any formula cell lacks a cached value, recalculate a
# COPY headlessly with LibreOffice (present in every native task image) and read from the copy.
# Strictly best-effort: any failure (no soffice, timeout, conversion error) falls back to reading
# the original file exactly as before. The original file is never modified.
# ---------------------------------------------------------------------------

_RECALC_TIMEOUT_SECONDS = 180


def _has_uncached_formula(path: Path) -> bool:
    from openpyxl import load_workbook

    wb_f = load_workbook(path, data_only=False, read_only=True)
    wb_v = load_workbook(path, data_only=True, read_only=True)
    try:
        for ws_f in wb_f.worksheets:
            ws_v = wb_v[ws_f.title]
            for row_f, row_v in zip(ws_f.iter_rows(), ws_v.iter_rows()):
                for cell_f, cell_v in zip(row_f, row_v):
                    if cell_f.data_type == "f" and cell_v.value is None:
                        return True
        return False
    finally:
        wb_f.close()
        wb_v.close()


_RECALC_CACHE: dict[tuple, Path] = {}


def _recalculated_copy_if_uncached(path: Path) -> Path:
    """Returns a recalculated temp copy of ``path`` when it has uncached formulas, else ``path``.

    Memoised per (path, size, mtime) so a workbook read by many criteria is converted once per run.
    """
    try:
        st = path.stat()
        key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    except OSError:
        key = None
    if key is not None and key in _RECALC_CACHE and _RECALC_CACHE[key].is_file():
        return _RECALC_CACHE[key]
    result = _recalc_uncached(path)
    if key is not None:
        _RECALC_CACHE[key] = result
    return result


def _recalc_uncached(path: Path) -> Path:
    import shutil
    import subprocess
    import tempfile

    try:
        if not _has_uncached_formula(path):
            return path
    except Exception:
        return path
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        return path
    try:
        out_dir = Path(tempfile.mkdtemp(prefix="xlsx_recalc_"))
        profile = Path(tempfile.mkdtemp(prefix="lo_profile_"))
        src = out_dir / ("src_" + path.name)
        shutil.copy2(path, src)
        subprocess.run(
            [
                soffice,
                f"-env:UserInstallation=file://{profile}",
                "--headless",
                "--norestore",
                "--convert-to",
                "xlsx",
                "--outdir",
                str(out_dir / "out"),
                str(src),
            ],
            check=True,
            capture_output=True,
            timeout=_RECALC_TIMEOUT_SECONDS,
        )
        converted = out_dir / "out" / src.name
        if converted.is_file() and not _has_uncached_formula(converted):
            return converted
        return path
    except Exception:
        return path


# ---------------------------------------------------------------------------
# formula recompute (fixes a grading-time blind spot: this file previously read every cell
# with data_only=True, which returns whatever value was cached in the workbook the LAST time
# a real spreadsheet engine saved it -- never the live result of the formula. A file saved by
# openpyxl (or produced by an xlsm->xlsx conversion step) commonly has every formula cell's
# cache blank or stale even though the formula itself is fine, which meant a judge reading this
# extractor's output saw blank/wrong numbers for perfectly good work. An external QC review
# found this exact defect graded 28 submissions across 18 tasks with every calculation
# invisible, with the harm concentrated on whichever submissions happened to lean more heavily
# on formulas rather than hardcoded values -- not a neutral, model-independent blind spot.
#
# Fix: recompute every formula cell with the same `formulas` engine
# deterministic-checks/formula_health_check.py already uses (same library, independently
# vendored here because this module ships standalone inside each task's grading sandbox and
# cannot import from the repo-root deterministic-checks/ tooling). The recompute is strictly
# best-effort and additive: any failure (missing dependency, a formula shape the engine can't
# resolve, a corrupt file) falls back to this file's ORIGINAL behavior -- reading whatever is
# cached on disk -- for every affected cell, never a hard failure of the whole extraction. A
# recomputed value is only ever used in place of a cache read; nothing here can produce a value
# that didn't already come out of the workbook's own formula.
#
# Trade-off, disclosed rather than silent: this recompute re-parses and re-evaluates the WHOLE
# workbook's formula graph on every extract_xlsx call, which costs real time and CPU beyond the
# original streaming, cache-only read -- proportional to the size of the workbook, not the size
# of the extracted text. For a benchmark's typical task-sized spreadsheet this is the same cost
# deterministic-checks/formula_health_check.py already pays per file today; it has not been
# load-tested against many-large-file batches inside the grading sandbox specifically.
# ---------------------------------------------------------------------------

_SOLUTION_KEY_RE = re.compile(r"^'\[[^\]]+\](?P<sheet>[^']+)'!(?P<cell>[A-Z]+\d+)$")
_ERR_TOKENS = ("#REF!", "#VALUE!", "#NAME?", "#DIV/0!", "#N/A", "#NULL!", "#NUM!")


def _unwrap_formulas_value(v):
    """A `formulas` calculate() value is usually a Ranges/Array wrapper; pull the scalar out.
    Verbatim logic from deterministic-checks/formula_health_check.py's own helper of the same
    name, duplicated here (not imported) since this module ships standalone per task."""
    try:
        v = v.value
    except AttributeError:
        pass
    try:
        arr = v
        seen = 0
        while hasattr(arr, "shape") and getattr(arr, "shape", None) not in (None, ()):
            arr = arr[0] if len(arr) else None
            seen += 1
            if seen > 10:
                break
        return arr
    except Exception:
        return v


def _is_usable_recompute(value: object) -> bool:
    """Whether a recomputed value is safe to substitute for a cached one: not missing, not
    NaN, and not an Excel error-shaped string. A cell the engine could not cleanly resolve is
    left on its original cached value rather than showing a fabricated or error-shaped result.
    """
    if value is None:
        return False
    if isinstance(value, float):
        try:
            import math

            if math.isnan(value):
                return False
        except Exception:
            pass
    if isinstance(value, str) and any(tok in value for tok in _ERR_TOKENS):
        return False
    return True


def _recompute_formula_map(path: Path) -> dict:
    """Best-effort formula recompute for one workbook. Returns
    {(SHEET_NAME_UPPER, cell_coordinate): recomputed_value} for every formula cell the
    `formulas` engine could resolve to a usable value. Returns {} on ANY failure -- missing
    dependency, unsupported formula shape, corrupt file -- so a recompute problem never blocks
    grading; the caller falls back to the pre-existing cached-value read for every cell in
    that case, exactly this file's behavior before this fix.
    """
    try:
        import os

        # Explicit opt-in (set by the single-underscore harbor genre's verifier_runtime.py, whose
        # requirements.txt pins `formulas`). Never keyed on whether the package merely happens to
        # be importable: that would make the extracted text -- and the reward -- depend on an
        # incidental property of the environment. Never installed at grade time.
        if os.environ.get("VERIFIER_XLSX_FORMULAS_RECOMPUTE") != "1":
            return {}
        import formulas

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xl_model = formulas.ExcelModel().loads(str(path)).finish()
            solution = xl_model.calculate()
    except Exception:
        return {}

    recomputed: dict = {}
    for key, cell_range in solution.items():
        m = _SOLUTION_KEY_RE.match(key)
        if not m:
            continue
        try:
            value = _unwrap_formulas_value(cell_range)
        except Exception:
            continue
        if not _is_usable_recompute(value):
            continue
        recomputed[(m.group("sheet").upper(), m.group("cell"))] = value
    return recomputed


def extract_xlsx(path: Path, limit: int) -> str:
    """Extracts sheet names and non-empty cell rows from a workbook.

    Every formula cell is recomputed with a real spreadsheet engine first (see the module-level
    comment above) and that live value is used in place of whatever was cached on disk, so a
    stale or blank formula cache can no longer read as blank/wrong to whichever judge grades
    this text. Non-formula cells are read exactly as before.

    Args:
        path: Resolved workbook path.
        limit: Maximum number of characters to return.

    Returns:
        Extracted workbook text capped to ``limit`` characters. If the workbook is longer than
        the cap, a truncation notice is appended so the cut is visible to the grader instead of
        silent.
    """
    from openpyxl import load_workbook

    path = _recalculated_copy_if_uncached(path)
    recomputed = _recompute_formula_map(path)

    wb_cached = load_workbook(path, data_only=True, read_only=True)
    wb_formula = None
    if recomputed:
        try:
            wb_formula = load_workbook(path, data_only=False, read_only=True)
        except Exception:
            wb_formula = None

    parts: list[str] = []
    total = 0
    truncated = False
    try:
        for sheet_name in wb_cached.sheetnames:
            total = append_limited(parts, f"[Sheet: {sheet_name}]", total, limit)
            if total >= limit:
                truncated = True
                break

            ws_cached = wb_cached[sheet_name]
            ws_formula = (
                wb_formula[sheet_name]
                if wb_formula is not None and sheet_name in wb_formula.sheetnames
                else None
            )
            row_pairs = (
                zip(ws_cached.iter_rows(), ws_formula.iter_rows())
                if ws_formula is not None
                else ((row, None) for row in ws_cached.iter_rows())
            )

            for cached_row, formula_row in row_pairs:
                cells = []
                for i, cached_cell in enumerate(cached_row):
                    value = cached_cell.value
                    if formula_row is not None and i < len(formula_row):
                        fcell = formula_row[i]
                        if isinstance(fcell.value, str) and fcell.value.startswith("="):
                            key = (sheet_name.upper(), fcell.coordinate)
                            if key in recomputed:
                                value = recomputed[key]
                    cells.append(str(value) if value is not None else "")
                if any(cell.strip() for cell in cells):
                    total = append_limited(parts, " | ".join(cells), total, limit)
                    if total >= limit:
                        truncated = True
                        break
            if truncated:
                break
        return finalize_accumulated(parts, limit, truncated)
    finally:
        wb_cached.close()
        if wb_formula is not None:
            wb_formula.close()


def append_limited(parts: list[str], text: str, total: int, limit: int) -> int:
    """Appends text and returns the updated character count.

    Args:
        parts: Text fragments collected so far.
        text: Text fragment to append.
        total: Current approximate character count.
        limit: Maximum desired character count.

    Returns:
        Updated approximate character count.
    """
    parts.append(text)
    return total + len(text) + 1


class XlsxFormulaAuthenticityInput(StrictModel):
    """Input for xlsx.formula_authenticity_check.

    Attributes:
        path: Workspace-relative XLSX or XLSM path.
        sheet: Sheet name containing the cells to check.
        cell_range: A single cell (e.g. ``"B4"``) or range (e.g. ``"B4:B10"``,
            ``"B4:E10"``) that should be formula-driven rather than typed-in
            literal values.
    """

    path: str
    sheet: str
    cell_range: str

    @field_validator("path")
    @classmethod
    def require_workbook_extension(cls, value: str) -> str:
        """Requires the path to target an XLSX/XLSM workbook.

        Args:
            value: Workspace-relative path from verifier.json.

        Returns:
            The unchanged path when it has a supported workbook extension.

        Raises:
            ValueError: If the path does not end in ``.xlsx`` or ``.xlsm``.
        """
        if Path(value).suffix.lower() not in {".xlsx", ".xlsm"}:
            raise ValueError("xlsx.formula_authenticity_check requires a .xlsx or .xlsm file")
        return value


class XlsxFormulaAuthenticityCell(StrictModel):
    """Per-cell finding from xlsx.formula_authenticity_check.

    Attributes:
        address: Cell address (e.g. ``"B4"``).
        is_formula: True if the cell's stored content is a live formula.
        formula_text: The formula string (e.g. ``"=A1+A2"``) when
            ``is_formula`` is true, else ``None``.
        cached_value: The cell's cached/computed value, as a string (or
            ``None`` if the cell is empty). For a formula cell this is the
            last-saved computed result; for a non-formula cell it is
            whatever literal was typed in.
        is_empty: True if the cell has no stored content at all.
    """

    address: str
    is_formula: bool
    formula_text: str | None
    cached_value: str | None
    is_empty: bool


class XlsxFormulaAuthenticityOutput(StrictModel):
    """Output from xlsx.formula_authenticity_check.

    Attributes:
        sheet: Sheet name that was checked.
        cell_range: Range that was checked, as given in the input.
        n_checked: Total number of cells in the range.
        n_formula: Number of cells that are live formulas.
        n_hardcoded: Number of non-empty cells that are NOT formulas (a typed
            literal sitting where the criterion expects a computed value).
        n_empty: Number of cells with no stored content.
        hardcoded_cells: Addresses (with their literal value) of every
            non-empty, non-formula cell in the range — the cells a
            "must be computed, not typed in" criterion should fail on.
        cells: Full per-cell detail for every cell in the range, in reading
            order, for a judge or a human to inspect directly.
    """

    sheet: str
    cell_range: str
    n_checked: int
    n_formula: int
    n_hardcoded: int
    n_empty: int
    hardcoded_cells: list[XlsxFormulaAuthenticityCell]
    cells: list[XlsxFormulaAuthenticityCell]


class FormulaAuthenticityCheck(SourceCommand):
    """Checks whether cells in a range are live formulas or typed-in literals.

    ``xlsx.extract_text`` (and any judge reading it) only ever sees a cell's
    *cached* value — a formula that computes 42 and a literal ``42`` typed
    directly into the cell are indistinguishable from that extraction alone.
    This command instead inspects each cell's stored content directly (via
    openpyxl's non-``data_only`` load, where a formula cell's ``.value`` is
    the formula string itself, e.g. ``"=A1+A2"``, rather than its result) to
    tell live computation apart from a hardcoded answer sitting in a cell
    that is supposed to be derived.
    """

    name = "formula_authenticity_check"
    input_model = XlsxFormulaAuthenticityInput
    output_model = XlsxFormulaAuthenticityOutput

    def run(
        self, source_input: XlsxFormulaAuthenticityInput, context: SourceContext
    ) -> XlsxFormulaAuthenticityOutput:
        """Runs the formula-authenticity check.

        Args:
            source_input: Validated workbook/sheet/range input.
            context: Source runtime context.

        Returns:
            Per-cell formula-vs-literal findings for the requested range.

        Raises:
            SourceAuthoringError: If the sheet or cell range is invalid.
        """
        resolved = context.resolve_path(source_input.path)
        return check_formula_authenticity(resolved, source_input.sheet, source_input.cell_range)


def _formula_text(value: object) -> str | None:
    """Formula source text for a cell value. openpyxl returns a plain str for ordinary formulas
    but an ArrayFormula / DataTableFormula OBJECT for array cells; feeding that object to a
    ``str | None`` field raised ValidationError and zeroed the whole task."""
    if isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    # A DataTableFormula carries no text (only r1/r2 refs). Return None rather than str(value):
    # the object's repr ("... at 0x10a3b") would otherwise read as a cell reference.
    return text if isinstance(text, str) else None


def check_formula_authenticity(path: Path, sheet: str, cell_range: str) -> XlsxFormulaAuthenticityOutput:
    """Compares formula presence against cached values for a cell range.

    Args:
        path: Resolved workbook path.
        sheet: Sheet name containing the range to check.
        cell_range: A single cell or range address.

    Returns:
        Formula-authenticity findings for every cell in the range.

    Raises:
        SourceAuthoringError: If the sheet name or range is invalid.
    """
    from openpyxl import load_workbook

    # Formula-visible load: a formula cell's .value is the formula string
    # itself (e.g. "=A1+A2"); a literal cell's .value is the typed value.
    wb_formulas = load_workbook(path, data_only=False, read_only=False)
    # Cached-value load: every cell's .value is its last-computed/typed result.
    wb_values = load_workbook(path, data_only=True, read_only=False)
    try:
        if sheet not in wb_formulas.sheetnames:
            raise SourceAuthoringError(
                f"formula_authenticity_check: sheet {sheet!r} not found (have: {wb_formulas.sheetnames})"
            )
        ws_formulas = wb_formulas[sheet]
        ws_values = wb_values[sheet]

        try:
            formula_cells = ws_formulas[cell_range]
        except (ValueError, KeyError) as exc:
            raise SourceAuthoringError(f"formula_authenticity_check: invalid cell_range {cell_range!r}") from exc

        # openpyxl returns a single Cell for one address, or a tuple of rows
        # of Cells for a range — normalize to a flat list either way.
        if not isinstance(formula_cells, tuple):
            flat_formula_cells = [formula_cells]
        else:
            flat_formula_cells = [cell for row in formula_cells for cell in row]

        cells: list[XlsxFormulaAuthenticityCell] = []
        for fcell in flat_formula_cells:
            is_formula = fcell.data_type == "f"
            vcell = ws_values[fcell.coordinate]
            is_empty = fcell.value is None
            cells.append(
                XlsxFormulaAuthenticityCell(
                    address=fcell.coordinate,
                    is_formula=is_formula,
                    formula_text=_formula_text(fcell.value) if is_formula else None,
                    cached_value=None if vcell.value is None else str(vcell.value),
                    is_empty=is_empty,
                )
            )

        hardcoded_cells = [c for c in cells if not c.is_formula and not c.is_empty]
        return XlsxFormulaAuthenticityOutput(
            sheet=sheet,
            cell_range=cell_range,
            n_checked=len(cells),
            n_formula=sum(1 for c in cells if c.is_formula),
            n_hardcoded=len(hardcoded_cells),
            n_empty=sum(1 for c in cells if c.is_empty),
            hardcoded_cells=hardcoded_cells,
            cells=cells,
        )
    finally:
        wb_formulas.close()
        wb_values.close()


class XlsxFormulaSummaryInput(StrictModel):
    """Input for a layout-independent workbook formula summary."""

    path: str

    @field_validator("path")
    @classmethod
    def require_workbook_extension(cls, value: str) -> str:
        if Path(value).suffix.lower() not in {".xlsx", ".xlsm"}:
            raise ValueError("xlsx.formula_summary requires a .xlsx or .xlsm file")
        return value


class XlsxFormulaSummaryOutput(StrictModel):
    """Workbook-wide formula counts that do not pin sheet names or cell ranges."""

    n_formula: int
    n_formula_with_refs: int
    n_cross_sheet_formula: int
    n_formula_sheets: int
    formula_sheets: list[str]


class FormulaSummary(SourceCommand):
    """Counts live formulas and cross-sheet links across the whole workbook."""

    name = "formula_summary"
    input_model = XlsxFormulaSummaryInput
    output_model = XlsxFormulaSummaryOutput

    def run(
        self, source_input: XlsxFormulaSummaryInput, context: SourceContext
    ) -> XlsxFormulaSummaryOutput:
        resolved = context.resolve_path(source_input.path)
        return summarize_formulas(resolved)


_CELL_REF_RE = re.compile(r"(\$?[A-Za-z]{1,3}\$?\d{1,7})|!|\[[^\]]+\]")


def summarize_formulas(path: Path) -> XlsxFormulaSummaryOutput:
    """Returns layout-independent formula and cross-sheet-link counts."""
    from openpyxl import load_workbook

    workbook = load_workbook(path, data_only=False, read_only=False)
    n_formula = 0
    n_formula_with_refs = 0
    n_cross_sheet_formula = 0
    formula_sheets: list[str] = []
    try:
        for worksheet in workbook.worksheets:
            sheet_formula_count = 0
            for row in worksheet.iter_rows():
                for cell in row:
                    if cell.data_type != "f":
                        continue
                    n_formula += 1
                    sheet_formula_count += 1
                    value = cell.value
                    text = _formula_text(value) or ""
                    # a formula like =1+1 is not evidence of a live model; require a cell/range/name reference
                    if _CELL_REF_RE.search(text):
                        n_formula_with_refs += 1
                    if isinstance(value, str) and "!" in value:
                        n_cross_sheet_formula += 1
            if sheet_formula_count:
                formula_sheets.append(worksheet.title)
        return XlsxFormulaSummaryOutput(
            n_formula=n_formula,
            n_formula_with_refs=n_formula_with_refs,
            n_cross_sheet_formula=n_cross_sheet_formula,
            n_formula_sheets=len(formula_sheets),
            formula_sheets=formula_sheets,
        )
    finally:
        workbook.close()


class XlsxFormulaAuthenticityScanInput(StrictModel):
    """Input for a layout-independent formula-authenticity scan.

    Attributes:
        path: Workspace-relative XLSX or XLSM path.
        sheet_name_contains: Case-insensitive substrings; a sheet is scanned
            when its name contains any of them. An empty list -- or a list
            that matches no sheet in the submitted workbook -- falls back to
            scanning every sheet, so a solver who names their tabs
            differently from the reference workbook is never penalised and
            the command never raises.
    """

    path: str
    sheet_name_contains: list[str] = []

    @field_validator("path")
    @classmethod
    def require_workbook_extension(cls, value: str) -> str:
        """Requires the path to target an XLSX/XLSM workbook.

        Args:
            value: Workspace-relative path from verifier.json.

        Returns:
            The unchanged path when it has a supported workbook extension.

        Raises:
            ValueError: If the path does not end in ``.xlsx`` or ``.xlsm``.
        """
        if Path(value).suffix.lower() not in {".xlsx", ".xlsm"}:
            raise ValueError("xlsx.formula_authenticity_scan requires a .xlsx or .xlsm file")
        return value


class XlsxFormulaAuthenticityScanOutput(StrictModel):
    """Output from xlsx.formula_authenticity_scan.

    Attributes:
        sheets_scanned: Names of the sheets that were scanned.
        fallback_all_sheets: True when no sheet name matched the requested
            substrings and every sheet was scanned instead.
        n_checked: Number of value-bearing cells scanned (live formula cells
            plus numeric literal cells). Text labels are ignored.
        n_formula: Number of those cells that are live formulas referencing at
            least one cell/range/sheet (a formula with no reference, e.g. =1+1,
            is ignored entirely so padding cannot inflate the share).
        n_hardcoded: Number of those cells that are typed-in numeric
            literals rather than formulas.
        formula_share: ``n_formula / n_checked`` (0.0 when nothing scanned).
        formula_share_all: the same share computed over EVERY sheet, regardless
            of the selector.
        formula_share_min: ``min(formula_share, formula_share_all)``. Do NOT
            assert on it: when the selector matches only an inputs tab it
            false-fails a compliant workbook (confirmed on task_31's golden once
            its tabs were renamed). Assert on ``formula_share_all`` (selector-
            independent). Known residual of any whole-workbook share: a
            constants-only graded tab next to a live non-graded tab; pair the
            share with ``n_formula_with_refs`` and the LLM criteria on the
            graded figures rather than a selector-scoped floor.
        hardcoded_cells: Up to 50 examples of typed-in numeric cells, for a
            human or a judge to inspect directly.
    """

    sheets_scanned: list[str]
    fallback_all_sheets: bool
    n_checked: int
    n_formula: int
    n_hardcoded: int
    formula_share: float
    formula_share_all: float
    formula_share_min: float
    hardcoded_cells: list[XlsxFormulaAuthenticityCell]


class FormulaAuthenticityScan(SourceCommand):
    """Measures how much of a workbook's numbers are computed vs. typed in.

    ``xlsx.formula_authenticity_check`` answers the same question but has to
    be pinned to one exact sheet name and cell range, and raises
    ``SourceAuthoringError`` (which aborts the whole verifier run and scores
    the submission 0.0) when a solver's workbook happens to use a different
    sheet name -- something no instruction in this corpus mandates. This
    command asks the same formula-vs-literal question without pinning any
    layout: it selects sheets by case-insensitive substring, falls back to
    the whole workbook when nothing matches, and reports the share of
    value-bearing cells that are live formulas. A workbook whose numbers
    were computed elsewhere and pasted in as constants scores 0.0 here while
    extracting to byte-identical text under ``extract_text``.
    """

    name = "formula_authenticity_scan"
    input_model = XlsxFormulaAuthenticityScanInput
    output_model = XlsxFormulaAuthenticityScanOutput

    def run(
        self, source_input: XlsxFormulaAuthenticityScanInput, context: SourceContext
    ) -> XlsxFormulaAuthenticityScanOutput:
        """Runs the layout-independent formula-authenticity scan.

        Args:
            source_input: Validated workbook/sheet-selector input.
            context: Source runtime context.

        Returns:
            Formula-vs-literal counts over the selected sheets.
        """
        resolved = context.resolve_path(source_input.path)
        return scan_formula_authenticity(resolved, source_input.sheet_name_contains)


_NUMERIC_TEXT_RE = re.compile(r"^[\s$€£(]*[-+]?\d[\d,]*(\.\d+)?\s*[%x)]*\s*$")


def _is_typed_number(value: object) -> bool:
    """A typed-in numeric literal, including a number stored as text ("1,234.5", "$410", "6.65x", "9.1%")."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, str) and bool(_NUMERIC_TEXT_RE.match(value)) and any(ch.isdigit() for ch in value)


def scan_formula_authenticity(path: Path, sheet_name_contains: list[str]) -> XlsxFormulaAuthenticityScanOutput:
    """Counts live-formula vs. typed-in numeric cells across selected sheets.

    Args:
        path: Resolved workbook path.
        sheet_name_contains: Case-insensitive sheet-name substrings; empty or
            non-matching selects every sheet.

    Returns:
        Formula-authenticity counts and example hardcoded cells.
    """
    from openpyxl import load_workbook

    wanted = [s.lower() for s in sheet_name_contains if s.strip()]
    workbook = load_workbook(path, data_only=False, read_only=False)
    try:
        selected = [ws for ws in workbook.worksheets if any(w in ws.title.lower() for w in wanted)]
        fallback = not selected
        if fallback:
            selected = list(workbook.worksheets)

        n_formula = 0
        hardcoded: list[XlsxFormulaAuthenticityCell] = []
        n_hardcoded = 0
        all_formula = 0
        all_hardcoded = 0
        selected_titles = {ws.title for ws in selected}
        for worksheet in workbook.worksheets:
            in_selection = worksheet.title in selected_titles
            for row in worksheet.iter_rows():
                for cell in row:
                    if cell.data_type == "f":
                        ftext = _formula_text(cell.value) or ""
                        if not _CELL_REF_RE.search(ftext):
                            continue  # =1+1 padding is neither a live link nor a typed constant
                        all_formula += 1
                        if in_selection:
                            n_formula += 1
                    elif _is_typed_number(cell.value):
                        all_hardcoded += 1
                        if not in_selection:
                            continue
                        n_hardcoded += 1
                        if len(hardcoded) < 50:
                            hardcoded.append(
                                XlsxFormulaAuthenticityCell(
                                    address=f"{worksheet.title}!{cell.coordinate}",
                                    is_formula=False,
                                    formula_text=None,
                                    cached_value=str(cell.value),
                                    is_empty=False,
                                )
                            )
        n_checked = n_formula + n_hardcoded
        share = (n_formula / n_checked) if n_checked else 0.0
        all_checked = all_formula + all_hardcoded
        share_all = (all_formula / all_checked) if all_checked else 0.0
        return XlsxFormulaAuthenticityScanOutput(
            sheets_scanned=[ws.title for ws in selected],
            fallback_all_sheets=fallback,
            n_checked=n_checked,
            n_formula=n_formula,
            n_hardcoded=n_hardcoded,
            formula_share=share,
            formula_share_all=share_all,
            formula_share_min=min(share, share_all),
            hardcoded_cells=hardcoded,
        )
    finally:
        workbook.close()


class XlsxHasChartInput(StrictModel):
    """Input for xlsx.has_chart.

    Attributes:
        path: Workspace-relative XLSX or XLSM path.
        min_series: Minimum number of chart data series required across all
            charts in the workbook.
    """

    path: str
    min_series: int = 1

    @field_validator("path")
    @classmethod
    def require_workbook_extension(cls, value: str) -> str:
        """Requires the path to target an XLSX/XLSM workbook.

        Args:
            value: Workspace-relative path from verifier.json.

        Returns:
            The unchanged path when it has a supported workbook extension.

        Raises:
            ValueError: If the path does not end in ``.xlsx`` or ``.xlsm``.
        """
        if Path(value).suffix.lower() not in {".xlsx", ".xlsm"}:
            raise ValueError("xlsx.has_chart requires a .xlsx or .xlsm file")
        return value


class XlsxHasChartOutput(StrictModel):
    """Output from xlsx.has_chart.

    Attributes:
        chart_count: Number of chart parts found in the workbook package.
        max_series_count: Largest number of data series found on any single
            chart part.
    """

    chart_count: int
    max_series_count: int


class HasChart(SourceCommand):
    """Deterministically checks whether a workbook contains a chart object."""

    name = "has_chart"
    input_model = XlsxHasChartInput
    output_model = XlsxHasChartOutput

    def run(self, source_input: XlsxHasChartInput, context: SourceContext) -> XlsxHasChartOutput:
        """Runs the chart-presence check by inspecting the workbook's chart XML parts directly.

        A workbook is a zip archive of XML parts; chart objects live under
        ``xl/charts/chartN.xml`` regardless of what sheet or drawing embeds
        them. This is checked at the archive level (not via openpyxl's chart
        API) because it must survive whatever chart-object model the
        authoring tool used.

        Args:
            source_input: Validated chart-check input.
            context: Source runtime context.

        Returns:
            Chart and data-series counts found in the workbook package.
        """
        import zipfile
        from xml.etree import ElementTree

        resolved = context.resolve_path(source_input.path)
        chart_count = 0
        max_series_count = 0
        c_ns = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"
        with zipfile.ZipFile(resolved) as archive:
            chart_names = [
                name
                for name in archive.namelist()
                if name.startswith("xl/charts/chart") and name.endswith(".xml")
            ]
            chart_count = len(chart_names)
            for name in chart_names:
                raw = archive.read(name)
                # The workbook is produced by the system under evaluation. OOXML chart parts never
                # declare a DOCTYPE or entities; refusing them removes entity-expansion attacks.
                if b"<!DOCTYPE" in raw[:4096] or b"<!ENTITY" in raw[:4096]:
                    continue
                tree = ElementTree.fromstring(raw)
                series_count = len(tree.findall(f".//{c_ns}ser"))
                max_series_count = max(max_series_count, series_count)
        return XlsxHasChartOutput(chart_count=chart_count, max_series_count=max_series_count)


COMMANDS = (ExtractText(), FormulaAuthenticityCheck(), FormulaSummary(), FormulaAuthenticityScan(), HasChart())

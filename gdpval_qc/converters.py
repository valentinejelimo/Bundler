"""Document conversion — Google Drive URL resolution and local Office-to-PDF conversion.

``DocumentConverter`` is an optional pre-processing step that runs between
``Package.load_samples()`` and ``DataSchema.infer()`` inside
``AnnotationEngine.setup()``.  It converts every Google Drive URL and every
local Office file path found in the batch to a locally-cached PDF, then
returns the updated batch.  Everything downstream (schema inference, assessor
generation, assessment) sees only plain ``.pdf`` file paths.

Backends
--------
Google Drive API
    Used for Google Drive URLs (always) and optionally for local Office files
    when LibreOffice is not available.  Requires one of:
    - ``credentials`` passed directly (``google.oauth2`` credentials object)
    - ``GOOGLE_SERVICE_ACCOUNT_JSON`` env var — path to a service-account JSON key
    - ``GOOGLE_API_KEY`` env var — for files shared as "anyone with the link"
    - Application Default Credentials (``gcloud auth application-default login``)
    Install: ``pip install google-api-python-client google-auth``

LibreOffice CLI
    Used for local Office files.  Auto-detected from common install locations.
    Install: ``brew install --cask libreoffice`` (macOS) or
             ``apt install libreoffice`` (Linux/Debian).

Backend selection for local Office files (``local_backend`` parameter):
    ``"auto"``        — LibreOffice if available, else Google Drive upload/export
    ``"libreoffice"`` — always LibreOffice; raises if not found
    ``"gdrive"``      — always Google Drive upload/export; raises if no credentials
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal


# ── URL / extension constants ──────────────────────────────────────────────────

# Matches individual Google Drive / Docs / Sheets / Slides file links.
# Captures the file ID in group 1.
GDRIVE_FILE_RE = re.compile(
    r"https://(?:"
    r"drive\.google\.com/file/d/"
    r"|docs\.google\.com/(?:document|spreadsheets|presentation|drawings)/d/"
    r")([a-zA-Z0-9_-]+)"
)

OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"}

# Spreadsheet extensions that are converted to CSV text instead of PDF.
# This avoids the multi-hundred-MB PDFs that LibreOffice produces for
# spreadsheets — the LLM reads CSV far more efficiently anyway.
SPREADSHEET_EXTENSIONS = {".xlsx", ".xls"}

# Plain-text extensions that should never be fed to LibreOffice.
# When a pipe-separated field contains one of these alongside an office file
# (which triggers _needs_conversion), _resolve_single copies the plain-text
# file into the converted dir as a .txt file rather than attempting PDF
# conversion.  Must stay in sync with _TEXT_FILE_EXTENSIONS in schema.py.
_PLAIN_TEXT_EXTENSIONS = {".csv", ".tsv", ".txt", ".md", ".rst", ".json",
                           ".yaml", ".yml", ".xml", ".html", ".htm"}

# Google Workspace MIME types that can be exported directly as PDF
_GOOGLE_WORKSPACE_EXPORT_MIMES = {
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
    "application/vnd.google-apps.drawing",
}

# Office MIME → extension (for naming temp files when downloading)
_MIME_TO_EXT: dict[str, str] = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/msword": ".doc",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.ms-powerpoint": ".ppt",
}

# Extension → Google Workspace MIME (for upload-and-convert path)
_EXT_TO_GOOGLE_MIME: dict[str, str] = {
    ".docx": "application/vnd.google-apps.document",
    ".doc":  "application/vnd.google-apps.document",
    ".xlsx": "application/vnd.google-apps.spreadsheet",
    ".xls":  "application/vnd.google-apps.spreadsheet",
    ".pptx": "application/vnd.google-apps.presentation",
    ".ppt":  "application/vnd.google-apps.presentation",
}

# Extension → upload MIME (original binary format)
_EXT_TO_UPLOAD_MIME: dict[str, str] = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":  "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":  "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt":  "application/vnd.ms-powerpoint",
}

_LIBREOFFICE_CANDIDATES = [
    "soffice",
    str(Path.home() / "Applications/LibreOffice.app/Contents/MacOS/soffice"),  # no-sudo user install
    "/usr/bin/soffice",
    "/usr/local/bin/soffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/opt/libreoffice/program/soffice",
]


# ── Public class ───────────────────────────────────────────────────────────────

class DocumentConverter:
    """Convert Google Drive–linked and local Office files to PDF before assessment.

    Pass an instance to ``AnnotationEngine`` via the ``document_converter``
    parameter.  The engine calls ``resolve()`` automatically between
    ``load_samples()`` and schema inference.

    Example::

        from quality_framework import AnthropicClientAdapter
        from quality_framework.converters import DocumentConverter
        from quality_framework.engine import AnnotationEngine

        converter = DocumentConverter()           # auto-detects LibreOffice
        engine = AnnotationEngine(
            client=AnthropicClientAdapter(),
            document_converter=converter,
        )
    """

    def __init__(
        self,
        credentials=None,
        local_backend: Literal["auto", "libreoffice", "gdrive"] = "auto",
        libreoffice_path: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        credentials:
            A ``google.oauth2`` credentials object.  If ``None``, the adapter
            tries ``GOOGLE_SERVICE_ACCOUNT_JSON``, ``GOOGLE_API_KEY``, then
            Application Default Credentials.
        local_backend:
            Backend for converting *local* Office files.  ``"auto"`` tries
            LibreOffice first, falls back to Google Drive upload/export.
        libreoffice_path:
            Explicit path to the ``soffice`` executable.  Auto-detected if None.
        """
        self._credentials = credentials
        self._local_backend = local_backend
        self._libreoffice_path = libreoffice_path
        self._drive_service = None   # lazy-initialised

    # ── Main entry point ───────────────────────────────────────────────────────

    def resolve(
        self,
        batch: list[dict],
        data_dir: Path | None,
        cache_dir: Path,
        skip_fields: set[str] | None = None,
        on_progress: "Callable[[int, int, str, bool], None] | None" = None,
    ) -> list[dict]:
        """Return a new batch with Drive URLs and Office paths replaced by local PDF paths.

        Converted PDFs are cached under ``cache_dir/converted/`` so repeated
        runs skip re-conversion.  The original batch is not mutated.

        For every field whose value was converted, a companion field
        ``{field}_original_exts`` is injected into each sample that had a
        conversion.  The companion value is the pipe-joined list of original
        file extensions (e.g. ``".xlsx|.pdf"``), preserving the same order as
        the (possibly pipe-separated) converted paths.  Drive URLs are recorded
        as ``".gdrive"``.  This lets downstream quality dimensions such as
        ``file_format_check`` inspect the true original file formats rather than
        seeing only ``.pdf`` for every converted document.

        Parameters
        ----------
        batch:        The loaded sample dicts.
        data_dir:     Base directory for resolving relative local file paths.
        cache_dir:    Directory for caching converted PDFs (will be created).
        skip_fields:  Field names to ignore entirely (e.g. fields marked
                      ``field_types: skip`` in the spec).  Values in these
                      fields are never converted, preventing timeouts when
                      a skipped field contains an unsupported file type.
        on_progress:  Optional callback invoked after each unique value is
                      resolved.  Signature:
                      ``on_progress(current, total, label, from_cache)``
                      where *current* is the 1-based index of the resolved
                      value, *total* is the number of unique values to
                      process, *label* is a short human-readable name for
                      the value (filename or "GDrive file"), and
                      *from_cache* is True when no conversion was needed.
        """
        converted_dir = cache_dir / "converted"
        converted_dir.mkdir(parents=True, exist_ok=True)

        # Collect unique values that need conversion (avoids duplicate work).
        # Values may be a single Drive URL / Office path, or a pipe-separated
        # list of them — both are handled uniformly.
        # Fields in skip_fields are excluded — their values will never be loaded
        # by the assessor so converting them is wasteful and can cause timeouts.
        _skip = skip_fields or set()
        pending: dict[str, str | None] = {}   # original_value → resolved string
        for sample in batch:
            for k, v in sample.items():
                if k in _skip:
                    continue
                if isinstance(v, str) and v not in pending:
                    if _needs_conversion(v):
                        pending[v] = None

        if not pending:
            return batch, 0, 0

        # Resolve each unique value; track how many were actually converted vs cached.
        # Progress is reported per individual file so the denominator equals the
        # total number of individual files to process (pipe-separated groups expand).
        n_converted = 0
        pending_keys = list(pending)
        n_files_total = sum(
            len([p for p in v.split("|") if p.strip()]) if "|" in v else 1
            for v in pending_keys
        )
        n_files_done = 0
        for value in pending_keys:
            if "|" in value:
                parts = [p.strip() for p in value.split("|") if p.strip()]
                pdf_parts = []
                for part in parts:
                    path, did_convert = self._resolve_single(part, data_dir, converted_dir)
                    pdf_parts.append(str(path))
                    if did_convert:
                        n_converted += 1
                    n_files_done += 1
                    if on_progress is not None:
                        on_progress(n_files_done, n_files_total, _progress_label(part), not did_convert)
                pending[value] = "|".join(pdf_parts)
            else:
                path, did_convert = self._resolve_single(value, data_dir, converted_dir)
                pending[value] = str(path)
                if did_convert:
                    n_converted += 1
                n_files_done += 1
                if on_progress is not None:
                    on_progress(n_files_done, n_files_total, _progress_label(value), not did_convert)

        # Pre-compute original-extension and original-filename strings for every
        # converted value.
        # _original_exts: pipe-joined extensions (".xlsx|.pdf") — kept for legacy.
        # _original_names: pipe-joined filenames ("model.xlsx|report.pdf") — used by
        #   format-check dimensions so the evaluating LLM sees the true file format
        #   without loading potentially large converted content.
        original_exts_map: dict[str, str] = {}
        original_names_map: dict[str, str] = {}
        for value in pending:
            if "|" in value:
                parts = [p.strip() for p in value.split("|") if p.strip()]
                original_exts_map[value] = "|".join(_original_ext(p) for p in parts)
                original_names_map[value] = "|".join(_original_name(p) for p in parts)
            else:
                original_exts_map[value] = _original_ext(value)
                original_names_map[value] = _original_name(value)

        # Build updated batch (new dicts, original unchanged).
        # Companion fields are collected separately and appended after the main
        # fields to preserve the original field ordering.
        updated = []
        for sample in batch:
            new_sample = {}
            companion: dict[str, str] = {}
            for k, v in sample.items():
                resolved = pending.get(v) if isinstance(v, str) else None
                new_sample[k] = resolved if resolved is not None else v
                if resolved is not None and v in original_exts_map:
                    companion[f"{k}_original_exts"] = original_exts_map[v]
                    companion[f"{k}_original_names"] = original_names_map[v]
            new_sample.update(companion)
            updated.append(new_sample)
        return updated, n_converted, len(pending)

    def convert_for_eval(
        self,
        src: Path,
        cache_dir: Path,
    ) -> tuple[Path, bool]:
        """Convert one file to its eval-ready cached form.

        Public wrapper around `_resolve_single` for v18-style callers that
        work in terms of a single file Path (not a batch dict). Returns
        `(dest_path, was_freshly_converted)`. The `dest_path` is:

          - the original file when it's already a `.pdf`
          - `<cache_dir>/converted/<stem>_<hash>.txt` for plain-text inputs
          - `<cache_dir>/converted/<stem>_<hash>.txt` for chart-free `.xlsx`
          - `<cache_dir>/converted/<stem>_<hash>.pdf` for all other Office files

        Cache key is the SHA-256 of the source file's bytes, so a re-run
        with the identical source is an O(1) lookup. A cached file produced
        by an earlier run is reused regardless of which process created it.
        """
        converted_dir = cache_dir / "converted"
        converted_dir.mkdir(parents=True, exist_ok=True)
        return self._resolve_single(str(src), data_dir=None, converted_dir=converted_dir)

    def _resolve_single(
        self, value: str, data_dir: Path | None, converted_dir: Path
    ) -> tuple[Path, bool]:
        """Resolve one Drive URL or local Office path to a cached local file.

        Returns ``(dest_path, converted)`` where ``converted`` is True if the
        file was freshly converted and False if the cached version was reused.

        Spreadsheets (.xlsx/.xls) are converted to a plain-text CSV file
        (``<stem>_<hash>.txt``) rather than PDF — this avoids the multi-hundred-MB
        PDFs LibreOffice produces and produces content that is more LLM-friendly.
        All other Office types and Drive URLs are still converted to PDF.
        """
        m = GDRIVE_FILE_RE.search(value)
        if m:
            file_id = m.group(1)
            dest = converted_dir / f"gdrive_{file_id}.pdf"
            if not dest.exists():
                self._download_gdrive(file_id, dest)
                return dest, True
            return dest, False
        else:
            full_path = (data_dir / value) if data_dir else Path(value)
            if full_path.suffix.lower() == ".pdf":
                # Already a PDF — pass through as-is, no conversion needed.
                return full_path, False
            elif full_path.suffix.lower() in _PLAIN_TEXT_EXTENSIONS:
                dest = converted_dir / f"{full_path.stem}_{_file_hash(full_path)}.txt"
                if not dest.exists():
                    dest.write_bytes(full_path.read_bytes())
                    return dest, True
            elif full_path.suffix.lower() in SPREADSHEET_EXTENSIONS and not _xlsx_has_charts(full_path):
                dest = converted_dir / f"{full_path.stem}_{_file_hash(full_path)}.txt"
                if not dest.exists():
                    _xlsx_to_text(full_path, dest)
                    return dest, True
            else:
                dest = converted_dir / f"{full_path.stem}_{_file_hash(full_path)}.pdf"
                if not dest.exists():
                    self._convert_local(full_path, dest)
                    return dest, True
            return dest, False

    # ── Google Drive backend ───────────────────────────────────────────────────

    def _get_drive_service(self):
        """Lazy-initialise and return the Drive v3 service."""
        if self._drive_service is not None:
            return self._drive_service
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "google-api-python-client is required for Google Drive support. "
                "Install with: pip install google-api-python-client google-auth"
            )

        creds = self._credentials
        if creds is None:
            sa_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
            api_key  = os.environ.get("GOOGLE_API_KEY")
            if sa_path:
                from google.oauth2 import service_account
                creds = service_account.Credentials.from_service_account_file(
                    sa_path,
                    scopes=["https://www.googleapis.com/auth/drive.readonly"],
                )
            elif api_key:
                from googleapiclient.discovery import build as _build
                self._drive_service = _build("drive", "v3", developerKey=api_key)
                return self._drive_service
            else:
                import google.auth
                creds, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/drive.readonly"]
                )

        from googleapiclient.discovery import build as _build
        self._drive_service = _build("drive", "v3", credentials=creds)
        return self._drive_service

    def _download_gdrive(self, file_id: str, dest: Path) -> None:
        """Download/export a single Drive file as PDF to ``dest``."""
        from googleapiclient.http import MediaIoBaseDownload

        service = self._get_drive_service()
        meta = service.files().get(fileId=file_id, fields="mimeType,name").execute()
        mime_type = meta["mimeType"]

        if mime_type in _GOOGLE_WORKSPACE_EXPORT_MIMES:
            # Google-native format → export directly as PDF
            request = service.files().export_media(
                fileId=file_id, mimeType="application/pdf"
            )
            _stream_to_file(request, dest)

        elif mime_type == "application/pdf":
            # Already a PDF — download as-is
            request = service.files().get_media(fileId=file_id)
            _stream_to_file(request, dest)

        else:
            # Uploaded binary Office file — download original, then convert locally
            ext = _MIME_TO_EXT.get(mime_type, ".bin")
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                request = service.files().get_media(fileId=file_id)
                _stream_to_file(request, tmp_path)
                self._convert_local(tmp_path, dest)
            finally:
                tmp_path.unlink(missing_ok=True)

    # ── Local Office conversion backend ───────────────────────────────────────

    def _convert_local(self, src: Path, dest: Path) -> None:
        """Convert a local Office file to PDF using the configured backend."""
        backend = self._local_backend
        if backend == "auto":
            backend = "libreoffice" if self._find_libreoffice() else "gdrive"

        if backend == "libreoffice":
            self._libreoffice_convert(src, dest)
        elif backend == "gdrive":
            self._gdrive_upload_convert(src, dest)
        else:
            raise ValueError(f"Unknown local_backend: {backend!r}")

    def _find_libreoffice(self) -> str | None:
        if self._libreoffice_path:
            return self._libreoffice_path if Path(self._libreoffice_path).exists() else None
        for candidate in _LIBREOFFICE_CANDIDATES:
            if shutil.which(candidate):
                return candidate
        return None

    def _libreoffice_convert(self, src: Path, dest: Path) -> None:
        soffice = self._find_libreoffice()
        if not soffice:
            raise RuntimeError(
                "LibreOffice not found. Install with 'brew install --cask libreoffice' "
                "(macOS) or 'apt install libreoffice' (Linux), or set "
                "local_backend='gdrive' to use Google Drive for conversion instead."
            )
        result = subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf",
             "--outdir", str(dest.parent), str(src)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LibreOffice conversion failed for {src.name}: {result.stderr.strip()}"
            )
        # LibreOffice writes <stem>.pdf into outdir; rename to the hashed dest name
        lo_output = dest.parent / (src.stem + ".pdf")
        if lo_output.exists() and lo_output != dest:
            lo_output.rename(dest)

    def _gdrive_upload_convert(self, src: Path, dest: Path) -> None:
        """Upload a local Office file to Drive, export as PDF, then delete the upload."""
        try:
            from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
        except ImportError:
            raise ImportError(
                "google-api-python-client is required for gdrive local_backend. "
                "Install with: pip install google-api-python-client google-auth"
            )

        ext = src.suffix.lower()
        upload_mime  = _EXT_TO_UPLOAD_MIME.get(ext, "application/octet-stream")
        google_mime  = _EXT_TO_GOOGLE_MIME.get(ext)
        if not google_mime:
            raise ValueError(
                f"Cannot convert {src.name} via Google Drive: "
                f"extension {ext!r} is not a supported Office format."
            )

        # Build a Drive service with write scope for the upload
        creds = self._credentials
        if creds is None:
            sa_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
            if sa_path:
                from google.oauth2 import service_account
                creds = service_account.Credentials.from_service_account_file(
                    sa_path,
                    scopes=["https://www.googleapis.com/auth/drive"],
                )
            else:
                import google.auth
                creds, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/drive"]
                )

        from googleapiclient.discovery import build as _build
        rw_service = _build("drive", "v3", credentials=creds)

        # Upload with conversion to Google Workspace format
        file_meta = {"name": src.name, "mimeType": google_mime}
        media = MediaFileUpload(str(src), mimetype=upload_mime, resumable=True)
        uploaded = rw_service.files().create(
            body=file_meta, media_body=media, fields="id"
        ).execute()
        upload_id = uploaded["id"]

        try:
            # Export the Google Workspace file as PDF
            request = rw_service.files().export_media(
                fileId=upload_id, mimeType="application/pdf"
            )
            _stream_to_file(request, dest)
        finally:
            # Always clean up the uploaded file
            try:
                rw_service.files().delete(fileId=upload_id).execute()
            except Exception:
                pass


# ── Module-level helpers ───────────────────────────────────────────────────────

def _xlsx_has_charts(path: Path) -> bool:
    """Return True if any sheet in the workbook contains charts.

    Falls back to True (→ PDF path) if openpyxl is not installed or the file
    cannot be opened, so the safer conversion is always chosen.
    Note: read_only=False is required — chart objects are not loaded in read-only mode.
    """
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True, read_only=False)
        has = any(getattr(wb[s], "_charts", None) for s in wb.sheetnames)
        wb.close()
        return has
    except Exception:
        return True  # safe fallback: treat as chart-containing → use PDF


_XLSX_SOLUTION_KEY_RE = re.compile(r"^'\[[^\]]+\](?P<sheet>[^']+)'!(?P<cell>[A-Z]+\d+)$")
_XLSX_ERR_TOKENS = ("#REF!", "#VALUE!", "#NAME?", "#DIV/0!", "#N/A", "#NULL!", "#NUM!")


def _xlsx_recompute_formula_map(src: Path) -> dict:
    """Best-effort formula recompute for one workbook, via the same ``formulas`` engine
    ``deterministic-checks/formula_health_check.py`` uses. Returns
    ``{(SHEET_NAME_UPPER, cell_coordinate): recomputed_value}`` for every formula cell it could
    resolve to a usable (non-null, non-NaN, non-error-shaped) value. Returns ``{}`` on ANY
    failure -- missing dependency, unsupported formula shape, corrupt file -- so a recompute
    problem never blocks conversion; the caller falls back to reading whatever is cached on
    disk for every cell, this function's pre-existing behavior before this fix.

    Why this exists: ``data_only=True`` below only ever returns the value that was cached in
    the workbook the last time a real spreadsheet engine saved it, never the live result of the
    formula -- a file saved by openpyxl (or produced by an xlsm->xlsx conversion step) commonly
    has every formula cell's cache blank or stale even though the formula itself is fine. This
    fed Internal QC's own judge blank/wrong numbers for perfectly good work. Mirrors the
    identical fix and reasoning in
    ``pipeline/templates/harbor/verifier_lib/sources/xlsx.py`` -- duplicated rather than shared
    because that module ships standalone inside each task's grading sandbox and this one runs
    inside the pipeline's own installed package.
    """
    try:
        try:
            import formulas
        except ImportError:
            return {}

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xl_model = formulas.ExcelModel().loads(str(src)).finish()
            solution = xl_model.calculate()
    except Exception:
        return {}

    def _unwrap(v):
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

    def _usable(v) -> bool:
        if v is None:
            return False
        if isinstance(v, float):
            try:
                import math

                if math.isnan(v):
                    return False
            except Exception:
                pass
        if isinstance(v, str) and any(tok in v for tok in _XLSX_ERR_TOKENS):
            return False
        return True

    recomputed: dict = {}
    for key, cell_range in solution.items():
        m = _XLSX_SOLUTION_KEY_RE.match(key)
        if not m:
            continue
        try:
            value = _unwrap(cell_range)
        except Exception:
            continue
        if not _usable(value):
            continue
        recomputed[(m.group("sheet").upper(), m.group("cell"))] = value
    return recomputed


def _xlsx_to_text(src: Path, dest: Path) -> None:
    """Convert an XLSX/XLS workbook to a plain-text CSV file.

    Each sheet is written as a CSV block preceded by a ``# Sheet: <name>``
    header.  This produces files that are orders of magnitude smaller than
    LibreOffice-rendered PDFs while being more readable for LLMs.

    Every formula cell is recomputed with a real spreadsheet engine first (best-effort, see
    ``_xlsx_recompute_formula_map``) and that live value is written in place of whatever was
    cached on disk, so a stale or blank formula cache can no longer read as blank/wrong to
    Internal QC's own judge. Non-formula cells are read exactly as before. Recompute is skipped
    silently (falling back to the original cached-only read) if the optional ``formulas``
    dependency isn't installed -- unlike the ``openpyxl`` requirement below, this is treated as
    a degrade, not a hard failure, since it was never a hard requirement before this fix.

    Requires ``openpyxl`` (``pip install 'quality-framework[converters]'``).
    Falls back to a one-line error message in the dest file if the import fails
    so that downstream errors are clear rather than silent.
    """
    try:
        import csv
        import io
        import openpyxl
    except ImportError:
        dest.write_text(
            f"[openpyxl not installed — cannot convert {src.name} to text. "
            "Run: pip install 'quality-framework[converters]']\n",
            encoding="utf-8",
        )
        return

    recomputed = _xlsx_recompute_formula_map(src)

    wb_cached = openpyxl.load_workbook(src, data_only=True, read_only=True)
    wb_formula = None
    if recomputed:
        try:
            wb_formula = openpyxl.load_workbook(src, data_only=False, read_only=True)
        except Exception:
            wb_formula = None

    buf = io.StringIO()
    for sheet_name in wb_cached.sheetnames:
        ws_cached = wb_cached[sheet_name]
        ws_formula = (
            wb_formula[sheet_name]
            if wb_formula is not None and sheet_name in wb_formula.sheetnames
            else None
        )
        buf.write(f"# Sheet: {sheet_name}\n")
        writer = csv.writer(buf)
        row_pairs = (
            zip(ws_cached.iter_rows(), ws_formula.iter_rows())
            if ws_formula is not None
            else ((row, None) for row in ws_cached.iter_rows())
        )
        for cached_row, formula_row in row_pairs:
            values = []
            for i, cached_cell in enumerate(cached_row):
                v = cached_cell.value
                if formula_row is not None and i < len(formula_row):
                    fcell = formula_row[i]
                    if isinstance(fcell.value, str) and fcell.value.startswith("="):
                        key = (sheet_name.upper(), fcell.coordinate)
                        if key in recomputed:
                            v = recomputed[key]
                values.append(v)
            writer.writerow(["" if v is None else v for v in values])
        buf.write("\n")
    wb_cached.close()
    if wb_formula is not None:
        wb_formula.close()
    dest.write_text(buf.getvalue(), encoding="utf-8")


def _original_ext(value: str) -> str:
    """Return the original file extension for a single (non-pipe) value.

    Returns ``".gdrive"`` for Google Drive URLs (format unknown from URL alone),
    or the lowercase suffix for local file paths (e.g. ``".xlsx"``).
    """
    if GDRIVE_FILE_RE.search(value):
        return ".gdrive"
    return Path(value).suffix.lower() or ""


def _original_name(value: str) -> str:
    """Return the original filename (basename) for a single (non-pipe) value.

    Returns ``"(gdrive)"`` for Google Drive URLs,
    or the basename for local file paths (e.g. ``"model.xlsx"``).
    """
    if GDRIVE_FILE_RE.search(value):
        return "(gdrive)"
    return Path(value).name or value


def _is_office_path(value: str) -> bool:
    return Path(value).suffix.lower() in OFFICE_EXTENSIONS


def _is_text_path(value: str) -> bool:
    """Return True if value is a plain-text file that needs copying to the converted dir.

    CSV/TSV files must be flagged here so they are always pre-processed to .txt
    files before schema inference.  Without this, the schema may infer
    ``file_path_list`` (PDF-only) from one sample, then at runtime encounter a
    CSV-only sample whose bytes get sent to the LLM as ``application/pdf`` —
    causing the API to reject it with "Unsupported MIME type: text/csv".
    """
    return Path(value).suffix.lower() in _PLAIN_TEXT_EXTENSIONS


def _needs_conversion(value: str) -> bool:
    """Return True if value is a Drive URL, Office path, text file, or pipe-list of any."""
    if "|" in value:
        parts = [p.strip() for p in value.split("|") if p.strip()]
        return len(parts) > 1 and any(
            GDRIVE_FILE_RE.search(p) or _is_office_path(p) or _is_text_path(p)
            for p in parts
        )
    return bool(GDRIVE_FILE_RE.search(value)) or _is_office_path(value) or _is_text_path(value)


def _progress_label(value: str) -> str:
    """Short human-readable label for a single conversion value (no pipes)."""
    if GDRIVE_FILE_RE.search(value):
        m = GDRIVE_FILE_RE.search(value)
        fid = m.group(1)[:10] if m else "?"
        return f"GDrive:{fid}"
    return Path(value).name or value


def _file_hash(path: Path) -> str:
    """Short content hash for cache-busting when a local file changes."""
    try:
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        return digest[:8]
    except OSError:
        return "unknown"


def _stream_to_file(request, dest: Path) -> None:
    """Stream a Drive API MediaIoBaseDownload request to a local file."""
    from googleapiclient.http import MediaIoBaseDownload
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    dest.write_bytes(buf.getvalue())

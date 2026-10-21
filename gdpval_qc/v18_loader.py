"""Task-corpus loader for v18 of the quality framework.

A v18 corpus is a directory of task subdirectories. Each task directory must
contain:

    task.toml             # metadata (TOML)
    instruction.md        # the prompt for the agent (markdown text)
    environment/          # input files the agent reads (directory)

Plus any number of additional paths the spec binds via `field_bindings`. The
canonical v18 spec binds:

    solution/golden_outputs/      → solution_files     (directory)
    solution/golden_outputs/Reasoning.docx → solution_reasoning (file, optional)
    tests/verifier.json           → verifier_spec_json (file, optional)
    [callback]                    → verifier_assertion_summary (optional)

The loader produces one `Sample` per task subdirectory with every field
either set or `None` (for optional bindings whose source is absent).
"""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

from .v18_spec import V18Spec, FieldBinding

# ── Public types ───────────────────────────────────────────────────────────

@dataclass
class FileContent:
    """One file from a directory- or file-bound field.

    ``text`` is the extracted text content (best-effort; head+tail truncated
    when the source exceeds `_MAX_FIELD_CHARS`). ``path`` is the original
    location on disk. When the loader is constructed with a
    `DocumentConverter` and a `cache_dir`, ``cached_path`` is populated with
    the eval-ready cached form (PDF or text) and ``is_pdf`` indicates
    whether that cached form can be sent to a multimodal judge as a
    DocumentBlock.

    Backward-compat: ``cached_path`` and ``is_pdf`` default to None / False,
    so callers that build FileContent directly (tests, etc.) keep working
    without modification.

    ``conversion_status`` records whether the DocumentConverter ran: ``"ok"``
    (or ``"not_attempted"`` when no converter was configured) vs ``"failed"``.
    On failure the judge falls back to ``text`` alone, which can be degraded
    (e.g. raw bytes for Office files), so ``conversion_error`` carries the
    reason and the loader surfaces it to the run output rather than silently
    proceeding.
    """

    name:        str
    path:        Path
    text:        str       # extracted text content; head+tail truncated if huge
    cached_path: Path | None = None       # PDF or text, after DocumentConverter
    is_pdf:      bool = False             # True iff cached_path is a .pdf
    conversion_status: str = "not_attempted"   # "ok" | "failed" | "not_attempted"
    conversion_error:  str | None = None        # exception text when status == "failed"

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "path":        str(self.path),
            "text":        self.text,
            "cached_path": str(self.cached_path) if self.cached_path else None,
            "is_pdf":      self.is_pdf,
            "conversion_status": self.conversion_status,
            "conversion_error":  self.conversion_error,
        }


@dataclass
class Sample:
    """A single task, fully loaded."""

    task_id:   str
    task_dir:  Path
    fields:    dict[str, Any] = field(default_factory=dict)

    def has(self, name: str) -> bool:
        v = self.fields.get(name)
        return v is not None and v != "" and v != []


# ── Default fields ─────────────────────────────────────────────────────────
# These are populated by the loader for every task regardless of the spec's
# field_bindings. Spec field_bindings layer on top.

DEFAULT_TASK_FIELDS: tuple[str, ...] = (
    "task_id", "category", "difficulty", "tags", "annotatorID",
    "instruction", "input_files",
)


# ── Loader ─────────────────────────────────────────────────────────────────

class TaskCorpusLoader:
    """Loads v18 samples from a task-corpus directory.

    When ``converter`` and ``cache_dir`` are both supplied, every file
    surfaced via file/directory bindings is also converted to its eval-ready
    cached form (PDF for Office files, copy for plain text, .txt CSV for
    chart-free spreadsheets) using ``DocumentConverter.convert_for_eval``.
    The cached path lands on ``FileContent.cached_path`` and the
    ``is_pdf`` flag indicates whether it can be sent to multimodal judges
    as a DocumentBlock. Repeat runs against the same source files are
    O(1) cache hits.
    """

    def __init__(
        self,
        root: Path,
        spec: V18Spec,
        callbacks: dict[str, Callable[["Sample"], Any]] | None = None,
        *,
        converter: Any | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Task corpus root not found: {self.root}")
        self.spec = spec
        # Built-in loader callbacks are auto-registered; user-supplied
        # callbacks override built-ins on name collision.
        from .v18_verifier_summary import build_verifier_assertion_summary
        from .v18_trajectory_digest import (
            build_intermodel_trajectory_digest,
            build_failed_trial_deliverable_digest,
            build_passed_trial_deliverable_digest,
        )
        from .upstream_deliverables import build_upstream_deliverable_inputs
        builtins = {
            "build_verifier_assertion_summary": build_verifier_assertion_summary,
            "build_intermodel_trajectory_digest": build_intermodel_trajectory_digest,
            "build_failed_trial_deliverable_digest": build_failed_trial_deliverable_digest,
            "build_passed_trial_deliverable_digest": build_passed_trial_deliverable_digest,
            "build_upstream_deliverable_inputs": build_upstream_deliverable_inputs,
        }
        self.callbacks = {**builtins, **(callbacks or {})}
        self.converter = converter
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        if (converter is not None) ^ (cache_dir is not None):
            raise ValueError(
                "converter and cache_dir must be supplied together "
                "(or neither, to disable conversion)."
            )

    def task_ids(self) -> list[str]:
        """All task IDs in the corpus, in deterministic order."""
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def load_sample(self, task_id: str) -> Sample:
        task_dir = self.root / task_id
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        sample = Sample(task_id=task_id, task_dir=task_dir, fields={})

        # Layer 1: standard fields from task.toml / instruction.md / environment/
        self._populate_default_fields(sample)

        # Layer 2: each binding from the spec
        for binding in self.spec.field_bindings:
            sample.fields[binding.field_name] = self._resolve_binding(sample, binding)

        return sample

    def load_all(self) -> list[Sample]:
        return [self.load_sample(tid) for tid in self.task_ids()]

    # ── Internals ──────────────────────────────────────────────────────────

    def _populate_default_fields(self, sample: Sample) -> None:
        sample.fields["task_id"] = sample.task_id

        # task.toml metadata
        toml_path = sample.task_dir / "task.toml"
        meta: dict[str, Any] = {}
        if toml_path.is_file():
            try:
                with open(toml_path, "rb") as f:
                    raw = tomllib.load(f)
                meta = raw.get("metadata") or {}
                # Valid TOML can still declare `metadata` as a non-table (e.g.
                # `metadata = "bad"`). Normalise to a recoverable parse error so
                # the later meta.get(...) calls don't raise AttributeError.
                if not isinstance(meta, dict):
                    meta = {"_parse_error":
                            f"metadata must be a table, got {type(meta).__name__}"}
            except Exception as exc:
                meta = {"_parse_error": str(exc)}

        # Surface a metadata parse/normalisation problem so reviewers can see it
        # rather than silently getting empty category/difficulty/etc.
        if meta.get("_parse_error"):
            sample.fields["_parse_error"] = meta["_parse_error"]

        # `category` falls back to `domain` (legacy field used by older v18 data)
        sample.fields["category"]    = str(meta.get("category", meta.get("domain", "")))
        sample.fields["difficulty"]  = str(meta.get("difficulty", ""))
        sample.fields["tags"]        = list(meta.get("tags", []))
        sample.fields["annotatorID"] = str(meta.get("annotatorID", ""))

        # instruction.md
        instr_path = sample.task_dir / "instruction.md"
        sample.fields["instruction"] = (
            instr_path.read_text(encoding="utf-8", errors="replace")
            if instr_path.is_file() else ""
        )

        # environment/ as input_files — but exclude infrastructure files
        # (Dockerfile, docker-compose, etc.) so quality dims don't grade
        # them as task inputs.
        env_dir = sample.task_dir / "environment"
        if env_dir.is_dir():
            files = _list_directory_files(env_dir, exclude=_ENVIRONMENT_DEFAULT_EXCLUDE)
            if self.converter is not None and self.cache_dir is not None:
                for fc in files:
                    _attach_cached_path(fc, self.converter, self.cache_dir)
                _warn_failed_conversions(sample.task_id, files)
            sample.fields["input_files"] = files
        else:
            sample.fields["input_files"] = []

    def _resolve_binding(self, sample: Sample, binding: FieldBinding) -> Any:
        if binding.source == "file":
            return self._resolve_file(sample, binding)
        if binding.source == "directory":
            return self._resolve_directory(sample, binding)
        if binding.source == "callback":
            return self._resolve_callback(sample, binding)
        if binding.source == "text":
            # `text` source means "this field is already on the sample" — no-op
            return sample.fields.get(binding.field_name)
        raise ValueError(f"Unknown field-binding source: {binding.source}")

    def _resolve_file(self, sample: Sample, binding: FieldBinding) -> str | None:
        if not binding.path:
            raise ValueError(
                f"Field binding '{binding.field_name}' source=file requires `path`"
            )
        full = sample.task_dir / binding.path
        if not full.is_file():
            if binding.optional:
                return None
            raise FileNotFoundError(
                f"Required file missing for field '{binding.field_name}': {full}"
            )
        return _extract_text(full)

    def _resolve_directory(
        self, sample: Sample, binding: FieldBinding
    ) -> list[FileContent] | None:
        if not binding.path:
            raise ValueError(
                f"Field binding '{binding.field_name}' source=directory requires `path`"
            )
        full = sample.task_dir / binding.path
        if not full.is_dir():
            if binding.optional:
                return None
            raise FileNotFoundError(
                f"Required directory missing for field '{binding.field_name}': {full}"
            )
        files = _list_directory_files(full, exclude=tuple(binding.exclude))
        if self.converter is not None and self.cache_dir is not None:
            for fc in files:
                _attach_cached_path(fc, self.converter, self.cache_dir)
            _warn_failed_conversions(sample.task_id, files)
        return files

    def _resolve_callback(self, sample: Sample, binding: FieldBinding) -> Any:
        if not binding.callback:
            raise ValueError(
                f"Field binding '{binding.field_name}' source=callback requires `callback`"
            )
        fn = self.callbacks.get(binding.callback)
        if fn is None:
            if binding.optional:
                return None
            raise KeyError(
                f"Callback not registered for field '{binding.field_name}': "
                f"{binding.callback!r}. Pass it via TaskCorpusLoader(callbacks=...)."
            )
        return fn(sample)


# ── File reading ───────────────────────────────────────────────────────────

# Suffixes that don't need conversion (already PDF, or plain text we can read).
# DocumentConverter._needs_conversion treats these uniformly; we mirror its
# behaviour here as a fast-path to avoid the conversion cache lookup when
# it's unnecessary.
_PLAIN_TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".md", ".rst", ".json",
                         ".log", ".yaml", ".yml"}

# Infrastructure files in environment/ that the framework treats as
# build-time setup, NOT as agent inputs. Matched against bare filename via
# fnmatch globs by `_list_directory_files`. Excluding these prevents the
# quality dims from grading the Dockerfile or docker-compose config as if
# they were part of the task's input materials.
_ENVIRONMENT_DEFAULT_EXCLUDE: tuple[str, ...] = (
    "Dockerfile",
    "Dockerfile.*",
    "docker-compose.yaml",
    "docker-compose.yml",
    "docker-compose.*.yaml",
    "docker-compose.*.yml",
    ".dockerignore",
    ".env",
    ".env.*",
    ".gitignore",
    ".DS_Store",
    "Makefile",
    "*.lock",   # poetry.lock, package-lock.json, Cargo.lock, etc.
)


def _attach_cached_path(fc: "FileContent", converter: Any, cache_dir: Path) -> None:
    """Populate `fc.cached_path` and `fc.is_pdf` by calling the converter.

    Catches exceptions to keep loading resilient — if a single file fails to
    convert, the rest of the corpus still loads. On failure the fallback is
    text-only (no cached_path, is_pdf stays False), but the failure is recorded
    in `fc.conversion_status`/`fc.conversion_error` so callers can surface it
    instead of silently trusting a degraded (text-only) assessment.
    """
    try:
        dest, _was_fresh = converter.convert_for_eval(fc.path, cache_dir)
        fc.cached_path = dest
        fc.is_pdf = dest.suffix.lower() == ".pdf"
        fc.conversion_status = "ok"
        fc.conversion_error = None
    except Exception as exc:
        # Conversion failed; the loader still has the extracted text, so the
        # judge can fall back to that. Record the failure so it is visible.
        fc.cached_path = None
        fc.is_pdf = False
        fc.conversion_status = "failed"
        fc.conversion_error = f"{type(exc).__name__}: {exc}"


def _warn_failed_conversions(task_id: str, files: "list[FileContent]") -> None:
    """Emit a stderr warning for any files whose conversion failed.

    A degraded (text-only) fallback can produce normal-looking QC artifacts over
    content the judge never actually saw (e.g. raw bytes for Office files), so
    make the failure visible at load time. The per-file status is also carried
    into the output via ``FileContent.to_dict`` for downstream inspection.
    """
    failed = [fc for fc in files if fc.conversion_status == "failed"]
    for fc in failed:
        print(
            f"WARNING [{task_id}]: document conversion failed for {fc.name!r} "
            f"({fc.conversion_error}); judging will fall back to extracted text only.",
            file=sys.stderr,
        )


def _list_directory_files(
    directory: Path, exclude: tuple[str, ...] = ()
) -> list[FileContent]:
    """Alphabetical listing of files under `directory`, with optional glob excludes.

    Subdirectories are walked recursively (alphabetically), files within each
    subdirectory ordered alphabetically too. This produces a deterministic
    ordering that matches reviewer expectations.
    """
    out: list[FileContent] = []
    for p in sorted(directory.rglob("*")):
        if not p.is_file():
            continue
        # Exclude globs match against bare filename
        if any(fnmatch(p.name, pat) for pat in exclude):
            continue
        out.append(FileContent(name=p.name, path=p.resolve(), text=_extract_text(p)))
    return out


def _extract_text(path: Path) -> str:
    """Best-effort text extraction for the common file formats in this corpus.

    Returns a short placeholder string for binary types the loader can't read
    directly. The loader never raises on extraction failure — failures become
    visible text the judge can see.
    """
    suffix = path.suffix.lower()

    if suffix in (".md", ".txt", ".csv", ".tsv", ".json", ".toml",
                  ".yaml", ".yml", ".log", ".py", ".sh", ".sql"):
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"[failed to read: {exc}]"

    if suffix == ".docx":
        return _docx_to_text(path)
    if suffix == ".xlsx":
        return _xlsx_to_text(path)
    if suffix == ".pdf":
        return _pdf_to_text(path)

    if suffix in (".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".webp"):
        return f"[binary image: {path.name}]"

    # Generic fallback
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return f"[unsupported file type: {path.name}]"


def _docx_to_text(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        chunks = [p.text for p in doc.paragraphs if p.text]
        # Tables — append as tab-separated rows
        for table in doc.tables:
            for row in table.rows:
                row_text = "\t".join(c.text for c in row.cells)
                if row_text.strip():
                    chunks.append(row_text)
        return "\n".join(chunks)
    except Exception as exc:
        return f"[docx extraction failed: {exc}]"


def _xlsx_to_text(path: Path) -> str:
    try:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        out: list[str] = []
        for name in wb.sheetnames:
            out.append(f"=== Sheet: {name} ===")
            ws = wb[name]
            for row in ws.iter_rows(values_only=True):
                cells = ["" if c is None else str(c) for c in row]
                if any(cells):
                    out.append("\t".join(cells))
        wb.close()
        return "\n".join(out)
    except Exception as exc:
        return f"[xlsx extraction failed: {exc}]"


def _pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        chunks = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception as exc:
                txt = f"[page {i+1} extraction failed: {exc}]"
            chunks.append(txt)
        return "\n\n".join(chunks)
    except Exception as exc:
        return f"[pdf extraction failed: {exc}]"

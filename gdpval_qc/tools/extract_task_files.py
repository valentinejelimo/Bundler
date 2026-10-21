#!/usr/bin/env python3
"""Extract an omega task's files as text for local QC.

Tasks follow the expert "Shared Drive" layout:

    <task>/
      Source Documents/   ← files referenced in the prompt (the QC source files)
      Seed Documents/     ← material used to generate inputs (NOT QC'd)
      Expert Solution/    ← the expert's golden solution & reasoning
      Claude Output/      ← a model's attempt (NOT QC'd, ignored)

The prompt is NOT stored in the task folder — pass it with --prompt-file, or
paste it straight into the QC conversation. Folder names are matched
case-insensitively and tolerate trailing spaces (e.g. "Source Documents ").

This prints the source files (and, with --solution, the Expert Solution) as
plain text so an agent can QC a task without binary-file tooling.

Usage:
    python extract_task_files.py <task_dir>                       # source files
    python extract_task_files.py <task_dir> --prompt-file p.md    # + the prompt
    python extract_task_files.py <task_dir> --solution           # + Expert Solution

Office formats need optional libraries (text/PDF work without them):
    pip install python-docx openpyxl pypdf
"""

from __future__ import annotations

import argparse
import sys
from fnmatch import fnmatch
from pathlib import Path

# Per-file truncation guard so a huge spreadsheet can't blow the context window.
DEFAULT_MAX_CHARS = 100_000

# Infrastructure files that are build setup, not task inputs.
INFRA_EXCLUDE: tuple[str, ...] = (
    "Dockerfile", "Dockerfile.*",
    "docker-compose.yaml", "docker-compose.yml",
    "docker-compose.*.yaml", "docker-compose.*.yml",
    ".dockerignore", ".env", ".env.*", ".gitignore", ".DS_Store",
    "Makefile", "*.lock",
)

# Shared Drive folder names (matched case-insensitively, trailing-space tolerant).
SOURCE_DIR_NAMES = ("Source Documents",)
SOLUTION_DIR_NAMES = ("Expert Solution",)

TEXT_SUFFIXES = {
    ".md", ".txt", ".csv", ".tsv", ".json", ".toml",
    ".yaml", ".yml", ".log", ".py", ".sh", ".sql", ".rst",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".bmp", ".webp", ".svg"}


# ── Text extraction ─────────────────────────────────────────────────────────

def extract_text(path: Path, max_chars: int) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"[failed to read: {exc}]"
    elif suffix == ".docx":
        text = _docx_to_text(path)
    elif suffix == ".xlsx":
        text = _xlsx_to_text(path)
    elif suffix == ".pdf":
        text = _pdf_to_text(path)
    elif suffix in IMAGE_SUFFIXES:
        return f"[binary image: {path.name} — open it with the Read tool to inspect visually]"
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return f"[unsupported file type: {path.name}]"
    return _truncate(text, max_chars)


def _docx_to_text(path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        return "[.docx not extracted — run: pip install python-docx]"
    try:
        doc = Document(str(path))
        chunks = [p.text for p in doc.paragraphs if p.text]
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
    except ImportError:
        return "[.xlsx not extracted — run: pip install openpyxl]"
    try:
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
    except ImportError:
        return "[.pdf not extracted — run: pip install pypdf (or open it with the Read tool)]"
    try:
        reader = PdfReader(str(path))
        chunks = []
        for i, page in enumerate(reader.pages):
            try:
                chunks.append(page.extract_text() or "")
            except Exception as exc:
                chunks.append(f"[page {i + 1} extraction failed: {exc}]")
        return "\n\n".join(chunks)
    except Exception as exc:
        return f"[pdf extraction failed: {exc}]"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2:]
    return head + f"\n…[truncated {len(text) - limit} chars]…\n" + tail


def _list_files(directory: Path,
                exclude: tuple[str, ...] = ()) -> tuple[list[Path], int]:
    """Return (kept files, count excluded by the infrastructure globs)."""
    out: list[Path] = []
    excluded = 0
    for p in sorted(directory.rglob("*")):
        if not p.is_file():
            continue
        if any(fnmatch(p.name, pat) for pat in exclude):
            excluded += 1
            continue
        out.append(p)
    return out, excluded


# ── Folder resolution ───────────────────────────────────────────────────────

def _find_subdir(task_dir: Path, *names: str) -> Path | None:
    """First path matching a candidate name/relative-path under `task_dir`.

    Each candidate is tried two ways, in order:
      1. As an exact relative path from `task_dir` — supports multi-segment
         paths like SVC's ``environment/data`` or ``solution/golden``.
      2. For single-segment candidates only, as a case-insensitive,
         trailing-space-tolerant match against `task_dir`'s immediate
         children — the original Shared-Drive folder-name convention
         (e.g. "Source Documents ").
    """
    for name in names:
        candidate = task_dir / name
        if candidate.is_dir():
            return candidate

    single_segment = {n.strip().casefold() for n in names if "/" not in n}
    if not single_segment:
        return None
    try:
        children = sorted(task_dir.iterdir())
    except OSError:
        return None
    for p in children:
        if p.is_dir() and p.name.strip().casefold() in single_segment:
            return p
    return None


def resolve_source_dir(task_dir: Path, names: tuple[str, ...] = SOURCE_DIR_NAMES) -> Path | None:
    return _find_subdir(task_dir, *names)


def resolve_solution_dir(task_dir: Path, names: tuple[str, ...] = SOLUTION_DIR_NAMES) -> Path | None:
    return _find_subdir(task_dir, *names)


# ── Printers ────────────────────────────────────────────────────────────────

def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _print_dir(label: str, directory: Path, max_chars: int,
               exclude: tuple[str, ...] = ()) -> None:
    files, excluded = _list_files(directory, exclude=exclude)
    suffix = f" ({len(files)} file(s)"
    suffix += f"; {excluded} excluded as infrastructure)" if excluded else ")"
    _hr(f"{label}{suffix}")
    if not files:
        print("[no files]")
        return
    for p in files:
        rel = p.relative_to(directory)
        print(f"\n--- {rel}  (original_extension={p.suffix.lower() or 'none'}) ---")
        print(extract_text(p, max_chars))


def print_overview(task_dir: Path,
                   source_dir: Path | None, solution_dir: Path | None) -> None:
    _hr(f"TASK: {task_dir.name}   ({task_dir})")
    print("\nContents:")
    src = f"present  → {source_dir.name!r}" if source_dir else "ABSENT"
    sol = f"present  → {solution_dir.name!r}" if solution_dir else "absent (solution may not be ready yet)"
    print(f"  - Source Documents (QC source files)   {src}")
    print(f"  - Expert Solution (golden)             {sol}")
    print("  - Other folders (Seed Documents, Claude Output, …) are not QC'd.")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir", type=Path, help="Path to a single task directory")
    ap.add_argument("--prompt-file", type=Path, default=None,
                    help="Path to the prompt text (the prompt is not in the task folder)")
    ap.add_argument("--solution", action="store_true",
                    help="Also extract the Expert Solution (golden)")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help=f"Per-file head+tail truncation limit (default {DEFAULT_MAX_CHARS})")
    ap.add_argument("--source-dir-names", nargs="+", metavar="NAME", default=list(SOURCE_DIR_NAMES),
                    help="Candidate source-folder names/relative paths, tried in order "
                         f"(default: {list(SOURCE_DIR_NAMES)!r}; e.g. --source-dir-names environment/data "
                         "for the SVC layout)")
    ap.add_argument("--solution-dir-names", nargs="+", metavar="NAME", default=list(SOLUTION_DIR_NAMES),
                    help="Candidate solution-folder names/relative paths, tried in order "
                         f"(default: {list(SOLUTION_DIR_NAMES)!r}; e.g. --solution-dir-names solution/golden "
                         "for the SVC layout)")
    args = ap.parse_args()

    task_dir: Path = args.task_dir.expanduser().resolve()
    if not task_dir.is_dir():
        print(f"ERROR: not a directory: {task_dir}", file=sys.stderr)
        return 2

    source_dir = resolve_source_dir(task_dir, tuple(args.source_dir_names))
    solution_dir = resolve_solution_dir(task_dir, tuple(args.solution_dir_names))

    print_overview(task_dir, source_dir, solution_dir)

    # Prompt — supplied by the expert, not stored in the task folder.
    _hr("PROMPT (provided by the expert — not stored in the task folder)")
    if args.prompt_file is not None:
        pf = args.prompt_file.expanduser()
        print(extract_text(pf, args.max_chars) if pf.is_file()
              else f"[missing: {pf} does not exist]")
    else:
        print("[no --prompt-file given — paste the prompt text into the QC "
              "conversation so it can be graded with the source files]")

    # Source files
    if source_dir is not None:
        _print_dir(f"SOURCE FILES — {source_dir.name}/", source_dir,
                   args.max_chars, exclude=INFRA_EXCLUDE)
    else:
        _hr("SOURCE FILES")
        print("[missing: no 'Source Documents/' folder found — nothing to QC]")

    # Expert solution (golden) — only on request.
    if args.solution:
        if solution_dir is not None:
            _print_dir(f"EXPERT SOLUTION (golden) — {solution_dir.name}/",
                       solution_dir, args.max_chars)
        else:
            _hr("EXPERT SOLUTION (golden)")
            print("[missing: no 'Expert Solution/' folder found — "
                  "solution may not be ready yet]")

    return 0


if __name__ == "__main__":
    sys.exit(main())

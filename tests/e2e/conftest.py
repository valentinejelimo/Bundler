"""Shared fixtures for the E2E test suite.

These tests exercise the CLI end-to-end by spawning the project's pinned venv
Python as a subprocess. ``cycle.py`` resolves its project root from its own
location (``BASE = Path(__file__).resolve().parent``), so it always reads and
writes the real ``<project>/cycles/`` tree. We isolate each test with a unique
cycle date (``e2e-<uuid>``) and tear it down afterwards so no test data leaks
into the real cycles/ folder.

Helpers are exposed as pytest fixtures so test modules don't need to import
this conftest directly (pytest auto-discovers it).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(r"C:\Users\LENOVO\Desktop\Bundler")
VENV_PY = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CYCLES = PROJECT_ROOT / "cycles"
CYCLE_PY = PROJECT_ROOT / "cycle.py"
SVC_NOEVAL_SPEC = PROJECT_ROOT / "specs" / "overlays" / "svc-noeval.md"


# ── Path constants exposed via fixtures (so tests don't import this module) ─

@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def venv_py() -> Path:
    return VENV_PY


@pytest.fixture
def cycles_dir() -> Path:
    return CYCLES


@pytest.fixture
def svc_noeval_spec() -> Path:
    return SVC_NOEVAL_SPEC


# ── Subprocess runners ──────────────────────────────────────────────────────

@pytest.fixture
def run_cycle():
    """Invoke ``cycle.py <args>`` with the venv Python, cwd at the project root."""
    def _run(args: list[str], **kw) -> subprocess.CompletedProcess[str]:
        cmd = [str(VENV_PY), str(CYCLE_PY), *args]
        return subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, **kw
        )
    return _run


@pytest.fixture
def run_cli_module():
    """Invoke ``python -m gdpval_qc.cli <args>`` with the venv Python."""
    def _run(args: list[str], **kw) -> subprocess.CompletedProcess[str]:
        cmd = [str(VENV_PY), "-m", "gdpval_qc.cli", *args]
        return subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, **kw
        )
    return _run


# ── Cycle-date isolation ────────────────────────────────────────────────────

@pytest.fixture
def cycle_date():
    """A unique cycle date directory under <project>/cycles/, cleaned up after."""
    date = f"e2e-{uuid.uuid4().hex[:12]}"
    cdir = CYCLES / date
    cdir.mkdir(parents=True, exist_ok=False)
    try:
        yield date
    finally:
        try:
            if cdir.exists():
                shutil.rmtree(cdir, ignore_errors=True)
        except Exception:
            pass


# ── Task factory ────────────────────────────────────────────────────────────

@pytest.fixture
def make_fetched_task():
    """Factory: create a raw task under cycles/<date>/00_fetched/<id>/.

    Returns a function(date, tid, *, prompt, rubric_json, row_extra,
    source_files, solution_files, shipped_verifier, source_subdir,
    solution_subdir) -> Path.
    """
    def _make(cycle_date: str, tid: str, *, prompt: str = "Analyze the report.",
              rubric_json: str | None = None, row_extra: dict | None = None,
              source_files: dict | None = None, solution_files: dict | None = None,
              shipped_verifier: str | None = None,
              source_subdir: str = "source",
              solution_subdir: str = "solution") -> Path:
        d = CYCLES / cycle_date / "00_fetched" / tid
        (d / source_subdir).mkdir(parents=True)
        (d / solution_subdir).mkdir(parents=True)
        row: dict = {"Prompt": prompt}
        if rubric_json is not None:
            row["Rubric JSON"] = rubric_json
        if row_extra:
            row.update(row_extra)
        (d / "_row.json").write_text(json.dumps(row), encoding="utf-8")
        for name, content in (source_files or {}).items():
            (d / source_subdir / name).write_text(content, encoding="utf-8")
        for name, content in (solution_files or {}).items():
            (d / solution_subdir / name).write_text(content, encoding="utf-8")
        if shipped_verifier is not None:
            (d / "verifier.json").write_text(shipped_verifier, encoding="utf-8")
        return d
    return _make

"""Integration tests for verifier_lib/sources/registry.py and source parsers.

Tests SourceRegistry with a temp workspace exercising csv, json, text, md
parsers. xlsx/docx/pdf/pptx are marked blocked (no libs installed).

Scope: area 3 — source registry + parsers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifier_lib.models import FileCommand, Source
from verifier_lib.source_types import SourceAuthoringError
from verifier_lib.sources.registry import SourceRegistry


# ── registry helpers ────────────────────────────────────────────────────────

def _file_source(file_type: str, command: str, path: str) -> Source:
    return Source(
        type="file",
        file=FileCommand(type=file_type, command=command, arguments={"path": path}),
    )


def _registry(workspace: Path, tmp_path: Path) -> SourceRegistry:
    return SourceRegistry(workspace, agent_logs_dir=tmp_path / "logs",
                          max_content_chars=90000)


# ── csv.extract_text ────────────────────────────────────────────────────────

def test_csv_extract_text(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    csv_text = "name,value\nalice,1\nbob,2\n"
    (workspace / "data.csv").write_text(csv_text, encoding="utf-8")
    reg = _registry(workspace, tmp_path)
    out = reg.run(_file_source("csv", "extract_text", "data.csv"))
    assert out.text == csv_text


def test_csv_extract_text_rejects_non_csv(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "data.txt").write_text("x", encoding="utf-8")
    reg = _registry(workspace, tmp_path)
    with pytest.raises(SourceAuthoringError, match="source argument validation failed"):
        reg.run(_file_source("csv", "extract_text", "data.txt"))


def test_csv_extract_text_caps_long_content(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    long_text = "a,b\n" + "x,1\n" * 5000
    (workspace / "big.csv").write_text(long_text, encoding="utf-8")
    reg = SourceRegistry(workspace, agent_logs_dir=tmp_path / "logs",
                         max_content_chars=200)
    out = reg.run(_file_source("csv", "extract_text", "big.csv"))
    assert len(out.text) <= 200
    assert "TRUNCATED" in out.text


# ── json.read_file ─────────────────────────────────────────────────────────

def test_json_read_file(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    data = {"count": 3, "items": ["a", "b", "c"]}
    (workspace / "out.json").write_text(json.dumps(data), encoding="utf-8")
    reg = _registry(workspace, tmp_path)
    out = reg.run(_file_source("json", "read_file", "out.json"))
    assert out.root == data


def test_json_read_file_rejects_non_json(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "data.csv").write_text("x", encoding="utf-8")
    reg = _registry(workspace, tmp_path)
    with pytest.raises(SourceAuthoringError, match="source argument validation failed"):
        reg.run(_file_source("json", "read_file", "data.csv"))


def test_json_read_file_missing(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    reg = _registry(workspace, tmp_path)
    # FileNotFoundError is treated as agent-output failure, not SourceAuthoring
    with pytest.raises(FileNotFoundError):
        reg.run(_file_source("json", "read_file", "missing.json"))


def test_json_read_file_invalid_json(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "out.json").write_text("{not json}", encoding="utf-8")
    reg = _registry(workspace, tmp_path)
    with pytest.raises(json.JSONDecodeError):
        reg.run(_file_source("json", "read_file", "out.json"))


# ── text.extract_text ──────────────────────────────────────────────────────

def test_text_extract_text(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "answer.txt").write_text("hello world\n", encoding="utf-8")
    reg = _registry(workspace, tmp_path)
    out = reg.run(_file_source("text", "extract_text", "answer.txt"))
    assert out.text == "hello world\n"


def test_text_extract_text_any_extension(tmp_path: Path):
    """text.extract_text intentionally accepts any extension."""
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "answer.log").write_text("log line\n", encoding="utf-8")
    reg = _registry(workspace, tmp_path)
    out = reg.run(_file_source("text", "extract_text", "answer.log"))
    assert out.text == "log line\n"


# ── md.extract_text ────────────────────────────────────────────────────────

def test_md_extract_text(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    md_text = "# Title\n\nSome **bold** markdown.\n"
    (workspace / "doc.md").write_text(md_text, encoding="utf-8")
    reg = _registry(workspace, tmp_path)
    out = reg.run(_file_source("md", "extract_text", "doc.md"))
    assert out.text == md_text


def test_md_extract_text_rejects_non_md(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "doc.txt").write_text("x", encoding="utf-8")
    reg = _registry(workspace, tmp_path)
    with pytest.raises(SourceAuthoringError, match="source argument validation failed"):
        reg.run(_file_source("md", "extract_text", "doc.txt"))


# ── registry error paths ───────────────────────────────────────────────────

def test_registry_unknown_source_raises(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    reg = _registry(workspace, tmp_path)
    with pytest.raises(SourceAuthoringError, match="unknown source"):
        reg.run(_file_source("xml", "read_file", "out.xml"))


def test_registry_unknown_command_raises(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    reg = _registry(workspace, tmp_path)
    with pytest.raises(SourceAuthoringError, match="unknown source"):
        reg.run(_file_source("json", "bogus_command", "out.json"))


def test_registry_path_must_be_relative(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "out.json").write_text("{}", encoding="utf-8")
    reg = _registry(workspace, tmp_path)
    abs_source = Source(
        type="file",
        file=FileCommand(type="json", command="read_file",
                         arguments={"path": str(workspace / "out.json")}),
    )
    with pytest.raises(SourceAuthoringError, match="must be relative to workspace"):
        reg.run(abs_source)


def test_registry_path_escape_workspace_raises(tmp_path: Path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    reg = _registry(workspace, tmp_path)
    with pytest.raises(SourceAuthoringError, match="must stay inside workspace"):
        reg.run(_file_source("text", "extract_text", "../escape.txt"))


# ── blocked: xlsx/docx/pdf/pptx ─────────────────────────────────────────────

def test_xlsx_source_blocked(tmp_path: Path):
    """xlsx source requires openpyxl, which is not installed.

    BLOCKED per instructions (no openpyxl). NOTE: when openpyxl is absent,
    xlsx.extract_text's run() raises a bare ModuleNotFoundError that is NOT
    caught by SourceRegistry (it is not a SourceAuthoringError nor an
    agent-output failure type) and propagates out of run_verifier as an
    infrastructure error. That uncaught ImportError is recorded as a
    FINDING in the integration report.
    """
    workspace = tmp_path / "ws"; workspace.mkdir()
    # Minimal xlsx bytes so file resolution succeeds and we reach the parser.
    (workspace / "out.xlsx").write_bytes(b"PK\x03\x04")
    reg = _registry(workspace, tmp_path)
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        with pytest.raises(ModuleNotFoundError):
            reg.run(_file_source("xlsx", "extract_text", "out.xlsx"))
    else:
        pytest.skip("openpyxl is installed; not a blocked test")


def test_docx_source_blocked(tmp_path: Path):
    """docx source requires python-docx, which is not installed.

    BLOCKED per instructions (no python-docx). Same uncaught-ModuleNotFoundError
    behavior as xlsx (see test_xlsx_source_blocked).
    """
    workspace = tmp_path / "ws"; workspace.mkdir()
    (workspace / "out.docx").write_bytes(b"PK\x03\x04")
    reg = _registry(workspace, tmp_path)
    try:
        import docx  # noqa: F401
    except ImportError:
        with pytest.raises(ModuleNotFoundError):
            reg.run(_file_source("docx", "extract_text", "out.docx"))
    else:
        pytest.skip("python-docx is installed; not a blocked test")


# ── registry source listing ────────────────────────────────────────────────

def test_registry_known_sources(tmp_path: Path):
    """The registry exposes the canonical dotted source names."""
    workspace = tmp_path / "ws"; workspace.mkdir()
    reg = _registry(workspace, tmp_path)
    names = sorted(reg._sources.keys())
    # spot-check the text/markdown/csv/json namespaces are present
    assert "text.extract_text" in names
    assert "md.extract_text" in names
    assert "csv.extract_text" in names
    assert "json.read_file" in names
    assert "response.extract_final_message" in names

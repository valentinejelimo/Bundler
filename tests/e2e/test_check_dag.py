"""E2E tests for `gdpval_qc.tools.check_dag` — cross-level DAG checks.

check_dag runs ONCE over a tasks/ directory of cirrus-* nodes (the SVC L1->L2->L3
decomposition). It reports:
  - construct_map: expected_value numbers already PRESENT in a node's inputs
    (by-design; L1 -> "extraction-expected", L2/L3 -> "confirm-anchor").
  - filename_map: deliverable filenames that name their content (informational).
  - anchor_consistency: recurring anchors (>=3 tasks) surfaced for divergence.
  - classify_node_inputs: which environment/data inputs are ANOTHER node's golden
    deliverable (by content hash) vs a source document.

We use .txt/.csv/.json golden files (openpyxl/python-docx/pypdf are NOT
installed, so xlsx/docx/pdf coverage is reported as SKIPPED by check_dag —
a full Office DAG test is BLOCKED and marked skip below).
"""
from __future__ import annotations

import importlib
import json
import subprocess

import pytest

check_dag = importlib.import_module("gdpval_qc.tools.check_dag")


def _make_node(tasks, name: str, level: str, *,
               rubric_criteria: list[dict], input_files: dict[str, str],
               golden_files: dict[str, str]):
    """Create a cirrus-* node with tests/rubric.json, environment/data/, solution/golden/."""
    n = tasks / name
    (n / "tests").mkdir(parents=True)
    (n / "environment" / "data").mkdir(parents=True)
    (n / "solution" / "golden").mkdir(parents=True)
    (n / "tests" / "rubric.json").write_text(
        json.dumps({"criteria": rubric_criteria}), encoding="utf-8")
    for fn, content in input_files.items():
        (n / "environment" / "data" / fn).write_text(content, encoding="utf-8")
    for fn, content in golden_files.items():
        (n / "solution" / "golden" / fn).write_text(content, encoding="utf-8")
    return n


# ── construct_map ────────────────────────────────────────────────────────────

def test_construct_map_l1_present_value_is_extraction_expected(tmp_path):
    """An L1 node whose input contains a rubric expected_value number reports a
    construct_map entry with disposition 'extraction-expected'."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_node(
        tasks, "cirrus-l1-01", "L1",
        rubric_criteria=[{"id": "c1", "expected_value": "result is 2186512"}],
        input_files={"source.txt": "The anchor start cash is 2186512 here."},
        golden_files={"output.txt": "2186512"},
    )
    per_task, recurring, skipped = check_dag.check_workbook(str(tasks))
    assert len(per_task) == 1
    entry = per_task[0]
    assert entry["level"] == "L1"
    assert entry["check"] == "construct_map"
    assert entry["disposition"] == "extraction-expected", entry
    assert "2186512" in entry["detail"]
    assert "extraction" in entry["detail"].lower()


def test_construct_map_l2_present_value_is_confirm_anchor(tmp_path):
    """An L2 node with a present expected_value reports disposition 'confirm-anchor'."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_node(
        tasks, "cirrus-l2-02", "L2",
        rubric_criteria=[{"id": "c1", "expected_value": "anchor 2186512"}],
        input_files={"raw.csv": "cash,2186512\n"},
        golden_files={"output.txt": "2186512"},
    )
    per_task, recurring, skipped = check_dag.check_workbook(str(tasks))
    assert len(per_task) == 1
    entry = per_task[0]
    assert entry["level"] == "L2"
    assert entry["disposition"] == "confirm-anchor", entry


def test_construct_map_absent_value_not_reported(tmp_path):
    """An expected_value NOT present in the node's inputs produces no construct_map
    entry (the value must be derived, not extracted)."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_node(
        tasks, "cirrus-l1-03", "L1",
        rubric_criteria=[{"id": "c1", "expected_value": "drawdown is 9999999"}],
        input_files={"source.txt": "unrelated content with 1234567 only."},
        golden_files={"output.txt": "9999999"},
    )
    per_task, recurring, skipped = check_dag.check_workbook(str(tasks))
    # 9999999 is in expected_value but NOT in input (input has 1234567) -> no entry.
    construct = [e for e in per_task if e["check"] == "construct_map"]
    assert construct == [], f"no construct_map expected when value absent: {construct}"


# ── anchor_consistency / recurring ──────────────────────────────────────────

def test_recurring_anchor_appears_in_three_tasks(tmp_path):
    """A value appearing in expected_value of >=3 tasks is listed in recurring."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    for i, lvl in enumerate(["l1", "l2", "l3"], start=1):
        _make_node(
            tasks, f"cirrus-{lvl}-0{i}", f"L{i}",
            rubric_criteria=[{"id": "c1", "expected_value": "start cash 2186512"}],
            input_files={"in.txt": f"anchor 2186512 node{i}"},
            golden_files={"out.txt": "2186512"},
        )
    per_task, recurring, skipped = check_dag.check_workbook(str(tasks))
    assert "2186512" in recurring, f"2186512 should be recurring: {recurring}"
    # recurring value maps to >=3 distinct tasks.
    assert len(recurring["2186512"]) >= 3


def test_non_recurring_value_not_in_recurring(tmp_path):
    """A value in only 1-2 tasks' expected_values is NOT in recurring (needs >=3)."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_node(tasks, "cirrus-l1-01", "L1",
               rubric_criteria=[{"id": "c1", "expected_value": "v 5551111"}],
               input_files={"in.txt": "x 5551111"}, golden_files={"o.txt": "5551111"})
    _make_node(tasks, "cirrus-l2-02", "L2",
               rubric_criteria=[{"id": "c1", "expected_value": "v 5551111"}],
               input_files={"in.txt": "x 5551111"}, golden_files={"o.txt": "5551111"})
    per_task, recurring, skipped = check_dag.check_workbook(str(tasks))
    assert "5551111" not in recurring, f"only-2-tasks value must not be recurring: {recurring}"


# ── filename_map ─────────────────────────────────────────────────────────────

def test_filename_map_flags_answer_token_in_golden_name(tmp_path):
    """A golden deliverable filename containing an ANSWER_TOKEN (e.g. 'answer')
    is reported in a filename_map informational entry."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_node(tasks, "cirrus-l1-01", "L1",
               rubric_criteria=[{"id": "c1", "expected_value": "x 2186512"}],
               input_files={"in.txt": "2186512"},
               golden_files={"the_answer_file.txt": "2186512"})
    per_task, recurring, skipped = check_dag.check_workbook(str(tasks))
    fn = [e for e in per_task if e["check"] == "filename_map"]
    assert len(fn) == 1, f"expected filename_map entry: {per_task}"
    assert fn[0]["disposition"] == "informational"
    assert "answer" in fn[0]["detail"].lower()


# ── classify_node_inputs ────────────────────────────────────────────────────

def test_classify_node_inputs_detects_upstream_deliverable_by_hash(tmp_path):
    """If node B's environment/data contains a file identical (by sha256) to node
    A's solution/golden deliverable, classify_node_inputs reports it as an
    upstream-deliverable input sourced from node A."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    golden_content = "GOLDEN_FROM_A_UNIQUE_CONTENT_12345"
    # Node A produces a golden.
    _make_node(tasks, "cirrus-l1-01", "L1",
               rubric_criteria=[{"id": "c1", "expected_value": "x 2186512"}],
               input_files={"source.txt": "raw 2186512"},
               golden_files={"result.txt": golden_content})
    # Node B consumes A's golden as an input (same content), plus a real source.
    _make_node(tasks, "cirrus-l2-02", "L2",
               rubric_criteria=[{"id": "c1", "expected_value": "x 2186512"}],
               input_files={"upstream_result.txt": golden_content,  # <- A's golden
                            "raw_source.csv": "data,1\ndata,2\n"},
               golden_files={"out.txt": "derived"})
    rows = check_dag.classify_node_inputs(str(tasks))
    by_nid = {nid: (deliv, nsrc) for nid, deliv, nsrc in rows}
    assert "cirrus-l2-02" in by_nid
    deliv_b, nsrc_b = by_nid["cirrus-l2-02"]
    # upstream_result.txt should be attributed to cirrus-l1-01.
    assert any(fn == "upstream_result.txt" and prod == "cirrus-l1-01"
               for fn, prod in deliv_b), f"expected upstream deliverable: {deliv_b}"
    # raw_source.csv is a source doc (no hash match) -> counted in nsrc.
    assert nsrc_b >= 1, f"expected >=1 source doc for B: nsrc={nsrc_b}, deliv={deliv_b}"


def test_classify_node_inputs_no_upstream_when_all_unique(tmp_path):
    """When every node's inputs are unique (no content-hash matches against any
    golden), classify_node_inputs reports 0 upstream deliverables."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_node(tasks, "cirrus-l1-01", "L1",
               rubric_criteria=[{"id": "c1", "expected_value": "x 2186512"}],
               input_files={"a.txt": "UNIQUE_A_CONTENT"},
               golden_files={"g.txt": "UNIQUE_GOLDEN_A"})
    _make_node(tasks, "cirrus-l2-02", "L2",
               rubric_criteria=[{"id": "c1", "expected_value": "x 2186512"}],
               input_files={"b.txt": "UNIQUE_B_CONTENT"},
               golden_files={"g2.txt": "UNIQUE_GOLDEN_B"})
    rows = check_dag.classify_node_inputs(str(tasks))
    tot_deliv = sum(len(d) for _, d, _ in rows)
    assert tot_deliv == 0, f"expected 0 upstream deliverables: {rows}"


# ── CLI invocation ───────────────────────────────────────────────────────────

def test_check_dag_cli_runs_and_exits_zero(tmp_path, venv_py, project_root):
    """`python -m gdpval_qc.tools.check_dag <tasks_dir>` runs end-to-end via CLI."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _make_node(tasks, "cirrus-l1-01", "L1",
               rubric_criteria=[{"id": "c1", "expected_value": "x 2186512"}],
               input_files={"in.txt": "2186512"}, golden_files={"out.txt": "2186512"})
    res = subprocess.run(
        [str(venv_py), "-m", "gdpval_qc.tools.check_dag", str(tasks),
         "--json", str(tmp_path / "report.json")],
        cwd=str(project_root), capture_output=True, text=True,
    )
    assert res.returncode == 0, f"check_dag CLI exited {res.returncode}\n{res.stdout}\n{res.stderr}"
    assert "SVC DAG checks" in res.stdout
    assert "construct_map" in res.stdout or "construct" in res.stdout.lower()
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "per_task" in report
    assert "recurring_anchors" in report


# ── Office-format coverage (openpyxl/python-docx/pypdf not installed) ───────

def test_check_dag_reports_skipped_coverage_for_office_files(tmp_path):
    """check_dag reports xlsx/docx/pdf files as SKIPPED coverage when the Office
    libs are missing (they are in this env). This is observable behavior, not a
    block: the file is listed in `skipped` rather than read."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    n = tasks / "cirrus-l1-01"
    (n / "tests").mkdir(parents=True)
    (n / "environment" / "data").mkdir(parents=True)
    (n / "solution" / "golden").mkdir(parents=True)
    (n / "tests" / "rubric.json").write_text(
        json.dumps({"criteria": [{"id": "c1", "expected_value": "x 2186512"}]}),
        encoding="utf-8")
    # Office files: just bytes; check_dag's file_text tries to import the lib and
    # records an ImportError in `skipped`.
    (n / "environment" / "data" / "sheet.xlsx").write_bytes(b"fake xlsx bytes")
    (n / "environment" / "data" / "doc.docx").write_bytes(b"fake docx bytes")
    (n / "environment" / "data" / "text.txt").write_text("anchor 2186512", encoding="utf-8")
    (n / "solution" / "golden" / "out.txt").write_text("2186512", encoding="utf-8")

    per_task, recurring, skipped = check_dag.check_workbook(str(tasks))
    # The .txt file is read normally -> construct_map entry still produced.
    construct = [e for e in per_task if e["check"] == "construct_map"]
    assert len(construct) == 1, f"txt input should still be read: {per_task}"
    # The Office files should be reported as skipped coverage (missing libs).
    skipped_text = " ".join(skipped)
    assert ".xlsx" in skipped_text or "openpyxl" in skipped_text, \
        f"xlsx skip not reported: {skipped}"
    assert ".docx" in skipped_text or "docx" in skipped_text, \
        f"docx skip not reported: {skipped}"


@pytest.mark.skip(reason="BLOCKED: openpyxl not installed; a full xlsx-content DAG test cannot run")
def test_check_dag_xlsx_content_extraction_blocked():
    """BLOCKED — would assert check_dag extracts >=4-digit numbers from xlsx
    cell values once openpyxl is installed. Cannot run in this environment."""

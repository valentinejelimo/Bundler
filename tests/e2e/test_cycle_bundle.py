"""E2E tests for `cycle.py bundle` — the core task-bundling workflow.

Validates the EXISTING intended behavior documented in README's "How a task is
shaped" and "Step 1 - Bundle tasks" sections, by spawning `cycle.py bundle` as
a subprocess against a temp cycle folder under <project>/cycles/.

Happy path + error paths + shipped-verifier override + --only filter + no-source
task. We DO NOT weaken assertions to make tests pass; discrepancies between
documented and actual behavior are recorded as findings.
"""
from __future__ import annotations

import json
import os
import stat


# A rubric with 6 verifiers (>= project minimum), weights summing to 1.0, two
# verifiers reading output/ paths so _deliverables resolves to those filenames.
def _good_rubric() -> dict:
    return {
        "verifiers": [
            {"name": "v1", "weight": 0.2,
             "source": {"file": {"arguments": {"path": "output/result.txt"}}}},
            {"name": "v2", "weight": 0.2, "source": {"file": {"arguments": {"path": "output/other.txt"}}}},
            {"name": "v3", "weight": 0.15},
            {"name": "v4", "weight": 0.15},
            {"name": "v5", "weight": 0.15},
            {"name": "v6", "weight": 0.15},
        ]
    }


# ── Happy path ──────────────────────────────────────────────────────────────

def test_bundle_happy_path_assembles_full_task_structure(cycle_date, make_fetched_task, run_cycle, cycles_dir):
    """A well-formed task is assembled into the documented harbor layout."""
    rubric = _good_rubric()
    make_fetched_task(
        cycle_date, "97",
        prompt="Analyze the quarterly report and write result.txt.",
        rubric_json=json.dumps(rubric),
        row_extra={"Sector": "Finance", "Subdomain": "Audit", "Task Type": "analysis"},
        source_files={"data.txt": "Q1 revenue 5000000"},
        solution_files={"result.txt": "golden result", "other.txt": "other golden"},
    )

    res = run_cycle(["bundle", cycle_date])

    out = cycles_dir / cycle_date / "02_bundle"
    task = out / "task_97"

    # --- exit code & stdout ---
    assert res.returncode == 0, f"bundle exited {res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "assembled 1 harbor tasks" in res.stdout, f"unexpected stdout:\n{res.stdout}"

    # --- documented file structure (README "Bundled task") ---
    assert (task / "instruction.md").is_file(), "instruction.md missing"
    assert (task / "environment" / "data" / "data.txt").is_file(), "source file not copied to environment/data/"
    assert (task / "environment" / "Dockerfile").is_file(), "Dockerfile not generated"
    assert (task / "solution" / "golden_outputs" / "result.txt").is_file(), "golden output not copied"
    assert (task / "solution" / "golden_outputs" / "other.txt").is_file(), "second golden output not copied"
    assert (task / "solution" / "solve.sh").is_file(), "solve.sh oracle missing"
    assert (task / "tests" / "verifier.json").is_file(), "verifier.json missing"
    assert (task / "tests" / "test.sh").is_file(), "generic test.sh missing"
    assert (task / "tests" / "verifier_runtime.py").is_file(), "verifier_runtime.py missing"
    assert (task / "tests" / "verifier_lib").is_dir(), "verifier_lib/ missing"
    assert (task / "tests" / "verifier_lib" / "runner.py").is_file(), "verifier_lib/runner.py missing"
    assert (task / "task.toml").is_file(), "task.toml missing"

    # --- content: instruction.md == prompt ---
    assert (task / "instruction.md").read_text(encoding="utf-8") == \
        "Analyze the quarterly report and write result.txt."

    # --- content: tests/verifier.json == rubric (parsed-equal, not byte-equal) ---
    written = json.loads((task / "tests" / "verifier.json").read_text(encoding="utf-8"))
    assert written == rubric, "verifier.json does not match the _row.json rubric"

    # --- content: solve.sh lists the graded deliverables (both have output/ paths) ---
    solve = (task / "solution" / "solve.sh").read_text(encoding="utf-8")
    assert '"result.txt"' in solve, f"solve.sh should list result.txt deliverable:\n{solve}"
    assert '"other.txt"' in solve, f"solve.sh should list other.txt deliverable:\n{solve}"
    # solve.sh must be marked executable (chmod 0o755). On POSIX this is the
    # exec bit; on Windows NTFS the exec bit is not honored by os.chmod, so we
    # only assert it where the platform supports it. cycle.py calls
    # os.chmod(solve, 0o755) unconditionally — that is the documented behavior.
    if os.name != "nt":
        mode = (task / "solution" / "solve.sh").stat().st_mode
        assert mode & stat.S_IXUSR, f"solve.sh not marked executable (mode={oct(mode)})"

    # --- content: task.toml metadata ---
    toml_text = (task / "task.toml").read_text(encoding="utf-8")
    assert 'task_id = "task_97"' in toml_text
    assert 'category = "Finance"' in toml_text
    assert 'subdomain = "Audit"' in toml_text
    assert 'task_type = "analysis"' in toml_text

    # --- _INCIDENTS.md written ---
    inc = out / "_INCIDENTS.md"
    assert inc.is_file(), "_INCIDENTS.md not written"
    inc_text = inc.read_text(encoding="utf-8")
    assert "Pipeline run report" in inc_text
    assert "Tasks assembled (with `verifier.json`): **1**" in inc_text


def test_bundle_happy_path_stdout_lists_next_step(cycle_date, make_fetched_task, run_cycle):
    """On a successful assembly, bundle prints the `qc` next-step hint."""
    make_fetched_task(
        cycle_date, "11",
        rubric_json=json.dumps(_good_rubric()),
        source_files={"data.txt": "x"},
        solution_files={"result.txt": "g"},
    )
    res = run_cycle(["bundle", cycle_date])
    assert res.returncode == 0
    assert "Next: run the review with" in res.stdout
    assert f"qc {cycle_date}" in res.stdout


# ── Error path: malformed rubric ────────────────────────────────────────────

def test_bundle_malformed_rubric_records_error_and_skips_task(cycle_date, make_fetched_task, run_cycle, cycles_dir):
    """A task whose Rubric JSON cell is malformed JSON produces a rubric-malformed
    ERROR incident and is NOT counted as assembled. The verifier.json must NOT
    be written for that task."""
    make_fetched_task(
        cycle_date, "42",
        prompt="Do something.",
        rubric_json="not valid json {{{ broken",
        source_files={"data.txt": "input"},
        solution_files={"result.txt": "golden"},
    )

    res = run_cycle(["bundle", cycle_date])
    assert res.returncode == 0, f"bundle should exit 0 even on data errors; got {res.returncode}"

    out = cycles_dir / cycle_date / "02_bundle"
    task = out / "task_42"

    # The task must NOT be counted as assembled.
    assert "assembled 0 harbor tasks" in res.stdout, f"expected 0 assembled:\n{res.stdout}"
    # The ERROR incident must be reported on stdout.
    assert "1 ERROR incident(s)" in res.stdout
    assert "task_42:rubric-malformed" in res.stdout, f"expected rubric-malformed ERROR:\n{res.stdout}"

    # verifier.json must NOT exist (no rubric -> no verifier).
    assert not (task / "tests" / "verifier.json").is_file(), \
        "verifier.json must not be written for a malformed-rubric task"

    # _INCIDENTS.md must record the ERROR.
    inc = (out / "_INCIDENTS.md").read_text(encoding="utf-8")
    assert "rubric-malformed" in inc
    assert "task_42" in inc


# ── Error path: no source files ─────────────────────────────────────────────

def test_bundle_no_source_files_task_skipped_entirely(cycle_date, run_cycle, cycles_dir):
    """A task with no source/ directory (or empty source/) is skipped entirely:
    no task_<id>/ directory is created, and a no-source-files ERROR incident is
    recorded."""
    d = cycles_dir / cycle_date / "00_fetched" / "7"
    d.mkdir(parents=True)
    (d / "_row.json").write_text(json.dumps({"Prompt": "p", "Rubric JSON": json.dumps(_good_rubric())}))
    # NOTE: no source/ subdir, no solution/ subdir

    res = run_cycle(["bundle", cycle_date])
    assert res.returncode == 0
    out = cycles_dir / cycle_date / "02_bundle"
    assert "assembled 0 harbor tasks" in res.stdout
    assert "task_7:no-source-files" in res.stdout
    # No task dir created.
    assert not (out / "task_7").is_dir(), "task_7/ dir must not be created when source is missing"
    inc = (out / "_INCIDENTS.md").read_text(encoding="utf-8")
    assert "no-source-files" in inc


# ── Shipped verifier.json override ──────────────────────────────────────────

def test_bundle_shipped_verifier_json_overrides_rubric_field(cycle_date, make_fetched_task, run_cycle, cycles_dir):
    """A shipped verifier.json next to source/solution wins over the Rubric JSON
    field in _row.json (the hand-repaired version), and an INFO incident is
    recorded."""
    shipped = {"verifiers": [{"name": "hand", "weight": 1.0}]}
    make_fetched_task(
        cycle_date, "5",
        rubric_json=json.dumps(_good_rubric()),  # would normally be used
        source_files={"data.txt": "input"},
        solution_files={"result.txt": "golden"},
        shipped_verifier=json.dumps(shipped),
    )

    res = run_cycle(["bundle", cycle_date])
    assert res.returncode == 0
    assert "assembled 1 harbor tasks" in res.stdout

    task = cycles_dir / cycle_date / "02_bundle" / "task_5"
    written = json.loads((task / "tests" / "verifier.json").read_text(encoding="utf-8"))
    assert written == shipped, "shipped verifier.json should override the _row.json rubric field"
    # INFO incident for using the shipped file
    inc = (cycles_dir / cycle_date / "02_bundle" / "_INCIDENTS.md").read_text(encoding="utf-8")
    assert "verifier-shipped" in inc


# ── --only filter ───────────────────────────────────────────────────────────

def test_bundle_only_filter_restricts_to_listed_ids(cycle_date, make_fetched_task, run_cycle, cycles_dir):
    """--only 97,160 restricts bundling to those task ids; other tasks are left
    alone (no task dir created for them)."""
    rubric = json.dumps(_good_rubric())
    make_fetched_task(cycle_date, "97", rubric_json=rubric,
                      source_files={"data.txt": "a"}, solution_files={"result.txt": "g"})
    make_fetched_task(cycle_date, "160", rubric_json=rubric,
                      source_files={"data.txt": "b"}, solution_files={"result.txt": "g"})
    make_fetched_task(cycle_date, "200", rubric_json=rubric,
                      source_files={"data.txt": "c"}, solution_files={"result.txt": "g"})

    res = run_cycle(["bundle", cycle_date, "--only", "97,160"])
    assert res.returncode == 0
    assert "assembled 2 harbor tasks" in res.stdout, f"expected 2 assembled:\n{res.stdout}"

    out = cycles_dir / cycle_date / "02_bundle"
    assert (out / "task_97").is_dir()
    assert (out / "task_160").is_dir()
    assert not (out / "task_200").is_dir(), "task_200 must NOT be assembled when not in --only"


# ── --out custom output subdir ──────────────────────────────────────────────

def test_bundle_custom_out_subdir(cycle_date, make_fetched_task, run_cycle, cycles_dir):
    """--out names a custom output subdir under the cycle (default 02_bundle)."""
    make_fetched_task(cycle_date, "3", rubric_json=json.dumps(_good_rubric()),
                      source_files={"data.txt": "a"}, solution_files={"result.txt": "g"})
    res = run_cycle(["bundle", cycle_date, "--out", "custom_bundle"])
    assert res.returncode == 0
    out = cycles_dir / cycle_date / "custom_bundle"
    assert (out / "task_3").is_dir(), "custom --out subdir not used"
    assert (out / "_INCIDENTS.md").is_file()


# ── rubric-recovered (trailing junk) ────────────────────────────────────────

def test_bundle_rubric_with_trailing_junk_is_recovered(cycle_date, make_fetched_task, run_cycle, cycles_dir):
    """A Rubric JSON cell with valid JSON followed by trailing junk (e.g. a stray
    </verifier> tag) is auto-recovered via raw_decode and a WARN incident
    'rubric-recovered' is recorded; the task IS assembled."""
    good = json.dumps(_good_rubric())
    rubric_with_junk = good + "</verifier>"  # trailing junk
    make_fetched_task(cycle_date, "8", rubric_json=rubric_with_junk,
                      source_files={"data.txt": "a"}, solution_files={"result.txt": "g"})

    res = run_cycle(["bundle", cycle_date])
    assert res.returncode == 0
    assert "assembled 1 harbor tasks" in res.stdout, f"expected recovery + assembly:\n{res.stdout}"
    # WARN incident rubric-recovered
    inc_text = (cycles_dir / cycle_date / "02_bundle" / "_INCIDENTS.md").read_text(encoding="utf-8")
    assert "rubric-recovered" in res.stdout or "rubric-recovered" in inc_text
    task = cycles_dir / cycle_date / "02_bundle" / "task_8"
    assert (task / "tests" / "verifier.json").is_file()
    written = json.loads((task / "tests" / "verifier.json").read_text(encoding="utf-8"))
    assert written == _good_rubric(), "recovered verifier.json should equal the valid JSON prefix"

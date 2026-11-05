"""E2E tests for `cycle.py report` — collects human-facing artifacts from
cycles/<date>/04_qc/ into cycles/<date>/05_report/.

Read README "Step 3 - Collect the human-facing report" and cmd_report in
cycle.py for expected behavior. cmd_report copies a fixed list of artifact
filenames (batch_summary.md, reviewer_queue.csv, dashboard.html, REPORT.md,
batch_review.json) from 04_qc/ into 05_report/, printing which were collected.
"""


def _put(cycles_dir, cycle_date: str, name: str, content: bytes) -> None:
    src = cycles_dir / cycle_date / "04_qc"
    src.mkdir(parents=True, exist_ok=True)
    (src / name).write_bytes(content)


def test_report_copies_known_artifacts_from_04qc(cycle_date, run_cycle, cycles_dir):
    """Each known artifact file present in 04_qc/ is byte-copied to 05_report/."""
    _put(cycles_dir, cycle_date, "batch_summary.md", b"# Verdict mix\naccept: 3\n")
    _put(cycles_dir, cycle_date, "reviewer_queue.csv", b"sample,issue,severity\ntask_1,missing,2\n")
    _put(cycles_dir, cycle_date, "batch_review.json", b'{"samples": []}')
    _put(cycles_dir, cycle_date, "dashboard.html", b"<html>dash</html>")
    _put(cycles_dir, cycle_date, "REPORT.md", b"# Report body")

    res = run_cycle(["report", cycle_date])
    assert res.returncode == 0, f"report exited {res.returncode}\n{res.stdout}\n{res.stderr}"

    dst = cycles_dir / cycle_date / "05_report"
    for name in ["batch_summary.md", "reviewer_queue.csv", "batch_review.json",
                 "dashboard.html", "REPORT.md"]:
        f = dst / name
        assert f.is_file(), f"{name} not copied to 05_report/"
    # Byte-equal content preserved (cmd_report uses read_bytes/write_bytes).
    assert (dst / "batch_summary.md").read_bytes() == b"# Verdict mix\naccept: 3\n"
    assert (dst / "batch_review.json").read_bytes() == b'{"samples": []}'

    # stdout lists the collected filenames.
    for name in ["batch_summary.md", "reviewer_queue.csv", "batch_review.json",
                 "dashboard.html", "REPORT.md"]:
        assert name in res.stdout, f"{name!r} should appear in collected list:\n{res.stdout}"


def test_report_handles_empty_04qc(cycle_date, run_cycle, cycles_dir):
    """If 04_qc/ has none of the known artifacts, report still succeeds and
    prints a '(nothing - run qc first)' message."""
    src = cycles_dir / cycle_date / "04_qc"
    src.mkdir(parents=True, exist_ok=True)
    # An unrelated file that is NOT in the known list must be ignored.
    (src / "unrelated.txt").write_bytes(b"noise")

    res = run_cycle(["report", cycle_date])
    assert res.returncode == 0, f"report should succeed with no artifacts:\n{res.stdout}\n{res.stderr}"
    assert "(nothing - run qc first)" in res.stdout, f"expected nothing-collected msg:\n{res.stdout}"
    dst = cycles_dir / cycle_date / "05_report"
    assert dst.is_dir(), "05_report/ should be created even when nothing collected"
    assert not (dst / "unrelated.txt").is_file(), "report must NOT copy unknown filenames"


def test_report_creates_05_report_dir_if_absent(cycle_date, run_cycle, cycles_dir):
    """report creates 05_report/ (mkdir exist_ok) even if it didn't exist."""
    _put(cycles_dir, cycle_date, "batch_summary.md", b"summary")
    dst = cycles_dir / cycle_date / "05_report"
    assert not dst.exists()
    res = run_cycle(["report", cycle_date])
    assert res.returncode == 0
    assert dst.is_dir(), "05_report/ was not created"
    assert (dst / "batch_summary.md").read_bytes() == b"summary"


def test_report_finishes_with_done_message(cycle_date, run_cycle, cycles_dir):
    """report prints the final 'Done. Review the artifacts in 05_report/.' line."""
    _put(cycles_dir, cycle_date, "batch_summary.md", b"summary")
    res = run_cycle(["report", cycle_date])
    assert res.returncode == 0
    assert "Done. Review the artifacts in 05_report/." in res.stdout

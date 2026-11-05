"""E2E tests for `cycle.py qc` and `cycle.py svc-qc` subprocess construction.

`cmd_qc` and `cmd_svc_qc` build a command list and shell out to the venv python
(``-m gdpval_qc.cli``). We test that the constructed command is correct WITHOUT
actually running the engine (which needs API keys). We mock ``subprocess.call``
(qc) and ``subprocess.Popen`` (svc-qc) to capture the args list passed and
assert it includes the right -m, --spec, --corpus, --output-dir, and default
flags (--skip-calibration, --no-conversion for svc).

We also assert that passthrough ``--`` args reach the subprocess, and that the
venv-existence check is satisfied (the real .venv exists in this project).
"""
from __future__ import annotations

import argparse
import shutil

import cycle  # type: ignore[import-not-found]  # imported from project root
import pytest


def _parse_args(argv: list[str]):
    """Replicate cycle.py's argparse so we can build a Namespace with the same
    defaults, then attach the real cmd_qc/cmd_svc_qc as fn."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    # qc
    q = sub.add_parser("qc")
    q.add_argument("date")
    q.add_argument("--spec", default=str(cycle.SPEC_DEFAULT))
    q.add_argument("--sample-parallelism", type=int, default=5, dest="sample_parallelism")
    q.add_argument("--dim-parallelism", type=int, default=4, dest="dim_parallelism")
    q.add_argument("--calibrate", action="store_true")
    q.add_argument("extra", nargs="*")
    # svc-qc
    sv = sub.add_parser("svc-qc")
    sv.add_argument("corpus")
    sv.add_argument("--only", default="")
    sv.add_argument("--out-dir", default="", dest="out_dir")
    sv.add_argument("--date", default="")
    sv.add_argument("--spec", default=str(cycle.SPEC_SVC))
    sv.add_argument("--sample-parallelism", type=int, default=3, dest="sample_parallelism")
    sv.add_argument("--with-conversion", action="store_true")
    sv.add_argument("extra", nargs="*")
    args = p.parse_args(argv)
    return args


def _seed_evals(cycles_dir, cycle_date: str):
    e = cycles_dir / cycle_date / "03_evals"
    e.mkdir(parents=True, exist_ok=True)
    (e / "dummy_eval.json").write_text("{}", encoding="utf-8")
    return e


# ── qc command construction ─────────────────────────────────────────────────

def test_qc_builds_command_with_spec_corpus_output_and_skip_calibration(cycle_date, monkeypatch, cycles_dir, venv_py):
    """cmd_qc constructs [VENV_PY, -m gdpval_qc.cli, --spec, --corpus, --output-dir,
    --sample-parallelism, --dim-parallelism, --skip-calibration] and calls
    subprocess.call with it."""
    _seed_evals(cycles_dir, cycle_date)
    captured = {}

    def fake_call(cmd, *a, **kw):
        captured["cmd"] = list(cmd)
        return 0

    monkeypatch.setattr(cycle.subprocess, "call", fake_call)
    monkeypatch.setattr(cycle.sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    args = _parse_args(["qc", cycle_date])
    args.fn = cycle.cmd_qc

    with pytest.raises(SystemExit):
        cycle.cmd_qc(args)

    cmd = captured["cmd"]
    assert cmd[0] == str(venv_py), f"first arg should be venv python: {cmd[0]}"
    assert "-m" in cmd and "gdpval_qc.cli" in cmd, f"missing -m gdpval_qc.cli: {cmd}"
    assert "--spec" in cmd
    spec_idx = cmd.index("--spec")
    assert cmd[spec_idx + 1] == str(cycle.SPEC_DEFAULT)
    assert "--corpus" in cmd
    corpus_idx = cmd.index("--corpus")
    assert cmd[corpus_idx + 1] == str(cycles_dir / cycle_date / "03_evals")
    assert "--output-dir" in cmd
    out_idx = cmd.index("--output-dir")
    assert cmd[out_idx + 1] == str(cycles_dir / cycle_date / "04_qc")
    # Default parallelism injected.
    assert "--sample-parallelism" in cmd
    sp_idx = cmd.index("--sample-parallelism")
    assert cmd[sp_idx + 1] == "5", f"qc default sample-parallelism should be 5: {cmd[sp_idx+1]}"
    assert "--dim-parallelism" in cmd
    dp_idx = cmd.index("--dim-parallelism")
    assert cmd[dp_idx + 1] == "4"
    # skip-calibration injected by default (calibrate off).
    assert "--skip-calibration" in cmd, f"--skip-calibration not in default cmd: {cmd}"


def test_qc_with_calibrate_omits_skip_calibration(cycle_date, monkeypatch, cycles_dir):
    """--calibrate must NOT inject --skip-calibration (calibration is opt-in)."""
    _seed_evals(cycles_dir, cycle_date)
    captured = {}

    def fake_call(cmd, *a, **kw):
        captured["cmd"] = list(cmd)
        return 0

    monkeypatch.setattr(cycle.subprocess, "call", fake_call)
    monkeypatch.setattr(cycle.sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    args = _parse_args(["qc", cycle_date, "--calibrate"])
    args.fn = cycle.cmd_qc

    with pytest.raises(SystemExit):
        cycle.cmd_qc(args)

    cmd = captured["cmd"]
    assert "--skip-calibration" not in cmd, \
        f"--calibrate should suppress --skip-calibration: {cmd}"


def test_qc_passthrough_extra_args_reach_subprocess(cycle_date, monkeypatch, cycles_dir):
    """Args after the `--` separator are passed through to the engine.

    README documents: 'Any args after `--` are passed through to the engine
    (e.g. `-- --no-multimodal` to force text-only judging).'
    """
    _seed_evals(cycles_dir, cycle_date)
    captured = {}

    def fake_call(cmd, *a, **kw):
        captured["cmd"] = list(cmd)
        return 0

    monkeypatch.setattr(cycle.subprocess, "call", fake_call)
    monkeypatch.setattr(cycle.sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    # The `--` separator makes argparse treat everything after it as positional
    # `extra` args (nargs="*"), so unknown flags like --no-multimodal are captured.
    args = _parse_args(["qc", cycle_date, "--", "--no-multimodal", "--limit", "2"])
    args.fn = cycle.cmd_qc

    with pytest.raises(SystemExit):
        cycle.cmd_qc(args)

    cmd = captured["cmd"]
    assert "--no-multimodal" in cmd, f"passthrough --no-multimodal missing: {cmd}"
    assert "--limit" in cmd and cmd[cmd.index("--limit") + 1] == "2", f"passthrough --limit missing: {cmd}"


def test_qc_refuses_to_run_with_empty_evals(cycle_date, monkeypatch):
    """qc exits with an error message if 03_evals/ is missing or empty."""
    # Don't seed evals -> empty/missing
    monkeypatch.setattr(cycle.sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
    args = _parse_args(["qc", cycle_date])
    args.fn = cycle.cmd_qc

    with pytest.raises(SystemExit) as exc:
        cycle.cmd_qc(args)
    assert exc.value.code and "no evals" in str(exc.value.code), \
        f"expected 'no evals' error, got: {exc.value.code!r}"


def test_qc_passthrough_sample_parallelism_not_doubled(cycle_date, monkeypatch, cycles_dir):
    """If the user passes --sample-parallelism via extra, qc must NOT inject its
    own default (the `not any(a.startswith(...))` guard)."""
    _seed_evals(cycles_dir, cycle_date)
    captured = {}

    def fake_call(cmd, *a, **kw):
        captured["cmd"] = list(cmd)
        return 0

    monkeypatch.setattr(cycle.subprocess, "call", fake_call)
    monkeypatch.setattr(cycle.sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    args = _parse_args(["qc", cycle_date, "--sample-parallelism", "1"])
    args.fn = cycle.cmd_qc

    with pytest.raises(SystemExit):
        cycle.cmd_qc(args)

    cmd = captured["cmd"]
    # Count occurrences of --sample-parallelism: should be exactly 1 (the user's).
    assert cmd.count("--sample-parallelism") == 1, \
        f"qc should not inject a second --sample-parallelism: {cmd}"
    assert cmd[cmd.index("--sample-parallelism") + 1] == "1"


# ── svc-qc command construction ─────────────────────────────────────────────

def _patch_svc(monkeypatch, captured, rc=0):
    """Mock subprocess.Popen for svc-qc and sys.exit."""
    class _FakeProc:
        stdout = iter([])  # no lines
        def wait(self):
            return rc

    def fake_popen(cmd, *a, **kw):
        captured["cmd"] = list(cmd)
        return _FakeProc()

    monkeypatch.setattr(cycle.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cycle.sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))


def test_svc_qc_builds_command_with_spec_corpus_output(tmp_path, monkeypatch, venv_py):
    """svc-qc constructs [VENV_PY, -m gdpval_qc.cli, --spec, --corpus,
    --output-dir, --skip-calibration, --no-conversion, --sample-parallelism]."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "task_1").mkdir()
    out_dir = tmp_path / "svc_out"

    captured = {}
    _patch_svc(monkeypatch, captured)

    args = _parse_args(["svc-qc", str(corpus), "--out-dir", str(out_dir)])
    args.fn = cycle.cmd_svc_qc

    with pytest.raises(SystemExit):
        cycle.cmd_svc_qc(args)

    cmd = captured["cmd"]
    assert cmd[0] == str(venv_py)
    assert "-m" in cmd and "gdpval_qc.cli" in cmd
    assert "--spec" in cmd
    spec_idx = cmd.index("--spec")
    assert cmd[spec_idx + 1] == str(cycle.SPEC_SVC), \
        f"svc-qc default spec should be svc.md: {cmd[spec_idx+1]}"
    assert "--corpus" in cmd
    corpus_idx = cmd.index("--corpus")
    assert cmd[corpus_idx + 1] == str(corpus.resolve())
    out_idx = cmd.index("--output-dir")
    assert cmd[out_idx + 1] == str(out_dir.resolve())
    # Default flags.
    assert "--skip-calibration" in cmd, f"--skip-calibration default missing: {cmd}"
    assert "--no-conversion" in cmd, f"--no-conversion default missing: {cmd}"
    assert "--sample-parallelism" in cmd
    sp_idx = cmd.index("--sample-parallelism")
    assert cmd[sp_idx + 1] == "3", f"svc-qc default sample-parallelism should be 3: {cmd[sp_idx+1]}"


def test_svc_qc_with_conversion_omits_no_conversion(tmp_path, monkeypatch):
    """--with-conversion must NOT inject --no-conversion."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    captured = {}
    _patch_svc(monkeypatch, captured)

    args = _parse_args(["svc-qc", str(corpus), "--with-conversion"])
    args.fn = cycle.cmd_svc_qc

    with pytest.raises(SystemExit):
        cycle.cmd_svc_qc(args)

    cmd = captured["cmd"]
    assert "--no-conversion" not in cmd, \
        f"--with-conversion should suppress --no-conversion: {cmd}"


def test_svc_qc_only_passes_tasks_flag(tmp_path, monkeypatch):
    """--only ids is passed to the engine as --tasks <ids>."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    captured = {}
    _patch_svc(monkeypatch, captured)

    args = _parse_args(["svc-qc", str(corpus), "--only", "task_1,task_2"])
    args.fn = cycle.cmd_svc_qc

    with pytest.raises(SystemExit):
        cycle.cmd_svc_qc(args)

    cmd = captured["cmd"]
    assert "--tasks" in cmd, f"--only should produce --tasks: {cmd}"
    assert cmd[cmd.index("--tasks") + 1] == "task_1,task_2"


def test_svc_qc_passthrough_extra_args(tmp_path, monkeypatch):
    """Args after the `--` separator reach the engine subprocess.

    README: 'Any args after `--` are passed through to the engine.'
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    captured = {}
    _patch_svc(monkeypatch, captured)

    args = _parse_args(["svc-qc", str(corpus), "--", "--quiet", "--limit", "1"])
    args.fn = cycle.cmd_svc_qc

    with pytest.raises(SystemExit):
        cycle.cmd_svc_qc(args)

    cmd = captured["cmd"]
    assert "--quiet" in cmd
    assert "--limit" in cmd and cmd[cmd.index("--limit") + 1] == "1"


def test_svc_qc_tees_run_log_to_outdir(tmp_path, monkeypatch):
    """svc-qc writes a _run.log inside out-dir (audit trail)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    out_dir = tmp_path / "svc_out"
    captured = {}
    _patch_svc(monkeypatch, captured)

    args = _parse_args(["svc-qc", str(corpus), "--out-dir", str(out_dir)])
    args.fn = cycle.cmd_svc_qc

    with pytest.raises(SystemExit):
        cycle.cmd_svc_qc(args)

    log = out_dir / "_run.log"
    assert log.is_file(), f"_run.log not written to out-dir: {out_dir}"
    log_text = log.read_text(encoding="utf-8")
    assert "running:" in log_text
    assert "gdpval_qc.cli" in log_text


def test_svc_qc_default_outdir_is_runs_svc_today(tmp_path, monkeypatch):
    """Without --out-dir, svc-qc writes to <project>/runs/svc_<today>/."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    captured = {}
    _patch_svc(monkeypatch, captured)
    # Use a fixed date to avoid race with the real today; also easier cleanup.
    args = _parse_args(["svc-qc", str(corpus), "--date", "2099-12-31"])
    args.fn = cycle.cmd_svc_qc

    expected = cycle.BASE / "runs" / "svc_2099-12-31"
    try:
        with pytest.raises(SystemExit):
            cycle.cmd_svc_qc(args)
        assert expected.is_dir(), f"default out-dir not created at {expected}"
        out_idx = captured["cmd"].index("--output-dir")
        assert captured["cmd"][out_idx + 1] == str(expected)
    finally:
        if expected.exists():
            shutil.rmtree(expected, ignore_errors=True)


def test_svc_qc_refuses_nonexistent_corpus(monkeypatch):
    """svc-qc exits with an error if the corpus dir doesn't exist."""
    monkeypatch.setattr(cycle.sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
    args = _parse_args(["svc-qc", "/no/such/dir/here_12345"])
    args.fn = cycle.cmd_svc_qc

    with pytest.raises(SystemExit) as exc:
        cycle.cmd_svc_qc(args)
    assert exc.value.code and "not a directory" in str(exc.value.code), \
        f"expected 'not a directory' error, got: {exc.value.code!r}"

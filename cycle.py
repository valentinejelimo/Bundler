#!/usr/bin/env python3
"""Task bundler + quality-review driver.

Bundles raw task folders (source documents + expert solution + rubric) into the
harbor evaluator format, then runs the multi-judge quality-review engine over
the bundled (or pre-bundled) corpus.

Every run is a dated folder under cycles/ with numbered stages:

    cycles/<YYYY-MM-DD>/
        00_fetched/   raw tasks (source/ + solution/ subfolders, optional verifier.json)
        02_bundle/    harbor-format package (instruction.md, environment/, solution/, tests/)
        03_evals/     evals received back (input to QC)
        04_qc/        review-engine output (batch_review.json, reviewer_queue.csv, ...)
        05_report/    collected human-facing artifacts

Usage:
    python3 cycle.py bundle <date> [--only 97,160] [--out 02_bundle]
    python3 cycle.py qc     <date> [--spec PATH] [--sample-parallelism N] [-- extra args]
    python3 cycle.py svc-qc <corpus> [--only ids] [--out-dir DIR] [--spec PATH]
    python3 cycle.py report <date>

bundle / qc shell out to the pinned .venv (create it first:
`python -m venv .venv` then `pip install -e .`).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent               # project root (this file's dir)
PIPELINE = BASE                                        # flat layout: no nested pipeline/ subdir
CYCLES = BASE / "cycles"
SPEC_DEFAULT = BASE / "specs" / "gdpval_task_quality_v18.md"
SPEC_SVC = BASE / "specs" / "overlays" / "svc.md"      # SVC/Cirrus front: extends the base spec
# venv interpreter: Scripts/python.exe on Windows, bin/python elsewhere
_VENV_BIN = "Scripts" if os.name == "nt" else "bin"
_VENV_EXE = "python.exe" if os.name == "nt" else "python"
VENV_PY = BASE / ".venv" / _VENV_BIN / _VENV_EXE
STAGES = ["00_fetched", "01_diff", "02_bundle", "03_evals", "04_qc", "05_report"]

# Change detection hashes the task's ACTUAL CONTENT: every artifact file
# (source documents, expert solution, any verifier.json) plus the task-defining
# tracker fields below. Administrative churn (status, links, dates, scores) is
# deliberately excluded so "changed" means the artifacts genuinely differ.
CONTENT_ROW_FIELDS = ("Prompt", "Rubric JSON")
_STUB_FILES = {"_source.txt", "_row.json"}


def _today() -> str:
    return _dt.date.today().isoformat()


def _cycle(date: str) -> Path:
    return CYCLES / date


def _read_version() -> str:
    vf = PIPELINE / "VERSION"
    return vf.read_text().strip().splitlines()[0] if vf.exists() else "unknown"


DOCKERFILE_TEMPLATE = """# EKW task - Python environment with data analysis + document skills

FROM alpine/git:2.45.2 AS skills

ARG ANTHROPIC_SKILLS_REF=da20c92503b2e8ff1cf28ca81a0df4673debdbf7

RUN git clone --filter=blob:none --no-checkout https://github.com/anthropics/skills.git /tmp/anthropic-skills \\
    && cd /tmp/anthropic-skills \\
    && git sparse-checkout init --cone \\
    && git sparse-checkout set skills/xlsx skills/pdf skills/docx \\
    && git -c advice.detachedHead=false checkout "${ANTHROPIC_SKILLS_REF}" \\
    && mkdir -p /harbor/skills \\
    && cp -a skills/xlsx /harbor/skills/xlsx \\
    && cp -a skills/pdf /harbor/skills/pdf \\
    && cp -a skills/docx /harbor/skills/docx

FROM python:3.11-slim
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends \\
    libreoffice poppler-utils && rm -rf /var/lib/apt/lists/*
COPY --from=skills /harbor/skills /harbor/skills
RUN chmod -R a+rX /harbor/skills
RUN pip install --no-cache-dir \\
    numpy==2.3.3 pandas==2.3.3 matplotlib==3.10.7 pypdf==6.7.0 openpyxl \\
    pdfplumber pdf2image pillow defusedxml lxml python-docx pytest pyyaml
COPY data/ /workspace/data/
"""


# Generic Harbor verifier runtime shipped verbatim in every task's tests/ dir
# (test.sh entrypoint + verifier_runtime.py + verifier_lib/). Only verifier.json
# varies per task; these are task-agnostic. Kept as a canonical template so a
# harbor upload validator finds tests/test.sh and a runnable test harness.
HARBOR_TEMPLATE = PIPELINE / "templates" / "harbor"


def _deliverables(vobj: dict | None, golden_dir: Path) -> list[str]:
    """The files solve.sh must reproduce in /workspace/output/.

    A deliverable is a golden file the verifier actually reads back as an
    ``output/<name>`` source path. This deliberately excludes expert-only
    artifacts like ``Reasoning.docx`` that are present in golden_outputs but
    never graded — copying those into output/ would be an unsolicited
    side-effect a side-effect-aware verifier can penalize. Falls back to every
    golden file except Reasoning.docx when no verifier paths are resolvable.
    """
    present = {f.name for f in golden_dir.iterdir() if f.is_file()}
    wanted: set[str] = set()
    if isinstance(vobj, dict):
        for v in vobj.get("verifiers", []) or []:
            fsrc = (v.get("source") or {}).get("file") or {}
            # source.file may be a single dict or a list of file-sources
            # (multi-file verifiers grading across several deliverables)
            for src in (fsrc if isinstance(fsrc, list) else [fsrc]):
                if not isinstance(src, dict):
                    continue
                p = ((src.get("arguments") or {}).get("path") or "")
                base = os.path.basename(p)
                if p.startswith("output/") and base in present:
                    wanted.add(base)
    if not wanted:
        wanted = {n for n in present if n.lower() != "reasoning.docx"}
    return sorted(wanted)


def _solve_sh(deliverables: list[str]) -> str:
    """Oracle solver: copies the graded golden deliverables into output/.

    Deliverables are held in a quoted bash array so filenames containing spaces
    (e.g. ``Output 1 - Task-8.docx``) survive iteration intact.
    """
    files = "".join(f'  "{d}"\n' for d in deliverables)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        'WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"\n'
        'OUTPUT_DIR="$WORKSPACE_DIR/output"\n'
        'SOLUTION_DIR="${SOLUTION_DIR:-/solution}"\n'
        'GOLDEN_DIR="$SOLUTION_DIR/golden_outputs"\n\n'
        'mkdir -p "$OUTPUT_DIR"\n\n'
        'if [ ! -d "$GOLDEN_DIR" ]; then\n'
        '  echo "Missing golden outputs directory: $GOLDEN_DIR" >&2\n'
        "  exit 1\n"
        "fi\n\n"
        f"required_files=(\n{files})\n"
        'for required_file in "${required_files[@]}"; do\n'
        '  if [ ! -f "$GOLDEN_DIR/$required_file" ]; then\n'
        '    echo "Missing golden output: $GOLDEN_DIR/$required_file" >&2\n'
        "    exit 1\n"
        "  fi\n"
        '  cp -p "$GOLDEN_DIR/$required_file" "$OUTPUT_DIR/$required_file"\n'
        "done\n"
    )


def _install_harbor_runtime(tdir: Path) -> None:
    """Copy the generic test.sh + verifier_runtime.py + verifier_lib/ into tests/."""
    tests = tdir / "tests"
    shutil.copy2(HARBOR_TEMPLATE / "test.sh", tests / "test.sh")
    shutil.copy2(HARBOR_TEMPLATE / "verifier_runtime.py", tests / "verifier_runtime.py")
    dst_lib = tests / "verifier_lib"
    if dst_lib.exists():
        shutil.rmtree(dst_lib)
    shutil.copytree(HARBOR_TEMPLATE / "verifier_lib", dst_lib,
                    ignore=shutil.ignore_patterns("__pycache__"))
    os.chmod(tests / "test.sh", 0o755)
    os.chmod(tests / "verifier_runtime.py", 0o755)


def _task_toml(tid: str, row: dict) -> str:
    def esc(v):
        return str(v or "").replace('"', "'").strip()
    return (
        'version = "1.0"\nartifacts = ["/workspace/output"]\n\n'
        "[verifier]\ntimeout_sec = 3600.0\n\n"
        '[verifier.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n\n'
        "[agent]\ntimeout_sec = 14400.0\n\n"
        "[environment]\nbuild_timeout_sec = 900.0\nmemory_mb = 4096\n"
        "storage_mb = 10240\nallow_internet = true\nskills_dir = \"/harbor/skills\"\n\n"
        "[metadata]\n"
        f'task_id = "task_{tid}"\n'
        f'category = "{esc(row.get("Sector"))}"\n'
        f'subdomain = "{esc(row.get("Subdomain"))}"\n'
        f'task_type = "{esc(row.get("Task Type"))}"\n'
        'difficulty = ""\ntags = []\n'
    )


def _subdir(task_dir: Path, *keywords: str) -> Path | None:
    for d in task_dir.iterdir():
        if d.is_dir() and any(k in d.name.lower() for k in keywords):
            return d
    return None


def _verifier_weight(v: dict):
    """Weight lives either at the verifier root or nested under metadata."""
    w = v.get("weight")
    if w is None:
        w = (v.get("metadata") or {}).get("weight")
    try:
        return float(w)
    except (TypeError, ValueError):
        return None


def _verifier_sanity(obj: dict):
    """Objective QC checks on a parsed verifier.json. Yields (level, category, detail)."""
    vs = obj.get("verifiers")
    if not isinstance(vs, list):
        yield ("ERROR", "verifier-shape", "no top-level 'verifiers' array")
        return
    n = len(vs)
    # Minimum-only: no upper limit on verifier count (a thorough rubric with many
    # checks is not a defect). Minimum aligns with the spec's verifier_count_in_range.
    if n < 6:
        yield ("WARN", "verifier-count",
               f"only {n} verifiers; project minimum is 6")
    # weight sum must equal 1.0
    ws = [_verifier_weight(v) for v in vs]
    if any(w is None for w in ws):
        missing = [v.get("name", f"#{i}") for i, (v, w) in enumerate(zip(vs, ws)) if w is None]
        yield ("WARN", "verifier-weight-missing",
               f"{len(missing)} verifier(s) lack a numeric weight: {', '.join(missing)}")
    else:
        total = round(sum(ws), 4)
        if abs(total - 1.0) > 0.001:
            yield ("WARN", "verifier-weight-sum",
                   f"weights sum to {total}, expected 1.0")
    # sequential-name numbering gaps (v1, v2, ...)
    nums = []
    for v in vs:
        nm = str(v.get("name", ""))
        if nm.startswith("v") and nm[1:].isdigit():
            nums.append(int(nm[1:]))
    if nums:
        gaps = sorted(set(range(min(nums), max(nums) + 1)) - set(nums))
        if gaps:
            yield ("INFO", "verifier-numbering-gap",
                   f"verifier names skip {', '.join('v'+str(g) for g in gaps)} "
                   f"(present: v{min(nums)}..v{max(nums)}, count {len(nums)})")
    # outcome mean weight should be >= process mean weight
    def role(v):
        return (v.get("grading_role") or (v.get("metadata") or {}).get("grading_role") or "").lower()
    out_w = [w for v, w in zip(vs, ws) if w is not None and role(v) == "outcome"]
    proc_w = [w for v, w in zip(vs, ws) if w is not None and role(v) == "process"]
    if out_w and proc_w:
        mo, mp = sum(out_w) / len(out_w), sum(proc_w) / len(proc_w)
        if mo < mp - 1e-9:
            yield ("WARN", "verifier-role-balance",
                   f"mean outcome weight {round(mo,4)} < mean process weight {round(mp,4)}")


_LEVEL_RANK = {"ERROR": 0, "WARN": 1, "INFO": 2}


def _write_incident_report(path: Path, date: str, out: Path,
                           assembled: list, no_files: list, incidents: list) -> None:
    """Write a shareable markdown report of all intercurrences for this bundle run."""
    by_level = {"ERROR": [], "WARN": [], "INFO": []}
    for tid, level, cat, detail in incidents:
        by_level.setdefault(level, []).append((tid, cat, detail))
    lines = [
        f"# Pipeline run report — cycle {date}",
        "",
        "_Auto-generated by `cycle.py bundle`. Lists every intercurrence (data or rubric",
        "issue) detected while assembling harbor tasks, for hand-off to the experts team._",
        "",
        "## Summary",
        "",
        f"- Bundle output: `{out.name}/`",
        f"- Tasks assembled (with `verifier.json`): **{len(assembled)}**",
        f"- Tasks skipped (no source files): **{len(no_files)}**",
        f"- Incidents: **{len(incidents)}** "
        f"({len(by_level['ERROR'])} error / {len(by_level['WARN'])} warn / {len(by_level['INFO'])} info)",
        "",
    ]
    if not incidents:
        lines += ["No intercurrences detected. ✅", ""]
    for level in ("ERROR", "WARN", "INFO"):
        rows = by_level[level]
        if not rows:
            continue
        icon = {"ERROR": "🔴", "WARN": "🟡", "INFO": "🔵"}[level]
        lines += [f"## {icon} {level} ({len(rows)})", "", "| Task | Category | Detail |",
                  "|------|----------|--------|"]
        for tid, cat, detail in sorted(rows, key=lambda r: (r[1], int(r[0]) if r[0].isdigit() else 0)):
            lines.append(f"| task_{tid} | `{cat}` | {detail.replace('|', chr(92)+'|')} |")
        lines.append("")
    if no_files:
        lines += ["## Not downloaded (skipped entirely)", "",
                  ", ".join(f"task_{t}" for t in no_files), ""]
    path.write_text("\n".join(lines))


def cmd_bundle(args) -> None:
    """Assemble harbor-format tasks from 00_fetched/<id>/ and report all intercurrences.

    --only 97,160   restrict to specific task ids (comma-separated)
    --out NAME       output subdir under the cycle (default 02_bundle)
    """
    c = _cycle(args.date)
    only = {t.strip() for t in (args.only or "").split(",") if t.strip()}
    out_name = getattr(args, "out", None) or "02_bundle"
    fetched, out = c / "00_fetched", c / out_name
    out.mkdir(exist_ok=True)
    exclude = set()
    exf = c / "_exclude.txt"
    if exf.exists():
        exclude = {ln.strip() for ln in exf.read_text().splitlines()
                   if ln.strip() and not ln.startswith("#")}
    assembled, no_files = [], []
    tid_fronts: dict[str, str] = {}
    incidents = []  # (tid, level, category, detail)
    # Task ids aren't always numeric once non-EKW fronts are in the mix (SVC's
    # "cirrus-y1-task-05", for instance) — accept any fetched dir, not just
    # digit-named ones, and sort as plain strings.
    for d in sorted((p for p in fetched.iterdir() if p.is_dir()), key=lambda p: p.name):
        tid = d.name
        if tid in exclude or (only and tid not in only):
            continue
        rj = d / "_row.json"
        row = json.loads(rj.read_text()) if rj.exists() else {}
        front = row.get("front", "ekw")
        tid_fronts[tid] = front

        # SVC tasks arrive pre-bundled (see SOP 1.7 — a decomposition-DAG corpus
        # built outside this pipeline, not raw source+solution files we assemble
        # from scratch). If fetch already dropped a complete harbor-shaped task
        # here, register it verbatim instead of running the EKW assembly path
        # below, which expects "source"/"solution" subfolders + a Prompt/Rubric
        # tracker cell neither of which a pre-bundled SVC task has.
        if front == "svc":
            has_verifier = (d / "tests" / "verifier.json").is_file() or (d / "tests" / "rubric.json").is_file()
            if (d / "instruction.md").is_file() and (d / "task.toml").is_file() and has_verifier:
                tdir = out / f"task_{tid}"
                if tdir.exists():
                    shutil.rmtree(tdir)
                shutil.copytree(d, tdir, ignore=shutil.ignore_patterns("_source.txt", "_row.json"))
                assembled.append(tid)
                incidents.append((tid, "INFO", "svc-prebundled",
                                  "SVC task arrived pre-bundled; registered into 02_bundle/ verbatim, "
                                  "no harbor assembly run"))
                continue
            incidents.append((tid, "ERROR", "svc-not-prebundled",
                              "front=svc but 00_fetched/<id>/ isn't a complete harbor-shaped task "
                              "(missing instruction.md/task.toml/tests/{verifier,rubric}.json) — "
                              "SVC tasks are expected pre-bundled; cmd_bundle has no from-scratch "
                              "assembly path for SVC's DAG/rubric.json shape"))
            no_files.append(tid)
            continue

        src = _subdir(d, "source")
        sol = _subdir(d, "solution", "expert")
        if not src or not any(src.iterdir()):
            no_files.append(tid)
            incidents.append((tid, "ERROR", "no-source-files",
                              "no source documents downloaded; task skipped entirely"))
            continue
        tdir = out / f"task_{tid}"
        if tdir.exists():
            shutil.rmtree(tdir)
        (tdir / "environment" / "data").mkdir(parents=True)
        (tdir / "solution" / "golden_outputs").mkdir(parents=True)
        (tdir / "tests").mkdir()
        # instruction.md
        (tdir / "instruction.md").write_text(str(row.get("Prompt", "")))
        # environment
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, tdir / "environment" / "data" / f.name)
        (tdir / "environment" / "Dockerfile").write_text(DOCKERFILE_TEMPLATE)
        # solution
        if sol and any(f.is_file() for f in sol.iterdir()):
            for f in sol.iterdir():
                if f.is_file():
                    shutil.copy2(f, tdir / "solution" / "golden_outputs" / f.name)
        else:
            incidents.append((tid, "WARN", "no-solution",
                              "no expert/golden solution files found"))
        # task.toml
        (tdir / "task.toml").write_text(_task_toml(tid, row))
        # verifier.json: prefer a shipped (possibly hand-repaired) file, else parse the Rubric cell
        vfile = d / "verifier.json"
        rub = (row.get("Rubric JSON") or row.get("Rubric") or "").strip()
        wrote_v, vobj = False, None
        if vfile.exists():
            try:
                vobj = json.loads(vfile.read_text())
                shutil.copy2(vfile, tdir / "tests" / "verifier.json")
                wrote_v = True
                incidents.append((tid, "INFO", "verifier-shipped",
                                  "used shipped verifier.json from 00_fetched/ "
                                  "(hand-repaired or externally supplied rubric)"))
            except Exception as e:
                incidents.append((tid, "ERROR", "verifier-shipped-bad",
                                  f"shipped verifier.json is invalid JSON: {e}"))
        elif rub:
            try:
                vobj = json.loads(rub)  # already valid
                (tdir / "tests" / "verifier.json").write_text(rub)
                wrote_v = True
            except Exception:
                # tolerate a single valid JSON value followed by trailing junk
                # (e.g. a stray "</verifier>" wrapper tag pasted into the cell)
                try:
                    vobj, _ = json.JSONDecoder().raw_decode(rub)
                    (tdir / "tests" / "verifier.json").write_text(
                        json.dumps(vobj, indent=2, ensure_ascii=False))
                    wrote_v = True
                    incidents.append((tid, "WARN", "rubric-recovered",
                                      "Rubric cell had trailing junk after valid JSON "
                                      "(e.g. stray </verifier> tag); auto-recovered via raw_decode"))
                except Exception as e:
                    incidents.append((tid, "ERROR", "rubric-malformed",
                                      f"Rubric cell is malformed JSON, not auto-recoverable ({e}); "
                                      f"hand-repair and drop verifier.json in 00_fetched/{tid}/"))
        else:
            incidents.append((tid, "ERROR", "rubric-empty",
                              "Rubric cell is empty in the tracker; no verifier.json produced"))
        if wrote_v:
            assembled.append(tid)
            if isinstance(vobj, dict):
                for level, cat, detail in _verifier_sanity(vobj):
                    incidents.append((tid, level, cat, detail))
        # harbor runtime: generic test.sh + verifier_runtime.py + verifier_lib/,
        # and a per-task solve.sh oracle. Shipped for every assembled task so a
        # strict harbor upload validator finds a runnable tests/ and solution/.
        _install_harbor_runtime(tdir)
        golden = tdir / "solution" / "golden_outputs"
        deliverables = _deliverables(vobj, golden) if any(golden.iterdir()) else []
        if deliverables:
            solve = tdir / "solution" / "solve.sh"
            solve.write_text(_solve_sh(deliverables))
            os.chmod(solve, 0o755)
        else:
            incidents.append((tid, "WARN", "no-solve-sh",
                              "no golden deliverables resolved; solution/solve.sh not written"))
    report = out / "_INCIDENTS.md"
    _write_incident_report(report, args.date, out, assembled, no_files, incidents)
    print(f"assembled {len(assembled)} harbor tasks -> {out}")
    errs = [i for i in incidents if i[1] == "ERROR"]
    warns = [i for i in incidents if i[1] == "WARN"]
    if errs:
        print(f"  {len(errs)} ERROR incident(s): "
              + ", ".join(f"task_{t}:{cat}" for t, _, cat, _ in errs))
    if warns:
        print(f"  {len(warns)} WARN incident(s): "
              + ", ".join(f"task_{t}:{cat}" for t, _, cat, _ in warns))
    print(f"  intercurrence report -> {report}")

    if not assembled:
        print("No tasks assembled - check the incident report above before proceeding.")
    else:
        print(f"Next: run the review with  python3 cycle.py qc {args.date}")

def cmd_qc(args) -> None:
    c = _cycle(args.date)
    corpus = c / "03_evals"
    outdir = c / "04_qc"
    if not corpus.is_dir() or not any(corpus.iterdir()):
        sys.exit(f"no evals in {corpus} — drop the returned evals there first")
    if not VENV_PY.exists():
        sys.exit(f"pipeline venv missing at {VENV_PY}")
    cmd = [str(VENV_PY), "-m", "gdpval_qc.cli",
           "--spec", str(args.spec), "--corpus", str(corpus), "--output-dir", str(outdir)]
    # Heavy parallelism by default: grade several tasks at once (default 1 in the
    # underlying CLI is far too slow for a full cycle). Only inject when the user
    # hasn't set the flag themselves via `extra`. The API token rate limiters
    # (Gemini 2M tpm, OpenAI 1M tpm/key) throttle gracefully under the load.
    if not any(a.startswith("--sample-parallelism") for a in args.extra):
        cmd += ["--sample-parallelism", str(args.sample_parallelism)]
    if not any(a.startswith("--dim-parallelism") for a in args.extra):
        cmd += ["--dim-parallelism", str(args.dim_parallelism)]
    # Calibration (5 known-verdict goldens) is OPT-IN: it adds a full task's worth
    # of grading + slow multimodal dims to every run, and is only useful as a
    # drift check on the full/final pass. Skip by default; re-enable with
    # --calibrate (or by passing --skip-calibration explicitly to override).
    if not getattr(args, "calibrate", False) and not any(
            a.startswith("--skip-calibration") for a in args.extra):
        cmd += ["--skip-calibration"]
    cmd += args.extra
    print("running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc == 0:
        print(f"QC done -> {outdir}.  Next: python3 cycle.py report {args.date}")
    sys.exit(rc)


def cmd_svc_qc(args) -> None:
    """SVC/Cirrus QC entrypoint — no fetch/diff/bundle: SVC tasks arrive pre-bundled
    (no tracker, no Shared Drive step), so this points straight at a corpus directory
    instead of a cycles/<date>/ stage folder.

    Wraps the invocation documented in SVC/RUN_FULL_QC.md against the SVC overlay
    spec (specs/overlays/svc.md by default; pass --spec specs/overlays/svc-noeval.md
    when evals/<model>/trial_*/ isn't populated yet). --no-conversion is the default
    (text-only judging — reliable, no libreoffice dependency; formula-driven xlsx
    sheets may extract as blank, a known false-"unfinished" risk noted in
    RUN_FULL_QC.md — pass --with-conversion to re-check a flagged node).

    Unlike `qc` (which just subprocess.call and leaves logging to whoever invoked
    it — most historical cycles never redirected it), this always tees its own
    run to <out-dir>/_run.log, so every SVC run keeps a timing/audit trail.

    cycle.py svc-qc <corpus> [--only ids] [--out-dir DIR] [--date YYYY-MM-DD]
                     [--spec PATH] [--sample-parallelism N] [--with-conversion]
                     [extra args passed through to gdpval-assess]
    """
    corpus = Path(args.corpus).expanduser().resolve()
    if not corpus.is_dir():
        sys.exit(f"not a directory: {corpus}")
    if not VENV_PY.exists():
        sys.exit(f"pipeline venv missing at {VENV_PY}")

    date = args.date or _today()
    outdir = (Path(args.out_dir).expanduser().resolve() if args.out_dir
              else (BASE / "runs" / f"svc_{date}"))
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [str(VENV_PY), "-m", "gdpval_qc.cli",
           "--spec", str(args.spec), "--corpus", str(corpus), "--output-dir", str(outdir)]
    if not any(a.startswith("--skip-calibration") for a in args.extra):
        cmd += ["--skip-calibration"]   # the spec's calibration goldens are EKW's, not SVC's
    if not args.with_conversion and not any(a.startswith("--no-conversion") for a in args.extra):
        cmd += ["--no-conversion"]
    if args.only and not any(a.startswith("--tasks") for a in args.extra):
        cmd += ["--tasks", args.only]
    if not any(a.startswith("--sample-parallelism") for a in args.extra):
        cmd += ["--sample-parallelism", str(args.sample_parallelism)]
    cmd += args.extra

    log_path = outdir / "_run.log"
    print("running:", " ".join(cmd))
    print(f"teeing output -> {log_path}")
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("running: " + " ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        rc = proc.wait()
    if rc == 0:
        print(f"SVC QC done -> {outdir}")
    sys.exit(rc)


def cmd_report(args) -> None:
    c = _cycle(args.date)
    src, dst = c / "04_qc", c / "05_report"
    dst.mkdir(exist_ok=True)
    picked = []
    for f in ["batch_summary.md", "reviewer_queue.csv", "dashboard.html", "REPORT.md",
              "batch_review.json"]:
        s = src / f
        if s.is_file():
            (dst / f).write_bytes(s.read_bytes())
            picked.append(f)
    print(f"collected into 05_report/: {', '.join(picked) or '(nothing - run qc first)'}")
    print("Done. Review the artifacts in 05_report/.")



def main() -> None:
    p = argparse.ArgumentParser(description="Task bundler + quality-review driver")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bundle", help="assemble harbor-format tasks from 00_fetched/<id>/")
    b.add_argument("date")
    b.add_argument("--only", default="",
                   help="comma-separated task ids to restrict the bundle to")
    b.add_argument("--out", default="02_bundle",
                   help="output subdir under the cycle (default 02_bundle)")
    b.set_defaults(fn=cmd_bundle)

    q = sub.add_parser("qc", help="run the quality-review engine over a cycle's 03_evals/")
    q.add_argument("date")
    q.add_argument("--spec", default=str(SPEC_DEFAULT))
    q.add_argument("--sample-parallelism", type=int, default=5, dest="sample_parallelism",
                   help="tasks graded concurrently (default 5)")
    q.add_argument("--dim-parallelism", type=int, default=4, dest="dim_parallelism",
                   help="dims graded concurrently per task (default 4)")
    q.add_argument("--calibrate", action="store_true",
                   help="also grade the calibration goldens (drift check); off by default")
    q.add_argument("extra", nargs="*",
                   help="extra args passed to the review engine (e.g. --no-multimodal)")
    q.set_defaults(fn=cmd_qc)

    sv = sub.add_parser("svc-qc",
                        help="review a pre-bundled corpus directly (no fetch/bundle step)")
    sv.add_argument("corpus", help="path to the tasks/ directory")
    sv.add_argument("--only", default="",
                    help="comma-separated task ids to restrict to (passed as --tasks)")
    sv.add_argument("--out-dir", default="", dest="out_dir",
                    help="output dir (default: runs/svc_<date>/)")
    sv.add_argument("--date", default="",
                    help="date stamp for the default out-dir (default: today)")
    sv.add_argument("--spec", default=str(SPEC_SVC),
                    help="spec to run (default: specs/overlays/svc.md; use "
                         "specs/overlays/svc-noeval.md when evals/ isn't populated yet)")
    sv.add_argument("--sample-parallelism", type=int, default=3, dest="sample_parallelism",
                    help="tasks graded concurrently (default 3)")
    sv.add_argument("--with-conversion", action="store_true",
                    help="enable libreoffice doc conversion (default: --no-conversion, text-only)")
    sv.add_argument("extra", nargs="*", help="extra args passed to the review engine")
    sv.set_defaults(fn=cmd_svc_qc)

    rp = sub.add_parser("report", help="collect human-facing artifacts from 04_qc/ into 05_report/")
    rp.add_argument("date")
    rp.set_defaults(fn=cmd_report)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

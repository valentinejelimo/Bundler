# Bundler — Task Bundling + Comprehensive Review

This project does two things, and only two things:

1. **Bundle** raw task folders (source documents + expert solution + rubric)
   into the harbor evaluator format (a self-contained `task_<id>/` package with
   `instruction.md`, `environment/`, `solution/`, and `tests/`).
2. **Review** those bundled tasks with a multi-judge LLM ensemble against a
   versioned quality spec — the same engine that grades a full corpus.

There is no tracker integration, no platform upload, no one-pager / message
drafting step, no auto-fix agent. If you want any of that, this is the wrong
starting point.

---

## What you need

- **Python 3.11+** (3.12 or 3.13 also fine).
- API keys for the two default judge models (see `.env.example`):
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- **LibreOffice** (optional) — only needed if your task corpus contains
  `.docx` / `.doc` / `.pptx` / `.ppt` / `.xls` files that must be converted to
  PDF for the judges to read as document blocks. If every input and golden
  file is `.xlsx`, `.pdf`, or plain text, you can skip LibreOffice and run
  text-only (the `svc-qc` command does this by default).

---

## Set up (once)

From this folder:

```bash
# 1. Create a virtual environment.
python -m venv .venv

# 2. Activate it.
#    Windows (Git Bash / PowerShell):     source .venv/Scripts/activate
#    macOS / Linux:                        source .venv/bin/activate

# 3. Install the engine in editable mode.
pip install -e .

# 4. Copy the env template and fill in your keys.
cp .env.example .env
#   then edit .env and set OPENAI_API_KEY and GEMINI_API_KEY
```

`pip install -e .` puts the `gdpval-assess` script on your PATH. The driver
(`cycle.py`) calls the engine through this same virtual environment, so
activate the venv before running any command below.

---

## How a task is shaped

### Raw task (input to `bundle`)

A dated cycle folder, with one subdirectory per task under `00_fetched/`:

```
cycles/2026-10-21/
  00_fetched/
    97/
      _row.json              # tracker row: at least a "Prompt" and "Rubric JSON" field
      source/                # the input files the agent receives
        report.xlsx
        memo.docx
      solution/              # the expert / golden solution
        Output - Task 97.xlsx
      verifier.json          # optional: a hand-repaired rubric (overrides the Rubric field)
    160/
      ...
```

`_row.json` carries the prompt and rubric. If a task already ships a
`verifier.json` next to its source/solution, that file wins over the
`_row.json` rubric field (it's the hand-repaired version).

### Bundled task (output of `bundle`, input to `qc`)

```
cycles/2026-10-21/
  02_bundle/
    task_97/
      instruction.md         # the agent-facing prompt (from _row.json "Prompt")
      environment/
        Dockerfile           # generated: Python 3.11 + libreoffice + data-analysis libs
        data/                # the source documents
      solution/
        golden_outputs/      # the expert deliverables
        solve.sh            # oracle: copies the golden deliverables into output/
      tests/
        verifier.json        # the rubric (from _row.json or the shipped file)
        test.sh              # generic harbor test entrypoint
        verifier_runtime.py # generic harbor runtime
        verifier_lib/        # generic harbor verifier library
      task.toml              # task metadata
```

### Pre-bundled corpus (alternative — skip `bundle`, use `svc-qc`)

If your tasks already arrive in the bundled shape above (each has
`instruction.md`, `task.toml`, `tests/verifier.json` or `tests/rubric.json`),
you can point `svc-qc` straight at the directory and skip the `bundle` step
entirely. `svc-qc` grades text-only by default (no LibreOffice).

---

## Step 1 — Bundle tasks

```bash
python cycle.py bundle <date> [--only 97,160] [--out 02_bundle]
```

- `<date>` is the cycle folder under `cycles/` (e.g. `2026-10-21`).
- `--only` restricts the bundle to specific task ids (comma-separated).
- `--out` names the output subdir under the cycle (default `02_bundle`).

What it does, for each task in `cycles/<date>/00_fetched/<id>/`:

1. Reads the prompt from `_row.json` and writes it to `instruction.md`.
2. Copies the source files into `environment/data/` and writes a `Dockerfile`.
3. Copies the expert solution into `solution/golden_outputs/`.
4. Writes the rubric to `tests/verifier.json` — preferring a shipped
   `verifier.json`, falling back to the `_row.json` rubric field.
5. Installs the generic harbor runtime (`test.sh`, `verifier_runtime.py`,
   `verifier_lib/`) into `tests/`.
6. Writes `solution/solve.sh` (the oracle that replays the golden deliverables).
7. Writes `task.toml` with the task metadata.

It prints a summary and writes an incident report to
`cycles/<date>/02_bundle/_INCIDENTS.md` listing every ERROR / WARN / INFO
it hit per task (missing source files, malformed rubric, weight mismatches,
etc.). Read it before proceeding — tasks with an ERROR were not assembled.

---

## Step 2 — Run the comprehensive review

There are two entry points. Use `qc` after bundling a cycle; use `svc-qc`
to review a pre-bundled corpus directly.

### `qc` — review a bundled cycle

```bash
python cycle.py qc <date> [--spec PATH] [--sample-parallelism N] [--dim-parallelism N] [-- calibrate] [-- extra args]
```

- `<date>` is the same cycle folder. The review reads from
  `cycles/<date>/03_evals/` and writes to `cycles/<date>/04_qc/`.
  **Before running**, drop the returned evals (model-run trajectories and
  verifier reports) into `03_evals/`. If you have no evals yet and just want
  to grade the task package itself, see "Review without evals" below.
- `--spec` overrides the quality spec (default
  `specs/gdpval_task_quality_v18.md`).
- `--sample-parallelism N` — tasks graded concurrently (default 5).
- `--dim-parallelism N` — quality dimensions graded concurrently per task
  (default 4).
- `--calibrate` — also grade the 5 calibration goldens as a drift check
  (off by default; only useful on a final pass).
- Any args after `--` are passed through to the engine (e.g.
  `-- --no-multimodal` to force text-only judging).

### `svc-qc` — review a pre-bundled corpus directly

```bash
python cycle.py svc-qc <corpus> [--only ids] [--out-dir DIR] [--spec PATH] [--sample-parallelism N] [--with-conversion] [-- extra args]
```

- `<corpus>` is the directory containing the bundled `task_*/` folders.
- `--only` restricts to specific task ids (passed as `--tasks`).
- `--out-dir DIR` — where to write the review output
  (default: `runs/svc_<today>/`).
- `--spec` — which spec to run. Default `specs/overlays/svc.md`. Use
  `specs/overlays/svc-noeval.md` when `evals/<model>/trial_*/` isn't
  populated yet.
- `--with-conversion` — enable LibreOffice document-to-PDF conversion.
  The default is text-only (`--no-conversion`), which is reliable and has
  no LibreOffice dependency.
- Any args after `--` are passed through to the engine.

This command tees its run log to `<out-dir>/_run.log`, so every run keeps
an audit trail.

### Review without evals

If you haven't run the platform evals yet, you can still grade the task
package against the spec — use the no-eval spec variant:

```bash
python cycle.py svc-qc ./my_corpus --spec specs/overlays/svc-noeval.md --out-dir runs/noeval
```

Or call the engine directly (bypassing `cycle.py`) for full control:

```bash
gdpval-assess --spec specs/gdpval_task_quality_v18.md \
              --corpus ./my_corpus \
              --output-dir runs/full \
              --dry-run          # validate spec + load samples, no API calls
```

`--dry-run` costs nothing and confirms your corpus loads before you spend
on judge calls. Remove it to run for real.

---

## Step 3 — Collect the human-facing report

```bash
python cycle.py report <date>
```

Copies the review artifacts (`batch_summary.md`, `reviewer_queue.csv`,
`batch_review.json`, etc.) from `cycles/<date>/04_qc/` into
`cycles/<date>/05_report/`. Read those.

---

## What the review produces

The engine runs **25 quality dimensions** over each task — 21 graded by LLM
judges, 4 deterministic — and writes:

| File | What it is |
|---|---|
| `batch_review.json` | Full machine-readable verdict + every dimension finding + every issue + every judge attempt (for audit / replay). |
| `reviewer_queue.csv` | One row per sample with the top 3 must-fix issues and up to 5 minor suggestions, each with severity / fixability / headline / action / evidence / detail. |
| `batch_summary.md` | Human-readable verdict mix + failure-mode histogram. |
| `raw/sample_<id>.json` | Per-sample LLM transcripts (one file per sample). |

Each sample lands in one of five verdict tiers:

| Tier | Meaning |
|---|---|
| `accept` | All judges happy, no findings above informational. |
| `accept_with_fixes` | Minor advisory issues only; ship it, address in next pass. |
| `needs_review` | Judges disagree, or low confidence at severity 2+. Human triage. |
| `reject_fixable` | Material issues; the task isn't shippable but is salvageable. |
| `reject_structural` | A blocking dimension hit `reject_at_severity` with structural fixability. |

The framework **does not modify your data** — it grades, classifies, and
queues. Action is yours.

---

## CLI reference

```
cycle.py bundle <date> [--only 97,160] [--out 02_bundle]
cycle.py qc     <date> [--spec PATH] [--sample-parallelism N] [--dim-parallelism N] [--calibrate] [-- extra...]
cycle.py svc-qc <corpus> [--only ids] [--out-dir DIR] [--spec PATH] [--sample-parallelism N] [--with-conversion] [-- extra...]
cycle.py report <date>
```

The engine itself (`gdpval-assess`) accepts more flags; pass them through
with `--`:

```
gdpval-assess --spec PATH --corpus PATH --output-dir PATH
              [--limit N] [--tasks task_1,task_3]
              [--sample-parallelism N] [--dim-parallelism N]
              [--dry-run] [--skip-calibration]
              [--no-multimodal] [--no-conversion] [--cache-dir PATH]
              [--quiet] [--heartbeat-only] [--heartbeat-seconds N]
```

---

## Project layout

```
Bundler/
├── cycle.py                  # driver: bundle, qc, svc-qc, report
├── gdpval_qc/                # the review engine (multi-judge LLM ensemble)
├── specs/                    # the quality specs the engine grades against
│   ├── gdpval_task_quality_v18.md   # the base spec (default for `qc`)
│   └── overlays/
│       ├── svc.md            # SVC/Cirrus overlay (default for `svc-qc`)
│       └── svc-noeval.md     # same overlay, no-eval gates (use before evals)
├── templates/harbor/         # generic verifier runtime copied into every bundle
├── pyproject.toml            # install metadata + dependencies
└── .env.example              # API key template (copy to .env)
```

`cycles/` and `runs/` are created as you work and are not part of the
project — never commit task data or API keys.

---

## Troubleshooting

**`pipeline venv missing at ...`** — the `qc` and `svc-qc` commands shell
out to the venv's Python. Create it (`python -m venv .venv`) and install the
engine (`pip install -e .`), then activate it before running.

**No evals in `03_evals/`** — `qc` refuses to run if `03_evals/` is empty.
Either drop the returned evals there, or use `svc-qc` against the bundled
corpus with the `svc-noeval.md` spec.

**Judge API errors / timeouts** — the engine is rate-limited but still
depends on your API quotas. Lower `--sample-parallelism` (e.g. 1 or 2) and
`--dim-parallelism` (e.g. 2) to reduce in-flight calls.

**`verifier.json` errors in the incident report** — open
`cycles/<date>/02_bundle/_INCIDENTS.md`. A `rubric-malformed` ERROR means
the `_row.json` rubric field isn't valid JSON; hand-repair it and drop a
`verifier.json` in `00_fetched/<id>/`, then re-run `bundle`.

**LibreOffice not found** — the `qc` command attempts document-to-PDF
conversion. If LibreOffice isn't installed, pass `-- --no-conversion` to
`qc` (text-only judging), or just use `svc-qc` (text-only by default).

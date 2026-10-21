#!/usr/bin/env python3
"""gdpval_qc.cli — drive the v18 quality framework end-to-end from the CLI.

Loads a v18 spec, walks a task-corpus directory, runs every active dimension
through the real LLM judge ensemble, computes verdicts, runs the synthesizer,
and writes batch_review.json + reviewer_queue.csv + batch_summary.md + raw
transcripts to an output directory.

After `pip install`, invoke as ``gdpval-assess`` (PATH-installed console
script) or ``python -m gdpval_qc.cli``.

Examples
--------

Assess the bundled sample corpus (1 task — task_11; good for a first run):

    gdpval-assess \\
        --spec specs/gdpval_task_quality_v18.md \\
        --corpus data/sample_corpus \\
        --output-dir runs/v18_smoke

Assess only the first N tasks of a larger corpus (~21 LLM dims × 2 judges
+ 1 synth ≈ 43 LLM calls per task):

    gdpval-assess \\
        --spec specs/gdpval_task_quality_v18.md \\
        --corpus /path/to/your/corpus \\
        --output-dir runs/v18_smoke \\
        --limit 3

Assess specific task IDs:

    gdpval-assess \\
        --spec specs/gdpval_task_quality_v18.md \\
        --corpus /path/to/your/corpus \\
        --output-dir runs/v18_pick \\
        --tasks task_1,task_10,task_28

Run the full batch (this will spend real money):

    gdpval-assess \\
        --spec specs/gdpval_task_quality_v18.md \\
        --corpus /path/to/your/corpus \\
        --output-dir runs/v18_full

Required environment (loaded from .env if present):
    OPENAI_API_KEY     OpenAI key for gpt-5.5 (judge + synthesizer)
    GEMINI_API_KEY     Gemini key for gemini-3.1-pro-preview judge
                       (GOOGLE_API_KEY also accepted)

Dry-run mode (no API calls) — use to validate spec + loader + preconditions
without spending tokens:

    gdpval-assess --spec ... --corpus ... --output-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

# Load .env early so API keys are available
load_dotenv()


# ── Progress printers ──────────────────────────────────────────────────────

class _ProgressPrinter:
    """Thread-safe stderr printer for engine progress events.

    Output is one line per event, prefixed with elapsed-since-batch-start
    so the user can see the framework is alive even when an individual
    judge call is taking minutes at high reasoning effort.

    Per-sample dim-completion counts are tracked so each sample shows its
    own [k/N] progress regardless of sample-parallelism interleaving.
    Tracks in-flight concurrency on every dim_start / dim_complete so the
    user has direct visibility into whether dim_parallelism is taking
    effect (a common confusion: dims completing serially-looking when they
    were actually parallel but had similar latency).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t0: float | None = None
        self._sample_state: dict[str, dict[str, Any]] = {}
        self._in_flight: int = 0
        self._peak_in_flight: int = 0

    def __call__(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._handle(event)

    def _stamp(self) -> str:
        if self._t0 is None:
            return "+0.0s"
        return f"+{time.time() - self._t0:5.1f}s"

    def _handle(self, e: dict[str, Any]) -> None:
        kind = e.get("kind")

        if kind == "batch_start":
            self._t0 = e.get("ts", time.time())
            n = e.get("total_samples", 0)
            d = e.get("total_dims_per_sample", 0)
            self._line(f"[{self._stamp()}] batch start — {n} sample(s) × {d} dims each")
            return

        if kind == "sample_start":
            sid = e.get("sample_id", "?")
            self._sample_state[sid] = {"start_ts": time.time(),
                                         "completed": 0,
                                         "total": e.get("total_dims", 0)}
            self._line(f"[{self._stamp()}] {sid}: starting (0/{e.get('total_dims')} dims)")
            return

        if kind == "dim_start":
            self._in_flight += 1
            self._peak_in_flight = max(self._peak_in_flight, self._in_flight)
            sid = e.get("sample_id", "?")
            dim = e.get("dim_name", "?")
            check = e.get("check_kind", "?")
            # Mark programmatic dims separately so they don't muddy the
            # in-flight concurrency picture (they complete in microseconds).
            tag = "[prog]" if check == "programmatic" else "[ llm]"
            self._line(
                f"[{self._stamp()}] {sid}: ⇢ {tag} {dim:<40} "
                f"(in-flight={self._in_flight}, peak={self._peak_in_flight})"
            )
            return

        if kind == "dim_complete":
            self._in_flight = max(0, self._in_flight - 1)
            sid = e.get("sample_id", "?")
            st = self._sample_state.setdefault(sid, {"completed": 0, "total": 0})
            st["completed"] = st.get("completed", 0) + 1
            sev = e.get("severity", 0)
            applicable = e.get("applicable", True)
            elapsed = e.get("elapsed_s", 0.0)
            tag = "skipped" if not applicable else f"sev {sev}"
            self._line(
                f"[{self._stamp()}] {sid}: "
                f"[{st['completed']:>2}/{st.get('total', '?')}] "
                f"{e.get('dim_name'):<40} {tag:>8} ({elapsed:5.1f}s, "
                f"in-flight={self._in_flight})"
            )
            return

        if kind == "sample_synth":
            sid = e.get("sample_id", "?")
            self._line(f"[{self._stamp()}] {sid}: synthesizing…")
            return

        if kind == "sample_complete":
            sid = e.get("sample_id", "?")
            v = e.get("verdict")
            elapsed = e.get("elapsed_s", 0.0)
            nti = e.get("top_issue_count", 0)
            mm = int(elapsed // 60)
            ss = elapsed - 60 * mm
            self._line(
                f"[{self._stamp()}] {sid}: DONE → {v}  "
                f"({nti} top issue{'s' if nti != 1 else ''}, "
                f"{mm}m {ss:4.1f}s)"
            )
            return

        if kind == "batch_complete":
            elapsed = e.get("elapsed_s", 0.0)
            mm = int(elapsed // 60)
            ss = elapsed - 60 * mm
            self._line(
                f"[{self._stamp()}] batch complete in {mm}m {ss:4.1f}s  "
                f"(peak dim concurrency observed: {self._peak_in_flight})"
            )
            return

    def _line(self, text: str) -> None:
        print(text, file=sys.stderr, flush=True)


class _HeartbeatPrinter:
    """Background heartbeat printer for non-TTY environments where the
    per-event printer's output may be buffered/lost. Emits a single line
    every `interval_s` seconds with the current per-sample dim progress.
    Stops when the batch_complete event fires."""

    def __init__(self, interval_s: float = 30.0) -> None:
        self._interval = interval_s
        self._lock = threading.Lock()
        self._sample_state: dict[str, dict[str, Any]] = {}
        self._t0: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __call__(self, event: dict[str, Any]) -> None:
        with self._lock:
            kind = event.get("kind")
            if kind == "batch_start":
                self._t0 = event.get("ts", time.time())
                self._thread = threading.Thread(
                    target=self._loop, daemon=True, name="v18-heartbeat",
                )
                self._thread.start()
            elif kind == "sample_start":
                sid = event["sample_id"]
                self._sample_state[sid] = {
                    "completed": 0, "total": event.get("total_dims", 0),
                }
            elif kind == "dim_complete":
                sid = event["sample_id"]
                st = self._sample_state.setdefault(sid, {"completed": 0, "total": 0})
                st["completed"] += 1
            elif kind == "sample_complete":
                sid = event["sample_id"]
                self._sample_state.pop(sid, None)
            elif kind == "batch_complete":
                self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._interval)
            if self._stop.is_set():
                break
            with self._lock:
                if not self._sample_state:
                    continue
                elapsed = time.time() - (self._t0 or time.time())
                parts = [
                    f"{sid} {st['completed']}/{st['total']}"
                    for sid, st in self._sample_state.items()
                ]
                print(
                    f"[heartbeat +{elapsed:.0f}s] in progress: " + ", ".join(parts),
                    file=sys.stderr, flush=True,
                )


def _make_progress_callback(quiet: bool, heartbeat_only: bool, heartbeat_seconds: int):
    if quiet:
        return None
    if heartbeat_only:
        return _HeartbeatPrinter(interval_s=heartbeat_seconds)
    return _ProgressPrinter()


def main() -> int:
    args = _parse_args()

    # ── Spec ──────────────────────────────────────────────────────────────
    from gdpval_qc.v18_spec import V18Spec, SpecValidationError
    try:
        spec = V18Spec.from_markdown(args.spec)
    except SpecValidationError as exc:
        print(f"ERROR: failed to load spec {args.spec}:\n{exc}", file=sys.stderr)
        return 2
    print(f"Loaded spec: {spec.name} v{spec.version} "
          f"({len(spec.dimensions)} dims, "
          f"{sum(1 for d in spec.dimensions if d.check_kind == 'llm')} LLM + "
          f"{sum(1 for d in spec.dimensions if d.check_kind == 'programmatic')} programmatic)")
    print(f"Judge models: {[m.model for m in spec.models]}")
    print(f"Synthesizer:  {spec.synthesis.judge_model}")

    # ── Loader (with optional document conversion + caching) ──────────────
    from gdpval_qc.v18_loader import TaskCorpusLoader
    converter = None
    cache_dir = None
    if not args.no_conversion:
        from gdpval_qc.converters import DocumentConverter
        converter = DocumentConverter()
        cache_dir = args.cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Document conversion: enabled (cache_dir={cache_dir})")
    else:
        print("Document conversion: disabled (--no-conversion)")
    loader = TaskCorpusLoader(args.corpus, spec, converter=converter, cache_dir=cache_dir)
    all_ids = loader.task_ids()
    print(f"Corpus has {len(all_ids)} task(s) at {args.corpus}")

    selected_ids = _select_task_ids(all_ids, args)
    print(f"Selected: {len(selected_ids)} task(s) — {selected_ids}")

    if not selected_ids:
        print("ERROR: no tasks selected", file=sys.stderr)
        return 2

    # ── Batch API: submit-only mode ────────────────────────────────────────
    # Collects every (sample, dim) prompt the chosen model would grade live,
    # submits them as one OpenAI Batch API job, and exits without grading.
    # Retrieve later with --retrieve-batch.
    if args.submit_batch:
        if not _check_api_keys(spec):
            return 2
        batch_model = args.batch_model or next(
            (m.model for m in spec.models if _is_openai_model(m.model)), None,
        )
        if batch_model is None:
            print("ERROR: --submit-batch found no OpenAI model in the spec's "
                  "ensemble; pass --batch-model explicitly.", file=sys.stderr)
            return 2
        from gdpval_qc.openai_batch import collect_batch_requests, submit_batch
        samples = [loader.load_sample(tid) for tid in selected_ids]
        rows = collect_batch_requests(
            spec, samples, batch_model,
            multimodal=not (args.no_multimodal or args.no_conversion),
        )
        if not rows:
            print("No LLM-graded (sample, dim) pairs matched — nothing to submit.")
            return 0
        manifest_path = (
            args.batch_manifest
            or Path(args.output_dir).expanduser().resolve() / "_batch_manifest.json"
        )
        batch_id = submit_batch(rows, manifest_path)
        print(f"\nSubmitted batch {batch_id} ({len(rows)} requests, model={batch_model!r}).")
        print(f"Manifest written to {manifest_path}.")
        print(f"Retrieve with: --retrieve-batch {batch_id} --batch-manifest {manifest_path}")
        return 0

    # ── Dry-run mode: just load and exit ──────────────────────────────────
    if args.dry_run:
        print("\nDRY-RUN: loading samples without any LLM calls…")
        for tid in selected_ids:
            sample = loader.load_sample(tid)
            print(f"  {tid}:")
            print(f"    category:        {sample.fields.get('category')!r}")
            print(f"    instruction len: {len(sample.fields.get('instruction', ''))}")
            print(f"    input_files:     {len(sample.fields.get('input_files') or [])}")
            print(f"    solution_files:  {len(sample.fields.get('solution_files') or [])}")
            print(f"    verifier_spec:   {'present' if sample.fields.get('verifier_spec_json') else 'absent'}")
        print("\nDry-run complete. Re-run without --dry-run to actually grade.")
        return 0

    # ── API keys ──────────────────────────────────────────────────────────
    if not _check_api_keys(spec):
        return 2

    # ── Build judge client factory ────────────────────────────────────────
    factory = _build_judge_client_factory(spec)
    synth_client = _build_synthesizer_client(spec)

    # Fresh per-run cost accounting (accumulator is process-wide).
    from gdpval_qc.llm_client import reset_accumulated_usage
    reset_accumulated_usage()

    # ── Batch API: retrieve mode ────────────────────────────────────────────
    # Resolves a batch submitted via --submit-batch into precomputed judge
    # attempts. Anything not resolved by --batch-max-wait is simply absent —
    # run_ensemble() falls back to a live call for it, so nothing is lost.
    precomputed_llm_attempts: dict[tuple[str, str], dict[str, Any]] = {}
    if args.retrieve_batch:
        manifest_path = (
            args.batch_manifest
            or Path(args.output_dir).expanduser().resolve() / "_batch_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        from gdpval_qc.openai_batch import fetch_batch_results, poll_batch
        print(f"\nPolling batch {args.retrieve_batch} "
              f"(max wait {args.batch_max_wait:.0f}s)…")
        status = poll_batch(args.retrieve_batch, max_wait_s=args.batch_max_wait)
        print(f"Batch status: {status}")
        precomputed_llm_attempts = fetch_batch_results(args.retrieve_batch, manifest)
        n_resolved = sum(len(v) for v in precomputed_llm_attempts.values())
        n_total = len(manifest.get("requests", {}))
        print(f"Resolved {n_resolved}/{n_total} requests from the batch; "
              f"the rest fall back to live calls.")

    # ── Engine ────────────────────────────────────────────────────────────
    from gdpval_qc.v18_engine import V18Engine
    progress_cb = _make_progress_callback(
        quiet=args.quiet,
        heartbeat_only=args.heartbeat_only,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    multimodal_enabled = not (args.no_multimodal or args.no_conversion)
    engine = V18Engine(
        spec=spec,
        judge_client_factory=factory,
        synthesizer_client=synth_client,
        dim_parallelism=args.dim_parallelism,
        sample_parallelism=args.sample_parallelism,
        progress_callback=progress_cb,
        multimodal=multimodal_enabled,
        precomputed_llm_attempts=precomputed_llm_attempts,
    )
    print(f"Parallelism: dim={args.dim_parallelism}, sample={args.sample_parallelism}, "
          f"judges-per-dim={len(spec.models)} (always parallel)")
    print(f"Multimodal:  {'enabled (PDFs sent as document blocks)' if multimodal_enabled else 'disabled (text-only judging)'}")
    print()

    # ── Run ───────────────────────────────────────────────────────────────
    print(f"\nGrading {len(selected_ids)} task(s)…")
    samples = []
    skipped: list[tuple[str, Exception]] = []
    for tid in selected_ids:
        try:
            samples.append(loader.load_sample(tid))
        except Exception as exc:  # noqa: BLE001 - a malformed sample (e.g. a
            # missing required input directory) must not crash the whole
            # batch and lose every other already-loadable sample's results.
            print(f"WARNING: skipping {tid!r} — failed to load: {exc}",
                  file=sys.stderr)
            skipped.append((tid, exc))
    if skipped:
        print(f"\n{len(skipped)} sample(s) skipped due to load errors "
              f"(excluded from grading, NOT silently treated as passing):")
        for tid, exc in skipped:
            print(f"  {tid}: {exc}")

    t0 = time.time()
    batch = engine.assess_batch(samples, run_calibration=not args.skip_calibration)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"\nVerdict distribution:")
    for v, n in sorted(batch.batch_summary.verdict_counts.items(),
                        key=lambda kv: kv[0]):
        pct = batch.batch_summary.verdict_percentages.get(v, 0.0)
        print(f"  {v:<22}  {n:>3}  ({pct:.1f}%)")
    print(f"\nJudge agreement rate:    {batch.batch_summary.judge_agreement_rate:.3f}")
    print(f"Samples with findings:   {batch.batch_summary.samples_with_findings} / {batch.batch_summary.sample_count}")

    if batch.batch_summary.failure_mode_histogram:
        print(f"\nTop failure modes:")
        for dim, n in sorted(batch.batch_summary.failure_mode_histogram.items(),
                              key=lambda kv: (-kv[1], kv[0]))[:10]:
            print(f"  {dim:<40}  {n}")

    # ── Emit artifacts ────────────────────────────────────────────────────
    from gdpval_qc.v18_emitter import collect_judge_errors, write_batch_artifacts
    out_dir = Path(args.output_dir).expanduser().resolve()
    paths = write_batch_artifacts(batch, out_dir,
                                   top_issues_cap=spec.synthesis.top_issues_cap)
    print(f"\nArtifacts written to {out_dir}:")
    for name, p in paths.items():
        print(f"  {name:<24} {p}")

    # ── Cost estimate (best-effort, via framework's per-model usage) ──────
    try:
        from gdpval_qc.llm_client import get_accumulated_usage
        usage = get_accumulated_usage()
        if usage:
            total_cost = sum(u.cost_usd for u in usage if hasattr(u, "cost_usd"))
            n_calls = len(usage)
            unknown = any(not getattr(u, "pricing_known", True) for u in usage)
            note = "  (some models lack pricing; cost is a lower bound)" if unknown else ""
            print(f"\nApprox cost (all judge calls, n={n_calls}): ${total_cost:.4f}{note}")
    except Exception as exc:
        print(f"\nWarning: unable to compute approximate cost: {exc}", file=sys.stderr)

    # ── Judge-error gate ─────────────────────────────────────────────────
    return judge_error_exit_code(batch, args.fail_on_judge_error_rate)


def judge_error_exit_code(batch, threshold: float) -> int:
    """Return process exit code given a graded BatchReview and a fail threshold.

    0 → run was clean (no errors, or below threshold).
    3 → at least one sample's judge-call error rate is ≥ threshold.

    Failed judge calls silently aggregate to ``applicable=False`` which makes
    broken runs look clean in the verdict mix. The artifacts always get
    written; the exit code is the signal for CI / automation.
    """
    from gdpval_qc.v18_emitter import collect_judge_errors
    errs = collect_judge_errors(batch)
    if errs["total_errors"] == 0:
        return 0

    print(f"\n⚠  Judge-call errors: {errs['total_errors']} / "
          f"{errs['total_attempts']} attempts "
          f"({errs['total_errors']/max(errs['total_attempts'],1):.1%})",
          file=sys.stderr)
    for reason, n in sorted(errs["by_reason"].items(),
                             key=lambda kv: (-kv[1], kv[0])):
        print(f"     {n}× {reason}", file=sys.stderr)

    if threshold > 1.0:
        return 0  # check disabled

    offenders = [
        (sid, counts["errors"] / max(counts["attempts"], 1))
        for sid, counts in errs["by_sample"].items()
        if counts["attempts"] > 0
        and counts["errors"] / counts["attempts"] >= threshold
    ]
    if not offenders:
        return 0

    print(
        f"\n✗ {len(offenders)} sample(s) hit judge-error rate "
        f"≥ {threshold:.0%}:",
        file=sys.stderr,
    )
    for sid, rate in sorted(offenders):
        print(f"     {sid:<24} {rate:.1%}", file=sys.stderr)
    print(
        "  Verdicts above may be wrong — check raw/sample_*.json "
        "transcripts before trusting them. Exiting with code 3.",
        file=sys.stderr,
    )
    return 3


# ── CLI parsing ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--spec", required=True, type=Path,
                   help="v18 quality spec markdown file")
    p.add_argument("--corpus", required=True, type=Path,
                   help="Task-corpus directory (each subdirectory is one task)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory to write batch_review.json + CSV + summary")
    p.add_argument("--limit", type=int, default=None,
                   help="Grade only the first N tasks (by sorted task ID). "
                        "Useful for first smoke run.")
    p.add_argument("--tasks", type=str, default=None,
                   help="Comma-separated explicit task IDs to grade "
                        "(overrides --limit)")
    p.add_argument("--dry-run", action="store_true",
                   help="Load samples but skip all LLM calls; print a summary "
                        "of what would be sent")
    p.add_argument("--skip-calibration", action="store_true",
                   help="Skip calibration goldens (faster smoke runs)")
    p.add_argument("--dim-parallelism", type=int, default=4,
                   help="Max dims graded concurrently per sample (default 4). "
                        "Each dim still parallelises judges internally. Lower to "
                        "1 to grade dims sequentially.")
    p.add_argument("--sample-parallelism", type=int, default=1,
                   help="Max samples graded concurrently within a batch "
                        "(default 1). Raise to 2-4 if API rate limits allow.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-event progress output on stderr.")
    p.add_argument("--heartbeat-only", action="store_true",
                   help="Replace the per-dim progress lines with a single "
                        "heartbeat line every N seconds (see --heartbeat-seconds).")
    p.add_argument("--heartbeat-seconds", type=int, default=30,
                   help="Heartbeat interval in seconds (default 30).")
    p.add_argument("--cache-dir", type=Path,
                   default=Path.home() / ".cache" / "gdpval-qc",
                   help="Directory for cached document conversions (default "
                        "~/.cache/gdpval-qc/). Office files convert to PDF "
                        "once and are reused across runs. Override via the "
                        "GDPVAL_QC_CACHE_DIR env var or this flag.")
    p.add_argument("--no-multimodal", action="store_true",
                   help="Disable multimodal: judges receive only text "
                        "extractions, never PDFs as document blocks. "
                        "Useful when judge models don't support PDF inputs.")
    p.add_argument("--fail-on-judge-error-rate", type=float, default=0.5,
                   help="Process exit with code 3 if ANY sample's judge-call "
                        "error rate (errors / total attempts) is >= this "
                        "threshold (default 0.5). Use a value > 1.0 to disable "
                        "the check. Artifacts are still written either way; "
                        "the exit code is the signal for CI / automation.")
    p.add_argument("--no-conversion", action="store_true",
                   help="Disable document conversion entirely (no PDF cache "
                        "produced). Implies --no-multimodal.")
    p.add_argument("--submit-batch", action="store_true",
                   help="Batch the OpenAI leg of the judge ensemble via the "
                        "OpenAI Batch API instead of grading live: collect "
                        "every (sample, dim) prompt for --batch-model, submit "
                        "as one job, write a manifest, print the batch_id, "
                        "and exit without grading. Retrieve later with "
                        "--retrieve-batch. Gemini/Anthropic judges are "
                        "unaffected — they always run live.")
    p.add_argument("--retrieve-batch", type=str, default=None, metavar="BATCH_ID",
                   help="Resume a batch submitted via --submit-batch: poll "
                        "BATCH_ID (bounded by --batch-max-wait), fetch "
                        "whatever completed, and run the normal grading pass "
                        "with those OpenAI judge calls pre-filled. Any "
                        "request not resolved in time falls back to a live "
                        "call automatically.")
    p.add_argument("--batch-model", type=str, default=None,
                   help="Which model to batch (must be one of the spec's "
                        "ensemble models). Defaults to the first OpenAI model "
                        "in --spec's ensemble.")
    p.add_argument("--batch-manifest", type=Path, default=None,
                   help="Path to the batch manifest JSON (custom_id -> "
                        "sample/dim/model, plus batch_id). Defaults to "
                        "<output-dir>/_batch_manifest.json.")
    p.add_argument("--batch-max-wait", type=float, default=1800.0,
                   help="Max seconds to poll a batch in --retrieve-batch "
                        "before giving up and falling back to live calls for "
                        "whatever hasn't completed (default 1800 = 30min).")
    return p.parse_args()


def _select_task_ids(all_ids: list[str], args: argparse.Namespace) -> list[str]:
    if args.tasks:
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [t for t in wanted if t not in all_ids]
        if missing:
            print(f"ERROR: requested tasks not in corpus: {missing}", file=sys.stderr)
            sys.exit(2)
        return wanted
    if args.limit is not None:
        return all_ids[:args.limit]
    return all_ids


# ── API key check ──────────────────────────────────────────────────────────

def _check_api_keys(spec) -> bool:
    needs_openai = any(_is_openai_model(m.model) for m in spec.models) \
                    or _is_openai_model(spec.synthesis.judge_model)
    needs_gemini = any(_is_gemini_model(m.model) for m in spec.models) \
                    or _is_gemini_model(spec.synthesis.judge_model)
    needs_anthropic = any(_is_anthropic_model(m.model) for m in spec.models) \
                       or _is_anthropic_model(spec.synthesis.judge_model)

    missing = []
    if needs_openai and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if needs_gemini and not (os.environ.get("GEMINI_API_KEY")
                              or os.environ.get("GOOGLE_API_KEY")):
        missing.append("GEMINI_API_KEY (or GOOGLE_API_KEY)")
    if needs_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        print(f"ERROR: missing env var(s): {missing}\n"
              f"Add them to your .env file or export them before running.",
              file=sys.stderr)
        return False
    return True


def _is_openai_model(name: str) -> bool:
    n = name.lower()
    return n.startswith("gpt-") or n.startswith("o1") or n.startswith("o3") or n.startswith("openai/")


def _is_gemini_model(name: str) -> bool:
    return name.lower().startswith("gemini")


def _is_anthropic_model(name: str) -> bool:
    return "claude" in name.lower()


# ── Client factory ─────────────────────────────────────────────────────────

def _build_judge_client_factory(spec) -> Callable[[str], object]:
    """Return a factory(model_name) → LLMClient using the right provider per model.

    Each provider's adapter is cached; the factory returns the cached instance
    for every call with the same model.
    """
    from gdpval_qc.openai_client import OpenAIClientAdapter
    from gdpval_qc.gemini_client import GeminiClientAdapter

    cache: dict[str, object] = {}

    def factory(model: str):
        if model in cache:
            return cache[model]
        if _is_openai_model(model):
            c = OpenAIClientAdapter()
        elif _is_gemini_model(model):
            c = GeminiClientAdapter()
        elif _is_anthropic_model(model):
            from gdpval_qc.llm_client import AnthropicClientAdapter
            c = AnthropicClientAdapter()
        else:
            raise ValueError(
                f"Unknown model provider for {model!r}. Recognised prefixes: "
                f"gpt-*, openai/*, gemini-*, *claude*."
            )
        cache[model] = c
        return c

    return factory


def _build_synthesizer_client(spec):
    """The synthesizer uses spec.synthesis.judge_model (gpt-5.5 by default)."""
    factory = _build_judge_client_factory(spec)
    return factory(spec.synthesis.judge_model)


if __name__ == "__main__":
    sys.exit(main())

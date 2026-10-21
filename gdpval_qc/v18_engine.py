"""V18 engine — drives the 8-stage per-sample pipeline.

Stages:
  1. Loader           Sample(task_dir, fields)
  2. Preconditions    for each dim, evaluate applies_when
  3. Judging          run ensemble per active dim → list[JudgeAttempt]
  4. Aggregation      per dim, attempts → DimensionFinding
  5. Verdict          deterministic from findings + spec
  6. Synthesizer      one LLM call per sample (or fallback to deterministic prose)
  7. BatchReview      assemble + compute batch summary
  8. Emit             written by the caller via v18_emitter

The engine is intentionally a thin driver. All the work happens in the
modules it calls; this file just sequences them.
"""

from __future__ import annotations

import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .finding import DimensionFinding
from .llm_client import LLMClient
from .precondition import Precondition
from .review import BatchReview, BatchSummary, SampleReview, Verdict
from .v18_aggregate import aggregate_attempts, make_skipped_finding
from .v18_callbacks import BUILTIN_CALLBACKS
from .v18_judge import (
    JudgeClientFactory, evaluate_applies_when, run_ensemble,
)
from .v18_loader import FileContent, Sample, TaskCorpusLoader
from .v18_spec import V18Dimension, V18Spec
from .v18_synthesizer import synthesize_sample_review

# Progress events emitted by the engine. Each is a dict with at minimum a
# 'kind' field; downstream callbacks can pattern-match on kind.
#
# Event kinds:
#   batch_start      → total_samples, total_dims_per_sample, ts
#   sample_start     → sample_id, total_dims, ts
#   dim_start        → sample_id, dim_name, check_kind, ts
#   dim_complete     → sample_id, dim_name, severity, applicable,
#                      elapsed_s, ts, completed_dims, total_dims
#   sample_synth     → sample_id, ts
#   sample_complete  → sample_id, verdict, top_issue_count, elapsed_s, ts,
#                      completed_samples, total_samples
#   batch_complete   → total_samples, elapsed_s, ts, verdict_counts
ProgressCallback = Callable[[dict[str, Any]], None]


# ── Programmatic-check registry (for v18) ──────────────────────────────────
# Programmatic dims declare a `callback:` name; the engine resolves them via
# this map. The four canonical v18 programmatic dims ship out-of-the-box.
# Callers can extend via the `programmatic_callbacks` parameter of V18Engine.

ProgrammaticCallback = Callable[[Sample, V18Dimension], DimensionFinding]

BUILTIN_PROGRAMMATIC_CALLBACKS: dict[str, ProgrammaticCallback] = dict(BUILTIN_CALLBACKS)


def register_programmatic_callback(name: str, fn: ProgrammaticCallback) -> None:
    BUILTIN_PROGRAMMATIC_CALLBACKS[name] = fn


def _collect_conversion_warnings(sample: Sample) -> list[str]:
    """Return one warning per FileContent whose document conversion failed.

    These files are judged from degraded fallback text (see
    ``v18_loader._attach_cached_path``); surfacing them on the SampleReview
    keeps the degradation visible to reviewers instead of only on stderr.
    """
    warnings: list[str] = []
    for fname, value in sample.fields.items():
        if not isinstance(value, list):
            continue
        for item in value:
            if (isinstance(item, FileContent)
                    and item.conversion_status == "failed"):
                reason = item.conversion_error or "unknown error"
                warnings.append(
                    f"{fname}: '{item.name}' could not be converted "
                    f"({reason}); judged from degraded text-only fallback."
                )
    return warnings


# ── Public API ─────────────────────────────────────────────────────────────

class V18Engine:
    """Orchestrate the per-sample pipeline.

    Construct with the spec, a judge-client factory, an (optional)
    synthesizer LLM client, and any callback registries. Then call
    `assess_batch(samples_or_loader)` to produce a BatchReview.

    Three parallelism dials (each defaults match v17 conventions where
    applicable):

      dim_parallelism      Max dims graded concurrently per sample.
                           Each dim still runs its own ensemble in parallel
                           (judge_parallelism handled inside run_ensemble).
                           Default 4. Set to 1 to grade dims sequentially.

      sample_parallelism   Max samples graded concurrently within a batch.
                           Default 1 — per-sample work is already token-heavy
                           because each sample drives ~21 LLM dims × 2 judges
                           plus a synthesizer call. Raise to 2-4 if your API
                           rate limits comfortably accommodate it.

      judge_parallelism    (Implicit) judges within one dim run in parallel
                           inside `run_ensemble`. No knob exposed here —
                           ensemble size is small (2 in v18).
    """

    def __init__(
        self,
        spec: V18Spec,
        judge_client_factory: JudgeClientFactory,
        synthesizer_client: LLMClient | None = None,
        programmatic_callbacks: dict[str, ProgrammaticCallback] | None = None,
        precondition_extras: dict[str, Precondition] | None = None,
        loader_callbacks: dict[str, Callable[[Sample], object]] | None = None,
        *,
        dim_parallelism: int = 4,
        sample_parallelism: int = 1,
        progress_callback: ProgressCallback | None = None,
        multimodal: bool = True,
        precomputed_llm_attempts: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> None:
        if dim_parallelism < 1:
            raise ValueError("dim_parallelism must be >= 1")
        if sample_parallelism < 1:
            raise ValueError("sample_parallelism must be >= 1")
        self.spec = spec
        self.judge_client_factory = judge_client_factory
        self.synthesizer_client = synthesizer_client
        self.programmatic_callbacks: dict[str, ProgrammaticCallback] = {
            **BUILTIN_PROGRAMMATIC_CALLBACKS,
            **(programmatic_callbacks or {}),
        }
        self.precondition_extras = precondition_extras or {}
        self.loader_callbacks = loader_callbacks or {}
        self.dim_parallelism = dim_parallelism
        self.sample_parallelism = sample_parallelism
        self.progress_callback = progress_callback
        self.multimodal = multimodal
        # (sample.task_id, dim.name) -> {model: JudgeAttempt}, resolved out-of-band
        # via the OpenAI Batch API (see openai_batch.py). Models not present for a
        # given (sample, dim) key — including every model when this is None — are
        # unaffected and still call live inside run_ensemble.
        self.precomputed_llm_attempts = precomputed_llm_attempts or {}

    def _emit(self, **event: Any) -> None:
        """Call progress_callback with the given event dict (timestamp added)."""
        if self.progress_callback is None:
            return
        event.setdefault("ts", time.time())
        try:
            self.progress_callback(event)
        except Exception:
            # Never let a buggy progress callback break the engine.
            pass

    # ── Per-dimension dispatch ─────────────────────────────────────────────

    def grade_dimension(
        self,
        dim: V18Dimension,
        sample: Sample,
    ) -> DimensionFinding:
        """Grade one dimension on one sample.

        Steps: applies_when → programmatic vs. LLM dispatch → aggregation.
        Returns a DimensionFinding regardless of outcome.
        """
        start = time.time()
        self._emit(kind="dim_start", sample_id=sample.task_id,
                   dim_name=dim.name, check_kind=dim.check_kind)

        if not evaluate_applies_when(dim, sample, extra=self.precondition_extras):
            reason_name = (
                dim.applies_when.builtin or dim.applies_when.callback
                if dim.applies_when else "no_precondition"
            )
            finding = make_skipped_finding(dim, f"{reason_name}=False")
            self._emit(kind="dim_complete", sample_id=sample.task_id,
                       dim_name=dim.name, severity=0, applicable=False,
                       elapsed_s=time.time() - start,
                       skipped_because=finding.skipped_because)
            return finding

        if dim.check_kind == "programmatic":
            finding = self._run_programmatic(dim, sample)
            self._emit(kind="dim_complete", sample_id=sample.task_id,
                       dim_name=dim.name, severity=finding.severity,
                       applicable=finding.applicable,
                       elapsed_s=time.time() - start)
            return finding

        # check_kind == "llm"
        attempts = run_ensemble(
            self.spec, dim, sample, self.judge_client_factory,
            parallel=True, multimodal=self.multimodal,
            precomputed=self.precomputed_llm_attempts.get((sample.task_id, dim.name)),
        )
        finding = aggregate_attempts(dim, attempts, self.spec)
        self._emit(kind="dim_complete", sample_id=sample.task_id,
                   dim_name=dim.name, severity=finding.severity,
                   applicable=finding.applicable,
                   elapsed_s=time.time() - start)
        return finding

    def _run_programmatic(
        self, dim: V18Dimension, sample: Sample,
    ) -> DimensionFinding:
        if not dim.callback:
            raise ValueError(
                f"Programmatic dim {dim.name!r} is missing `callback`"
            )
        fn = self.programmatic_callbacks.get(dim.callback)
        if fn is None:
            raise KeyError(
                f"Programmatic callback {dim.callback!r} not registered. "
                f"Pass it via `programmatic_callbacks` to V18Engine."
            )
        # A programmatic callback that declares a 3rd parameter opts in to
        # receiving the judge_client_factory, so it can make its own LLM
        # call(s) (e.g. a blind two-call extract-then-verdict decomposition)
        # without the engine needing any cross-dimension override plumbing.
        # 2-arg callbacks (the common case) are called exactly as before.
        import inspect
        if len(inspect.signature(fn).parameters) >= 3:
            return fn(sample, dim, self.judge_client_factory)
        return fn(sample, dim)

    # ── Per-sample pipeline ────────────────────────────────────────────────

    def assess_sample(
        self,
        sample: Sample,
        raw_trace_path: str | None = None,
    ) -> SampleReview:
        """Grade every dim, then synthesize. Dims grade in parallel up to
        `self.dim_parallelism`. Order in the returned `all_findings` matches
        the spec's dimension order regardless of completion order."""
        start = time.time()
        self._emit(kind="sample_start", sample_id=sample.task_id,
                   total_dims=len(self.spec.dimensions))
        findings = self._grade_all_dimensions(sample)
        self._emit(kind="sample_synth", sample_id=sample.task_id)
        review = synthesize_sample_review(
            sample.task_id, findings, self.spec, self.synthesizer_client,
            raw_trace_path=raw_trace_path,
        )
        # Carry input/solution file conversion failures into the reviewer-facing
        # output. The judge prompt already flags the degraded text inline; this
        # makes the same failure visible in batch_review.json / the summary so a
        # reviewer doesn't trust a verdict formed over content the judge never saw.
        review.data_quality_warnings = _collect_conversion_warnings(sample)
        self._emit(kind="sample_complete", sample_id=sample.task_id,
                   verdict=review.verdict,
                   top_issue_count=len(review.top_issues),
                   elapsed_s=time.time() - start)
        return review

    def _grade_all_dimensions(self, sample: Sample) -> list[DimensionFinding]:
        n = len(self.spec.dimensions)
        if self.dim_parallelism <= 1 or n <= 1:
            return [self.grade_dimension(d, sample) for d in self.spec.dimensions]

        # Parallel: submit all, collect by original index to preserve order
        findings: list[DimensionFinding | None] = [None] * n
        with ThreadPoolExecutor(max_workers=self.dim_parallelism,
                                  thread_name_prefix=f"v18-dim-{sample.task_id}") as pool:
            futs = {
                pool.submit(self.grade_dimension, d, sample): i
                for i, d in enumerate(self.spec.dimensions)
            }
            for f in futs:
                i = futs[f]
                findings[i] = f.result()
        return findings   # type: ignore[return-value]

    # ── Batch ──────────────────────────────────────────────────────────────

    def assess_batch(
        self,
        samples: list[Sample],
        *,
        run_calibration: bool = True,
    ) -> BatchReview:
        """Assess every sample and assemble a BatchReview.

        Samples grade concurrently up to `self.sample_parallelism`. Order
        in the returned `BatchReview.samples` matches input order.
        Calibration goldens (if any) are evaluated by default; pass
        `run_calibration=False` to skip them.
        """
        batch_start = time.time()
        self._emit(kind="batch_start",
                   total_samples=len(samples),
                   total_dims_per_sample=len(self.spec.dimensions))
        reviews = self._assess_samples_concurrently(samples)
        summary = self._build_batch_summary(reviews)
        calibration = None
        if run_calibration:
            from .v18_calibration import run_calibration as _run_calib
            calibration = _run_calib(self.spec, self)
        self._emit(kind="batch_complete",
                   total_samples=len(samples),
                   elapsed_s=time.time() - batch_start,
                   verdict_counts=dict(summary.verdict_counts))
        return BatchReview(
            spec_name=self.spec.name,
            spec_version=self.spec.version,
            samples=reviews,
            batch_summary=summary,
            calibration=calibration,
        )

    def _assess_samples_concurrently(
        self, samples: list[Sample],
    ) -> list[SampleReview]:
        n = len(samples)
        if self.sample_parallelism <= 1 or n <= 1:
            return [self._assess_sample_safe(s) for s in samples]

        reviews: list[SampleReview | None] = [None] * n
        with ThreadPoolExecutor(max_workers=self.sample_parallelism,
                                  thread_name_prefix="v18-sample") as pool:
            futs = {
                pool.submit(self._assess_sample_safe, s): i
                for i, s in enumerate(samples)
            }
            for f in futs:
                i = futs[f]
                reviews[i] = f.result()
        return reviews   # type: ignore[return-value]

    def _assess_sample_safe(self, sample: Sample) -> SampleReview:
        """`assess_sample`, but a failure never propagates past this point.

        The synthesizer already retries and falls back internally on its own
        LLM call (see `v18_synthesizer.synthesize_sample_review`); this is the
        outer backstop for anything else unexpected (e.g. a dimension-judging
        failure). One sample's crash must never cost the rest of the batch its
        results — a `Future.result()` re-raising an uncaught exception here
        used to take the whole `assess_batch()` down before any artifacts were
        written, silently discarding every already-completed sample.
        """
        try:
            return self.assess_sample(
                sample, raw_trace_path=f"raw/sample_{sample.task_id}.json",
            )
        except Exception as exc:  # noqa: BLE001 - last-resort batch backstop
            print(
                f"ERROR [{sample.task_id}]: unhandled exception during "
                f"assessment; degrading this sample to needs_review instead "
                f"of losing the batch.\n{traceback.format_exc()}",
                file=sys.stderr,
            )
            return SampleReview(
                sample_id=sample.task_id,
                verdict="needs_review",
                one_line=f"Assessment crashed for this sample: {exc}",
                review_required_because=f"engine error during assessment: {exc!r}",
                synthesizer_fell_back=True,
            )

    def assess_from_corpus(self, root_dir, task_ids: list[str] | None = None) -> BatchReview:
        """Convenience: build a TaskCorpusLoader from the spec, load samples, assess."""
        loader = TaskCorpusLoader(root_dir, self.spec, callbacks=self.loader_callbacks)
        ids = task_ids if task_ids is not None else loader.task_ids()
        samples = [loader.load_sample(tid) for tid in ids]
        return self.assess_batch(samples)

    # ── Internals ──────────────────────────────────────────────────────────

    def _build_batch_summary(self, reviews: list[SampleReview]) -> BatchSummary:
        verdict_counts: Counter[Verdict] = Counter()
        failure_modes: Counter[str] = Counter()
        samples_with_findings = 0
        agree_total = 0
        agree_count = 0

        for r in reviews:
            verdict_counts[r.verdict] += 1
            if any(f.severity >= 1 and f.gate_role != "informational"
                   for f in r.all_findings):
                samples_with_findings += 1
            for issue in r.top_issues:
                failure_modes[issue.dimension] += 1
            for f in r.all_findings:
                if not f.applicable:
                    continue
                agree_total += 1
                if f.judges_agreed:
                    agree_count += 1

        agree_rate = (agree_count / agree_total) if agree_total > 0 else 1.0
        total = sum(verdict_counts.values()) or 1
        pcts = {v: (n * 100.0 / total) for v, n in verdict_counts.items()}

        return BatchSummary(
            verdict_counts=dict(verdict_counts),
            verdict_percentages=pcts,
            failure_mode_histogram=dict(failure_modes),
            judge_agreement_rate=agree_rate,
            sample_count=len(reviews),
            samples_with_findings=samples_with_findings,
        )

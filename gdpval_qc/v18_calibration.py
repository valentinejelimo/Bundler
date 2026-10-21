"""Calibration goldens runner.

The spec carries a `calibration.goldens_path` pointing at a directory of
known-good samples with recorded expected verdicts. After a full assess run,
the calibration runner re-evaluates the goldens against the current spec +
judge ensemble and reports drift — goldens whose verdict no longer matches
what we previously recorded.

Layout under `goldens_path` (per spec):

    <goldens_path>/
        <golden_sample_id_1>/
            ... full task layout (task.toml, instruction.md, etc.)
            expected_verdict.txt    # single-line: one of the 5 verdict tiers
        <golden_sample_id_2>/
            ...

If `goldens_path` does not exist OR contains no goldens with
`expected_verdict.txt`, the runner emits a CalibrationReport with
`enabled=True, goldens_count=0, warning="..."` and returns. The framework
does NOT fail the run on missing goldens — calibration is opt-in.
"""

from __future__ import annotations

from pathlib import Path

from .review import CalibrationReport, Verdict, VERDICT_ORDER
from .v18_engine import V18Engine
from .v18_loader import TaskCorpusLoader
from .v18_spec import V18Spec


def run_calibration(spec: V18Spec, engine: V18Engine) -> CalibrationReport:
    """Run goldens against the current engine and report drift.

    Returns a CalibrationReport. Never raises — diagnostic problems become
    `warning` strings on the report.
    """
    if not spec.calibration.enabled:
        return CalibrationReport(
            enabled=False, goldens_count=0, drift=[],
            warning=None,
        )

    if not spec.calibration.goldens_path:
        return CalibrationReport(
            enabled=True, goldens_count=0, drift=[],
            warning="calibration.goldens_path is not set in the spec",
        )

    goldens_root = _resolve_goldens_path(spec)
    if not goldens_root.is_dir():
        return CalibrationReport(
            enabled=True, goldens_count=0, drift=[],
            warning=f"calibration.goldens_path does not exist: {goldens_root}",
        )

    goldens = _enumerate_goldens(goldens_root)
    if not goldens:
        return CalibrationReport(
            enabled=True, goldens_count=0, drift=[],
            warning=f"calibration.goldens_path is empty (no expected_verdict.txt found): {goldens_root}",
        )

    loader = TaskCorpusLoader(goldens_root, spec, callbacks=engine.loader_callbacks)

    drift: list[dict] = []
    for sample_id, expected in goldens:
        try:
            sample = loader.load_sample(sample_id)
            review = engine.assess_sample(sample)
            observed = review.verdict
        except Exception as exc:
            drift.append({
                "sample_id": sample_id,
                "expected":  expected,
                "observed":  "ERROR",
                "error":     f"{type(exc).__name__}: {exc}",
            })
            continue

        if observed != expected:
            drift.append({
                "sample_id": sample_id,
                "expected":  expected,
                "observed":  observed,
            })

    return CalibrationReport(
        enabled=True,
        goldens_count=len(goldens),
        drift=drift,
        warning=None,
    )


# ── Internals ──────────────────────────────────────────────────────────────

def _resolve_goldens_path(spec: V18Spec) -> Path:
    p = Path(spec.calibration.goldens_path)
    if not p.is_absolute() and spec.source_path is not None:
        # Resolve relative to the spec file's directory
        p = (spec.source_path.parent / p).resolve()
    return p.expanduser().resolve()


def _enumerate_goldens(goldens_root: Path) -> list[tuple[str, Verdict]]:
    """Return [(sample_id, expected_verdict)] for every golden in the dir.

    Goldens are immediate subdirectories that contain an `expected_verdict.txt`
    file with a single line naming one of the five verdict tiers.
    """
    out: list[tuple[str, Verdict]] = []
    for entry in sorted(goldens_root.iterdir()):
        if not entry.is_dir():
            continue
        ev_file = entry / "expected_verdict.txt"
        if not ev_file.is_file():
            continue
        line = ev_file.read_text(encoding="utf-8").strip()
        if line in VERDICT_ORDER:
            out.append((entry.name, line))  # type: ignore[arg-type]
    return out

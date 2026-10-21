"""Synthesise pass/fail patterns from a task's verifier-report.json files.

For tasks that have `evals/<model>/trial_<n>/verifier/verifier-report.json`
(or `.../reward.json`) outputs from earlier model runs, this module turns
the raw outcomes into a compact summary text that judge dimensions can read
as the `verifier_assertion_summary` field.

The summary surfaces the signals a quality reviewer would want when grading
verifier construct validity, outcome coverage, and discriminative power:

  - Per-verifier pass-rate matrix (model × pass-rate)
  - Aggregate patterns: always-passes, always-fails, discriminates,
    judge-unreliable
  - Per-model overall reward (from the verifier-report top-level `reward`)

Tolerant: when a task has no `evals/` dir, no reports, or unparseable
reports, the function returns an empty string. The dim's judges then
operate on whatever other inputs they have (spec text + solution_files).

Cache key for results is the set of (mtime, size) of all eval files
plus the task_dir path; we don't memoise yet because the summary is built
once per sample in `load_sample()` anyway.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .v18_loader import Sample

# Filenames the verifier framework writes per trial (in priority order).
_REPORT_CANDIDATES = (
    "verifier/verifier-report.json",
    "verifier/reward.json",
    "reward.json",
)


def build_verifier_assertion_summary(sample: Sample) -> str | None:
    """Return a compact summary of verifier outcomes across all model
    trials under `sample.task_dir/evals/`. Returns None when there's
    nothing meaningful to summarise (no evals dir, no parseable reports)."""
    evals_dir = sample.task_dir / "evals"
    if not evals_dir.is_dir():
        return None

    per_model_trials = _collect_reports(evals_dir)
    if not per_model_trials:
        return None

    return _format_summary(per_model_trials)


# Trials whose reward is exactly this are treated as harness/model crashes
# (empty output, infra failure) — excluded from reward stats so a crashing
# model can't fake a floor or mask headroom. (QC decision, 2026-07-06.)
_CRASH_REWARD = 0.0


def collect_reward_stats(sample: Sample) -> dict | None:
    """Structured per-model reward stats across all eval trials, shared by the
    programmatic run-based callbacks (headroom, capability inversion).

    Returns None when there is no eval data. Otherwise:
      {
        "models": {model: {"clean_mean": float|None, "n": int,
                           "n_crash": int, "clean": [float, ...]}},
        "best_model": str|None, "best_mean": float|None,
      }
    "clean" excludes crash trials (reward == 0.0); "clean_mean" is over clean
    trials only. Mirrors build_verifier_assertion_summary's collection so the
    text digest (for LLM gates) and the deterministic gates read the same runs.
    """
    evals_dir = sample.task_dir / "evals"
    if not evals_dir.is_dir():
        return None
    per_model = _collect_reports(evals_dir)
    if not per_model:
        return None

    models: dict[str, dict] = {}
    for model, reports in per_model.items():
        rewards = [r.get("reward") for r in reports
                   if isinstance(r.get("reward"), (int, float))]
        clean = [x for x in rewards if x > _CRASH_REWARD]
        models[model] = {
            "clean": clean,
            "clean_mean": (sum(clean) / len(clean)) if clean else None,
            "n": len(rewards),
            "n_crash": sum(1 for x in rewards if x == _CRASH_REWARD),
        }
    means = {m: v["clean_mean"] for m, v in models.items() if v["clean_mean"] is not None}
    best_model = max(means, key=means.get) if means else None
    return {
        "models": models,
        "best_model": best_model,
        "best_mean": means.get(best_model) if best_model else None,
    }


# ── Collection ─────────────────────────────────────────────────────────────

def _collect_reports(evals_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Walk evals/<model>/trial_*/ and return {model: [report_dict, ...]}."""
    out: dict[str, list[dict[str, Any]]] = {}
    for model_dir in sorted(p for p in evals_dir.iterdir() if p.is_dir()):
        reports: list[dict[str, Any]] = []
        for trial_dir in sorted(t for t in model_dir.iterdir()
                                  if t.is_dir() and t.name.startswith("trial_")):
            payload = _read_first_report(trial_dir)
            if payload is None:
                continue
            payload["_trial_name"] = trial_dir.name
            reports.append(payload)
        if reports:
            out[model_dir.name] = reports
    return out


def _read_first_report(trial_dir: Path) -> dict[str, Any] | None:
    """Return the first verifier report under ``trial_dir`` as a dict, else None.

    Reports are only treated as valid when they parse to a JSON object; arrays,
    scalars, and unparseable files are skipped so a malformed eval artifact
    (e.g. a report containing ``[]``) becomes a controlled absence rather than a
    ``TypeError`` later in ``_collect_reports``.

    Two on-disk layouts are supported:
      - Flat: ``<trial>/verifier/verifier-report.json`` (and the other candidates).
      - Harbor: the report is nested under
        ``<trial>/job/<timestamp>/<run-id>/verifier/verifier-report.json``. When
        the flat path is absent, fall back to a recursive glob and prefer the
        newest match by mtime so it tracks the latest trial execution.
    """
    def _load_object(p: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    # Flat layout.
    for rel in _REPORT_CANDIDATES:
        p = trial_dir / rel
        if p.is_file():
            obj = _load_object(p)
            if obj is not None:
                return obj

    # Harbor layout: recursively locate a nested report, newest first.
    for rel in _REPORT_CANDIDATES:
        candidates = sorted(
            (p for p in trial_dir.glob(f"**/{rel}") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in candidates:
            obj = _load_object(p)
            if obj is not None:
                return obj

    return None


# ── Formatting ─────────────────────────────────────────────────────────────

def _format_summary(per_model: dict[str, list[dict[str, Any]]]) -> str:
    models = sorted(per_model.keys())

    # Build the verifier-name → classification map as a UNION across every
    # report. All reports for a task should target the same verifier.json spec,
    # but an incomplete report (e.g. a failed trial that wrote only reward.json
    # or a partial `assertion_results`) would otherwise narrow the verifier set
    # and silently drop verifiers that the pass-rate loop below should visit.
    # First-seen insertion order is preserved; a concrete classification wins
    # over the "?" placeholder when reports disagree.
    name_to_class: dict[str, str] = {}
    for reports in per_model.values():
        for report in reports:
            for vname, cls in _classification_map(report).items():
                if vname not in name_to_class or name_to_class[vname] == "?":
                    name_to_class[vname] = cls
    verifier_names = list(name_to_class.keys())

    # Per (verifier, model) → list of bool outcomes across trials
    outcomes: dict[tuple[str, str], list[bool]] = defaultdict(list)
    scores:   dict[tuple[str, str], list[float | None]] = defaultdict(list)
    for model in models:
        for report in per_model[model]:
            for ar in report.get("assertion_results", []) or []:
                name = ar.get("name")
                if not name:
                    continue
                res = ar.get("result", {}) or {}
                outcomes[(name, model)].append(bool(res.get("success")))
                s = res.get("score")
                scores[(name, model)].append(float(s) if isinstance(s, (int, float)) else None)

    lines: list[str] = []
    lines.append("=== Verifier outcomes across model evaluations ===")
    lines.append(f"Models evaluated: {', '.join(models)}")
    n_trials_per_model = {m: len(per_model[m]) for m in models}
    lines.append("Trials per model: " + ", ".join(
        f"{m}={n}" for m, n in n_trials_per_model.items()
    ))
    lines.append("")

    # Per-verifier table
    lines.append("Per-verifier pass rates (passes / trials per model):")
    always_pass: list[str] = []
    always_fail: list[str] = []
    discriminates: list[str] = []
    judge_unreliable: list[str] = []

    for vname in verifier_names:
        cls = name_to_class.get(vname, "?")
        per_model_pass = []
        all_outcomes_flat: list[bool] = []
        per_model_outcomes: dict[str, list[bool]] = {}
        for m in models:
            o = outcomes.get((vname, m), [])
            if not o:
                continue
            per_model_outcomes[m] = o
            all_outcomes_flat.extend(o)
            per_model_pass.append(f"{m}={sum(o)}/{len(o)}")

        if not all_outcomes_flat:
            continue

        pattern = _classify_pattern(per_model_outcomes)
        if pattern == "always_passes":
            always_pass.append(vname)
        elif pattern == "always_fails":
            always_fail.append(vname)
        elif pattern == "judge_unreliable":
            judge_unreliable.append(vname)
        elif pattern == "discriminates":
            discriminates.append(vname)

        lines.append(
            f"  {vname:<46} [{cls:<16}] "
            f"{'  '.join(per_model_pass):<48} pattern: {pattern}"
        )

    lines.append("")
    lines.append("Aggregate patterns:")
    n_total = len([v for v in verifier_names if any(outcomes.get((v, m), []) for m in models)])

    def _row(label: str, names: list[str]) -> str:
        if not names:
            return f"  {label}: none"
        sample = names[:5]
        more = f", +{len(names) - 5} more" if len(names) > 5 else ""
        return f"  {label}: {len(names)}/{n_total}  ({', '.join(sample)}{more})"

    lines.append(_row("always-passes (low discriminative power)", always_pass))
    lines.append(_row("always-fails (possibly broken / too strict)", always_fail))
    lines.append(_row("discriminates (varies by model)", discriminates))
    lines.append(_row("judge-unreliable (varies within model trials)", judge_unreliable))

    # Per-model overall rewards
    lines.append("")
    lines.append("Per-model overall rewards (mean across trials):")
    for m in models:
        rewards = [r.get("reward") for r in per_model[m]
                   if isinstance(r.get("reward"), (int, float))]
        if rewards:
            mean = sum(rewards) / len(rewards)
            trials_str = " ".join(f"{r:.2f}" for r in rewards)
            lines.append(f"  {m:<30}  mean={mean:.3f}   per-trial=[{trials_str}]")
        else:
            lines.append(f"  {m:<30}  (no rewards reported)")

    return "\n".join(lines)


def _classification_map(report: dict[str, Any]) -> dict[str, str]:
    """Return {verifier_name: classification_label} from one report's assertion
    definitions. Handles two schemas:

      - GDPVal-216 schema: `metadata.classification` ("core" | "secondary")
      - Omega schema: `metadata.importance` ("core" | "secondary") +
        `metadata.grading_role` ("outcome" | "process")

    When both `importance` and `grading_role` are present we render them
    joined as "{importance}/{grading_role}" (e.g. "core/outcome") so the
    judge sees the full taxonomy.
    """
    out: dict[str, str] = {}
    for ar in report.get("assertion_results", []) or []:
        name = ar.get("name")
        if not name:
            continue
        meta = (ar.get("definition") or {}).get("metadata") or {}
        cls = meta.get("classification")
        if not cls:
            importance = meta.get("importance")
            grading_role = meta.get("grading_role")
            if importance and grading_role:
                cls = f"{importance}/{grading_role}"
            elif importance:
                cls = importance
            elif grading_role:
                cls = grading_role
        out[name] = str(cls) if cls else "?"
    return out


def _classify_pattern(per_model_outcomes: dict[str, list[bool]]) -> str:
    """Classify a single verifier's outcomes across models and trials.

    Returns one of:
      - "always_passes"     — pass on every trial of every model
      - "always_fails"      — fail on every trial of every model
      - "judge_unreliable"  — within at least one model, trials disagree
                              (only meaningful when n_trials >= 2 for that model)
      - "discriminates"     — different models produce different overall pass rates
      - "single_trial"      — fewer than 2 distinct trial outcomes; no further pattern
    """
    flat = [o for outs in per_model_outcomes.values() for o in outs]
    if not flat:
        return "no_data"
    if all(flat):
        return "always_passes"
    if not any(flat):
        return "always_fails"

    # Intra-model disagreement check
    for outs in per_model_outcomes.values():
        if len(outs) >= 2 and any(o != outs[0] for o in outs):
            return "judge_unreliable"

    # Inter-model variance: any model's pass-rate differs from another's
    per_model_rates = [sum(o) / len(o) for o in per_model_outcomes.values() if o]
    if max(per_model_rates) - min(per_model_rates) >= 0.5:
        return "discriminates"
    # Mild variation
    return "discriminates"

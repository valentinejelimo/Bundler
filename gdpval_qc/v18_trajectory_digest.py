"""Condense model *trajectories* into a text digest for trajectory-reliant QC.

The existing `v18_verifier_summary` digest reads only pass/fail outcomes. This
module reads the actual agent trajectories (`agent/trajectory.json`) AND their
verifier reports, and renders a compact side-by-side digest across all models
that ran a task. It feeds the `intermodel_trajectory_consistency` dimension (E),
whose job is the investigator-agent move from OpenAI's "signal vs noise" work:
look at HOW each model worked, not just its score, and flag when the reward
ordering contradicts the visible trajectory quality (a stronger model scoring
below a weaker one — construct-invalid reward) or when several capable models
fail the SAME core rubric identically (an over-strict / underspecified signal).

Design (mirrors v18_verifier_summary):
  - Returns a STRING (rendered digest) or None; tolerant of missing/broken data.
  - String output means it renders through the judge prompt unchanged — no
    engine/judge changes beyond a `_FIELD_CHAR_LIMITS` entry.

Bulk control: agent/trajectory.json is 300-600 KB, almost all of it tool-output
(`observation.results[].content`). We DROP that content (keep only a size
marker), keep the tool-call sequence + a per-model representative trajectory, and
aggregate rubric outcomes across the k trials. Result is tens of KB, not MB.

Trajectory facts this handles (verified against the k=10/6-model drop):
  - ATIF schema + tool function-names differ per agent (codex exec_command,
    gemini run_shell_command/write_file, opus …) — we are name-agnostic.
  - Per-rubric judge reasoning lives in
    assertion_results[].result.evaluation.attempts[].justification.
  - Deliverables land in <bundle>/artifacts/output/.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .v18_loader import Sample, _extract_text
from .v18_verifier_summary import _CRASH_REWARD, _classification_map

# ── Budget knobs ────────────────────────────────────────────────────────────
_ARG_SNIPPET_CHARS = 120      # per tool-call argument snippet
_ACTIONS_HEAD = 20            # first N action lines kept per model
_ACTIONS_TAIL = 5             # last N action lines kept per model
_FINAL_MSG_CHARS = 500        # final agent message per model
_JUSTIFICATION_CHARS = 240    # per-rubric representative reasoning
# Argument keys that best summarise a tool call, in priority order.
_ARG_KEYS = ("cmd", "command", "file_path", "path", "name", "query",
             "content", "topic", "input")


# ── Public entry point ──────────────────────────────────────────────────────

def build_intermodel_trajectory_digest(sample: Sample) -> str | None:
    """One string comparing every model's condensed trajectory + reward +
    per-rubric outcomes side by side. None when fewer than 2 models have a
    parseable trajectory (E needs a cross-model comparison)."""
    evals_dir = sample.task_dir / "evals"
    if not evals_dir.is_dir():
        return None

    per_model = _iter_trial_bundles(evals_dir)
    if not per_model:
        return None

    # Build per-model summaries (reward stats + representative trajectory).
    model_blocks: dict[str, dict] = {}
    for model, bundles in per_model.items():
        block = _summarise_model(model, bundles)
        if block is not None:
            model_blocks[model] = block

    n_with_traj = sum(1 for b in model_blocks.values() if b.get("trajectory"))
    if n_with_traj < 2:
        return None

    return _format_digest(sample.task_id, model_blocks, per_model)


# ── Trial / bundle discovery ────────────────────────────────────────────────

def _iter_trial_bundles(evals_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """{model: [(trial_name, bundle_dir), ...]}. A bundle_dir contains both
    agent/ and verifier/. Robust to the Harbor job/<ts>/<run-id>/ nesting and
    to duplicated trial_N/trial_N trees (glob + newest-mtime dedupe)."""
    out: dict[str, list[tuple[str, Path]]] = {}
    for model_dir in sorted(p for p in evals_dir.iterdir() if p.is_dir()):
        trials: list[tuple[str, Path]] = []
        for trial_dir in sorted(
            (t for t in model_dir.iterdir()
             if t.is_dir() and t.name.startswith("trial_")),
            key=lambda p: _trial_sort_key(p.name),
        ):
            bundle = _find_bundle(trial_dir)
            if bundle is not None:
                trials.append((trial_dir.name, bundle))
        if trials:
            out[model_dir.name] = trials
    return out


def _trial_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name.split("_", 1)[1]), name)
    except (IndexError, ValueError):
        return (10**9, name)


def _find_bundle(trial_dir: Path) -> Path | None:
    """Return the dir holding verifier/verifier-report.json (newest by mtime)."""
    candidates = sorted(
        (p for p in trial_dir.glob("**/verifier/verifier-report.json") if p.is_file()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    for report in candidates:
        return report.parent.parent      # <bundle>/verifier/verifier-report.json
    # Fall back to a bundle that at least has a trajectory.
    for tj in sorted(trial_dir.glob("**/agent/trajectory.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True):
        return tj.parent.parent
    return None


# ── Per-model summarisation ─────────────────────────────────────────────────

def _summarise_model(model: str, bundles: list[tuple[str, Path]]) -> dict | None:
    """Aggregate a model's k trials: reward stats + a representative condensed
    trajectory (the clean trial whose reward is closest to the model mean)."""
    trials: list[dict] = []
    for trial_name, bundle in bundles:
        report = _load_json(bundle / "verifier" / "verifier-report.json")
        reward = report.get("reward") if isinstance(report, dict) else None
        trials.append({
            "trial": trial_name, "bundle": bundle,
            "reward": reward if isinstance(reward, (int, float)) else None,
            "report": report,
        })
    rewards = [t["reward"] for t in trials if isinstance(t["reward"], (int, float))]
    clean = [r for r in rewards if r > _CRASH_REWARD]
    mean = (sum(clean) / len(clean)) if clean else None

    # Representative = clean trial closest to the mean; else the first trial.
    rep = None
    if mean is not None:
        rep = min((t for t in trials if isinstance(t["reward"], (int, float))
                   and t["reward"] > _CRASH_REWARD),
                  key=lambda t: abs(t["reward"] - mean), default=None)
    if rep is None:
        rep = trials[0] if trials else None

    traj = _load_condensed_trajectory(rep["bundle"]) if rep else None

    return {
        "model": model,
        "n_trials": len(trials),
        "n_crash": sum(1 for r in rewards if r == _CRASH_REWARD),
        "rewards": rewards,
        "mean": mean,
        "best": max(clean) if clean else None,
        "worst": min(clean) if clean else None,
        "rep_trial": rep["trial"] if rep else None,
        "rep_reward": rep["reward"] if rep else None,
        "trajectory": traj,
        "trials": trials,          # kept for rubric aggregation
    }


def _load_condensed_trajectory(bundle: Path) -> dict | None:
    """Parse agent/trajectory.json into a compact dict. DROP observation bulk."""
    data = _load_json(bundle / "agent" / "trajectory.json")
    if not isinstance(data, dict):
        return None
    steps = data.get("steps") or []
    agent = data.get("agent") or {}
    fm = data.get("final_metrics") or {}

    tool_hist: dict[str, int] = defaultdict(int)
    actions: list[str] = []
    last_agent_msg = ""
    for s in steps:
        if s.get("source") == "agent" and isinstance(s.get("message"), str) and s["message"].strip():
            last_agent_msg = s["message"]
        for call in (s.get("tool_calls") or []):
            fn = call.get("function_name") or "?"
            tool_hist[fn] += 1
            snippet = _arg_snippet(call.get("arguments"))
            out_chars = _observation_size(s.get("observation"))
            actions.append(f"{fn}: {snippet}"
                           + (f"  → {out_chars} chars out" if out_chars else ""))

    action_lines = _elide(actions, _ACTIONS_HEAD, _ACTIONS_TAIL)

    return {
        "agent_name": agent.get("name"),
        "model_name": agent.get("model_name"),
        "schema": data.get("schema_version"),
        "n_steps": len(steps),
        "total_tokens": fm.get("total_tokens") or (fm.get("extra") or {}).get("total_tokens"),
        "cost_usd": fm.get("total_cost_usd"),
        "tool_hist": dict(tool_hist),
        "actions": action_lines,
        "final_message": _clip(last_agent_msg, _FINAL_MSG_CHARS),
        "deliverables": _list_deliverables(bundle),
    }


def _arg_snippet(args: Any) -> str:
    if isinstance(args, dict):
        for k in _ARG_KEYS:
            if k in args and isinstance(args[k], (str, int, float)):
                return _clip(str(args[k]).replace("\n", " "), _ARG_SNIPPET_CHARS)
        return _clip(json.dumps(args, ensure_ascii=False).replace("\n", " "),
                     _ARG_SNIPPET_CHARS)
    if args is None:
        return ""
    return _clip(str(args).replace("\n", " "), _ARG_SNIPPET_CHARS)


def _observation_size(obs: Any) -> int:
    if not isinstance(obs, dict):
        return 0
    total = 0
    for r in (obs.get("results") or []):
        if isinstance(r, dict):
            total += len(str(r.get("content", "")))
    return total


def _list_deliverables(bundle: Path) -> list[str]:
    out_dir = bundle / "artifacts" / "output"
    if not out_dir.is_dir():
        return []
    return sorted(p.name for p in out_dir.iterdir() if p.is_file())


# ── Rubric aggregation across trials ────────────────────────────────────────

def _aggregate_rubrics(per_model: dict[str, list[tuple[str, Path]]],
                       model_blocks: dict[str, dict]) -> tuple[list[str], dict, dict, dict]:
    """Return (ordered_rubric_names, meta, outcomes, justification).

    outcomes[(vname, model)]  = [bool, ...] across trials
    meta[vname]               = {importance, role, weight, tag}
    justification[vname]      = (model, passed, text) — one representative,
                                preferring a FAILED attempt (more informative).
    """
    outcomes: dict[tuple[str, str], list[bool]] = defaultdict(list)
    meta: dict[str, dict] = {}
    justification: dict[str, tuple[str, bool, str]] = {}
    order: list[str] = []

    for model, block in model_blocks.items():
        for t in block["trials"]:
            report = t.get("report")
            if not isinstance(report, dict):
                continue
            if not meta:
                # taxonomy map from the first good report
                for ar in report.get("assertion_results", []) or []:
                    name = ar.get("name")
                    if not name:
                        continue
                    m = (ar.get("definition") or {}).get("metadata") or {}
                    if name not in meta:
                        meta[name] = {
                            "importance": m.get("importance") or m.get("classification") or "?",
                            "role": m.get("grading_role") or "?",
                            "weight": m.get("weight"),
                            "tag": m.get("tag") or "",
                        }
                        order.append(name)
            for ar in report.get("assertion_results", []) or []:
                name = ar.get("name")
                if not name:
                    continue
                res = ar.get("result", {}) or {}
                success = bool(res.get("success"))
                outcomes[(name, model)].append(success)
                if name not in meta:
                    m = (ar.get("definition") or {}).get("metadata") or {}
                    meta[name] = {
                        "importance": m.get("importance") or m.get("classification") or "?",
                        "role": m.get("grading_role") or "?",
                        "weight": m.get("weight"), "tag": m.get("tag") or "",
                    }
                    order.append(name)
                # Capture a representative justification (prefer a failure).
                text = _extract_justification(res)
                if text and (name not in justification or
                             (not success and justification[name][1])):
                    justification[name] = (model, success, _clip(text, _JUSTIFICATION_CHARS))

    # Sort: core first, then by weight desc, then name.
    def _key(n: str) -> tuple:
        m = meta.get(n, {})
        core = 0 if str(m.get("importance", "")).startswith("core") else 1
        w = m.get("weight") if isinstance(m.get("weight"), (int, float)) else 0.0
        return (core, -w, n)

    order = sorted(set(order) | {n for (n, _m) in outcomes}, key=_key)
    return order, meta, outcomes, justification


def _extract_justification(res: dict) -> str:
    ev = res.get("evaluation") or {}
    for att in (ev.get("attempts") or []):
        j = att.get("justification")
        if isinstance(j, str) and j.strip():
            return j.strip().replace("\n", " ")
    for k in ("reason", "rationale"):
        if isinstance(res.get(k), str) and res[k].strip():
            return res[k].strip().replace("\n", " ")
    return ""


# ── Formatting ──────────────────────────────────────────────────────────────

def _format_digest(task_id: str, model_blocks: dict[str, dict],
                   per_model: dict[str, list[tuple[str, Path]]]) -> str:
    models = sorted(model_blocks.keys())
    lines: list[str] = []
    lines.append("=== Intermodel trajectory consistency digest ===")
    lines.append(f"Task: {task_id}   Models: {', '.join(models)}")
    lines.append("")

    # Reward ordering up front — the headline signal.
    ranked = sorted(
        (m for m in models if model_blocks[m]["mean"] is not None),
        key=lambda m: model_blocks[m]["mean"], reverse=True,
    )
    lines.append("Mean reward ranking (clean trials, high → low):")
    for m in ranked:
        b = model_blocks[m]
        lines.append(f"  {m:<44} mean={b['mean']:.3f}  "
                     f"[best={_fmt(b['best'])} worst={_fmt(b['worst'])} "
                     f"n={b['n_trials']} crash={b['n_crash']}]")
    lines.append("")

    # Per-model condensed trajectory.
    for m in models:
        b = model_blocks[m]
        t = b.get("trajectory")
        lines.append(f"--- {m}  (mean reward {_fmt(b['mean'])}) ---")
        if not t:
            lines.append("  [no parseable trajectory]")
            lines.append("")
            continue
        lines.append(f"  agent={t['agent_name']} ({t['schema']})  steps={t['n_steps']}"
                     f"  tokens={t['total_tokens']}  cost={_fmt_cost(t['cost_usd'])}"
                     f"  rep_trial={b['rep_trial']}(reward={_fmt(b['rep_reward'])})")
        if t["tool_hist"]:
            hist = " ".join(f"{k}×{v}" for k, v in
                            sorted(t["tool_hist"].items(), key=lambda kv: -kv[1]))
            lines.append(f"  tools: {hist}")
        if t["deliverables"]:
            lines.append(f"  deliverables: {', '.join(t['deliverables'])}")
        lines.append("  actions:")
        for i, a in enumerate(t["actions"]):
            lines.append(f"    {a}")
        if t["final_message"]:
            lines.append(f"  final message: \"{t['final_message']}\"")
        lines.append("")

    # Per-rubric outcomes across trials.
    order, meta, outcomes, justification = _aggregate_rubrics(per_model, model_blocks)
    lines.append("--- Per-rubric outcomes (passes/trials per model), core first ---")
    for vname in order:
        m_meta = meta.get(vname, {})
        w = m_meta.get("weight")
        w_str = f"w={w:.3f}" if isinstance(w, (int, float)) else "w=?"
        head = (f"[{vname}] {m_meta.get('tag','')} "
                f"({m_meta.get('importance','?')}/{m_meta.get('role','?')}, {w_str})")
        lines.append(head)
        cells = []
        for m in models:
            o = outcomes.get((vname, m))
            if o:
                cells.append(f"{m.split('__')[-1]}={sum(o)}/{len(o)}")
        lines.append("   " + "  ".join(cells))
        if vname in justification:
            jm, jok, jt = justification[vname]
            lines.append(f"   e.g. {jm.split('__')[-1]} "
                         f"{'PASS' if jok else 'FAIL'}: \"{jt}\"")
    return "\n".join(lines)


# ── Single-trial deliverable digests (feed A: over-strict, C: leniency) ──────

_DELIVERABLE_CHARS = 25_000   # total extracted deliverable text budget (head+tail)


def build_failed_trial_deliverable_digest(sample: Sample) -> str | None:
    """The lowest-reward non-crash trial: its actual deliverable text + full
    per-rubric outcomes. Feeds `verifier_rubric_construct_validity` (A) with a
    concrete failed answer, so the judge can tell an over-strict rubric (a
    defensible answer marked wrong) from a genuinely weak one."""
    return _build_extreme_deliverable_digest(sample, pick="min",
                                             lens="verifier over-strictness")


def build_passed_trial_deliverable_digest(sample: Sample) -> str | None:
    """The highest-reward non-crash trial: its actual deliverable text + full
    per-rubric outcomes. Feeds `verifier_discriminative_power` (C), so the judge
    can tell whether the top-scoring answer is genuinely strong or a thin
    solution the verifier waved through (leniency)."""
    return _build_extreme_deliverable_digest(sample, pick="max",
                                             lens="verifier leniency")


def _build_extreme_deliverable_digest(sample: Sample, *, pick: str,
                                      lens: str) -> str | None:
    evals_dir = sample.task_dir / "evals"
    if not evals_dir.is_dir():
        return None
    trials = _all_trials(_iter_trial_bundles(evals_dir))
    scored = [t for t in trials
              if isinstance(t["reward"], (int, float)) and t["reward"] > _CRASH_REWARD]
    if not scored:
        return None
    chosen = (min if pick == "min" else max)(scored, key=lambda t: t["reward"])
    # Only useful if the chosen trial actually has deliverable files to read;
    # otherwise (e.g. a slimmed eval tree with verifier/ but no artifacts/output)
    # return None so A/C fall back to their prior evidence instead of seeing a
    # misleading "[no deliverable files found]" digest.
    out_dir = chosen["bundle"] / "artifacts" / "output"
    if not (out_dir.is_dir() and any(p.is_file() for p in out_dir.iterdir())):
        return None
    return _format_deliverable_digest(sample.task_id, chosen, lens)


def _all_trials(per_model: dict[str, list[tuple[str, Path]]]) -> list[dict]:
    out: list[dict] = []
    for model, bundles in per_model.items():
        for trial_name, bundle in bundles:
            report = _load_json(bundle / "verifier" / "verifier-report.json")
            reward = report.get("reward") if isinstance(report, dict) else None
            out.append({
                "model": model, "trial": trial_name, "bundle": bundle,
                "reward": reward if isinstance(reward, (int, float)) else None,
                "report": report,
            })
    return out


def _format_deliverable_digest(task_id: str, trial: dict, lens: str) -> str:
    lines: list[str] = []
    label = "LOWEST-reward" if lens.endswith("strictness") else "HIGHEST-reward"
    lines.append(f"=== {label} non-crash trial exemplar ({lens} lens) ===")
    lines.append(f"Task: {task_id}   Trial: {trial['model']} / {trial['trial']}   "
                 f"reward={_fmt(trial['reward'])}")
    lines.append("")

    # Deliverable text (head+tail truncated to a shared budget).
    lines.append("--- Deliverable files (artifacts/output/) ---")
    out_dir = trial["bundle"] / "artifacts" / "output"
    files = sorted(p for p in out_dir.iterdir() if p.is_file()) if out_dir.is_dir() else []
    if not files:
        lines.append("  [no deliverable files found]")
    else:
        per_file = max(2000, _DELIVERABLE_CHARS // max(1, len(files)))
        for p in files:
            lines.append(f"[file: {p.name}]")
            lines.append(_head_tail(_extract_text(p), per_file))
            lines.append("")

    # Full per-rubric outcomes for THIS trial.
    report = trial.get("report") or {}
    lines.append("--- Verifier outcome for this trial, per rubric ---")
    for ar in report.get("assertion_results", []) or []:
        name = ar.get("name")
        if not name:
            continue
        meta = (ar.get("definition") or {}).get("metadata") or {}
        res = ar.get("result", {}) or {}
        w = meta.get("weight")
        w_str = f"w={w:.3f}" if isinstance(w, (int, float)) else "w=?"
        imp = meta.get("importance") or meta.get("classification") or "?"
        role = meta.get("grading_role") or "?"
        ok = "PASS" if res.get("success") else "FAIL"
        lines.append(f"[{name}] {meta.get('tag','')} ({imp}/{role}, {w_str})  {ok}")
        just = _extract_justification(res)
        if just:
            lines.append(f"   {_clip(just, _JUSTIFICATION_CHARS)}")
    return "\n".join(lines)


def _head_tail(s: str, budget: int) -> str:
    s = s.strip()
    if len(s) <= budget:
        return s
    half = budget // 2
    return s[:half] + f"\n…[{len(s) - budget} chars elided]…\n" + s[-half:]


# ── Small helpers ───────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "…"


def _elide(items: list[str], head: int, tail: int) -> list[str]:
    if len(items) <= head + tail:
        return [f"#{i:02d} {a}" for i, a in enumerate(items)]
    kept = [f"#{i:02d} {a}" for i, a in enumerate(items[:head])]
    kept.append(f"…[{len(items) - head - tail} steps elided]…")
    start = len(items) - tail
    kept += [f"#{i:02d} {a}" for i, a in enumerate(items[start:], start=start)]
    return kept


def _fmt(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


def _fmt_cost(x: Any) -> str:
    return f"${x:.2f}" if isinstance(x, (int, float)) else "n/a"

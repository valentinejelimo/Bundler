#!/usr/bin/env python3
"""Hard-cutover v3 Harbor task verifier entrypoint."""

import json
import os
import sys
import logging
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from verifier_lib.runner import (  # noqa: E402
    build_failure_payload,
    run_verifier as _run_verifier,
    write_outputs,
)


TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
VERIFIER_DIR = Path(os.environ.get("VERIFIER_DIR", "/logs/verifier"))
WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
AGENT_LOGS_DIR = Path(os.environ.get("AGENT_LOGS_DIR", "/logs/agent"))
VERIFIER_JSON = TESTS_DIR / "verifier.json"
MAX_CONTENT_CHARS = int(os.environ.get("VERIFIER_MAX_CONTENT", "90000"))
# This genre keeps its `formulas`-engine recompute of formula cells (pinned in verifier_lib/requirements.txt).
# The shared library only runs it when asked, so the rewardkit genre is unaffected.
os.environ.setdefault("VERIFIER_XLSX_FORMULAS_RECOMPUTE", "1")


def configure_logging() -> None:
    """Configures timestamped runtime logging for the verifier process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    for logger_name in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def write_reward_details(payload: dict, verifier_dir: Path) -> None:
    """Emit reward-kit ``reward-details.json`` alongside reward.json/reward.txt.

    Harbor's runtime grades from reward.json, but the platform surfaces the
    per-check breakdown ("tests") from the reward-kit file
    ``reward-details.json``. Without it the verifier panel renders empty and the
    run reads as "no tests ran". This translates each verifier in our
    verifier-report payload into one reward-kit entry (name -> reward) with a
    single scored criterion, so every verifier shows as a scored test — even
    when a judge errors (it then shows as an errored test rather than nothing).

    Schema (from harbor's viewer): dict name -> {kind, score, criteria[],
    judge?{model}, warnings?[]}; criterion = {name, description, weight, value,
    raw, reasoning, error}.
    """
    details: dict[str, dict] = {}
    for r in payload.get("assertion_results", []) or []:
        name = r.get("name") or f"verifier_{len(details) + 1}"
        res = r.get("result", {}) or {}
        meta = (r.get("definition", {}) or {}).get("metadata", {}) or {}
        evaluation = res.get("evaluation", {}) or {}
        success = bool(res.get("success"))
        value = 1.0 if success else 0.0
        weight = meta.get("weight")
        if not isinstance(weight, (int, float)):
            weight = res.get("effective_weight", 1)
        judge_models = evaluation.get("judge_models") or []
        attempts = evaluation.get("attempts") or []
        reasoning = next((a.get("justification") for a in attempts if a.get("justification")), None)
        reasoning = reasoning or res.get("reason")
        entry: dict = {
            "kind": meta.get("grading_role") or "rubric",
            "score": value,
            "criteria": [
                {
                    "name": meta.get("tag") or name,
                    "description": meta.get("how_justification"),
                    "weight": weight,
                    "value": value,
                    "raw": value,
                    "reasoning": reasoning,
                    "error": res.get("error"),
                }
            ],
        }
        if judge_models:
            entry["judge"] = {"model": judge_models[0]}
        if res.get("error"):
            entry["warnings"] = [res["error"]]
        details[name] = entry
    (verifier_dir / "reward-details.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def main() -> int:
    """Runs the Harbor verifier entrypoint.

    Returns:
        Process exit code. Zero means grading completed and reward artifacts
        were written; non-zero means verifier infrastructure failed.
    """
    configure_logging()
    logger = logging.getLogger(__name__)

    if not VERIFIER_JSON.exists():
        message = f"{VERIFIER_JSON} not found"
        logger.error(message)
        write_outputs(build_failure_payload(message), VERIFIER_DIR)
        return 1
    try:
        payload = _run_verifier(
            VERIFIER_JSON,
            workspace_dir=WORKSPACE_DIR,
            verifier_dir=VERIFIER_DIR,
            agent_logs_dir=AGENT_LOGS_DIR,
            max_content_chars=MAX_CONTENT_CHARS,
        )
    except Exception as exc:
        logger.exception("Verifier failed")
        if not (VERIFIER_DIR / "verifier-report.json").exists():
            write_outputs(build_failure_payload(str(exc)), VERIFIER_DIR)
        return 1
    try:
        write_reward_details(payload, VERIFIER_DIR)
    except Exception:
        logger.exception("Failed to write reward-details.json (grading unaffected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import VerifierSpec, effective_weights
from .sources.registry import SourceRegistry
from .source_types import SourceAuthoringError
from .verifiers import VerifierInfrastructureError, result_entry, verify_definition


logger = logging.getLogger(__name__)
REWARD_DECIMAL_PLACES = 3


def load_spec(path: Path) -> VerifierSpec:
    """Loads and validates a verifier spec.

    Args:
        path: Path to verifier.json.

    Returns:
        Parsed verifier specification.

    Raises:
        ValueError: If the JSON content does not match the v3 schema.
    """
    try:
        return VerifierSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ValueError(f"invalid verifier schema: {exc}") from exc


def run_verifier(
    spec_path: Path,
    *,
    workspace_dir: Path,
    verifier_dir: Path,
    agent_logs_dir: Path = Path("/logs/agent"),
    llm_completion: Callable[..., Any] | None = None,
    max_content_chars: int = 90000,
) -> dict[str, Any]:
    """Runs all verifiers from a spec and writes output artifacts.

    Args:
        spec_path: Path to verifier.json.
        workspace_dir: Workspace containing task output artifacts.
        verifier_dir: Directory where verifier artifacts should be written.
        agent_logs_dir: Harbor agent log artifact directory.
        llm_completion: Optional LiteLLM-compatible completion function for
            tests.
        max_content_chars: Maximum text content returned by file sources.

    Returns:
        The verifier report payload written to verifier-report.json.

    Raises:
        ValueError: If the spec is invalid.
        RuntimeError: If verifier infrastructure cannot produce trustworthy
            grading.
    """
    results: list[dict[str, Any]] = []
    try:
        spec = load_spec(spec_path)
        registry = SourceRegistry(
            workspace_dir,
            agent_logs_dir=agent_logs_dir,
            max_content_chars=max_content_chars,
        )
        weights = effective_weights(spec.verifiers)
        for definition in spec.verifiers:
            try:
                results.append(
                    verify_definition(
                        definition,
                        registry,
                        weights[definition.name],
                        config=spec.config,
                        completion_fn=llm_completion,
                    )
                )
            except SourceAuthoringError as exc:
                # A source command that cannot resolve what the criterion names (a pinned sheet
                # the submission called something else, an unknown argument) fails THAT criterion.
                # It must not discard every other criterion's verdict by zeroing the whole task.
                results.append(
                    result_entry(
                        definition,
                        success=False,
                        expected=definition.assertion.expected,
                        actual=None,
                        effective_weight=weights[definition.name],
                        error=f"source authoring error: {exc}",
                        reason="source_authoring_error",
                        evaluation={},
                        source_data=None,
                    )
                )
    except VerifierInfrastructureError as exc:
        if exc.partial_result is not None:
            results.append(exc.partial_result)
        payload = build_payload(spec, results, fatal_error=str(exc))
        write_outputs(payload, verifier_dir)
        raise
    except Exception as exc:
        payload = (
            build_payload(spec, results, fatal_error=str(exc))
            if "spec" in locals()
            else build_failure_payload(str(exc))
        )
        write_outputs(payload, verifier_dir)
        raise

    payload = build_payload(spec, results)
    write_outputs(payload, verifier_dir)
    print_summary(payload)
    return payload


def build_payload(
    spec: VerifierSpec,
    results: list[dict[str, Any]],
    fatal_error: str | None = None,
) -> dict[str, Any]:
    """Builds the verifier-owned report payload.

    Args:
        spec: Parsed verifier specification.
        results: Assertion results produced so far.
        fatal_error: Optional infrastructure/authoring error.

    Returns:
        Dictionary payload containing reward, summary, and assertion results.
    """
    passed_weight = (
        0.0
        if fatal_error
        else float(round(sum(r["result"]["effective_weight"] for r in results if r["result"]["success"]), REWARD_DECIMAL_PLACES))
    )
    assertion_errors = sum(1 for r in results if not r["result"]["success"] and r["result"].get("error"))
    fatal_error_count = 1 if fatal_error and assertion_errors == 0 else 0
    errored = assertion_errors + fatal_error_count
    failed = sum(1 for r in results if not r["result"]["success"] and not r["result"].get("error"))
    summary = {
        "total_verifiers": len(spec.verifiers),
        "passed": sum(1 for r in results if r["result"]["success"]),
        "failed": failed,
        "errored": errored,
        "total_weight": 1.0,
        "passed_weight": passed_weight,
    }
    payload: dict[str, Any] = {
        "task_id": spec.task_id,
        "reward": passed_weight,
        "summary": summary,
        "assertion_results": results,
    }
    if fatal_error:
        payload["error"] = fatal_error
    return payload


def build_failure_payload(error: str, task_id: str | None = None) -> dict[str, Any]:
    """Builds a best-effort payload for fatal runtime failures.

    Args:
        error: Fatal error message.
        task_id: Optional task id if known.

    Returns:
        Failure payload with no reward.txt companion.
    """
    return {
        "task_id": task_id,
        "reward": 0.0,
        "summary": {
            "total_verifiers": 0,
            "passed": 0,
            "failed": 0,
            "errored": 1,
            "total_weight": 1.0,
            "passed_weight": 0.0,
        },
        "assertion_results": [],
        "error": error,
    }


def write_outputs(payload: dict[str, Any], verifier_dir: Path) -> None:
    """Writes verifier output artifacts.

    Args:
        payload: Full verifier report payload.
        verifier_dir: Output directory for reward artifacts.
    """
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "verifier-report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (verifier_dir / "reward.json").write_text(
        json.dumps({"reward": payload["reward"]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (verifier_dir / "reward.txt").write_text(str(payload["reward"]), encoding="utf-8")


def print_summary(payload: dict[str, Any]) -> None:
    """Logs a concise verifier summary.

    Args:
        payload: Full verifier reward payload.
    """
    summary = payload["summary"]
    logger.info(
        "Verifier complete: reward=%s passed=%s failed=%s errored=%s",
        payload["reward"],
        summary["passed"],
        summary["failed"],
        summary["errored"],
    )
    for result in payload["assertion_results"]:
        status = "PASS" if result["result"]["success"] else "FAIL"
        if result["result"].get("error"):
            suffix = f" - {result['result']['error']}"
        elif not result["result"]["success"]:
            suffix = f" - {result['result'].get('reason') or 'assertion failed'}"
        else:
            suffix = ""
        logger.info("[%s] %s%s", status, result["name"], suffix)

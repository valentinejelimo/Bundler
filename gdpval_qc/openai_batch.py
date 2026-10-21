"""OpenAI Batch API support for the QC engine's judge-ensemble calls.

Batches only ONE model's leg of the judge ensemble (the OpenAI one) at a
time. Other ensemble members (e.g. the Gemini judge) are unaffected — they
keep calling live inside ``run_ensemble``. Aggregation, synthesis, and the
emitter are untouched; this module only produces
``V18Engine.precomputed_llm_attempts``-shaped data for the CLI to feed back
into the engine.

Flow
----
1. ``collect_batch_requests`` — build one request row per (sample, dim) pair
   the target model would grade live, mirroring the gating in
   ``V18Engine.grade_dimension`` (``applies_when`` + ``check_kind == "llm"``).
2. ``submit_batch`` — upload the rows as one Batch API job; persist a
   resumable manifest to disk.
3. ``poll_batch`` — wait for the job to finish, bounded by a max wait.
4. ``fetch_batch_results`` — parse whatever completed into JudgeAttempts.
   Anything missing (still running, errored, or past the wait deadline) is
   simply absent from the result — the caller falls back to a live call for
   those via ``V18Engine.precomputed_llm_attempts``.
"""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .finding import JudgeAttempt
from .llm_client import LLMClientConfig, LLMMessage, ModelUsage, _store_usage
from .openai_client import build_request_body
from .precondition import Precondition
from .v18_judge import (
    ParseError, build_system_prompt, build_user_prompt, build_user_prompt_blocks,
    evaluate_applies_when, parse_judge_response, sample_has_multimodal_content,
)
from .v18_loader import Sample
from .v18_spec import V18Spec


@dataclass
class BatchRequestRow:
    """One pending judge call, ready to be serialized into the batch JSONL."""

    custom_id: str
    sample_id: str
    dim_name: str
    model: str
    body: dict[str, Any]


def collect_batch_requests(
    spec: V18Spec,
    samples: list[Sample],
    model: str,
    *,
    precondition_extras: dict[str, Precondition] | None = None,
    multimodal: bool = True,
) -> list[BatchRequestRow]:
    """Build one Batch API request row per (sample, dim) pair that ``model``
    would grade live.

    Only ``model`` is collected — pass the OpenAI model from the spec's
    ensemble. Other ensemble members are left to call live.
    """
    model_cfg = next((m for m in spec.models if m.model == model), None)
    if model_cfg is None:
        raise ValueError(
            f"{model!r} is not one of the spec's ensemble models: "
            f"{[m.model for m in spec.models]}"
        )

    system = build_system_prompt(spec)
    rows: list[BatchRequestRow] = []
    for sample in samples:
        for dim in spec.dimensions:
            if dim.check_kind != "llm":
                continue
            if not evaluate_applies_when(dim, sample, extra=precondition_extras):
                continue
            if multimodal and sample_has_multimodal_content(dim, sample):
                user_content = build_user_prompt_blocks(dim, sample)
            else:
                user_content = build_user_prompt(dim, sample)
            config = LLMClientConfig(
                model=model_cfg.model,
                max_tokens=model_cfg.max_tokens or 4096,
                system=system,
                reasoning_effort=model_cfg.reasoning_effort,
            )
            body = build_request_body(
                [LLMMessage(role="user", content=user_content)], config,
            )
            rows.append(BatchRequestRow(
                custom_id=f"req-{len(rows)}",
                sample_id=sample.task_id,
                dim_name=dim.name,
                model=model,
                body=body,
            ))
    return rows


def _default_client():
    import os

    import openai as _openai
    kwargs: dict[str, Any] = {}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return _openai.OpenAI(**kwargs)


def submit_batch(
    rows: list[BatchRequestRow],
    manifest_path: Path,
    *,
    client: Any = None,
    completion_window: str = "24h",
) -> str:
    """Upload ``rows`` as one Batch API job and persist a resumable manifest.

    The manifest (custom_id -> sample_id/dim_name/model, plus batch_id and
    input_file_id) is written to ``manifest_path`` so ``poll_batch`` /
    ``fetch_batch_results`` can run later, in a separate CLI invocation.
    """
    client = client or _default_client()

    lines = [
        json.dumps({
            "custom_id": row.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": row.body,
        })
        for row in rows
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8",
    ) as f:
        f.write("\n".join(lines))
        input_path = Path(f.name)

    try:
        with input_path.open("rb") as fh:
            uploaded = client.files.create(file=fh, purpose="batch")
    finally:
        input_path.unlink(missing_ok=True)

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )

    manifest = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "requests": {
            row.custom_id: {
                "sample_id": row.sample_id,
                "dim_name": row.dim_name,
                "model": row.model,
            }
            for row in rows
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return batch.id


_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def poll_batch(
    batch_id: str,
    *,
    max_wait_s: float = 1800.0,
    poll_interval_s: float = 15.0,
    client: Any = None,
) -> str:
    """Poll until the batch reaches a terminal status or ``max_wait_s`` elapses.

    Returns the final observed status. A non-terminal status after
    ``max_wait_s`` (typically "in_progress" or "validating") means some
    requests are still outstanding — the caller should treat any custom_id
    missing from ``fetch_batch_results`` as needing a live fallback call.
    """
    client = client or _default_client()
    started = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status in _TERMINAL_STATUSES:
            return batch.status
        if time.time() - started >= max_wait_s:
            return batch.status
        time.sleep(poll_interval_s)


def fetch_batch_results(
    batch_id: str,
    manifest: dict[str, Any],
    *,
    client: Any = None,
) -> dict[tuple[str, str], dict[str, JudgeAttempt]]:
    """Download whatever the batch has completed and parse it into JudgeAttempts.

    Returns ``{(sample_id, dim_name): {model: JudgeAttempt}}``. Any custom_id
    in ``manifest`` that isn't present (or errored) in the output is simply
    absent from the result — feed this straight into
    ``V18Engine(precomputed_llm_attempts=...)`` and ``run_ensemble`` falls
    back to a live call for whatever's missing.
    """
    client = client or _default_client()
    batch = client.batches.retrieve(batch_id)
    result: dict[tuple[str, str], dict[str, JudgeAttempt]] = {}
    if not batch.output_file_id:
        return result

    content = client.files.content(batch.output_file_id).text
    requests = manifest.get("requests", {})
    for line in content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = requests.get(row.get("custom_id"))
        if meta is None:
            continue
        response = row.get("response") or {}
        if row.get("error") or response.get("status_code") != 200:
            continue
        body = response.get("body") or {}
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            continue

        model = meta["model"]
        try:
            attempt = parse_judge_response(text, model)
        except ParseError as exc:
            attempt = JudgeAttempt(
                model=model, severity=0, confidence="low",
                applicable=False, headline="judge response unparseable",
                justification=str(exc),
                fixability="n/a", evidence_quote=None,
                raw_response=text, error=f"ParseError: {exc}",
            )

        usage = body.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        _store_usage([ModelUsage.from_tokens(
            model=model,
            input_tokens=usage.get("prompt_tokens", 0) - cached,
            output_tokens=usage.get("completion_tokens", 0),
            cache_read_tokens=cached,
            batch_discount=0.5,
        )])

        key = (meta["sample_id"], meta["dim_name"])
        result.setdefault(key, {})[model] = attempt

    return result

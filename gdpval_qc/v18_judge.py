"""Per-judge runtime for v18.

Given a `V18Dimension` and a loaded `Sample`, this module builds the judge
prompts, invokes the ensemble of judges declared on the spec, parses each
response into a `JudgeAttempt`, and hands the list off to the aggregator
(`v18_aggregate`) which produces the final `DimensionFinding`.

Three responsibilities, kept as pure functions where possible:

  1. `build_system_prompt(spec)` — system message common to every judge call.
     Encodes the global severity scale, evidence policy, confidence policy,
     and the required JSON output schema.

  2. `build_user_prompt(dim, sample)` — per-dim user message. Includes the
     dimension's prose (criterion + severity rubric + hard-fail patterns +
     anti-overcall guidance) and the bound input fields.

  3. `parse_judge_response(text)` — strict JSON parser that returns a
     JudgeAttempt or raises a ParseError.

The runner wires these together: per dim, evaluate `applies_when`; if active,
call each judge in the ensemble (parallel via ThreadPoolExecutor) and produce
a `JudgeAttempt` per model. Aggregation lives in `v18_aggregate`.
"""

from __future__ import annotations

import base64
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

from .finding import JudgeAttempt
from .llm_client import (
    ContentBlock, DocumentBlock, ImageBlock, LLMClient, LLMClientConfig,
    LLMMessage, TextBlock,
)
from .precondition import Precondition, get_precondition
from .v18_loader import FileContent, Sample
from .v18_spec import V18Dimension, V18Spec


# ── Public surface ─────────────────────────────────────────────────────────

class JudgeClientFactory(Protocol):
    """Returns the LLMClient to use for a given judge model name."""

    def __call__(self, model: str) -> LLMClient: ...


class ParseError(ValueError):
    """Raised when a judge response cannot be parsed into JudgeAttempt fields."""


# ── System prompt ──────────────────────────────────────────────────────────

# The system prompt is identical across dims and judges. It defines the
# output contract (JSON shape), severity meaning, evidence requirement, and
# confidence calibration. Per-dim guidance comes via the user prompt.

_SYSTEM_PROMPT_TEMPLATE = """\
You are a strict quality reviewer for a knowledge-work task dataset.

You will be given:
  - A criterion (the dimension's full prose, including a severity rubric and
    common hard-fail patterns)
  - Per-sample input field values (the inputs you are grading)

Your job is to return a single JSON object with this exact shape:

{{
  "severity":        0 | 1 | 2 | 3,
  "confidence":      "low" | "medium" | "high",
  "applicable":      true | false,
  "evidence_quote":  string | null,
  "headline":        string,
  "justification":   string,
  "fixability":      "trivial" | "fixable" | "structural" | "n/a"
}}

Severity scale (use ANCHORED levels from the criterion's severity rubric):
{severity_scale}

Required fields:

  severity        Integer 0–3 following the dimension's severity rubric.

  confidence      Your subjective confidence in this severity call.
                  - "high":   you can point to clear, unambiguous evidence
                  - "medium": evidence is solid but interpretive
                  - "low":    evidence is weak, ambiguous, or partial

  applicable      false ONLY when the criterion does not apply to this
                  sample given the inputs (e.g. a chart-quality criterion on
                  a chart-free sample). When applicable=false, also set
                  severity=0, evidence_quote=null, headline="not applicable",
                  fixability="n/a".

  evidence_quote  REQUIRED when severity >= 1. A verbatim short quote
                  (≤ 30 words) from one of the inputs that supports the
                  finding. If you cannot quote concrete evidence, set
                  severity=0.
                  When severity=0 or applicable=false, set this to null.

  headline        One sentence (≤ 25 words) stating what is wrong, in
                  sample-specific language. For severity=0 use "no issue".

  justification   1–3 sentences describing the reasoning. Refer to the
                  evidence_quote where applicable.

  fixability      Choose based on what fixing the issue would require:
                  - "trivial":    a one-line edit
                  - "fixable":    rewrite/regenerate one section or one file
                  - "structural": the underlying sample would need to be
                                  redone or different inputs supplied
                  - "n/a":        when severity=0 or applicable=false

Important policies:
  - Quote evidence verbatim. Do not paraphrase.
  - Default to the LOWER severity when the rubric is ambiguous between two
    levels.
  - Real professional artifacts are often dryly written and structured —
    structure alone is not a defect.
  - Return ONLY the JSON object; no prose before or after.
"""


def build_system_prompt(spec: V18Spec) -> str:
    scale_lines = "\n".join(
        f"  {k}: {spec.severity_scale[k]}" for k in sorted(spec.severity_scale.keys())
    )
    return _SYSTEM_PROMPT_TEMPLATE.format(severity_scale=scale_lines)


# ── User prompt ────────────────────────────────────────────────────────────

# Each input field is rendered into a labelled section. For document-typed
# fields, the file name and extracted text are included; lists become
# enumerated subsections.

_MAX_FIELD_CHARS = 30_000   # protect judges from over-large field values (lowered from 90k: data-heavy tasks summed >gpt-5.5 context across many source files)

# Per-field overrides. The 30k default exists to cap large SOURCE documents;
# it must NOT truncate verifier_spec_json — that field is a single structured
# rubric that has to stay valid/whole, and cutting it mid-JSON made judges
# report phantom "truncated rubric" / "missing coverage" findings on any
# verifier.json > 30k (e.g. task_48 at 30.7k). Verifier specs are bounded, so a
# generous cap is safe and never approaches the context limit on its own.
_FIELD_CHAR_LIMITS = {
    "verifier_spec_json": 200_000,
    # Cross-model trajectory digest: bounded by design (~30k on the k=10/6-model
    # corpus), but give headroom so a many-model / many-rubric task renders whole
    # rather than truncated mid-rubric (which reads as phantom "missing model").
    "intermodel_trajectory_digest": 60_000,
    # Single-trial deliverable exemplars (~25k deliverable text + per-rubric
    # outcomes); a little headroom above the deliverable budget.
    "failed_trial_deliverable_digest": 40_000,
    "passed_trial_deliverable_digest": 40_000,
}


def _limit_for(name: str) -> int:
    return _FIELD_CHAR_LIMITS.get(name, _MAX_FIELD_CHARS)


def _conversion_note(item: FileContent) -> str:
    """Inline warning when a file's conversion to a judge-ready form failed.

    On failure the judge sees only ``_extract_text`` output, which can be
    degraded (e.g. raw ZIP bytes for Office files) or a placeholder. Surfacing
    the failure in the prompt stops judges from grading degraded text as if it
    were the real deliverable, and the same failure is carried into the
    reviewer-facing output via ``SampleReview.data_quality_warnings``.
    """
    if getattr(item, "conversion_status", None) != "failed":
        return ""
    reason = item.conversion_error or "unknown error"
    return (
        f"\n  [CONVERSION FAILED for {item.name}: {reason}. "
        f"The content below is degraded best-effort text extraction, NOT the "
        f"rendered deliverable; it may be incomplete or unreadable. Do not "
        f"penalise layout/format quality on the basis of this fallback text.]"
    )


def build_user_prompt(dim: V18Dimension, sample: Sample) -> str:
    """Build the user message for one judge call on one (dim, sample) pair.

    Text-only path: every bound input field is rendered as inline text
    (extracted from the source if it's a document). For PDF-as-DocumentBlock
    multimodal support, use `build_user_prompt_blocks` instead.
    """
    parts: list[str] = []
    parts.append(f"<dimension>\n{dim.dimension_name}\n</dimension>")
    parts.append(f"<criterion>\n{dim.prose.strip()}\n</criterion>")
    parts.append("<inputs>")
    for fname in dim.inputs:
        value = sample.fields.get(fname)
        rendered = _render_field(fname, value)
        parts.append(rendered)
    parts.append("</inputs>")
    parts.append(
        "Return only the JSON object specified in the system prompt. "
        "Do not include any preamble or commentary."
    )
    return "\n\n".join(parts)


def build_user_prompt_blocks(
    dim: V18Dimension, sample: Sample,
) -> list[ContentBlock]:
    """Build a multimodal user message: TextBlocks for prose, DocumentBlocks
    for any bound FileContent whose `cached_path` is a PDF.

    Layout:
        TextBlock  <dimension> + <criterion> + opening <inputs>
        For each bound field:
            TextBlock with the field tag and a brief inline summary
            DocumentBlock for each FileContent with is_pdf=True
            TextBlock with the extracted text for any non-PDF content
        TextBlock closing </inputs> + return-format reminder

    Files that have a cached PDF are sent as DocumentBlocks; their inline
    text is omitted to avoid duplicating context. Files without a cached
    PDF fall back to text extraction with head+tail truncation.
    """
    blocks: list[ContentBlock] = []
    header = (
        f"<dimension>\n{dim.dimension_name}\n</dimension>\n\n"
        f"<criterion>\n{dim.prose.strip()}\n</criterion>\n\n"
        f"<inputs>"
    )
    blocks.append(TextBlock(text=header))

    for fname in dim.inputs:
        value = sample.fields.get(fname)
        for b in _render_field_as_blocks(fname, value):
            blocks.append(b)

    blocks.append(TextBlock(text=(
        "</inputs>\n\n"
        "Return only the JSON object specified in the system prompt. "
        "Do not include any preamble or commentary."
    )))
    return blocks


# Raster image extensions the Anthropic/OpenAI multimodal APIs accept as
# image blocks, mapped to their media type. Formats outside this set
# (.tif/.bmp) aren't accepted by the vision APIs, so they keep the text
# placeholder fallback rather than being sent as an unsupported image.
_IMAGE_MEDIA_TYPES = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def _render_field_as_blocks(name: str, value: Any) -> list[ContentBlock]:
    """Render one field as a list of ContentBlocks."""
    if value is None:
        return [TextBlock(text=f"\n\n<{name}>\n[absent]\n</{name}>")]

    if isinstance(value, str):
        return [TextBlock(text=f"\n\n<{name}>\n{_truncate(value, _limit_for(name))}\n</{name}>")]

    if isinstance(value, list):
        if not value:
            return [TextBlock(text=f"\n\n<{name}>\n[empty]\n</{name}>")]

        out: list[ContentBlock] = [TextBlock(text=f"\n\n<{name}>")]
        for i, item in enumerate(value, 1):
            if isinstance(item, FileContent):
                original_ext = item.path.suffix.lower()
                image_media_type = _IMAGE_MEDIA_TYPES.get(original_ext)
                # Raster image deliverable → send the ORIGINAL bytes as an
                # ImageBlock so multimodal chart-quality/readability judges
                # actually see the chart, instead of the loader's
                # "[binary image: ...]" text placeholder. Checked before the
                # PDF branch so an image that was also converted to a transport
                # PDF is still judged from its native (lossless) pixels.
                if image_media_type and item.path.is_file():
                    out.append(TextBlock(text=(
                        f"\n  <file index=\"{i}\" name=\"{item.name}\" "
                        f"original_extension=\"{original_ext}\" "
                        f"sent_to_judge_as=\"image\"/>"
                    )))
                    out.append(_image_to_image_block(item.path, image_media_type))
                # PDF available → send as DocumentBlock for legibility. The
                # `<file>` tag exposes the ORIGINAL extension explicitly so
                # judges grading format-match don't confuse the transport
                # PDF for the actual deliverable format.
                elif item.is_pdf and item.cached_path and item.cached_path.is_file():
                    is_transport_pdf = original_ext != ".pdf"
                    if is_transport_pdf:
                        attrs = (f"name=\"{item.name}\" "
                                 f"original_extension=\"{original_ext}\" "
                                 f"sent_to_judge_as=\"pdf-conversion-of-original\"")
                    else:
                        attrs = (f"name=\"{item.name}\" "
                                 f"original_extension=\"{original_ext}\"")
                    out.append(TextBlock(text=(
                        f"\n  <file index=\"{i}\" {attrs}/>"
                    )))
                    out.append(_pdf_to_document_block(item.cached_path))
                else:
                    # Text-only: render the extracted text with head+tail truncation.
                    out.append(TextBlock(text=(
                        f"\n  <file index=\"{i}\" name=\"{item.name}\" "
                        f"original_extension=\"{original_ext}\">"
                        f"{_conversion_note(item)}\n"
                        f"{_truncate(item.text)}\n  </file>"
                    )))
            else:
                out.append(TextBlock(text=(
                    f"\n  <item index=\"{i}\">\n{_truncate(str(item))}\n  </item>"
                )))
        out.append(TextBlock(text=f"\n</{name}>"))
        return out

    return [TextBlock(text=f"\n\n<{name}>\n{str(value)}\n</{name}>")]


def _pdf_to_document_block(path: Path) -> DocumentBlock:
    """Read a PDF file and wrap it as a DocumentBlock(base64).

    Errors during read are turned into a fallback TextBlock-shaped
    DocumentBlock placeholder so the judge gets a clear signal that the file
    was meant to be here but couldn't be read. (Callers should also include
    the text extraction in FileContent.text as a redundant fallback when
    multimodal isn't viable.)
    """
    data = path.read_bytes()
    b64 = base64.standard_b64encode(data).decode("ascii")
    return DocumentBlock(source={
        "type":       "base64",
        "media_type": "application/pdf",
        "data":       b64,
    })


def _image_to_image_block(path: Path, media_type: str) -> ImageBlock:
    """Read a raster image file and wrap it as an ImageBlock(base64)."""
    data = path.read_bytes()
    b64 = base64.standard_b64encode(data).decode("ascii")
    return ImageBlock(source={
        "type":       "base64",
        "media_type": media_type,
        "data":       b64,
    })


def sample_has_multimodal_content(dim: V18Dimension, sample: Sample) -> bool:
    """Return True iff any bound field has at least one FileContent that can be
    sent to a multimodal judge — either an available cached PDF (DocumentBlock)
    or a raster image in a vision-API-supported format (ImageBlock) — i.e., a
    multimodal user prompt is meaningful for this (dim, sample) pair."""
    for fname in dim.inputs:
        value = sample.fields.get(fname)
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, FileContent):
                    continue
                if (item.is_pdf
                        and item.cached_path
                        and item.cached_path.is_file()):
                    return True
                if (item.path.suffix.lower() in _IMAGE_MEDIA_TYPES
                        and item.path.is_file()):
                    return True
    return False


def _render_field(name: str, value: Any) -> str:
    if value is None:
        return f"<{name}>\n[absent]\n</{name}>"
    if isinstance(value, str):
        return f"<{name}>\n{_truncate(value, _limit_for(name))}\n</{name}>"
    if isinstance(value, list):
        if not value:
            return f"<{name}>\n[empty]\n</{name}>"
        sub_parts: list[str] = []
        for i, item in enumerate(value, 1):
            sub_parts.append(_render_list_item(item, i))
        body = "\n\n".join(sub_parts)
        return f"<{name}>\n{body}\n</{name}>"
    return f"<{name}>\n{str(value)}\n</{name}>"


def _render_list_item(item: Any, index: int) -> str:
    if isinstance(item, FileContent):
        return (f"  <file index=\"{index}\" name=\"{item.name}\">"
                f"{_conversion_note(item)}\n"
                f"{_truncate(item.text)}\n  </file>")
    return f"  <item index=\"{index}\">\n{_truncate(str(item))}\n  </item>"


def _truncate(text: str, limit: int = _MAX_FIELD_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return head + f"\n…[truncated {len(text) - limit} chars]…\n" + tail


# ── Response parsing ───────────────────────────────────────────────────────

# Even with explicit JSON instructions, judges occasionally wrap the JSON in
# code fences or a one-line prefix. The parser strips these and attempts to
# pull out the first {...} block.

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_BARE_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

VALID_CONFIDENCE = {"low", "medium", "high"}
VALID_FIXABILITY = {"trivial", "fixable", "structural", "n/a"}


def parse_judge_response(text: str, model: str) -> JudgeAttempt:
    """Parse a judge's text response into a JudgeAttempt.

    Strict shape: severity ∈ {0,1,2,3}, confidence ∈ {low,medium,high},
    applicable ∈ {true,false}, fixability ∈ {trivial,fixable,structural,n/a},
    plus headline / justification / evidence_quote.

    Raises ParseError on any malformed response. The caller is responsible
    for deciding what to do with an unparseable response (retry, or surface
    as a failed JudgeAttempt with error set).
    """
    if not text or not text.strip():
        raise ParseError("empty response")

    payload = _extract_json_object(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ParseError(f"not valid JSON: {exc}; payload preview: {payload[:200]!r}") from exc

    if not isinstance(data, dict):
        raise ParseError(f"JSON payload is not an object: {type(data).__name__}")

    severity = data.get("severity")
    if not isinstance(severity, int) or severity < 0 or severity > 3:
        raise ParseError(f"severity must be int 0-3, got {severity!r}")

    confidence = data.get("confidence")
    if confidence not in VALID_CONFIDENCE:
        raise ParseError(
            f"confidence must be one of {sorted(VALID_CONFIDENCE)}, got {confidence!r}"
        )

    applicable = data.get("applicable")
    if not isinstance(applicable, bool):
        raise ParseError(f"applicable must be bool, got {applicable!r}")

    fixability = data.get("fixability", "n/a")
    if fixability not in VALID_FIXABILITY:
        raise ParseError(
            f"fixability must be one of {sorted(VALID_FIXABILITY)}, got {fixability!r}"
        )

    headline = data.get("headline")
    if not isinstance(headline, str):
        raise ParseError(f"headline must be str, got {type(headline).__name__}")

    justification = data.get("justification", "")
    if not isinstance(justification, str):
        raise ParseError(f"justification must be str, got {type(justification).__name__}")

    evidence_quote = data.get("evidence_quote")
    if evidence_quote is not None and not isinstance(evidence_quote, str):
        raise ParseError(
            f"evidence_quote must be str or null, got {type(evidence_quote).__name__}"
        )

    return JudgeAttempt(
        model=model,
        severity=severity,
        confidence=confidence,
        applicable=applicable,
        headline=headline,
        justification=justification,
        fixability=fixability,
        evidence_quote=evidence_quote,
        raw_response=text,
        error=None,
    )


def _extract_json_object(text: str) -> str:
    """Find the JSON object in a possibly-fenced response."""
    # First, try a fenced ```json ... ``` block
    m = _JSON_FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith("{"):
            return candidate
    # Otherwise, return the first {...} block in the response
    m = _BARE_OBJECT_RE.search(text)
    if m:
        return m.group(0)
    # Fall back to the raw text
    return text.strip()


# ── Judge invocation ───────────────────────────────────────────────────────

def run_one_judge(
    client: LLMClient,
    spec: V18Spec,
    dim: V18Dimension,
    sample: Sample,
    model: str,
    *,
    max_tokens: int = 4096,
    reasoning_effort: str | None = None,
    multimodal: bool = True,
) -> JudgeAttempt:
    """Send one judge request and return a JudgeAttempt (failure-tolerant).

    When `multimodal=True` AND any bound field carries a cached PDF, the
    user prompt is sent as a `list[ContentBlock]` with DocumentBlocks for
    PDFs and TextBlocks for prose. Otherwise it's sent as a plain text
    string. Multimodal is silently disabled per-call when the dim's
    inputs don't include any PDF-bearing fields.

    Network/parse failures are converted into a JudgeAttempt with `error`
    set and severity=0/applicable=False. This way aggregation always has a
    list of attempts to work with; failed judges contribute nothing to the
    aggregate but the failure is visible in the audit trail.
    """
    system = build_system_prompt(spec)
    user_content: Any
    if multimodal and sample_has_multimodal_content(dim, sample):
        user_content = build_user_prompt_blocks(dim, sample)
    else:
        user_content = build_user_prompt(dim, sample)
    config = LLMClientConfig(
        model=model,
        max_tokens=max_tokens,
        system=system,
        reasoning_effort=reasoning_effort,
    )
    try:
        text = client.complete(
            [LLMMessage(role="user", content=user_content)], config,
        )
    except Exception as exc:
        # Printed live (not just aggregated into the end-of-run summary) so a
        # systemic problem — e.g. a dead API key — is visible within seconds
        # of the first failure, not after the whole batch finishes.
        print(
            f"JUDGE-CALL FAILED [{sample.task_id} / {dim.name} / {model}]: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return JudgeAttempt(
            model=model, severity=0, confidence="low",
            applicable=False, headline="judge call failed",
            justification=str(exc),
            fixability="n/a", evidence_quote=None,
            raw_response="", error=f"{type(exc).__name__}: {exc}",
        )
    try:
        return parse_judge_response(text, model)
    except ParseError as exc:
        return JudgeAttempt(
            model=model, severity=0, confidence="low",
            applicable=False, headline="judge response unparseable",
            justification=str(exc),
            fixability="n/a", evidence_quote=None,
            raw_response=text, error=f"ParseError: {exc}",
        )


def run_ensemble(
    spec: V18Spec,
    dim: V18Dimension,
    sample: Sample,
    client_factory: JudgeClientFactory,
    *,
    parallel: bool = True,
    multimodal: bool = True,
    precomputed: dict[str, JudgeAttempt] | None = None,
) -> list[JudgeAttempt]:
    """Run every judge in the spec's ensemble for one (dim, sample) pair.

    ``precomputed`` (model name -> JudgeAttempt) short-circuits models whose
    attempt was already resolved out-of-band, e.g. via the OpenAI Batch API
    (see ``openai_batch.py``). Models absent from ``precomputed`` — including
    every model when it's None — still call live. This lets one leg of the
    ensemble be batched while the rest of the ensemble is untouched.
    """
    def _one(model_cfg) -> JudgeAttempt:
        if precomputed and model_cfg.model in precomputed:
            return precomputed[model_cfg.model]
        client = client_factory(model_cfg.model)
        return run_one_judge(
            client, spec, dim, sample,
            model=model_cfg.model,
            max_tokens=model_cfg.max_tokens or 4096,
            reasoning_effort=model_cfg.reasoning_effort,
            multimodal=multimodal,
        )

    if not parallel or len(spec.models) <= 1:
        return [_one(m) for m in spec.models]

    attempts: list[JudgeAttempt] = [None] * len(spec.models)   # type: ignore
    with ThreadPoolExecutor(max_workers=len(spec.models)) as pool:
        futs = {pool.submit(_one, m): i for i, m in enumerate(spec.models)}
        for f in as_completed(futs):
            attempts[futs[f]] = f.result()
    return attempts


# ── Precondition evaluation ────────────────────────────────────────────────

def evaluate_applies_when(
    dim: V18Dimension,
    sample: Sample,
    extra: dict[str, Precondition] | None = None,
) -> bool:
    """Return True if the dimension applies to this sample.

    A dim with no `applies_when` always applies. Otherwise the precondition
    function is looked up via `quality_framework.precondition.get_precondition`
    (falling back to `extra`). The function is called with the sample and
    must return a bool.
    """
    if dim.applies_when is None or dim.applies_when.is_unconditional():
        return True
    name = dim.applies_when.builtin or dim.applies_when.callback
    if name is None:
        return True
    fn = get_precondition(name, extra=extra)
    return bool(fn(sample))

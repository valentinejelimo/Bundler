"""MockLLMClient — a deterministic, controllable LLMClient for tests.

Two usage modes:

  1. **Canned responses** — supply a dict mapping (model, dim_name) → JSON
     string. The client looks up by dimension on each request and returns the
     matching canned string. Use for end-to-end pipeline tests where you want
     to control judge outputs precisely.

  2. **Programmatic responses** — supply a callable that takes the parsed
     request (model, system prompt, user prompt) and returns a string. Use
     for prompt-template tests that need to inspect what the framework sent.

The client conforms to the framework's `LLMClient` Protocol and is safe to
substitute anywhere a real client is used.

Helper functions (`make_attempt_json`, `make_canned_response`) build
well-formed judge-response JSON strings without callers needing to write
the JSON by hand.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Literal

from .llm_client import LLMClientConfig, LLMMessage


# ── Helpers for building canned responses ──────────────────────────────────

Confidence  = Literal["low", "medium", "high"]
Fixability  = Literal["trivial", "fixable", "structural", "n/a"]


def make_attempt_json(
    *,
    severity: int = 0,
    confidence: Confidence = "medium",
    applicable: bool = True,
    evidence_quote: str | None = None,
    headline: str = "no issue",
    justification: str = "All criteria met.",
    fixability: Fixability = "n/a",
) -> str:
    """Build a JudgeAttempt-shaped JSON string."""
    payload = {
        "severity":       severity,
        "confidence":     confidence,
        "applicable":     applicable,
        "evidence_quote": evidence_quote,
        "headline":       headline,
        "justification":  justification,
        "fixability":     fixability,
    }
    return json.dumps(payload)


def make_canned_response(
    severity: int = 0,
    confidence: Confidence = "medium",
    evidence_quote: str | None = None,
    headline: str = "no issue",
    justification: str = "All criteria met.",
    fixability: Fixability = "n/a",
    applicable: bool = True,
) -> str:
    """Alias for `make_attempt_json` with the common kw-args."""
    return make_attempt_json(
        severity=severity,
        confidence=confidence,
        applicable=applicable,
        evidence_quote=evidence_quote,
        headline=headline,
        justification=justification,
        fixability=fixability,
    )


# ── Recorded request (for assertions in tests) ─────────────────────────────

@dataclass
class RecordedCall:
    model:    str
    system:   str | None
    user:     str

    @property
    def dim_name_from_user(self) -> str | None:
        """Extract <dimension>...</dimension> content from the user prompt."""
        m = re.search(r"<dimension>\s*(.*?)\s*</dimension>", self.user, re.DOTALL)
        return m.group(1).strip() if m else None


# ── Mock client ────────────────────────────────────────────────────────────

class MockLLMClient:
    """Deterministic LLMClient for tests.

    Resolution order on each `complete()`:
      1. If a key (model, dim_name) exists in `canned`, return it.
      2. If a key (model,) exists in `canned`, return it.
      3. If `responder` callable is set, call it with the recorded request
         and use its return.
      4. If `default` is set, return it.
      5. Raise RuntimeError describing what was missing.

    Calls are recorded in `calls` for test assertions.
    """

    def __init__(
        self,
        *,
        canned: dict | None = None,
        responder: Callable[[RecordedCall], str] | None = None,
        default: str | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        # Normalize canned keys to tuple form: ("model", "dim_name") or ("model",)
        self._canned: dict[tuple, str] = {}
        for k, v in (canned or {}).items():
            if isinstance(k, str):
                self._canned[(k,)] = v
            elif isinstance(k, tuple):
                self._canned[k] = v
            else:
                raise TypeError(f"canned key must be str or tuple, got {type(k)}")
        self._responder = responder
        self._default = default
        self._raise = raise_on_call
        self._lock = threading.Lock()
        self.calls: list[RecordedCall] = []

    def complete(
        self,
        messages: list[LLMMessage],
        config: LLMClientConfig,
    ) -> str:
        if self._raise is not None:
            raise self._raise

        # Concatenate user content into a single string for inspection
        user_text = ""
        for m in messages:
            if m.role == "user":
                if isinstance(m.content, str):
                    user_text += m.content
                else:
                    for block in m.content:
                        if hasattr(block, "text"):
                            user_text += block.text

        call = RecordedCall(model=config.model or "", system=config.system, user=user_text)
        with self._lock:
            self.calls.append(call)

        # Resolution
        dim_name = call.dim_name_from_user

        if dim_name and config.model and (config.model, dim_name) in self._canned:
            return self._canned[(config.model, dim_name)]
        if config.model and (config.model,) in self._canned:
            return self._canned[(config.model,)]
        if self._responder is not None:
            return self._responder(call)
        if self._default is not None:
            return self._default

        raise RuntimeError(
            f"MockLLMClient: no canned response for model={config.model!r} "
            f"dim={dim_name!r}. Set canned[(model, dim)], canned[(model,)], "
            f"a responder, or default."
        )

    def calls_for_model(self, model: str) -> list[RecordedCall]:
        return [c for c in self.calls if c.model == model]

    def calls_for_dim(self, dim_name: str) -> list[RecordedCall]:
        return [c for c in self.calls if c.dim_name_from_user == dim_name]

"""GDPVal Quality Control framework — v18 only, standalone distribution.

This package contains the v18 quality-assessment pipeline for grading
GDPVal-style knowledge-work task datasets. The entry-point CLI lives at
the repository root as `v18_assess.py`.

Public API
----------

Core types and engine::

    from gdpval_qc import (
        V18Spec, V18Engine, TaskCorpusLoader,
        BatchReview, SampleReview, DimensionFinding, Issue,
        write_batch_artifacts,
    )

LLM client adapters::

    from gdpval_qc import (
        OpenAIClientAdapter, GeminiClientAdapter, AnthropicClientAdapter,
        LLMClient, LLMClientConfig, LLMMessage, MockLLMClient,
    )

Document conversion + multimodal helpers::

    from gdpval_qc import DocumentConverter
"""

# Core data types
from .finding import (
    DimensionFinding,
    Issue,
    JudgeAttempt,
    Confidence,
    Fixability,
    GateRole,
    OnDisagree,
)
from .review import (
    BatchReview,
    BatchSummary,
    CalibrationReport,
    SampleReview,
    Verdict,
    VERDICT_ORDER,
)

# LLM client layer
from .llm_client import (
    AnthropicClientAdapter,
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    LLMClient,
    LLMClientConfig,
    LLMMessage,
    ModelUsage,
    TextBlock,
    get_last_usage,
)
from .openai_client import OpenAIClientAdapter
from .gemini_client import GeminiClientAdapter
from .mock_llm import MockLLMClient, make_attempt_json

# Document conversion
from .converters import DocumentConverter

# v18 pipeline
from .v18_spec import (
    AppliesWhen,
    CalibrationConfig,
    ConfidenceDemotion,
    FieldBinding,
    GateConfig,
    JudgeModelConfig,
    SpecIssue,
    SpecValidationError,
    SynthesisConfig,
    V18Dimension,
    V18Spec,
)
from .v18_loader import FileContent, Sample, TaskCorpusLoader
from .v18_engine import (
    BUILTIN_PROGRAMMATIC_CALLBACKS,
    ProgressCallback,
    V18Engine,
    register_programmatic_callback,
)
from .v18_emitter import (
    collect_judge_errors,
    to_csv,
    to_json,
    to_markdown_summary,
    write_batch_artifacts,
)

__all__ = [
    # Core types
    "DimensionFinding", "Issue", "JudgeAttempt",
    "Confidence", "Fixability", "GateRole", "OnDisagree",
    "BatchReview", "BatchSummary", "CalibrationReport",
    "SampleReview", "Verdict", "VERDICT_ORDER",
    # LLM client layer
    "AnthropicClientAdapter", "OpenAIClientAdapter", "GeminiClientAdapter",
    "MockLLMClient", "make_attempt_json",
    "ContentBlock", "DocumentBlock", "ImageBlock", "TextBlock",
    "LLMClient", "LLMClientConfig", "LLMMessage",
    "ModelUsage", "get_last_usage",
    # Document conversion
    "DocumentConverter",
    # Spec
    "V18Spec", "V18Dimension", "GateConfig", "AppliesWhen",
    "FieldBinding", "SynthesisConfig", "CalibrationConfig",
    "ConfidenceDemotion", "JudgeModelConfig",
    "SpecIssue", "SpecValidationError",
    # Loader
    "FileContent", "Sample", "TaskCorpusLoader",
    # Engine
    "V18Engine", "ProgressCallback",
    "BUILTIN_PROGRAMMATIC_CALLBACKS", "register_programmatic_callback",
    # Emitters
    "to_csv", "to_json", "to_markdown_summary", "write_batch_artifacts",
    "collect_judge_errors",
]

__version__ = "18.5"

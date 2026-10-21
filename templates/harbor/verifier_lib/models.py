import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator


Comparison = Literal[
    "equals",
    "not_equals",
    "approx_equals",
    "greater_than",
    "less_than",
    "greater_than_equal",
    "less_than_equal",
    "contains",
    "not_contains",
    "regex_match",
    "not_regex_match",
    "in_array",
    "not_in_array",
    "length_equals",
]


class StrictModel(BaseModel):
    """Base Pydantic model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Config(StrictModel):
    """Optional global verifier configuration.

    Attributes:
        models: LiteLLM model names used for rubric judge attempts.
        temperature: Temperature passed to LiteLLM rubric judge calls.
        reasoning_effort: Optional reasoning effort for judge calls.
    """

    models: list[str] | None = Field(default=None, min_length=1, max_length=3)
    temperature: float | None = None
    reasoning_effort: str | None = None

    @field_validator("models")
    @classmethod
    def validate_models(cls, value: list[str] | None) -> list[str] | None:
        """Validates LiteLLM-native model names.

        Args:
            value: Configured LiteLLM model names.

        Returns:
            The unchanged model list when valid.

        Raises:
            ValueError: If a model name is blank or uses colon-style provider
                separators.
        """
        if value is None:
            return value
        for model in value:
            if not model.strip():
                raise ValueError("config.models entries must be non-empty")
            if ":" in model:
                raise ValueError("config.models must use LiteLLM-native model names such as openai/gpt-5.4")
        return value


class VerifierMetadata(StrictModel):
    """Author-provided verifier metadata.

    Metadata is an annotation surface for QC and reporting tools, so unknown
    fields are preserved instead of rejected.

    Attributes:
        how_justification: Explanation of how this verifier checks the task.
        why_justification: Explanation of why this verifier matters.
        tag: Optional human grouping label.
        weight: Optional authored weight from zero to one.
    """

    model_config = ConfigDict(extra="allow")

    how_justification: str
    why_justification: str
    tag: str | None = None
    weight: float | None = Field(default=None, gt=0, le=1)


class FileCommand(StrictModel):
    """File source command declaration.

    Attributes:
        type: Source namespace, such as ``json`` or ``docx``.
        command: Command name within the namespace.
        arguments: Command-specific arguments.
    """

    type: str
    command: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ResponseCommand(StrictModel):
    """Response source command declaration.

    Attributes:
        command: Response command name.
        arguments: Optional empty argument object. Non-empty arguments are not
            allowed for response commands yet.
    """

    command: str
    arguments: dict[str, Any] = Field(default_factory=dict, max_length=0)


class Source(StrictModel):
    """Verifier source declaration.

    Attributes:
        type: Source family.
        file: File source command declaration when ``type`` is ``file``.
        response: Response source command declaration when ``type`` is
            ``response``.
    """

    type: Literal["file", "response"]
    file: FileCommand | None = None
    response: ResponseCommand | None = None

    @model_validator(mode="after")
    def validate_branch(self) -> "Source":
        """Validates that source details match the selected source family.

        Returns:
            The validated source declaration.

        Raises:
            ValueError: If source branch config does not match the selected
                source type.
        """
        if self.type == "file" and self.file is None:
            raise ValueError("file source requires file config")
        if self.type == "file" and self.response is not None:
            raise ValueError("file source must not include response config")
        if self.type == "response" and self.response is None:
            raise ValueError("response source requires response config")
        if self.type == "response" and self.file is not None:
            raise ValueError("response source must not include file config")
        return self

    @property
    def source_name(self) -> str:
        """Returns the registered source command name.

        Returns:
            Dotted source command name used by ``SourceRegistry``.
        """
        if self.file is not None:
            return f"{self.file.type}.{self.file.command}"
        assert self.response is not None
        return f"response.{self.response.command}"

    @property
    def arguments(self) -> dict[str, Any]:
        """Returns command arguments for the selected source branch.

        Returns:
            Command-specific argument mapping.
        """
        if self.file is not None:
            return self.file.arguments
        assert self.response is not None
        return {}


class Tolerance(StrictModel):
    """Numeric comparison tolerance config.

    Attributes:
        absolute: Optional absolute tolerance.
        relative: Optional relative tolerance.
    """

    absolute: float | None = None
    relative: float | None = None


class DeterministicConfig(StrictModel):
    """Deterministic assertion details.

    Attributes:
        path: JSONPath evaluated against the source output root.
        comparison: Comparator used to compare actual and expected values.
        tolerance: Optional numeric tolerance for approximate comparisons.
    """

    path: str
    comparison: Comparison
    tolerance: Tolerance | None = None

    @field_validator("path")
    @classmethod
    def validate_jsonpath(cls, value: str) -> str:
        """Validates deterministic JSONPath syntax at schema parse time.

        Args:
            value: JSONPath expression.

        Returns:
            The unchanged JSONPath when valid.

        Raises:
            ValueError: If the JSONPath cannot be parsed.
        """
        try:
            from jsonpath_ng.ext import parse as jsonpath_parse  # lazy: see compare.py's module note

            jsonpath_parse(value)
        except Exception as exc:
            raise ValueError(f"invalid JSONPath: {value}") from exc
        return value


class RubricConfig(StrictModel):
    """LLM rubric assertion details.

    Attributes:
        prompt: Rubric text evaluated by the judge.
    """

    prompt: str


class Assertion(StrictModel):
    """Verifier assertion declaration.

    Attributes:
        type: Assertion family, either deterministic or rubric.
        expected: Expected value. Rubric expected values must be booleans.
        deterministic: Deterministic assertion details.
        rubric: Rubric assertion details.
    """

    type: Literal["deterministic", "rubric"]
    expected: Any
    deterministic: DeterministicConfig | None = None
    rubric: RubricConfig | None = None

    @model_validator(mode="after")
    def validate_branch(self) -> "Assertion":
        """Validates that assertion details match the selected type.

        Returns:
            The validated assertion.

        Raises:
            ValueError: If required branch config is missing or forbidden branch
                config is present.
        """
        if self.type == "deterministic":
            if self.deterministic is None:
                raise ValueError("deterministic assertion requires deterministic config")
            if self.rubric is not None:
                raise ValueError("deterministic assertion must not include rubric config")
            validate_deterministic_expected(self.expected, self.deterministic.comparison)
        if self.type == "rubric":
            if self.rubric is None:
                raise ValueError("rubric assertion requires rubric config")
            if self.deterministic is not None:
                raise ValueError("rubric assertion must not include deterministic config")
            if not isinstance(self.expected, bool):
                raise ValueError("rubric assertion expected value must be boolean")
        return self


def validate_deterministic_expected(expected: Any, comparison: Comparison) -> None:
    """Validates expected value shape for deterministic comparisons.

    Args:
        expected: Expected value from the verifier assertion.
        comparison: Comparator selected by the verifier assertion.

    Raises:
        ValueError: If the expected value cannot be used by the comparator.
    """
    numeric_comparisons = {
        "approx_equals",
        "greater_than",
        "less_than",
        "greater_than_equal",
        "less_than_equal",
    }
    if comparison in numeric_comparisons and not is_json_number(expected):
        raise ValueError(f"{comparison} expected value must be a JSON number")
    if comparison in {"in_array", "not_in_array"} and not isinstance(expected, list):
        raise ValueError(f"{comparison} expected value must be an array")
    if comparison in {"regex_match", "not_regex_match"} and not isinstance(expected, str):
        raise ValueError(f"{comparison} expected value must be a regex string")


def is_json_number(value: Any) -> bool:
    """Checks whether a value is a JSON number in Python form.

    Args:
        value: Value to inspect.

    Returns:
        True for integer and float values except booleans.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class VerifierDefinition(StrictModel):
    """One verifier definition from verifier.json.

    Attributes:
        name: Unique verifier name.
        metadata: Author-provided metadata.
        source: Source declaration.
        assertion: Assertion declaration.
    """

    name: str = Field(min_length=1)
    metadata: VerifierMetadata
    source: Source
    assertion: Assertion


class VerifierSpec(StrictModel):
    """Root verifier.json schema."""

    task_id: str
    config: Config | None = None
    verifiers: list[VerifierDefinition]

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_keys(cls, data: Any) -> Any:
        """Rejects old verifier schema keys before normal validation.

        Args:
            data: Raw verifier spec object.

        Returns:
            The unchanged object when it does not include legacy keys.

        Raises:
            ValueError: If a legacy top-level key is present.
        """
        if isinstance(data, dict):
            for key in ("schema_version", "judge", "assertions"):
                if key in data:
                    raise ValueError(f"legacy top-level key is not supported: {key}")
        return data

    @field_validator("verifiers")
    @classmethod
    def require_verifiers(cls, value: list[VerifierDefinition]) -> list[VerifierDefinition]:
        """Requires at least one verifier.

        Args:
            value: Parsed verifier definitions.

        Returns:
            The verifier list when it is non-empty.

        Raises:
            ValueError: If no verifiers are configured.
        """
        if not value:
            raise ValueError("verifier.json must contain at least one verifier")
        return value

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "VerifierSpec":
        """Validates cross-verifier schema rules.

        Returns:
            The validated verifier spec.

        Raises:
            ValueError: If names, judge config, or authored weights are invalid.
        """
        names = [verifier.name for verifier in self.verifiers]
        if len(names) != len(set(names)):
            raise ValueError("verifier names must be unique")

        has_rubric = any(v.assertion.type == "rubric" for v in self.verifiers)
        if has_rubric:
            if self.config is None or self.config.models is None or self.config.temperature is None:
                raise ValueError("config.models and config.temperature are required for rubric assertions")

        validate_weight_distribution(self.verifiers)
        return self


class LLMVerdict(StrictModel):
    """Structured verdict returned by the LLM judge.

    Attributes:
        verdict: Boolean rubric verdict.
        justification: Short evidence-based explanation for the verdict.
    """

    verdict: bool
    justification: str


def validate_weight_distribution(verifiers: list[VerifierDefinition]) -> None:
    """Validates authored verifier weights.

    Args:
        verifiers: Parsed verifier definitions.

    Raises:
        ValueError: If explicit weights cannot be distributed to one total.
    """
    weights = [v.metadata.weight for v in verifiers]
    explicit = [w for w in weights if w is not None]
    if not explicit:
        return

    explicit_sum = sum(explicit)
    unweighted = len([w for w in weights if w is None])
    if explicit_sum > 1 + 1e-9:
        raise ValueError("verifier weights must sum to at most 1")
    if unweighted and explicit_sum >= 1 - 1e-9:
        raise ValueError("unweighted verifiers require remaining positive weight")
    if not unweighted and abs(explicit_sum - 1) > 1e-9:
        raise ValueError("explicit verifier weights must sum to 1 when all verifiers are weighted")


def effective_weights(verifiers: list[VerifierDefinition]) -> dict[str, float]:
    """Computes effective verifier weights.

    Args:
        verifiers: Parsed verifier definitions.

    Returns:
        Mapping from verifier name to effective weight.
    """
    explicit_sum = sum(v.metadata.weight or 0 for v in verifiers)
    unweighted = [v for v in verifiers if v.metadata.weight is None]
    default = (1 - explicit_sum) / len(unweighted) if unweighted else 0
    return {
        verifier.name: round(verifier.metadata.weight if verifier.metadata.weight is not None else default, 10)
        for verifier in verifiers
    }


def source_output_to_data(value: BaseModel | RootModel[Any]) -> Any:
    """Converts a Pydantic source output into JSON-like data.

    Args:
        value: Source output model.

    Returns:
        Model data suitable for JSONPath, serialization, or prompt rendering.
    """
    if isinstance(value, RootModel):
        return value.root
    return value.model_dump()


def jsonable(value: Any) -> Any:
    """Returns a JSON-serializable representation of a value.

    Args:
        value: Arbitrary Python value.

    Returns:
        The original value when JSON serialization supports it; otherwise its
        string representation.
    """
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)

"""Unit tests for verifier_lib.models — Pydantic models.

Tests VerifierSpec validation with valid/invalid JSON, effective_weights,
validate_weight_distribution, validate_deterministic_expected. Builds
minimal valid verifier specs and asserts validation behavior. pydantic installed.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from verifier_lib.models import (
    Assertion, Config, DeterministicConfig, FileCommand, Source,
    VerifierDefinition, VerifierMetadata, VerifierSpec,
    effective_weights, validate_deterministic_expected,
    validate_weight_distribution,
)


# ── Builders ──────────────────────────────────────────────────────────────────

def _valid_verifier(name="v1", weight=None, assertion_type="deterministic") -> dict:
    assertion = {
        "type": assertion_type,
        "expected": True,
    }
    if assertion_type == "deterministic":
        assertion["deterministic"] = {
            "path": "$.result",
            "comparison": "equals",
        }
    else:
        assertion["rubric"] = {"prompt": "Is the answer correct?"}
        assertion["expected"] = True
    meta = {"how_justification": "how", "why_justification": "why"}
    if weight is not None:
        meta["weight"] = weight
    return {
        "name": name,
        "metadata": meta,
        "source": {
            "type": "file",
            "file": {"type": "json", "command": "read"},
        },
        "assertion": assertion,
    }


def _valid_spec(verifiers=None, config=None) -> dict:
    spec = {
        "task_id": "task_1",
        # NOTE: use "is None" (not "or") so an explicit [] is preserved.
        # `verifiers or [...]` would treat [] as falsy and silently
        # substitute a non-empty list, masking the empty-list tests below.
        "verifiers": [_valid_verifier()] if verifiers is None else verifiers,
    }
    if config is not None:
        spec["config"] = config
    return spec


# ── VerifierSpec valid parse ──────────────────────────────────────────────────

class TestVerifierSpecValid:
    def test_minimal_valid(self):
        spec = VerifierSpec(**_valid_spec())
        assert spec.task_id == "task_1"
        assert len(spec.verifiers) == 1
        assert spec.verifiers[0].name == "v1"

    def test_config_optional(self):
        spec = VerifierSpec(**_valid_spec())
        assert spec.config is None

    def test_with_config(self):
        config = {"models": ["openai/gpt-5.4"], "temperature": 0.0}
        spec = VerifierSpec(**_valid_spec(config=config))
        assert spec.config.models == ["openai/gpt-5.4"]
        assert spec.config.temperature == 0.0

    def test_deterministic_assertion_parsed(self):
        spec = VerifierSpec(**_valid_spec())
        assert spec.verifiers[0].assertion.type == "deterministic"
        assert spec.verifiers[0].assertion.deterministic.comparison == "equals"

    def test_source_parsed(self):
        spec = VerifierSpec(**_valid_spec())
        assert spec.verifiers[0].source.file.type == "json"
        assert spec.verifiers[0].source.file.command == "read"


# ── VerifierSpec invalid ──────────────────────────────────────────────────────

class TestVerifierSpecInvalid:
    def test_empty_verifiers_rejected(self):
        # Pydantic's field_validator catches the empty list: the
        # require_verifiers validator raises ValueError which Pydantic wraps.
        with pytest.raises(ValidationError) as exc_info:
            VerifierSpec(**_valid_spec(verifiers=[]))
        assert "at least one verifier" in str(exc_info.value).lower()

    def test_empty_verifiers_rejected_via_direct_construct(self):
        # Direct construction with empty list also fails (field_validator runs)
        with pytest.raises(ValidationError):
            VerifierSpec(task_id="t1", verifiers=[])

    def test_missing_task_id(self):
        spec_data = _valid_spec()
        del spec_data["task_id"]
        with pytest.raises(ValidationError):
            VerifierSpec(**spec_data)

    def test_duplicate_verifier_names(self):
        verifiers = [_valid_verifier(name="dup"), _valid_verifier(name="dup")]
        with pytest.raises(ValidationError) as exc_info:
            VerifierSpec(**_valid_spec(verifiers=verifiers))
        assert "unique" in str(exc_info.value).lower()

    def test_legacy_key_rejected(self):
        spec_data = _valid_spec()
        spec_data["schema_version"] = "1.0"
        with pytest.raises(ValidationError) as exc_info:
            VerifierSpec(**spec_data)
        assert "legacy" in str(exc_info.value).lower()

    def test_legacy_assertions_key_rejected(self):
        spec_data = _valid_spec()
        spec_data["assertions"] = []
        with pytest.raises(ValidationError) as exc_info:
            VerifierSpec(**spec_data)

    def test_unknown_field_rejected(self):
        # StrictModel forbids extra fields
        with pytest.raises(ValidationError):
            VerifierSpec(**_valid_spec(), bogus_field="x")

    def test_rubric_requires_config_models(self):
        verifier = _valid_verifier(assertion_type="rubric")
        # No config provided -> rubric requires config.models and temperature
        with pytest.raises(ValidationError) as exc_info:
            VerifierSpec(**_valid_spec(verifiers=[verifier]))
        assert "config.models" in str(exc_info.value)

    def test_rubric_with_config_ok(self):
        verifier = _valid_verifier(assertion_type="rubric")
        config = {"models": ["openai/gpt-5.4"], "temperature": 0.0}
        spec = VerifierSpec(**_valid_spec(verifiers=[verifier], config=config))
        assert spec.verifiers[0].assertion.type == "rubric"

    def test_invalid_model_colon_rejected(self):
        config = {"models": ["gpt-5.4:latest"], "temperature": 0.0}
        with pytest.raises(ValidationError):
            Config(**config)

    def test_blank_model_rejected(self):
        config = {"models": [""], "temperature": 0.0}
        with pytest.raises(ValidationError):
            Config(**config)


# ── Assertion validation ──────────────────────────────────────────────────────

class TestAssertionValidation:
    def test_deterministic_requires_config(self):
        with pytest.raises(ValidationError) as exc_info:
            Assertion(type="deterministic", expected=True)
        assert "deterministic assertion requires" in str(exc_info.value)

    def test_deterministic_must_not_have_rubric(self):
        with pytest.raises(ValidationError):
            Assertion(
                type="deterministic", expected=True,
                deterministic=DeterministicConfig(path="$.x", comparison="equals"),
                rubric={"prompt": "x"},
            )

    def test_rubric_requires_config(self):
        with pytest.raises(ValidationError) as exc_info:
            Assertion(type="rubric", expected=True)
        assert "rubric assertion requires" in str(exc_info.value)

    def test_rubric_expected_must_be_bool(self):
        with pytest.raises(ValidationError):
            Assertion(type="rubric", expected="yes",
                      rubric={"prompt": "x"})

    def test_rubric_must_not_have_deterministic(self):
        with pytest.raises(ValidationError):
            Assertion(
                type="rubric", expected=True,
                rubric={"prompt": "x"},
                deterministic=DeterministicConfig(path="$.x", comparison="equals"),
            )


# ── Source validation ─────────────────────────────────────────────────────────

class TestSourceValidation:
    def test_file_source_requires_file(self):
        with pytest.raises(ValidationError):
            Source(type="file")

    def test_file_source_no_response(self):
        with pytest.raises(ValidationError):
            Source(type="file", file={"type": "json", "command": "read"},
                   response={"command": "x"})

    def test_response_source_requires_response(self):
        with pytest.raises(ValidationError):
            Source(type="response")

    def test_response_source_no_file(self):
        with pytest.raises(ValidationError):
            Source(type="response", response={"command": "x"},
                   file={"type": "json", "command": "read"})


# ── effective_weights ─────────────────────────────────────────────────────────

class TestEffectiveWeights:
    def test_all_none_equal_distribution(self):
        specs = [
            VerifierDefinition(**_valid_verifier(name=f"v{i}"))
            for i in range(3)
        ]
        weights = effective_weights(specs)
        # Each gets 1/3
        assert abs(weights["v0"] - 1/3) < 1e-9
        assert abs(weights["v1"] - 1/3) < 1e-9
        assert abs(weights["v2"] - 1/3) < 1e-9

    def test_explicit_weight(self):
        v0 = VerifierDefinition(**_valid_verifier(name="v0", weight=0.5))
        v1 = VerifierDefinition(**_valid_verifier(name="v1"))
        v2 = VerifierDefinition(**_valid_verifier(name="v2"))
        weights = effective_weights([v0, v1, v2])
        assert weights["v0"] == 0.5
        # Remaining 0.5 split between v1 and v2 = 0.25 each
        assert abs(weights["v1"] - 0.25) < 1e-9
        assert abs(weights["v2"] - 0.25) < 1e-9

    def test_all_explicit_sum_to_1(self):
        v0 = VerifierDefinition(**_valid_verifier(name="v0", weight=0.3))
        v1 = VerifierDefinition(**_valid_verifier(name="v1", weight=0.7))
        weights = effective_weights([v0, v1])
        assert weights["v0"] == 0.3
        assert weights["v1"] == 0.7

    def test_no_unweighted_returns_zero_default(self):
        v0 = VerifierDefinition(**_valid_verifier(name="v0", weight=1.0))
        v1 = VerifierDefinition(**_valid_verifier(name="v1", weight=1.0))
        # This won't happen in practice (sum > 1 rejected), but test the math
        weights = effective_weights([v0, v1])
        # explicit_sum = 2.0, unweighted = 0 -> default = (1-2)/0 -> no unweighted
        # default = 0 since unweighted is empty
        assert weights["v0"] == 1.0
        assert weights["v1"] == 1.0


# ── validate_weight_distribution ──────────────────────────────────────────────

class TestValidateWeightDistribution:
    def test_all_none_ok(self):
        specs = [
            VerifierDefinition(**_valid_verifier(name=f"v{i}"))
            for i in range(2)
        ]
        # No explicit weights -> no error
        validate_weight_distribution(specs)

    def test_sum_exceeds_1_raises(self):
        v0 = VerifierDefinition(**_valid_verifier(name="v0", weight=0.7))
        v1 = VerifierDefinition(**_valid_verifier(name="v1", weight=0.5))
        with pytest.raises(ValueError, match="sum to at most 1"):
            validate_weight_distribution([v0, v1])

    def test_all_weighted_must_sum_to_1(self):
        v0 = VerifierDefinition(**_valid_verifier(name="v0", weight=0.3))
        v1 = VerifierDefinition(**_valid_verifier(name="v1", weight=0.6))
        with pytest.raises(ValueError, match="must sum to 1"):
            validate_weight_distribution([v0, v1])

    def test_unweighted_needs_remaining(self):
        v0 = VerifierDefinition(**_valid_verifier(name="v0", weight=1.0))
        v1 = VerifierDefinition(**_valid_verifier(name="v1"))
        with pytest.raises(ValueError, match="remaining positive weight"):
            validate_weight_distribution([v0, v1])


# ── validate_deterministic_expected ──────────────────────────────────────────

class TestValidateDeterministicExpected:
    def test_numeric_comparison_requires_number(self):
        with pytest.raises(ValueError, match="must be a JSON number"):
            validate_deterministic_expected("x", "greater_than")

    def test_numeric_comparison_ok_with_number(self):
        validate_deterministic_expected(5, "greater_than")  # no raise
        validate_deterministic_expected(3.14, "less_than")

    def test_bool_not_number(self):
        with pytest.raises(ValueError):
            validate_deterministic_expected(True, "approx_equals")

    def test_in_array_requires_list(self):
        with pytest.raises(ValueError, match="must be an array"):
            validate_deterministic_expected(1, "in_array")

    def test_in_array_ok_with_list(self):
        validate_deterministic_expected([1, 2], "in_array")

    def test_regex_requires_string(self):
        with pytest.raises(ValueError, match="must be a regex string"):
            validate_deterministic_expected(123, "regex_match")

    def test_regex_ok_with_string(self):
        validate_deterministic_expected(r"\d+", "regex_match")

    def test_equals_no_restriction(self):
        validate_deterministic_expected("anything", "equals")
        validate_deterministic_expected(42, "equals")
        validate_deterministic_expected(None, "equals")


# ── VerifierMetadata ──────────────────────────────────────────────────────────

class TestVerifierMetadata:
    def test_valid(self):
        m = VerifierMetadata(how_justification="h", why_justification="w")
        assert m.how_justification == "h"
        assert m.tag is None
        assert m.weight is None

    def test_weight_range(self):
        m = VerifierMetadata(how_justification="h", why_justification="w",
                              weight=0.5)
        assert m.weight == 0.5

    def test_weight_zero_raises(self):
        with pytest.raises(ValidationError):
            VerifierMetadata(how_justification="h", why_justification="w",
                             weight=0.0)

    def test_weight_over_1_raises(self):
        with pytest.raises(ValidationError):
            VerifierMetadata(how_justification="h", why_justification="w",
                             weight=1.5)

    def test_extra_fields_allowed(self):
        m = VerifierMetadata(how_justification="h", why_justification="w",
                             custom_field="extra")
        # extra="allow" preserves unknown fields
        assert m.model_dump().get("custom_field") == "extra"


# ── DeterministicConfig path validation ────────────────────────────────────────

class TestDeterministicConfig:
    def test_valid_path(self):
        dc = DeterministicConfig(path="$.result", comparison="equals")
        assert dc.path == "$.result"

    def test_invalid_jsonpath_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            DeterministicConfig(path="invalid[", comparison="equals")
        assert "invalid JSONPath" in str(exc_info.value)

    def test_valid_jsonpath_filter(self):
        dc = DeterministicConfig(path="$.items[?(@.active)]",
                                  comparison="equals")
        assert dc.path == "$.items[?(@.active)]"


# ── Comparison type ────────────────────────────────────────────────────────────

class TestComparisonType:
    def test_all_comparisons_accepted(self):
        comparisons = [
            "equals", "not_equals", "approx_equals", "greater_than",
            "less_than", "greater_than_equal", "less_than_equal",
            "contains", "not_contains", "regex_match", "not_regex_match",
            "in_array", "not_in_array", "length_equals",
        ]
        for c in comparisons:
            dc = DeterministicConfig(path="$.x", comparison=c)
            assert dc.comparison == c

    def test_invalid_comparison_rejected(self):
        with pytest.raises(ValidationError):
            DeterministicConfig(path="$.x", comparison="bogus")

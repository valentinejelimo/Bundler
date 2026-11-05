"""Unit tests for verifier_lib.compare — the comparison engine.

Tests compare() for all 13 comparators (equals, not_equals, approx_equals,
contains, not_contains, regex_match, not_regex_match, in_array, not_in_array,
length_equals, greater_than, less_than, greater_than_equal, less_than_equal)
and extract_jsonpath_values. Tests matching/non-matching cases and edge cases
(None, empty, type mismatches). jsonpath_ng is installed.
"""
from __future__ import annotations

import math
import pytest

from verifier_lib.compare import (
    actual_from_matches, assert_list, assert_length_target, assert_number_expected,
    assert_number_pair, assert_string, compare, contains,
    extract_jsonpath_values, is_json_number,
)
from verifier_lib.models import Tolerance


# ── equals ────────────────────────────────────────────────────────────────────

class TestEquals:
    def test_equal_numbers(self):
        assert compare(actual=1, expected=1, comparison="equals") is True

    def test_unequal_numbers(self):
        assert compare(actual=1, expected=2, comparison="equals") is False

    def test_equal_strings(self):
        assert compare(actual="abc", expected="abc", comparison="equals") is True

    def test_unequal_strings(self):
        assert compare(actual="abc", expected="abd", comparison="equals") is False

    def test_equal_lists(self):
        assert compare(actual=[1, 2], expected=[1, 2], comparison="equals") is True

    def test_none_equal_none(self):
        assert compare(actual=None, expected=None, comparison="equals") is True

    def test_none_not_equal_value(self):
        assert compare(actual=None, expected=1, comparison="equals") is False

    def test_bool_vs_int(self):
        # In Python True == 1 is True
        assert compare(actual=True, expected=1, comparison="equals") is True


# ── not_equals ────────────────────────────────────────────────────────────────

class TestNotEquals:
    def test_different_values(self):
        assert compare(actual=1, expected=2, comparison="not_equals") is True

    def test_same_values(self):
        assert compare(actual=1, expected=1, comparison="not_equals") is False

    def test_none_vs_value(self):
        assert compare(actual=None, expected=1, comparison="not_equals") is True


# ── approx_equals ─────────────────────────────────────────────────────────────

class TestApproxEquals:
    def test_close_numbers(self):
        assert compare(actual=1.0, expected=1.0000000001,
                       comparison="approx_equals") is True

    def test_far_numbers(self):
        assert compare(actual=1.0, expected=2.0,
                       comparison="approx_equals") is False

    def test_exact(self):
        assert compare(actual=5, expected=5, comparison="approx_equals") is True

    def test_with_tolerance_absolute(self):
        tol = Tolerance(absolute=0.5)
        assert compare(actual=10.0, expected=10.3,
                       comparison="approx_equals", tolerance=tol) is True

    def test_with_tolerance_relative(self):
        tol = Tolerance(relative=0.1)
        assert compare(actual=100.0, expected=105.0,
                       comparison="approx_equals", tolerance=tol) is True

    def test_string_raises(self):
        with pytest.raises(TypeError):
            compare(actual="x", expected=1, comparison="approx_equals")

    def test_none_raises(self):
        with pytest.raises(TypeError):
            compare(actual=None, expected=1, comparison="approx_equals")


# ── contains ──────────────────────────────────────────────────────────────────

class TestContains:
    def test_string_contains_substring(self):
        assert compare(actual="hello world", expected="world",
                       comparison="contains") is True

    def test_string_not_contains(self):
        assert compare(actual="hello", expected="xyz",
                       comparison="contains") is False

    def test_list_contains_element(self):
        assert compare(actual=[1, 2, 3], expected=2,
                       comparison="contains") is True

    def test_list_not_contains(self):
        assert compare(actual=[1, 2, 3], expected=4,
                       comparison="contains") is False

    def test_int_actual_raises(self):
        with pytest.raises(TypeError):
            compare(actual=42, expected=1, comparison="contains")


# ── not_contains ──────────────────────────────────────────────────────────────

class TestNotContains:
    def test_string_not_contains(self):
        assert compare(actual="hello", expected="xyz",
                       comparison="not_contains") is True

    def test_string_contains_false(self):
        assert compare(actual="hello world", expected="world",
                       comparison="not_contains") is False

    def test_list_not_contains(self):
        assert compare(actual=[1, 2], expected=3,
                       comparison="not_contains") is True


# ── regex_match ──────────────────────────────────────────────────────────────

class TestRegexMatch:
    def test_match(self):
        assert compare(actual="hello123", expected=r"\d+",
                       comparison="regex_match") is True

    def test_no_match(self):
        assert compare(actual="hello", expected=r"\d+",
                       comparison="regex_match") is False

    def test_non_string_actual_raises(self):
        with pytest.raises(TypeError):
            compare(actual=42, expected=r"\d+", comparison="regex_match")

    def test_empty_string(self):
        assert compare(actual="", expected=r"^$",
                       comparison="regex_match") is True


# ── not_regex_match ───────────────────────────────────────────────────────────

class TestNotRegexMatch:
    def test_no_match_is_true(self):
        assert compare(actual="hello", expected=r"\d+",
                       comparison="not_regex_match") is True

    def test_match_is_false(self):
        assert compare(actual="hello123", expected=r"\d+",
                       comparison="not_regex_match") is False


# ── in_array ──────────────────────────────────────────────────────────────────

class TestInArray:
    def test_in_array(self):
        assert compare(actual=2, expected=[1, 2, 3],
                       comparison="in_array") is True

    def test_not_in_array(self):
        assert compare(actual=4, expected=[1, 2, 3],
                       comparison="in_array") is False

    def test_non_list_expected_raises(self):
        with pytest.raises(TypeError):
            compare(actual=1, expected="not a list", comparison="in_array")

    def test_string_in_string_list(self):
        assert compare(actual="a", expected=["a", "b"],
                       comparison="in_array") is True


# ── not_in_array ───────────────────────────────────────────────────────────────

class TestNotInArray:
    def test_not_in(self):
        assert compare(actual=4, expected=[1, 2, 3],
                       comparison="not_in_array") is True

    def test_in_is_false(self):
        assert compare(actual=1, expected=[1, 2, 3],
                       comparison="not_in_array") is False

    def test_non_list_raises(self):
        with pytest.raises(TypeError):
            compare(actual=1, expected=42, comparison="not_in_array")


# ── length_equals ─────────────────────────────────────────────────────────────

class TestLengthEquals:
    def test_list_length(self):
        assert compare(actual=[1, 2, 3], expected=3,
                       comparison="length_equals") is True

    def test_list_length_wrong(self):
        assert compare(actual=[1, 2, 3], expected=2,
                       comparison="length_equals") is False

    def test_string_length(self):
        assert compare(actual="hello", expected=5,
                       comparison="length_equals") is True

    def test_dict_length(self):
        assert compare(actual={"a": 1, "b": 2}, expected=2,
                       comparison="length_equals") is True

    def test_int_actual_raises(self):
        with pytest.raises(TypeError):
            compare(actual=42, expected=1, comparison="length_equals")

    def test_non_number_expected_raises(self):
        with pytest.raises(TypeError):
            compare(actual=[1, 2], expected="two",
                    comparison="length_equals")


# ── greater_than / less_than ──────────────────────────────────────────────────

class TestGreaterThan:
    def test_greater(self):
        assert compare(actual=5, expected=3, comparison="greater_than") is True

    def test_not_greater(self):
        assert compare(actual=3, expected=5, comparison="greater_than") is False

    def test_equal_not_greater(self):
        assert compare(actual=5, expected=5, comparison="greater_than") is False

    def test_string_raises(self):
        with pytest.raises(TypeError):
            compare(actual="5", expected=3, comparison="greater_than")


class TestLessThan:
    def test_less(self):
        assert compare(actual=3, expected=5, comparison="less_than") is True

    def test_not_less(self):
        assert compare(actual=5, expected=3, comparison="less_than") is False

    def test_equal_not_less(self):
        assert compare(actual=5, expected=5, comparison="less_than") is False


# ── greater_than_equal / less_than_equal ───────────────────────────────────────

class TestGreaterEqual:
    def test_greater(self):
        assert compare(actual=5, expected=3,
                       comparison="greater_than_equal") is True

    def test_equal(self):
        assert compare(actual=5, expected=5,
                       comparison="greater_than_equal") is True

    def test_less(self):
        assert compare(actual=3, expected=5,
                       comparison="greater_than_equal") is False


class TestLessEqual:
    def test_less(self):
        assert compare(actual=3, expected=5,
                       comparison="less_than_equal") is True

    def test_equal(self):
        assert compare(actual=5, expected=5,
                       comparison="less_than_equal") is True

    def test_greater(self):
        assert compare(actual=7, expected=5,
                       comparison="less_than_equal") is False


# ── unsupported comparison ────────────────────────────────────────────────────

class TestUnsupported:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="unsupported comparison"):
            compare(actual=1, expected=1, comparison="bogus_comparator")


# ── is_json_number ─────────────────────────────────────────────────────────────

class TestIsJsonNumber:
    def test_int(self):
        assert is_json_number(42) is True

    def test_float(self):
        assert is_json_number(3.14) is True

    def test_bool_not_number(self):
        assert is_json_number(True) is False
        assert is_json_number(False) is False

    def test_string_not_number(self):
        assert is_json_number("42") is False

    def test_none_not_number(self):
        assert is_json_number(None) is False


# ── extract_jsonpath_values ───────────────────────────────────────────────────

class TestExtractJsonpathValues:
    def test_root_path(self):
        data = {"a": 1}
        assert extract_jsonpath_values(data, "$") == [{"a": 1}]

    def test_simple_path(self):
        data = {"name": "test", "value": 42}
        assert extract_jsonpath_values(data, "$.name") == ["test"]

    def test_nested_path(self):
        data = {"a": {"b": {"c": 10}}}
        assert extract_jsonpath_values(data, "$.a.b.c") == [10]

    def test_array_index(self):
        data = {"items": [10, 20, 30]}
        assert extract_jsonpath_values(data, "$.items[1]") == [20]

    def test_wildcard(self):
        data = {"items": [{"id": 1}, {"id": 2}]}
        result = extract_jsonpath_values(data, "$.items[*].id")
        assert result == [1, 2]

    def test_no_match(self):
        data = {"a": 1}
        assert extract_jsonpath_values(data, "$.nonexistent") == []

    def test_filter_expression(self):
        data = {"items": [{"id": 1, "active": True},
                          {"id": 2, "active": False}]}
        # jsonpath_ng.ext requires explicit ==true for boolean filter
        result = extract_jsonpath_values(data, "$.items[?(@.active==true)].id")
        assert result == [1]


# ── actual_from_matches ────────────────────────────────────────────────────────

class TestActualFromMatches:
    def test_empty_matches(self):
        assert actual_from_matches([]) is None

    def test_single_match(self):
        assert actual_from_matches([42]) == 42

    def test_multiple_matches(self):
        assert actual_from_matches([1, 2, 3]) == [1, 2, 3]


# ── assert helpers ─────────────────────────────────────────────────────────────

class TestAssertHelpers:
    def test_assert_list_valid(self):
        assert_list([1, 2], "in_array")  # no raise

    def test_assert_list_invalid(self):
        with pytest.raises(TypeError):
            assert_list("not a list", "in_array")

    def test_assert_string_valid(self):
        assert_string("hello", "regex_match")  # no raise

    def test_assert_string_invalid(self):
        with pytest.raises(TypeError):
            assert_string(42, "regex_match")

    def test_assert_number_pair_valid(self):
        assert_number_pair(1, 2, "greater_than")  # no raise

    def test_assert_number_pair_invalid(self):
        with pytest.raises(TypeError):
            assert_number_pair("1", 2, "greater_than")

    def test_assert_length_target_valid_list(self):
        assert_length_target([1, 2], "length_equals")

    def test_assert_length_target_valid_string(self):
        assert_length_target("hello", "length_equals")

    def test_assert_length_target_valid_dict(self):
        assert_length_target({"a": 1}, "length_equals")

    def test_assert_length_target_invalid(self):
        with pytest.raises(TypeError):
            assert_length_target(42, "length_equals")

    def test_assert_number_expected_valid(self):
        assert_number_expected(3, "length_equals")

    def test_assert_number_expected_invalid(self):
        with pytest.raises(TypeError):
            assert_number_expected("three", "length_equals")


# ── contains helper ────────────────────────────────────────────────────────────

class TestContainsHelper:
    def test_string_contains(self):
        assert contains("hello world", "world") is True

    def test_string_not_contains(self):
        assert contains("hello", "xyz") is False

    def test_list_contains(self):
        assert contains([1, 2, 3], 2) is True

    def test_list_not_contains(self):
        assert contains([1, 2, 3], 5) is False

    def test_int_raises(self):
        with pytest.raises(TypeError):
            contains(42, 1)

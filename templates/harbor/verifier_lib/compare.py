import math
import re
from typing import Any

# jsonpath_ng is imported lazily inside extract_jsonpath_values(), not at module
# level: the grading platform's upload-time rewardkit.discover() imports every
# tests/<dir>/*.py it finds (including this one, via tests/__verifier_lib/) in
# its own bare environment, which does not have jsonpath_ng installed -- a
# top-level import here breaks that discovery pass with "Unable to parse tests
# under tests/" even though grading itself runs fine inside this task's own
# container.

from .models import Tolerance


def extract_jsonpath_values(data: Any, path: str) -> list[Any]:
    """Extracts all values matching a JSONPath expression.

    Args:
        data: JSON-like data to inspect.
        path: JSONPath expression to evaluate.

    Returns:
        List of matched values. The root path ``$`` returns the full input as a
        single matched value.
    """
    if path == "$":
        return [data]
    from jsonpath_ng.ext import parse as jsonpath_parse  # lazy: see module note below
    return [match.value for match in jsonpath_parse(path).find(data)]


def actual_from_matches(matches: list[Any]) -> Any:
    """Converts JSONPath matches into the result actual value.

    Args:
        matches: JSONPath matched values.

    Returns:
        ``None`` for no matches, the single value for one match, or the full
        match list for multiple matches.
    """
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return matches


def compare(
    *,
    actual: Any,
    expected: Any,
    comparison: str,
    tolerance: Tolerance | None = None,
) -> bool:
    """Compares an actual value against an expected value.

    Args:
        actual: Value produced by a source command and optional JSONPath.
        expected: Expected value from the verifier assertion.
        comparison: Comparator name from the verifier schema.
        tolerance: Optional absolute/relative tolerance.

    Returns:
        True when the comparison succeeds; otherwise false.

    Raises:
        TypeError: If values have unsupported types for the comparator.
        ValueError: If the comparator name is unsupported.
    """
    if comparison == "equals":
        return actual == expected
    if comparison == "not_equals":
        return actual != expected
    if comparison == "approx_equals":
        assert_number_pair(actual, expected, comparison)
        return math.isclose(
            actual,
            expected,
            abs_tol=1e-9 if tolerance is None or tolerance.absolute is None else tolerance.absolute,
            rel_tol=1e-9 if tolerance is None or tolerance.relative is None else tolerance.relative,
        )
    if comparison == "contains":
        return contains(actual, expected)
    if comparison == "not_contains":
        return not contains(actual, expected)
    if comparison == "regex_match":
        assert_string(actual, comparison)
        return bool(re.search(str(expected), actual))
    if comparison == "not_regex_match":
        assert_string(actual, comparison)
        return not bool(re.search(str(expected), actual))
    if comparison == "in_array":
        assert_list(expected, comparison)
        return actual in expected
    if comparison == "not_in_array":
        assert_list(expected, comparison)
        return actual not in expected
    if comparison == "length_equals":
        assert_length_target(actual, comparison)
        assert_number_expected(expected, comparison)
        return len(actual) == expected
    if comparison == "greater_than":
        assert_number_pair(actual, expected, comparison)
        return actual > expected
    if comparison == "less_than":
        assert_number_pair(actual, expected, comparison)
        return actual < expected
    if comparison == "greater_than_equal":
        assert_number_pair(actual, expected, comparison)
        return actual >= expected
    if comparison == "less_than_equal":
        assert_number_pair(actual, expected, comparison)
        return actual <= expected
    raise ValueError(f"unsupported comparison: {comparison}")


def contains(actual: Any, expected: Any) -> bool:
    """Checks string substring or list element containment.

    Args:
        actual: Container value.
        expected: Expected contained value.

    Returns:
        True when ``actual`` contains ``expected``.

    Raises:
        TypeError: If ``actual`` is not a string or list.
    """
    if isinstance(actual, str):
        return str(expected) in actual
    if isinstance(actual, list):
        return expected in actual
    raise TypeError(f"contains requires string or list actual: actual_type={type(actual).__name__}")


def assert_number_pair(actual: Any, expected: Any, comparison: str) -> None:
    """Requires both actual and expected values to be JSON numbers.

    Args:
        actual: Actual value.
        expected: Expected value.
        comparison: Comparator name for error messages.

    Raises:
        TypeError: If either value is not a number.
    """
    if not is_json_number(actual) or not is_json_number(expected):
        raise TypeError(
            f"{comparison} requires JSON numbers: "
            f"actual_type={type(actual).__name__} expected_type={type(expected).__name__}"
        )


def is_json_number(value: Any) -> bool:
    """Checks whether a value is a JSON number in Python form.

    Args:
        value: Value to inspect.

    Returns:
        True for int/float values except booleans.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def assert_string(actual: Any, comparison: str) -> None:
    """Requires actual to be a string.

    Args:
        actual: Actual value.
        comparison: Comparator name for error messages.

    Raises:
        TypeError: If actual is not a string.
    """
    if not isinstance(actual, str):
        raise TypeError(f"{comparison} requires string actual: actual_type={type(actual).__name__}")


def assert_list(expected: Any, comparison: str) -> None:
    """Requires expected to be a list.

    Args:
        expected: Expected value.
        comparison: Comparator name for error messages.

    Raises:
        TypeError: If expected is not a list.
    """
    if not isinstance(expected, list):
        raise TypeError(f"{comparison} requires expected to be an array")


def assert_length_target(actual: Any, comparison: str) -> None:
    """Requires actual to have a stable length."""
    if not isinstance(actual, (list, dict, str)):
        raise TypeError(f"{comparison} requires list, object, or string actual: actual_type={type(actual).__name__}")


def assert_number_expected(expected: Any, comparison: str) -> None:
    """Requires expected to be a JSON number."""
    if not is_json_number(expected):
        raise TypeError(f"{comparison} requires numeric expected: expected_type={type(expected).__name__}")

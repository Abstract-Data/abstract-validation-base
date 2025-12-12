"""Tests for ValidationError and ValidationResult."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from abstract_validation_base.results import ValidationError, ValidationResult

from .conftest import field_names, messages, optional_strings

# =============================================================================
# ValidationError Unit Tests
# =============================================================================


class TestValidationErrorUnit:
    """Unit tests for ValidationError dataclass."""

    def test_validation_error_creation(self) -> None:
        """Test creating a ValidationError with all fields."""
        error = ValidationError(
            field="email",
            message="Invalid email format",
            value="not-an-email",
        )

        assert error.field == "email"
        assert error.message == "Invalid email format"
        assert error.value == "not-an-email"

    def test_validation_error_optional_value(self) -> None:
        """Test that value field defaults to None."""
        error = ValidationError(field="name", message="Required")

        assert error.field == "name"
        assert error.message == "Required"
        assert error.value is None

    def test_validation_error_equality(self) -> None:
        """Test that two identical errors are equal."""
        error1 = ValidationError(field="test", message="error", value="val")
        error2 = ValidationError(field="test", message="error", value="val")

        assert error1 == error2

    def test_validation_error_repr(self) -> None:
        """Test string representation."""
        error = ValidationError(field="x", message="y", value="z")
        repr_str = repr(error)

        assert "ValidationError" in repr_str
        assert "x" in repr_str
        assert "y" in repr_str


# =============================================================================
# ValidationResult Unit Tests
# =============================================================================


class TestValidationResultUnit:
    """Unit tests for ValidationResult dataclass."""

    def test_validation_result_valid(self) -> None:
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.errors == []

    def test_validation_result_invalid(self) -> None:
        """Test creating an invalid result."""
        result = ValidationResult(is_valid=False)

        assert result.is_valid is False
        assert result.errors == []

    def test_add_error_sets_invalid(self) -> None:
        """Test that add_error sets is_valid to False."""
        result = ValidationResult(is_valid=True)

        result.add_error("field", "message")

        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_add_error_with_value(self) -> None:
        """Test add_error with optional value."""
        result = ValidationResult(is_valid=True)

        result.add_error("age", "Must be positive", "-5")

        assert result.errors[0].field == "age"
        assert result.errors[0].message == "Must be positive"
        assert result.errors[0].value == "-5"

    def test_add_multiple_errors(self) -> None:
        """Test adding multiple errors."""
        result = ValidationResult(is_valid=True)

        result.add_error("field1", "error1")
        result.add_error("field2", "error2")
        result.add_error("field3", "error3")

        assert len(result.errors) == 3
        assert not result.is_valid

    def test_merge_two_valid_results(self) -> None:
        """Test merging two valid results stays valid."""
        result1 = ValidationResult(is_valid=True)
        result2 = ValidationResult(is_valid=True)

        merged = result1.merge(result2)

        assert merged.is_valid is True
        assert len(merged.errors) == 0
        assert merged is result1  # Returns self

    def test_merge_invalid_into_valid(self) -> None:
        """Test merging invalid result into valid makes it invalid."""
        result1 = ValidationResult(is_valid=True)
        result2 = ValidationResult(is_valid=True)
        result2.add_error("test", "error")

        result1.merge(result2)

        assert result1.is_valid is False
        assert len(result1.errors) == 1

    def test_merge_valid_into_invalid(self) -> None:
        """Test merging valid result into invalid keeps it invalid."""
        result1 = ValidationResult(is_valid=True)
        result1.add_error("existing", "error")
        result2 = ValidationResult(is_valid=True)

        result1.merge(result2)

        assert result1.is_valid is False
        assert len(result1.errors) == 1

    def test_merge_two_invalid_results(self) -> None:
        """Test merging two invalid results combines errors."""
        result1 = ValidationResult(is_valid=True)
        result1.add_error("field1", "error1")
        result2 = ValidationResult(is_valid=True)
        result2.add_error("field2", "error2")

        result1.merge(result2)

        assert result1.is_valid is False
        assert len(result1.errors) == 2

    def test_merge_returns_self_for_chaining(self) -> None:
        """Test that merge returns self for method chaining."""
        result1 = ValidationResult(is_valid=True)
        result2 = ValidationResult(is_valid=True)
        result3 = ValidationResult(is_valid=True)

        # Chain merges
        final = result1.merge(result2).merge(result3)

        assert final is result1


# =============================================================================
# ValidationError Property-Based Tests
# =============================================================================


class TestValidationErrorProperties:
    """Property-based tests for ValidationError."""

    @given(
        field=field_names,
        message=messages,
        value=optional_strings,
    )
    @settings(max_examples=100)
    def test_validation_error_roundtrip(self, field: str, message: str, value: str | None) -> None:
        """Any field/message/value combination creates valid error."""
        error = ValidationError(field=field, message=message, value=value)

        assert error.field == field
        assert error.message == message
        assert error.value == value


# =============================================================================
# ValidationResult Property-Based Tests
# =============================================================================


class TestValidationResultProperties:
    """Property-based tests for ValidationResult."""

    @given(
        field=field_names,
        message=messages,
    )
    @settings(max_examples=100)
    def test_add_error_always_invalidates(self, field: str, message: str) -> None:
        """Adding any error sets is_valid to False."""
        result = ValidationResult(is_valid=True)

        result.add_error(field, message)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == field
        assert result.errors[0].message == message

    @given(initial_valid=st.booleans())
    @settings(max_examples=50)
    def test_merge_identity(self, initial_valid: bool) -> None:
        """Merging with a valid result with no errors preserves state."""
        result = ValidationResult(is_valid=initial_valid)
        other = ValidationResult(is_valid=True)

        result.merge(other)

        assert result.is_valid == initial_valid

    @given(
        error_count=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50)
    def test_merge_invalid_propagates(self, error_count: int) -> None:
        """Merging invalid result always invalidates."""
        result = ValidationResult(is_valid=True)
        other = ValidationResult(is_valid=True)

        for i in range(error_count):
            other.add_error(f"field_{i}", f"message_{i}")

        result.merge(other)

        assert result.is_valid is False
        assert len(result.errors) == error_count

    @given(
        count1=st.integers(min_value=0, max_value=5),
        count2=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50)
    def test_merge_errors_accumulate(self, count1: int, count2: int) -> None:
        """Merged errors = sum of both error lists."""
        result1 = ValidationResult(is_valid=True)
        result2 = ValidationResult(is_valid=True)

        for i in range(count1):
            result1.add_error(f"r1_field_{i}", f"r1_msg_{i}")

        for i in range(count2):
            result2.add_error(f"r2_field_{i}", f"r2_msg_{i}")

        result1.merge(result2)

        assert len(result1.errors) == count1 + count2

    @given(
        results=st.lists(
            st.tuples(st.booleans(), st.integers(min_value=0, max_value=3)),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=50)
    def test_merge_chain_error_count(self, results: list[tuple[bool, int]]) -> None:
        """Chaining multiple merges accumulates all errors."""
        base = ValidationResult(is_valid=True)
        expected_errors = 0

        for _, error_count in results:
            other = ValidationResult(is_valid=True)
            for i in range(error_count):
                other.add_error(f"f_{i}", f"m_{i}")
            base.merge(other)
            expected_errors += error_count

        assert len(base.errors) == expected_errors


# =============================================================================
# ValidationResult Stateful Test
# =============================================================================


class ValidationResultStateMachine(RuleBasedStateMachine):
    """Stateful test for ValidationResult state transitions."""

    def __init__(self) -> None:
        super().__init__()
        self.result = ValidationResult(is_valid=True)
        self.expected_error_count = 0
        self.was_initially_valid = True

    @rule(
        field=st.text(
            min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
        ),
        message=st.text(min_size=1, max_size=50),
    )
    def add_error(self, field: str, message: str) -> None:
        """Add an error to the result."""
        self.result.add_error(field, message)
        self.expected_error_count += 1
        self.was_initially_valid = False

        # Verify state
        assert not self.result.is_valid
        assert len(self.result.errors) == self.expected_error_count

    @rule(
        other_has_errors=st.booleans(),
        error_count=st.integers(min_value=0, max_value=3),
    )
    def merge_result(self, other_has_errors: bool, error_count: int) -> None:
        """Merge another result into this one."""
        other = ValidationResult(is_valid=True)

        actual_count = error_count if other_has_errors else 0
        for i in range(actual_count):
            other.add_error(f"merged_field_{i}", f"merged_message_{i}")

        self.result.merge(other)
        self.expected_error_count += actual_count

        if actual_count > 0:
            self.was_initially_valid = False

        # Verify state
        assert len(self.result.errors) == self.expected_error_count

    @invariant()
    def error_count_matches(self) -> None:
        """Error count always matches expected."""
        assert len(self.result.errors) == self.expected_error_count

    @invariant()
    def validity_consistent(self) -> None:
        """is_valid is consistent with error count when errors were added."""
        if self.expected_error_count > 0:
            assert not self.result.is_valid


# Create the test class for pytest
TestValidationResultStateful = ValidationResultStateMachine.TestCase

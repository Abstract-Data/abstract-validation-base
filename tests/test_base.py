"""Tests for ValidationBase model."""

from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule
from pydantic_core import PydanticCustomError

from abstract_validation_base.process_log import ProcessLog

from .conftest import SampleModel, field_names, messages

# =============================================================================
# ValidationBase Unit Tests
# =============================================================================


class TestValidationBaseUnit:
    """Unit tests for ValidationBase."""

    def test_validation_base_subclass_creation(self) -> None:
        """Test creating a ValidationBase subclass."""
        model = SampleModel(name="test", value=42)

        assert model.name == "test"
        assert model.value == 42
        assert isinstance(model.process_log, ProcessLog)

    def test_process_log_excluded_from_dict(self) -> None:
        """Test that process_log is excluded from model_dump()."""
        model = SampleModel(name="test")

        data = model.model_dump()

        assert "process_log" not in data
        assert "name" in data
        assert "value" in data

    def test_process_log_excluded_from_json(self) -> None:
        """Test that process_log is excluded from model_dump_json()."""
        model = SampleModel(name="test")

        json_str = model.model_dump_json()

        assert "process_log" not in json_str
        assert "test" in json_str

    def test_has_errors_false_initially(self) -> None:
        """Test has_errors returns False when no errors logged."""
        model = SampleModel(name="test")

        assert model.has_errors is False

    def test_has_errors_true_after_add_error(self) -> None:
        """Test has_errors returns True after add_error()."""
        model = SampleModel(name="test")

        model.add_error("name", "Invalid name")

        assert model.has_errors is True

    def test_has_cleaning_false_initially(self) -> None:
        """Test has_cleaning returns False when no cleaning logged."""
        model = SampleModel(name="test")

        assert model.has_cleaning is False

    def test_has_cleaning_true_after_add_cleaning(self) -> None:
        """Test has_cleaning returns True after add_cleaning_process()."""
        model = SampleModel(name="test")

        model.add_cleaning_process("name", "  test  ", "test", "Trimmed whitespace")

        assert model.has_cleaning is True

    def test_error_count_zero_initially(self) -> None:
        """Test error_count is 0 initially."""
        model = SampleModel(name="test")

        assert model.error_count == 0

    def test_error_count_increases(self) -> None:
        """Test error_count increases with add_error()."""
        model = SampleModel(name="test")

        model.add_error("field1", "error1")
        assert model.error_count == 1

        model.add_error("field2", "error2")
        assert model.error_count == 2

    def test_cleaning_count_zero_initially(self) -> None:
        """Test cleaning_count is 0 initially."""
        model = SampleModel(name="test")

        assert model.cleaning_count == 0

    def test_cleaning_count_increases(self) -> None:
        """Test cleaning_count increases with add_cleaning_process()."""
        model = SampleModel(name="test")

        model.add_cleaning_process("name", "a", "b", "Reason 1")
        assert model.cleaning_count == 1

        model.add_cleaning_process("value", "1", "2", "Reason 2")
        assert model.cleaning_count == 2


class TestAddError:
    """Tests for add_error method."""

    def test_add_error_creates_entry(self) -> None:
        """Test that add_error creates a ProcessEntry."""
        model = SampleModel(name="test")

        model.add_error("name", "Invalid")

        assert len(model.process_log.errors) == 1
        entry = model.process_log.errors[0]
        assert entry.entry_type == "error"
        assert entry.field == "name"
        assert entry.message == "Invalid"

    def test_add_error_with_value(self) -> None:
        """Test add_error with value parameter."""
        model = SampleModel(name="test", value=-1)

        model.add_error("value", "Must be positive", value=-1)

        entry = model.process_log.errors[0]
        assert entry.original_value == "-1"

    def test_add_error_with_context(self) -> None:
        """Test add_error with context dict."""
        model = SampleModel(name="test")

        model.add_error("name", "Failed check", context={"rule": "length", "min": 3})

        entry = model.process_log.errors[0]
        assert entry.context == {"rule": "length", "min": 3}

    def test_add_error_with_none_value(self) -> None:
        """Test add_error with None value."""
        model = SampleModel(name="test")

        model.add_error("field", "Missing", value=None)

        entry = model.process_log.errors[0]
        assert entry.original_value is None

    def test_add_error_raise_exception(self) -> None:
        """Test add_error with raise_exception=True raises."""
        model = SampleModel(name="test")

        with pytest.raises(PydanticCustomError):
            model.add_error("name", "Critical error", raise_exception=True)

        # Error should still be logged
        assert model.error_count == 1

    def test_add_error_raise_exception_message(self) -> None:
        """Test that raised exception has correct message format."""
        model = SampleModel(name="test")

        with pytest.raises(PydanticCustomError) as exc_info:
            model.add_error("email", "Invalid format", value="bad", raise_exception=True)

        # The exception type should be validation_error
        assert exc_info.value.type == "validation_error"


class TestAddCleaningProcess:
    """Tests for add_cleaning_process method."""

    def test_add_cleaning_process_creates_entry(self) -> None:
        """Test that add_cleaning_process creates a ProcessEntry."""
        model = SampleModel(name="test")

        model.add_cleaning_process("name", "  test  ", "test", "Trimmed whitespace")

        assert len(model.process_log.cleaning) == 1
        entry = model.process_log.cleaning[0]
        assert entry.entry_type == "cleaning"
        assert entry.field == "name"
        assert entry.message == "Trimmed whitespace"
        assert entry.original_value == "  test  "
        assert entry.new_value == "test"

    def test_add_cleaning_process_default_operation_type(self) -> None:
        """Test default operation_type is 'cleaning'."""
        model = SampleModel(name="test")

        model.add_cleaning_process("name", "old", "new", "Changed")

        entry = model.process_log.cleaning[0]
        assert entry.context.get("operation_type") == "cleaning"

    def test_add_cleaning_process_custom_operation_type(self) -> None:
        """Test custom operation_type parameter."""
        model = SampleModel(name="test")

        model.add_cleaning_process(
            "name", "old", "new", "Normalized", operation_type="normalization"
        )

        entry = model.process_log.cleaning[0]
        assert entry.context.get("operation_type") == "normalization"

    def test_add_cleaning_process_none_values(self) -> None:
        """Test add_cleaning_process with None values."""
        model = SampleModel(name="test")

        model.add_cleaning_process("field", None, "new", "Set default")

        entry = model.process_log.cleaning[0]
        assert entry.original_value is None
        assert entry.new_value == "new"


class TestAuditLog:
    """Tests for audit_log method."""

    def test_audit_log_empty(self) -> None:
        """Test audit_log returns empty list when no entries."""
        model = SampleModel(name="test")

        log = model.audit_log()

        assert log == []

    def test_audit_log_cleaning_only(self) -> None:
        """Test audit_log with only cleaning entries."""
        model = SampleModel(name="test")
        model.add_cleaning_process("name", "old", "new", "Changed")

        log = model.audit_log()

        assert len(log) == 1
        assert log[0]["entry_type"] == "cleaning"

    def test_audit_log_errors_only(self) -> None:
        """Test audit_log with only error entries."""
        model = SampleModel(name="test")
        model.add_error("name", "Invalid")

        log = model.audit_log()

        assert len(log) == 1
        assert log[0]["entry_type"] == "error"

    def test_audit_log_combined(self) -> None:
        """Test audit_log combines cleaning and errors."""
        model = SampleModel(name="test")
        model.add_cleaning_process("name", "a", "b", "Changed")
        model.add_error("value", "Invalid")

        log = model.audit_log()

        assert len(log) == 2
        entry_types = {e["entry_type"] for e in log}
        assert entry_types == {"cleaning", "error"}

    def test_audit_log_with_source(self) -> None:
        """Test audit_log adds source field when provided."""
        model = SampleModel(name="test")
        model.add_error("field", "Error")

        log = model.audit_log(source="test_source")

        assert len(log) == 1
        assert log[0]["source"] == "test_source"

    def test_audit_log_sorted_by_timestamp(self) -> None:
        """Test audit_log entries are sorted by timestamp."""
        model = SampleModel(name="test")

        # Add entries with small delay to ensure different timestamps
        model.add_cleaning_process("field1", "a", "b", "First")
        time.sleep(0.01)
        model.add_error("field2", "Second")
        time.sleep(0.01)
        model.add_cleaning_process("field3", "c", "d", "Third")

        log = model.audit_log()

        # Should be sorted by timestamp
        timestamps = [e["timestamp"] for e in log]
        assert timestamps == sorted(timestamps)


class TestClearProcessLog:
    """Tests for clear_process_log method."""

    def test_clear_process_log_empties_both_lists(self) -> None:
        """Test clear_process_log empties cleaning and errors."""
        model = SampleModel(name="test")
        model.add_error("field", "Error")
        model.add_cleaning_process("field", "a", "b", "Clean")

        model.clear_process_log()

        assert model.error_count == 0
        assert model.cleaning_count == 0
        assert model.has_errors is False
        assert model.has_cleaning is False

    def test_clear_process_log_idempotent(self) -> None:
        """Test clear_process_log can be called multiple times."""
        model = SampleModel(name="test")

        model.clear_process_log()
        model.clear_process_log()

        assert model.error_count == 0
        assert model.cleaning_count == 0


# =============================================================================
# ValidationBase Property-Based Tests
# =============================================================================


class TestValidationBaseProperties:
    """Property-based tests for ValidationBase."""

    @given(
        error_count=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50)
    def test_error_count_matches_list(self, error_count: int) -> None:
        """error_count == len(process_log.errors)."""
        model = SampleModel(name="test")

        for i in range(error_count):
            model.add_error(f"field_{i}", f"error_{i}")

        assert model.error_count == len(model.process_log.errors)
        assert model.error_count == error_count

    @given(
        cleaning_count=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50)
    def test_cleaning_count_matches_list(self, cleaning_count: int) -> None:
        """cleaning_count == len(process_log.cleaning)."""
        model = SampleModel(name="test")

        for i in range(cleaning_count):
            model.add_cleaning_process(f"field_{i}", f"old_{i}", f"new_{i}", f"reason_{i}")

        assert model.cleaning_count == len(model.process_log.cleaning)
        assert model.cleaning_count == cleaning_count

    @given(
        error_count=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50)
    def test_has_errors_consistency(self, error_count: int) -> None:
        """has_errors == (error_count > 0)."""
        model = SampleModel(name="test")

        for i in range(error_count):
            model.add_error(f"field_{i}", f"error_{i}")

        assert model.has_errors == (model.error_count > 0)

    @given(
        cleaning_count=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50)
    def test_has_cleaning_consistency(self, cleaning_count: int) -> None:
        """has_cleaning == (cleaning_count > 0)."""
        model = SampleModel(name="test")

        for i in range(cleaning_count):
            model.add_cleaning_process(f"field_{i}", f"old_{i}", f"new_{i}", f"reason_{i}")

        assert model.has_cleaning == (model.cleaning_count > 0)

    @given(
        error_count=st.integers(min_value=0, max_value=5),
        cleaning_count=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50)
    def test_audit_log_contains_all_entries(self, error_count: int, cleaning_count: int) -> None:
        """audit_log length == cleaning + errors."""
        model = SampleModel(name="test")

        for i in range(error_count):
            model.add_error(f"e_field_{i}", f"error_{i}")

        for i in range(cleaning_count):
            model.add_cleaning_process(f"c_field_{i}", f"old_{i}", f"new_{i}", f"reason_{i}")

        log = model.audit_log()

        assert len(log) == error_count + cleaning_count

    @given(
        operations=st.lists(
            st.tuples(
                st.sampled_from(["error", "cleaning"]),
                field_names,
                messages,
            ),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=30)
    def test_audit_log_sorted(self, operations: list[tuple[str, str, str]]) -> None:
        """audit_log always sorted by timestamp."""
        model = SampleModel(name="test")

        for op_type, field, msg in operations:
            if op_type == "error":
                model.add_error(field, msg)
            else:
                model.add_cleaning_process(field, "old", "new", msg)
            # Small delay to ensure different timestamps
            time.sleep(0.001)

        log = model.audit_log()

        timestamps = [e["timestamp"] for e in log]
        assert timestamps == sorted(timestamps)


# =============================================================================
# ValidationBase Stateful Test
# =============================================================================


class ValidationBaseStateMachine(RuleBasedStateMachine):
    """Stateful test for ValidationBase operations."""

    def __init__(self) -> None:
        super().__init__()
        self.model = SampleModel(name="test")
        self.expected_errors = 0
        self.expected_cleaning = 0

    @rule(
        field=st.text(
            min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
        ),
        message=st.text(min_size=1, max_size=50),
    )
    def add_error(self, field: str, message: str) -> None:
        """Add an error to the model."""
        self.model.add_error(field, message)
        self.expected_errors += 1

        assert self.model.error_count == self.expected_errors
        assert self.model.has_errors

    @rule(
        field=st.text(
            min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
        ),
        original=st.text(max_size=30),
        new=st.text(max_size=30),
        reason=st.text(min_size=1, max_size=50),
    )
    def add_cleaning(self, field: str, original: str, new: str, reason: str) -> None:
        """Add a cleaning operation to the model."""
        self.model.add_cleaning_process(field, original, new, reason)
        self.expected_cleaning += 1

        assert self.model.cleaning_count == self.expected_cleaning
        assert self.model.has_cleaning

    @rule()
    def clear_log(self) -> None:
        """Clear the process log."""
        self.model.clear_process_log()
        self.expected_errors = 0
        self.expected_cleaning = 0

        assert self.model.error_count == 0
        assert self.model.cleaning_count == 0
        assert not self.model.has_errors
        assert not self.model.has_cleaning

    @rule()
    def get_audit_log(self) -> None:
        """Get the audit log and verify it."""
        log = self.model.audit_log()

        assert len(log) == self.expected_errors + self.expected_cleaning

        # Verify sorted by timestamp
        timestamps = [e["timestamp"] for e in log]
        assert timestamps == sorted(timestamps)

    @invariant()
    def counts_match(self) -> None:
        """Error and cleaning counts always match list lengths."""
        assert self.model.error_count == len(self.model.process_log.errors)
        assert self.model.cleaning_count == len(self.model.process_log.cleaning)

    @invariant()
    def has_properties_consistent(self) -> None:
        """has_errors and has_cleaning are consistent with counts."""
        assert self.model.has_errors == (self.model.error_count > 0)
        assert self.model.has_cleaning == (self.model.cleaning_count > 0)

    @invariant()
    def expected_counts_match(self) -> None:
        """Tracked expected counts match actual."""
        assert self.model.error_count == self.expected_errors
        assert self.model.cleaning_count == self.expected_cleaning


# Create the test class for pytest
TestValidationBaseStateful = ValidationBaseStateMachine.TestCase

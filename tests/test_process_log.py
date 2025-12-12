"""Tests for ProcessEntry and ProcessLog models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from abstract_validation_base.process_log import ProcessEntry, ProcessLog

from .conftest import context_dicts, entry_types, field_names, messages, optional_strings

# =============================================================================
# ProcessEntry Unit Tests
# =============================================================================


class TestProcessEntryUnit:
    """Unit tests for ProcessEntry."""

    def test_process_entry_creation_cleaning(self) -> None:
        """Test creating a cleaning entry with all fields."""
        entry = ProcessEntry(
            entry_type="cleaning",
            field="name",
            message="Trimmed whitespace",
            original_value="  test  ",
            new_value="test",
            context={"operation_type": "trimming"},
        )

        assert entry.entry_type == "cleaning"
        assert entry.field == "name"
        assert entry.message == "Trimmed whitespace"
        assert entry.original_value == "  test  "
        assert entry.new_value == "test"
        assert entry.context == {"operation_type": "trimming"}
        assert entry.timestamp  # Should be auto-generated

    def test_process_entry_creation_error(self) -> None:
        """Test creating an error entry."""
        entry = ProcessEntry(
            entry_type="error",
            field="email",
            message="Invalid email format",
        )

        assert entry.entry_type == "error"
        assert entry.field == "email"
        assert entry.message == "Invalid email format"
        assert entry.original_value is None
        assert entry.new_value is None

    def test_process_entry_defaults(self) -> None:
        """Test that defaults are applied correctly."""
        entry = ProcessEntry(
            entry_type="cleaning",
            field="test",
            message="test message",
        )

        assert entry.original_value is None
        assert entry.new_value is None
        assert entry.context == {}
        # Timestamp should be auto-generated
        assert entry.timestamp is not None
        assert len(entry.timestamp) > 0

    def test_process_entry_timestamp_is_iso_format(self) -> None:
        """Test that timestamp is valid ISO format."""
        entry = ProcessEntry(
            entry_type="cleaning",
            field="test",
            message="test",
        )

        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(entry.timestamp)
        assert isinstance(parsed, datetime)

    def test_process_entry_model_dump(self) -> None:
        """Test serialization with model_dump."""
        entry = ProcessEntry(
            entry_type="error",
            field="value",
            message="Invalid",
            original_value="bad",
            context={"code": 123},
        )

        data = entry.model_dump()
        assert data["entry_type"] == "error"
        assert data["field"] == "value"
        assert data["message"] == "Invalid"
        assert data["original_value"] == "bad"
        assert data["context"] == {"code": 123}


# =============================================================================
# ProcessLog Unit Tests
# =============================================================================


class TestProcessLogUnit:
    """Unit tests for ProcessLog."""

    def test_process_log_empty(self) -> None:
        """Test that a new ProcessLog has empty lists."""
        log = ProcessLog()

        assert log.cleaning == []
        assert log.errors == []

    def test_process_log_add_cleaning_entry(self) -> None:
        """Test adding entries to cleaning list."""
        log = ProcessLog()
        entry = ProcessEntry(
            entry_type="cleaning",
            field="name",
            message="Normalized",
        )

        log.cleaning.append(entry)

        assert len(log.cleaning) == 1
        assert log.cleaning[0] == entry

    def test_process_log_add_error_entry(self) -> None:
        """Test adding entries to errors list."""
        log = ProcessLog()
        entry = ProcessEntry(
            entry_type="error",
            field="email",
            message="Invalid format",
        )

        log.errors.append(entry)

        assert len(log.errors) == 1
        assert log.errors[0] == entry

    def test_process_log_multiple_entries(self) -> None:
        """Test adding multiple entries to both lists."""
        log = ProcessLog()

        for i in range(3):
            log.cleaning.append(
                ProcessEntry(entry_type="cleaning", field=f"field_{i}", message=f"Cleaned {i}")
            )
            log.errors.append(
                ProcessEntry(entry_type="error", field=f"field_{i}", message=f"Error {i}")
            )

        assert len(log.cleaning) == 3
        assert len(log.errors) == 3

    def test_process_log_clear(self) -> None:
        """Test clearing lists."""
        log = ProcessLog()
        log.cleaning.append(ProcessEntry(entry_type="cleaning", field="test", message="test"))
        log.errors.append(ProcessEntry(entry_type="error", field="test", message="test"))

        log.cleaning.clear()
        log.errors.clear()

        assert log.cleaning == []
        assert log.errors == []


# =============================================================================
# ProcessEntry Property-Based Tests
# =============================================================================


class TestProcessEntryProperties:
    """Property-based tests for ProcessEntry."""

    @given(
        entry_type=entry_types,
        field=field_names,
        message=messages,
    )
    @settings(max_examples=100)
    def test_process_entry_arbitrary_fields(
        self, entry_type: str, field: str, message: str
    ) -> None:
        """Any valid field/message combination creates a valid entry."""
        entry = ProcessEntry(
            entry_type=entry_type,
            field=field,
            message=message,
        )

        assert entry.entry_type == entry_type
        assert entry.field == field
        assert entry.message == message

    @given(
        entry_type=entry_types,
        field=field_names,
        message=messages,
    )
    @settings(max_examples=50)
    def test_process_entry_timestamp_format(
        self, entry_type: str, field: str, message: str
    ) -> None:
        """Timestamp is always valid ISO format."""
        entry = ProcessEntry(
            entry_type=entry_type,
            field=field,
            message=message,
        )

        # Should always be parseable as ISO format
        parsed = datetime.fromisoformat(entry.timestamp)
        assert isinstance(parsed, datetime)

    @given(
        entry_type=entry_types,
        field=field_names,
        message=messages,
        original_value=optional_strings,
        new_value=optional_strings,
        context=context_dicts,
    )
    @settings(max_examples=100)
    def test_process_entry_all_fields_roundtrip(
        self,
        entry_type: str,
        field: str,
        message: str,
        original_value: str | None,
        new_value: str | None,
        context: dict[str, Any],
    ) -> None:
        """Entry can be created with any combination of valid fields and serialized."""
        entry = ProcessEntry(
            entry_type=entry_type,
            field=field,
            message=message,
            original_value=original_value,
            new_value=new_value,
            context=context,
        )

        # Serialize and verify
        data = entry.model_dump()
        assert data["entry_type"] == entry_type
        assert data["field"] == field
        assert data["message"] == message
        assert data["original_value"] == original_value
        assert data["new_value"] == new_value
        assert data["context"] == context


# =============================================================================
# ProcessLog Property-Based Tests
# =============================================================================


class TestProcessLogProperties:
    """Property-based tests for ProcessLog."""

    @given(
        entries=st.lists(
            st.fixed_dictionaries(
                {
                    "entry_type": entry_types,
                    "field": field_names,
                    "message": messages,
                }
            ),
            max_size=20,
        )
    )
    @settings(max_examples=50)
    def test_process_log_entry_preservation(self, entries: list[dict[str, str]]) -> None:
        """Entries added are retrievable unchanged."""
        log = ProcessLog()

        for entry_data in entries:
            entry = ProcessEntry(
                entry_type=entry_data["entry_type"],
                field=entry_data["field"],
                message=entry_data["message"],
            )
            if entry_data["entry_type"] == "cleaning":
                log.cleaning.append(entry)
            else:
                log.errors.append(entry)

        # Count should match
        cleaning_count = sum(1 for e in entries if e["entry_type"] == "cleaning")
        error_count = sum(1 for e in entries if e["entry_type"] == "error")

        assert len(log.cleaning) == cleaning_count
        assert len(log.errors) == error_count

    @given(
        cleaning_count=st.integers(min_value=0, max_value=10),
        error_count=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50)
    def test_process_log_counts_independent(self, cleaning_count: int, error_count: int) -> None:
        """Cleaning and error lists maintain independent counts."""
        log = ProcessLog()

        for i in range(cleaning_count):
            log.cleaning.append(
                ProcessEntry(entry_type="cleaning", field=f"c{i}", message=f"clean {i}")
            )

        for i in range(error_count):
            log.errors.append(ProcessEntry(entry_type="error", field=f"e{i}", message=f"error {i}"))

        assert len(log.cleaning) == cleaning_count
        assert len(log.errors) == error_count

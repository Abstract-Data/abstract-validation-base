"""Tests for output writers: CSVFailedWriter, JSONLinesFailedWriter, AuditReportWriter."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from abstract_validation_base.results import ValidationResult
from abstract_validation_base.runner import RowResult
from abstract_validation_base.writers import (
    AuditReportWriter,
    CSVFailedWriter,
    FailedRecordWriter,
    JSONLinesFailedWriter,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .conftest import SampleModel

# =============================================================================
# Test Fixtures
# =============================================================================


def make_failed_result(
    row_index: int,
    raw_data: dict[str, Any],
    pydantic_errors: list[dict[str, Any]] | None = None,
    validator_errors: list[tuple[str, str]] | None = None,
) -> RowResult[SampleModel]:
    """Create a failed RowResult for testing."""
    result: RowResult[SampleModel] = RowResult(
        row_index=row_index,
        raw_data=raw_data,
        pydantic_errors=pydantic_errors or [],
    )

    if validator_errors:
        validator_result = ValidationResult(is_valid=False)
        for field, msg in validator_errors:
            validator_result.add_error(field, msg)
        result.validator_result = validator_result

    return result


def failed_results_iterator(count: int = 3) -> Iterator[RowResult[SampleModel]]:
    """Create an iterator of failed RowResults for testing."""
    for i in range(count):
        yield make_failed_result(
            row_index=i,
            raw_data={"name": f"test_{i}", "value": i * 10},
            pydantic_errors=[{"type": "value_error", "loc": ("name",), "msg": f"Error {i}"}],
        )


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Test that writers comply with FailedRecordWriter protocol."""

    def test_csv_writer_is_protocol_instance(self, temp_dir: Path) -> None:
        """Test CSVFailedWriter implements FailedRecordWriter."""
        writer = CSVFailedWriter(temp_dir / "test.csv")
        assert isinstance(writer, FailedRecordWriter)

    def test_json_writer_is_protocol_instance(self, temp_dir: Path) -> None:
        """Test JSONLinesFailedWriter implements FailedRecordWriter."""
        writer = JSONLinesFailedWriter(temp_dir / "test.jsonl")
        assert isinstance(writer, FailedRecordWriter)


# =============================================================================
# CSVFailedWriter Tests
# =============================================================================


class TestCSVFailedWriter:
    """Tests for CSVFailedWriter."""

    def test_write_all_creates_file(self, temp_dir: Path) -> None:
        """Test write_all creates the CSV file."""
        path = temp_dir / "failed.csv"
        writer = CSVFailedWriter(path)

        count = writer.write_all(failed_results_iterator(2))

        assert count == 2
        assert path.exists()

    def test_write_all_returns_count(self, temp_dir: Path) -> None:
        """Test write_all returns the number of records written."""
        writer = CSVFailedWriter(temp_dir / "failed.csv")

        count = writer.write_all(failed_results_iterator(5))

        assert count == 5

    def test_csv_contains_row_index(self, temp_dir: Path) -> None:
        """Test CSV includes _row_index column."""
        path = temp_dir / "failed.csv"
        writer = CSVFailedWriter(path)

        writer.write_all(iter([make_failed_result(42, {"name": "test"}, [])]))

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["_row_index"] == "42"

    def test_csv_contains_raw_data_fields(self, temp_dir: Path) -> None:
        """Test CSV includes raw data fields."""
        path = temp_dir / "failed.csv"
        writer = CSVFailedWriter(path)

        writer.write_all(iter([make_failed_result(0, {"name": "alice", "value": "123"}, [])]))

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["name"] == "alice"
        assert rows[0]["value"] == "123"

    def test_csv_contains_error_columns(self, temp_dir: Path) -> None:
        """Test CSV includes error columns."""
        path = temp_dir / "failed.csv"
        writer = CSVFailedWriter(path)

        result = make_failed_result(
            0,
            {"name": "test"},
            pydantic_errors=[
                {"type": "error1", "loc": ("field1",), "msg": "Error message 1"},
                {"type": "error2", "loc": ("field2",), "msg": "Error message 2"},
            ],
        )

        writer.write_all(iter([result]))

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["_error_1_field"] == "field1"
        assert rows[0]["_error_1_message"] == "Error message 1"
        assert rows[0]["_error_2_field"] == "field2"
        assert rows[0]["_error_2_message"] == "Error message 2"
        assert rows[0]["_error_count"] == "2"

    def test_csv_max_errors_per_row(self, temp_dir: Path) -> None:
        """Test CSV respects max_errors_per_row setting."""
        path = temp_dir / "failed.csv"
        writer = CSVFailedWriter(path, max_errors_per_row=1)

        result = make_failed_result(
            0,
            {"name": "test"},
            pydantic_errors=[
                {"type": "e1", "loc": ("f1",), "msg": "Msg 1"},
                {"type": "e2", "loc": ("f2",), "msg": "Msg 2"},
            ],
        )

        writer.write_all(iter([result]))

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Only 1 error column should exist
        assert "_error_1_field" in rows[0]
        assert "_error_2_field" not in rows[0]
        assert rows[0]["_error_count"] == "2"  # Still shows total count

    def test_csv_exclude_raw_data(self, temp_dir: Path) -> None:
        """Test CSV can exclude raw data fields."""
        path = temp_dir / "failed.csv"
        writer = CSVFailedWriter(path, include_raw_data=False)

        writer.write_all(iter([make_failed_result(0, {"name": "alice", "value": "123"}, [])]))

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert "name" not in rows[0]
        assert "value" not in rows[0]
        assert "_row_index" in rows[0]

    def test_context_manager_write_one(self, temp_dir: Path) -> None:
        """Test using context manager with write_one."""
        path = temp_dir / "failed.csv"

        with CSVFailedWriter(path) as writer:
            writer.write_one(make_failed_result(0, {"name": "test1"}, []))
            writer.write_one(make_failed_result(1, {"name": "test2"}, []))

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

    def test_write_one_without_context_raises(self, temp_dir: Path) -> None:
        """Test write_one raises error if not in context."""
        writer = CSVFailedWriter(temp_dir / "failed.csv")

        with pytest.raises(RuntimeError, match="Writer not opened"):
            writer.write_one(make_failed_result(0, {}, []))


# =============================================================================
# JSONLinesFailedWriter Tests
# =============================================================================


class TestJSONLinesFailedWriter:
    """Tests for JSONLinesFailedWriter."""

    def test_write_all_creates_file(self, temp_dir: Path) -> None:
        """Test write_all creates the JSON Lines file."""
        path = temp_dir / "failed.jsonl"
        writer = JSONLinesFailedWriter(path)

        count = writer.write_all(failed_results_iterator(2))

        assert count == 2
        assert path.exists()

    def test_write_all_returns_count(self, temp_dir: Path) -> None:
        """Test write_all returns the number of records written."""
        writer = JSONLinesFailedWriter(temp_dir / "failed.jsonl")

        count = writer.write_all(failed_results_iterator(5))

        assert count == 5

    def test_jsonl_format_valid(self, temp_dir: Path) -> None:
        """Test each line is valid JSON."""
        path = temp_dir / "failed.jsonl"
        writer = JSONLinesFailedWriter(path)

        writer.write_all(failed_results_iterator(3))

        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                assert isinstance(record, dict)

    def test_jsonl_contains_row_index(self, temp_dir: Path) -> None:
        """Test JSON Lines includes row_index."""
        path = temp_dir / "failed.jsonl"
        writer = JSONLinesFailedWriter(path)

        writer.write_all(iter([make_failed_result(42, {"name": "test"}, [])]))

        with open(path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert record["row_index"] == 42

    def test_jsonl_contains_errors(self, temp_dir: Path) -> None:
        """Test JSON Lines includes error details."""
        path = temp_dir / "failed.jsonl"
        writer = JSONLinesFailedWriter(path)

        result = make_failed_result(
            0,
            {"name": "test"},
            pydantic_errors=[{"type": "error1", "loc": ("field1",), "msg": "Error 1"}],
        )

        writer.write_all(iter([result]))

        with open(path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert "errors" in record
        assert len(record["errors"]) == 1
        assert record["errors"][0]["field"] == "field1"
        assert record["errors"][0]["message"] == "Error 1"

    def test_jsonl_contains_raw_data(self, temp_dir: Path) -> None:
        """Test JSON Lines includes raw_data."""
        path = temp_dir / "failed.jsonl"
        writer = JSONLinesFailedWriter(path)

        writer.write_all(iter([make_failed_result(0, {"name": "alice", "value": 123}, [])]))

        with open(path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert record["raw_data"]["name"] == "alice"
        assert record["raw_data"]["value"] == 123

    def test_jsonl_exclude_raw_data(self, temp_dir: Path) -> None:
        """Test JSON Lines can exclude raw_data."""
        path = temp_dir / "failed.jsonl"
        writer = JSONLinesFailedWriter(path, include_raw_data=False)

        writer.write_all(iter([make_failed_result(0, {"name": "alice"}, [])]))

        with open(path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert "raw_data" not in record

    def test_jsonl_contains_pydantic_errors(self, temp_dir: Path) -> None:
        """Test JSON Lines includes full Pydantic error details."""
        path = temp_dir / "failed.jsonl"
        writer = JSONLinesFailedWriter(path)

        result = make_failed_result(
            0,
            {"name": "test"},
            pydantic_errors=[
                {"type": "string_type", "loc": ("name",), "msg": "Input should be a string"}
            ],
        )

        writer.write_all(iter([result]))

        with open(path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert "pydantic_errors" in record
        assert record["pydantic_errors"][0]["type"] == "string_type"

    def test_jsonl_with_indent(self, temp_dir: Path) -> None:
        """Test JSON Lines with indentation for readability."""
        path = temp_dir / "failed.jsonl"
        writer = JSONLinesFailedWriter(path, indent=2)

        writer.write_all(iter([make_failed_result(0, {"name": "test"}, [])]))

        with open(path, encoding="utf-8") as f:
            content = f.read()

        # Indented JSON spans multiple lines per record
        assert "\n  " in content

    def test_context_manager_write_one(self, temp_dir: Path) -> None:
        """Test using context manager with write_one."""
        path = temp_dir / "failed.jsonl"

        with JSONLinesFailedWriter(path) as writer:
            writer.write_one(make_failed_result(0, {"name": "test1"}, []))
            writer.write_one(make_failed_result(1, {"name": "test2"}, []))

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2

    def test_write_one_without_context_raises(self, temp_dir: Path) -> None:
        """Test write_one raises error if not in context."""
        writer = JSONLinesFailedWriter(temp_dir / "failed.jsonl")

        with pytest.raises(RuntimeError, match="Writer not opened"):
            writer.write_one(make_failed_result(0, {}, []))


# =============================================================================
# AuditReportWriter Tests
# =============================================================================


class TestAuditReportWriter:
    """Tests for AuditReportWriter."""

    @pytest.fixture
    def sample_report(self) -> dict[str, Any]:
        """Create a sample audit report for testing."""
        return {
            "summary": {
                "total_rows": 1000,
                "valid_rows": 850,
                "error_rows": 150,
                "success_rate": "85.0%",
                "pydantic_failures": 100,
                "validator_failures": 50,
                "duration_ms": 1234.56,
            },
            "top_errors": [
                {"field": "email", "message": "Invalid format", "count": 50, "percentage": "33.3%"},
                {"field": "age", "message": "Must be positive", "count": 30, "percentage": "20.0%"},
            ],
            "failed_samples": [
                {"name": "test1", "email": "bad"},
                {"name": "test2", "age": -5},
            ],
        }

    def test_write_creates_file(self, temp_dir: Path, sample_report: dict[str, Any]) -> None:
        """Test write creates the summary CSV file."""
        path = temp_dir / "audit.csv"
        writer = AuditReportWriter(path)

        writer.write(sample_report)

        assert path.exists()

    def test_write_summary_section(self, temp_dir: Path, sample_report: dict[str, Any]) -> None:
        """Test summary section is written correctly."""
        path = temp_dir / "audit.csv"
        writer = AuditReportWriter(path)

        writer.write(sample_report)

        with open(path, encoding="utf-8") as f:
            content = f.read()

        assert "VALIDATION SUMMARY" in content
        assert "Total Rows" in content
        assert "1000" in content
        assert "Valid Rows" in content
        assert "850" in content
        assert "Success Rate" in content
        assert "85.0%" in content

    def test_write_errors_in_main_file(self, temp_dir: Path, sample_report: dict[str, Any]) -> None:
        """Test top errors are included in main file by default."""
        path = temp_dir / "audit.csv"
        writer = AuditReportWriter(path)

        writer.write(sample_report)

        with open(path, encoding="utf-8") as f:
            content = f.read()

        assert "TOP ERRORS" in content
        assert "email" in content
        assert "Invalid format" in content

    def test_write_errors_separate_file(
        self, temp_dir: Path, sample_report: dict[str, Any]
    ) -> None:
        """Test top errors can be written to separate file."""
        summary_path = temp_dir / "audit.csv"
        errors_path = temp_dir / "errors.csv"
        writer = AuditReportWriter(summary_path, errors_path=errors_path)

        writer.write(sample_report)

        assert errors_path.exists()

        # Errors should be in separate file, not main file
        with open(summary_path, encoding="utf-8") as f:
            summary_content = f.read()
        assert "TOP ERRORS" not in summary_content

        with open(errors_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["Field"] == "email"
        assert rows[0]["Count"] == "50"

    def test_write_samples(self, temp_dir: Path, sample_report: dict[str, Any]) -> None:
        """Test failed samples can be included."""
        path = temp_dir / "audit.csv"
        writer = AuditReportWriter(path, include_samples=True)

        writer.write(sample_report)

        samples_path = path.with_suffix(".samples.csv")
        assert samples_path.exists()

        with open(samples_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

    def test_write_samples_separate_path(
        self, temp_dir: Path, sample_report: dict[str, Any]
    ) -> None:
        """Test samples can be written to custom path."""
        path = temp_dir / "audit.csv"
        samples_path = temp_dir / "custom_samples.csv"
        writer = AuditReportWriter(path, include_samples=True, samples_path=samples_path)

        writer.write(sample_report)

        assert samples_path.exists()

    def test_write_empty_report(self, temp_dir: Path) -> None:
        """Test writing an empty report."""
        path = temp_dir / "audit.csv"
        writer = AuditReportWriter(path)

        empty_report: dict[str, Any] = {
            "summary": {},
            "top_errors": [],
            "failed_samples": [],
        }

        writer.write(empty_report)

        assert path.exists()

    def test_write_no_errors(self, temp_dir: Path) -> None:
        """Test report with no errors."""
        path = temp_dir / "audit.csv"
        writer = AuditReportWriter(path)

        report: dict[str, Any] = {
            "summary": {
                "total_rows": 100,
                "valid_rows": 100,
                "error_rows": 0,
                "success_rate": "100.0%",
                "pydantic_failures": 0,
                "validator_failures": 0,
                "duration_ms": 50.0,
            },
            "top_errors": [],
            "failed_samples": [],
        }

        writer.write(report)

        with open(path, encoding="utf-8") as f:
            content = f.read()

        # Should still have summary but no errors section
        assert "VALIDATION SUMMARY" in content
        assert "100.0%" in content


# =============================================================================
# Integration Tests
# =============================================================================


class TestWriterIntegration:
    """Integration tests for writers with runner output."""

    def test_csv_writer_with_validator_errors(self, temp_dir: Path) -> None:
        """Test CSV writer handles both Pydantic and validator errors."""
        path = temp_dir / "failed.csv"
        writer = CSVFailedWriter(path)

        result = make_failed_result(
            0,
            {"name": "test"},
            pydantic_errors=[{"type": "e1", "loc": ("f1",), "msg": "Pydantic error"}],
            validator_errors=[("f2", "Validator error")],
        )

        writer.write_all(iter([result]))

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["_error_count"] == "2"

    def test_jsonl_writer_with_validator_errors(self, temp_dir: Path) -> None:
        """Test JSON Lines writer handles both error types."""
        path = temp_dir / "failed.jsonl"
        writer = JSONLinesFailedWriter(path)

        result = make_failed_result(
            0,
            {"name": "test"},
            pydantic_errors=[{"type": "e1", "loc": ("f1",), "msg": "Pydantic error"}],
            validator_errors=[("f2", "Validator error")],
        )

        writer.write_all(iter([result]))

        with open(path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert len(record["errors"]) == 2

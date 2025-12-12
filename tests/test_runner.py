"""Tests for ValidationRunner, RowResult, and RunnerStats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from abstract_validation_base import ValidationEvent, ValidationEventType
from abstract_validation_base.results import ValidationResult
from abstract_validation_base.runner import RowResult, RunnerStats, ValidationRunner
from abstract_validation_base.validators import BaseValidator

from .conftest import SampleModel

if TYPE_CHECKING:
    from collections.abc import Iterator

# =============================================================================
# RowResult Unit Tests
# =============================================================================


class TestRowResult:
    """Unit tests for RowResult dataclass."""

    def test_row_result_valid_with_model(self) -> None:
        """Test RowResult is valid when it has a model and no errors."""
        model = SampleModel(name="test", value=42)
        result = RowResult(row_index=0, raw_data={"name": "test", "value": 42}, model=model)

        assert result.is_valid
        assert result.model is model
        assert result.row_index == 0
        assert result.error_summary == []

    def test_row_result_invalid_with_pydantic_errors(self) -> None:
        """Test RowResult is invalid when it has Pydantic errors."""
        result = RowResult(
            row_index=1,
            raw_data={"name": 123},
            pydantic_errors=[
                {"type": "string_type", "loc": ("name",), "msg": "Input should be a valid string"}
            ],
        )

        assert not result.is_valid
        assert result.model is None
        assert len(result.error_summary) == 1
        assert result.error_summary[0] == ("name", "Input should be a valid string")

    def test_row_result_invalid_with_validator_errors(self) -> None:
        """Test RowResult is invalid when validator_result has errors."""
        model = SampleModel(name="test", value=-5)
        validator_result = ValidationResult(is_valid=False)
        validator_result.add_error("value", "Value must be positive")

        result = RowResult(
            row_index=2,
            raw_data={"name": "test", "value": -5},
            model=model,
            validator_result=validator_result,
        )

        assert not result.is_valid
        assert len(result.error_summary) == 1
        assert result.error_summary[0] == ("value", "Value must be positive")

    def test_row_result_combined_errors(self) -> None:
        """Test RowResult combines both Pydantic and validator errors."""
        validator_result = ValidationResult(is_valid=False)
        validator_result.add_error("custom_field", "Custom error")

        result = RowResult(
            row_index=3,
            raw_data={"name": "test"},
            pydantic_errors=[
                {"type": "missing", "loc": ("required_field",), "msg": "Field required"}
            ],
            validator_result=validator_result,
        )

        assert not result.is_valid
        assert len(result.error_summary) == 2

    def test_error_summary_handles_missing_loc(self) -> None:
        """Test error_summary handles Pydantic errors without loc field."""
        result = RowResult(
            row_index=0,
            raw_data={},
            pydantic_errors=[{"msg": "General error"}],
        )

        errors = result.error_summary
        assert len(errors) == 1
        assert errors[0][0] == "unknown"  # Default when loc missing

    def test_error_summary_uses_type_when_msg_missing(self) -> None:
        """Test error_summary falls back to type when msg is missing."""
        result = RowResult(
            row_index=0,
            raw_data={},
            pydantic_errors=[{"type": "value_error", "loc": ("field",)}],
        )

        errors = result.error_summary
        assert errors[0][1] == "value_error"


# =============================================================================
# RunnerStats Unit Tests
# =============================================================================


class TestRunnerStats:
    """Unit tests for RunnerStats dataclass."""

    def test_default_stats(self) -> None:
        """Test default stats values."""
        stats = RunnerStats()

        assert stats.total_rows == 0
        assert stats.valid_rows == 0
        assert stats.pydantic_failures == 0
        assert stats.validator_failures == 0
        assert stats.error_rows == 0
        assert stats.success_rate == 0.0

    def test_error_rows_calculated(self) -> None:
        """Test error_rows property is calculated correctly."""
        stats = RunnerStats(total_rows=100, valid_rows=85)

        assert stats.error_rows == 15

    def test_success_rate_calculation(self) -> None:
        """Test success_rate calculation."""
        stats = RunnerStats(total_rows=100, valid_rows=80)

        assert stats.success_rate == 80.0

    def test_success_rate_zero_rows(self) -> None:
        """Test success_rate returns 0 when no rows."""
        stats = RunnerStats()

        assert stats.success_rate == 0.0

    def test_duration_ms_calculation(self) -> None:
        """Test duration_ms calculation."""
        stats = RunnerStats(start_time=1.0, end_time=2.5)

        assert stats.duration_ms == 1500.0

    def test_record_error(self) -> None:
        """Test recording error counts."""
        stats = RunnerStats()

        stats.record_error("email", "Invalid format")
        stats.record_error("email", "Invalid format")
        stats.record_error("name", "Required")

        assert stats.error_counts[("email", "Invalid format")] == 2
        assert stats.error_counts[("name", "Required")] == 1

    def test_top_errors_ordering(self) -> None:
        """Test top_errors returns errors in descending count order."""
        stats = RunnerStats()

        for _ in range(5):
            stats.record_error("field_a", "error_a")
        for _ in range(10):
            stats.record_error("field_b", "error_b")
        for _ in range(3):
            stats.record_error("field_c", "error_c")

        top = stats.top_errors(2)

        assert len(top) == 2
        assert top[0][0] == ("field_b", "error_b")
        assert top[0][1] == 10
        assert top[1][0] == ("field_a", "error_a")
        assert top[1][1] == 5

    def test_top_errors_percentage(self) -> None:
        """Test top_errors calculates percentage correctly."""
        stats = RunnerStats()

        stats.record_error("field", "error")
        stats.record_error("field", "error")
        stats.record_error("other", "msg")
        stats.record_error("other", "msg")

        top = stats.top_errors(10)

        # 50% each
        assert top[0][2] == 50.0
        assert top[1][2] == 50.0

    def test_add_failed_sample(self) -> None:
        """Test adding failed samples."""
        stats = RunnerStats(max_samples=2)

        stats.add_failed_sample({"name": "test1"})
        stats.add_failed_sample({"name": "test2"})
        stats.add_failed_sample({"name": "test3"})  # Should be ignored

        assert len(stats.failed_samples) == 2
        assert stats.failed_samples[0]["name"] == "test1"
        assert stats.failed_samples[1]["name"] == "test2"


# =============================================================================
# ValidationRunner Unit Tests
# =============================================================================


class RecordingObserver:
    """Observer that records all events."""

    def __init__(self) -> None:
        self.events: list[ValidationEvent] = []

    def on_event(self, event: ValidationEvent) -> None:
        self.events.append(event)


def make_data_iterator(items: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Create an iterator from a list of dicts."""
    return iter(items)


class TestValidationRunnerBasic:
    """Basic unit tests for ValidationRunner."""

    def test_runner_processes_valid_data(self) -> None:
        """Test runner processes valid data correctly."""
        data = [
            {"name": "alice", "value": 1},
            {"name": "bob", "value": 2},
            {"name": "charlie", "value": 3},
        ]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        results = list(runner.run())

        assert len(results) == 3
        assert all(r.is_valid for r in results)
        assert runner.stats.total_rows == 3
        assert runner.stats.valid_rows == 3
        assert runner.stats.error_rows == 0

    def test_runner_handles_pydantic_errors(self) -> None:
        """Test runner handles Pydantic validation errors."""
        data = [
            {"name": "valid", "value": 10},
            {"name": 123, "value": "not_an_int"},  # Invalid
            {"name": "another_valid", "value": 20},
        ]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        results = list(runner.run())

        assert len(results) == 3
        assert results[0].is_valid
        assert not results[1].is_valid
        assert results[2].is_valid

        assert runner.stats.valid_rows == 2
        assert runner.stats.pydantic_failures == 1

    def test_runner_with_custom_validators(self) -> None:
        """Test runner with custom validators."""
        data = [
            {"name": "test1", "value": 10},
            {"name": "test2", "value": -5},  # Should fail validator
        ]

        class NegativeValueValidator(BaseValidator[SampleModel]):
            @property
            def name(self) -> str:
                return "negative_check"

            def validate(self, item: SampleModel) -> ValidationResult:
                result = ValidationResult(is_valid=True)
                if item.value < 0:
                    result.add_error("value", "Value must not be negative")
                return result

        runner = ValidationRunner(
            make_data_iterator(data),
            SampleModel,
            validators=NegativeValueValidator(),
        )

        results = list(runner.run())

        assert results[0].is_valid
        assert not results[1].is_valid
        assert runner.stats.validator_failures == 1

    def test_runner_fail_fast(self) -> None:
        """Test runner stops on first error when fail_fast=True."""
        data = [
            {"name": "valid1"},
            {"name": 123},  # Invalid
            {"name": "valid2"},
            {"name": "valid3"},
        ]
        runner = ValidationRunner(
            make_data_iterator(data),
            SampleModel,
            fail_fast=True,
        )

        results = list(runner.run())

        assert len(results) == 2  # Stopped after the invalid row
        assert results[0].is_valid
        assert not results[1].is_valid

    def test_runner_run_collect_valid(self) -> None:
        """Test run_collect_valid yields only valid models."""
        data = [
            {"name": "valid1", "value": 1},
            {"name": 123},  # Invalid
            {"name": "valid2", "value": 2},
        ]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        valid_models = list(runner.run_collect_valid())

        assert len(valid_models) == 2
        assert all(isinstance(m, SampleModel) for m in valid_models)
        assert valid_models[0].name == "valid1"
        assert valid_models[1].name == "valid2"

    def test_runner_run_collect_failed(self) -> None:
        """Test run_collect_failed yields only failed results."""
        data = [
            {"name": "valid"},
            {"name": 123},  # Invalid
            {"name": "another_valid"},
        ]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        failed_results = list(runner.run_collect_failed())

        assert len(failed_results) == 1
        assert not failed_results[0].is_valid
        assert failed_results[0].row_index == 1


class TestValidationRunnerBatching:
    """Tests for ValidationRunner batching functionality."""

    def test_run_batch_valid_exact_batches(self) -> None:
        """Test batching with exact batch sizes."""
        data = [{"name": f"item{i}", "value": i} for i in range(6)]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        batches = list(runner.run_batch_valid(batch_size=2))

        assert len(batches) == 3
        assert all(len(b) == 2 for b in batches)

    def test_run_batch_valid_partial_last_batch(self) -> None:
        """Test batching with partial last batch."""
        data = [{"name": f"item{i}", "value": i} for i in range(5)]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        batches = list(runner.run_batch_valid(batch_size=2))

        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1

    def test_run_batch_valid_with_invalid_rows(self) -> None:
        """Test batching excludes invalid rows."""
        data = [
            {"name": "valid1"},
            {"name": 123},  # Invalid
            {"name": "valid2"},
            {"name": "valid3"},
        ]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        batches = list(runner.run_batch_valid(batch_size=2))

        # 3 valid items, batch_size=2 -> 2 batches (2 + 1)
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

    def test_run_batch_valid_emits_events(self) -> None:
        """Test that batching emits BATCH_STARTED and BATCH_COMPLETED events."""
        data = [{"name": f"item{i}"} for i in range(3)]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)
        observer = RecordingObserver()
        runner.add_observer(observer)

        list(runner.run_batch_valid(batch_size=2))

        batch_types = (ValidationEventType.BATCH_STARTED, ValidationEventType.BATCH_COMPLETED)
        batch_events = [e for e in observer.events if e.event_type in batch_types]

        # 2 batches: each has START + COMPLETE
        started = [e for e in batch_events if e.event_type == ValidationEventType.BATCH_STARTED]
        completed = [e for e in batch_events if e.event_type == ValidationEventType.BATCH_COMPLETED]

        # BATCH_STARTED only emitted when first item enters a batch
        assert len(started) >= 1
        assert len(completed) == 2


class TestValidationRunnerEvents:
    """Tests for ValidationRunner observer events."""

    def test_emits_validation_started(self) -> None:
        """Test that VALIDATION_STARTED event is emitted."""
        data = [{"name": "test"}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel, total_hint=100)
        observer = RecordingObserver()
        runner.add_observer(observer)

        list(runner.run())

        started_events = [
            e for e in observer.events if e.event_type == ValidationEventType.VALIDATION_STARTED
        ]

        assert len(started_events) == 1
        assert started_events[0].data["model_class"] == "SampleModel"
        assert started_events[0].data["total_hint"] == 100

    def test_emits_validation_completed(self) -> None:
        """Test that VALIDATION_COMPLETED event is emitted."""
        data = [{"name": "test1"}, {"name": "test2"}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)
        observer = RecordingObserver()
        runner.add_observer(observer)

        list(runner.run())

        completed_events = [
            e for e in observer.events if e.event_type == ValidationEventType.VALIDATION_COMPLETED
        ]

        assert len(completed_events) == 1
        assert "stats" in completed_events[0].data

    def test_emits_row_processed_events(self) -> None:
        """Test that ROW_PROCESSED events are emitted for each row."""
        data = [{"name": "valid"}, {"name": 123}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)
        observer = RecordingObserver()
        runner.add_observer(observer)

        list(runner.run())

        row_events = [
            e for e in observer.events if e.event_type == ValidationEventType.ROW_PROCESSED
        ]

        assert len(row_events) == 2
        assert row_events[0].data["row_index"] == 0
        assert row_events[0].data["is_valid"] is True
        assert row_events[1].data["row_index"] == 1
        assert row_events[1].data["is_valid"] is False

    def test_row_processed_includes_stats_snapshot(self) -> None:
        """Test that ROW_PROCESSED includes running stats."""
        data = [{"name": "test1"}, {"name": "test2"}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel, total_hint=10)
        observer = RecordingObserver()
        runner.add_observer(observer)

        list(runner.run())

        row_events = [
            e for e in observer.events if e.event_type == ValidationEventType.ROW_PROCESSED
        ]

        # After first row
        assert row_events[0].data["stats_snapshot"]["total"] == 1
        assert row_events[0].data["stats_snapshot"]["valid"] == 1
        assert row_events[0].data["stats_snapshot"]["total_hint"] == 10

        # After second row
        assert row_events[1].data["stats_snapshot"]["total"] == 2


class TestValidationRunnerAuditReport:
    """Tests for ValidationRunner audit_report method."""

    def test_audit_report_structure(self) -> None:
        """Test audit_report returns correct structure."""
        data = [{"name": "test"}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)
        list(runner.run())

        report = runner.audit_report()

        assert "summary" in report
        assert "top_errors" in report
        assert "failed_samples" in report

    def test_audit_report_summary(self) -> None:
        """Test audit_report summary contains correct metrics."""
        data = [
            {"name": "valid1"},
            {"name": 123},  # Invalid
            {"name": "valid2"},
        ]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)
        list(runner.run())

        report = runner.audit_report()
        summary = report["summary"]

        assert summary["total_rows"] == 3
        assert summary["valid_rows"] == 2
        assert summary["error_rows"] == 1
        assert summary["success_rate"] == "66.7%"
        assert summary["pydantic_failures"] == 1
        assert "duration_ms" in summary

    def test_audit_report_top_errors(self) -> None:
        """Test audit_report includes top errors."""
        data = [
            {"name": 123},
            {"name": 456},
            {"name": 789},
        ]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)
        list(runner.run())

        report = runner.audit_report()

        assert len(report["top_errors"]) > 0
        for error in report["top_errors"]:
            assert "field" in error
            assert "message" in error
            assert "count" in error
            assert "percentage" in error

    def test_audit_report_failed_samples(self) -> None:
        """Test audit_report includes failed samples."""
        data = [{"name": 123, "extra": "data"}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)
        list(runner.run())

        report = runner.audit_report()

        assert len(report["failed_samples"]) == 1
        assert report["failed_samples"][0]["name"] == 123


class TestValidationRunnerEdgeCases:
    """Edge case tests for ValidationRunner."""

    def test_empty_data(self) -> None:
        """Test runner handles empty data."""
        runner = ValidationRunner(make_data_iterator([]), SampleModel)

        results = list(runner.run())

        assert len(results) == 0
        assert runner.stats.total_rows == 0

    def test_generator_is_single_use(self) -> None:
        """Test that running twice uses the data only once."""
        data = [{"name": "test"}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        first_run = list(runner.run())
        second_run = list(runner.run())

        assert len(first_run) == 1
        assert len(second_run) == 0  # Iterator exhausted

    def test_stats_reset_between_runs(self) -> None:
        """Test that stats are reset on each run."""
        data = [{"name": "test1"}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        list(runner.run())
        assert runner.stats.total_rows == 1

        # Running again (empty iterator now)
        list(runner.run())
        # Stats should be reset (new RunnerStats instance)
        assert runner.stats.total_rows == 0

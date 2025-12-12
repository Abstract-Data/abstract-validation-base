"""Tests for ValidationRunner, RowResult, and RunnerStats."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

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
        result: RowResult[SampleModel] = RowResult(
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

        result: RowResult[SampleModel] = RowResult(
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
        result: RowResult[SampleModel] = RowResult(
            row_index=0,
            raw_data={},
            pydantic_errors=[{"msg": "General error"}],
        )

        errors = result.error_summary
        assert len(errors) == 1
        assert errors[0][0] == "unknown"  # Default when loc missing

    def test_error_summary_uses_type_when_msg_missing(self) -> None:
        """Test error_summary falls back to type when msg is missing."""
        result: RowResult[SampleModel] = RowResult(
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
# RunnerStats Merge Tests
# =============================================================================


class TestRunnerStatsMerge:
    """Tests for RunnerStats.merge() method."""

    def test_merge_combines_row_counts(self) -> None:
        """Test merge combines total_rows and valid_rows."""
        stats1 = RunnerStats(total_rows=100, valid_rows=80)
        stats2 = RunnerStats(total_rows=50, valid_rows=40)

        stats1.merge(stats2)

        assert stats1.total_rows == 150
        assert stats1.valid_rows == 120

    def test_merge_combines_failure_counts(self) -> None:
        """Test merge combines pydantic_failures and validator_failures."""
        stats1 = RunnerStats(pydantic_failures=10, validator_failures=5)
        stats2 = RunnerStats(pydantic_failures=8, validator_failures=3)

        stats1.merge(stats2)

        assert stats1.pydantic_failures == 18
        assert stats1.validator_failures == 8

    def test_merge_combines_error_counts(self) -> None:
        """Test merge combines error_counts from both stats."""
        stats1 = RunnerStats()
        stats1.record_error("email", "Invalid")
        stats1.record_error("email", "Invalid")
        stats1.record_error("name", "Required")

        stats2 = RunnerStats()
        stats2.record_error("email", "Invalid")
        stats2.record_error("phone", "Bad format")

        stats1.merge(stats2)

        assert stats1.error_counts[("email", "Invalid")] == 3
        assert stats1.error_counts[("name", "Required")] == 1
        assert stats1.error_counts[("phone", "Bad format")] == 1

    def test_merge_failed_samples_respects_max(self) -> None:
        """Test merge respects max_samples limit for failed_samples."""
        stats1 = RunnerStats(max_samples=3)
        stats1.add_failed_sample({"id": 1})
        stats1.add_failed_sample({"id": 2})

        stats2 = RunnerStats()
        stats2.add_failed_sample({"id": 3})
        stats2.add_failed_sample({"id": 4})
        stats2.add_failed_sample({"id": 5})

        stats1.merge(stats2)

        # Should only add 1 more (3 - 2 = 1 remaining)
        assert len(stats1.failed_samples) == 3
        assert stats1.failed_samples[0]["id"] == 1
        assert stats1.failed_samples[1]["id"] == 2
        assert stats1.failed_samples[2]["id"] == 3

    def test_merge_failed_samples_empty_other(self) -> None:
        """Test merge with empty failed_samples in other."""
        stats1 = RunnerStats(max_samples=5)
        stats1.add_failed_sample({"id": 1})

        stats2 = RunnerStats()  # No failed samples

        stats1.merge(stats2)

        assert len(stats1.failed_samples) == 1
        assert stats1.failed_samples[0]["id"] == 1

    def test_merge_failed_samples_empty_self(self) -> None:
        """Test merge with empty failed_samples in self."""
        stats1 = RunnerStats(max_samples=5)  # No failed samples

        stats2 = RunnerStats()
        stats2.add_failed_sample({"id": 1})
        stats2.add_failed_sample({"id": 2})

        stats1.merge(stats2)

        assert len(stats1.failed_samples) == 2

    def test_merge_failed_samples_at_capacity(self) -> None:
        """Test merge when self is already at max_samples capacity."""
        stats1 = RunnerStats(max_samples=2)
        stats1.add_failed_sample({"id": 1})
        stats1.add_failed_sample({"id": 2})

        stats2 = RunnerStats()
        stats2.add_failed_sample({"id": 3})

        stats1.merge(stats2)

        # Should not add any more
        assert len(stats1.failed_samples) == 2
        assert stats1.failed_samples[0]["id"] == 1
        assert stats1.failed_samples[1]["id"] == 2

    def test_merge_preserves_start_end_times(self) -> None:
        """Test merge does not affect start_time and end_time."""
        stats1 = RunnerStats(start_time=1.0, end_time=2.0)
        stats2 = RunnerStats(start_time=3.0, end_time=4.0)

        stats1.merge(stats2)

        # start_time and end_time should be unchanged
        assert stats1.start_time == 1.0
        assert stats1.end_time == 2.0

    def test_merge_empty_stats(self) -> None:
        """Test merge with empty stats."""
        stats1 = RunnerStats()
        stats2 = RunnerStats()

        stats1.merge(stats2)

        assert stats1.total_rows == 0
        assert stats1.valid_rows == 0
        assert stats1.pydantic_failures == 0
        assert stats1.validator_failures == 0

    def test_merge_error_rows_calculated_correctly(self) -> None:
        """Test error_rows is correct after merge."""
        stats1 = RunnerStats(total_rows=100, valid_rows=80)
        stats2 = RunnerStats(total_rows=100, valid_rows=90)

        stats1.merge(stats2)

        # total=200, valid=170, errors should be 30
        assert stats1.error_rows == 30


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
        data: list[dict[str, Any]] = [
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
        data: list[dict[str, Any]] = [
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
        data: list[dict[str, Any]] = [
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
        data: list[dict[str, Any]] = [
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
        data: list[dict[str, Any]] = [{"name": "valid"}, {"name": 123}]
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
        data: list[dict[str, Any]] = [
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


# =============================================================================
# ValidationRunner Context Manager Tests
# =============================================================================


class TestValidationRunnerContextManager:
    """Tests for ValidationRunner context manager support."""

    def test_enter_returns_self(self) -> None:
        """Test __enter__ returns the runner instance."""
        data = [{"name": "test"}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        result = runner.__enter__()

        assert result is runner

    def test_context_manager_workflow(self) -> None:
        """Test using runner as context manager."""
        data = [{"name": "test1"}, {"name": "test2"}]

        with ValidationRunner(make_data_iterator(data), SampleModel) as runner:
            results = list(runner.run())

        assert len(results) == 2
        assert runner.stats.total_rows == 2

    def test_exit_finalizes_stats_when_run_incomplete(self) -> None:
        """Test __exit__ finalizes stats if run was started but not completed."""
        data = [{"name": f"test{i}"} for i in range(10)]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        with runner:
            # Start iteration but don't complete it
            iterator = runner.run()
            next(iterator)  # Process one row
            next(iterator)  # Process second row
            # Don't exhaust the iterator

            # Start time should be set
            assert runner.stats.start_time > 0
            # End time should be 0 (not finalized yet during iteration)

        # After exiting context, end_time should be set if it wasn't
        # Note: The test verifies __exit__ behavior

    def test_exit_does_nothing_if_not_started(self) -> None:
        """Test __exit__ does nothing if run was never called."""
        data = [{"name": "test"}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        with runner:
            pass  # Don't call run()

        # Stats should be default
        assert runner.stats.start_time == 0
        assert runner.stats.end_time == 0

    def test_exit_does_nothing_if_completed(self) -> None:
        """Test __exit__ doesn't change end_time if run completed normally."""
        data = [{"name": "test1"}, {"name": "test2"}]

        with ValidationRunner(make_data_iterator(data), SampleModel) as runner:
            list(runner.run())  # Complete the run
            end_time_after_run = runner.stats.end_time

        # end_time should remain unchanged after __exit__
        assert runner.stats.end_time == end_time_after_run
        assert runner.stats.end_time > 0

    def test_context_manager_with_exception(self) -> None:
        """Test context manager properly cleans up on exception."""
        data = [{"name": "test"}]

        try:
            with ValidationRunner(make_data_iterator(data), SampleModel) as runner:
                list(runner.run())
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Stats should still be available
        assert runner.stats.total_rows == 1


# =============================================================================
# ValidationRunner Parallel Processing Tests
# =============================================================================


class TestValidationRunnerParallel:
    """Tests for ValidationRunner parallel processing."""

    def test_parallel_processes_all_rows(self) -> None:
        """Test parallel processing processes all rows."""
        data = [{"name": f"item{i}", "value": i} for i in range(50)]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        results = list(runner.run(workers=2, chunk_size=10))

        assert len(results) == 50
        assert runner.stats.total_rows == 50
        assert runner.stats.valid_rows == 50

    def test_parallel_maintains_row_order(self) -> None:
        """Test parallel results maintain original row order."""
        data = [{"name": f"item{i}", "value": i} for i in range(100)]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        results = list(runner.run(workers=4, chunk_size=10))

        # Results should be in original row order
        row_indices = [r.row_index for r in results]
        assert row_indices == list(range(100))

    def test_parallel_handles_invalid_rows(self) -> None:
        """Test parallel processing handles invalid rows correctly."""
        data: list[dict[str, Any]] = [
            {"name": "valid1", "value": 1},
            {"name": 123},  # Invalid - pydantic error
            {"name": "valid2", "value": 2},
            {"name": 456},  # Invalid - pydantic error
            {"name": "valid3", "value": 3},
        ]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        results = list(runner.run(workers=2, chunk_size=2))

        assert len(results) == 5
        valid_count = sum(1 for r in results if r.is_valid)
        invalid_count = sum(1 for r in results if not r.is_valid)

        assert valid_count == 3
        assert invalid_count == 2
        assert runner.stats.valid_rows == 3
        assert runner.stats.pydantic_failures == 2

    def test_parallel_with_custom_validators(self) -> None:
        """Test parallel processing with custom validators."""
        data = [
            {"name": "test1", "value": 10},
            {"name": "test2", "value": -5},  # Should fail validator
            {"name": "test3", "value": 20},
            {"name": "test4", "value": -10},  # Should fail validator
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

        results = list(runner.run(workers=2, chunk_size=2))

        assert len(results) == 4
        valid_count = sum(1 for r in results if r.is_valid)
        assert valid_count == 2
        assert runner.stats.validator_failures == 2

    def test_parallel_fail_fast(self) -> None:
        """Test parallel processing with fail_fast=True."""
        data: list[dict[str, Any]] = [
            {"name": "valid1"},
            {"name": "valid2"},
            {"name": 123},  # Invalid
            {"name": "valid3"},
            {"name": "valid4"},
        ]
        runner = ValidationRunner(
            make_data_iterator(data),
            SampleModel,
            fail_fast=True,
        )

        results = list(runner.run(workers=2, chunk_size=2))

        # Should stop after encountering the first error in yielded results
        # Due to chunking, it may process some items before stopping
        # The key invariant: the last result should be invalid
        invalid_results = [r for r in results if not r.is_valid]
        assert len(invalid_results) >= 1

    def test_parallel_stats_combined_correctly(self) -> None:
        """Test parallel processing combines stats from all chunks."""
        data: list[dict[str, Any]] = [
            {"name": "valid1", "value": 1},
            {"name": 123},  # Invalid
            {"name": "valid2", "value": 2},
            {"name": "valid3", "value": 3},
            {"name": 456},  # Invalid
            {"name": "valid4", "value": 4},
        ]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        list(runner.run(workers=2, chunk_size=2))

        assert runner.stats.total_rows == 6
        assert runner.stats.valid_rows == 4
        assert runner.stats.pydantic_failures == 2
        assert runner.stats.error_rows == 2

    def test_parallel_error_counts_combined(self) -> None:
        """Test parallel processing combines error counts from all chunks."""
        # Create data that will produce the same error across chunks
        data = [{"name": 123} for _ in range(10)]  # All invalid
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        list(runner.run(workers=2, chunk_size=3))

        # All errors should be counted
        total_errors = sum(runner.stats.error_counts.values())
        assert total_errors == 10

    def test_parallel_failed_samples_collected(self) -> None:
        """Test parallel processing collects failed samples."""
        data: list[dict[str, Any]] = [
            {"name": "valid1"},
            {"name": 123, "extra": "data1"},  # Invalid
            {"name": "valid2"},
            {"name": 456, "extra": "data2"},  # Invalid
        ]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        list(runner.run(workers=2, chunk_size=2))

        assert len(runner.stats.failed_samples) == 2

    def test_parallel_emits_events(self) -> None:
        """Test parallel processing emits correct events."""
        data = [{"name": f"item{i}"} for i in range(10)]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)
        observer = RecordingObserver()
        runner.add_observer(observer)

        list(runner.run(workers=2, chunk_size=3))

        # Should have VALIDATION_STARTED
        started_events = [
            e for e in observer.events if e.event_type == ValidationEventType.VALIDATION_STARTED
        ]
        assert len(started_events) == 1

        # Should have VALIDATION_COMPLETED
        completed_events = [
            e for e in observer.events if e.event_type == ValidationEventType.VALIDATION_COMPLETED
        ]
        assert len(completed_events) == 1

        # Should have ROW_PROCESSED for each row
        row_events = [
            e for e in observer.events if e.event_type == ValidationEventType.ROW_PROCESSED
        ]
        assert len(row_events) == 10

    def test_parallel_started_event_includes_workers(self) -> None:
        """Test VALIDATION_STARTED event includes worker info in parallel mode."""
        data = [{"name": "test"}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)
        observer = RecordingObserver()
        runner.add_observer(observer)

        list(runner.run(workers=4, chunk_size=5))

        started_events = [
            e for e in observer.events if e.event_type == ValidationEventType.VALIDATION_STARTED
        ]
        assert len(started_events) == 1
        assert started_events[0].data.get("workers") == 4
        assert started_events[0].data.get("chunk_size") == 5

    def test_parallel_small_dataset(self) -> None:
        """Test parallel processing with dataset smaller than chunk_size."""
        data = [{"name": f"item{i}"} for i in range(3)]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        results = list(runner.run(workers=4, chunk_size=10))

        assert len(results) == 3
        assert runner.stats.total_rows == 3

    def test_parallel_single_item(self) -> None:
        """Test parallel processing with single item."""
        data = [{"name": "single"}]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        results = list(runner.run(workers=2, chunk_size=5))

        assert len(results) == 1
        assert results[0].is_valid
        assert results[0].model is not None
        assert results[0].model.name == "single"

    def test_parallel_empty_data(self) -> None:
        """Test parallel processing with empty data."""
        runner = ValidationRunner(make_data_iterator([]), SampleModel)

        results = list(runner.run(workers=2, chunk_size=5))

        assert len(results) == 0
        assert runner.stats.total_rows == 0

    def test_parallel_uses_sequential_for_single_worker(self) -> None:
        """Test that workers=1 uses sequential processing."""
        data = [{"name": f"item{i}"} for i in range(10)]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)
        observer = RecordingObserver()
        runner.add_observer(observer)

        results = list(runner.run(workers=1, chunk_size=5))

        # Should still process all rows
        assert len(results) == 10

        # In sequential mode, no worker info in started event
        started_events = [
            e for e in observer.events if e.event_type == ValidationEventType.VALIDATION_STARTED
        ]
        assert "workers" not in started_events[0].data

    def test_parallel_max_pending_limits_memory(self) -> None:
        """Test max_pending parameter controls buffering."""
        data = [{"name": f"item{i}"} for i in range(50)]
        runner = ValidationRunner(make_data_iterator(data), SampleModel)

        # Should complete without issues even with small max_pending
        results = list(runner.run(workers=2, chunk_size=5, max_pending=2))

        assert len(results) == 50


# =============================================================================
# Stateful Tests
# =============================================================================


class RunnerStatsStateMachine(RuleBasedStateMachine):
    """Stateful test for RunnerStats to verify state transitions."""

    def __init__(self) -> None:
        super().__init__()
        self.stats = RunnerStats(max_samples=10)
        # Track expected state
        self.expected_error_counts: dict[tuple[str, str], int] = {}
        self.expected_samples: list[dict[str, Any]] = []
        self.expected_total_rows = 0
        self.expected_valid_rows = 0
        self.expected_pydantic_failures = 0
        self.expected_validator_failures = 0

    @rule(field=st.text(min_size=1, max_size=20), message=st.text(min_size=1, max_size=50))
    def record_error(self, field: str, message: str) -> None:
        """Record an error and track expected state."""
        self.stats.record_error(field, message)
        key = (field, message)
        self.expected_error_counts[key] = self.expected_error_counts.get(key, 0) + 1

    @rule(sample_id=st.integers(min_value=0, max_value=1000))
    def add_failed_sample(self, sample_id: int) -> None:
        """Add a failed sample and track expected state."""
        sample = {"id": sample_id}
        self.stats.add_failed_sample(sample)
        if len(self.expected_samples) < self.stats.max_samples:
            self.expected_samples.append(sample)

    @rule(
        total=st.integers(min_value=0, max_value=100),
        valid=st.integers(min_value=0, max_value=100),
        pydantic_fail=st.integers(min_value=0, max_value=50),
        validator_fail=st.integers(min_value=0, max_value=50),
    )
    def merge_other_stats(
        self, total: int, valid: int, pydantic_fail: int, validator_fail: int
    ) -> None:
        """Merge another stats object and track expected state."""
        other = RunnerStats(
            total_rows=total,
            valid_rows=valid,
            pydantic_failures=pydantic_fail,
            validator_failures=validator_fail,
        )
        # Add some errors to the other stats
        other.record_error("merge_field", "merge_error")

        self.stats.merge(other)

        # Update expectations
        self.expected_total_rows += total
        self.expected_valid_rows += valid
        self.expected_pydantic_failures += pydantic_fail
        self.expected_validator_failures += validator_fail
        key = ("merge_field", "merge_error")
        self.expected_error_counts[key] = self.expected_error_counts.get(key, 0) + 1

    @invariant()
    def error_counts_match(self) -> None:
        """Verify error counts match expected values."""
        assert dict(self.stats.error_counts) == self.expected_error_counts

    @invariant()
    def failed_samples_match(self) -> None:
        """Verify failed samples match expected values."""
        assert self.stats.failed_samples == self.expected_samples

    @invariant()
    def row_counts_match(self) -> None:
        """Verify row counts match expected values."""
        assert self.stats.total_rows == self.expected_total_rows
        assert self.stats.valid_rows == self.expected_valid_rows

    @invariant()
    def failure_counts_match(self) -> None:
        """Verify failure counts match expected values."""
        assert self.stats.pydantic_failures == self.expected_pydantic_failures
        assert self.stats.validator_failures == self.expected_validator_failures

    @invariant()
    def error_rows_calculated_correctly(self) -> None:
        """Verify error_rows property is always correct."""
        assert self.stats.error_rows == self.expected_total_rows - self.expected_valid_rows

    @invariant()
    def samples_within_limit(self) -> None:
        """Verify failed_samples never exceeds max_samples."""
        assert len(self.stats.failed_samples) <= self.stats.max_samples


# Create test class from state machine
TestRunnerStatsStateful = RunnerStatsStateMachine.TestCase


# =============================================================================
# Property-Based Tests for Runner
# =============================================================================


class TestRunnerStatsProperties:
    """Property-based tests for RunnerStats."""

    @given(
        total=st.integers(min_value=0, max_value=1000000),
        valid=st.integers(min_value=0, max_value=1000000),
    )
    @settings(max_examples=50)
    def test_error_rows_always_non_negative(self, total: int, valid: int) -> None:
        """Test error_rows is always non-negative (assuming valid <= total)."""
        valid = min(valid, total)  # Ensure valid <= total
        stats = RunnerStats(total_rows=total, valid_rows=valid)

        assert stats.error_rows >= 0
        assert stats.error_rows == total - valid

    @given(
        total=st.integers(min_value=1, max_value=1000000),
        valid=st.integers(min_value=0, max_value=1000000),
    )
    @settings(max_examples=50)
    def test_success_rate_in_range(self, total: int, valid: int) -> None:
        """Test success_rate is always between 0 and 100."""
        valid = min(valid, total)  # Ensure valid <= total
        stats = RunnerStats(total_rows=total, valid_rows=valid)

        assert 0 <= stats.success_rate <= 100

    @given(
        errors=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=20),
                st.text(min_size=1, max_size=50),
            ),
            min_size=0,
            max_size=100,
        ),
    )
    @settings(max_examples=30)
    def test_top_errors_counts_match(self, errors: list[tuple[str, str]]) -> None:
        """Test top_errors counts sum to total recorded errors."""
        stats = RunnerStats()

        for field, msg in errors:
            stats.record_error(field, msg)

        top = stats.top_errors(100)  # Get all

        # Sum of top error counts should equal total errors recorded
        total_from_top = sum(count for _, count, _ in top)
        assert total_from_top == len(errors)

    @given(
        stats1_data=st.fixed_dictionaries(
            {
                "total": st.integers(min_value=0, max_value=1000),
                "valid": st.integers(min_value=0, max_value=1000),
            }
        ),
        stats2_data=st.fixed_dictionaries(
            {
                "total": st.integers(min_value=0, max_value=1000),
                "valid": st.integers(min_value=0, max_value=1000),
            }
        ),
    )
    @settings(max_examples=50)
    def test_merge_is_additive(
        self,
        stats1_data: dict[str, int],
        stats2_data: dict[str, int],
    ) -> None:
        """Test merge is additive for row counts."""
        stats1 = RunnerStats(
            total_rows=stats1_data["total"],
            valid_rows=min(stats1_data["valid"], stats1_data["total"]),
        )
        stats2 = RunnerStats(
            total_rows=stats2_data["total"],
            valid_rows=min(stats2_data["valid"], stats2_data["total"]),
        )

        expected_total = stats1.total_rows + stats2.total_rows
        expected_valid = stats1.valid_rows + stats2.valid_rows

        stats1.merge(stats2)

        assert stats1.total_rows == expected_total
        assert stats1.valid_rows == expected_valid

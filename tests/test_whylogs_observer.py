"""Tests for WhylogsObserver, ProfilePair, and ProfileComparison."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from abstract_validation_base.events import ValidationEvent, ValidationEventType
from abstract_validation_base.results import ValidationResult
from abstract_validation_base.validators import BaseValidator

from .conftest import SampleModel

# Skip entire module if whylogs is not installed
whylogs = pytest.importorskip("whylogs")

from abstract_validation_base.whylogs_observer import (  # noqa: E402
    ProfileComparison,
    ProfilePair,
    WhylogsObserver,
)

# =============================================================================
# Helper Functions
# =============================================================================


def make_event(
    event_type: ValidationEventType,
    data: dict[str, Any] | None = None,
) -> ValidationEvent:
    """Create a ValidationEvent for testing."""
    return ValidationEvent(
        event_type=event_type,
        source=None,
        data=data or {},
    )


def make_row_event(
    raw_data: dict[str, Any] | None = None,
    model_dict: dict[str, Any] | None = None,
    is_valid: bool = True,
) -> ValidationEvent:
    """Create a ROW_PROCESSED event with the expected data structure."""
    data: dict[str, Any] = {
        "row_index": 0,
        "is_valid": is_valid,
        "stats_snapshot": {
            "total": 1,
            "valid": 1 if is_valid else 0,
            "failed": 0 if is_valid else 1,
        },
        "errors": [] if is_valid else [("field", "error")],
        "raw_data": raw_data or {"field": "value"},
    }
    if is_valid and model_dict is not None:
        data["model_dict"] = model_dict
    return make_event(ValidationEventType.ROW_PROCESSED, data)


# =============================================================================
# WhylogsObserver Init Tests
# =============================================================================


class TestWhylogsObserverInit:
    """Tests for WhylogsObserver initialization."""

    def test_init_defaults(self) -> None:
        """Test that __init__ sets correct default values."""
        observer = WhylogsObserver()

        assert observer.chunk_size == 10000
        assert observer.profile_raw is True
        assert observer.profile_valid is True
        assert observer.schema is None
        assert observer._is_running is False
        assert observer._raw_buffer == []
        assert observer._valid_buffer == []
        assert observer._raw_profile is None
        assert observer._valid_profile is None
        assert observer._raw_row_count == 0
        assert observer._valid_row_count == 0

    def test_init_custom_chunk_size(self) -> None:
        """Test that custom chunk_size is stored."""
        observer = WhylogsObserver(chunk_size=500)

        assert observer.chunk_size == 500

    def test_init_disable_raw_profiling(self) -> None:
        """Test that raw profiling can be disabled."""
        observer = WhylogsObserver(profile_raw=False)

        assert observer.profile_raw is False
        assert observer.profile_valid is True

    def test_init_disable_valid_profiling(self) -> None:
        """Test that valid profiling can be disabled."""
        observer = WhylogsObserver(profile_valid=False)

        assert observer.profile_raw is True
        assert observer.profile_valid is False


# =============================================================================
# WhylogsObserver Event Handling Tests
# =============================================================================


class TestWhylogsObserverValidationStarted:
    """Tests for WhylogsObserver handling VALIDATION_STARTED events."""

    def test_handles_validation_started(self) -> None:
        """Test VALIDATION_STARTED sets is_running to True."""
        observer = WhylogsObserver()

        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        assert observer._is_running is True

    def test_validation_started_resets_state(self) -> None:
        """Test VALIDATION_STARTED clears previous state."""
        observer = WhylogsObserver()
        # Simulate some previous state
        observer._raw_buffer = [{"a": 1}]
        observer._valid_buffer = [{"b": 2}]
        observer._raw_row_count = 100
        observer._valid_row_count = 50

        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        assert observer._raw_buffer == []
        assert observer._valid_buffer == []
        assert observer._raw_row_count == 0
        assert observer._valid_row_count == 0


class TestWhylogsObserverRowProcessed:
    """Tests for WhylogsObserver handling ROW_PROCESSED events."""

    def test_buffers_raw_data(self) -> None:
        """Test ROW_PROCESSED buffers raw data."""
        observer = WhylogsObserver()
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        raw_data = {"name": "test", "value": 42}
        observer.on_event(make_row_event(raw_data=raw_data, is_valid=False))

        assert len(observer._raw_buffer) == 1
        assert observer._raw_buffer[0] == raw_data

    def test_buffers_valid_model_dict(self) -> None:
        """Test ROW_PROCESSED buffers model_dict for valid rows."""
        observer = WhylogsObserver()
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        model_dict = {"name": "test", "value": 42}
        observer.on_event(make_row_event(raw_data={"x": 1}, model_dict=model_dict, is_valid=True))

        assert len(observer._valid_buffer) == 1
        assert observer._valid_buffer[0] == model_dict

    def test_does_not_buffer_model_dict_for_invalid(self) -> None:
        """Test ROW_PROCESSED does not buffer model_dict for invalid rows."""
        observer = WhylogsObserver()
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        observer.on_event(make_row_event(raw_data={"x": 1}, is_valid=False))

        assert len(observer._valid_buffer) == 0

    def test_skips_raw_buffer_when_disabled(self) -> None:
        """Test ROW_PROCESSED skips raw buffering when profile_raw=False."""
        observer = WhylogsObserver(profile_raw=False)
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        observer.on_event(make_row_event(raw_data={"x": 1}))

        assert len(observer._raw_buffer) == 0

    def test_skips_valid_buffer_when_disabled(self) -> None:
        """Test ROW_PROCESSED skips valid buffering when profile_valid=False."""
        observer = WhylogsObserver(profile_valid=False)
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        observer.on_event(make_row_event(raw_data={"x": 1}, model_dict={"y": 2}, is_valid=True))

        assert len(observer._valid_buffer) == 0

    def test_flushes_at_chunk_size(self) -> None:
        """Test buffer is flushed when chunk_size is reached."""
        observer = WhylogsObserver(chunk_size=3)
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        # Add rows up to chunk_size
        for i in range(3):
            observer.on_event(make_row_event(raw_data={"index": i}))

        # Buffer should have been flushed
        assert len(observer._raw_buffer) == 0
        assert observer._raw_profile is not None
        assert observer._raw_row_count == 3


class TestWhylogsObserverValidationCompleted:
    """Tests for WhylogsObserver handling VALIDATION_COMPLETED events."""

    def test_final_flush_on_complete(self) -> None:
        """Test VALIDATION_COMPLETED flushes remaining buffers."""
        observer = WhylogsObserver(chunk_size=100)  # Large chunk to prevent auto-flush
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        # Add some data (less than chunk_size)
        observer.on_event(make_row_event(raw_data={"a": 1}, model_dict={"b": 2}, is_valid=True))
        observer.on_event(make_row_event(raw_data={"c": 3}, model_dict={"d": 4}, is_valid=True))

        # Buffers should have data
        assert len(observer._raw_buffer) == 2
        assert len(observer._valid_buffer) == 2

        # Complete
        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED))

        # Buffers should be flushed
        assert len(observer._raw_buffer) == 0
        assert len(observer._valid_buffer) == 0
        assert observer._is_running is False

    def test_sets_is_running_false(self) -> None:
        """Test VALIDATION_COMPLETED sets is_running to False."""
        observer = WhylogsObserver()
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED))

        assert observer._is_running is False


# =============================================================================
# WhylogsObserver get_profiles Tests
# =============================================================================


class TestWhylogsObserverGetProfiles:
    """Tests for WhylogsObserver.get_profiles method."""

    def test_get_profiles_raises_while_running(self) -> None:
        """Test get_profiles raises RuntimeError while validation is running."""
        observer = WhylogsObserver()
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        with pytest.raises(RuntimeError, match="Cannot get profiles while validation is running"):
            observer.get_profiles()

    def test_get_profiles_returns_profile_pair(self) -> None:
        """Test get_profiles returns a ProfilePair after validation completes."""
        observer = WhylogsObserver(chunk_size=100)
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))
        observer.on_event(make_row_event(raw_data={"a": 1}, model_dict={"b": 2}, is_valid=True))
        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED))

        result = observer.get_profiles()

        assert isinstance(result, ProfilePair)
        assert result.raw_profile is not None
        assert result.valid_profile is not None

    def test_get_profiles_empty_when_no_data(self) -> None:
        """Test get_profiles returns empty ProfilePair when no data was processed."""
        observer = WhylogsObserver()
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))
        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED))

        result = observer.get_profiles()

        assert isinstance(result, ProfilePair)
        assert result.raw_profile is None
        assert result.valid_profile is None

    def test_profile_disabled_returns_none(self) -> None:
        """Test get_profiles returns None for disabled profile types."""
        observer = WhylogsObserver(profile_raw=False, profile_valid=True)
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))
        observer.on_event(make_row_event(raw_data={"a": 1}, model_dict={"b": 2}, is_valid=True))
        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED))

        result = observer.get_profiles()

        assert result.raw_profile is None
        assert result.valid_profile is not None


# =============================================================================
# WhylogsObserver compare_profiles Tests
# =============================================================================


class TestWhylogsObserverCompareProfiles:
    """Tests for WhylogsObserver.compare_profiles method."""

    def test_compare_profiles_raises_while_running(self) -> None:
        """Test compare_profiles raises RuntimeError while validation is running."""
        observer = WhylogsObserver()
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        with pytest.raises(
            RuntimeError, match="Cannot compare profiles while validation is running"
        ):
            observer.compare_profiles()

    def test_compare_profiles_raises_when_both_disabled(self) -> None:
        """Test compare_profiles raises ValueError when both profiles disabled."""
        observer = WhylogsObserver(profile_raw=False, profile_valid=False)

        with pytest.raises(ValueError, match="Cannot compare profiles when both"):
            observer.compare_profiles()

    def test_compare_profiles_returns_comparison(self) -> None:
        """Test compare_profiles returns a ProfileComparison."""
        observer = WhylogsObserver(chunk_size=100)
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))
        observer.on_event(
            make_row_event(raw_data={"a": 1, "extra": "x"}, model_dict={"a": 1}, is_valid=True)
        )
        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED))

        result = observer.compare_profiles()

        assert isinstance(result, ProfileComparison)
        assert result.raw_row_count == 1
        assert result.valid_row_count == 1
        assert result.pass_rate == 1.0

    def test_compare_profiles_pass_rate_calculation(self) -> None:
        """Test pass_rate is calculated correctly."""
        observer = WhylogsObserver(chunk_size=100)
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        # Add 4 rows, 2 valid, 2 invalid
        observer.on_event(make_row_event(raw_data={"a": 1}, model_dict={"a": 1}, is_valid=True))
        observer.on_event(make_row_event(raw_data={"a": 2}, is_valid=False))
        observer.on_event(make_row_event(raw_data={"a": 3}, model_dict={"a": 3}, is_valid=True))
        observer.on_event(make_row_event(raw_data={"a": 4}, is_valid=False))

        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED))

        result = observer.compare_profiles()

        assert result.raw_row_count == 4
        assert result.valid_row_count == 2
        assert result.pass_rate == 0.5


# =============================================================================
# WhylogsObserver reset Tests
# =============================================================================


class TestWhylogsObserverReset:
    """Tests for WhylogsObserver.reset method."""

    def test_reset(self) -> None:
        """Test reset clears all state."""
        observer = WhylogsObserver(chunk_size=100)
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))
        observer.on_event(make_row_event(raw_data={"a": 1}, model_dict={"b": 2}, is_valid=True))
        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED))

        # Verify we have state
        assert observer._raw_profile is not None

        observer.reset()

        assert observer._raw_buffer == []
        assert observer._valid_buffer == []
        assert observer._raw_profile is None
        assert observer._valid_profile is None
        assert observer._is_running is False
        assert observer._raw_row_count == 0
        assert observer._valid_row_count == 0


# =============================================================================
# ProfilePair Tests
# =============================================================================


class TestProfilePairToPandas:
    """Tests for ProfilePair.to_pandas method."""

    def test_to_pandas_empty(self) -> None:
        """Test to_pandas with empty profiles returns None values."""
        pair = ProfilePair()

        result = pair.to_pandas()

        assert result["raw"] is None
        assert result["valid"] is None

    def test_to_pandas_with_profiles(self) -> None:
        """Test to_pandas with profiles returns DataFrames."""
        import pandas as pd
        import whylogs as why

        # Create actual profiles
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        profile = why.log(df).profile().view()

        pair = ProfilePair(raw_profile=profile, valid_profile=profile)

        result = pair.to_pandas()

        assert result["raw"] is not None
        assert result["valid"] is not None
        assert isinstance(result["raw"], pd.DataFrame)
        assert isinstance(result["valid"], pd.DataFrame)


class TestProfilePairWrite:
    """Tests for ProfilePair.write method."""

    def test_write_empty_profiles(self) -> None:
        """Test write with empty profiles returns None paths."""
        pair = ProfilePair()

        result = pair.write()

        assert result["raw"] is None
        assert result["valid"] is None

    def test_write_with_profiles(self) -> None:
        """Test write creates files on disk."""
        import pandas as pd
        import whylogs as why

        # Create actual profiles
        df = pd.DataFrame({"a": [1, 2, 3]})
        profile = why.log(df).profile().view()

        pair = ProfilePair(raw_profile=profile, valid_profile=profile)

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "raw.bin"
            valid_path = Path(tmpdir) / "valid.bin"

            result = pair.write(raw_path=raw_path, valid_path=valid_path)

            assert result["raw"] == raw_path
            assert result["valid"] == valid_path
            assert raw_path.exists()
            assert valid_path.exists()


# =============================================================================
# ProfileComparison Tests
# =============================================================================


class TestProfileComparison:
    """Tests for ProfileComparison dataclass."""

    def test_to_dict(self) -> None:
        """Test to_dict returns all fields."""
        comparison = ProfileComparison(
            raw_column_count=5,
            valid_column_count=4,
            raw_row_count=100,
            valid_row_count=80,
            columns_only_in_raw=["extra"],
            columns_only_in_valid=["computed"],
            common_columns=["a", "b", "c"],
            pass_rate=0.8,
        )

        result = comparison.to_dict()

        assert result["raw_column_count"] == 5
        assert result["valid_column_count"] == 4
        assert result["raw_row_count"] == 100
        assert result["valid_row_count"] == 80
        assert result["columns_only_in_raw"] == ["extra"]
        assert result["columns_only_in_valid"] == ["computed"]
        assert result["common_columns"] == ["a", "b", "c"]
        assert result["pass_rate"] == 0.8


# =============================================================================
# Integration Tests with ValidationRunner
# =============================================================================


class TestWhylogsObserverRunnerIntegration:
    """Integration tests for WhylogsObserver with ValidationRunner."""

    def test_with_runner_sequential(self) -> None:
        """Test WhylogsObserver with ValidationRunner in sequential mode."""
        from abstract_validation_base.runner import ValidationRunner

        data = [
            {"name": "Alice", "value": 10},
            {"name": "Bob", "value": 20},
            {"name": "Charlie", "value": 30},
        ]

        observer = WhylogsObserver(chunk_size=100)
        runner: ValidationRunner[SampleModel] = ValidationRunner(
            data=iter(data),
            model_class=SampleModel,
        )
        runner.add_observer(observer)

        # Process all rows
        results = list(runner.run())

        # All should be valid
        assert all(r.is_valid for r in results)
        assert len(results) == 3

        # Check profiles
        profiles = observer.get_profiles()
        assert profiles.raw_profile is not None
        assert profiles.valid_profile is not None

        # Check comparison
        comparison = observer.compare_profiles()
        assert comparison.raw_row_count == 3
        assert comparison.valid_row_count == 3
        assert comparison.pass_rate == 1.0

    def test_with_runner_partial_failures(self) -> None:
        """Test WhylogsObserver with some validation failures."""
        from abstract_validation_base.runner import ValidationRunner

        class NegativeValueValidator(BaseValidator[SampleModel]):
            @property
            def name(self) -> str:
                return "negative_value"

            def validate(self, item: SampleModel) -> ValidationResult:
                result = ValidationResult(is_valid=True)
                if item.value < 0:
                    result.add_error("value", "Value must be non-negative")
                return result

        data = [
            {"name": "Valid", "value": 10},
            {"name": "Invalid", "value": -5},
            {"name": "Valid2", "value": 20},
        ]

        observer = WhylogsObserver(chunk_size=100)
        runner: ValidationRunner[SampleModel] = ValidationRunner(
            data=iter(data),
            model_class=SampleModel,
            validators=NegativeValueValidator(),
        )
        runner.add_observer(observer)

        # Process all rows
        results = list(runner.run())

        # 2 valid, 1 invalid
        valid_count = sum(1 for r in results if r.is_valid)
        assert valid_count == 2

        # Check comparison
        comparison = observer.compare_profiles()
        assert comparison.raw_row_count == 3
        assert comparison.valid_row_count == 2
        assert abs(comparison.pass_rate - (2 / 3)) < 0.01

    def test_with_runner_parallel(self) -> None:
        """Test WhylogsObserver with ValidationRunner in parallel mode."""
        from abstract_validation_base.runner import ValidationRunner

        # Generate larger dataset for parallel processing
        data = [{"name": f"Item{i}", "value": i} for i in range(50)]

        observer = WhylogsObserver(chunk_size=20)
        runner: ValidationRunner[SampleModel] = ValidationRunner(
            data=iter(data),
            model_class=SampleModel,
            total_hint=50,
        )
        runner.add_observer(observer)

        # Process with workers
        results = list(runner.run(workers=2, chunk_size=10))

        # All should be valid
        assert len(results) == 50
        assert all(r.is_valid for r in results)

        # Check profiles
        profiles = observer.get_profiles()
        assert profiles.raw_profile is not None
        assert profiles.valid_profile is not None

        # Check row counts (profiles should have captured all rows)
        comparison = observer.compare_profiles()
        assert comparison.raw_row_count == 50
        assert comparison.valid_row_count == 50


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestWhylogsObserverProperties:
    """Property-based tests for WhylogsObserver."""

    @given(
        chunk_size=st.integers(min_value=1, max_value=100),
        num_rows=st.integers(min_value=1, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_row_counts_accurate(self, chunk_size: int, num_rows: int) -> None:
        """Test that row counts accurately reflect processed data."""
        observer = WhylogsObserver(chunk_size=chunk_size)
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        # Process rows
        for i in range(num_rows):
            is_valid = i % 2 == 0  # Every other row is valid
            observer.on_event(
                make_row_event(
                    raw_data={"index": i},
                    model_dict={"index": i} if is_valid else None,
                    is_valid=is_valid,
                )
            )

        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED))

        comparison = observer.compare_profiles()

        # Raw should have all rows
        assert comparison.raw_row_count == num_rows

        # Valid should have half (rounded up for odd counts)
        expected_valid = (num_rows + 1) // 2
        assert comparison.valid_row_count == expected_valid

    @given(
        profile_raw=st.booleans(),
        profile_valid=st.booleans(),
    )
    @settings(max_examples=10, deadline=None)
    def test_profile_flags_respected(self, profile_raw: bool, profile_valid: bool) -> None:
        """Test that profile_raw and profile_valid flags are respected."""
        observer = WhylogsObserver(
            chunk_size=10, profile_raw=profile_raw, profile_valid=profile_valid
        )
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))

        # Process some rows
        for i in range(5):
            observer.on_event(make_row_event(raw_data={"i": i}, model_dict={"i": i}, is_valid=True))

        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED))

        # Can only get profiles if at least one is enabled
        if not profile_raw and not profile_valid:
            with pytest.raises(ValueError):
                observer.compare_profiles()
        else:
            comparison = observer.compare_profiles()
            if profile_raw:
                assert comparison.raw_row_count == 5
            else:
                assert comparison.raw_row_count == 0
            if profile_valid:
                assert comparison.valid_row_count == 5
            else:
                assert comparison.valid_row_count == 0


# =============================================================================
# Stateful Testing
# =============================================================================


class WhylogsObserverStateMachine(RuleBasedStateMachine):
    """Stateful test for WhylogsObserver lifecycle."""

    def __init__(self) -> None:
        super().__init__()
        self.observer = WhylogsObserver(chunk_size=5)
        self.is_running = False
        self.raw_rows_added = 0
        self.valid_rows_added = 0

    @rule()
    def start_validation(self) -> None:
        """Start a new validation run."""
        self.observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED))
        self.is_running = True
        self.raw_rows_added = 0
        self.valid_rows_added = 0

    @rule(is_valid=st.booleans())
    def process_row(self, is_valid: bool) -> None:
        """Process a row (only when running)."""
        if not self.is_running:
            return

        self.observer.on_event(
            make_row_event(
                raw_data={"x": self.raw_rows_added},
                model_dict={"x": self.raw_rows_added} if is_valid else None,
                is_valid=is_valid,
            )
        )
        self.raw_rows_added += 1
        if is_valid:
            self.valid_rows_added += 1

    @rule()
    def complete_validation(self) -> None:
        """Complete the validation run."""
        if not self.is_running:
            return

        self.observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED))
        self.is_running = False

    @rule()
    def reset_observer(self) -> None:
        """Reset the observer."""
        self.observer.reset()
        self.is_running = False
        self.raw_rows_added = 0
        self.valid_rows_added = 0

    @invariant()
    def is_running_matches(self) -> None:
        """Observer's is_running should match our tracking."""
        assert self.observer._is_running == self.is_running

    @invariant()
    def cannot_get_profiles_while_running(self) -> None:
        """Should not be able to get profiles while running."""
        if self.is_running:
            with pytest.raises(RuntimeError):
                self.observer.get_profiles()


TestWhylogsObserverStateful = WhylogsObserverStateMachine.TestCase

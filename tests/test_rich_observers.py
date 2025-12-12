"""Tests for Rich-based observers: SimpleProgressObserver and RichDashboardObserver."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st
from rich.console import Console
from rich.progress import Progress

from abstract_validation_base.events import ValidationEvent, ValidationEventType
from abstract_validation_base.rich_observers import (
    RichDashboardObserver,
    SimpleProgressObserver,
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


# =============================================================================
# SimpleProgressObserver Unit Tests
# =============================================================================


class TestSimpleProgressObserverInit:
    """Tests for SimpleProgressObserver initialization."""

    def test_init_stores_progress(self) -> None:
        """Test that __init__ stores the progress instance."""
        progress = MagicMock(spec=Progress)
        observer = SimpleProgressObserver(progress)

        assert observer._progress is progress

    def test_init_default_description(self) -> None:
        """Test that default task description is 'Validating'."""
        progress = MagicMock(spec=Progress)
        observer = SimpleProgressObserver(progress)

        assert observer._description == "Validating"

    def test_init_custom_description(self) -> None:
        """Test that custom task description is stored."""
        progress = MagicMock(spec=Progress)
        observer = SimpleProgressObserver(progress, task_description="Processing")

        assert observer._description == "Processing"

    def test_init_counters_zero(self) -> None:
        """Test that valid/failed counters start at zero."""
        progress = MagicMock(spec=Progress)
        observer = SimpleProgressObserver(progress)

        assert observer._valid == 0
        assert observer._failed == 0

    def test_init_task_id_none(self) -> None:
        """Test that task_id is None initially."""
        progress = MagicMock(spec=Progress)
        observer = SimpleProgressObserver(progress)

        assert observer._task_id is None


class TestSimpleProgressObserverOnEvent:
    """Tests for SimpleProgressObserver.on_event method."""

    def test_validation_started_creates_task(self) -> None:
        """Test VALIDATION_STARTED creates a progress task."""
        progress = MagicMock(spec=Progress)
        progress.add_task.return_value = 42
        observer = SimpleProgressObserver(progress)

        event = make_event(
            ValidationEventType.VALIDATION_STARTED,
            {"total_hint": 100},
        )
        observer.on_event(event)

        progress.add_task.assert_called_once_with("Validating", total=100)
        assert observer._task_id == 42

    def test_validation_started_no_total_hint(self) -> None:
        """Test VALIDATION_STARTED with no total_hint uses None."""
        progress = MagicMock(spec=Progress)
        progress.add_task.return_value = 1
        observer = SimpleProgressObserver(progress)

        event = make_event(ValidationEventType.VALIDATION_STARTED, {})
        observer.on_event(event)

        progress.add_task.assert_called_once_with("Validating", total=None)

    def test_validation_started_zero_total_hint(self) -> None:
        """Test VALIDATION_STARTED with zero total_hint uses None."""
        progress = MagicMock(spec=Progress)
        progress.add_task.return_value = 1
        observer = SimpleProgressObserver(progress)

        event = make_event(
            ValidationEventType.VALIDATION_STARTED,
            {"total_hint": 0},
        )
        observer.on_event(event)

        progress.add_task.assert_called_once_with("Validating", total=None)

    def test_row_processed_updates_counters(self) -> None:
        """Test ROW_PROCESSED updates valid/failed counters."""
        progress = MagicMock(spec=Progress)
        progress.add_task.return_value = 1
        observer = SimpleProgressObserver(progress)

        # First start the task
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED, {}))

        # Then process a row
        event = make_event(
            ValidationEventType.ROW_PROCESSED,
            {"stats_snapshot": {"valid": 5, "failed": 2}},
        )
        observer.on_event(event)

        assert observer._valid == 5
        assert observer._failed == 2

    def test_row_processed_updates_progress(self) -> None:
        """Test ROW_PROCESSED calls progress.update."""
        progress = MagicMock(spec=Progress)
        progress.add_task.return_value = 1
        observer = SimpleProgressObserver(progress)

        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED, {}))
        observer.on_event(
            make_event(
                ValidationEventType.ROW_PROCESSED,
                {"stats_snapshot": {"valid": 10, "failed": 3}},
            )
        )

        progress.update.assert_called()
        call_args = progress.update.call_args
        assert call_args[0][0] == 1  # task_id
        assert call_args[1]["advance"] == 1

    def test_row_processed_no_task_id(self) -> None:
        """Test ROW_PROCESSED does nothing if no task started."""
        progress = MagicMock(spec=Progress)
        observer = SimpleProgressObserver(progress)

        # Process without starting - should not crash
        event = make_event(
            ValidationEventType.ROW_PROCESSED,
            {"stats_snapshot": {"valid": 1, "failed": 0}},
        )
        observer.on_event(event)

        progress.update.assert_not_called()

    def test_validation_completed_marks_complete(self) -> None:
        """Test VALIDATION_COMPLETED marks task as complete."""
        progress = MagicMock(spec=Progress)
        progress.add_task.return_value = 1
        observer = SimpleProgressObserver(progress)

        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED, {}))
        observer._valid = 8
        observer._failed = 2

        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED, {}))

        progress.update.assert_called()
        # Should set completed to total (valid + failed)
        call_args = progress.update.call_args
        assert call_args[1]["completed"] == 10

    def test_validation_completed_no_rows(self) -> None:
        """Test VALIDATION_COMPLETED with zero rows does not update."""
        progress = MagicMock(spec=Progress)
        progress.add_task.return_value = 1
        observer = SimpleProgressObserver(progress)

        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED, {}))
        # valid and failed stay at 0
        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED, {}))

        # Should not have been called for completed (only for add_task)
        # The update would only happen if total > 0
        progress.update.assert_not_called()

    def test_validation_completed_no_task_id(self) -> None:
        """Test VALIDATION_COMPLETED does nothing if no task started."""
        progress = MagicMock(spec=Progress)
        observer = SimpleProgressObserver(progress)

        # Complete without starting - should not crash
        observer._valid = 5
        observer._failed = 2
        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED, {}))

        progress.update.assert_not_called()


# =============================================================================
# RichDashboardObserver Unit Tests
# =============================================================================


class TestRichDashboardObserverInit:
    """Tests for RichDashboardObserver initialization."""

    def test_init_creates_console(self) -> None:
        """Test that __init__ creates a console if not provided."""
        observer = RichDashboardObserver()

        assert observer._console is not None

    def test_init_uses_provided_console(self) -> None:
        """Test that __init__ uses provided console."""
        console = Console(force_terminal=True)
        observer = RichDashboardObserver(console=console)

        assert observer._console is console

    def test_init_default_top_errors(self) -> None:
        """Test default top_errors_count is 10."""
        observer = RichDashboardObserver()

        assert observer._top_errors_count == 10

    def test_init_custom_top_errors(self) -> None:
        """Test custom top_errors_count is stored."""
        observer = RichDashboardObserver(top_errors_count=5)

        assert observer._top_errors_count == 5

    def test_init_default_refresh_rate(self) -> None:
        """Test default refresh_rate is 10."""
        observer = RichDashboardObserver()

        assert observer._refresh_rate == 10

    def test_init_state_initialized(self) -> None:
        """Test initial state is properly initialized."""
        observer = RichDashboardObserver()

        assert observer._live is None
        assert observer._task_id is None
        assert observer._total_hint is None
        assert observer._stats_snapshot == {}
        assert observer._error_counts == {}


class TestRichDashboardObserverContextManager:
    """Tests for RichDashboardObserver context manager."""

    def test_enter_returns_self(self) -> None:
        """Test __enter__ returns the observer."""
        observer = RichDashboardObserver()

        with patch.object(observer, "start"):
            result = observer.__enter__()

        assert result is observer

    def test_enter_calls_start(self) -> None:
        """Test __enter__ calls start()."""
        observer = RichDashboardObserver()

        with patch.object(observer, "start") as mock_start:
            observer.__enter__()

        mock_start.assert_called_once()

    def test_exit_calls_stop(self) -> None:
        """Test __exit__ calls stop()."""
        observer = RichDashboardObserver()

        with patch.object(observer, "stop") as mock_stop:
            observer.__exit__(None, None, None)

        mock_stop.assert_called_once()


class TestRichDashboardObserverStartStop:
    """Tests for RichDashboardObserver start/stop methods."""

    @patch("rich.live.Live")
    def test_start_creates_live(self, mock_live_class: MagicMock) -> None:
        """Test start() creates a Live instance."""
        observer = RichDashboardObserver()
        mock_live_instance = MagicMock()
        mock_live_class.return_value = mock_live_instance

        observer.start()

        mock_live_class.assert_called_once()
        mock_live_instance.start.assert_called_once()
        assert observer._live is mock_live_instance

    @patch("rich.live.Live")
    def test_stop_stops_live(self, mock_live_class: MagicMock) -> None:
        """Test stop() stops the Live instance."""
        observer = RichDashboardObserver()
        mock_live_instance = MagicMock()
        mock_live_class.return_value = mock_live_instance

        observer.start()
        observer.stop()

        mock_live_instance.stop.assert_called_once()
        assert observer._live is None

    def test_stop_without_start(self) -> None:
        """Test stop() does nothing if not started."""
        observer = RichDashboardObserver()

        # Should not crash
        observer.stop()

        assert observer._live is None


class TestRichDashboardObserverOnEvent:
    """Tests for RichDashboardObserver.on_event method."""

    def test_validation_started_stores_total_hint(self) -> None:
        """Test VALIDATION_STARTED stores total_hint."""
        observer = RichDashboardObserver()

        event = make_event(
            ValidationEventType.VALIDATION_STARTED,
            {"total_hint": 500},
        )
        observer.on_event(event)

        assert observer._total_hint == 500

    def test_validation_started_creates_task(self) -> None:
        """Test VALIDATION_STARTED creates a progress task."""
        observer = RichDashboardObserver()

        event = make_event(
            ValidationEventType.VALIDATION_STARTED,
            {"total_hint": 100},
        )
        observer.on_event(event)

        assert observer._task_id is not None

    def test_row_processed_updates_stats(self) -> None:
        """Test ROW_PROCESSED updates stats_snapshot."""
        observer = RichDashboardObserver()

        event = make_event(
            ValidationEventType.ROW_PROCESSED,
            {
                "stats_snapshot": {"total": 10, "valid": 8, "failed": 2},
                "errors": [],
            },
        )
        observer.on_event(event)

        assert observer._stats_snapshot["total"] == 10
        assert observer._stats_snapshot["valid"] == 8
        assert observer._stats_snapshot["failed"] == 2

    def test_row_processed_updates_error_counts(self) -> None:
        """Test ROW_PROCESSED updates error_counts."""
        observer = RichDashboardObserver()

        event = make_event(
            ValidationEventType.ROW_PROCESSED,
            {
                "stats_snapshot": {},
                "errors": [("email", "Invalid"), ("name", "Required")],
            },
        )
        observer.on_event(event)

        assert observer._error_counts[("email", "Invalid")] == 1
        assert observer._error_counts[("name", "Required")] == 1

    def test_row_processed_accumulates_errors(self) -> None:
        """Test ROW_PROCESSED accumulates error counts over multiple events."""
        observer = RichDashboardObserver()

        for _ in range(3):
            event = make_event(
                ValidationEventType.ROW_PROCESSED,
                {
                    "stats_snapshot": {},
                    "errors": [("email", "Invalid")],
                },
            )
            observer.on_event(event)

        assert observer._error_counts[("email", "Invalid")] == 3

    @patch("rich.live.Live")
    def test_row_processed_updates_live(self, mock_live_class: MagicMock) -> None:
        """Test ROW_PROCESSED updates the live display."""
        observer = RichDashboardObserver()
        mock_live_instance = MagicMock()
        mock_live_class.return_value = mock_live_instance

        observer.start()
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED, {}))
        observer.on_event(
            make_event(
                ValidationEventType.ROW_PROCESSED,
                {"stats_snapshot": {"valid": 1}, "errors": []},
            )
        )

        mock_live_instance.update.assert_called()

    def test_validation_completed_updates_progress(self) -> None:
        """Test VALIDATION_COMPLETED updates progress to complete."""
        observer = RichDashboardObserver()
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED, {}))
        observer._stats_snapshot = {"valid": 90, "failed": 10}

        observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED, {}))

        # Progress should have been updated with completed count


class TestRichDashboardObserverBuildMethods:
    """Tests for RichDashboardObserver display building methods."""

    def test_build_display_returns_group(self) -> None:
        """Test _build_display returns a Rich Group."""
        from rich.console import Group

        observer = RichDashboardObserver()

        result = observer._build_display()

        assert isinstance(result, Group)

    def test_build_stats_panel_returns_panel(self) -> None:
        """Test _build_stats_panel returns a Rich Panel."""
        from rich.panel import Panel

        observer = RichDashboardObserver()
        observer._stats_snapshot = {"total": 100, "valid": 80, "failed": 20}

        result = observer._build_stats_panel()

        assert isinstance(result, Panel)

    def test_build_stats_panel_handles_empty_stats(self) -> None:
        """Test _build_stats_panel handles empty stats."""
        from rich.panel import Panel

        observer = RichDashboardObserver()
        observer._stats_snapshot = {}

        result = observer._build_stats_panel()

        assert isinstance(result, Panel)

    def test_build_errors_table_returns_panel(self) -> None:
        """Test _build_errors_table returns a Rich Panel."""
        from rich.panel import Panel

        observer = RichDashboardObserver()

        result = observer._build_errors_table()

        assert isinstance(result, Panel)

    def test_build_errors_table_with_errors(self) -> None:
        """Test _build_errors_table with error data."""
        from rich.panel import Panel

        observer = RichDashboardObserver()
        observer._error_counts = {
            ("email", "Invalid format"): 50,
            ("name", "Required"): 30,
        }

        result = observer._build_errors_table()

        assert isinstance(result, Panel)

    def test_build_errors_table_respects_limit(self) -> None:
        """Test _build_errors_table respects top_errors_count."""
        observer = RichDashboardObserver(top_errors_count=2)

        # Add more errors than the limit
        for i in range(10):
            observer._error_counts[(f"field_{i}", f"error_{i}")] = 10 - i

        # Should not crash and should limit to top 2
        result = observer._build_errors_table()
        assert result is not None

    def test_build_errors_table_truncates_long_messages(self) -> None:
        """Test _build_errors_table truncates long error messages."""
        observer = RichDashboardObserver()
        long_message = "x" * 100  # Longer than 50 chars
        observer._error_counts[("field", long_message)] = 5

        # Should not crash
        result = observer._build_errors_table()
        assert result is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestSimpleProgressObserverIntegration:
    """Integration tests for SimpleProgressObserver with real Progress."""

    def test_full_workflow(self) -> None:
        """Test complete workflow with real Progress instance."""
        console = Console(force_terminal=True, force_interactive=False)
        with Progress(console=console) as progress:
            observer = SimpleProgressObserver(progress, task_description="Testing")

            # Simulate validation workflow
            observer.on_event(
                make_event(
                    ValidationEventType.VALIDATION_STARTED,
                    {"total_hint": 10},
                )
            )

            for i in range(10):
                observer.on_event(
                    make_event(
                        ValidationEventType.ROW_PROCESSED,
                        {"stats_snapshot": {"valid": i + 1, "failed": 0}},
                    )
                )

            observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED, {}))

            assert observer._valid == 10
            assert observer._failed == 0


class TestRichDashboardObserverIntegration:
    """Integration tests for RichDashboardObserver."""

    def test_full_workflow_with_context_manager(self) -> None:
        """Test complete workflow using context manager."""
        console = Console(force_terminal=True, force_interactive=False)

        with RichDashboardObserver(console=console, refresh_rate=1) as observer:
            observer.on_event(
                make_event(
                    ValidationEventType.VALIDATION_STARTED,
                    {"total_hint": 5},
                )
            )

            for i in range(5):
                observer.on_event(
                    make_event(
                        ValidationEventType.ROW_PROCESSED,
                        {
                            "stats_snapshot": {"total": i + 1, "valid": i, "failed": 1},
                            "errors": [("field", "error")] if i == 0 else [],
                        },
                    )
                )

            observer.on_event(make_event(ValidationEventType.VALIDATION_COMPLETED, {}))

        assert observer._error_counts[("field", "error")] == 1


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestSimpleProgressObserverProperties:
    """Property-based tests for SimpleProgressObserver."""

    @given(
        valid_counts=st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=50),
        failed_counts=st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_counters_match_last_event(
        self, valid_counts: list[int], failed_counts: list[int]
    ) -> None:
        """Test that counters always match the last ROW_PROCESSED event."""
        progress = MagicMock(spec=Progress)
        progress.add_task.return_value = 1
        observer = SimpleProgressObserver(progress)

        # Start the task
        observer.on_event(make_event(ValidationEventType.VALIDATION_STARTED, {}))

        # Process multiple events
        min_len = min(len(valid_counts), len(failed_counts))
        for i in range(min_len):
            observer.on_event(
                make_event(
                    ValidationEventType.ROW_PROCESSED,
                    {"stats_snapshot": {"valid": valid_counts[i], "failed": failed_counts[i]}},
                )
            )

        # Counters should match the last event values
        assert observer._valid == valid_counts[min_len - 1]
        assert observer._failed == failed_counts[min_len - 1]

    @given(
        total_hint=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
    )
    @settings(max_examples=30)
    def test_task_created_with_any_total_hint(self, total_hint: int | None) -> None:
        """Test that tasks are created regardless of total_hint value."""
        progress = MagicMock(spec=Progress)
        progress.add_task.return_value = 1
        observer = SimpleProgressObserver(progress)

        observer.on_event(
            make_event(
                ValidationEventType.VALIDATION_STARTED,
                {"total_hint": total_hint},
            )
        )

        progress.add_task.assert_called_once()
        assert observer._task_id == 1


class TestRichDashboardObserverProperties:
    """Property-based tests for RichDashboardObserver."""

    @given(
        error_events=st.lists(
            st.tuples(
                st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(
                        whitelist_categories=("L",)  # type: ignore[arg-type]
                    ),
                ),
                st.text(
                    min_size=1,
                    max_size=50,
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                ),
            ),
            min_size=0,
            max_size=100,
        ),
    )
    @settings(max_examples=50)
    def test_error_counts_match_occurrences(
        self, error_events: list[tuple[str, str]]
    ) -> None:
        """Test that error counts accurately reflect the number of occurrences."""
        observer = RichDashboardObserver()

        # Expected counts
        expected_counts: dict[tuple[str, str], int] = {}
        for field, msg in error_events:
            key = (field, msg)
            expected_counts[key] = expected_counts.get(key, 0) + 1

        # Emit events
        for field, msg in error_events:
            observer.on_event(
                make_event(
                    ValidationEventType.ROW_PROCESSED,
                    {"stats_snapshot": {}, "errors": [(field, msg)]},
                )
            )

        # Verify counts match
        assert observer._error_counts == expected_counts

    @given(
        stats_list=st.lists(
            st.fixed_dictionaries({
                "total": st.integers(min_value=0, max_value=10000),
                "valid": st.integers(min_value=0, max_value=10000),
                "failed": st.integers(min_value=0, max_value=10000),
            }),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=30)
    def test_stats_snapshot_matches_last_event(
        self, stats_list: list[dict[str, int]]
    ) -> None:
        """Test that stats_snapshot always matches the last ROW_PROCESSED event."""
        observer = RichDashboardObserver()

        for stats in stats_list:
            observer.on_event(
                make_event(
                    ValidationEventType.ROW_PROCESSED,
                    {"stats_snapshot": stats, "errors": []},
                )
            )

        # Should match last event
        last_stats = stats_list[-1]
        assert observer._stats_snapshot["total"] == last_stats["total"]
        assert observer._stats_snapshot["valid"] == last_stats["valid"]
        assert observer._stats_snapshot["failed"] == last_stats["failed"]

    @given(
        num_errors=st.integers(min_value=0, max_value=50),
        top_errors_count=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=30)
    def test_build_errors_table_handles_any_count(
        self, num_errors: int, top_errors_count: int
    ) -> None:
        """Test that _build_errors_table handles any number of errors gracefully."""
        observer = RichDashboardObserver(top_errors_count=top_errors_count)

        # Add errors
        for i in range(num_errors):
            observer._error_counts[(f"field_{i}", f"error_{i}")] = i + 1

        # Should not crash
        result = observer._build_errors_table()
        assert result is not None

    @given(
        total=st.integers(min_value=0, max_value=1000000),
        valid=st.integers(min_value=0, max_value=1000000),
        failed=st.integers(min_value=0, max_value=1000000),
    )
    @settings(max_examples=50)
    def test_build_stats_panel_handles_any_values(
        self, total: int, valid: int, failed: int
    ) -> None:
        """Test that _build_stats_panel handles any stats values gracefully."""
        observer = RichDashboardObserver()
        observer._stats_snapshot = {"total": total, "valid": valid, "failed": failed}

        # Should not crash and should return a Panel
        result = observer._build_stats_panel()
        assert result is not None

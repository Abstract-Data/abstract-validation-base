"""Rich-based observers for validation progress display.

Provides Rich console UI components for displaying validation progress,
statistics, and error patterns in real-time.

Requires the 'rich' package: pip install rich
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from abstract_validation_base.events import (
    ValidationEvent,
    ValidationEventType,
    ValidationObserver,
)

if TYPE_CHECKING:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, TaskID

__all__ = ["SimpleProgressObserver", "RichDashboardObserver"]


class SimpleProgressObserver(ValidationObserver):
    """Simple progress bar with pass/fail counts.

    Displays a progress bar with valid/failed counters using Rich.
    Must be used within a Rich Progress context.

    Example:
        from rich.progress import Progress

        with Progress() as progress:
            observer = SimpleProgressObserver(progress)
            runner.add_observer(observer)

            for result in runner.run():
                process(result)

    Requires:
        pip install rich
    """

    def __init__(
        self,
        progress: Progress,
        task_description: str = "Validating",
    ) -> None:
        """Initialize the progress observer.

        Args:
            progress: A Rich Progress instance (must be started).
            task_description: Description text shown in the progress bar.
        """
        self._progress = progress
        self._task_id: TaskID | None = None
        self._description = task_description
        self._valid = 0
        self._failed = 0

    def on_event(self, event: ValidationEvent) -> None:
        """Handle validation events to update the progress bar.

        Args:
            event: The validation event to handle.
        """
        if event.event_type == ValidationEventType.VALIDATION_STARTED:
            total = event.data.get("total_hint") or 0
            self._task_id = self._progress.add_task(
                self._description,
                total=total if total > 0 else None,
            )

        elif event.event_type == ValidationEventType.ROW_PROCESSED:
            stats = event.data.get("stats_snapshot", {})
            self._valid = stats.get("valid", 0)
            self._failed = stats.get("failed", 0)

            if self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    advance=1,
                    description=(
                        f"{self._description} [green]✓{self._valid}[/] [red]✗{self._failed}[/]"
                    ),
                )

        elif event.event_type == ValidationEventType.VALIDATION_COMPLETED:
            if self._task_id is not None:
                # Mark task as complete
                total = self._valid + self._failed
                if total > 0:
                    self._progress.update(self._task_id, completed=total)


class RichDashboardObserver(ValidationObserver):
    """Live dashboard showing validation progress and top errors.

    Displays:
    - Progress bar with pass/fail counts
    - Live table of top errors with percentages
    - Summary statistics panel

    Example:
        observer = RichDashboardObserver()
        runner.add_observer(observer)

        with observer:  # Context manager starts/stops Live display
            for result in runner.run():
                process(result)

        # Or manually:
        observer.start()
        for result in runner.run():
            process(result)
        observer.stop()

    Requires:
        pip install rich
    """

    def __init__(
        self,
        console: Console | None = None,
        top_errors_count: int = 10,
        refresh_rate: int = 10,
    ) -> None:
        """Initialize the dashboard observer.

        Args:
            console: Rich Console instance. If None, creates a new one.
            top_errors_count: Number of top errors to display.
            refresh_rate: Display refresh rate per second.
        """
        # Import Rich components here to make them optional
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
        )

        self._console = console or Console()
        self._top_errors_count = top_errors_count
        self._refresh_rate = refresh_rate

        # State
        self._live: Live | None = None
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TextColumn("[green]✓ {task.fields[valid]}[/]"),
            TextColumn("[red]✗ {task.fields[failed]}[/]"),
            console=self._console,
        )
        self._task_id: TaskID | None = None

        # Stats for display
        self._total_hint: int | None = None
        self._stats_snapshot: dict[str, int | None] = {}
        self._error_counts: dict[tuple[str, str], int] = {}

    def __enter__(self) -> RichDashboardObserver:
        """Start the live display."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop the live display."""
        self.stop()

    def start(self) -> None:
        """Start the live display."""
        from rich.live import Live

        self._live = Live(
            self._build_display(),
            console=self._console,
            refresh_per_second=self._refresh_rate,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def on_event(self, event: ValidationEvent) -> None:
        """Handle validation events to update the dashboard.

        Args:
            event: The validation event to handle.
        """
        if event.event_type == ValidationEventType.VALIDATION_STARTED:
            self._total_hint = event.data.get("total_hint")
            self._task_id = self._progress.add_task(
                "Processing",
                total=self._total_hint,
                valid=0,
                failed=0,
            )

        elif event.event_type == ValidationEventType.ROW_PROCESSED:
            self._stats_snapshot = event.data.get("stats_snapshot", {})

            # Update error counts
            for field, msg in event.data.get("errors", []):
                key = (field, msg)
                self._error_counts[key] = self._error_counts.get(key, 0) + 1

            # Update progress
            if self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    advance=1,
                    valid=self._stats_snapshot.get("valid", 0),
                    failed=self._stats_snapshot.get("failed", 0),
                )

            # Refresh display
            if self._live:
                self._live.update(self._build_display())

        elif event.event_type == ValidationEventType.VALIDATION_COMPLETED:
            if self._task_id is not None:
                total = (self._stats_snapshot.get("valid", 0) or 0) + (
                    self._stats_snapshot.get("failed", 0) or 0
                )
                if total > 0:
                    self._progress.update(self._task_id, completed=total)
            if self._live:
                self._live.update(self._build_display())

    def _build_display(self) -> Group:
        """Build the full dashboard display.

        Returns:
            Rich Group containing all dashboard components.
        """
        from rich.console import Group

        return Group(
            self._progress,
            "",  # Spacer
            self._build_stats_panel(),
            self._build_errors_table(),
        )

    def _build_stats_panel(self) -> Panel:
        """Build the statistics panel.

        Returns:
            Rich Panel containing statistics.
        """
        from rich.panel import Panel
        from rich.text import Text

        total = self._stats_snapshot.get("total", 0) or 0
        valid = self._stats_snapshot.get("valid", 0) or 0
        failed = self._stats_snapshot.get("failed", 0) or 0

        rate = (valid / total * 100) if total > 0 else 0

        text = Text()
        text.append(f"Total: {total:,}  ", style="bold")
        text.append(f"Valid: {valid:,}  ", style="green")
        text.append(f"Failed: {failed:,}  ", style="red")
        text.append(f"Success Rate: {rate:.1f}%", style="bold cyan")

        return Panel(text, title="[bold]Statistics[/]", border_style="blue")

    def _build_errors_table(self) -> Panel:
        """Build the top errors table.

        Returns:
            Rich Panel containing the errors table.
        """
        from rich.panel import Panel
        from rich.table import Table

        table = Table(
            title="Top Errors",
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Error", style="yellow")
        table.add_column("Count", justify="right", style="red", width=10)
        table.add_column("%", justify="right", width=8)

        # Sort by count
        total_errors = sum(self._error_counts.values())
        sorted_errors = sorted(
            self._error_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[: self._top_errors_count]

        for (field, msg), count in sorted_errors:
            pct = (count / total_errors * 100) if total_errors > 0 else 0
            # Truncate long messages
            display_msg = msg[:50] + "..." if len(msg) > 50 else msg
            table.add_row(field, display_msg, f"{count:,}", f"{pct:.1f}%")

        if not sorted_errors:
            table.add_row("-", "No errors yet", "-", "-")

        return Panel(table, border_style="red")

"""Streaming validation runner for large files.

Provides memory-efficient batch validation using iterators and generators.
Results are streamed rather than stored in memory. Supports parallel
processing with configurable worker count for large datasets.
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import islice
from threading import Lock
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from pydantic import ValidationError

from abstract_validation_base.base import ValidationBase
from abstract_validation_base.events import (
    ObservableMixin,
    ValidationEvent,
    ValidationEventType,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from abstract_validation_base.results import ValidationResult
    from abstract_validation_base.validators import BaseValidator

__all__ = ["RowResult", "RunnerStats", "ValidationRunner"]

T = TypeVar("T", bound=ValidationBase)


@dataclass
class RowResult(Generic[T]):
    """Result for a single row validation.

    Attributes:
        row_index: Zero-based index of the row in the input data.
        raw_data: The original dict data for this row.
        model: The validated model instance if Pydantic validation passed.
        pydantic_errors: List of Pydantic validation errors if any.
        validator_result: Result from custom validators if any were run.

    Example:
        result = RowResult(row_index=0, raw_data={"name": "test"})
        if result.is_valid:
            save(result.model)
        else:
            log_errors(result.error_summary)
    """

    row_index: int
    raw_data: dict[str, Any]
    model: T | None = None
    pydantic_errors: list[dict[str, Any]] = field(default_factory=list)
    validator_result: ValidationResult | None = None

    @property
    def is_valid(self) -> bool:
        """Check if the row passed all validation stages."""
        if self.pydantic_errors:
            return False
        return not (self.validator_result and not self.validator_result.is_valid)

    @property
    def error_summary(self) -> list[tuple[str, str]]:
        """Get (field, message) tuples for all errors.

        Returns:
            List of (field_name, error_message) tuples from both
            Pydantic and custom validator errors.
        """
        errors: list[tuple[str, str]] = []
        for pydantic_err in self.pydantic_errors:
            field_name = str(pydantic_err.get("loc", ["unknown"])[-1])
            msg = pydantic_err.get("msg", pydantic_err.get("type", "validation_error"))
            errors.append((field_name, msg))
        if self.validator_result:
            for validator_err in self.validator_result.errors:
                errors.append((validator_err.field, validator_err.message))
        return errors


@dataclass
class RunnerStats:
    """Streaming statistics tracker for validation runs.

    Tracks counts, timing, error patterns, and sample failures
    during validation without storing all records in memory.

    Attributes:
        total_rows: Total number of rows processed.
        valid_rows: Number of rows that passed all validation.
        pydantic_failures: Number of rows that failed Pydantic validation.
        validator_failures: Number of rows that failed custom validators.
        start_time: Start time of the validation run.
        end_time: End time of the validation run.
        error_counts: Counter of (field, message) -> occurrence count.
        failed_samples: Sample of failed raw data for debugging.
        max_samples: Maximum number of failed samples to keep.
    """

    total_rows: int = 0
    valid_rows: int = 0
    pydantic_failures: int = 0
    validator_failures: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    # Error tracking (field, message) -> count
    error_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    # Sample of failed raw data for debugging (limited)
    failed_samples: list[dict[str, Any]] = field(default_factory=list)
    max_samples: int = 100

    @property
    def error_rows(self) -> int:
        """Get the number of rows with errors."""
        return self.total_rows - self.valid_rows

    @property
    def duration_ms(self) -> float:
        """Get the duration of the validation run in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def success_rate(self) -> float:
        """Get the success rate as a percentage (0-100)."""
        if self.total_rows == 0:
            return 0.0
        return self.valid_rows / self.total_rows * 100

    def top_errors(self, n: int = 10) -> list[tuple[tuple[str, str], int, float]]:
        """Get top N errors with counts and percentages.

        Args:
            n: Maximum number of errors to return.

        Returns:
            List of ((field, message), count, percentage) tuples,
            sorted by count descending.
        """
        total_errors = sum(self.error_counts.values())
        result: list[tuple[tuple[str, str], int, float]] = []
        for (field_name, msg), count in self.error_counts.most_common(n):
            pct = (count / total_errors * 100) if total_errors > 0 else 0
            result.append(((field_name, msg), count, pct))
        return result

    def record_error(self, field: str, message: str) -> None:
        """Record an error occurrence.

        Args:
            field: The field name where the error occurred.
            message: The error message.
        """
        self.error_counts[(field, message)] += 1

    def add_failed_sample(self, raw_data: dict[str, Any]) -> None:
        """Add a failed sample if under the sample limit.

        Args:
            raw_data: The raw dict data that failed validation.
        """
        if len(self.failed_samples) < self.max_samples:
            self.failed_samples.append(raw_data)

    def merge(self, other: RunnerStats) -> None:
        """Merge another RunnerStats into this one.

        Used for combining partial stats from parallel workers.

        Args:
            other: Another RunnerStats to merge into this one.
        """
        self.total_rows += other.total_rows
        self.valid_rows += other.valid_rows
        self.pydantic_failures += other.pydantic_failures
        self.validator_failures += other.validator_failures
        self.error_counts.update(other.error_counts)

        # Add failed samples up to the limit
        remaining = self.max_samples - len(self.failed_samples)
        if remaining > 0:
            self.failed_samples.extend(other.failed_samples[:remaining])


class ValidationRunner(ObservableMixin, Generic[T]):
    """Streaming batch validator for large files.

    Memory-efficient: yields results as iterator, doesn't store all records.
    Tracks statistics and error patterns for audit reporting. Supports
    parallel processing for large datasets.

    Example:
        runner = ValidationRunner(csv_reader, MyModel)

        # Streaming: process records without storing all in memory
        for result in runner.run():
            if result.is_valid:
                save_to_database(result.model)
            else:
                log_failure(result.raw_data)

        # After iteration, stats are available
        print(runner.stats)
        print(runner.stats.top_errors())

        # Parallel processing for large files (>1M rows)
        for result in runner.run(workers=4, chunk_size=10000):
            process(result)

        # Context manager support
        with ValidationRunner(data, MyModel) as runner:
            for result in runner.run():
                process(result)
            # Stats automatically finalized on exit

        # Or use convenience methods
        for model in runner.run_collect_valid():
            db.insert(model)

        # Batch inserts
        for batch in runner.run_batch_valid(batch_size=100):
            db.insert_many(batch)
    """

    def __init__(
        self,
        data: Iterator[dict[str, Any]],
        model_class: type[T],
        *,
        validators: BaseValidator[T] | None = None,
        fail_fast: bool = False,
        total_hint: int | None = None,
    ) -> None:
        """Initialize the validation runner.

        Args:
            data: Iterator of dicts (e.g., csv.DictReader). NOT materialized.
            model_class: The ValidationBase subclass to instantiate.
            validators: Optional validator(s) for custom business logic.
            fail_fast: If True, stop on first error.
            total_hint: Optional total count for progress percentage.
        """
        self._data = data
        self._model_class = model_class
        self._validators = validators
        self._fail_fast = fail_fast
        self._total_hint = total_hint

        # Statistics (updated during iteration)
        self._stats = RunnerStats()
        self._stats_lock = Lock()

    def __enter__(self) -> ValidationRunner[T]:
        """Enter context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager, ensuring stats are finalized."""
        if self._stats.end_time == 0 and self._stats.start_time > 0:
            self._stats.end_time = time.perf_counter()

    def run(
        self,
        *,
        workers: int | None = None,
        chunk_size: int = 10000,
        max_pending: int | None = None,
    ) -> Iterator[RowResult[T]]:
        """Run validation, yielding results as they're processed.

        This is a generator - results are not stored in memory.
        Statistics are tracked internally for audit_report().

        Args:
            workers: Number of parallel workers. None or 1 for sequential
                processing. Values > 1 enable parallel chunk processing.
            chunk_size: Number of rows per chunk when using parallel processing.
                Defaults to 10000. Ignored for sequential processing.
            max_pending: Maximum number of chunks to buffer in parallel mode.
                Controls memory usage for streaming data sources.
                Defaults to workers * 2 if not specified.

        Yields:
            RowResult for each row processed, in original order.

        Note:
            Emits VALIDATION_STARTED at the beginning, ROW_PROCESSED for
            each row, and VALIDATION_COMPLETED at the end.

            When workers > 1, uses streaming parallel processing with bounded
            buffers. Only max_pending chunks are held in memory at a time,
            making it efficient for large iterator-based data sources.
        """
        if workers and workers > 1:
            effective_max_pending = max_pending if max_pending else workers * 2
            yield from self._run_parallel(workers, chunk_size, effective_max_pending)
        else:
            yield from self._run_sequential()

    def _run_sequential(self) -> Iterator[RowResult[T]]:
        """Run validation sequentially (original behavior).

        Yields:
            RowResult for each row processed.
        """
        self._stats = RunnerStats(start_time=time.perf_counter())

        # Emit start event
        self.notify(
            ValidationEvent(
                event_type=ValidationEventType.VALIDATION_STARTED,
                source=self,
                data={
                    "model_class": self._model_class.__name__,
                    "total_hint": self._total_hint,
                },
            )
        )

        for row_index, item in enumerate(self._data):
            result = self._validate_row(row_index, item)
            self._update_stats(result)

            # Emit progress event
            self._emit_row_processed_event(result)

            yield result

            if self._fail_fast and not result.is_valid:
                break

        self._stats.end_time = time.perf_counter()

        # Emit completion event
        self.notify(
            ValidationEvent(
                event_type=ValidationEventType.VALIDATION_COMPLETED,
                source=self,
                data={
                    "stats": self._stats,
                },
            )
        )

    def _run_parallel(
        self,
        workers: int,
        chunk_size: int,
        max_pending: int,
    ) -> Iterator[RowResult[T]]:
        """Run validation in parallel using streaming chunked processing.

        Processes data in chunks using a thread pool with bounded buffers,
        maintaining original row order in output. Only max_pending chunks
        are held in memory at a time (backpressure).

        Args:
            workers: Number of parallel workers.
            chunk_size: Number of rows per chunk.
            max_pending: Maximum chunks to buffer (controls memory).

        Yields:
            RowResult for each row processed, in original order.
        """
        self._stats = RunnerStats(start_time=time.perf_counter())

        # Emit start event
        self.notify(
            ValidationEvent(
                event_type=ValidationEventType.VALIDATION_STARTED,
                source=self,
                data={
                    "model_class": self._model_class.__name__,
                    "total_hint": self._total_hint,
                    "workers": workers,
                    "chunk_size": chunk_size,
                    "max_pending": max_pending,
                },
            )
        )

        # State for streaming
        global_row_index = 0
        chunk_index = 0
        next_chunk_to_yield = 0
        completed_chunks: dict[int, tuple[list[RowResult[T]], RunnerStats]] = {}
        data_exhausted = False

        def read_next_chunk() -> tuple[int, list[tuple[int, dict[str, Any]]]] | None:
            """Read the next chunk from the input iterator."""
            nonlocal global_row_index, chunk_index, data_exhausted

            if data_exhausted:
                return None

            chunk_data: list[tuple[int, dict[str, Any]]] = []
            for item in islice(self._data, chunk_size):
                chunk_data.append((global_row_index, item))
                global_row_index += 1

            if not chunk_data:
                data_exhausted = True
                return None

            idx = chunk_index
            chunk_index += 1
            return (idx, chunk_data)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Track pending futures: future -> chunk_index
            pending_futures: dict[Any, int] = {}

            # Initial fill: submit up to max_pending chunks
            while len(pending_futures) < max_pending:
                chunk = read_next_chunk()
                if chunk is None:
                    break
                idx, chunk_data = chunk
                future = executor.submit(self._validate_chunk, chunk_data)
                pending_futures[future] = idx

            # Process until all done
            while pending_futures:
                # Wait for at least one future to complete
                done_futures = set()
                for future in as_completed(pending_futures):
                    done_futures.add(future)
                    # Process one at a time to allow yielding
                    break

                for future in done_futures:
                    idx = pending_futures.pop(future)
                    results, partial_stats = future.result()
                    completed_chunks[idx] = (results, partial_stats)

                    # Submit next chunk if available (maintain pressure)
                    if len(pending_futures) < max_pending:
                        chunk = read_next_chunk()
                        if chunk is not None:
                            new_idx, chunk_data = chunk
                            new_future = executor.submit(self._validate_chunk, chunk_data)
                            pending_futures[new_future] = new_idx

                # Yield completed chunks in order
                while next_chunk_to_yield in completed_chunks:
                    chunk_results, partial_stats = completed_chunks.pop(next_chunk_to_yield)

                    # Merge stats (thread-safe)
                    with self._stats_lock:
                        self._stats.merge(partial_stats)

                    # Yield results and emit events
                    for result in chunk_results:
                        self._emit_row_processed_event(result)
                        yield result

                        if self._fail_fast and not result.is_valid:
                            self._stats.end_time = time.perf_counter()
                            return

                    next_chunk_to_yield += 1

        self._stats.end_time = time.perf_counter()

        # Emit completion event
        self.notify(
            ValidationEvent(
                event_type=ValidationEventType.VALIDATION_COMPLETED,
                source=self,
                data={
                    "stats": self._stats,
                },
            )
        )

    def _validate_chunk(
        self,
        chunk_data: list[tuple[int, dict[str, Any]]],
    ) -> tuple[list[RowResult[T]], RunnerStats]:
        """Validate a chunk of rows (called by worker threads).

        Args:
            chunk_data: List of (row_index, raw_data) tuples.

        Returns:
            Tuple of (results list, partial stats for this chunk).
        """
        results: list[RowResult[T]] = []
        partial_stats = RunnerStats()

        for row_index, item in chunk_data:
            result = self._validate_row(row_index, item)
            results.append(result)

            # Update partial stats
            partial_stats.total_rows += 1
            if result.is_valid:
                partial_stats.valid_rows += 1
            else:
                if result.pydantic_errors:
                    partial_stats.pydantic_failures += 1
                if result.validator_result and not result.validator_result.is_valid:
                    partial_stats.validator_failures += 1
                for field_name, msg in result.error_summary:
                    partial_stats.record_error(field_name, msg)
                partial_stats.add_failed_sample(result.raw_data)

        return results, partial_stats

    def run_collect_valid(self) -> Iterator[T]:
        """Convenience: yield only valid models.

        Yields:
            Valid model instances (T) for rows that passed validation.
        """
        for result in self.run():
            if result.is_valid and result.model is not None:
                yield result.model

    def run_collect_failed(self) -> Iterator[RowResult[T]]:
        """Convenience: yield only failed results.

        Yields:
            RowResult instances for rows that failed validation.
        """
        for result in self.run():
            if not result.is_valid:
                yield result

    def run_batch_valid(self, batch_size: int = 100) -> Iterator[list[T]]:
        """Yield batches of valid models for bulk operations.

        Args:
            batch_size: Number of models per batch.

        Yields:
            Lists of valid models, each up to batch_size length.
            The final batch may be smaller.

        Note:
            Emits BATCH_STARTED and BATCH_COMPLETED events for each batch.
        """
        batch: list[T] = []
        batch_number = 0

        for result in self.run():
            if result.is_valid and result.model is not None:
                if len(batch) == 0:
                    # Emit batch started
                    self.notify(
                        ValidationEvent(
                            event_type=ValidationEventType.BATCH_STARTED,
                            source=self,
                            data={
                                "batch_number": batch_number,
                                "batch_size": batch_size,
                            },
                        )
                    )

                batch.append(result.model)

                if len(batch) >= batch_size:
                    # Emit batch completed
                    self.notify(
                        ValidationEvent(
                            event_type=ValidationEventType.BATCH_COMPLETED,
                            source=self,
                            data={
                                "batch_number": batch_number,
                                "batch_size": len(batch),
                            },
                        )
                    )
                    yield batch
                    batch = []
                    batch_number += 1

        # Yield remaining items
        if batch:
            self.notify(
                ValidationEvent(
                    event_type=ValidationEventType.BATCH_COMPLETED,
                    source=self,
                    data={
                        "batch_number": batch_number,
                        "batch_size": len(batch),
                    },
                )
            )
            yield batch

    def _validate_row(self, row_index: int, item: dict[str, Any]) -> RowResult[T]:
        """Validate a single row.

        Args:
            row_index: The index of the row being validated.
            item: The raw dict data for this row.

        Returns:
            RowResult containing validation status and any errors.
        """
        result: RowResult[T] = RowResult(row_index=row_index, raw_data=item)

        # Stage 1: Pydantic validation
        try:
            model = self._model_class(**item)
            result.model = model
        except ValidationError as e:
            result.pydantic_errors = cast("list[dict[str, Any]]", e.errors())
            return result

        # Stage 2: Custom validators
        if self._validators is not None:
            result.validator_result = self._validators.validate(model)

        return result

    def _update_stats(self, result: RowResult[T]) -> None:
        """Update running statistics.

        Args:
            result: The RowResult to incorporate into statistics.
        """
        self._stats.total_rows += 1

        if result.is_valid:
            self._stats.valid_rows += 1
        else:
            if result.pydantic_errors:
                self._stats.pydantic_failures += 1
            if result.validator_result and not result.validator_result.is_valid:
                self._stats.validator_failures += 1

            # Track error patterns
            for field_name, msg in result.error_summary:
                self._stats.record_error(field_name, msg)

            # Keep sample of failures
            self._stats.add_failed_sample(result.raw_data)

    def _emit_row_processed_event(self, result: RowResult[T]) -> None:
        """Emit a ROW_PROCESSED event for UI observers.

        Args:
            result: The RowResult that was just processed.
        """
        self.notify(
            ValidationEvent(
                event_type=ValidationEventType.ROW_PROCESSED,
                source=self,
                data={
                    "row_index": result.row_index,
                    "is_valid": result.is_valid,
                    "stats_snapshot": {
                        "total": self._stats.total_rows,
                        "valid": self._stats.valid_rows,
                        "failed": self._stats.error_rows,
                        "total_hint": self._total_hint,
                    },
                    "errors": result.error_summary if not result.is_valid else [],
                },
            )
        )

    @property
    def stats(self) -> RunnerStats:
        """Get current statistics."""
        return self._stats

    def audit_report(self) -> dict[str, Any]:
        """Get audit report after run completes.

        Returns summary statistics and error patterns,
        NOT full record data (for memory efficiency).

        Returns:
            Dict with 'summary', 'top_errors', and 'failed_samples' keys.
        """
        return {
            "summary": {
                "total_rows": self._stats.total_rows,
                "valid_rows": self._stats.valid_rows,
                "error_rows": self._stats.error_rows,
                "success_rate": f"{self._stats.success_rate:.1f}%",
                "pydantic_failures": self._stats.pydantic_failures,
                "validator_failures": self._stats.validator_failures,
                "duration_ms": self._stats.duration_ms,
            },
            "top_errors": [
                {"field": f, "message": m, "count": c, "percentage": f"{p:.1f}%"}
                for (f, m), c, p in self._stats.top_errors(20)
            ],
            "failed_samples": self._stats.failed_samples,
        }

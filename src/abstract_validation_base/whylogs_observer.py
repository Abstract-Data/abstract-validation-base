"""whylogs observer for data profiling during validation.

Provides integration with whylogs for automated data profiling of both
raw input data and validated/cleaned model outputs during validation runs.

Requires the 'whylogs' package: pip install abstract-validation-base[whylogs]
"""

from __future__ import annotations

try:
    from whylogs.api.writer.local import LocalWriter  # type: ignore[import-untyped]
except ImportError:
    LocalWriter = None

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from abstract_validation_base.events import (
    ValidationEvent,
    ValidationEventType,
    ValidationObserver,
)

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]
    from whylogs.core import DatasetProfileView  # type: ignore[import-untyped]
    from whylogs.core.schema import DatasetSchema  # type: ignore[import-untyped]

__all__ = ["WhylogsObserver", "ProfilePair", "ProfileComparison"]


@dataclass
class ProfileComparison:
    """Comparison results between raw and valid data profiles.

    Provides summary statistics and drift metrics comparing the raw input
    data profile against the valid (post-validation) data profile.

    Attributes:
        raw_column_count: Number of columns in raw data profile.
        valid_column_count: Number of columns in valid data profile.
        raw_row_count: Number of rows profiled in raw data.
        valid_row_count: Number of rows profiled in valid data.
        columns_only_in_raw: Column names present only in raw profile.
        columns_only_in_valid: Column names present only in valid profile.
        common_columns: Column names present in both profiles.
        pass_rate: Percentage of raw rows that passed validation.
    """

    raw_column_count: int
    valid_column_count: int
    raw_row_count: int
    valid_row_count: int
    columns_only_in_raw: list[str]
    columns_only_in_valid: list[str]
    common_columns: list[str]
    pass_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the comparison.
        """
        return {
            "raw_column_count": self.raw_column_count,
            "valid_column_count": self.valid_column_count,
            "raw_row_count": self.raw_row_count,
            "valid_row_count": self.valid_row_count,
            "columns_only_in_raw": self.columns_only_in_raw,
            "columns_only_in_valid": self.columns_only_in_valid,
            "common_columns": self.common_columns,
            "pass_rate": self.pass_rate,
        }


@dataclass
class ProfilePair:
    """Container for raw and valid data profiles.

    Holds the whylogs profiles generated during validation. The raw profile
    contains statistics about all input data, while the valid profile contains
    statistics about only the records that passed validation.

    Attributes:
        raw_profile: Profile of raw input data (None if profiling disabled).
        valid_profile: Profile of valid model data (None if profiling disabled).
    """

    raw_profile: DatasetProfileView | None = None
    valid_profile: DatasetProfileView | None = None

    def to_pandas(self) -> dict[str, pd.DataFrame | None]:
        """Convert profiles to pandas DataFrames.

        Returns:
            Dictionary with 'raw' and 'valid' keys mapping to DataFrames
            containing profile statistics, or None if that profile is empty.

        Example:
            profiles = observer.get_profiles()
            dfs = profiles.to_pandas()
            print(dfs['raw'].describe())
        """
        result: dict[str, Any] = {"raw": None, "valid": None}

        if self.raw_profile is not None:
            result["raw"] = self.raw_profile.to_pandas()

        if self.valid_profile is not None:
            result["valid"] = self.valid_profile.to_pandas()

        return result

    def write(
        self,
        raw_path: str | Path | None = None,
        valid_path: str | Path | None = None,
    ) -> dict[str, Path | None]:
        """Write profiles to disk in whylogs binary format.

        Args:
            raw_path: Path for raw profile. Defaults to 'raw_profile.bin'.
            valid_path: Path for valid profile. Defaults to 'valid_profile.bin'.

        Returns:
            Dictionary with 'raw' and 'valid' keys mapping to the actual
            paths written, or None if that profile was empty.

        Example:
            profiles = observer.get_profiles()
            paths = profiles.write(
                raw_path='output/raw.bin',
                valid_path='output/valid.bin'
            )
        """
        result: dict[str, Path | None] = {"raw": None, "valid": None}

        if self.raw_profile is not None:
            path = Path(raw_path) if raw_path else Path("raw_profile.bin")
            writer = LocalWriter()
            writer.write(self.raw_profile, str(path))
            result["raw"] = path

        if self.valid_profile is not None:
            path = Path(valid_path) if valid_path else Path("valid_profile.bin")
            writer = LocalWriter()
            writer.write(self.valid_profile, str(path))
            result["valid"] = path

        return result


@dataclass
class WhylogsObserver(ValidationObserver):
    """Observer that profiles data during validation using whylogs.

    Collects statistics on both raw input data and validated model outputs,
    enabling data quality monitoring and drift detection. Profiles are
    accumulated in chunks for memory efficiency.

    Thread-safe for use with parallel validation runners.

    Example:
        from abstract_validation_base import ValidationRunner, WhylogsObserver

        observer = WhylogsObserver(
            chunk_size=10000,
            profile_raw=True,
            profile_valid=True,
        )
        runner = ValidationRunner(data, MyModel)
        runner.add_observer(observer)

        for result in runner.run():
            process(result)

        # Get profiles after validation completes
        profiles = observer.get_profiles()
        comparison = observer.compare_profiles()

        # Export profiles
        profiles.write(raw_path='raw.bin', valid_path='valid.bin')

    Requires:
        pip install abstract-validation-base[whylogs]
    """

    chunk_size: int = field(default=10000)
    """Number of rows to buffer before flushing to profile."""

    profile_raw: bool = field(default=True)
    """Whether to profile raw input data."""

    profile_valid: bool = field(default=True)
    """Whether to profile valid model outputs."""

    schema: DatasetSchema | None = field(default=None)
    """Optional whylogs schema for type inference. If None, inferred automatically."""

    # Internal state (not part of init signature)
    _lock: Lock = field(default_factory=Lock, repr=False)
    _raw_buffer: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _valid_buffer: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _raw_profile: DatasetProfileView | None = field(default=None, repr=False)
    _valid_profile: DatasetProfileView | None = field(default=None, repr=False)
    _is_running: bool = field(default=False, repr=False)
    _raw_row_count: int = field(default=0, repr=False)
    _valid_row_count: int = field(default=0, repr=False)

    def on_event(self, event: ValidationEvent) -> None:
        """Handle validation events to build profiles.

        Args:
            event: The validation event to handle.
        """
        if event.event_type == ValidationEventType.VALIDATION_STARTED:
            self._handle_start(event)
        elif event.event_type == ValidationEventType.ROW_PROCESSED:
            self._handle_row(event)
        elif event.event_type == ValidationEventType.VALIDATION_COMPLETED:
            self._handle_complete(event)

    def _handle_start(self, event: ValidationEvent) -> None:
        """Handle VALIDATION_STARTED event - reset state.

        Args:
            event: The validation started event.
        """
        with self._lock:
            self._is_running = True
            self._raw_buffer = []
            self._valid_buffer = []
            self._raw_profile = None
            self._valid_profile = None
            self._raw_row_count = 0
            self._valid_row_count = 0

    def _handle_row(self, event: ValidationEvent) -> None:
        """Handle ROW_PROCESSED event - buffer data for profiling.

        Args:
            event: The row processed event containing raw_data and optional model_dict.
        """
        data = event.data

        with self._lock:
            # Buffer raw data if profiling enabled
            if self.profile_raw:
                raw_data = data.get("raw_data")
                if raw_data is not None:
                    self._raw_buffer.append(raw_data)
                    if len(self._raw_buffer) >= self.chunk_size:
                        self._flush_raw_buffer()

            # Buffer valid model data if profiling enabled and row is valid
            if self.profile_valid and data.get("is_valid", False):
                model_dict = data.get("model_dict")
                if model_dict is not None:
                    self._valid_buffer.append(model_dict)
                    if len(self._valid_buffer) >= self.chunk_size:
                        self._flush_valid_buffer()

    def _handle_complete(self, event: ValidationEvent) -> None:
        """Handle VALIDATION_COMPLETED event - flush remaining buffers.

        Args:
            event: The validation completed event.
        """
        with self._lock:
            # Final flush of any remaining buffered data
            if self._raw_buffer:
                self._flush_raw_buffer()
            if self._valid_buffer:
                self._flush_valid_buffer()
            self._is_running = False

    def _flush_raw_buffer(self) -> None:
        """Profile and merge buffered raw data.

        Must be called while holding _lock.
        """
        if not self._raw_buffer:
            return

        import pandas as pd
        import whylogs as why  # type: ignore[import-untyped]

        df = pd.DataFrame(self._raw_buffer)
        self._raw_row_count += len(df)

        # Profile the chunk
        result = why.log(df, schema=self.schema) if self.schema is not None else why.log(df)

        # Get view from profile (views support merge, profiles don't)
        chunk_view = result.profile().view()

        # Merge with existing profile view
        if self._raw_profile is None:
            self._raw_profile = chunk_view
        else:
            self._raw_profile = self._raw_profile.merge(chunk_view)

        # Clear buffer (create new list to avoid race conditions)
        self._raw_buffer = []

    def _flush_valid_buffer(self) -> None:
        """Profile and merge buffered valid model data.

        Must be called while holding _lock.
        """
        if not self._valid_buffer:
            return

        import pandas as pd
        import whylogs as why

        df = pd.DataFrame(self._valid_buffer)
        self._valid_row_count += len(df)

        # Profile the chunk
        result = why.log(df, schema=self.schema) if self.schema is not None else why.log(df)

        # Get view from profile (views support merge, profiles don't)
        chunk_view = result.profile().view()

        # Merge with existing profile view
        if self._valid_profile is None:
            self._valid_profile = chunk_view
        else:
            self._valid_profile = self._valid_profile.merge(chunk_view)

        # Clear buffer (create new list to avoid race conditions)
        self._valid_buffer = []

    def get_profiles(self) -> ProfilePair:
        """Get the accumulated profiles after validation completes.

        Returns:
            ProfilePair containing views of raw and valid profiles.

        Raises:
            RuntimeError: If called while validation is still running.

        Example:
            profiles = observer.get_profiles()
            raw_df = profiles.to_pandas()['raw']
        """
        with self._lock:
            if self._is_running:
                raise RuntimeError(
                    "Cannot get profiles while validation is running. "
                    "Wait for validation to complete."
                )

            # Profiles are already stored as views
            return ProfilePair(raw_profile=self._raw_profile, valid_profile=self._valid_profile)

    def compare_profiles(self) -> ProfileComparison:
        """Compare raw and valid profiles.

        Computes summary statistics comparing the raw input data profile
        against the valid output data profile.

        Returns:
            ProfileComparison with drift and summary statistics.

        Raises:
            RuntimeError: If called while validation is still running.
            ValueError: If neither raw nor valid profiling is enabled.

        Example:
            comparison = observer.compare_profiles()
            print(f"Pass rate: {comparison.pass_rate:.1%}")
            print(f"Columns only in raw: {comparison.columns_only_in_raw}")
        """
        with self._lock:
            if self._is_running:
                raise RuntimeError(
                    "Cannot compare profiles while validation is running. "
                    "Wait for validation to complete."
                )

            if not self.profile_raw and not self.profile_valid:
                raise ValueError(
                    "Cannot compare profiles when both profile_raw and profile_valid are disabled."
                )

            # Get column names from profile views
            raw_columns: set[str] = set()
            valid_columns: set[str] = set()

            if self._raw_profile is not None:
                raw_columns = set(self._raw_profile.get_columns().keys())

            if self._valid_profile is not None:
                valid_columns = set(self._valid_profile.get_columns().keys())

            # Compute column differences
            columns_only_in_raw = sorted(raw_columns - valid_columns)
            columns_only_in_valid = sorted(valid_columns - raw_columns)
            common_columns = sorted(raw_columns & valid_columns)

            # Compute pass rate
            pass_rate = 0.0
            if self._raw_row_count > 0:
                pass_rate = self._valid_row_count / self._raw_row_count

            return ProfileComparison(
                raw_column_count=len(raw_columns),
                valid_column_count=len(valid_columns),
                raw_row_count=self._raw_row_count,
                valid_row_count=self._valid_row_count,
                columns_only_in_raw=columns_only_in_raw,
                columns_only_in_valid=columns_only_in_valid,
                common_columns=common_columns,
                pass_rate=pass_rate,
            )

    def reset(self) -> None:
        """Reset observer state for reuse.

        Clears all buffered data and accumulated profiles. Call this
        before attaching the observer to a new validation run.

        Example:
            observer.reset()
            runner2.add_observer(observer)
            for result in runner2.run():
                process(result)
        """
        with self._lock:
            self._raw_buffer = []
            self._valid_buffer = []
            self._raw_profile = None
            self._valid_profile = None
            self._is_running = False
            self._raw_row_count = 0
            self._valid_row_count = 0

"""Output writers for validation results.

Provides writers for exporting failed records and audit reports
to CSV and JSON formats.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator

    from abstract_validation_base.runner import RowResult

__all__ = [
    "FailedRecordWriter",
    "CSVFailedWriter",
    "JSONLinesFailedWriter",
    "AuditReportWriter",
]


@runtime_checkable
class FailedRecordWriter(Protocol):
    """Protocol for writing failed validation records.

    Implement this protocol to create custom writers for failed records.
    Writers should handle streaming data efficiently.

    Example:
        class MyCustomWriter:
            def write_all(self, results: Iterator[RowResult]) -> int:
                count = 0
                for result in results:
                    self._write_record(result)
                    count += 1
                return count
    """

    def write_all(self, results: Iterator[RowResult[Any]]) -> int:
        """Write all failed records from an iterator.

        Args:
            results: Iterator of RowResult objects (typically from
                runner.run_collect_failed()).

        Returns:
            Number of records written.
        """
        ...


class CSVFailedWriter:
    """Write failed records to a CSV file.

    Outputs each failed record with its row index, raw data fields,
    and error information.

    Example:
        writer = CSVFailedWriter("failed_records.csv")
        count = writer.write_all(runner.run_collect_failed())
        print(f"Wrote {count} failed records")

        # Or use context manager for explicit control
        with CSVFailedWriter("failed.csv") as writer:
            for result in runner.run_collect_failed():
                writer.write_one(result)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        include_raw_data: bool = True,
        max_errors_per_row: int = 5,
    ) -> None:
        """Initialize the CSV writer.

        Args:
            path: Path to the output CSV file.
            include_raw_data: If True, include all raw data fields in output.
            max_errors_per_row: Maximum number of error columns per row.
        """
        self._path = Path(path)
        self._include_raw_data = include_raw_data
        self._max_errors = max_errors_per_row
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames: list[str] | None = None
        self._rows_written = 0

    def __enter__(self) -> CSVFailedWriter:
        """Open the file for writing."""
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        return self

    def __exit__(self, *args: object) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None
        self._writer = None
        self._fieldnames = None

    def write_one(self, result: RowResult[Any]) -> None:
        """Write a single failed record.

        Args:
            result: The RowResult to write.

        Note:
            The first call determines the CSV columns based on the
            raw data fields. Subsequent rows should have the same fields.
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Use 'with' statement or call __enter__")

        # Build the row data
        row: dict[str, Any] = {"_row_index": result.row_index}

        if self._include_raw_data:
            for key, value in result.raw_data.items():
                row[key] = value

        # Add error columns
        errors = result.error_summary
        for i, (field, msg) in enumerate(errors[: self._max_errors]):
            row[f"_error_{i + 1}_field"] = field
            row[f"_error_{i + 1}_message"] = msg

        row["_error_count"] = len(errors)

        # Initialize writer on first row
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._fieldnames,
                extrasaction="ignore",
            )
            self._writer.writeheader()

        self._writer.writerow(row)
        self._rows_written += 1

    def write_all(self, results: Iterator[RowResult[Any]]) -> int:
        """Write all failed records from an iterator.

        Args:
            results: Iterator of RowResult objects.

        Returns:
            Number of records written.
        """
        with self:
            for result in results:
                self.write_one(result)
        return self._rows_written


class JSONLinesFailedWriter:
    """Write failed records to a JSON Lines file.

    Each line is a complete JSON object containing the row index,
    raw data, and error information.

    Example:
        writer = JSONLinesFailedWriter("failed_records.jsonl")
        count = writer.write_all(runner.run_collect_failed())

        # Or with context manager
        with JSONLinesFailedWriter("failed.jsonl") as writer:
            for result in runner.run_collect_failed():
                writer.write_one(result)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        include_raw_data: bool = True,
        indent: int | None = None,
    ) -> None:
        """Initialize the JSON Lines writer.

        Args:
            path: Path to the output file.
            include_raw_data: If True, include all raw data fields.
            indent: JSON indentation (None for compact, int for pretty).
        """
        self._path = Path(path)
        self._include_raw_data = include_raw_data
        self._indent = indent
        self._file: TextIO | None = None
        self._rows_written = 0

    def __enter__(self) -> JSONLinesFailedWriter:
        """Open the file for writing."""
        self._file = open(self._path, "w", encoding="utf-8")
        return self

    def __exit__(self, *args: object) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None

    def write_one(self, result: RowResult[Any]) -> None:
        """Write a single failed record as a JSON line.

        Args:
            result: The RowResult to write.
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Use 'with' statement or call __enter__")

        record: dict[str, Any] = {
            "row_index": result.row_index,
            "errors": [{"field": field, "message": msg} for field, msg in result.error_summary],
        }

        if self._include_raw_data:
            record["raw_data"] = result.raw_data

        # Add Pydantic errors with full detail if present
        if result.pydantic_errors:
            record["pydantic_errors"] = result.pydantic_errors

        line = json.dumps(record, indent=self._indent, default=str)
        self._file.write(line + "\n")
        self._rows_written += 1

    def write_all(self, results: Iterator[RowResult[Any]]) -> int:
        """Write all failed records from an iterator.

        Args:
            results: Iterator of RowResult objects.

        Returns:
            Number of records written.
        """
        with self:
            for result in results:
                self.write_one(result)
        return self._rows_written


class AuditReportWriter:
    """Export audit reports to CSV or JSON files.

    For CSV, creates a file with two sections:
    1. Summary statistics (total rows, valid rows, error rates, timing)
    2. Top errors table (field, message, count, percentage)

    For JSON, creates a structured file with summary, top_errors,
    and optionally failed_samples keys.

    Example:
        # After running validation
        for result in runner.run():
            process(result)

        # Export as CSV (default for .csv extension)
        writer = AuditReportWriter("validation_audit.csv")
        writer.write(runner.audit_report())

        # Export as JSON (auto-detected from .json extension)
        writer = AuditReportWriter("validation_audit.json")
        writer.write(runner.audit_report())

        # Explicit format override
        writer = AuditReportWriter("audit_report.txt", format="json")
        writer.write(runner.audit_report())

        # CSV with separate error file
        writer = AuditReportWriter(
            "audit.csv",
            errors_path="top_errors.csv",
        )
        writer.write(runner.audit_report())
    """

    def __init__(
        self,
        path: str | Path,
        *,
        format: str = "auto",
        errors_path: str | Path | None = None,
        include_samples: bool = False,
        samples_path: str | Path | None = None,
        json_indent: int | None = 2,
    ) -> None:
        """Initialize the audit report writer.

        Args:
            path: Path for the main output file.
            format: Output format - "csv", "json", or "auto" (detect from extension).
                Defaults to "auto".
            errors_path: Optional separate path for top errors CSV.
                If None, errors are included in the main file.
                Only applicable for CSV format.
            include_samples: If True, include failed samples section.
            samples_path: Optional separate path for failed samples CSV.
                Only applicable for CSV format.
            json_indent: Indentation for JSON output. None for compact.
                Defaults to 2.
        """
        self._path = Path(path)
        self._format = self._detect_format(format)
        self._errors_path = Path(errors_path) if errors_path else None
        self._include_samples = include_samples
        self._samples_path = Path(samples_path) if samples_path else None
        self._json_indent = json_indent

    def _detect_format(self, format: str) -> str:
        """Detect output format from file extension or explicit parameter.

        Args:
            format: Explicit format or "auto" for extension detection.

        Returns:
            "csv" or "json".
        """
        if format != "auto":
            return format.lower()

        suffix = self._path.suffix.lower()
        if suffix == ".json":
            return "json"
        return "csv"

    def write(self, report: dict[str, Any]) -> None:
        """Write the audit report to file(s).

        Args:
            report: Audit report dict from ValidationRunner.audit_report().
        """
        if self._format == "json":
            self._write_json(report)
        else:
            self._write_csv(report)

    def _write_json(self, report: dict[str, Any]) -> None:
        """Write the audit report as structured JSON.

        Args:
            report: Audit report dict from ValidationRunner.audit_report().
        """
        output = {
            "summary": report.get("summary", {}),
            "top_errors": report.get("top_errors", []),
        }

        if self._include_samples:
            output["failed_samples"] = report.get("failed_samples", [])

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=self._json_indent, default=str)

    def _write_csv(self, report: dict[str, Any]) -> None:
        """Write the audit report as CSV file(s).

        Args:
            report: Audit report dict from ValidationRunner.audit_report().
        """
        self._write_summary(report)

        if self._errors_path:
            self._write_errors_separate(report)

        if self._include_samples and report.get("failed_samples"):
            self._write_samples(report)

    def _write_summary(self, report: dict[str, Any]) -> None:
        """Write summary section to the main file."""
        summary = report.get("summary", {})
        top_errors = report.get("top_errors", [])

        with open(self._path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Summary section header
            writer.writerow(["=== VALIDATION SUMMARY ==="])
            writer.writerow([])

            # Summary metrics
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Rows", summary.get("total_rows", 0)])
            writer.writerow(["Valid Rows", summary.get("valid_rows", 0)])
            writer.writerow(["Error Rows", summary.get("error_rows", 0)])
            writer.writerow(["Success Rate", summary.get("success_rate", "0.0%")])
            writer.writerow(["Pydantic Failures", summary.get("pydantic_failures", 0)])
            writer.writerow(["Validator Failures", summary.get("validator_failures", 0)])
            writer.writerow(["Duration (ms)", f"{summary.get('duration_ms', 0):.2f}"])
            writer.writerow([])

            # Top errors section (if not writing to separate file)
            if not self._errors_path and top_errors:
                writer.writerow(["=== TOP ERRORS ==="])
                writer.writerow([])
                writer.writerow(["Field", "Message", "Count", "Percentage"])
                for error in top_errors:
                    writer.writerow(
                        [
                            error.get("field", ""),
                            error.get("message", ""),
                            error.get("count", 0),
                            error.get("percentage", "0.0%"),
                        ]
                    )

    def _write_errors_separate(self, report: dict[str, Any]) -> None:
        """Write top errors to a separate CSV file."""
        top_errors = report.get("top_errors", [])

        if not self._errors_path:
            return

        with open(self._errors_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Field", "Message", "Count", "Percentage"])
            for error in top_errors:
                writer.writerow(
                    [
                        error.get("field", ""),
                        error.get("message", ""),
                        error.get("count", 0),
                        error.get("percentage", "0.0%"),
                    ]
                )

    def _write_samples(self, report: dict[str, Any]) -> None:
        """Write failed samples to CSV."""
        samples = report.get("failed_samples", [])
        if not samples:
            return

        # Determine output path
        output_path = self._samples_path or self._path.with_suffix(".samples.csv")

        # Get all unique keys from samples
        all_keys: set[str] = set()
        for sample in samples:
            all_keys.update(sample.keys())
        fieldnames = sorted(all_keys)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for sample in samples:
                writer.writerow(sample)

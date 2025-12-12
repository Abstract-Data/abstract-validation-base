"""Process logging for tracking cleaning operations and errors.

Provides Pydantic models for unified process tracking across all models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ProcessEntry(BaseModel):
    """Base entry for any process log record.

    Tracks individual cleaning operations or errors that occur during
    model processing.

    Attributes:
        entry_type: Type of entry - "cleaning" or "error".
        field: Name of the field this entry relates to.
        message: Description of what happened (reason for cleaning, error message).
        original_value: The original value before transformation (if applicable).
        new_value: The value after transformation (if applicable).
        timestamp: ISO format timestamp of when the operation occurred.
        context: Additional context dict for operation-specific metadata.
    """

    entry_type: Literal["cleaning", "error"]
    field: str
    message: str
    original_value: str | None = None
    new_value: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    context: dict[str, Any] = Field(default_factory=dict)


class ProcessLog(BaseModel):
    """Tracks cleaning operations and errors during model processing.

    Provides separate lists for cleaning operations and errors, allowing
    for easy filtering and aggregation.

    Attributes:
        cleaning: List of cleaning/transformation operations performed.
        errors: List of errors encountered during processing.
    """

    cleaning: list[ProcessEntry] = Field(default_factory=list)
    errors: list[ProcessEntry] = Field(default_factory=list)

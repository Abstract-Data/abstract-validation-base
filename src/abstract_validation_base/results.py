"""Validation result containers.

Generic validation result classes for tracking validation status and errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationError:
    """A single validation error."""

    field: str
    message: str
    value: str | None = None


@dataclass
class ValidationResult:
    """Result of validation with error aggregation."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    def add_error(self, field: str, message: str, value: str | None = None) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(field=field, message=message, value=value))
        self.is_valid = False

    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge another validation result into this one.

        This method mutates the current instance in-place by extending
        its errors list with errors from the other result.

        Args:
            other: Another ValidationResult to merge into this one.

        Returns:
            Self, for method chaining (e.g., result.merge(a).merge(b)).

        Example:
            combined = ValidationResult(is_valid=True)
            combined.merge(validator1.validate(item))
            combined.merge(validator2.validate(item))
            # Or chained:
            combined.merge(v1_result).merge(v2_result)
        """
        if not other.is_valid:
            self.is_valid = False
            self.errors.extend(other.errors)
        return self

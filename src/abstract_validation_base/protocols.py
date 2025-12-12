"""Validation protocols for type checking.

Generic validator protocol that can be used for type hints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from abstract_validation_base.results import ValidationResult

T = TypeVar("T", contravariant=True)


@runtime_checkable
class ValidatorProtocol(Protocol[T]):
    """Protocol for validation implementations.

    Use this for type hints when accepting any validator.
    Generic over T, the type of object being validated.
    """

    def validate(self, item: T) -> ValidationResult:
        """Validate an item."""
        ...

    @property
    def name(self) -> str:
        """Name of this validator."""
        ...

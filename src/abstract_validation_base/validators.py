"""Abstract base validators and composite validator.

Provides generic base classes for creating validators for any model type,
plus a composite validator for combining multiple validators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from abstract_validation_base.results import ValidationResult

__all__ = ["BaseValidator", "CompositeValidator"]

T = TypeVar("T")


class BaseValidator(ABC, Generic[T]):
    """Abstract base class for validators.

    Generic over T, the type of object being validated.
    Subclass this to create validators for specific model types.

    Example:
        from abstract_validation_base import BaseValidator, ValidationResult

        class UserValidator(BaseValidator[User]):
            @property
            def name(self) -> str:
                return "user_validator"

            def validate(self, item: User) -> ValidationResult:
                result = ValidationResult(is_valid=True)
                if not item.email:
                    result.add_error("email", "Email is required")
                if item.age and item.age < 0:
                    result.add_error("age", "Age cannot be negative", str(item.age))
                return result
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this validator for error reporting and identification."""
        ...

    @abstractmethod
    def validate(self, item: T) -> ValidationResult:
        """Validate an item.

        Args:
            item: Object to validate.

        Returns:
            ValidationResult containing validation status and any errors.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class CompositeValidator(BaseValidator[T], Generic[T]):
    """Validator that combines multiple validators.

    Runs all validators and aggregates their results. Useful for building
    validation pipelines where multiple independent checks need to run.

    Example:
        from abstract_validation_base import CompositeValidator

        composite = CompositeValidator([
            EmailValidator(),
            AgeValidator(),
            NameValidator(),
        ])
        result = composite.validate(user)
        # result contains errors from all validators that failed
    """

    def __init__(
        self,
        validators: list[BaseValidator[T]] | None = None,
        *,
        name: str = "composite",
        fail_fast: bool = False,
    ) -> None:
        """Initialize composite validator.

        Args:
            validators: List of validators to run. Defaults to empty list.
            name: Name for this composite validator. Defaults to "composite".
            fail_fast: If True, stop on first validation failure.
                Defaults to False (run all validators).
        """
        self._validators: list[BaseValidator[T]] = validators or []
        self._name = name
        self._fail_fast = fail_fast

    @property
    def name(self) -> str:
        """Name of this validator."""
        return self._name

    def validate(self, item: T) -> ValidationResult:
        """Run all validators and combine results.

        Args:
            item: Object to validate.

        Returns:
            ValidationResult with merged errors from all validators.
        """
        result = ValidationResult(is_valid=True)

        for validator in self._validators:
            validator_result = validator.validate(item)
            result.merge(validator_result)

            if self._fail_fast and not result.is_valid:
                break

        return result

    def add_validator(self, validator: BaseValidator[T]) -> None:
        """Add a validator to the composite.

        Args:
            validator: Validator to add.
        """
        self._validators.append(validator)

    def remove_validator(self, name: str) -> bool:
        """Remove a validator by name.

        Args:
            name: Name of the validator to remove.

        Returns:
            True if a validator was removed, False if not found.
        """
        for i, v in enumerate(self._validators):
            if v.name == name:
                self._validators.pop(i)
                return True
        return False

    def has_validator(self, name: str) -> bool:
        """Check if a validator with the given name exists.

        Args:
            name: Name of the validator to check for.

        Returns:
            True if found, False otherwise.
        """
        return any(v.name == name for v in self._validators)

    def get_validator(self, name: str) -> BaseValidator[T] | None:
        """Get a validator by name.

        Args:
            name: Name of the validator to retrieve.

        Returns:
            The validator if found, None otherwise.
        """
        for v in self._validators:
            if v.name == name:
                return v
        return None

    @property
    def validators(self) -> list[BaseValidator[T]]:
        """Get copy of validators list."""
        return self._validators.copy()

    @property
    def validator_names(self) -> list[str]:
        """Get list of validator names in order."""
        return [v.name for v in self._validators]

    def __len__(self) -> int:
        """Return number of validators in the composite."""
        return len(self._validators)

    def __repr__(self) -> str:
        names = ", ".join(self.validator_names)
        return f"CompositeValidator(name={self._name!r}, validators=[{names}])"

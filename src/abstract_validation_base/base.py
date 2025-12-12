"""Pydantic base model with built-in process logging.

Provides ValidationBase, a Pydantic base model that automatically tracks
cleaning operations and errors during data processing.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import PydanticCustomError

from abstract_validation_base.events import (
    ObservableMixin,
    ValidationEvent,
    ValidationEventType,
)
from abstract_validation_base.process_log import ProcessEntry, ProcessLog

__all__ = ["ValidationBase"]


class ValidationBase(ObservableMixin, BaseModel):
    """Base model with built-in process logging for cleaning and errors.

    All models inheriting from this class automatically get:
    - process_log: ProcessLog field (excluded from serialization)
    - add_error(): Log an error and optionally raise ValidationError
    - add_cleaning_process(): Log a cleaning/transformation operation
    - audit_log(): Export combined entries for DataFrame analysis
    - audit_log_recursive(): Export entries including nested models
    - has_errors: Property to check if any errors were logged
    - has_cleaning: Property to check if any cleaning operations occurred
    - Observer support: add_observer(), remove_observer(), notify()

    Example:
        from abstract_validation_base import ValidationBase

        class MyModel(ValidationBase):
            name: str
            value: int

        model = MyModel(name="test", value=42)
        model.add_cleaning_process("name", "  test  ", "test", "Trimmed whitespace")
        model.add_error("value", "Value seems low", 42)

        print(model.has_errors)  # True
        print(model.audit_log())

        # Observer support
        model.add_observer(my_observer)
        model.add_error("field", "error")  # Observer notified
    """

    model_config = ConfigDict(
        # Subclasses can override this
        extra="ignore",
    )

    process_log: ProcessLog = Field(default_factory=ProcessLog, exclude=True)

    @property
    def has_errors(self) -> bool:
        """Check if any errors have been logged."""
        return len(self.process_log.errors) > 0

    @property
    def has_cleaning(self) -> bool:
        """Check if any cleaning operations have been logged."""
        return len(self.process_log.cleaning) > 0

    @property
    def error_count(self) -> int:
        """Get the number of logged errors."""
        return len(self.process_log.errors)

    @property
    def cleaning_count(self) -> int:
        """Get the number of logged cleaning operations."""
        return len(self.process_log.cleaning)

    def _create_error(
        self,
        error_type: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> Exception:
        """Create an error to raise. Override in subclasses for custom error types.

        Default implementation returns PydanticCustomError.

        Args:
            error_type: Type/category of the error.
            message: Error message describing the issue.
            context: Additional context dict (optional).

        Returns:
            Exception instance to be raised.
        """
        return PydanticCustomError(
            error_type,
            message,
            context or {},
        )

    def add_error(
        self,
        field: str,
        message: str,
        value: Any = None,
        context: dict[str, Any] | None = None,
        raise_exception: bool = False,
    ) -> None:
        """Log an error and optionally raise an exception.

        Args:
            field: Name of the field with the error.
            message: Error message describing the issue.
            value: The problematic value (optional).
            context: Additional context dict (optional).
            raise_exception: If True, raise an exception after logging.

        Raises:
            Exception: If raise_exception is True. The exception type is
                determined by _create_error() which can be overridden.

        Note:
            This method emits a ValidationEventType.ERROR_ADDED event
            to all registered observers.
        """
        entry = ProcessEntry(
            entry_type="error",
            field=field,
            message=message,
            original_value=str(value) if value is not None else None,
            context=context or {},
        )
        self.process_log.errors.append(entry)

        # Notify observers of the error
        self.notify(
            ValidationEvent(
                event_type=ValidationEventType.ERROR_ADDED,
                source=self,
                data={
                    "field": field,
                    "message": message,
                    "value": value,
                    "context": context or {},
                },
            )
        )

        if raise_exception:
            raise self._create_error(
                error_type="validation_error",
                message=f"{field}: {message}",
                context={"field": field, "value": value, **(context or {})},
            )

    def add_cleaning_process(
        self,
        field: str,
        original_value: Any,
        new_value: Any,
        reason: str,
        operation_type: str = "cleaning",
    ) -> None:
        """Log a cleaning/transformation operation.

        Args:
            field: Name of the field that was cleaned.
            original_value: The original value before transformation.
            new_value: The value after transformation.
            reason: Explanation of why the cleaning was performed.
            operation_type: Category of operation (cleaning, normalization,
                formatting, expansion, etc.).

        Note:
            This method emits a ValidationEventType.CLEANING_ADDED event
            to all registered observers.
        """
        entry = ProcessEntry(
            entry_type="cleaning",
            field=field,
            message=reason,
            original_value=str(original_value) if original_value is not None else None,
            new_value=str(new_value) if new_value is not None else None,
            context={"operation_type": operation_type},
        )
        self.process_log.cleaning.append(entry)

        # Notify observers of the cleaning operation
        self.notify(
            ValidationEvent(
                event_type=ValidationEventType.CLEANING_ADDED,
                source=self,
                data={
                    "field": field,
                    "original_value": original_value,
                    "new_value": new_value,
                    "reason": reason,
                    "operation_type": operation_type,
                },
            )
        )

    def audit_log(self, source: str | None = None) -> list[dict[str, Any]]:
        """Export combined cleaning and error entries for DataFrame analysis.

        Args:
            source: Optional source identifier to add to each entry.

        Returns:
            List of dicts suitable for pd.DataFrame(), sorted by timestamp.
            Each entry includes entry_type, field, message, timestamps, etc.
        """
        entries: list[dict[str, Any]] = []
        for entry in self.process_log.cleaning:
            d = entry.model_dump()
            if source:
                d["source"] = source
            entries.append(d)
        for entry in self.process_log.errors:
            d = entry.model_dump()
            if source:
                d["source"] = source
            entries.append(d)
        return sorted(entries, key=lambda x: x.get("timestamp", ""))

    def clear_process_log(self) -> None:
        """Clear all logged entries. Useful for reprocessing."""
        self.process_log.cleaning.clear()
        self.process_log.errors.clear()

    def audit_log_recursive(self, source: str | None = None) -> list[dict[str, Any]]:
        """Export audit logs from this model and all nested ValidationBase models.

        Traverses nested models and aggregates their process logs with qualified
        field names (e.g., "address.city" for nested address model).

        Args:
            source: Optional source identifier prefix.

        Returns:
            Combined list of dicts from this model and all nested models,
            sorted by timestamp.

        Example:
            class Address(ValidationBase):
                city: str

            class Person(ValidationBase):
                name: str
                address: Address

            person = Person(name="John", address=Address(city="NYC"))
            person.address.add_error("city", "Invalid city")
            person.add_error("name", "Name too short")

            # Gets ALL entries with qualified names
            all_logs = person.audit_log_recursive(source="import_batch_1")
            # Returns entries with source like "import_batch_1.address"
        """
        entries = self.audit_log(source=source)

        for field_name in self.__class__.model_fields:
            if field_name == "process_log":
                continue
            value = getattr(self, field_name, None)

            # Handle single nested model
            if isinstance(value, ValidationBase):
                nested_source = f"{source}.{field_name}" if source else field_name
                entries.extend(value.audit_log_recursive(source=nested_source))

            # Handle list of nested models
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ValidationBase):
                        nested_source = (
                            f"{source}.{field_name}[{i}]" if source else f"{field_name}[{i}]"
                        )
                        entries.extend(item.audit_log_recursive(source=nested_source))

        return sorted(entries, key=lambda x: x.get("timestamp", ""))

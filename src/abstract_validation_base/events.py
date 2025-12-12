"""Observer pattern implementation for validation events.

Provides event types, observer protocol, and mixin for adding observer
support to validation classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ValidationEventType",
    "ValidationEvent",
    "ValidationObserver",
    "ObservableMixin",
]


class ValidationEventType(Enum):
    """Types of validation events that can be observed."""

    ERROR_ADDED = auto()
    """Emitted when an error is added to a model's process log."""

    CLEANING_ADDED = auto()
    """Emitted when a cleaning operation is added to a model's process log."""

    VALIDATION_STARTED = auto()
    """Emitted when validation begins on a model."""

    VALIDATION_COMPLETED = auto()
    """Emitted when validation completes on a model."""

    ROW_PROCESSED = auto()
    """Emitted when a single row is processed during batch validation."""

    BATCH_STARTED = auto()
    """Emitted when a batch of rows begins processing."""

    BATCH_COMPLETED = auto()
    """Emitted when a batch of rows finishes processing."""


@dataclass
class ValidationEvent:
    """A validation event that can be observed.

    Attributes:
        event_type: The type of event that occurred.
        source: The object that emitted the event (model or validator).
        data: Event-specific data dictionary.

    Example:
        event = ValidationEvent(
            event_type=ValidationEventType.ERROR_ADDED,
            source=my_model,
            data={"field": "email", "message": "Invalid format", "value": "bad"}
        )
    """

    event_type: ValidationEventType
    source: object
    data: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ValidationObserver(Protocol):
    """Protocol for validation event observers.

    Implement this protocol to receive validation events. Observers
    can be used for logging, metrics collection, alerting, etc.

    Example:
        class LoggingObserver:
            def on_event(self, event: ValidationEvent) -> None:
                print(f"{event.event_type.name}: {event.data}")

        class MetricsObserver:
            def on_event(self, event: ValidationEvent) -> None:
                if event.event_type == ValidationEventType.ERROR_ADDED:
                    metrics.increment("validation.errors")
    """

    def on_event(self, event: ValidationEvent) -> None:
        """Handle a validation event.

        Args:
            event: The validation event to handle.
        """
        ...


class ObservableMixin:
    """Mixin class to add observer support to any class.

    This mixin provides methods to add, remove, and notify observers
    of validation events. Classes that include this mixin can emit
    events that observers will receive.

    Example:
        class MyModel(ObservableMixin, BaseModel):
            def do_something(self):
                self.notify(ValidationEvent(
                    event_type=ValidationEventType.ERROR_ADDED,
                    source=self,
                    data={"field": "name", "message": "Invalid"}
                ))

        model = MyModel()
        model.add_observer(LoggingObserver())
    """

    _observers: list[ValidationObserver]

    def _ensure_observers(self) -> None:
        """Ensure the observers list is initialized."""
        if not hasattr(self, "_observers") or self._observers is None:
            self._observers = []

    def add_observer(self, observer: ValidationObserver) -> None:
        """Add an observer to receive validation events.

        Args:
            observer: An object implementing the ValidationObserver protocol.
        """
        self._ensure_observers()
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: ValidationObserver) -> None:
        """Remove an observer from receiving validation events.

        Args:
            observer: The observer to remove.
        """
        self._ensure_observers()
        if observer in self._observers:
            self._observers.remove(observer)

    def notify(self, event: ValidationEvent) -> None:
        """Notify all observers of a validation event.

        Args:
            event: The validation event to broadcast to observers.
        """
        self._ensure_observers()
        for observer in self._observers:
            observer.on_event(event)

    @property
    def observers(self) -> list[ValidationObserver]:
        """Get a copy of the current observers list."""
        self._ensure_observers()
        return self._observers.copy()

    def clear_observers(self) -> None:
        """Remove all observers."""
        self._ensure_observers()
        self._observers.clear()

"""Tests for observer pattern and events."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from abstract_validation_base import (
    ValidationBase,
    ValidationEvent,
    ValidationEventType,
    ValidationObserver,
)

# =============================================================================
# Test Models
# =============================================================================


class SimpleModel(ValidationBase):
    """Simple model for testing observer functionality."""

    name: str
    value: int = 0


# =============================================================================
# Test Observers
# =============================================================================


class RecordingObserver:
    """Observer that records all events for testing."""

    def __init__(self) -> None:
        self.events: list[ValidationEvent] = []

    def on_event(self, event: ValidationEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()

    @property
    def event_types(self) -> list[ValidationEventType]:
        return [e.event_type for e in self.events]


class CountingObserver:
    """Observer that counts events by type."""

    def __init__(self) -> None:
        self.counts: dict[ValidationEventType, int] = {}

    def on_event(self, event: ValidationEvent) -> None:
        if event.event_type not in self.counts:
            self.counts[event.event_type] = 0
        self.counts[event.event_type] += 1


# =============================================================================
# ValidationEventType Tests
# =============================================================================


class TestValidationEventType:
    """Tests for ValidationEventType enum."""

    def test_event_types_exist(self) -> None:
        """Test that all expected event types exist."""
        assert ValidationEventType.ERROR_ADDED is not None
        assert ValidationEventType.CLEANING_ADDED is not None
        assert ValidationEventType.VALIDATION_STARTED is not None
        assert ValidationEventType.VALIDATION_COMPLETED is not None

    def test_event_types_are_distinct(self) -> None:
        """Test that all event types have distinct values."""
        types = [
            ValidationEventType.ERROR_ADDED,
            ValidationEventType.CLEANING_ADDED,
            ValidationEventType.VALIDATION_STARTED,
            ValidationEventType.VALIDATION_COMPLETED,
        ]
        assert len(set(types)) == len(types)


# =============================================================================
# ValidationEvent Tests
# =============================================================================


class TestValidationEvent:
    """Tests for ValidationEvent dataclass."""

    def test_event_creation(self) -> None:
        """Test creating a validation event."""
        event = ValidationEvent(
            event_type=ValidationEventType.ERROR_ADDED,
            source="test_source",
            data={"field": "name", "message": "Invalid"},
        )

        assert event.event_type == ValidationEventType.ERROR_ADDED
        assert event.source == "test_source"
        assert event.data == {"field": "name", "message": "Invalid"}

    def test_event_default_data(self) -> None:
        """Test that data defaults to empty dict."""
        event = ValidationEvent(
            event_type=ValidationEventType.CLEANING_ADDED,
            source="test",
        )

        assert event.data == {}


# =============================================================================
# ValidationObserver Protocol Tests
# =============================================================================


class TestValidationObserverProtocol:
    """Tests for ValidationObserver protocol."""

    def test_recording_observer_is_protocol_instance(self) -> None:
        """Test RecordingObserver satisfies the protocol."""
        observer = RecordingObserver()
        assert isinstance(observer, ValidationObserver)

    def test_counting_observer_is_protocol_instance(self) -> None:
        """Test CountingObserver satisfies the protocol."""
        observer = CountingObserver()
        assert isinstance(observer, ValidationObserver)


# =============================================================================
# ObservableMixin Tests via ValidationBase
# =============================================================================


class TestObservableMixinViaValidationBase:
    """Tests for ObservableMixin functionality in ValidationBase."""

    def test_add_observer(self) -> None:
        """Test adding an observer."""
        model = SimpleModel(name="test")
        observer = RecordingObserver()

        model.add_observer(observer)

        assert observer in model.observers

    def test_add_observer_no_duplicates(self) -> None:
        """Test that same observer isn't added twice."""
        model = SimpleModel(name="test")
        observer = RecordingObserver()

        model.add_observer(observer)
        model.add_observer(observer)

        assert len(model.observers) == 1

    def test_remove_observer(self) -> None:
        """Test removing an observer."""
        model = SimpleModel(name="test")
        observer = RecordingObserver()

        model.add_observer(observer)
        model.remove_observer(observer)

        assert observer not in model.observers

    def test_remove_nonexistent_observer(self) -> None:
        """Test removing an observer that was never added."""
        model = SimpleModel(name="test")
        observer = RecordingObserver()

        # Should not raise
        model.remove_observer(observer)

    def test_clear_observers(self) -> None:
        """Test clearing all observers."""
        model = SimpleModel(name="test")
        observer1 = RecordingObserver()
        observer2 = RecordingObserver()

        model.add_observer(observer1)
        model.add_observer(observer2)
        model.clear_observers()

        assert len(model.observers) == 0

    def test_observers_property_returns_copy(self) -> None:
        """Test that observers property returns a copy."""
        model = SimpleModel(name="test")
        observer = RecordingObserver()
        model.add_observer(observer)

        observers = model.observers
        observers.clear()

        assert len(model.observers) == 1


# =============================================================================
# ValidationBase Event Emission Tests
# =============================================================================


class TestValidationBaseEventEmission:
    """Tests for event emission in ValidationBase."""

    def test_add_error_emits_event(self) -> None:
        """Test that add_error emits ERROR_ADDED event."""
        model = SimpleModel(name="test")
        observer = RecordingObserver()
        model.add_observer(observer)

        model.add_error("name", "Invalid name", value="test")

        assert len(observer.events) == 1
        event = observer.events[0]
        assert event.event_type == ValidationEventType.ERROR_ADDED
        assert event.source is model
        assert event.data["field"] == "name"
        assert event.data["message"] == "Invalid name"
        assert event.data["value"] == "test"

    def test_add_cleaning_emits_event(self) -> None:
        """Test that add_cleaning_process emits CLEANING_ADDED event."""
        model = SimpleModel(name="test")
        observer = RecordingObserver()
        model.add_observer(observer)

        model.add_cleaning_process("name", "  test  ", "test", "Trimmed whitespace")

        assert len(observer.events) == 1
        event = observer.events[0]
        assert event.event_type == ValidationEventType.CLEANING_ADDED
        assert event.source is model
        assert event.data["field"] == "name"
        assert event.data["original_value"] == "  test  "
        assert event.data["new_value"] == "test"
        assert event.data["reason"] == "Trimmed whitespace"

    def test_multiple_observers_receive_events(self) -> None:
        """Test that all observers receive events."""
        model = SimpleModel(name="test")
        observer1 = RecordingObserver()
        observer2 = CountingObserver()

        model.add_observer(observer1)
        model.add_observer(observer2)

        model.add_error("field", "error")

        assert len(observer1.events) == 1
        assert observer2.counts.get(ValidationEventType.ERROR_ADDED) == 1

    def test_no_observers_no_error(self) -> None:
        """Test that methods work without observers."""
        model = SimpleModel(name="test")

        # Should not raise
        model.add_error("field", "error")
        model.add_cleaning_process("field", "old", "new", "reason")

        assert model.error_count == 1
        assert model.cleaning_count == 1


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestObserverProperties:
    """Property-based tests for observer functionality."""

    @given(
        error_count=st.integers(min_value=0, max_value=10),
        cleaning_count=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=30)
    def test_event_count_matches_operations(self, error_count: int, cleaning_count: int) -> None:
        """Test that event count matches operation count."""
        model = SimpleModel(name="test")
        observer = CountingObserver()
        model.add_observer(observer)

        for i in range(error_count):
            model.add_error(f"field_{i}", f"error_{i}")

        for i in range(cleaning_count):
            model.add_cleaning_process(f"field_{i}", f"old_{i}", f"new_{i}", f"reason_{i}")

        assert observer.counts.get(ValidationEventType.ERROR_ADDED, 0) == error_count
        assert observer.counts.get(ValidationEventType.CLEANING_ADDED, 0) == cleaning_count

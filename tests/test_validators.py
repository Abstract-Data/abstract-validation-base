"""Tests for BaseValidator and CompositeValidator."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from abstract_validation_base import (
    ValidationEvent,
    ValidationEventType,
    ValidatorPipelineBuilder,
)
from abstract_validation_base.protocols import ValidatorProtocol
from abstract_validation_base.results import ValidationResult
from abstract_validation_base.validators import BaseValidator, CompositeValidator

from .conftest import (
    ConditionalValidator,
    FailingValidator,
    SampleModel,
    SimpleValidator,
)

# =============================================================================
# BaseValidator Unit Tests
# =============================================================================


class TestBaseValidatorUnit:
    """Unit tests for BaseValidator."""

    def test_base_validator_is_abstract(self) -> None:
        """Test that BaseValidator cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseValidator()  # type: ignore[abstract]

        assert "abstract" in str(exc_info.value).lower()

    def test_base_validator_subclass_works(self) -> None:
        """Test that a concrete subclass can be instantiated."""
        validator = SimpleValidator(name="test_validator")

        assert validator.name == "test_validator"

    def test_base_validator_validate_returns_result(self) -> None:
        """Test that validate returns a ValidationResult."""
        validator = SimpleValidator()
        model = SampleModel(name="test")

        result = validator.validate(model)

        assert isinstance(result, ValidationResult)

    def test_base_validator_repr(self) -> None:
        """Test __repr__ format."""
        validator = SimpleValidator(name="my_validator")

        repr_str = repr(validator)

        assert "SimpleValidator" in repr_str
        assert "my_validator" in repr_str

    def test_failing_validator_adds_error(self) -> None:
        """Test that FailingValidator adds an error."""
        validator = FailingValidator(
            name="fail_test", error_field="email", error_msg="Invalid email"
        )
        model = SampleModel(name="test")

        result = validator.validate(model)

        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].field == "email"
        assert result.errors[0].message == "Invalid email"

    def test_conditional_validator_passes(self) -> None:
        """Test ConditionalValidator passes for valid data."""
        validator = ConditionalValidator()
        model = SampleModel(name="test", value=10)

        result = validator.validate(model)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_conditional_validator_fails(self) -> None:
        """Test ConditionalValidator fails for invalid data."""
        validator = ConditionalValidator()
        model = SampleModel(name="test", value=-5)

        result = validator.validate(model)

        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].field == "value"


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Test that validators comply with ValidatorProtocol."""

    def test_simple_validator_is_protocol_instance(self) -> None:
        """Test SimpleValidator is instance of ValidatorProtocol."""
        validator = SimpleValidator()

        assert isinstance(validator, ValidatorProtocol)

    def test_failing_validator_is_protocol_instance(self) -> None:
        """Test FailingValidator is instance of ValidatorProtocol."""
        validator = FailingValidator()

        assert isinstance(validator, ValidatorProtocol)

    def test_composite_validator_is_protocol_instance(self) -> None:
        """Test CompositeValidator is instance of ValidatorProtocol."""
        validator = CompositeValidator[SampleModel]()

        assert isinstance(validator, ValidatorProtocol)


# =============================================================================
# CompositeValidator Unit Tests
# =============================================================================


class TestCompositeValidatorUnit:
    """Unit tests for CompositeValidator."""

    def test_composite_validator_empty(self) -> None:
        """Test empty composite validator returns valid."""
        composite = CompositeValidator[SampleModel]()
        model = SampleModel(name="test")

        result = composite.validate(model)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_composite_validator_default_name(self) -> None:
        """Test default name is 'composite'."""
        composite = CompositeValidator[SampleModel]()

        assert composite.name == "composite"

    def test_composite_validator_custom_name(self) -> None:
        """Test custom name parameter."""
        composite = CompositeValidator[SampleModel](name="my_composite")

        assert composite.name == "my_composite"

    def test_composite_validator_with_validators(self) -> None:
        """Test composite with multiple validators."""
        composite = CompositeValidator[SampleModel](
            validators=[
                SimpleValidator("v1"),
                SimpleValidator("v2"),
            ]
        )

        assert len(composite) == 2

    def test_composite_validator_all_pass(self) -> None:
        """Test composite where all validators pass."""
        composite = CompositeValidator[SampleModel](
            validators=[
                SimpleValidator("v1"),
                SimpleValidator("v2"),
                ConditionalValidator(),
            ]
        )
        model = SampleModel(name="test", value=10)

        result = composite.validate(model)

        assert result.is_valid

    def test_composite_validator_one_fails(self) -> None:
        """Test composite where one validator fails."""
        composite = CompositeValidator[SampleModel](
            validators=[
                SimpleValidator("v1"),
                FailingValidator("fail"),
                SimpleValidator("v2"),
            ]
        )
        model = SampleModel(name="test")

        result = composite.validate(model)

        assert not result.is_valid
        assert len(result.errors) == 1

    def test_composite_validator_multiple_fail(self) -> None:
        """Test composite where multiple validators fail."""
        composite = CompositeValidator[SampleModel](
            validators=[
                FailingValidator("fail1", "field1", "error1"),
                FailingValidator("fail2", "field2", "error2"),
            ]
        )
        model = SampleModel(name="test")

        result = composite.validate(model)

        assert not result.is_valid
        assert len(result.errors) == 2

    def test_composite_validator_fail_fast_false(self) -> None:
        """Test fail_fast=False continues after failure."""
        composite = CompositeValidator[SampleModel](
            validators=[
                FailingValidator("fail1", "f1", "e1"),
                FailingValidator("fail2", "f2", "e2"),
            ],
            fail_fast=False,
        )
        model = SampleModel(name="test")

        result = composite.validate(model)

        # Should have errors from both validators
        assert len(result.errors) == 2

    def test_composite_validator_fail_fast_true(self) -> None:
        """Test fail_fast=True stops on first failure."""
        composite = CompositeValidator[SampleModel](
            validators=[
                FailingValidator("fail1", "f1", "e1"),
                FailingValidator("fail2", "f2", "e2"),
            ],
            fail_fast=True,
        )
        model = SampleModel(name="test")

        result = composite.validate(model)

        # Should only have error from first validator
        assert len(result.errors) == 1
        assert result.errors[0].field == "f1"


class TestCompositeValidatorManagement:
    """Tests for CompositeValidator add/remove/query methods."""

    def test_add_validator(self) -> None:
        """Test add_validator adds to the list."""
        composite = CompositeValidator[SampleModel]()

        composite.add_validator(SimpleValidator("v1"))

        assert len(composite) == 1
        assert composite.has_validator("v1")

    def test_add_multiple_validators(self) -> None:
        """Test adding multiple validators."""
        composite = CompositeValidator[SampleModel]()

        composite.add_validator(SimpleValidator("v1"))
        composite.add_validator(SimpleValidator("v2"))
        composite.add_validator(SimpleValidator("v3"))

        assert len(composite) == 3

    def test_remove_validator_found(self) -> None:
        """Test remove_validator returns True when found."""
        composite = CompositeValidator[SampleModel](
            validators=[SimpleValidator("v1"), SimpleValidator("v2")]
        )

        removed = composite.remove_validator("v1")

        assert removed is True
        assert len(composite) == 1
        assert not composite.has_validator("v1")
        assert composite.has_validator("v2")

    def test_remove_validator_not_found(self) -> None:
        """Test remove_validator returns False when not found."""
        composite = CompositeValidator[SampleModel](validators=[SimpleValidator("v1")])

        removed = composite.remove_validator("nonexistent")

        assert removed is False
        assert len(composite) == 1

    def test_has_validator_true(self) -> None:
        """Test has_validator returns True when exists."""
        composite = CompositeValidator[SampleModel](validators=[SimpleValidator("exists")])

        assert composite.has_validator("exists") is True

    def test_has_validator_false(self) -> None:
        """Test has_validator returns False when not exists."""
        composite = CompositeValidator[SampleModel]()

        assert composite.has_validator("nonexistent") is False

    def test_get_validator_found(self) -> None:
        """Test get_validator returns validator when found."""
        v1 = SimpleValidator("v1")
        composite = CompositeValidator[SampleModel](validators=[v1])

        found = composite.get_validator("v1")

        assert found is v1

    def test_get_validator_not_found(self) -> None:
        """Test get_validator returns None when not found."""
        composite = CompositeValidator[SampleModel]()

        found = composite.get_validator("nonexistent")

        assert found is None

    def test_validators_property_returns_copy(self) -> None:
        """Test validators property returns a copy."""
        composite = CompositeValidator[SampleModel](validators=[SimpleValidator("v1")])

        validators = composite.validators

        # Modifying returned list shouldn't affect composite
        validators.append(SimpleValidator("v2"))

        assert len(composite) == 1

    def test_validator_names_property(self) -> None:
        """Test validator_names returns list of names."""
        composite = CompositeValidator[SampleModel](
            validators=[
                SimpleValidator("alpha"),
                SimpleValidator("beta"),
                SimpleValidator("gamma"),
            ]
        )

        names = composite.validator_names

        assert names == ["alpha", "beta", "gamma"]

    def test_len(self) -> None:
        """Test __len__ returns correct count."""
        composite = CompositeValidator[SampleModel](
            validators=[SimpleValidator(f"v{i}") for i in range(5)]
        )

        assert len(composite) == 5

    def test_repr(self) -> None:
        """Test __repr__ format."""
        composite = CompositeValidator[SampleModel](
            name="test_composite",
            validators=[SimpleValidator("v1"), SimpleValidator("v2")],
        )

        repr_str = repr(composite)

        assert "CompositeValidator" in repr_str
        assert "test_composite" in repr_str
        assert "v1" in repr_str
        assert "v2" in repr_str


# =============================================================================
# CompositeValidator Property-Based Tests
# =============================================================================


class TestCompositeValidatorProperties:
    """Property-based tests for CompositeValidator."""

    @given(data=st.data())
    @settings(max_examples=50)
    def test_composite_empty_always_valid(self, data: st.DataObject) -> None:
        """Empty composite returns valid for any input."""
        composite = CompositeValidator[SampleModel]()

        name = data.draw(st.text(min_size=1, max_size=20))
        value = data.draw(st.integers())
        model = SampleModel(name=name, value=value)

        result = composite.validate(model)

        assert result.is_valid

    @given(
        count=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50)
    def test_composite_validator_count(self, count: int) -> None:
        """len(composite) == len(validators list)."""
        validators: list[BaseValidator[SampleModel]] = [
            SimpleValidator(f"v{i}") for i in range(count)
        ]
        composite = CompositeValidator[SampleModel](validators=validators)

        assert len(composite) == count

    @given(
        names=st.lists(
            st.text(
                min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
            ),
            min_size=1,
            max_size=10,
            unique=True,
        )
    )
    @settings(max_examples=50)
    def test_validator_names_match(self, names: list[str]) -> None:
        """validator_names matches actual validator.name values."""
        validators: list[BaseValidator[SampleModel]] = [SimpleValidator(name) for name in names]
        composite = CompositeValidator[SampleModel](validators=validators)

        assert composite.validator_names == names

    @given(
        fail_count=st.integers(min_value=0, max_value=5),
        pass_count=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50)
    def test_composite_error_count(self, fail_count: int, pass_count: int) -> None:
        """Total errors equals sum of failing validators."""
        validators: list[BaseValidator[SampleModel]] = []

        for i in range(fail_count):
            validators.append(FailingValidator(f"fail_{i}", f"field_{i}", f"error_{i}"))

        for i in range(pass_count):
            validators.append(SimpleValidator(f"pass_{i}"))

        composite = CompositeValidator[SampleModel](validators=validators)
        model = SampleModel(name="test")

        result = composite.validate(model)

        assert len(result.errors) == fail_count
        if fail_count > 0:
            assert not result.is_valid
        else:
            assert result.is_valid


# =============================================================================
# CompositeValidator Stateful Test
# =============================================================================


class CompositeValidatorStateMachine(RuleBasedStateMachine):
    """Stateful test for CompositeValidator add/remove operations."""

    def __init__(self) -> None:
        super().__init__()
        self.composite: CompositeValidator[SampleModel] = CompositeValidator()
        self.added_names: list[str] = []  # Track all added names (allows duplicates)

    @rule(
        name=st.text(
            min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
        ),
    )
    def add_validator(self, name: str) -> None:
        """Add a validator to the composite."""
        validator = SimpleValidator(name)
        self.composite.add_validator(validator)
        self.added_names.append(name)

        assert self.composite.has_validator(name)

    @rule(
        name=st.text(
            min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))
        ),
    )
    def remove_validator(self, name: str) -> None:
        """Remove a validator by name."""
        removed = self.composite.remove_validator(name)

        if name in self.added_names:
            assert removed
            # Remove first occurrence from tracking list
            self.added_names.remove(name)
        else:
            assert not removed

    @rule()
    def validate_model(self) -> None:
        """Validate a sample model."""
        model = SampleModel(name="test", value=1)
        result = self.composite.validate(model)

        # SimpleValidator always passes, so should be valid
        assert result.is_valid

    @invariant()
    def length_matches(self) -> None:
        """Length of composite matches tracked additions."""
        assert len(self.composite) == len(self.added_names)

    @invariant()
    def all_names_present(self) -> None:
        """All tracked names are present in composite."""
        for name in set(self.added_names):
            assert self.composite.has_validator(name)

    @invariant()
    def validator_names_consistent(self) -> None:
        """validator_names property matches tracked names."""
        assert self.composite.validator_names == self.added_names


# Create the test class for pytest
TestCompositeValidatorStateful = CompositeValidatorStateMachine.TestCase


# =============================================================================
# Additional Integration Tests
# =============================================================================


class TestValidatorIntegration:
    """Integration tests for validators."""

    def test_nested_composite_validators(self) -> None:
        """Test that composite validators can be nested."""
        inner = CompositeValidator[SampleModel](
            name="inner",
            validators=[SimpleValidator("inner_v1"), SimpleValidator("inner_v2")],
        )

        # CompositeValidator is a BaseValidator, so it can be added to another
        outer = CompositeValidator[SampleModel](
            name="outer",
            validators=[inner, SimpleValidator("outer_v1")],
        )

        model = SampleModel(name="test")
        result = outer.validate(model)

        assert result.is_valid
        assert len(outer) == 2

    def test_validator_order_preserved(self) -> None:
        """Test that validators run in order they were added."""
        call_order: list[str] = []

        class OrderTrackingValidator(BaseValidator[SampleModel]):
            def __init__(self, name: str, tracker: list[str]) -> None:
                self._name = name
                self._tracker = tracker

            @property
            def name(self) -> str:
                return self._name

            def validate(self, item: SampleModel) -> ValidationResult:
                self._tracker.append(self._name)
                return ValidationResult(is_valid=True)

        composite = CompositeValidator[SampleModel](
            validators=[
                OrderTrackingValidator("first", call_order),
                OrderTrackingValidator("second", call_order),
                OrderTrackingValidator("third", call_order),
            ]
        )

        model = SampleModel(name="test")
        composite.validate(model)

        assert call_order == ["first", "second", "third"]

    def test_fail_fast_stops_at_correct_validator(self) -> None:
        """Test fail_fast stops exactly at the failing validator."""
        call_order: list[str] = []

        class TrackingFailValidator(BaseValidator[SampleModel]):
            def __init__(self, name: str, tracker: list[str], should_fail: bool) -> None:
                self._name = name
                self._tracker = tracker
                self._should_fail = should_fail

            @property
            def name(self) -> str:
                return self._name

            def validate(self, item: SampleModel) -> ValidationResult:
                self._tracker.append(self._name)
                result = ValidationResult(is_valid=True)
                if self._should_fail:
                    result.add_error("test", "failed")
                return result

        composite = CompositeValidator[SampleModel](
            validators=[
                TrackingFailValidator("v1", call_order, False),
                TrackingFailValidator("v2", call_order, True),  # This one fails
                TrackingFailValidator("v3", call_order, False),  # Should not run
            ],
            fail_fast=True,
        )

        model = SampleModel(name="test")
        result = composite.validate(model)

        assert call_order == ["v1", "v2"]  # v3 should not have run
        assert not result.is_valid
        assert len(result.errors) == 1


# =============================================================================
# ValidatorPipelineBuilder Tests
# =============================================================================


class TestValidatorPipelineBuilder:
    """Tests for ValidatorPipelineBuilder."""

    def test_build_empty_pipeline(self) -> None:
        """Test building an empty pipeline."""
        pipeline = ValidatorPipelineBuilder[SampleModel]("empty").build()

        assert pipeline.name == "empty"
        assert len(pipeline) == 0

    def test_build_with_validators(self) -> None:
        """Test building a pipeline with validators."""
        pipeline = (
            ValidatorPipelineBuilder[SampleModel]("test_pipeline")
            .add(SimpleValidator("v1"))
            .add(SimpleValidator("v2"))
            .build()
        )

        assert pipeline.name == "test_pipeline"
        assert len(pipeline) == 2
        assert pipeline.validator_names == ["v1", "v2"]

    def test_fluent_interface(self) -> None:
        """Test that all methods return self for chaining."""
        builder = ValidatorPipelineBuilder[SampleModel]("test")

        result = builder.add(SimpleValidator("v1"))
        assert result is builder

        result = builder.fail_fast()
        assert result is builder

        result = builder.with_name("new_name")
        assert result is builder

    def test_fail_fast_setting(self) -> None:
        """Test fail_fast setting is applied."""
        pipeline = (
            ValidatorPipelineBuilder[SampleModel]("test")
            .add(FailingValidator("fail1", "f1", "e1"))
            .add(FailingValidator("fail2", "f2", "e2"))
            .fail_fast()
            .build()
        )

        model = SampleModel(name="test")
        result = pipeline.validate(model)

        # With fail_fast, only first error should be present
        assert len(result.errors) == 1

    def test_fail_fast_disabled(self) -> None:
        """Test fail_fast can be explicitly disabled."""
        pipeline = (
            ValidatorPipelineBuilder[SampleModel]("test")
            .add(FailingValidator("fail1", "f1", "e1"))
            .add(FailingValidator("fail2", "f2", "e2"))
            .fail_fast(False)
            .build()
        )

        model = SampleModel(name="test")
        result = pipeline.validate(model)

        # Without fail_fast, both errors should be present
        assert len(result.errors) == 2

    def test_with_name(self) -> None:
        """Test changing name after initialization."""
        pipeline = ValidatorPipelineBuilder[SampleModel]("initial").with_name("final").build()

        assert pipeline.name == "final"

    def test_validators_copied(self) -> None:
        """Test that validators list is copied, not shared."""
        builder = ValidatorPipelineBuilder[SampleModel]("test")
        builder.add(SimpleValidator("v1"))

        pipeline1 = builder.build()
        builder.add(SimpleValidator("v2"))
        pipeline2 = builder.build()

        assert len(pipeline1) == 1
        assert len(pipeline2) == 2

    def test_repr(self) -> None:
        """Test __repr__ output."""
        builder = (
            ValidatorPipelineBuilder[SampleModel]("my_pipeline")
            .add(SimpleValidator("v1"))
            .fail_fast()
        )

        repr_str = repr(builder)

        assert "ValidatorPipelineBuilder" in repr_str
        assert "my_pipeline" in repr_str
        assert "1" in repr_str  # validator count
        assert "True" in repr_str  # fail_fast


# =============================================================================
# CompositeValidator Observer Tests
# =============================================================================


class RecordingObserver:
    """Observer that records all events."""

    def __init__(self) -> None:
        self.events: list[ValidationEvent] = []

    def on_event(self, event: ValidationEvent) -> None:
        self.events.append(event)


class TestCompositeValidatorObserver:
    """Tests for CompositeValidator observer functionality."""

    def test_emits_validation_started_event(self) -> None:
        """Test that VALIDATION_STARTED event is emitted."""
        composite = CompositeValidator[SampleModel](
            validators=[SimpleValidator("v1")],
            name="test_composite",
        )
        observer = RecordingObserver()
        composite.add_observer(observer)

        model = SampleModel(name="test")
        composite.validate(model)

        started_events = [
            e for e in observer.events if e.event_type == ValidationEventType.VALIDATION_STARTED
        ]
        assert len(started_events) == 1

        event = started_events[0]
        assert event.source is composite
        assert event.data["validator_name"] == "test_composite"
        assert event.data["validator_count"] == 1

    def test_emits_validation_completed_event(self) -> None:
        """Test that VALIDATION_COMPLETED event is emitted."""
        composite = CompositeValidator[SampleModel](
            validators=[SimpleValidator("v1")],
            name="test_composite",
        )
        observer = RecordingObserver()
        composite.add_observer(observer)

        model = SampleModel(name="test")
        composite.validate(model)

        completed_events = [
            e for e in observer.events if e.event_type == ValidationEventType.VALIDATION_COMPLETED
        ]
        assert len(completed_events) == 1

        event = completed_events[0]
        assert event.source is composite
        assert event.data["validator_name"] == "test_composite"
        assert event.data["is_valid"] is True
        assert event.data["error_count"] == 0
        assert "duration_ms" in event.data

    def test_completed_event_reflects_failure(self) -> None:
        """Test that VALIDATION_COMPLETED reflects validation failure."""
        composite = CompositeValidator[SampleModel](
            validators=[FailingValidator("fail", "field", "error")],
        )
        observer = RecordingObserver()
        composite.add_observer(observer)

        model = SampleModel(name="test")
        composite.validate(model)

        completed_events = [
            e for e in observer.events if e.event_type == ValidationEventType.VALIDATION_COMPLETED
        ]

        event = completed_events[0]
        assert event.data["is_valid"] is False
        assert event.data["error_count"] == 1

    def test_events_in_correct_order(self) -> None:
        """Test that events are emitted in correct order."""
        composite = CompositeValidator[SampleModel](
            validators=[SimpleValidator("v1")],
        )
        observer = RecordingObserver()
        composite.add_observer(observer)

        model = SampleModel(name="test")
        composite.validate(model)

        assert len(observer.events) == 2
        assert observer.events[0].event_type == ValidationEventType.VALIDATION_STARTED
        assert observer.events[1].event_type == ValidationEventType.VALIDATION_COMPLETED

    def test_duration_measured(self) -> None:
        """Test that duration is measured in completed event."""
        composite = CompositeValidator[SampleModel](
            validators=[SimpleValidator("v1")],
        )
        observer = RecordingObserver()
        composite.add_observer(observer)

        model = SampleModel(name="test")
        composite.validate(model)

        completed_event = observer.events[1]
        assert completed_event.data["duration_ms"] >= 0

    def test_no_observers_no_error(self) -> None:
        """Test that validation works without observers."""
        composite = CompositeValidator[SampleModel](
            validators=[SimpleValidator("v1")],
        )

        model = SampleModel(name="test")
        result = composite.validate(model)

        assert result.is_valid

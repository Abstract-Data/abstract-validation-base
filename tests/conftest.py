"""Shared fixtures and Hypothesis strategies for tests."""

from __future__ import annotations

import pytest
from hypothesis import strategies as st

from abstract_validation_base import ValidationBase
from abstract_validation_base.results import ValidationResult
from abstract_validation_base.validators import BaseValidator

# -----------------------------------------------------------------------------
# Hypothesis Strategies
# -----------------------------------------------------------------------------

# Strategy for valid field names (letters and numbers only)
field_names = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=("L", "N")),
)

# Strategy for messages
messages = st.text(min_size=1, max_size=200)

# Strategy for entry types
entry_types = st.sampled_from(["cleaning", "error"])

# Strategy for optional string values
optional_strings = st.one_of(st.none(), st.text(max_size=100))

# Strategy for context dicts
context_dicts = st.dictionaries(
    keys=st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(
            whitelist_categories=("L",)  # type: ignore[arg-type]
        ),
    ),
    values=st.one_of(st.integers(), st.text(max_size=50), st.booleans()),
    max_size=5,
)


# -----------------------------------------------------------------------------
# Test Model Classes
# -----------------------------------------------------------------------------


class SampleModel(ValidationBase):
    """Simple model for testing ValidationBase functionality."""

    name: str
    value: int = 0


class AnotherModel(ValidationBase):
    """Another test model with different fields."""

    title: str
    count: int = 0
    active: bool = True


# -----------------------------------------------------------------------------
# Test Validator Classes
# -----------------------------------------------------------------------------


class SimpleValidator(BaseValidator[SampleModel]):
    """Simple validator that always passes."""

    def __init__(self, name: str = "simple") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def validate(self, item: SampleModel) -> ValidationResult:
        return ValidationResult(is_valid=True)


class FailingValidator(BaseValidator[SampleModel]):
    """Validator that always fails with a configurable error."""

    def __init__(
        self, name: str = "failing", error_field: str = "test", error_msg: str = "Failed"
    ) -> None:
        self._name = name
        self._error_field = error_field
        self._error_msg = error_msg

    @property
    def name(self) -> str:
        return self._name

    def validate(self, item: SampleModel) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        result.add_error(self._error_field, self._error_msg)
        return result


class ConditionalValidator(BaseValidator[SampleModel]):
    """Validator that fails if value is negative."""

    def __init__(self, name: str = "conditional") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def validate(self, item: SampleModel) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        if item.value < 0:
            result.add_error("value", "Value cannot be negative", str(item.value))
        return result


# -----------------------------------------------------------------------------
# Pytest Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_model() -> SampleModel:
    """Create a fresh SampleModel instance."""
    return SampleModel(name="test")


@pytest.fixture
def validation_result() -> ValidationResult:
    """Create a fresh valid ValidationResult."""
    return ValidationResult(is_valid=True)


@pytest.fixture
def simple_validator() -> SimpleValidator:
    """Create a SimpleValidator instance."""
    return SimpleValidator()


@pytest.fixture
def failing_validator() -> FailingValidator:
    """Create a FailingValidator instance."""
    return FailingValidator()


@pytest.fixture
def conditional_validator() -> ConditionalValidator:
    """Create a ConditionalValidator instance."""
    return ConditionalValidator()

# abstract-validation-base

[![Tests](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/tests.yml/badge.svg)](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/tests.yml)
[![Lint](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/lint.yml/badge.svg)](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/lint.yml)
[![Version Check](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/version-check.yml/badge.svg)](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/version-check.yml)
[![codecov](https://codecov.io/gh/Abstract-Data/abstract-validation-base/graph/badge.svg)](https://codecov.io/gh/Abstract-Data/abstract-validation-base)
[![PyPI version](https://img.shields.io/pypi/v/abstract-validation-base.svg)](https://pypi.org/project/abstract-validation-base/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Validation and process tracking for data transformation pipelines.

## Features

- **ValidationBase**: Pydantic base model with built-in process logging for tracking cleaning operations and errors
- **ProcessLog**: Unified process tracking with separate lists for cleaning operations and errors
- **ValidationResult**: Generic validation result containers with error aggregation
- **BaseValidator**: Abstract base class for creating type-safe validators
- **CompositeValidator**: Combine multiple validators into validation pipelines

## Installation

```bash
pip install abstract-validation-base
```

Or with uv:

```bash
uv add abstract-validation-base
```

## Quick Start

### Using ValidationBase

```python
from abstract_validation_base import ValidationBase

class MyModel(ValidationBase):
    name: str
    value: int

# Create a model instance
model = MyModel(name="test", value=42)

# Log cleaning operations
model.add_cleaning_process("name", "  test  ", "test", "Trimmed whitespace")

# Log errors (without raising)
model.add_error("value", "Value seems low", value=42)

# Check status
print(model.has_errors)      # True
print(model.has_cleaning)    # True
print(model.error_count)     # 1
print(model.cleaning_count)  # 1

# Export for analysis
audit_data = model.audit_log(source="my_pipeline")
```

### Using Validators

```python
from abstract_validation_base import BaseValidator, ValidationResult, CompositeValidator

class AgeValidator(BaseValidator[User]):
    @property
    def name(self) -> str:
        return "age_validator"

    def validate(self, item: User) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        if item.age < 0:
            result.add_error("age", "Age cannot be negative", str(item.age))
        return result

class EmailValidator(BaseValidator[User]):
    @property
    def name(self) -> str:
        return "email_validator"

    def validate(self, item: User) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        if "@" not in item.email:
            result.add_error("email", "Invalid email format", item.email)
        return result

# Combine validators
composite = CompositeValidator[User](
    validators=[AgeValidator(), EmailValidator()],
    fail_fast=False,  # Run all validators even if one fails
)

result = composite.validate(user)
if not result.is_valid:
    for error in result.errors:
        print(f"{error.field}: {error.message}")
```

## API Reference

### ValidationBase

A Pydantic base model with built-in process logging.

| Method/Property | Description |
|-----------------|-------------|
| `has_errors` | Check if any errors have been logged |
| `has_cleaning` | Check if any cleaning operations have been logged |
| `error_count` | Get the number of logged errors |
| `cleaning_count` | Get the number of logged cleaning operations |
| `add_error(field, message, value?, context?, raise_exception?)` | Log an error |
| `add_cleaning_process(field, original, new, reason, operation_type?)` | Log a cleaning operation |
| `audit_log(source?)` | Export entries for DataFrame analysis |
| `clear_process_log()` | Clear all logged entries |

### CompositeValidator

Combine multiple validators into a pipeline.

| Method/Property | Description |
|-----------------|-------------|
| `validate(item)` | Run all validators and combine results |
| `add_validator(validator)` | Add a validator to the composite |
| `remove_validator(name)` | Remove a validator by name |
| `has_validator(name)` | Check if a validator exists |
| `get_validator(name)` | Get a validator by name |
| `validators` | Get copy of validators list |
| `validator_names` | Get list of validator names |

## Development

```bash
# Clone the repository
git clone https://github.com/Abstract-Data/abstract-validation-base.git
cd abstract-validation-base

# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linters
uv run ruff check src tests
uv run mypy src
```

## License

MIT License


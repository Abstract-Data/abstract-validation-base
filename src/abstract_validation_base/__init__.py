"""Validation and process tracking for data transformation pipelines."""

from abstract_validation_base.base import ValidationBase
from abstract_validation_base.process_log import ProcessEntry, ProcessLog
from abstract_validation_base.protocols import ValidatorProtocol
from abstract_validation_base.results import ValidationError, ValidationResult
from abstract_validation_base.sqlmodel_support import ValidatedRecord
from abstract_validation_base.validators import BaseValidator, CompositeValidator

__all__ = [
    # Process tracking
    "ProcessEntry",
    "ProcessLog",
    # Pydantic base
    "ValidationBase",
    # SQLModel integration
    "ValidatedRecord",
    # Validation results
    "ValidationResult",
    "ValidationError",
    # Validator abstractions
    "BaseValidator",
    "CompositeValidator",
    "ValidatorProtocol",
]

__version__ = "0.2.0a1"

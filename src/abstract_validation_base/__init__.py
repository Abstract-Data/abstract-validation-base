"""Validation and process tracking for data transformation pipelines."""

from abstract_validation_base.base import ValidationBase
from abstract_validation_base.events import (
    ObservableMixin,
    ValidationEvent,
    ValidationEventType,
    ValidationObserver,
)
from abstract_validation_base.process_log import ProcessEntry, ProcessLog
from abstract_validation_base.protocols import ValidatorProtocol
from abstract_validation_base.results import ValidationError, ValidationResult
from abstract_validation_base.rich_observers import (
    RichDashboardObserver,
    SimpleProgressObserver,
)
from abstract_validation_base.runner import RowResult, RunnerStats, ValidationRunner
from abstract_validation_base.sqlmodel_support import ValidatedRecord
from abstract_validation_base.validators import (
    BaseValidator,
    CompositeValidator,
    ValidatorPipelineBuilder,
)
from abstract_validation_base.writers import (
    AuditReportWriter,
    CSVFailedWriter,
    FailedRecordWriter,
    JSONLinesFailedWriter,
)

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
    "ValidatorPipelineBuilder",
    "ValidatorProtocol",
    # Observer pattern
    "ObservableMixin",
    "ValidationEvent",
    "ValidationEventType",
    "ValidationObserver",
    # Streaming runner
    "RowResult",
    "RunnerStats",
    "ValidationRunner",
    # Output writers
    "AuditReportWriter",
    "CSVFailedWriter",
    "FailedRecordWriter",
    "JSONLinesFailedWriter",
    # Rich observers
    "RichDashboardObserver",
    "SimpleProgressObserver",
]

__version__ = "0.3.0a1"

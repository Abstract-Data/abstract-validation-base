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

# Lazy imports for optional dependencies (whylogs)
_WHYLOGS_NAMES = frozenset({"WhylogsObserver", "ProfilePair", "ProfileComparison"})


def __getattr__(name: str) -> type:
    """Lazy import for optional dependencies.

    This function enables lazy loading of whylogs components, which require
    the optional 'whylogs' package. The components are only loaded when
    first accessed, avoiding import errors when whylogs is not installed.

    Args:
        name: The attribute name being accessed.

    Returns:
        The requested class from whylogs_observer module.

    Raises:
        ImportError: If whylogs is not installed and a whylogs component is requested.
        AttributeError: If the requested attribute doesn't exist.
    """
    if name in _WHYLOGS_NAMES:
        try:
            from abstract_validation_base.whylogs_observer import (
                ProfileComparison,
                ProfilePair,
                WhylogsObserver,
            )

            _components = {
                "WhylogsObserver": WhylogsObserver,
                "ProfilePair": ProfilePair,
                "ProfileComparison": ProfileComparison,
            }
            return _components[name]
        except ImportError as e:
            raise ImportError(
                f"{name} requires whylogs. Install with: "
                "pip install abstract-validation-base[whylogs]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # whylogs observers (lazy-loaded, requires whylogs optional dependency)
    "WhylogsObserver",
    "ProfilePair",
    "ProfileComparison",
]

__version__ = "0.4.0"

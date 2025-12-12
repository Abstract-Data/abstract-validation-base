# abstract-validation-base

[![Tests](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/tests.yml)
[![Ruff](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/lint.yml)
[![Version Check](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/version-check.yml/badge.svg?branch=main)](https://github.com/Abstract-Data/abstract-validation-base/actions/workflows/version-check.yml)
[![codecov](https://codecov.io/gh/Abstract-Data/abstract-validation-base/graph/badge.svg?token=WPV48XDDO9)](https://codecov.io/gh/Abstract-Data/abstract-validation-base)
[![MyPy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

> **Production-ready validation and process tracking for data transformation pipelines.**

Build robust ETL pipelines with full audit trails, composable validators, streaming validation for large files, and seamless Pydantic integration.

---

## Table of Contents

- [Why Abstract Validation Base?](#why-abstract-validation-base)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [ValidationBase Model](#validationbase-model)
  - [Validators & Pipelines](#validators--pipelines)
  - [Streaming Validation](#streaming-validation)
  - [Observer Pattern](#observer-pattern)
  - [Output Writers](#output-writers)
- [Complete Examples](#complete-examples)
- [SQLModel Integration](#sqlmodel-integration)
- [Rich Console Integration](#rich-console-integration)
- [API Reference](#api-reference)
- [Development](#development)
- [License](#license)

---

## Why Abstract Validation Base?

| Challenge | Solution |
|-----------|----------|
| **Silent data corruption** | Every cleaning operation and error is logged with timestamps |
| **Scattered validation logic** | Composable validators that can be combined into pipelines |
| **No audit trail** | Export complete process history to DataFrames for analysis |
| **Large file processing** | Streaming ValidationRunner processes millions of rows without loading into memory |
| **Real-time monitoring** | Observer pattern with Rich console dashboards |
| **Pydantic boilerplate** | Built-in process logging that works with your existing models |
| **Type safety gaps** | Full generic type support with `mypy` strict mode compatibility |

### Key Features

- **ValidationBase** — Pydantic base model with automatic process logging
- **ProcessLog** — Unified tracking for cleaning operations and errors
- **ValidationResult** — Generic result containers with error aggregation
- **BaseValidator** — Abstract base class for creating type-safe validators
- **CompositeValidator** — Combine multiple validators into validation pipelines
- **ValidationRunner** — Memory-efficient streaming validation for large files
- **Observer Pattern** — Real-time event notifications for progress tracking
- **Rich Integration** — Beautiful console dashboards and progress bars
- **Output Writers** — Export failed records and audit reports to CSV/JSON
- **ValidatedRecord** — SQLModel integration with auto-generated DB models

---

## Installation

```bash
pip install abstract-validation-base
```

Or with uv:

```bash
uv add abstract-validation-base
```

For Rich console features:

```bash
pip install abstract-validation-base[rich]
# or
pip install rich
```

---

## Quick Start

### 1. Create a Model with Process Logging

```python
from abstract_validation_base import ValidationBase

class Contact(ValidationBase):
    name: str
    email: str
    phone: str | None = None

# Instantiate and track operations
contact = Contact(name="john doe", email="JOHN@EXAMPLE.COM", phone="555-1234")

# Log a cleaning operation
contact.add_cleaning_process(
    field="name",
    original_value="john doe",
    new_value="John Doe",
    reason="Title case normalization"
)

# Log an error (without raising)
contact.add_error(
    field="phone",
    message="Phone format not standardized",
    value="555-1234"
)

# Check status
print(f"Has errors: {contact.has_errors}")        # True
print(f"Has cleaning: {contact.has_cleaning}")    # True
```

### 2. Build a Validation Pipeline

```python
from abstract_validation_base import BaseValidator, CompositeValidator, ValidationResult

class EmailValidator(BaseValidator[Contact]):
    @property
    def name(self) -> str:
        return "email_validator"

    def validate(self, item: Contact) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        if "@" not in item.email:
            result.add_error("email", "Invalid email format", item.email)
        return result

# Combine validators into a pipeline
pipeline = CompositeValidator[Contact](
    validators=[EmailValidator()],
    fail_fast=False
)

result = pipeline.validate(contact)
```

### 3. Process Large Files with Streaming

```python
import csv
from abstract_validation_base import ValidationRunner

# Process millions of rows without loading into memory
with open("large_file.csv") as f:
    runner = ValidationRunner(csv.DictReader(f), Contact)
    
    for result in runner.run():
        if result.is_valid:
            save_to_database(result.model)
        else:
            log_failure(result.error_summary)

# Get statistics after processing
print(f"Success rate: {runner.stats.success_rate:.1f}%")
```

---

## Core Concepts

### ValidationBase Model

The foundation for all validated models. Extends Pydantic's `BaseModel` with process logging.

```python
from abstract_validation_base import ValidationBase

class CustomerRecord(ValidationBase):
    name: str
    email: str
    
record = CustomerRecord(name="Test", email="test@example.com")

# Log cleaning operations
record.add_cleaning_process(
    field="email",
    original_value="TEST@EXAMPLE.COM",
    new_value="test@example.com",
    reason="Normalized to lowercase",
    operation_type="normalization"  # Optional
)

# Log errors
record.add_error(
    field="email",
    message="Domain not in allowlist",
    value=record.email,
    context={"allowed_domains": ["company.com"]}  # Optional metadata
)

# Check status
record.has_errors      # bool
record.has_cleaning    # bool
record.error_count     # int
record.cleaning_count  # int

# Export audit log
entries = record.audit_log(source="import_batch_1")  # list[dict]

# For nested ValidationBase models
entries = record.audit_log_recursive(source="import_batch_1")
```

### Validators & Pipelines

Create reusable validators and combine them into pipelines.

```python
from abstract_validation_base import (
    BaseValidator,
    CompositeValidator,
    ValidatorPipelineBuilder,
    ValidationResult,
)

# Create a validator
class RequiredFieldsValidator(BaseValidator[CustomerRecord]):
    @property
    def name(self) -> str:
        return "required_fields"
    
    def validate(self, item: CustomerRecord) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        if not item.name:
            result.add_error("name", "Name is required")
        if not item.email:
            result.add_error("email", "Email is required")
        return result

# Combine with CompositeValidator
pipeline = CompositeValidator[CustomerRecord](
    validators=[RequiredFieldsValidator(), EmailFormatValidator()],
    name="customer_pipeline",
    fail_fast=False,  # Run all validators (default)
)

# Or use the fluent builder
pipeline = (
    ValidatorPipelineBuilder[CustomerRecord]("customer_pipeline")
    .add(RequiredFieldsValidator())
    .add(EmailFormatValidator())
    .fail_fast()  # Stop on first error
    .build()
)

# Validate
result = pipeline.validate(record)
if not result.is_valid:
    for error in result.errors:
        print(f"{error.field}: {error.message}")

# Dynamic validator management
pipeline.add_validator(PhoneValidator())
pipeline.remove_validator("phone_validator")
pipeline.has_validator("required_fields")  # True
pipeline.validator_names  # ["required_fields", "email_format"]
```

### Streaming Validation

Process large files efficiently with `ValidationRunner`.

```python
import csv
from abstract_validation_base import ValidationRunner, CSVFailedWriter

# Basic streaming
with open("data.csv") as f:
    runner = ValidationRunner(
        data=csv.DictReader(f),  # Iterator - NOT loaded into memory
        model_class=CustomerRecord,
        validators=pipeline,  # Optional custom validators
        total_hint=1_000_000,  # Optional: for progress percentage
    )
    
    for result in runner.run():
        if result.is_valid:
            db.insert(result.model)
        else:
            for field, msg in result.error_summary:
                log.warning(f"{field}: {msg}")

# Convenience methods
for model in runner.run_collect_valid():
    db.insert(model)

for result in runner.run_collect_failed():
    log_failure(result.raw_data)

# Batch inserts for bulk operations
for batch in runner.run_batch_valid(batch_size=1000):
    db.insert_many(batch)  # batch is List[Model]

# Parallel processing for very large files
for result in runner.run(workers=4, chunk_size=10000):
    process(result)

# Statistics and audit report
stats = runner.stats
print(f"Total: {stats.total_rows}")
print(f"Valid: {stats.valid_rows}")
print(f"Success Rate: {stats.success_rate:.1f}%")
print(f"Duration: {stats.duration_ms:.0f}ms")

# Top errors
for (field, msg), count, pct in stats.top_errors(10):
    print(f"{field}: {msg} ({count} occurrences, {pct:.1f}%)")

# Full audit report
report = runner.audit_report()
# Returns: {"summary": {...}, "top_errors": [...], "failed_samples": [...]}
```

### Observer Pattern

Track validation events in real-time.

```python
from abstract_validation_base import (
    ValidationObserver,
    ValidationEvent,
    ValidationEventType,
)

# Create a custom observer
class MetricsObserver:
    def on_event(self, event: ValidationEvent) -> None:
        if event.event_type == ValidationEventType.ERROR_ADDED:
            metrics.increment("validation.errors")
        elif event.event_type == ValidationEventType.ROW_PROCESSED:
            stats = event.data.get("stats_snapshot", {})
            if stats.get("total", 0) % 10000 == 0:
                print(f"Processed {stats['total']:,} rows...")

# Attach to a model
model = CustomerRecord(name="Test", email="test@example.com")
model.add_observer(MetricsObserver())
model.add_error("field", "error")  # Observer is notified

# Attach to a runner
runner = ValidationRunner(data, CustomerRecord)
runner.add_observer(MetricsObserver())
```

**Event Types:**

| Event Type | Emitted By | Data Keys |
|------------|------------|-----------|
| `ERROR_ADDED` | ValidationBase | `field`, `message`, `value`, `context` |
| `CLEANING_ADDED` | ValidationBase | `field`, `original_value`, `new_value`, `reason`, `operation_type` |
| `VALIDATION_STARTED` | ValidationRunner | `model_class`, `total_hint` |
| `VALIDATION_COMPLETED` | ValidationRunner | `stats` |
| `ROW_PROCESSED` | ValidationRunner | `row_index`, `is_valid`, `stats_snapshot`, `errors` |
| `BATCH_STARTED` | ValidationRunner | `batch_number`, `batch_size` |
| `BATCH_COMPLETED` | ValidationRunner | `batch_number`, `batch_size` |

### Output Writers

Export failed records and audit reports.

```python
from abstract_validation_base import (
    CSVFailedWriter,
    JSONLinesFailedWriter,
    AuditReportWriter,
)

# Write failed records to CSV
writer = CSVFailedWriter(
    "failed_records.csv",
    include_raw_data=True,
    max_errors_per_row=5,
)
count = writer.write_all(runner.run_collect_failed())

# Write failed records to JSON Lines
writer = JSONLinesFailedWriter(
    "failed_records.jsonl",
    include_raw_data=True,
    indent=None,  # Compact
)
count = writer.write_all(runner.run_collect_failed())

# Stream writes with context manager
with CSVFailedWriter("failed.csv") as writer:
    for result in runner.run_collect_failed():
        writer.write_one(result)

# Write audit reports
writer = AuditReportWriter("audit.json")  # Auto-detect format
writer.write(runner.audit_report())

writer = AuditReportWriter(
    "audit.csv",
    errors_path="top_errors.csv",     # Separate file for errors
    include_samples=True,
    samples_path="failed_samples.csv",
)
writer.write(runner.audit_report())
```

---

## Complete Examples

### Data Cleaning Pipeline

```python
import re
from pydantic import field_validator
from abstract_validation_base import ValidationBase

class CustomerRecord(ValidationBase):
    first_name: str
    last_name: str
    email: str
    phone: str | None = None
    
    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower() if isinstance(v, str) else v
    
    @field_validator("phone", mode="before")
    @classmethod
    def normalize_phone(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return re.sub(r"\D", "", v) or None

record = CustomerRecord(
    first_name="John",
    last_name="Doe",
    email="  JOHN@EXAMPLE.COM  ",
    phone="(555) 123-4567"
)

print(record.email)  # john@example.com
print(record.phone)  # 5551234567
```

### Large File Processing with Progress

```python
import csv
from rich.progress import Progress
from abstract_validation_base import (
    ValidationRunner,
    SimpleProgressObserver,
    CSVFailedWriter,
    AuditReportWriter,
)

with open("million_records.csv") as f:
    runner = ValidationRunner(
        csv.DictReader(f),
        CustomerRecord,
        validators=pipeline,
        total_hint=1_000_000,
    )
    
    # Add Rich progress bar
    with Progress() as progress:
        observer = SimpleProgressObserver(progress, "Processing")
        runner.add_observer(observer)
        
        # Process and write failures
        with CSVFailedWriter("failures.csv") as writer:
            for result in runner.run():
                if result.is_valid:
                    db.insert(result.model)
                else:
                    writer.write_one(result)
    
    # Export audit report
    AuditReportWriter("audit.json").write(runner.audit_report())
```

### Multi-Validator Pipeline

```python
from abstract_validation_base import (
    BaseValidator,
    CompositeValidator,
    ValidationResult,
)

class RequiredFieldsValidator(BaseValidator[Order]):
    @property
    def name(self) -> str:
        return "required_fields"
    
    def validate(self, item: Order) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        if not item.order_id:
            result.add_error("order_id", "Order ID is required")
        if not item.items:
            result.add_error("items", "At least one item required")
        return result

class BusinessRulesValidator(BaseValidator[Order]):
    @property
    def name(self) -> str:
        return "business_rules"
    
    def validate(self, item: Order) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        if item.total < 0:
            result.add_error("total", "Cannot be negative", str(item.total))
        if item.total > 10000:
            result.add_error("total", "Exceeds maximum", str(item.total))
        return result

pipeline = CompositeValidator[Order](
    validators=[RequiredFieldsValidator(), BusinessRulesValidator()],
    name="order_pipeline",
    fail_fast=False,
)
```

---

## SQLModel Integration

`ValidatedRecord` provides seamless SQLModel integration.

```python
from sqlmodel import Session, create_engine, select
from abstract_validation_base import ValidatedRecord

# Define once, get both validation and DB model
class CustomerRecord(ValidatedRecord, table_name="customers"):
    email: str
    name: str
    tier: str = "standard"

# Use ValidationBase features
customer = CustomerRecord(email="john@example.com", name="John")
customer.add_error("email", "Domain not allowed", customer.email)
customer.add_cleaning_process("name", "  john  ", "John", "Trimmed")

# Convert to SQLModel for database
db_customer = customer.to_db()
session.add(db_customer)
session.commit()

# Or with field overrides
db_customer = customer.to_db(id=123, tier="premium")

# Access the generated SQLModel class for queries
CustomerDB = CustomerRecord.db_model()
customers = session.exec(select(CustomerDB)).all()
```

---

## Rich Console Integration

Beautiful progress displays using Rich.

### Simple Progress Bar

```python
from rich.progress import Progress
from abstract_validation_base import ValidationRunner, SimpleProgressObserver

with Progress() as progress:
    observer = SimpleProgressObserver(progress, "Validating")
    runner.add_observer(observer)
    
    for result in runner.run():
        process(result)
```

### Full Dashboard

```python
from abstract_validation_base import RichDashboardObserver

observer = RichDashboardObserver(
    top_errors_count=10,  # Show top 10 errors
    refresh_rate=10,      # Updates per second
)
runner.add_observer(observer)

with observer:  # Starts/stops live display
    for result in runner.run():
        process(result)
```

The dashboard shows:
- Progress bar with valid/failed counts
- Live statistics panel
- Top errors table with counts and percentages

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `ValidationBase` | Pydantic model with process logging |
| `ProcessEntry` | Individual log entry (cleaning or error) |
| `ProcessLog` | Container for cleaning and error entries |
| `ValidationResult` | Result container with errors |
| `ValidationError` | Individual validation error |

### Validators

| Class | Description |
|-------|-------------|
| `BaseValidator[T]` | Abstract base class for validators |
| `CompositeValidator[T]` | Combine multiple validators |
| `ValidatorPipelineBuilder[T]` | Fluent builder for pipelines |
| `ValidatorProtocol[T]` | Runtime-checkable validator protocol |

### Streaming

| Class | Description |
|-------|-------------|
| `ValidationRunner[T]` | Memory-efficient streaming validator |
| `RowResult[T]` | Result for a single validated row |
| `RunnerStats` | Statistics tracker |

### Observer Pattern

| Class | Description |
|-------|-------------|
| `ValidationEvent` | Event data container |
| `ValidationEventType` | Enum of event types |
| `ValidationObserver` | Observer protocol |
| `ObservableMixin` | Mixin for adding observer support |

### Rich Integration

| Class | Description |
|-------|-------------|
| `SimpleProgressObserver` | Progress bar with counts |
| `RichDashboardObserver` | Full dashboard with stats and errors |

### Writers

| Class | Description |
|-------|-------------|
| `CSVFailedWriter` | Export failed records to CSV |
| `JSONLinesFailedWriter` | Export failed records to JSONL |
| `AuditReportWriter` | Export audit reports (CSV/JSON) |
| `FailedRecordWriter` | Protocol for custom writers |

### SQLModel

| Class | Description |
|-------|-------------|
| `ValidatedRecord` | ValidationBase with auto DB model |

---

## Development

```bash
# Clone the repository
git clone https://github.com/Abstract-Data/abstract-validation-base.git
cd abstract-validation-base

# Install dependencies
uv sync --dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run linters
uv run ruff check src tests
uv run mypy src tests
```

---

## License

MIT License

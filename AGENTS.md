# AI Agent Guidelines for abstract-validation-base

This document provides instructions for AI assistants working with code that uses the `abstract-validation-base` package.

---

## Quick Reference

| Task | Correct Approach |
|------|------------------|
| Create a validator | Inherit `BaseValidator[T]`, implement `name` property + `validate()` method |
| Report validation errors | Use `result.add_error()`, never raise exceptions |
| Log data transformations | Use `model.add_cleaning_process()` |
| Combine validators | Use `CompositeValidator` or `ValidatorPipelineBuilder` |
| Process large files | Use `ValidationRunner` with iterator input |
| Export failures | Use `CSVFailedWriter` or `JSONLinesFailedWriter` |
| Track progress | Add observer implementing `ValidationObserver` protocol |
| Report package issues | Use GitHub MCP to create issue on `Abstract-Data/abstract-validation-base` |

---

## Implementing Validators

### Required Pattern

```python
from abstract_validation_base import BaseValidator, ValidationResult

class MyValidator(BaseValidator[MyModel]):
    @property
    def name(self) -> str:
        return "my_validator"  # Used for identification and error reporting
    
    def validate(self, item: MyModel) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        
        # Add errors using result.add_error() - this auto-sets is_valid=False
        if not item.required_field:
            result.add_error("required_field", "This field is required")
        
        if item.number < 0:
            result.add_error("number", "Must be non-negative", str(item.number))
        
        return result
```

### Rules

1. **Always inherit from `BaseValidator[T]`** with the correct type parameter
2. **Implement both `name` property and `validate()` method** - both are abstract
3. **Return `ValidationResult`** - never raise exceptions for validation failures
4. **Use `result.add_error(field, message, value?)`** - this automatically sets `is_valid=False`
5. **Validators should be stateless** - don't store state between `validate()` calls
6. **Don't modify the item being validated** - validators should be side-effect free

### Type Parameter Must Match

```python
# CORRECT: Type parameter matches the model
class UserValidator(BaseValidator[User]):
    def validate(self, item: User) -> ValidationResult: ...

# INCORRECT: Type mismatch will cause type checker errors
class UserValidator(BaseValidator[Contact]):  # Wrong type!
    def validate(self, item: User) -> ValidationResult: ...
```

---

## Using ValidationBase Models

### Adding Errors vs Cleaning

```python
from abstract_validation_base import ValidationBase

class MyModel(ValidationBase):
    name: str
    email: str

model = MyModel(name="test", email="TEST@EXAMPLE.COM")

# For validation failures - something is wrong with the data
model.add_error(
    field="email",
    message="Domain not in allowlist",
    value=model.email,
    context={"allowed_domains": ["company.com"]},  # Optional context
)

# For data transformations - data was changed/cleaned
model.add_cleaning_process(
    field="email",
    original_value="TEST@EXAMPLE.COM",
    new_value="test@example.com",
    reason="Normalized to lowercase",
    operation_type="normalization",  # Optional: cleaning, normalization, formatting, etc.
)
```

### Checking Status

```python
if model.has_errors:
    print(f"Found {model.error_count} errors")

if model.has_cleaning:
    print(f"Applied {model.cleaning_count} transformations")
```

### Exporting Audit Logs

```python
# For single model
entries = model.audit_log(source="import_batch_1")

# For models with nested ValidationBase fields
entries = model.audit_log_recursive(source="import_batch_1")

# Convert to DataFrame
import pandas as pd
df = pd.DataFrame(entries)
```

### Using Observers

```python
from abstract_validation_base import ValidationObserver, ValidationEvent, ValidationEventType

class MetricsObserver:
    def on_event(self, event: ValidationEvent) -> None:
        if event.event_type == ValidationEventType.ERROR_ADDED:
            metrics.increment("validation.errors")
        elif event.event_type == ValidationEventType.CLEANING_ADDED:
            metrics.increment("validation.cleaning_operations")

model.add_observer(MetricsObserver())
model.add_error("field", "error")  # Observer is notified
```

---

## Combining Validators

### Using CompositeValidator

```python
from abstract_validation_base import CompositeValidator

# Combine validators - runs all and merges results
pipeline = CompositeValidator[MyModel](
    validators=[
        RequiredFieldsValidator(),
        FormatValidator(),
        BusinessRulesValidator(),
    ],
    name="my_pipeline",
    fail_fast=False,  # Run all validators (default)
)

# With fail_fast=True, stops on first failure
pipeline = CompositeValidator[MyModel](
    validators=[...],
    fail_fast=True,
)

result = pipeline.validate(model)
```

### Using ValidatorPipelineBuilder

```python
from abstract_validation_base import ValidatorPipelineBuilder

pipeline = (
    ValidatorPipelineBuilder[MyModel]("my_pipeline")
    .add(RequiredFieldsValidator())
    .add(FormatValidator())
    .add(BusinessRulesValidator())
    .fail_fast()  # Optional
    .build()
)
```

### Dynamic Validator Management

```python
composite = CompositeValidator[MyModel](validators=[])

# Add validators dynamically
composite.add_validator(EmailValidator())
composite.add_validator(PhoneValidator())

# Query validators
if composite.has_validator("email_validator"):
    validator = composite.get_validator("email_validator")

# Remove by name
composite.remove_validator("phone_validator")

# List all
print(composite.validator_names)  # ["email_validator"]
```

---

## Streaming Large Files

### Basic Pattern

```python
import csv
from abstract_validation_base import ValidationRunner

with open("large_file.csv") as f:
    reader = csv.DictReader(f)  # Iterator - NOT materialized
    
    runner = ValidationRunner(
        data=reader,              # Pass iterator directly
        model_class=MyModel,
        validators=pipeline,      # Optional custom validators
        total_hint=1_000_000,     # Optional: for progress percentage
    )
    
    for result in runner.run():
        if result.is_valid:
            db.insert(result.model)
        else:
            for field, msg in result.error_summary:
                log.warning(f"{field}: {msg}")
```

### Convenience Methods

```python
# Yield only valid models
for model in runner.run_collect_valid():
    db.insert(model)

# Yield only failed results
for result in runner.run_collect_failed():
    log_failure(result.raw_data)

# Batch valid models for bulk insert
for batch in runner.run_batch_valid(batch_size=1000):
    db.insert_many(batch)  # batch is List[MyModel]
```

### Parallel Processing

```python
# For very large files (>1M rows)
for result in runner.run(workers=4, chunk_size=10000):
    process(result)
```

### Accessing Statistics

```python
# After iteration completes
stats = runner.stats
print(f"Success rate: {stats.success_rate:.1f}%")
print(f"Duration: {stats.duration_ms:.0f}ms")

# Top errors
for (field, msg), count, pct in stats.top_errors(10):
    print(f"{field}: {msg} ({count} occurrences, {pct:.1f}%)")

# Full audit report
report = runner.audit_report()
# Returns: {"summary": {...}, "top_errors": [...], "failed_samples": [...]}
```

---

## Output Writers

### Writing Failed Records

```python
from abstract_validation_base import CSVFailedWriter, JSONLinesFailedWriter

# CSV format
writer = CSVFailedWriter(
    "failed_records.csv",
    include_raw_data=True,    # Include original fields
    max_errors_per_row=5,     # Limit error columns
)
count = writer.write_all(runner.run_collect_failed())

# JSON Lines format
writer = JSONLinesFailedWriter(
    "failed_records.jsonl",
    include_raw_data=True,
    indent=None,  # Compact (default) or int for pretty
)
count = writer.write_all(runner.run_collect_failed())

# Using context manager for streaming
with CSVFailedWriter("failed.csv") as writer:
    for result in runner.run_collect_failed():
        writer.write_one(result)
```

### Writing Audit Reports

```python
from abstract_validation_base import AuditReportWriter

# Auto-detect format from extension
writer = AuditReportWriter("audit.json")  # JSON format
writer = AuditReportWriter("audit.csv")   # CSV format

# Explicit format
writer = AuditReportWriter("report.txt", format="json")

# With options
writer = AuditReportWriter(
    "audit.csv",
    errors_path="top_errors.csv",     # Separate file for errors
    include_samples=True,              # Include failed samples
    samples_path="failed_samples.csv", # Separate file for samples
)

writer.write(runner.audit_report())
```

---

## Observer Pattern for Progress

### Simple Progress Bar (Rich)

```python
from rich.progress import Progress
from abstract_validation_base import SimpleProgressObserver

with Progress() as progress:
    observer = SimpleProgressObserver(progress, task_description="Validating")
    runner.add_observer(observer)
    
    for result in runner.run():
        process(result)
```

### Full Dashboard (Rich)

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

### Custom Observer

```python
from abstract_validation_base import ValidationObserver, ValidationEvent, ValidationEventType

class CustomObserver:
    def on_event(self, event: ValidationEvent) -> None:
        if event.event_type == ValidationEventType.VALIDATION_STARTED:
            print(f"Starting: {event.data.get('model_class')}")
        
        elif event.event_type == ValidationEventType.ROW_PROCESSED:
            stats = event.data.get("stats_snapshot", {})
            if stats.get("total", 0) % 10000 == 0:
                print(f"Processed {stats['total']:,} rows...")
        
        elif event.event_type == ValidationEventType.VALIDATION_COMPLETED:
            print(f"Complete: {event.data.get('stats')}")

runner.add_observer(CustomObserver())
```

---

## SQLModel Integration

### Basic Usage

```python
from abstract_validation_base import ValidatedRecord

class UserRecord(ValidatedRecord, table_name="users"):
    email: str
    name: str
    tier: str = "free"

# Use as ValidationBase
user = UserRecord(email="test@example.com", name="Test")
user.add_error("email", "Domain blocked")
user.add_cleaning_process("name", "  Test  ", "Test", "Trimmed whitespace")

# Convert to SQLModel for database
db_user = user.to_db()
session.add(db_user)

# Or with field overrides
db_user = user.to_db(id=123, tier="premium")

# Access the generated SQLModel class
UserDB = UserRecord.db_model()
users = session.exec(select(UserDB)).all()
```

### Key Points

- `table_name` parameter sets the database table name
- Auto-generated DB model includes `id: int` primary key
- `process_log` is excluded from database model
- DB model is lazily generated on first access

---

## Anti-Patterns to Avoid

### 1. Raising Exceptions in Validators

```python
# WRONG - Don't raise exceptions for validation failures
class BadValidator(BaseValidator[MyModel]):
    def validate(self, item: MyModel) -> ValidationResult:
        if not item.email:
            raise ValueError("Email required")  # DON'T DO THIS
        return ValidationResult(is_valid=True)

# CORRECT - Use ValidationResult
class GoodValidator(BaseValidator[MyModel]):
    def validate(self, item: MyModel) -> ValidationResult:
        result = ValidationResult(is_valid=True)
        if not item.email:
            result.add_error("email", "Email required")
        return result
```

### 2. Modifying Items in Validators

```python
# WRONG - Don't modify the item being validated
class BadValidator(BaseValidator[MyModel]):
    def validate(self, item: MyModel) -> ValidationResult:
        item.email = item.email.lower()  # DON'T MODIFY
        return ValidationResult(is_valid=True)

# CORRECT - Validators should only inspect, not modify
# Use ValidationBase.add_cleaning_process() separately if cleaning is needed
```

### 3. Materializing Large Iterators

```python
# WRONG - Loads entire file into memory
with open("huge_file.csv") as f:
    data = list(csv.DictReader(f))  # DON'T DO THIS
    runner = ValidationRunner(iter(data), MyModel)

# CORRECT - Pass iterator directly
with open("huge_file.csv") as f:
    runner = ValidationRunner(csv.DictReader(f), MyModel)
    for result in runner.run():
        process(result)
```

### 4. Accessing Private Attributes

```python
# WRONG - Don't access private attributes
entries = model._process_log.errors  # DON'T DO THIS

# CORRECT - Use public methods
entries = model.audit_log()
has_errors = model.has_errors
count = model.error_count
```

### 5. Forgetting Type Parameters

```python
# WRONG - Missing type parameter loses type safety
class MyValidator(BaseValidator):  # Missing [T]
    def validate(self, item) -> ValidationResult: ...

# CORRECT - Always specify type parameter
class MyValidator(BaseValidator[MyModel]):
    def validate(self, item: MyModel) -> ValidationResult: ...
```

### 6. Creating ValidationResult with is_valid=False

```python
# WRONG - Manually setting is_valid without errors
result = ValidationResult(is_valid=False)  # No errors listed!

# CORRECT - Use add_error() which handles is_valid
result = ValidationResult(is_valid=True)
result.add_error("field", "message")  # Sets is_valid=False automatically
```

---

## Event Types Reference

| Event Type | Emitted By | Data Keys |
|------------|------------|-----------|
| `ERROR_ADDED` | ValidationBase | `field`, `message`, `value`, `context` |
| `CLEANING_ADDED` | ValidationBase | `field`, `original_value`, `new_value`, `reason`, `operation_type` |
| `VALIDATION_STARTED` | CompositeValidator, ValidationRunner | `model_class`, `validator_name`, `total_hint` |
| `VALIDATION_COMPLETED` | CompositeValidator, ValidationRunner | `is_valid`, `error_count`, `duration_ms`, `stats` |
| `ROW_PROCESSED` | ValidationRunner | `row_index`, `is_valid`, `stats_snapshot`, `errors` |
| `BATCH_STARTED` | ValidationRunner | `batch_number`, `batch_size` |
| `BATCH_COMPLETED` | ValidationRunner | `batch_number`, `batch_size` |

---

## Reporting Issues to This Package

If you're an AI agent assisting with a project that uses `abstract-validation-base` and you discover a bug, missing feature, or documentation issue, you can file an issue against this package.

**Repository:** `Abstract-Data/abstract-validation-base`

### Using MCP GitHub Server (Recommended for AI Agents)

If your AI client (Cursor, Claude Desktop, etc.) has the GitHub MCP server configured, use the `create_issue` tool targeting `Abstract-Data/abstract-validation-base`.

**Cursor** (`~/.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your-token>"
      }
    }
  }
}
```

### Issue Guidelines

When filing issues, please include:

1. **Clear title** prefixed with `[Bug]`, `[Feature]`, or `[Docs]`
2. **Package version** (`abstract_validation_base.__version__`)
3. **Minimal reproduction** - smallest code that demonstrates the issue
4. **Expected vs actual behavior**
5. **Environment** - Python version, OS

### Issue Templates

**Bug Report:**
```
## Description
[What went wrong]

## Steps to Reproduce
1. [First step]
2. [Second step]

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- Package Version: X.Y.Z
- Python Version: 3.X
- OS: [macOS/Linux/Windows]

## Additional Context
[Stack traces, related issues, etc.]
```

**Feature Request:**
```
## Problem Statement
[What limitation or pain point does this address?]

## Proposed Solution
[What would you like to see?]

## Alternatives Considered
[Other approaches you've thought about]

## Use Case
[Example code showing how this would be used]
```


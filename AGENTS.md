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
| Profile data during validation | Use `WhylogsObserver` |
| Write tests | Use Hypothesis for property-based, stateful, and unit tests |
| Report package issues | Use `mcp_github_create_issue` with `[Bug]:`, `[Feature]:`, or `[Docs]:` title prefix |
| Submit a PR | Run `ruff check`, `ruff format`, `mypy`, `pytest` locally first |
| Plan complex tasks | Use `STEP# - AGENT#:` prefix format for parallel-safe task breakdown |

---

## Task Planning for Parallel Execution

When planning complex tasks, structure them to show parallelization opportunities. This enables:
- **Multi-agent execution** — Multiple Cursor sessions or agents can work simultaneously
- **Clear dependencies** — Steps are sequential barriers; agents within steps are parallel-safe
- **Single-agent fallback** — One agent can work through tasks top-to-bottom sequentially

### Plan Format

```
STEP 1 - AGENT A: [Task that can run in parallel with B and C]
STEP 1 - AGENT B: [Task that can run in parallel with A and C]
STEP 1 - AGENT C: [Task that can run in parallel with A and B]
STEP 2 - AGENT A: [Depends on all STEP 1 tasks completing]
STEP 2 - AGENT B: [Depends on all STEP 1 tasks completing]
STEP 3 - AGENT A: [Final integration task]
```

### Planning Rules

1. **Steps are sequential barriers** — All tasks in STEP N must complete before STEP N+1 begins
2. **Agents within a step are parallel-safe** — No dependencies between same-step tasks
3. **Single agent mode** — Work through tasks top-to-bottom, treating agent labels as informational
4. **Multi-agent mode** — Different sessions claim different agent letters for true parallelism

### Dependency Checklist

Before assigning parallel agents to the same step, verify:

- [ ] **No shared file sections** — Agents edit different files or non-overlapping sections
- [ ] **No import dependencies** — Task B doesn't import something Task A is creating
- [ ] **No shared test fixtures** — Parallel test changes don't conflict
- [ ] **No database migrations** — Only one agent modifies schema at a time

### Example Plan: Adding New Validator Types

```
STEP 1 - AGENT A: Create EmailValidator class in src/abstract_validation_base/validators.py
STEP 1 - AGENT B: Create PhoneValidator class in src/abstract_validation_base/validators.py (separate section)
STEP 1 - AGENT C: Add Hypothesis strategies for email/phone generation in tests/conftest.py

STEP 2 - AGENT A: Write property-based tests for EmailValidator in tests/test_validators.py
STEP 2 - AGENT B: Write property-based tests for PhoneValidator in tests/test_validators.py
STEP 2 - AGENT C: Write stateful tests for validator combinations in tests/test_validators.py

STEP 3 - AGENT A: Add exports to __init__.py and update type stubs
STEP 3 - AGENT B: Run full test suite, fix any integration issues

STEP 4 - AGENT A: Update README.md with usage examples
```

### How Agents Should Use This

**When creating a plan:**
1. Identify independent work units
2. Group truly parallel tasks into the same STEP
3. Assign different AGENT letters to parallel tasks
4. Put dependent work in later STEPs
5. **Name each task with the `STEP# - AGENT#:` prefix** so it appears clearly in the Agents panel

**Task Naming Convention:**
```
✅ Good: "STEP 1 - AGENT A: Create EmailValidator class"
✅ Good: "STEP 2 - AGENT B: Write property tests for PhoneValidator"
❌ Bad:  "Create EmailValidator class"
❌ Bad:  "Task 1: Create EmailValidator"
```

The prefix makes it easy to:
- Identify which step a task belongs to
- See which agent slot is assigned
- Track parallel work in the Agents panel
- Understand task dependencies at a glance

**When executing a plan (single agent):**
1. Work through tasks in order: STEP 1A → 1B → 1C → STEP 2A → 2B → ...
2. Complete all tasks regardless of agent assignment

**When executing a plan (multiple agents/sessions):**
1. Each session claims an agent letter (A, B, or C)
2. Work only on tasks matching your agent letter
3. Wait at step boundaries for other agents to complete
4. Coordinate via git commits or shared status file

### Agent Awareness

| Environment | Agent Awareness |
|-------------|-----------------|
| Single Cursor session | Agent sees full plan, executes sequentially |
| Multiple Cursor sessions | Each session independent; coordinate manually via git |
| Cursor Plan Mode (multi-agent) | Cursor may auto-distribute; agents see their assignments |
| External orchestration (AutoGen, CrewAI) | Orchestrator assigns tasks; agents receive specific work |

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

### Data Profiling with whylogs

Use `WhylogsObserver` to automatically profile your data during validation. This enables data quality monitoring, drift detection, and statistical comparison between raw input data and validated outputs.

**Installation:**

```bash
pip install abstract-validation-base[whylogs]
```

**Basic Usage:**

```python
import csv
from abstract_validation_base import ValidationRunner, WhylogsObserver

# Create observer with default settings
observer = WhylogsObserver(
    chunk_size=10000,      # Rows to buffer before profiling (memory vs latency tradeoff)
    profile_raw=True,      # Profile all input data
    profile_valid=True,    # Profile only records that pass validation
)

with open("data.csv") as f:
    runner = ValidationRunner(csv.DictReader(f), MyModel)
    runner.add_observer(observer)
    
    for result in runner.run():
        process(result)

# After validation completes, retrieve profiles
profiles = observer.get_profiles()
```

**Exporting Profiles:**

```python
# Write to whylogs binary format (recommended for large profiles)
paths = profiles.write(
    raw_path="output/raw_profile.bin",
    valid_path="output/valid_profile.bin",
)

# Convert to pandas DataFrames for analysis
dfs = profiles.to_pandas()
raw_stats = dfs["raw"]   # DataFrame with column statistics
valid_stats = dfs["valid"]
```

**Comparing Raw vs Valid Profiles:**

```python
# Get comparison statistics
comparison = observer.compare_profiles()

print(f"Pass rate: {comparison.pass_rate:.1%}")
print(f"Raw columns: {comparison.raw_column_count}")
print(f"Valid columns: {comparison.valid_column_count}")
print(f"Columns only in raw: {comparison.columns_only_in_raw}")
print(f"Columns only in valid: {comparison.columns_only_in_valid}")

# Serialize comparison for reporting
report = comparison.to_dict()
```

**Selective Profiling:**

```python
# Profile only raw data (useful for diagnosing input issues)
observer = WhylogsObserver(profile_raw=True, profile_valid=False)

# Profile only valid data (useful for downstream quality checks)
observer = WhylogsObserver(profile_raw=False, profile_valid=True)
```

**With Custom Schema:**

```python
from whylogs.core.schema import DatasetSchema, DeclarativeSchema
from whylogs.core.resolvers import StandardResolver

# Define explicit types for better profiling accuracy
schema = DeclarativeSchema([
    StandardResolver(),
])

observer = WhylogsObserver(schema=schema)
```

**Reusing Observer:**

```python
# Reset state before reusing with another runner
observer.reset()
runner2 = ValidationRunner(other_data, MyModel)
runner2.add_observer(observer)
for result in runner2.run():
    process(result)
```

**Key Points:**

- `get_profiles()` and `compare_profiles()` raise `RuntimeError` if called while validation is running
- Thread-safe for use with parallel validation (`runner.run(workers=4)`)
- Profiles are accumulated incrementally using whylogs merge for memory efficiency
- The `chunk_size` parameter controls memory usage vs profiling latency tradeoff

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

### 7. Writing Example-Based Tests Instead of Property Tests

```python
# WRONG - Hardcoded examples miss edge cases
def test_add_error():
    result = ValidationResult(is_valid=True)
    result.add_error("email", "Invalid")
    assert result.is_valid is False

# CORRECT - Property-based test covers all inputs
from hypothesis import given, strategies as st

@given(st.text(min_size=1), st.text(min_size=1))
def test_add_error(field: str, message: str):
    result = ValidationResult(is_valid=True)
    result.add_error(field, message)
    assert result.is_valid is False
```

---

## Event Types Reference

| Event Type | Emitted By | Data Keys |
|------------|------------|-----------|
| `ERROR_ADDED` | ValidationBase | `field`, `message`, `value`, `context` |
| `CLEANING_ADDED` | ValidationBase | `field`, `original_value`, `new_value`, `reason`, `operation_type` |
| `VALIDATION_STARTED` | CompositeValidator, ValidationRunner | `model_class`, `validator_name`, `total_hint` |
| `VALIDATION_COMPLETED` | CompositeValidator, ValidationRunner | `is_valid`, `error_count`, `duration_ms`, `stats` |
| `ROW_PROCESSED` | ValidationRunner | `row_index`, `is_valid`, `stats_snapshot`, `errors`, `raw_data`, `model_dict` |
| `BATCH_STARTED` | ValidationRunner | `batch_number`, `batch_size` |
| `BATCH_COMPLETED` | ValidationRunner | `batch_number`, `batch_size` |

---

## Testing with Hypothesis

All tests in this repository **must** use [Hypothesis](https://hypothesis.readthedocs.io/) for property-based, stateful, and unit testing. Hypothesis finds edge cases that traditional example-based tests miss.

### Property-Based Testing

Use `@given` to test invariants that should hold for all valid inputs:

```python
from hypothesis import given, strategies as st
from abstract_validation_base import ValidationResult

@given(st.text(), st.text())
def test_add_error_always_sets_invalid(field: str, message: str):
    """Property: Adding an error always results in is_valid=False."""
    result = ValidationResult(is_valid=True)
    result.add_error(field, message)
    assert result.is_valid is False
    assert len(result.errors) >= 1


@given(st.lists(st.tuples(st.text(min_size=1), st.text(min_size=1)), min_size=1))
def test_error_count_matches_errors_added(errors: list[tuple[str, str]]):
    """Property: Error count equals number of errors added."""
    result = ValidationResult(is_valid=True)
    for field, message in errors:
        result.add_error(field, message)
    assert len(result.errors) == len(errors)
```

### Stateful Testing

Use `RuleBasedStateMachine` to test complex interactions and state transitions:

```python
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
from abstract_validation_base import ValidationBase

class ValidationBaseStateMachine(RuleBasedStateMachine):
    """Test that ValidationBase maintains consistent state through operations."""
    
    def __init__(self):
        super().__init__()
        self.model = MyModel(name="test")
        self.expected_error_count = 0
        self.expected_cleaning_count = 0
    
    @rule(field=st.text(min_size=1), message=st.text(min_size=1))
    def add_error(self, field: str, message: str):
        self.model.add_error(field, message)
        self.expected_error_count += 1
    
    @rule(
        field=st.text(min_size=1),
        original=st.text(),
        new=st.text(),
        reason=st.text(min_size=1)
    )
    def add_cleaning(self, field: str, original: str, new: str, reason: str):
        self.model.add_cleaning_process(field, original, new, reason)
        self.expected_cleaning_count += 1
    
    @invariant()
    def error_count_consistent(self):
        assert self.model.error_count == self.expected_error_count
    
    @invariant()
    def cleaning_count_consistent(self):
        assert self.model.cleaning_count == self.expected_cleaning_count
    
    @invariant()
    def has_errors_reflects_count(self):
        assert self.model.has_errors == (self.expected_error_count > 0)


TestValidationBase = ValidationBaseStateMachine.TestCase
```

### Unit Testing with Hypothesis

Even simple unit tests benefit from Hypothesis strategies:

```python
from hypothesis import given, strategies as st, assume
from abstract_validation_base import CompositeValidator

# Strategy for generating valid model data
model_data = st.fixed_dictionaries({
    "name": st.text(min_size=1, max_size=100),
    "email": st.emails(),
    "age": st.integers(min_value=0, max_value=150),
})

@given(model_data)
def test_composite_validator_runs_all_validators(data: dict):
    """All validators in composite are executed."""
    model = MyModel(**data)
    composite = CompositeValidator[MyModel](
        validators=[ValidatorA(), ValidatorB()],
        name="test_composite"
    )
    result = composite.validate(model)
    # Verify both validators contributed to the result
    assert isinstance(result.is_valid, bool)
```

### Custom Strategies

Define reusable strategies for domain types:

```python
from hypothesis import strategies as st

# Strategy for generating ValidationBase models
@st.composite
def validation_models(draw, with_errors: bool = False, with_cleaning: bool = False):
    """Generate MyModel instances with optional errors/cleaning."""
    model = MyModel(
        name=draw(st.text(min_size=1)),
        email=draw(st.emails()),
    )
    
    if with_errors:
        num_errors = draw(st.integers(min_value=1, max_value=5))
        for _ in range(num_errors):
            model.add_error(
                draw(st.text(min_size=1, max_size=20)),
                draw(st.text(min_size=1, max_size=100))
            )
    
    if with_cleaning:
        num_cleaning = draw(st.integers(min_value=1, max_value=5))
        for _ in range(num_cleaning):
            model.add_cleaning_process(
                draw(st.text(min_size=1, max_size=20)),
                draw(st.text()),
                draw(st.text()),
                draw(st.text(min_size=1, max_size=100))
            )
    
    return model


@given(validation_models(with_errors=True))
def test_models_with_errors_report_has_errors(model):
    assert model.has_errors is True
```

### Settings and Profiles

Configure Hypothesis appropriately for CI vs local development:

```python
from hypothesis import settings, Phase

# In conftest.py - register profiles
settings.register_profile("ci", max_examples=1000, deadline=None)
settings.register_profile("dev", max_examples=100, deadline=500)
settings.register_profile("debug", max_examples=10, phases=[Phase.generate])

# Load profile from environment
import os
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
```

### Testing Guidelines

1. **Prefer property-based tests** — They find edge cases you won't think of
2. **Use stateful tests for stateful components** — ValidationBase, ProcessLog, etc.
3. **Define custom strategies** in `conftest.py` for reuse across test modules
4. **Use `@example` decorator** to pin specific regression cases
5. **Set `deadline=None`** for tests involving I/O or complex operations
6. **Use `assume()`** to filter invalid combinations rather than complex strategies

### Example Test Structure

```
tests/
├── conftest.py           # Shared fixtures and Hypothesis strategies
├── strategies.py         # Custom Hypothesis strategies (optional)
├── test_base.py          # Property + stateful tests for ValidationBase
├── test_validators.py    # Property tests for validator behavior
├── test_runner.py        # Stateful tests for ValidationRunner
└── test_writers.py       # Property tests for output writers
```

---

## Reporting Issues to This Package

If you're an AI agent assisting with a project that uses `abstract-validation-base` and you discover a bug, missing feature, or documentation issue, you can file an issue against this package.

**Repository:** `Abstract-Data/abstract-validation-base`

### Using MCP GitHub Server (Recommended for AI Agents)

If your AI client (Cursor, Claude Desktop, etc.) has the GitHub MCP server configured, use the `mcp_github_create_issue` tool targeting `Abstract-Data/abstract-validation-base`.

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

### Issue Templates (Structured Forms)

This repository uses GitHub's YAML-based issue forms. When creating issues via MCP, format the body to match the template fields:

**Bug Report** — Use title prefix `[Bug]: `
```markdown
### Prerequisites
- [x] I have searched existing issues
- [x] I am using the latest version

### Bug Description
[Clear description of the bug]

### Steps to Reproduce
```python
from abstract_validation_base import ValidationBase

# Minimal code that reproduces the issue
```

### Expected Behavior
[What should happen]

### Actual Behavior
[What actually happens]

### Affected Component
[ValidationBase / Validators / Runner / Writers / SQLModel / Rich / Events]

### Error Output
```
[Stack trace if applicable]
```

### Environment
- Package Version: [e.g., 0.3.0a1]
- Python Version: [3.10 / 3.11 / 3.12]
- OS: [macOS / Linux / Windows]
```

**Feature Request** — Use title prefix `[Feature]: `
```markdown
### Prerequisites
- [x] I have searched existing issues
- [x] I have read the documentation

### Problem Statement
[What limitation or pain point does this address?]

### Proposed Solution
[What would you like to see?]

### Alternatives Considered
[Other approaches you've thought about]

### Affected Component
[ValidationBase / Validators / Runner / Writers / SQLModel / Rich / New Component]

### Use Case Example
```python
# Example code showing how this feature would be used
```

### Priority
[Nice to have / Would significantly improve workflow / Blocking use case]
```

**Documentation Issue** — Use title prefix `[Docs]: `
```markdown
### Issue Type
[Missing / Incorrect / Unclear / Needs example / Typo / Outdated]

### Location
[README.md / AGENTS.md / Docstrings / API reference]

### Problem Description
[What's wrong or missing?]

### Suggested Improvement
[How should the documentation be improved?]
```

### Auto-Labeling

Issues are automatically labeled based on content:

| Keywords in Issue | Label Applied |
|-------------------|---------------|
| ValidationBase, add_error, ProcessLog | `component:base` |
| BaseValidator, CompositeValidator, pipeline | `component:validators` |
| ValidationRunner, streaming, large file | `component:runner` |
| CSVFailedWriter, JSONLines, AuditReport | `component:writers` |
| ValidatedRecord, SQLModel, to_db | `component:sqlmodel` |
| RichDashboard, SimpleProgress, observer | `component:rich` |
| WhylogsObserver, ProfilePair, profiling, whylogs | `component:whylogs` |

Bug reports automatically receive a helpful comment with relevant documentation links.

---

## Contributing Pull Requests

When submitting PRs to this repository, ensure the following checks pass locally:

### Pre-submission Checklist

```bash
# Linting
uv run ruff check src tests
uv run ruff format src tests

# Type checking
uv run mypy src

# Tests
uv run pytest
```

### PR Requirements

1. **Link to related issue** — Reference with `Closes #123`
2. **Type of change** — Bug fix, feature, docs, refactor, tests
3. **Tests** — Add Hypothesis-based tests (property, stateful, or unit) for new functionality
4. **Documentation** — Update docs/docstrings for user-facing changes

### CI Checks (Automated)

These run automatically on PRs:
- `ruff check` and `ruff format --check`
- `mypy src`
- `pytest --cov`

PRs cannot be merged until all CI checks pass


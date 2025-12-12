"""Tests for SQLModel integration."""

from __future__ import annotations

import pytest
from sqlmodel import SQLModel

from abstract_validation_base import ValidatedRecord


class TestValidatedRecord:
    """Tests for ValidatedRecord base class."""

    def test_creates_db_model_with_table_name(self) -> None:
        """ValidatedRecord creates a DB model when table_name is provided."""

        class UserRecord(ValidatedRecord, table_name="users"):
            email: str
            name: str

        assert UserRecord.__table_name__ == "users"
        # DB model is created lazily on first access
        db_model = UserRecord.db_model()
        assert db_model is not None
        assert UserRecord.__db_model__ is not None

    def test_db_model_has_correct_tablename(self) -> None:
        """Generated DB model has the correct __tablename__."""

        class ProductRecord(ValidatedRecord, table_name="products"):
            sku: str
            price: float

        db_model = ProductRecord.db_model()
        assert db_model.__tablename__ == "products"

    def test_db_model_has_id_field(self) -> None:
        """Generated DB model has an auto-generated id field."""

        class ItemRecord(ValidatedRecord, table_name="items"):
            name: str

        db_model = ItemRecord.db_model()
        assert "id" in db_model.model_fields
        assert db_model.model_fields["id"].annotation == int | None

    def test_db_model_has_matching_fields(self) -> None:
        """Generated DB model has all fields from validation model."""

        class OrderRecord(ValidatedRecord, table_name="orders"):
            order_id: str
            total: float
            status: str = "pending"

        db_model = OrderRecord.db_model()
        assert "order_id" in db_model.model_fields
        assert "total" in db_model.model_fields
        assert "status" in db_model.model_fields

    def test_db_model_excludes_process_log(self) -> None:
        """Generated DB model does not include process_log field."""

        class TestRecord(ValidatedRecord, table_name="test"):
            value: str

        db_model = TestRecord.db_model()
        assert "process_log" not in db_model.model_fields

    def test_db_model_preserves_defaults(self) -> None:
        """Generated DB model preserves field defaults."""

        class ConfigRecord(ValidatedRecord, table_name="configs"):
            key: str
            value: str = "default_value"
            enabled: bool = True

        db_model = ConfigRecord.db_model()
        instance = db_model(key="test", value="custom")
        assert instance.enabled is True  # type: ignore[attr-defined]

    def test_to_db_converts_instance(self) -> None:
        """to_db() converts validation model to DB model instance."""

        class CustomerRecord(ValidatedRecord, table_name="customers"):
            email: str
            name: str

        customer = CustomerRecord(email="test@example.com", name="Test User")
        db_customer = customer.to_db()

        assert isinstance(db_customer, SQLModel)
        assert db_customer.email == "test@example.com"  # type: ignore[attr-defined]
        assert db_customer.name == "Test User"  # type: ignore[attr-defined]
        assert db_customer.id is None  # type: ignore[attr-defined]

    def test_to_db_with_overrides(self) -> None:
        """to_db() accepts field overrides."""

        class RecordWithOverrides(ValidatedRecord, table_name="overrides"):
            value: str

        record = RecordWithOverrides(value="original")
        db_record = record.to_db(id=123, value="overridden")

        assert db_record.id == 123  # type: ignore[attr-defined]
        assert db_record.value == "overridden"  # type: ignore[attr-defined]

    def test_db_model_class_method(self) -> None:
        """db_model() returns the generated SQLModel class."""

        class EntityRecord(ValidatedRecord, table_name="entities"):
            name: str

        db_class = EntityRecord.db_model()

        assert db_class.__name__ == "EntityRecordDB"
        assert issubclass(db_class, SQLModel)

    def test_validation_features_work(self) -> None:
        """ValidationBase features work on ValidatedRecord."""

        class AuditRecord(ValidatedRecord, table_name="audit"):
            action: str
            user_id: str

        record = AuditRecord(action="login", user_id="user123")

        # Test add_error
        record.add_error("action", "Suspicious action detected", value="login")
        assert record.has_errors
        assert record.error_count == 1

        # Test add_cleaning_process
        record.add_cleaning_process(
            field="user_id",
            original_value="USER123",
            new_value="user123",
            reason="Lowercased user ID",
        )
        assert record.has_cleaning
        assert record.cleaning_count == 1

        # Test audit_log
        audit = record.audit_log(source="test")
        assert len(audit) == 2
        assert all(entry["source"] == "test" for entry in audit)

    def test_without_table_name_raises_on_to_db(self) -> None:
        """to_db() raises ValueError if table_name was not set."""

        class NoTableRecord(ValidatedRecord):
            value: str

        record = NoTableRecord(value="test")

        with pytest.raises(ValueError, match="was not created with table_name"):
            record.to_db()

    def test_without_table_name_raises_on_db_model(self) -> None:
        """db_model() raises ValueError if table_name was not set."""

        class AnotherNoTableRecord(ValidatedRecord):
            value: str

        with pytest.raises(ValueError, match="has no associated DB model"):
            AnotherNoTableRecord.db_model()

    def test_multiple_records_independent_db_models(self) -> None:
        """Each ValidatedRecord subclass gets its own DB model."""

        class FirstRecord(ValidatedRecord, table_name="first"):
            field_a: str

        class SecondRecord(ValidatedRecord, table_name="second"):
            field_b: int

        first_db = FirstRecord.db_model()
        second_db = SecondRecord.db_model()

        assert first_db.__tablename__ == "first"
        assert second_db.__tablename__ == "second"
        assert "field_a" in first_db.model_fields
        assert "field_b" in second_db.model_fields
        assert "field_a" not in second_db.model_fields
        assert "field_b" not in first_db.model_fields

    def test_optional_fields_handled(self) -> None:
        """Optional fields are handled correctly in DB model."""

        class OptionalRecord(ValidatedRecord, table_name="optional"):
            required_field: str
            optional_field: str | None = None

        record = OptionalRecord(required_field="test")
        db_record = record.to_db()

        assert db_record.required_field == "test"  # type: ignore[attr-defined]
        assert db_record.optional_field is None  # type: ignore[attr-defined]

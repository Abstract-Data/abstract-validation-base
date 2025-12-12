"""Tests for nested model support."""

from __future__ import annotations

import time

from hypothesis import given, settings
from hypothesis import strategies as st

from abstract_validation_base import ValidationBase

# =============================================================================
# Test Models
# =============================================================================


class Address(ValidationBase):
    """Nested model representing an address."""

    street: str
    city: str
    zip_code: str | None = None


class Contact(ValidationBase):
    """Nested model representing a contact."""

    email: str
    phone: str | None = None


class Person(ValidationBase):
    """Model with nested ValidationBase models."""

    name: str
    age: int = 0
    address: Address | None = None
    contact: Contact | None = None


class PersonWithContacts(ValidationBase):
    """Model with a list of nested ValidationBase models."""

    name: str
    contacts: list[Contact] = []


class Company(ValidationBase):
    """Deeply nested model."""

    name: str
    address: Address
    employees: list[Person] = []


# =============================================================================
# audit_log_recursive() Tests
# =============================================================================


class TestAuditLogRecursive:
    """Tests for audit_log_recursive() method."""

    def test_empty_model_returns_empty(self) -> None:
        """Test that empty model returns empty audit log."""
        person = Person(name="John")

        log = person.audit_log_recursive()

        assert log == []

    def test_flat_model_same_as_audit_log(self) -> None:
        """Test that flat model returns same as audit_log."""
        person = Person(name="John")
        person.add_error("name", "Invalid")

        regular_log = person.audit_log()
        recursive_log = person.audit_log_recursive()

        assert len(regular_log) == len(recursive_log)
        assert regular_log[0]["field"] == recursive_log[0]["field"]

    def test_nested_model_includes_child_errors(self) -> None:
        """Test that nested model errors are included."""
        address = Address(street="123 Main", city="NYC")
        person = Person(name="John", address=address)

        person.address.add_error("city", "Invalid city")

        log = person.audit_log_recursive()

        assert len(log) == 1
        assert log[0]["field"] == "city"
        assert "address" in log[0].get("source", "")

    def test_nested_model_includes_child_cleaning(self) -> None:
        """Test that nested model cleaning operations are included."""
        address = Address(street="123 Main", city="NYC")
        person = Person(name="John", address=address)

        person.address.add_cleaning_process("city", "nyc", "NYC", "Uppercased")

        log = person.audit_log_recursive()

        assert len(log) == 1
        assert log[0]["entry_type"] == "cleaning"
        assert "address" in log[0].get("source", "")

    def test_parent_and_child_errors_combined(self) -> None:
        """Test that parent and child errors are combined."""
        address = Address(street="123 Main", city="NYC")
        person = Person(name="John", address=address)

        person.add_error("name", "Name too short")
        person.address.add_error("city", "Invalid city")

        log = person.audit_log_recursive()

        assert len(log) == 2

        # Check both errors are present
        fields = {entry["field"] for entry in log}
        assert "name" in fields
        assert "city" in fields

    def test_source_prefix_applied(self) -> None:
        """Test that source prefix is applied correctly."""
        address = Address(street="123 Main", city="NYC")
        person = Person(name="John", address=address)

        person.add_error("name", "Error")
        person.address.add_error("city", "Error")

        log = person.audit_log_recursive(source="batch_1")

        # Find the entries by field
        name_entry = next(e for e in log if e["field"] == "name")
        city_entry = next(e for e in log if e["field"] == "city")

        assert name_entry["source"] == "batch_1"
        assert city_entry["source"] == "batch_1.address"

    def test_multiple_nested_models(self) -> None:
        """Test with multiple nested models."""
        address = Address(street="123 Main", city="NYC")
        contact = Contact(email="john@example.com")
        person = Person(name="John", address=address, contact=contact)

        person.address.add_error("street", "Invalid street")
        person.contact.add_error("email", "Invalid email")

        log = person.audit_log_recursive()

        assert len(log) == 2

        sources = {entry.get("source", "") for entry in log}
        assert "address" in sources
        assert "contact" in sources

    def test_list_of_nested_models(self) -> None:
        """Test with a list of nested models."""
        contact1 = Contact(email="a@example.com")
        contact2 = Contact(email="b@example.com")
        person = PersonWithContacts(name="John", contacts=[contact1, contact2])

        person.contacts[0].add_error("email", "Error 1")
        person.contacts[1].add_error("email", "Error 2")

        log = person.audit_log_recursive()

        assert len(log) == 2

        sources = {entry.get("source", "") for entry in log}
        assert "contacts[0]" in sources
        assert "contacts[1]" in sources

    def test_list_with_source_prefix(self) -> None:
        """Test list of nested models with source prefix."""
        contact = Contact(email="a@example.com")
        person = PersonWithContacts(name="John", contacts=[contact])

        person.contacts[0].add_error("email", "Error")

        log = person.audit_log_recursive(source="import")

        assert log[0]["source"] == "import.contacts[0]"

    def test_deeply_nested_models(self) -> None:
        """Test deeply nested model structure."""
        address = Address(street="123 Main", city="NYC")
        employee1 = Person(name="Alice", address=Address(street="456 Oak", city="LA"))
        employee2 = Person(name="Bob")
        company = Company(name="Acme", address=address, employees=[employee1, employee2])

        company.address.add_error("city", "Company city error")
        company.employees[0].add_error("name", "Employee name error")
        company.employees[0].address.add_error("street", "Employee address error")

        log = company.audit_log_recursive(source="company_import")

        assert len(log) == 3

        # Check all sources are correctly qualified
        sources = {entry.get("source", "") for entry in log}
        assert "company_import.address" in sources
        assert "company_import.employees[0]" in sources
        assert "company_import.employees[0].address" in sources

    def test_sorted_by_timestamp(self) -> None:
        """Test that results are sorted by timestamp."""
        address = Address(street="123 Main", city="NYC")
        person = Person(name="John", address=address)

        # Add entries with small delays
        person.add_error("name", "First")
        time.sleep(0.01)
        person.address.add_error("city", "Second")
        time.sleep(0.01)
        person.add_error("age", "Third")

        log = person.audit_log_recursive()

        timestamps = [e["timestamp"] for e in log]
        assert timestamps == sorted(timestamps)

    def test_none_nested_model_ignored(self) -> None:
        """Test that None nested models are safely ignored."""
        person = Person(name="John", address=None, contact=None)
        person.add_error("name", "Error")

        log = person.audit_log_recursive()

        assert len(log) == 1

    def test_empty_list_ignored(self) -> None:
        """Test that empty lists are safely handled."""
        person = PersonWithContacts(name="John", contacts=[])
        person.add_error("name", "Error")

        log = person.audit_log_recursive()

        assert len(log) == 1


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestAuditLogRecursiveProperties:
    """Property-based tests for audit_log_recursive."""

    @given(
        parent_errors=st.integers(min_value=0, max_value=5),
        child_errors=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=30)
    def test_total_count_correct(self, parent_errors: int, child_errors: int) -> None:
        """Test that total entry count equals sum of parent and child entries."""
        address = Address(street="123 Main", city="NYC")
        person = Person(name="John", address=address)

        for i in range(parent_errors):
            person.add_error(f"field_{i}", f"error_{i}")

        for i in range(child_errors):
            person.address.add_error(f"child_field_{i}", f"child_error_{i}")

        log = person.audit_log_recursive()

        assert len(log) == parent_errors + child_errors

    @given(
        num_contacts=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=20)
    def test_list_entries_all_included(self, num_contacts: int) -> None:
        """Test that all list item entries are included."""
        contacts = [Contact(email=f"contact{i}@example.com") for i in range(num_contacts)]
        person = PersonWithContacts(name="John", contacts=contacts)

        for contact in person.contacts:
            contact.add_error("email", "Invalid")

        log = person.audit_log_recursive()

        assert len(log) == num_contacts

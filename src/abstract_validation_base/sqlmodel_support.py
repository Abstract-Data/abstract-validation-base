"""SQLModel integration for ValidationBase.

Provides ValidatedRecord, a base class that auto-generates SQLModel
database models from validation model definitions.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import ConfigDict
from sqlmodel import Field, SQLModel
from sqlmodel.main import SQLModelMetaclass

from abstract_validation_base.base import ValidationBase

__all__ = ["ValidatedRecord"]


def _make_sqlmodel_class(
    name: str,
    table_name: str,
    field_definitions: dict[str, tuple[Any, Any]],
) -> type[SQLModel]:
    """Create a SQLModel class with proper field registration.

    Uses SQLModelMetaclass directly to ensure proper processing.

    Args:
        name: Name for the generated class.
        table_name: Database table name.
        field_definitions: Dict of field_name -> (annotation, default_value).

    Returns:
        A properly configured SQLModel class.
    """
    # Build namespace with annotations and defaults
    namespace: dict[str, Any] = {
        "__module__": __name__,
        "__qualname__": name,
        "__tablename__": table_name,
        "__annotations__": {},
    }

    for field_name, (annotation, default) in field_definitions.items():
        namespace["__annotations__"][field_name] = annotation
        if default is not ...:
            namespace[field_name] = default

    # Create the class using SQLModelMetaclass directly
    # The table=True keyword tells SQLModel to create a table model
    model = SQLModelMetaclass(
        name,
        (SQLModel,),
        namespace,
        table=True,
    )

    return model  # type: ignore[return-value]


class ValidatedRecord(ValidationBase):
    """Base class that auto-generates a SQLModel from field definitions.

    Define your fields once and get both a ValidationBase model (with full
    process logging) and a SQLModel (for database persistence).

    Example:
        class UserRecord(ValidatedRecord, table_name="users"):
            email: str
            name: str
            age: int

        # Create and validate
        user = UserRecord(email="john@example.com", name="John", age=30)
        user.add_cleaning_process("name", "  john  ", "John", "Trimmed whitespace")

        # Convert to DB model
        db_user = user.to_db()
        session.add(db_user)
        session.commit()

        # Access the auto-generated SQLModel class
        UserDB = UserRecord.db_model()
        users = session.exec(select(UserDB)).all()
    """

    model_config = ConfigDict(extra="ignore")

    __table_name__: ClassVar[str | None] = None
    __db_model__: ClassVar[type[SQLModel] | None] = None
    __db_model_created__: ClassVar[bool] = False

    def __init_subclass__(cls, table_name: str | None = None, **kwargs: Any) -> None:
        """Store table_name for lazy DB model creation."""
        super().__init_subclass__(**kwargs)

        if table_name:
            cls.__table_name__ = table_name
            cls.__db_model__ = None
            cls.__db_model_created__ = False

    @classmethod
    def _ensure_db_model(cls) -> None:
        """Lazily create the DB model on first access."""
        if cls.__table_name__ and not cls.__db_model_created__:
            cls.__db_model__ = cls._create_db_model()
            cls.__db_model_created__ = True

    @classmethod
    def _create_db_model(cls) -> type[SQLModel]:
        """Dynamically create a SQLModel class from field definitions.

        Returns:
            A new SQLModel class with matching fields and an auto-increment id.
        """
        from pydantic_core import PydanticUndefined

        # Build field definitions: field_name -> (annotation, default)
        field_definitions: dict[str, tuple[Any, Any]] = {
            "id": (int | None, Field(default=None, primary_key=True)),
        }

        # Copy fields from the validation model (excluding process_log)
        for name, info in cls.model_fields.items():
            if name == "process_log":
                continue

            annotation = info.annotation

            # Determine the default value
            # Note: PydanticUndefined means no default was provided
            if info.default is not PydanticUndefined and info.default is not None:
                default = info.default
            elif info.default_factory is not None:
                # The factory is already callable, just pass it through
                # Type ignore needed because FieldInfo.default_factory has a broader type
                default = Field(default_factory=info.default_factory)  # type: ignore[arg-type]
            elif info.is_required():
                default = ...  # Required field (Ellipsis)
            else:
                default = None

            field_definitions[name] = (annotation, default)

        return _make_sqlmodel_class(
            name=f"{cls.__name__}DB",
            table_name=cls.__table_name__ or cls.__name__.lower(),
            field_definitions=field_definitions,
        )

    def to_db(self, **overrides: Any) -> SQLModel:
        """Convert this validated record to a database model instance.

        Args:
            **overrides: Field values to override in the DB model.
                Useful for setting computed fields or overriding values.

        Returns:
            An instance of the auto-generated SQLModel.

        Raises:
            ValueError: If the class was not created with a table_name.

        Example:
            user = UserRecord(email="test@example.com", name="Test", age=25)
            db_user = user.to_db()
            db_user_with_id = user.to_db(id=123)
        """
        # Ensure DB model is created (lazy initialization)
        self.__class__._ensure_db_model()

        if self.__db_model__ is None:
            raise ValueError(
                f"{self.__class__.__name__} was not created with table_name. "
                f"Use: class {self.__class__.__name__}(ValidatedRecord, table_name='...')"
            )

        # Map fields from validation model to DB model
        db_model_fields = self.__db_model__.model_fields
        kwargs = {
            field: getattr(self, field)
            for field in self.__class__.model_fields
            if field != "process_log" and field in db_model_fields
        }
        kwargs.update(overrides)

        return self.__db_model__(**kwargs)

    @classmethod
    def db_model(cls) -> type[SQLModel]:
        """Get the auto-generated SQLModel class.

        Returns:
            The SQLModel class associated with this validation model.

        Raises:
            ValueError: If the class was not created with a table_name.

        Example:
            UserDB = UserRecord.db_model()
            users = session.exec(select(UserDB)).all()
        """
        # Ensure DB model is created (lazy initialization)
        cls._ensure_db_model()

        if cls.__db_model__ is None:
            raise ValueError(
                f"{cls.__name__} has no associated DB model. "
                f"Define with: class {cls.__name__}(ValidatedRecord, table_name='...')"
            )
        return cls.__db_model__

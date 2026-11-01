from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING

from sqlalchemy import DateTime
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column


def _to_snake_case(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy ORM models.

    Includes automatic plural/snake_case table naming convention.
    """

    if TYPE_CHECKING:
        # Declarative resolves this directive to a class-level string. Expose
        # that resolved type so explicit names can override it in subclasses.
        __tablename__: str
    else:
        @declared_attr.directive
        def __tablename__(cls) -> str:
            name = _to_snake_case(cls.__name__)
            return f"{name}s" if not name.endswith("s") else name


class TimestampMixin:
    """Mixin providing created_at and updated_at timezone-aware timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

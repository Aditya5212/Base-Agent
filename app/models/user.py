from typing import TYPE_CHECKING, Any
from sqlalchemy import Boolean, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.thread import Thread, ThreadAsset


class User(Base, TimestampMixin):
    """User account entity.
    
    Default root user is seeded automatically for production setup without requiring
    an external authentication provider initially.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(50), default="user", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    threads: Mapped[list["Thread"]] = relationship(
        "Thread",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    assets: Mapped[list["ThreadAsset"]] = relationship(
        "ThreadAsset",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

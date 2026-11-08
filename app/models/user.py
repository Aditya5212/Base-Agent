from typing import TYPE_CHECKING, Any
from sqlalchemy import Boolean, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.agent import Agent
    from app.models.document import Document
    from app.models.structured_schema import StructuredSchema
    from app.models.thread import Thread, ThreadAsset
    from app.models.tool import Tool
    from app.models.workflow import Workflow, WorkflowRun


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
    agents: Mapped[list["Agent"]] = relationship(
        "Agent",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    structured_schemas: Mapped[list["StructuredSchema"]] = relationship(
        "StructuredSchema",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    workflows: Mapped[list["Workflow"]] = relationship(
        "Workflow",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    tools: Mapped[list["Tool"]] = relationship(
        "Tool",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    workflow_runs: Mapped[list["WorkflowRun"]] = relationship(
        "WorkflowRun",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


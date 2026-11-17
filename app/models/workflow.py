from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
import uuid

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.user import User


class Workflow(Base, TimestampMixin):
    """Workflow entity for the Agent Builder graph orchestration engine.

    Stores graph topology (nodes and edges), validation schemas, and versioning.
    """

    __tablename__ = "workflows"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    nodes: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    edges: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    input_schema: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    output_schema: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    recursion_limit: Mapped[int] = mapped_column(Integer, default=25, nullable=False)
    is_published: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="workflows")
    runs: Mapped[list["WorkflowRun"]] = relationship(
        "WorkflowRun",
        back_populates="workflow",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class WorkflowRun(Base, TimestampMixin):
    """Execution run record for a compiled workflow.

    Tracks execution status, inputs, outputs, errors, and granular node step traces.
    """

    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    workflow_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    checkpoint_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    definition_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="PENDING", nullable=False, index=True)
    control_request: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    control_set_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    budget_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    consumed_seconds: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    execution_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    input_data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    output_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_log: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    workflow: Mapped["Workflow"] = relationship("Workflow", back_populates="runs")
    user: Mapped["User"] = relationship("User", back_populates="workflow_runs")
    events: Mapped[list["WorkflowRunEvent"]] = relationship(
        "WorkflowRunEvent",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="WorkflowRunEvent.sequence",
    )


class WorkflowRunEvent(Base):
    """Granular execution trace event for a workflow run."""

    __tablename__ = "workflow_run_events"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("workflow_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    invocation_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    node_id: Mapped[str] = mapped_column(String(255), nullable=False)
    node_type: Mapped[str] = mapped_column(String(50), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    output_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    model_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tool_invocation_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    run: Mapped["WorkflowRun"] = relationship("WorkflowRun", back_populates="events")

    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_workflow_run_events_run_sequence"),
        Index("ix_workflow_run_events_run_node", "run_id", "node_id"),
        Index("ix_workflow_run_events_run_sequence", "run_id", "sequence"),
    )

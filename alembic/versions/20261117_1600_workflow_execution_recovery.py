"""Add execution generation, next_event_sequence, workflow_dispatches, and workflow_tool_invocations.

Revision ID: workflow_execution_recovery
Revises: workflow_run_durability
Create Date: 2026-11-17 16:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "workflow_execution_recovery"
down_revision = "workflow_run_durability"
branch_labels = None
depends_on = None


def upgrade():
    # 1. Add generation and next_event_sequence to workflow_runs
    op.add_column("workflow_runs", sa.Column("generation", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("workflow_runs", sa.Column("next_event_sequence", sa.Integer(), nullable=False, server_default="1"))

    # 2. Create workflow_dispatches table
    op.create_table(
        "workflow_dispatches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflow_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("state", sa.String(length=50), nullable=False, server_default="PENDING"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_workflow_dispatches_run_id", "workflow_dispatches", ["run_id"])
    op.create_index("ix_workflow_dispatches_state", "workflow_dispatches", ["state"])
    op.create_index("ix_workflow_dispatches_next_retry_at", "workflow_dispatches", ["next_retry_at"])

    # 3. Create workflow_tool_invocations table
    op.create_table(
        "workflow_tool_invocations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflow_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("logical_occurrence", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("idempotency_key", sa.String(length=512), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="prepared"),
        sa.Column("tool_name", sa.String(length=100), nullable=False),
        sa.Column("arguments_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("result_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("idempotency_key", name="uq_workflow_tool_invocations_idempotency_key"),
    )
    op.create_index("ix_workflow_tool_invocations_run_id", "workflow_tool_invocations", ["run_id"])
    op.create_index("ix_workflow_tool_invocations_node_id", "workflow_tool_invocations", ["node_id"])
    op.create_index("ix_workflow_tool_invocations_idempotency_key", "workflow_tool_invocations", ["idempotency_key"])
    op.create_index(
        "ix_workflow_tool_invocations_run_node",
        "workflow_tool_invocations",
        ["run_id", "node_id", "logical_occurrence"],
    )


def downgrade():
    op.drop_index("ix_workflow_tool_invocations_run_node", table_name="workflow_tool_invocations")
    op.drop_index("ix_workflow_tool_invocations_idempotency_key", table_name="workflow_tool_invocations")
    op.drop_index("ix_workflow_tool_invocations_node_id", table_name="workflow_tool_invocations")
    op.drop_index("ix_workflow_tool_invocations_run_id", table_name="workflow_tool_invocations")
    op.drop_table("workflow_tool_invocations")

    op.drop_index("ix_workflow_dispatches_next_retry_at", table_name="workflow_dispatches")
    op.drop_index("ix_workflow_dispatches_state", table_name="workflow_dispatches")
    op.drop_index("ix_workflow_dispatches_run_id", table_name="workflow_dispatches")
    op.drop_table("workflow_dispatches")

    op.drop_column("workflow_runs", "next_event_sequence")
    op.drop_column("workflow_runs", "generation")

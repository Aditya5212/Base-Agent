"""Create workflow_run_events table for granular execution history and inspection.

Revision ID: workflow_run_events
Revises: document_thread_scoping
Create Date: 2026-11-17 12:30:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "workflow_run_events"
down_revision = "document_thread_scoping"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "workflow_run_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("workflow_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("invocation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("node_type", sa.String(length=50), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("input_summary", postgresql.JSONB(), nullable=True),
        sa.Column("output_summary", postgresql.JSONB(), nullable=True),
        sa.Column("model_id", sa.String(length=100), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("tool_name", sa.String(length=100), nullable=True),
        sa.Column("tool_invocation_id", sa.String(length=255), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )

    op.create_unique_constraint(
        "uq_workflow_run_events_run_sequence",
        "workflow_run_events",
        ["run_id", "sequence"],
    )
    op.create_index("ix_workflow_run_events_run_node", "workflow_run_events", ["run_id", "node_id"])
    op.create_index("ix_workflow_run_events_run_sequence", "workflow_run_events", ["run_id", "sequence"])


def downgrade():
    op.drop_index("ix_workflow_run_events_run_sequence", table_name="workflow_run_events")
    op.drop_index("ix_workflow_run_events_run_node", table_name="workflow_run_events")
    op.drop_constraint("uq_workflow_run_events_run_sequence", "workflow_run_events", type_="unique")
    op.drop_table("workflow_run_events")

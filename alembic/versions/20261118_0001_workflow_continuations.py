"""Persist queued continuations and unique logical tool occurrences.

Revision ID: workflow_continuations
Revises: workflow_execution_recovery
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "workflow_continuations"
down_revision = "workflow_execution_recovery"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("workflow_dispatches", sa.Column("resume_payload", postgresql.JSONB(), nullable=True))
    op.create_unique_constraint(
        "uq_tool_invocation_occurrence", "workflow_tool_invocations",
        ["run_id", "node_id", "logical_occurrence"],
    )


def downgrade():
    op.drop_constraint("uq_tool_invocation_occurrence", "workflow_tool_invocations", type_="unique")
    op.drop_column("workflow_dispatches", "resume_payload")

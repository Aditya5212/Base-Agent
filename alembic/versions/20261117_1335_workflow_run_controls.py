"""Add control_request and control_set_at columns to workflow_runs.

Revision ID: workflow_run_controls
Revises: workflow_run_events
Create Date: 2026-11-17 13:35:00
"""

from alembic import op
import sqlalchemy as sa

revision = "workflow_run_controls"
down_revision = "workflow_run_events"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("workflow_runs", sa.Column("control_request", sa.String(length=50), nullable=True))
    op.add_column("workflow_runs", sa.Column("control_set_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_workflow_runs_control_request", "workflow_runs", ["control_request"])


def downgrade():
    op.drop_index("ix_workflow_runs_control_request", table_name="workflow_runs")
    op.drop_column("workflow_runs", "control_set_at")
    op.drop_column("workflow_runs", "control_request")

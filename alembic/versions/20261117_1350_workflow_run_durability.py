"""Add durability, budget, and worker lease columns to workflow_runs.

Revision ID: workflow_run_durability
Revises: workflow_run_controls
Create Date: 2026-11-17 13:50:00
"""

from alembic import op
import sqlalchemy as sa

revision = "workflow_run_durability"
down_revision = "workflow_run_controls"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("workflow_runs", sa.Column("budget_seconds", sa.Integer(), nullable=True))
    op.add_column("workflow_runs", sa.Column("consumed_seconds", sa.Float(), nullable=False, server_default="0.0"))
    op.add_column("workflow_runs", sa.Column("execution_started_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("workflow_runs", sa.Column("worker_id", sa.String(length=128), nullable=True))
    op.add_column("workflow_runs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_workflow_runs_lease_expires_at", "workflow_runs", ["lease_expires_at"])


def downgrade():
    op.drop_index("ix_workflow_runs_lease_expires_at", table_name="workflow_runs")
    op.drop_column("workflow_runs", "lease_expires_at")
    op.drop_column("workflow_runs", "worker_id")
    op.drop_column("workflow_runs", "execution_started_at")
    op.drop_column("workflow_runs", "consumed_seconds")
    op.drop_column("workflow_runs", "budget_seconds")

"""Add partial unique index on agents for single default agent per user.

Revision ID: agent_unique_default
Revises: audit_execution_identity
Create Date: 2026-11-07 13:20:00
"""

from alembic import op
import sqlalchemy as sa

revision = "agent_unique_default"
down_revision = "audit_execution_identity"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "uq_agents_user_default",
        "agents",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("is_default = TRUE"),
    )


def downgrade():
    op.drop_index("uq_agents_user_default", table_name="agents")


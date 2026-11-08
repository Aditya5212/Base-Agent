"""Add foreign key constraint on documents.user_id -> users.id with CASCADE delete.

Revision ID: document_user_fk_cascade
Revises: agent_unique_default
Create Date: 2026-11-08 11:20:00
"""

from alembic import op
import sqlalchemy as sa

revision = "document_user_fk_cascade"
down_revision = "agent_unique_default"
branch_labels = None
depends_on = None


def upgrade():
    op.create_foreign_key(
        "fk_documents_user_id_users",
        "documents",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade():
    op.drop_constraint("fk_documents_user_id_users", "documents", type_="foreignkey")


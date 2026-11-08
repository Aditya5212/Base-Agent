"""Add thread_id to documents table with foreign key to threads and CASCADE delete.

Revision ID: document_thread_scoping
Revises: document_user_fk_cascade
Create Date: 2026-11-08 17:30:00
"""

from alembic import op
import sqlalchemy as sa

revision = "document_thread_scoping"
down_revision = "document_user_fk_cascade"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("documents", sa.Column("thread_id", sa.String(length=128), nullable=True))
    op.create_index("ix_documents_thread_id", "documents", ["thread_id"])
    op.create_foreign_key(
        "fk_documents_thread_id_threads",
        "documents",
        "threads",
        ["thread_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade():
    op.drop_constraint("fk_documents_thread_id_threads", "documents", type_="foreignkey")
    op.drop_index("ix_documents_thread_id", table_name="documents")
    op.drop_column("documents", "thread_id")


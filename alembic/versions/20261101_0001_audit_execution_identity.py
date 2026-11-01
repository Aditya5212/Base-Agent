"""Pin workflow execution definitions and isolate checkpoint identities."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "audit_execution_identity"
down_revision = "69df16e3a58a"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("workflow_runs", sa.Column("checkpoint_id", sa.String(255), nullable=True))
    op.add_column("workflow_runs", sa.Column("definition_snapshot", postgresql.JSONB(), nullable=True))
    op.add_column("documents", sa.Column("indexing_config", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")))
    op.add_column("documents", sa.Column("provider_status", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")))


def downgrade():
    op.drop_column("documents", "provider_status")
    op.drop_column("documents", "indexing_config")
    op.drop_column("workflow_runs", "definition_snapshot")
    op.drop_column("workflow_runs", "checkpoint_id")

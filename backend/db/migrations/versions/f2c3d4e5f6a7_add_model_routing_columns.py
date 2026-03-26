"""Add model routing columns to query_logs

Revision ID: f2c3d4e5f6a7
Revises: a3f8b1c2d4e5
Create Date: 2026-03-25 23:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2c3d4e5f6a7'
down_revision: Union[str, None] = 'a3f8b1c2d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.add_column('query_logs', sa.Column('model_used', sa.String(length=50), nullable=True))
    op.add_column('query_logs', sa.Column('mode_used', sa.String(length=50), nullable=True))
    op.add_column('query_logs', sa.Column('routing_reason', sa.String(), nullable=True))

def downgrade() -> None:
    op.drop_column('query_logs', 'routing_reason')
    op.drop_column('query_logs', 'mode_used')
    op.drop_column('query_logs', 'model_used')

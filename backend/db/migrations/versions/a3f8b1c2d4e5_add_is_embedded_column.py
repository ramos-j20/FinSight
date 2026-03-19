"""Add is_embedded column to filing_metadata

Revision ID: a3f8b1c2d4e5
Revises: 1216b262dbfb
Create Date: 2026-03-13 18:39:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f8b1c2d4e5'
down_revision: Union[str, None] = '1216b262dbfb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.add_column(
        'filing_metadata',
        sa.Column('is_embedded', sa.Boolean(), nullable=False, server_default=sa.text('false')),
    )

def downgrade() -> None:
    op.drop_column('filing_metadata', 'is_embedded')

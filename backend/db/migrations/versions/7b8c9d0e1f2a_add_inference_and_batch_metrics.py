"""Add inference and batch metrics tables

Revision ID: 7b8c9d0e1f2a
Revises: f2c3d4e5f6a7
Create Date: 2026-03-26 16:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7b8c9d0e1f2a'
down_revision: Union[str, None] = 'f2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Inference metrics table
    op.create_table(
        'inference_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('query_log_id', sa.Integer(), nullable=True),
        sa.Column('model_used', sa.String(length=100), nullable=False),
        sa.Column('mode_used', sa.String(length=50), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=False),
        sa.Column('output_tokens', sa.Integer(), nullable=False),
        sa.Column('cache_read_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('cache_write_tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('latency_ms', sa.Integer(), nullable=False),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=False),
        sa.Column('caching_enabled', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_inference_metrics_query_log_id'), 'inference_metrics', ['query_log_id'], unique=False)

    # Batch job metrics table
    op.create_table(
        'batch_job_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('batch_job_id', sa.String(length=255), nullable=False),
        sa.Column('total_requests', sa.Integer(), nullable=False),
        sa.Column('succeeded', sa.Integer(), nullable=False),
        sa.Column('failed', sa.Integer(), nullable=False),
        sa.Column('input_tokens_total', sa.Integer(), nullable=False),
        sa.Column('output_tokens_total', sa.Integer(), nullable=False),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=False),
        sa.Column('estimated_cost_without_batch_usd', sa.Float(), nullable=False),
        sa.Column('savings_usd', sa.Float(), nullable=False),
        sa.Column('savings_pct', sa.Float(), nullable=False),
        sa.Column('duration_seconds', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_batch_job_metrics_batch_job_id'), 'batch_job_metrics', ['batch_job_id'], unique=False)

def downgrade() -> None:
    op.drop_index(op.f('ix_batch_job_metrics_batch_job_id'), table_name='batch_job_metrics')
    op.drop_table('batch_job_metrics')
    op.drop_index(op.f('ix_inference_metrics_query_log_id'), table_name='inference_metrics')
    op.drop_table('inference_metrics')

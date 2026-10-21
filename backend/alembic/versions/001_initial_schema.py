"""initial_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-21 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')

    # Candidate Profiles Table
    op.create_table(
        'candidate_profiles',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('phone', sa.String(length=50), nullable=True),
        sa.Column('location', sa.String(length=255), nullable=True),
        sa.Column('visa_status', sa.String(length=100), nullable=True),
        sa.Column('target_role', sa.String(length=255), nullable=True),
        sa.Column('years_of_experience', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('skills', sa.JSON(), nullable=False),
        sa.Column('work_history', sa.JSON(), nullable=False),
        sa.Column('education', sa.JSON(), nullable=False),
        sa.Column('certifications', sa.JSON(), nullable=False),
        sa.Column('raw_text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(768), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_candidate_profiles_email'), 'candidate_profiles', ['email'], unique=False)

    # Job Postings Table
    op.create_table(
        'job_postings',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('company', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('requirements', sa.Text(), nullable=False),
        sa.Column('location', sa.String(length=255), nullable=True),
        sa.Column('remote_policy', sa.String(length=50), nullable=False, server_default='Hybrid'),
        sa.Column('min_salary', sa.Integer(), nullable=True),
        sa.Column('max_salary', sa.Integer(), nullable=True),
        sa.Column('salary_currency', sa.String(length=10), nullable=False, server_default='USD'),
        sa.Column('min_years_experience', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('required_skills', sa.JSON(), nullable=False),
        sa.Column('preferred_skills', sa.JSON(), nullable=False),
        sa.Column('visa_sponsorship', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('embedding', Vector(768), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_job_postings_title'), 'job_postings', ['title'], unique=False)

    # Match Results Table
    op.create_table(
        'match_results',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('candidate_id', sa.String(length=36), nullable=False),
        sa.Column('job_id', sa.String(length=36), nullable=False),
        sa.Column('overall_score', sa.Float(), nullable=False),
        sa.Column('skills_score', sa.Float(), nullable=False),
        sa.Column('experience_score', sa.Float(), nullable=False),
        sa.Column('vector_score', sa.Float(), nullable=False),
        sa.Column('hard_constraints_score', sa.Float(), nullable=False),
        sa.Column('hard_constraints_passed', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('matching_skills', sa.JSON(), nullable=False),
        sa.Column('missing_must_have_skills', sa.JSON(), nullable=False),
        sa.Column('missing_nice_to_have_skills', sa.JSON(), nullable=False),
        sa.Column('recommendations', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['candidate_id'], ['candidate_profiles.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['job_id'], ['job_postings.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_match_results_candidate_id'), 'match_results', ['candidate_id'], unique=False)
    op.create_index(op.f('ix_match_results_job_id'), 'match_results', ['job_id'], unique=False)


def downgrade() -> None:
    op.drop_table('match_results')
    op.drop_table('job_postings')
    op.drop_table('candidate_profiles')

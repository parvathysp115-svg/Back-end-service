from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy import String, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from app.models.base import generate_uuid


class MatchResult(Base):
    __tablename__ = "match_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    candidate_id: Mapped[str] = mapped_column(String(36), ForeignKey("candidate_profiles.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("job_postings.id", ondelete="CASCADE"), index=True)

    # Dimensional scores (0 - 100)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    skills_score: Mapped[float] = mapped_column(Float, nullable=False)
    experience_score: Mapped[float] = mapped_column(Float, nullable=False)
    vector_score: Mapped[float] = mapped_column(Float, nullable=False)
    hard_constraints_score: Mapped[float] = mapped_column(Float, nullable=False)
    hard_constraints_passed: Mapped[bool] = mapped_column(Boolean, default=True)

    # Detailed Gap Analysis
    matching_skills: Mapped[List[str]] = mapped_column(JSON, default=list)
    missing_must_have_skills: Mapped[List[str]] = mapped_column(JSON, default=list)
    missing_nice_to_have_skills: Mapped[List[str]] = mapped_column(JSON, default=list)

    # Generated feedback & recommendation notes
    recommendations: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    candidate = relationship("CandidateProfile", back_populates="matches")
    job = relationship("JobPosting", back_populates="matches")

    def __repr__(self) -> str:
        return f"<MatchResult(candidate='{self.candidate_id}', job='{self.job_id}', score={self.overall_score})>"

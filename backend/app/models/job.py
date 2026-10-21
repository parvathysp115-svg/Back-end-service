from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Float, Integer, Boolean, Text, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from app.models.base import generate_uuid, PortableVector


class JobPosting(Base):
    __tablename__ = "job_postings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    requirements: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    remote_policy: Mapped[str] = mapped_column(String(50), default="Hybrid")  # Remote, Hybrid, Onsite

    min_salary: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    max_salary: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    salary_currency: Mapped[str] = mapped_column(String(10), default="USD")

    min_years_experience: Mapped[float] = mapped_column(Float, default=0.0)
    required_skills: Mapped[List[str]] = mapped_column(JSON, default=list)  # Must-have skills
    preferred_skills: Mapped[List[str]] = mapped_column(JSON, default=list)  # Nice-to-have skills

    visa_sponsorship: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # 768-dimensional dense vector embedding (pgvector)
    embedding: Mapped[Optional[List[float]]] = mapped_column(PortableVector(768), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    matches = relationship("MatchResult", back_populates="job", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<JobPosting(id='{self.id}', title='{self.title}', company='{self.company}')>"

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy import String, Float, Text, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from app.models.base import generate_uuid, PortableVector


class CandidateProfile(Base):
    __tablename__ = "candidate_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    visa_status: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default="Citizen/Permanent Resident")
    target_role: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    years_of_experience: Mapped[float] = mapped_column(Float, default=0.0)

    # Structured parsed items stored as JSON
    skills: Mapped[List[str]] = mapped_column(JSON, default=list)
    work_history: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    education: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    certifications: Mapped[List[str]] = mapped_column(JSON, default=list)

    # Raw extracted text from resume
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

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
    matches = relationship("MatchResult", back_populates="candidate", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<CandidateProfile(id='{self.id}', name='{self.full_name}', exp={self.years_of_experience})>"

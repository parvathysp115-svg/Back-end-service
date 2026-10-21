import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.candidate import CandidateProfile
from app.schemas.resume_schema import (
    CandidateProfileCreate,
    CandidateProfileResponse,
    ResumeUploadResponse,
)
from app.services.parser import DocumentParser
from app.services.embeddings import embedding_service
from app.services.agents.extraction_agent import extraction_agent

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/upload",
    response_model=ResumeUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Parse Candidate Resume",
    description="Uploads a PDF, DOCX, or TXT resume, extracts structured candidate data via AI, computes 768-dim embedding, and persists to database.",
)
async def upload_resume(
    file: Optional[UploadFile] = File(None, description="Resume file (PDF, DOCX, or TXT)"),
    raw_text: Optional[str] = Form(None, description="Direct text input if not uploading file"),
    db: AsyncSession = Depends(get_db),
):
    extracted_text = ""
    filename = "resume.txt"

    if file is not None and file.filename:
        filename = file.filename
        content = await file.read()
        extracted_text, _ = DocumentParser.parse_document(content, filename)
    elif raw_text and raw_text.strip():
        extracted_text = DocumentParser.clean_text(raw_text)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either a resume file (PDF/DOCX/TXT) or raw_text must be provided.",
        )

    if not extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract readable text from the provided resume.",
        )

    # 1. Trigger Subagent 1: Ingestion & Extraction Agent
    try:
        parsed_profile: CandidateProfileCreate = await extraction_agent.extract_candidate_profile(extracted_text)
    except Exception as e:
        logger.error(f"Failed to extract candidate profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction agent failed: {str(e)}",
        )

    # 2. Compute Dense Vector Embedding (768-dim pgvector)
    try:
        embedding_vec = await embedding_service.get_embedding(extracted_text)
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}")
        embedding_vec = None

    # 3. Store Candidate Profile in Database
    db_profile = CandidateProfile(
        full_name=parsed_profile.full_name,
        email=parsed_profile.email,
        phone=parsed_profile.phone,
        location=parsed_profile.location,
        visa_status=parsed_profile.visa_status,
        target_role=parsed_profile.target_role,
        years_of_experience=parsed_profile.years_of_experience,
        skills=parsed_profile.skills,
        work_history=[w.model_dump() for w in parsed_profile.work_history],
        education=[e.model_dump() for e in parsed_profile.education],
        certifications=parsed_profile.certifications,
        raw_text=extracted_text,
        embedding=embedding_vec,
    )

    db.add(db_profile)
    await db.commit()
    await db.refresh(db_profile)

    return ResumeUploadResponse(
        message="Resume successfully processed, structured, and vector indexed.",
        candidate_id=db_profile.id,
        full_name=db_profile.full_name,
        parsed_skills_count=len(db_profile.skills),
        years_of_experience=db_profile.years_of_experience,
        embedding_indexed=embedding_vec is not None,
        profile=CandidateProfileResponse.model_validate(db_profile),
    )


@router.get(
    "/{id}",
    response_model=CandidateProfileResponse,
    summary="Get Candidate Profile by ID",
)
async def get_candidate_profile(
    id: str,
    db: AsyncSession = Depends(get_db),
):
    query = select(CandidateProfile).where(CandidateProfile.id == id)
    result = await db.execute(query)
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate profile with ID '{id}' was not found.",
        )
    return CandidateProfileResponse.model_validate(candidate)


@router.get(
    "/",
    response_model=List[CandidateProfileResponse],
    summary="List Candidate Profiles",
)
async def list_candidate_profiles(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    query = select(CandidateProfile).offset(skip).limit(limit).order_by(CandidateProfile.created_at.desc())
    result = await db.execute(query)
    profiles = result.scalars().all()
    return [CandidateProfileResponse.model_validate(p) for p in profiles]


@router.delete(
    "/{id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Candidate Profile",
)
async def delete_candidate_profile(
    id: str,
    db: AsyncSession = Depends(get_db),
):
    query = select(CandidateProfile).where(CandidateProfile.id == id)
    result = await db.execute(query)
    candidate = result.scalar_one_or_none()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate profile with ID '{id}' was not found.",
        )

    await db.delete(candidate)
    await db.commit()
    return None

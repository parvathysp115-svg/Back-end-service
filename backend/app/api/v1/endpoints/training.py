import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.candidate import CandidateProfile
from app.models.job import JobPosting
from app.schemas.coaching_schema import (
    ResumeTrainRequest,
    ResumeOptimizationResponse,
    CareerRoadmapResponse,
    CoverLetterRequest,
    CoverLetterResponse,
)
from app.services.agents.optimization_agent import optimization_agent
from app.services.agents.coaching_agent import coaching_agent

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/{id}/train",
    response_model=ResumeOptimizationResponse,
    summary="Optimize & Train Candidate Resume",
    description="Evaluates candidate resume against target role benchmarks or specific job posting, identifies weak bullets (passive voice, missing metrics), computes ATS score, and generates STAR rewrites.",
)
async def train_resume(
    id: str,
    train_request: Optional[ResumeTrainRequest] = None,
    db: AsyncSession = Depends(get_db),
):
    # 1. Fetch Candidate Profile
    cand_query = select(CandidateProfile).where(CandidateProfile.id == id)
    cand_res = await db.execute(cand_query)
    candidate = cand_res.scalar_one_or_none()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate profile with ID '{id}' was not found.",
        )

    # 2. Extract context
    target_role = (train_request.target_role if train_request else None) or candidate.target_role or "Senior Software Engineer"
    job_description = None

    if train_request and train_request.job_id:
        job_query = select(JobPosting).where(JobPosting.id == train_request.job_id)
        job_res = await db.execute(job_query)
        job = job_res.scalar_one_or_none()
        if job:
            job_description = f"{job.title} at {job.company}\n\nDescription: {job.description}\n\nRequirements: {job.requirements}"
            target_role = job.title

    cand_dict = {
        "id": candidate.id,
        "full_name": candidate.full_name,
        "target_role": target_role,
        "years_of_experience": candidate.years_of_experience,
        "skills": candidate.skills,
        "work_history": candidate.work_history,
        "education": candidate.education,
    }

    # 3. Trigger Subagent 2: Optimization & Training Agent
    try:
        optimization_result = await optimization_agent.optimize_resume(
            cand_dict,
            target_role=target_role,
            job_description=job_description,
        )
        return optimization_result
    except Exception as e:
        logger.error(f"Resume training failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization agent encountered an error: {str(e)}",
        )


@router.post(
    "/{id}/coaching-roadmap",
    response_model=CareerRoadmapResponse,
    summary="Generate Tailored Career Roadmap & Interview Prep",
    description="Analyzes candidate profile against target role, creates a structured multi-week learning plan and generated mock interview questions.",
)
async def get_coaching_roadmap(
    id: str,
    target_role: Optional[str] = None,
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

    role = target_role or candidate.target_role or "Staff Backend Engineer"

    cand_dict = {
        "id": candidate.id,
        "full_name": candidate.full_name,
        "target_role": role,
        "skills": candidate.skills,
        "years_of_experience": candidate.years_of_experience,
    }

    # Common missing skills projection for upskilling
    missing_sample = ["Distributed Systems Architecture", "Advanced Query Optimization", "Event-Driven Microservices"]

    try:
        roadmap = await coaching_agent.generate_career_roadmap(
            cand_dict,
            target_role=role,
            missing_skills=missing_sample,
            experience_gap_years=max(0.0, 5.0 - candidate.years_of_experience),
        )
        return roadmap
    except Exception as e:
        logger.error(f"Career roadmap generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coaching agent encountered an error: {str(e)}",
        )


@router.post(
    "/{id}/cover-letter",
    response_model=CoverLetterResponse,
    summary="Generate Tailored Cover Letter",
    description="Synthesizes candidate background with job specifications into a high-converting cover letter.",
)
async def generate_cover_letter(
    id: str,
    request: CoverLetterRequest,
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

    target_company = request.target_company or "Target Enterprise"
    target_role = request.target_role or candidate.target_role or "Senior Backend Engineer"
    job_desc = None

    if request.job_id:
        job_q = select(JobPosting).where(JobPosting.id == request.job_id)
        job_res = await db.execute(job_q)
        job = job_res.scalar_one_or_none()
        if job:
            target_company = job.company
            target_role = job.title
            job_desc = f"{job.description}\n\nRequirements:\n{job.requirements}"

    cand_dict = {
        "id": candidate.id,
        "full_name": candidate.full_name,
        "skills": candidate.skills,
        "years_of_experience": candidate.years_of_experience,
    }

    try:
        cover_letter = await coaching_agent.generate_cover_letter(
            cand_dict,
            target_company=target_company,
            target_role=target_role,
            job_description=job_desc,
            tone=request.tone,
        )
        return cover_letter
    except Exception as e:
        logger.error(f"Cover letter generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coaching agent error: {str(e)}",
        )

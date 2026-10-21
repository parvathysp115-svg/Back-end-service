import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.job import JobPosting
from app.schemas.job_schema import (
    JobPostingCreate,
    JobPostingResponse,
)
from app.services.embeddings import embedding_service
from app.services.agents.extraction_agent import extraction_agent

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/",
    response_model=JobPostingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create & Index New Job Vacancy",
    description="Creates a structured job posting, parses missing skills/requirements, computes 768-dim vector embedding, and indexes it in PostgreSQL pgvector.",
)
async def create_job_posting(
    job_in: JobPostingCreate,
    db: AsyncSession = Depends(get_db),
):
    # If required_skills are empty or unstructured, run ExtractionAgent to enrich
    if not job_in.required_skills and (job_in.description or job_in.requirements):
        try:
            combined_text = f"{job_in.description}\n\nRequirements:\n{job_in.requirements}"
            extracted_jd = await extraction_agent.extract_job_posting(
                combined_text,
                title_hint=job_in.title,
                company_hint=job_in.company,
            )
            job_in.required_skills = extracted_jd.required_skills
            job_in.preferred_skills = extracted_jd.preferred_skills
            if not job_in.min_years_experience:
                job_in.min_years_experience = extracted_jd.min_years_experience
        except Exception as e:
            logger.warning(f"Failed to auto-enrich job skills: {e}")

    # Generate 768-dim dense vector embedding
    embedding_text = f"Job Title: {job_in.title}\nCompany: {job_in.company}\nDescription: {job_in.description}\nRequirements: {job_in.requirements}\nSkills: {', '.join(job_in.required_skills + job_in.preferred_skills)}"
    try:
        embedding_vec = await embedding_service.get_embedding(embedding_text)
    except Exception as e:
        logger.warning(f"Job embedding generation failed: {e}")
        embedding_vec = None

    db_job = JobPosting(
        title=job_in.title,
        company=job_in.company,
        description=job_in.description,
        requirements=job_in.requirements,
        location=job_in.location,
        remote_policy=job_in.remote_policy,
        min_salary=job_in.min_salary,
        max_salary=job_in.max_salary,
        salary_currency=job_in.salary_currency,
        min_years_experience=job_in.min_years_experience,
        required_skills=job_in.required_skills,
        preferred_skills=job_in.preferred_skills,
        visa_sponsorship=job_in.visa_sponsorship,
        is_active=True,
        embedding=embedding_vec,
    )

    db.add(db_job)
    await db.commit()
    await db.refresh(db_job)

    return JobPostingResponse.model_validate(db_job)


@router.get(
    "/{id}",
    response_model=JobPostingResponse,
    summary="Get Job Vacancy by ID",
)
async def get_job_posting(
    id: str,
    db: AsyncSession = Depends(get_db),
):
    query = select(JobPosting).where(JobPosting.id == id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job posting with ID '{id}' was not found.",
        )
    return JobPostingResponse.model_validate(job)


@router.get(
    "/",
    response_model=List[JobPostingResponse],
    summary="List Job Vacancies with Filters",
)
async def list_job_postings(
    title: Optional[str] = Query(None, description="Filter by title keyword"),
    remote_policy: Optional[str] = Query(None, description="Filter by Remote, Hybrid, or Onsite"),
    visa_sponsorship: Optional[bool] = Query(None, description="Filter by visa sponsorship"),
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    query = select(JobPosting).where(JobPosting.is_active == True)

    if title:
        query = query.where(JobPosting.title.ilike(f"%{title}%"))
    if remote_policy:
        query = query.where(JobPosting.remote_policy.ilike(f"%{remote_policy}%"))
    if visa_sponsorship is not None:
        query = query.where(JobPosting.visa_sponsorship == visa_sponsorship)

    query = query.offset(skip).limit(limit).order_by(JobPosting.created_at.desc())
    result = await db.execute(query)
    jobs = result.scalars().all()

    return [JobPostingResponse.model_validate(j) for j in jobs]


@router.delete(
    "/{id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Job Vacancy",
)
async def delete_job_posting(
    id: str,
    db: AsyncSession = Depends(get_db),
):
    query = select(JobPosting).where(JobPosting.id == id)
    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job posting with ID '{id}' was not found.",
        )

    await db.delete(job)
    await db.commit()
    return None

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.candidate import CandidateProfile
from app.models.job import JobPosting
from app.models.match import MatchResult
from app.schemas.match_schema import (
    MatchResumeResponse,
    MatchJobResponse,
    JobMatchCandidateItem,
    CandidateMatchJobItem,
    DirectCompareRequest,
    DirectCompareResponse,
)
from app.services.agents.matching_agent import matching_agent

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/match-resume/{resume_id}",
    response_model=MatchResumeResponse,
    summary="Match Candidate to Vacancies (Top-K)",
    description="Calculates multi-dimensional hybrid scores (vector, skills, experience, hard constraints) across all vacancies for a candidate.",
)
async def match_resume_to_jobs(
    resume_id: str,
    top_k: int = Query(10, ge=1, le=50, description="Number of top vacancies to return"),
    min_score: float = Query(0.0, ge=0.0, le=100.0, description="Minimum overall score cutoff"),
    db: AsyncSession = Depends(get_db),
):
    # 1. Fetch Candidate
    cand_query = select(CandidateProfile).where(CandidateProfile.id == resume_id)
    cand_res = await db.execute(cand_query)
    candidate = cand_res.scalar_one_or_none()

    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate profile with ID '{resume_id}' was not found.",
        )

    # 2. Fetch Active Jobs
    jobs_query = select(JobPosting).where(JobPosting.is_active == True)
    jobs_res = await db.execute(jobs_query)
    jobs = jobs_res.scalars().all()

    if not jobs:
        return MatchResumeResponse(
            candidate_id=candidate.id,
            candidate_name=candidate.full_name,
            total_jobs_evaluated=0,
            top_matches=[],
        )

    cand_dict = {
        "id": candidate.id,
        "full_name": candidate.full_name,
        "target_role": candidate.target_role,
        "years_of_experience": candidate.years_of_experience,
        "skills": candidate.skills,
        "visa_status": candidate.visa_status,
        "location": candidate.location,
    }

    evaluated_matches: List[JobMatchCandidateItem] = []

    for job in jobs:
        job_dict = {
            "id": job.id,
            "title": job.title,
            "company": job.company,
            "min_years_experience": job.min_years_experience,
            "required_skills": job.required_skills,
            "preferred_skills": job.preferred_skills,
            "visa_sponsorship": job.visa_sponsorship,
            "location": job.location,
            "remote_policy": job.remote_policy,
        }

        overall, breakdown, matching_skills, missing_must, missing_pref, rec = (
            matching_agent.compute_hybrid_match(
                cand_dict,
                job_dict,
                candidate_embedding=candidate.embedding,
                job_embedding=job.embedding,
            )
        )

        if overall >= min_score:
            item = JobMatchCandidateItem(
                job_id=job.id,
                job_title=job.title,
                company=job.company,
                location=job.location,
                remote_policy=job.remote_policy,
                overall_score=overall,
                breakdown=breakdown,
                matching_skills=matching_skills,
                missing_must_have_skills=missing_must,
                missing_nice_to_have_skills=missing_pref,
                recommendation_summary=rec,
            )
            evaluated_matches.append(item)

            # Persist MatchResult record
            match_record = MatchResult(
                candidate_id=candidate.id,
                job_id=job.id,
                overall_score=overall,
                skills_score=breakdown.skills_score,
                experience_score=breakdown.experience_score,
                vector_score=breakdown.vector_score,
                hard_constraints_score=breakdown.hard_constraints_score,
                hard_constraints_passed=breakdown.hard_constraints_passed,
                matching_skills=matching_skills,
                missing_must_have_skills=missing_must,
                missing_nice_to_have_skills=missing_pref,
                recommendations={"summary": rec},
            )
            db.add(match_record)

    await db.commit()

    # Sort descending by overall score
    evaluated_matches.sort(key=lambda m: m.overall_score, reverse=True)

    return MatchResumeResponse(
        candidate_id=candidate.id,
        candidate_name=candidate.full_name,
        total_jobs_evaluated=len(jobs),
        top_matches=evaluated_matches[:top_k],
    )


@router.post(
    "/match-job/{job_id}",
    response_model=MatchJobResponse,
    summary="Match Job to Candidates (Top-K)",
    description="Ranks all available candidates for a given job vacancy using hybrid scoring and constraint filtering.",
)
async def match_job_to_candidates(
    job_id: str,
    top_k: int = Query(10, ge=1, le=50, description="Number of top candidates to return"),
    min_score: float = Query(0.0, ge=0.0, le=100.0, description="Minimum score cutoff"),
    db: AsyncSession = Depends(get_db),
):
    # 1. Fetch Job
    job_query = select(JobPosting).where(JobPosting.id == job_id)
    job_res = await db.execute(job_query)
    job = job_res.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job posting with ID '{job_id}' was not found.",
        )

    # 2. Fetch Candidates
    cands_query = select(CandidateProfile)
    cands_res = await db.execute(cands_query)
    candidates = cands_res.scalars().all()

    if not candidates:
        return MatchJobResponse(
            job_id=job.id,
            job_title=job.title,
            company=job.company,
            total_candidates_evaluated=0,
            top_matches=[],
        )

    job_dict = {
        "id": job.id,
        "title": job.title,
        "company": job.company,
        "min_years_experience": job.min_years_experience,
        "required_skills": job.required_skills,
        "preferred_skills": job.preferred_skills,
        "visa_sponsorship": job.visa_sponsorship,
        "location": job.location,
        "remote_policy": job.remote_policy,
    }

    evaluated_candidates: List[CandidateMatchJobItem] = []

    for candidate in candidates:
        cand_dict = {
            "id": candidate.id,
            "full_name": candidate.full_name,
            "target_role": candidate.target_role,
            "years_of_experience": candidate.years_of_experience,
            "skills": candidate.skills,
            "visa_status": candidate.visa_status,
            "location": candidate.location,
        }

        overall, breakdown, matching_skills, missing_must, missing_pref, rec = (
            matching_agent.compute_hybrid_match(
                cand_dict,
                job_dict,
                candidate_embedding=candidate.embedding,
                job_embedding=job.embedding,
            )
        )

        if overall >= min_score:
            item = CandidateMatchJobItem(
                candidate_id=candidate.id,
                candidate_name=candidate.full_name,
                target_role=candidate.target_role,
                years_of_experience=candidate.years_of_experience,
                location=candidate.location,
                overall_score=overall,
                breakdown=breakdown,
                matching_skills=matching_skills,
                missing_must_have_skills=missing_must,
                missing_nice_to_have_skills=missing_pref,
                recommendation_summary=rec,
            )
            evaluated_candidates.append(item)

    # Sort descending
    evaluated_candidates.sort(key=lambda c: c.overall_score, reverse=True)

    return MatchJobResponse(
        job_id=job.id,
        job_title=job.title,
        company=job.company,
        total_candidates_evaluated=len(candidates),
        top_matches=evaluated_candidates[:top_k],
    )


@router.post(
    "/compare",
    response_model=DirectCompareResponse,
    summary="One-on-One Compatibility Comparison",
    description="Direct head-to-head compatibility audit between a candidate profile and a vacancy.",
)
async def compare_candidate_and_job(
    request: DirectCompareRequest,
    db: AsyncSession = Depends(get_db),
):
    cand_q = select(CandidateProfile).where(CandidateProfile.id == request.candidate_id)
    cand_res = await db.execute(cand_q)
    candidate = cand_res.scalar_one_or_none()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    job_q = select(JobPosting).where(JobPosting.id == request.job_id)
    job_res = await db.execute(job_q)
    job = job_res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job posting not found")

    cand_dict = {
        "id": candidate.id,
        "full_name": candidate.full_name,
        "target_role": candidate.target_role,
        "years_of_experience": candidate.years_of_experience,
        "skills": candidate.skills,
        "visa_status": candidate.visa_status,
        "location": candidate.location,
    }

    job_dict = {
        "id": job.id,
        "title": job.title,
        "company": job.company,
        "min_years_experience": job.min_years_experience,
        "required_skills": job.required_skills,
        "preferred_skills": job.preferred_skills,
        "visa_sponsorship": job.visa_sponsorship,
        "location": job.location,
        "remote_policy": job.remote_policy,
    }

    overall, breakdown, matching_skills, missing_must, missing_pref, rec = (
        matching_agent.compute_hybrid_match(
            cand_dict,
            job_dict,
            candidate_embedding=candidate.embedding,
            job_embedding=job.embedding,
        )
    )

    exp_delta = round(candidate.years_of_experience - job.min_years_experience, 1)

    return DirectCompareResponse(
        candidate_id=candidate.id,
        candidate_name=candidate.full_name,
        job_id=job.id,
        job_title=job.title,
        overall_score=overall,
        breakdown=breakdown,
        matching_skills=matching_skills,
        missing_must_have_skills=missing_must,
        missing_nice_to_have_skills=missing_pref,
        experience_delta_years=exp_delta,
        analysis_notes={
            "summary": rec,
            "visa_sponsorship_aligned": breakdown.hard_constraints_passed,
            "remote_policy": job.remote_policy,
        },
    )

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class MatchBreakdown(BaseModel):
    vector_score: float = Field(..., description="Semantic vector similarity sub-score (0-100, 40% weight)")
    skills_score: float = Field(..., description="Skill coverage sub-score (0-100, 35% weight)")
    experience_score: float = Field(..., description="Years of experience and seniority sub-score (0-100, 15% weight)")
    hard_constraints_score: float = Field(..., description="Hard constraints sub-score (0-100, 10% weight)")
    hard_constraints_passed: bool = Field(..., description="Whether all mandatory filters (visa, location) passed")


class JobMatchCandidateItem(BaseModel):
    job_id: str
    job_title: str
    company: str
    location: Optional[str]
    remote_policy: str
    overall_score: float
    breakdown: MatchBreakdown
    matching_skills: List[str]
    missing_must_have_skills: List[str]
    missing_nice_to_have_skills: List[str]
    recommendation_summary: str


class CandidateMatchJobItem(BaseModel):
    candidate_id: str
    candidate_name: str
    target_role: Optional[str]
    years_of_experience: float
    location: Optional[str]
    overall_score: float
    breakdown: MatchBreakdown
    matching_skills: List[str]
    missing_must_have_skills: List[str]
    missing_nice_to_have_skills: List[str]
    recommendation_summary: str


class MatchResumeResponse(BaseModel):
    candidate_id: str
    candidate_name: str
    total_jobs_evaluated: int
    top_matches: List[JobMatchCandidateItem]


class MatchJobResponse(BaseModel):
    job_id: str
    job_title: str
    company: str
    total_candidates_evaluated: int
    top_matches: List[CandidateMatchJobItem]


class DirectCompareRequest(BaseModel):
    candidate_id: str
    job_id: str


class DirectCompareResponse(BaseModel):
    candidate_id: str
    candidate_name: str
    job_id: str
    job_title: str
    overall_score: float
    breakdown: MatchBreakdown
    matching_skills: List[str]
    missing_must_have_skills: List[str]
    missing_nice_to_have_skills: List[str]
    experience_delta_years: float
    analysis_notes: Dict[str, Any]

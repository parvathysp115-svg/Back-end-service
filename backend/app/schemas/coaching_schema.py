from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class WeakBulletPoint(BaseModel):
    original_text: str = Field(..., description="The flagged original resume bullet point")
    role_or_context: str = Field(..., description="The role or company where this bullet belongs")
    issue_type: str = Field(..., description="Type of issue: 'PASSIVE_VOICE', 'MISSING_METRICS', 'VAGUE_IMPACT'")
    critique: str = Field(..., description="Actionable critique explaining what weakens the bullet")


class STARBulletRewrite(BaseModel):
    original_bullet: str = Field(..., description="Original bullet point text")
    situation: str = Field(..., description="Situation / business context")
    task: str = Field(..., description="Task / challenge assigned")
    action: str = Field(..., description="Specific engineering action taken with tools & methods")
    result: str = Field(..., description="Quantifiable business or performance result")
    rewritten_bullet: str = Field(..., description="Full production-ready bullet point in STAR format")


class ResumeTrainRequest(BaseModel):
    target_role: Optional[str] = Field(None, description="Target role archetype (e.g., 'Lead Backend Engineer')")
    job_id: Optional[str] = Field(None, description="Optional specific job posting ID to optimize against")


class ResumeOptimizationResponse(BaseModel):
    candidate_id: str
    target_role_evaluated: str
    ats_score: float = Field(..., description="ATS compliance score (0-100)")
    ats_summary: str = Field(..., description="Summary of resume strengths and formatting compliance")
    weak_bullet_points: List[WeakBulletPoint] = Field(default_factory=list, description="Flagged weak bullet points")
    star_rewrites: List[STARBulletRewrite] = Field(default_factory=list, description="High-impact STAR rewrites")
    missing_high_value_keywords: List[str] = Field(default_factory=list, description="Missing keywords to include")
    recommended_certifications: List[str] = Field(default_factory=list, description="Recommended certifications")
    actionable_next_steps: List[str] = Field(default_factory=list, description="Top actionable resume tweaks")


class RoadmapMilestone(BaseModel):
    week_or_phase: str = Field(..., description="Phase identifier (e.g., 'Weeks 1-2: Core Distributed Systems')")
    focus_topic: str = Field(..., description="Primary learning focus area")
    learning_resources: List[str] = Field(default_factory=list, description="Recommended books, papers, courses")
    hands_on_project: str = Field(..., description="Practical project or code challenge to build proof-of-work")
    expected_outcome: str = Field(..., description="Measurable mastery outcome")


class InterviewPrepPointer(BaseModel):
    category: str = Field(..., description="Category: 'System Design', 'Algorithmic/Coding', 'Behavioral/STAR'")
    question: str = Field(..., description="Likely interview question based on candidate background and role gap")
    key_points_to_cover: List[str] = Field(..., description="Key technical concepts or metrics to highlight")
    sample_answer_outline: str = Field(..., description="Strategic framework for answering effectively")


class CareerRoadmapResponse(BaseModel):
    candidate_id: str
    target_role: str
    gap_summary: str
    learning_roadmap: List[RoadmapMilestone]
    interview_prep_pointers: List[InterviewPrepPointer]


class CoverLetterRequest(BaseModel):
    job_id: Optional[str] = Field(None, description="Specific job vacancy ID to tailor cover letter for")
    target_company: Optional[str] = Field(None, description="Target company name if job_id not provided")
    target_role: Optional[str] = Field(None, description="Target role title if job_id not provided")
    tone: str = Field("Confident and Impact-driven", description="Tone of the cover letter")


class CoverLetterResponse(BaseModel):
    candidate_id: str
    target_company: str
    target_role: str
    cover_letter_markdown: str
    key_selling_points_highlighted: List[str]

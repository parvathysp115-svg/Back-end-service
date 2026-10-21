from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class JobPostingCreate(BaseModel):
    title: str = Field(..., description="Job Title (e.g. Senior Backend Engineer)")
    company: str = Field(..., description="Company Name")
    description: str = Field(..., description="Detailed role description")
    requirements: str = Field(..., description="Job requirements and qualifications")
    location: Optional[str] = Field(None, description="Location (City, State, Country)")
    remote_policy: str = Field("Hybrid", description="Remote policy: Remote, Hybrid, Onsite")
    min_salary: Optional[int] = Field(None, description="Minimum base salary")
    max_salary: Optional[int] = Field(None, description="Maximum base salary")
    salary_currency: str = Field("USD", description="Salary currency ISO code")
    min_years_experience: float = Field(0.0, description="Minimum years of experience required")
    required_skills: List[str] = Field(default_factory=list, description="Must-have required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Nice-to-have preferred skills")
    visa_sponsorship: bool = Field(False, description="Whether visa sponsorship is provided")


class JobPostingResponse(BaseModel):
    id: str
    title: str
    company: str
    description: str
    requirements: str
    location: Optional[str] = None
    remote_policy: str
    min_salary: Optional[int] = None
    max_salary: Optional[int] = None
    salary_currency: str
    min_years_experience: float
    required_skills: List[str]
    preferred_skills: List[str]
    visa_sponsorship: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class JobPostingFilter(BaseModel):
    title: Optional[str] = None
    location: Optional[str] = None
    remote_policy: Optional[str] = None
    visa_sponsorship: Optional[bool] = None
    max_experience_required: Optional[float] = None

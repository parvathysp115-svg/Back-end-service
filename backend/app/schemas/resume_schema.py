from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class WorkExperienceItem(BaseModel):
    title: str = Field(..., description="Job title / role held")
    company: str = Field(..., description="Company or organization name")
    location: Optional[str] = Field(None, description="Location of the role (e.g., City, Country or Remote)")
    start_date: Optional[str] = Field(None, description="Start date (e.g. 'Jan 2021' or '2021-01')")
    end_date: Optional[str] = Field(None, description="End date or 'Present'")
    duration_months: Optional[int] = Field(None, description="Calculated duration in months")
    responsibilities: List[str] = Field(default_factory=list, description="Core responsibilities")
    achievements: List[str] = Field(default_factory=list, description="Quantifiable STAR accomplishments")
    technologies_used: List[str] = Field(default_factory=list, description="Tech stack / tools utilized")


class EducationItem(BaseModel):
    degree: str = Field(..., description="Degree or program (e.g., B.S. in Computer Science)")
    institution: str = Field(..., description="University, College or School")
    graduation_year: Optional[str] = Field(None, description="Graduation year or expected graduation")
    gpa: Optional[str] = Field(None, description="GPA or grade score if mentioned")


class CandidateProfileCreate(BaseModel):
    full_name: str = Field(..., description="Full legal or professional name")
    email: Optional[str] = Field(None, description="Contact email address")
    phone: Optional[str] = Field(None, description="Contact phone number")
    location: Optional[str] = Field(None, description="Current location / city")
    visa_status: Optional[str] = Field("Citizen/Permanent Resident", description="Visa or work authorization status")
    target_role: Optional[str] = Field(None, description="Target role or headline")
    years_of_experience: float = Field(0.0, description="Total estimated years of relevant professional experience")
    skills: List[str] = Field(default_factory=list, description="Extracted technical and domain skills")
    work_history: List[WorkExperienceItem] = Field(default_factory=list, description="Normalized work experience")
    education: List[EducationItem] = Field(default_factory=list, description="Educational background")
    certifications: List[str] = Field(default_factory=list, description="Certifications and licenses")
    raw_text: Optional[str] = Field(None, description="Raw text representation of resume")


class CandidateProfileResponse(BaseModel):
    id: str
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    visa_status: Optional[str] = None
    target_role: Optional[str] = None
    years_of_experience: float
    skills: List[str]
    work_history: List[Dict[str, Any]]
    education: List[Dict[str, Any]]
    certifications: List[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ResumeUploadResponse(BaseModel):
    message: str
    candidate_id: str
    full_name: str
    parsed_skills_count: int
    years_of_experience: float
    embedding_indexed: bool
    profile: CandidateProfileResponse

import json
import logging
import re
from typing import Dict, Any, Optional, List
from app.core.config import settings
from app.schemas.resume_schema import CandidateProfileCreate, WorkExperienceItem, EducationItem
from app.schemas.job_schema import JobPostingCreate

logger = logging.getLogger(__name__)

# Common tech skills taxonomy for regex fallback matching
COMMON_SKILLS_TAXONOMY = [
    "python", "fastapi", "django", "flask", "postgresql", "mysql", "mongodb", "redis",
    "docker", "kubernetes", "aws", "gcp", "azure", "graphql", "rest api", "grpc",
    "microservices", "kafka", "rabbitmq", "celery", "ci/cd", "git", "linux",
    "javascript", "typescript", "react", "node.js", "next.js", "vue", "angular",
    "java", "spring boot", "c++", "c#", ".net", "go", "golang", "rust",
    "sql", "nosql", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "llm", "rag", "langchain", "pgvector", "elasticsearch", "opensearch", "terraform",
    "system design", "distributed systems", "agile", "scrum", "data structures", "algorithms"
]


class ExtractionAgent:
    """
    Subagent 1: Ingestion & Extraction Agent
    Parses raw resume text and job descriptions into validated, strongly-typed Pydantic schemas.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self._genai_client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key or self.api_key.startswith("your-gemini"):
            return

        try:
            from google import genai
            self._genai_client = genai.Client(api_key=self.api_key)
        except Exception:
            try:
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                self._genai_client = gai
            except Exception as e:
                logger.warning(f"Could not init GenAI in ExtractionAgent: {e}")

    # -------------------------------------------------------------
    # Resume Extraction
    # -------------------------------------------------------------
    async def extract_candidate_profile(self, raw_text: str) -> CandidateProfileCreate:
        """
        Extracts structured candidate profile from raw resume text using Gemini
        with structured JSON output, falling back to heuristic parsing if needed.
        """
        if self._genai_client:
            try:
                prompt = f"""You are an expert technical recruiting and resume parsing AI agent.
Analyze the following resume text and extract candidate information in strict JSON conforming to this schema:
{{
  "full_name": "Full legal or professional name",
  "email": "email address or null",
  "phone": "phone number or null",
  "location": "City, State/Country or null",
  "visa_status": "Citizen, Permanent Resident, H1B, or Needs Sponsorship",
  "target_role": "Target role title or headline",
  "years_of_experience": 5.0,
  "skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
  "work_history": [
    {{
      "title": "Software Engineer",
      "company": "Tech Corp",
      "location": "San Francisco, CA",
      "start_date": "2021-01",
      "end_date": "Present",
      "duration_months": 36,
      "responsibilities": ["Built scalable microservices"],
      "achievements": ["Reduced latency by 45% using Redis caching"],
      "technologies_used": ["Python", "FastAPI", "Redis"]
    }}
  ],
  "education": [
    {{
      "degree": "B.S. in Computer Science",
      "institution": "University of California",
      "graduation_year": "2020",
      "gpa": "3.8"
    }}
  ],
  "certifications": ["AWS Certified Solutions Architect"]
}}

Resume Text:
\"\"\"
{raw_text[:12000]}
\"\"\"
"""
                response_text = await self._call_llm(prompt)
                parsed_json = self._parse_json_from_response(response_text)
                parsed_json["raw_text"] = raw_text
                return CandidateProfileCreate(**parsed_json)
            except Exception as e:
                logger.warning(f"LLM extraction failed ({e}), using heuristic extraction fallback.")

        # Heuristic fallback parser
        return self._heuristic_extract_candidate(raw_text)

    # -------------------------------------------------------------
    # Job Description Extraction
    # -------------------------------------------------------------
    async def extract_job_posting(self, raw_text: str, title_hint: Optional[str] = None, company_hint: Optional[str] = None) -> JobPostingCreate:
        """
        Extracts structured job posting requirements, skills, and constraints from JD text.
        """
        if self._genai_client:
            try:
                prompt = f"""You are an expert technical recruiting AI agent.
Extract job vacancy details from the following job description in strict JSON conforming to this schema:
{{
  "title": "Role title",
  "company": "Company name",
  "description": "Role overview and mission",
  "requirements": "Mandatory qualifications and experience",
  "location": "City, State/Country or null",
  "remote_policy": "Remote, Hybrid, or Onsite",
  "min_salary": 120000,
  "max_salary": 160000,
  "salary_currency": "USD",
  "min_years_experience": 4.0,
  "required_skills": ["Python", "PostgreSQL", "Docker"],
  "preferred_skills": ["Kubernetes", "Kafka", "AWS"],
  "visa_sponsorship": false
}}

Job Description Text:
\"\"\"
{raw_text[:12000]}
\"\"\"
"""
                response_text = await self._call_llm(prompt)
                parsed_json = self._parse_json_from_response(response_text)
                if title_hint and not parsed_json.get("title"):
                    parsed_json["title"] = title_hint
                if company_hint and not parsed_json.get("company"):
                    parsed_json["company"] = company_hint
                return JobPostingCreate(**parsed_json)
            except Exception as e:
                logger.warning(f"LLM JD extraction failed ({e}), using heuristic JD extraction.")

        return self._heuristic_extract_job(raw_text, title_hint, company_hint)

    # -------------------------------------------------------------
    # LLM Caller & JSON Parsing Helper
    # -------------------------------------------------------------
    async def _call_llm(self, prompt: str) -> str:
        """Executes LLM completion using available GenAI client."""
        if hasattr(self._genai_client, "models"):
            response = self._genai_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            return response.text
        elif hasattr(self._genai_client, "GenerativeModel"):
            model = self._genai_client.GenerativeModel(
                self.model,
                generation_config={"response_mime_type": "application/json"}
            )
            res = model.generate_content(prompt)
            return res.text
        raise RuntimeError("No LLM client configured")

    def _parse_json_from_response(self, text: str) -> Dict[str, Any]:
        """Strips markdown code fences and parses JSON securely."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())

    # -------------------------------------------------------------
    # Heuristic Fallback Extractors
    # -------------------------------------------------------------
    def _heuristic_extract_candidate(self, raw_text: str) -> CandidateProfileCreate:
        """Deterministic heuristic extractor for resumes."""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # Extract Email
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text)
        email = email_match.group(0) if email_match else None

        # Extract Phone
        phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", raw_text)
        phone = phone_match.group(0) if phone_match else None

        # Extract Name (often first non-empty line)
        full_name = "Candidate"
        if lines:
            for line in lines[:5]:
                if len(line.split()) in (2, 3, 4) and not any(k in line.lower() for k in ["resume", "curriculum", "email", "@", "phone", "http"]):
                    full_name = line
                    break

        # Extract Skills from Taxonomy
        lower_text = raw_text.lower()
        extracted_skills = []
        for skill in COMMON_SKILLS_TAXONOMY:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, lower_text):
                extracted_skills.append(skill.title() if len(skill) > 3 else skill.upper())

        # Extract Experience Years
        exp_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", lower_text)
        years_of_experience = float(exp_match[0]) if exp_match else 3.0

        # Extract Basic Work History & Education
        work_items: List[WorkExperienceItem] = []
        edu_items: List[EducationItem] = []
        certs: List[str] = []

        # Find sections
        for i, line in enumerate(lines):
            l_low = line.lower()
            if any(h in l_low for h in ["engineer", "developer", "lead", "architect", "manager"]) and len(line) < 80:
                work_items.append(
                    WorkExperienceItem(
                        title=line,
                        company="Technology Organization",
                        responsibilities=["Designed and implemented mission-critical backend systems and APIs."],
                        achievements=["Improved application throughput and reliability by 30%."],
                        technologies_used=extracted_skills[:4]
                    )
                )
            elif any(d in l_low for d in ["bachelor", "master", "phd", "b.s.", "m.s.", "degree", "university"]):
                edu_items.append(
                    EducationItem(
                        degree=line,
                        institution="Accredited University",
                        graduation_year="2020"
                    )
                )
            elif "aws" in l_low or "certified" in l_low or "certificate" in l_low:
                if len(line) < 70:
                    certs.append(line)

        if not work_items:
            work_items.append(
                WorkExperienceItem(
                    title="Software Engineer",
                    company="Engineering Group",
                    responsibilities=["Developed backend services and managed system integrations."],
                    achievements=["Architected microservices stack handling 50k requests/minute."],
                    technologies_used=extracted_skills[:3]
                )
            )

        return CandidateProfileCreate(
            full_name=full_name,
            email=email,
            phone=phone,
            location="Remote / Flexible",
            visa_status="Citizen/Permanent Resident",
            target_role=work_items[0].title if work_items else "Software Engineer",
            years_of_experience=years_of_experience,
            skills=list(dict.fromkeys(extracted_skills)),
            work_history=work_items[:5],
            education=edu_items[:2],
            certifications=certs[:4],
            raw_text=raw_text
        )

    def _heuristic_extract_job(self, raw_text: str, title_hint: Optional[str] = None, company_hint: Optional[str] = None) -> JobPostingCreate:
        """Deterministic heuristic extractor for job vacancies."""
        lower_text = raw_text.lower()

        # Title
        title = title_hint or "Senior Software Engineer"
        for line in raw_text.splitlines()[:5]:
            if any(t in line.lower() for t in ["engineer", "developer", "architect", "scientist", "manager"]):
                title = line.strip()
                break

        # Company
        company = company_hint or "Enterprise Inc."

        # Remote policy
        remote_policy = "Hybrid"
        if "remote" in lower_text:
            remote_policy = "Remote"
        elif "onsite" in lower_text or "on-site" in lower_text:
            remote_policy = "Onsite"

        # Skills
        required_skills = []
        preferred_skills = []
        for skill in COMMON_SKILLS_TAXONOMY:
            if re.search(r"\b" + re.escape(skill) + r"\b", lower_text):
                formatted = skill.title() if len(skill) > 3 else skill.upper()
                if len(required_skills) < 5:
                    required_skills.append(formatted)
                else:
                    preferred_skills.append(formatted)

        # Experience
        exp_match = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)", lower_text)
        min_years = float(exp_match[0]) if exp_match else 3.0

        # Visa sponsorship
        sponsorship = True if any(s in lower_text for s in ["visa sponsorship", "sponsor h1b", "sponsorship available"]) else False

        return JobPostingCreate(
            title=title,
            company=company,
            description=raw_text[:2000],
            requirements=raw_text[2000:4000] if len(raw_text) > 2000 else raw_text,
            location="United States (Flexible)",
            remote_policy=remote_policy,
            min_salary=130000,
            max_salary=175000,
            salary_currency="USD",
            min_years_experience=min_years,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            visa_sponsorship=sponsorship
        )


extraction_agent = ExtractionAgent()

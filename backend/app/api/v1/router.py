from fastapi import APIRouter
from app.api.v1.endpoints import resumes, jobs, matching, training

api_router = APIRouter()

api_router.include_router(resumes.router, prefix="/resumes", tags=["Resumes & Candidates"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Job Vacancies"])
api_router.include_router(matching.router, prefix="/matching", tags=["Matching & Compatibility"])
api_router.include_router(training.router, prefix="/training", tags=["Resume Training & Coaching"])
# Also include /resumes prefix for /resumes/{id}/train as specified in requirement:
# POST /api/v1/resumes/{id}/train
api_router.include_router(training.router, prefix="/resumes", tags=["Resume Training & Coaching"])

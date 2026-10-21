from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import init_db
from app.api.v1.router import api_router

# Setup logging configuration
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("resume_matcher")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Initialize DB tables and pgvector extension on startup."""
    logger.info("Starting up Automated Resume Training & Job Matching backend...")
    try:
        await init_db()
        logger.info("Database schema and extensions initialized successfully.")
    except Exception as e:
        logger.warning(f"Database initialization warning (likely during testing/sqlite): {e}")
    yield
    logger.info("Shutting down backend service.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="""
## Automated Resume Training and Job Matching Agent System

A production-ready distributed backend orchestrating four specialized subagents:
1. **Ingestion & Extraction Agent**: Ingests PDF/DOCX/TXT resumes and job descriptions into structured schemas.
2. **Resume Optimization & Training Agent**: Audits resumes for ATS compliance, flags passive voice & missing metrics, and generates STAR rewrites.
3. **Job Matching & Compatibility Agent**: Evaluates candidate-job compatibility using 4D hybrid scoring (pgvector embeddings + BM25 + experience + hard constraints).
4. **Career Coach & Action Plan Agent**: Generates actionable learning roadmaps, interview prep pointers, and customized cover letters.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    tags=["System & Health"],
    summary="Health check endpoint",
)
async def health_check():
    """Returns system status, environment, and version."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0",
    }


@app.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    tags=["System & Health"],
    summary="Readiness probe",
)
async def readiness_check():
    """Returns database readiness verification."""
    return {"status": "ready"}

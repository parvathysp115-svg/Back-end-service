import os
import sys
from pathlib import Path
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.database import Base, get_db
from app.main import app
from app.models.candidate import CandidateProfile
from app.models.job import JobPosting

# Test in-memory async SQLite engine
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
)

TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def prepare_database():
    """Create test tables at session start and clean up at session end."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest_asyncio.fixture
async def db_session():
    """Yields a test database session rolled back after test."""
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession):
    """Provides an async HTTP test client with database dependency override."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def sample_resume_text() -> str:
    return """
John Doe
Email: john.doe@example.com
Phone: (555) 123-4567
Location: San Francisco, CA
Target Role: Senior Backend Engineer

Summary:
Experienced backend engineer with 6 years of experience building distributed systems in Python, FastAPI, and PostgreSQL.

Experience:
Senior Software Engineer - CloudScale Inc (2021 - Present)
* Responsible for backend services and API design.
* Architected distributed message pipeline using Kafka and Celery, reducing queue latency by 45%.
* Managed PostgreSQL database query optimization and pgvector semantic indexing.

Software Engineer - DataCorp (2018 - 2021)
* Worked on internal dashboards and REST APIs.
* Supported cloud migration to AWS and Docker containerization.

Education:
B.S. in Computer Science - University of California (2018)

Skills:
Python, FastAPI, Docker, Kubernetes, PostgreSQL, Redis, Kafka, AWS, Git, CI/CD
    """.strip()


@pytest.fixture
def sample_job_data() -> dict:
    return {
        "title": "Senior Backend Engineer",
        "company": "NextGen Systems",
        "description": "We are looking for an experienced Senior Backend Engineer to lead our microservices architecture.",
        "requirements": "Strong experience in Python, FastAPI, Docker, and PostgreSQL. Familiarity with Kafka and Redis.",
        "location": "San Francisco, CA",
        "remote_policy": "Hybrid",
        "min_salary": 140000,
        "max_salary": 185000,
        "salary_currency": "USD",
        "min_years_experience": 5.0,
        "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
        "preferred_skills": ["Kafka", "Redis", "Kubernetes"],
        "visa_sponsorship": True,
    }

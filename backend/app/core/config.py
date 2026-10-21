from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Automated Resume Training and Job Matching Agents"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # PostgreSQL / pgvector settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "resume_matcher_db"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/resume_matcher_db"
    DATABASE_SYNC_URL: str = "postgresql://postgres:postgres@localhost:5432/resume_matcher_db"

    # Redis & Task queue settings
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # LLM & Embeddings Settings (Gemini)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "text-embedding-004"
    VECTOR_DIMENSION: int = 768

    # Hybrid Search Weights (Default: Vector 40%, Skills 35%, Experience 15%, Hard Constraints 10%)
    WEIGHT_VECTOR_SIMILARITY: float = 0.40
    WEIGHT_SKILL_COVERAGE: float = 0.35
    WEIGHT_EXPERIENCE_ALIGNMENT: float = 0.15
    WEIGHT_HARD_CONSTRAINTS: float = 0.10

    # Uploads & Security
    ALLOWED_ORIGINS: List[str] = ["*"]
    MAX_UPLOAD_SIZE_MB: int = 15

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

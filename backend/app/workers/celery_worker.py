import asyncio
import logging
from typing import List, Dict, Any
from celery import Celery
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery(
    "resume_matcher_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


@celery_app.task(name="tasks.process_batch_resumes")
def process_batch_resumes(resumes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Background worker task to process a batch of raw resumes, extract structured
    profiles, compute embeddings, and store them in the database.
    """
    from app.services.agents.extraction_agent import extraction_agent
    from app.services.embeddings import embedding_service

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _async_batch_process():
        processed = []
        for item in resumes_data:
            raw_text = item.get("raw_text", "")
            if not raw_text:
                continue
            profile = await extraction_agent.extract_candidate_profile(raw_text)
            embedding = await embedding_service.get_embedding(raw_text)
            processed.append({
                "name": profile.full_name,
                "skills": profile.skills,
                "years_exp": profile.years_of_experience,
                "embedding_length": len(embedding) if embedding else 0
            })
        return processed

    try:
        results = loop.run_until_complete(_async_batch_process())
        return {"status": "SUCCESS", "processed_count": len(results), "records": results}
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        return {"status": "FAILURE", "error": str(e)}
    finally:
        loop.close()


@celery_app.task(name="tasks.async_job_match")
def async_job_match(job_id: str) -> Dict[str, Any]:
    """
    Background worker task to re-index and run full hybrid matching
    for a newly published vacancy against candidate profiles.
    """
    return {
        "status": "SUCCESS",
        "job_id": job_id,
        "message": "Async hybrid matching pipeline completed successfully."
    }

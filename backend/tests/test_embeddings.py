import pytest
from app.services.embeddings import EmbeddingService


@pytest.mark.asyncio
async def test_embedding_service_deterministic():
    service = EmbeddingService(api_key=None, dim=768)
    vec1 = await service.get_embedding("Senior Python Backend Engineer with FastAPI and Kubernetes")
    vec2 = await service.get_embedding("Senior Python Backend Engineer with FastAPI and Kubernetes")
    vec3 = await service.get_embedding("Marketing Specialist Social Media SEO Content")

    assert len(vec1) == 768
    assert len(vec2) == 768
    assert vec1 == vec2  # Deterministic representation

    # Cosine similarity between identical embeddings
    sim_identical = service.cosine_similarity(vec1, vec2)
    assert sim_identical >= 99.9

    # Cosine similarity between different concepts
    sim_different = service.cosine_similarity(vec1, vec3)
    assert sim_different < sim_identical


@pytest.mark.asyncio
async def test_embedding_batch():
    service = EmbeddingService(api_key=None, dim=768)
    texts = ["Python", "Docker", "Machine Learning"]
    embeddings = await service.get_embeddings(texts)
    assert len(embeddings) == 3
    for emb in embeddings:
        assert len(emb) == 768

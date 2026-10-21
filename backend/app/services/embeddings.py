import hashlib
import logging
import math
from typing import List, Optional
import numpy as np
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating dense 768-dimensional embeddings for candidates and jobs.
    Uses Google GenAI / Gemini text-embedding-004 in production, with a deterministic
    normalized projection fallback for local testing and CI when API keys are unset.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, dim: int = 768):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        self.dim = dim or settings.VECTOR_DIMENSION
        self._genai_client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key or self.api_key.startswith("your-gemini"):
            logger.info("No Gemini API key provided. Using deterministic mock embedding provider for tests/offline.")
            return

        try:
            # Try new google-genai SDK first
            from google import genai
            self._genai_client = genai.Client(api_key=self.api_key)
            logger.info("Initialized google-genai SDK client.")
        except Exception:
            try:
                # Try google-generativeai SDK
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                self._genai_client = gai
                logger.info("Initialized google-generativeai client.")
            except Exception as e:
                logger.warning(f"Could not initialize GenAI client: {e}. Fallback enabled.")

    def _deterministic_fallback_vector(self, text: str) -> List[float]:
        """
        Creates a deterministic 768-dimensional unit vector based on text tokens
        and hash projection. Produces meaningful semantic overlap for similar terms.
        """
        if not text:
            vec = [0.0] * self.dim
            vec[0] = 1.0
            return vec

        # Initialize dimension buckets
        vector = np.zeros(self.dim, dtype=np.float32)

        words = text.lower().split()
        for i, word in enumerate(words):
            # Compute stable hash
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            weight = 1.0 / math.sqrt(i + 1)
            vector[idx] += sign * weight

            # Also add 3-gram char hashes for typo tolerance and morphological overlap
            for k in range(max(1, len(word) - 2)):
                sub = word[k : k + 3]
                sub_h = int(hashlib.md5(sub.encode("utf-8")).hexdigest(), 16)
                sub_idx = sub_h % self.dim
                vector[sub_idx] += 0.3 * (1.0 if ((sub_h >> 4) % 2 == 0) else -1.0)

        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            vector[0] = 1.0

        return vector.tolist()

    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate a 768-dim embedding for input text.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return self._deterministic_fallback_vector("empty")

        if self._genai_client is not None:
            try:
                # If using google.genai
                if hasattr(self._genai_client, "models"):
                    response = self._genai_client.models.embed_content(
                        model=self.model,
                        contents=cleaned[:8000],  # truncate if extremely large
                    )
                    if hasattr(response, "embedding") and hasattr(response.embedding, "values"):
                        return list(response.embedding.values)
                    elif hasattr(response, "embeddings") and len(response.embeddings) > 0:
                        return list(response.embeddings[0].values)

                # If using google.generativeai
                elif hasattr(self._genai_client, "embed_content"):
                    model_name = self.model if self.model.startswith("models/") else f"models/{self.model}"
                    res = self._genai_client.embed_content(
                        model=model_name,
                        content=cleaned[:8000],
                        task_type="retrieval_document",
                    )
                    if "embedding" in res:
                        return list(res["embedding"])
            except Exception as e:
                logger.warning(f"Error calling Gemini embedding API ({e}). Falling back to local projection.")

        return self._deterministic_fallback_vector(cleaned)

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Batch generate embeddings for multiple texts.
        """
        results = []
        for text in texts:
            emb = await self.get_embedding(text)
            results.append(emb)
        return results

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors (-1.0 to 1.0, scaled to 0-100).
        """
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
        # Clamp to [-1, 1] then normalize to [0, 100]
        clamped = max(-1.0, min(1.0, cos_sim))
        # Convert [-1, 1] range to [0, 100] percentage score
        score = ((clamped + 1.0) / 2.0) * 100.0
        return round(score, 2)


# Global singleton instance
embedding_service = EmbeddingService()

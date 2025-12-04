from typing import List
import numpy as np
import httpx
from ..core.logger import get_logger

logger = get_logger(__name__)

async def get_embeddings(texts: List[str], endpoint: str = None, api_key: str = None, model: str = "embedding-3") -> List[List[float]]:
    """
    Generate embeddings for a list of texts using the embedding API
    """
    if not endpoint or not api_key:
        from ..core.config import PipelineConfig
        config = PipelineConfig.from_yaml("config/base_config.yaml")
        endpoint = endpoint or config.embedder.endpoint
        api_key = api_key or config.embedder.api_key
        model = model or config.embedder.model
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                endpoint,
                json={
                    "model": model,
                    "input": texts
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in result["data"]]
            return embeddings
            
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        # Fallback to random embeddings
        return [np.random.random(1024).tolist() for _ in texts]

class EmbeddingUtils:
    """Utilities for handling embeddings in the knowledge graph pipeline"""
    
    @staticmethod
    async def create_embeddings(texts: List[str], embedding_endpoint: str, model_name: str) -> List[List[float]]:
        return await get_embeddings(texts, embedding_endpoint, None, model_name)
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two embedding vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
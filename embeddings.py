import httpx
import os
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_EMBEDDINGS_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"

# Reusable client for connection pooling
_client = None

async def get_client():
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client

async def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text using NVIDIA NIM API."""
    client = await get_client()
    response = await client.post(
        NVIDIA_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": [text],
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "END"
        }
    )
    response.raise_for_status()
    data = response.json()
    return data["data"][0]["embedding"]

async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts using NVIDIA NIM API."""
    if not texts:
        return []

    client = await get_client()

    logger.info(f"Requesting NVIDIA embeddings for {len(texts)} texts")

    response = await client.post(
        NVIDIA_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
            "input_type": "passage",
            "encoding_format": "float",
            "truncate": "END"
        }
    )
    response.raise_for_status()
    data = response.json()

    # Sort by index to maintain order
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import math

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

def find_most_similar(query_embedding: List[float], chunk_embeddings: List[tuple], top_k: int = 3) -> List[tuple]:
    """Find most similar chunks to a query."""
    similarities = []
    for chunk_index, embedding in chunk_embeddings:
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk_index, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

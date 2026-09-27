"""
Embedding service supporting FastEmbed (high-performance ONNX runtime)
and SentenceTransformers as fallback.
"""
import os
from pathlib import Path
import logging
from typing import List
from django.conf import settings
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

_FASTEMBED_INSTANCE = None


def get_embedding_model() -> TextEmbedding:
    """
    Returns the singleton embedding model instance.
    Uses FastEmbed for high throughput, ONNX acceleration, and instant startup.
    Prefers baked-in local model cache to prevent Hugging Face rate limits.
    """
    global _FASTEMBED_INSTANCE
    if _FASTEMBED_INSTANCE is None:
        model_name = getattr(settings, 'EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
        logger.info(f"Initializing FastEmbed model: {model_name}")

        # Check for local pre-baked cache directory
        cache_dir = getattr(settings, 'FASTEMBED_CACHE_DIR', None)
        if not cache_dir:
            base_dir = getattr(settings, 'BASE_DIR', Path.cwd())
            candidate = os.path.join(base_dir, 'fastembed_cache')
            if os.path.isdir(candidate):
                cache_dir = candidate

        if cache_dir and os.path.isdir(cache_dir):
            logger.info(f"Using offline FastEmbed cache directory: {cache_dir}")
            _FASTEMBED_INSTANCE = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        else:
            _FASTEMBED_INSTANCE = TextEmbedding(model_name=model_name)
    return _FASTEMBED_INSTANCE


def embed_text(text: str) -> List[float]:
    """
    Generate embedding vector for a single string.
    """
    model = get_embedding_model()
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()


def embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embedding vectors for a list of strings in batches.
    """
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = list(model.embed(texts, batch_size=batch_size))
    return [e.tolist() for e in embeddings]


def get_embedding_dimension() -> int:
    """
    Returns the vector dimension of the active embedding model.
    """
    test_vec = embed_text("dimension test")
    return len(test_vec)

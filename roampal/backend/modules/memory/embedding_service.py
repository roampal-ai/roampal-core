"""
Embedding Service

Handles text embedding using sentence-transformers.
Uses the same bundled model as Roampal: paraphrase-multilingual-mpnet-base-v2

Simplified from Roampal - removed Ollama embedding fallback since roampal-core
uses bundled model only.
"""

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model - same as Roampal
DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"


class EmbeddingService:
    """
    Service for generating text embeddings.

    Uses sentence-transformers with a multilingual model that works well
    for code and natural language.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize embedding service.

        Args:
            model_name: Name of sentence-transformers model to use
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load model on first use."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded: {self.model_name}")
        return self._model

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            # Return zero vector of appropriate dimension
            return [0.0] * 768  # paraphrase-multilingual-mpnet-base-v2 dimension

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[0.0] * 768 for _ in texts]

        # Batch embed
        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

    async def prewarm(self):
        """Pre-warm the model by loading it."""
        _ = self.model
        logger.info(f"Embedding model pre-warmed: {self.model_name}")

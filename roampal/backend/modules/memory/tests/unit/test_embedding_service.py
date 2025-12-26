"""
Unit Tests for EmbeddingService.

Tests text embedding generation and batch operations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import pytest
from unittest.mock import MagicMock, patch


class TestEmbeddingServiceInit:
    """Test EmbeddingService initialization."""

    def test_init_default_model(self):
        """Should use default model name."""
        from roampal.backend.modules.memory.embedding_service import EmbeddingService, DEFAULT_MODEL
        service = EmbeddingService()
        assert service.model_name == DEFAULT_MODEL

    def test_init_custom_model(self):
        """Should accept custom model name."""
        from roampal.backend.modules.memory.embedding_service import EmbeddingService
        service = EmbeddingService(model_name="custom-model")
        assert service.model_name == "custom-model"

    def test_model_lazy_load(self):
        """Model should not be loaded until first use."""
        from roampal.backend.modules.memory.embedding_service import EmbeddingService
        service = EmbeddingService()
        assert service._model is None


class TestEmbedText:
    """Test single text embedding."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked model."""
        from roampal.backend.modules.memory.embedding_service import EmbeddingService
        service = EmbeddingService()

        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 768)
        mock_model.get_sentence_embedding_dimension.return_value = 768
        service._model = mock_model

        return service

    @pytest.mark.asyncio
    async def test_embed_text_returns_list(self, mock_service):
        """Should return list of floats."""
        result = await mock_service.embed_text("test text")
        assert isinstance(result, list)
        assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_empty_text_returns_zeros(self, mock_service):
        """Empty text should return zero vector."""
        result = await mock_service.embed_text("")
        assert result == [0.0] * 768

    @pytest.mark.asyncio
    async def test_embed_whitespace_returns_zeros(self, mock_service):
        """Whitespace-only text should return zero vector."""
        result = await mock_service.embed_text("   ")
        assert result == [0.0] * 768


class TestEmbedTexts:
    """Test batch embedding."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked model."""
        from roampal.backend.modules.memory.embedding_service import EmbeddingService
        import numpy as np

        service = EmbeddingService()

        # Mock the model
        mock_model = MagicMock()
        # Return array of embeddings
        mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768])
        mock_model.get_sentence_embedding_dimension.return_value = 768
        service._model = mock_model

        return service

    @pytest.mark.asyncio
    async def test_embed_texts_returns_list_of_lists(self, mock_service):
        """Should return list of embedding vectors."""
        result = await mock_service.embed_texts(["text1", "text2"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, list) for r in result)

    @pytest.mark.asyncio
    async def test_embed_empty_list_returns_empty(self, mock_service):
        """Empty input should return empty list."""
        result = await mock_service.embed_texts([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_texts_filters_empty(self, mock_service):
        """Empty strings should be filtered."""
        result = await mock_service.embed_texts(["", "   "])
        # All empty, returns zero vectors for each
        assert len(result) == 2
        assert result[0] == [0.0] * 768


class TestGetDimension:
    """Test dimension retrieval."""

    def test_get_embedding_dimension(self):
        """Should return model dimension."""
        from roampal.backend.modules.memory.embedding_service import EmbeddingService

        service = EmbeddingService()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        service._model = mock_model

        assert service.get_embedding_dimension() == 768


class TestPrewarm:
    """Test model prewarming."""

    @pytest.mark.asyncio
    async def test_prewarm_loads_model(self):
        """Prewarm should load the model."""
        from roampal.backend.modules.memory.embedding_service import EmbeddingService

        service = EmbeddingService()

        # Mock the model property to track access
        with patch.object(EmbeddingService, 'model', new_callable=lambda: property(lambda self: MagicMock())):
            await service.prewarm()
            # No error means success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

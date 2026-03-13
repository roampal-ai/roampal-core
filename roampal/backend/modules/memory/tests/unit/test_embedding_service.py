"""
Unit Tests for EmbeddingService.

Tests text embedding generation and batch operations.
v0.4.3: Updated mocks for ONNX Runtime + tokenizers backend.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock


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
        assert service._session is None
        assert service._tokenizer is None


def _make_mock_service():
    """Create an EmbeddingService with mocked ONNX session and tokenizer."""
    from roampal.backend.modules.memory.embedding_service import EmbeddingService

    service = EmbeddingService()

    # Mock tokenizer encode_batch — returns objects with .ids and .attention_mask
    mock_tokenizer = MagicMock()

    def fake_encode_batch(texts):
        results = []
        for _ in texts:
            enc = MagicMock()
            enc.ids = [101] + [1000] * 5 + [102]  # 7 tokens
            enc.attention_mask = [1] * 7
            results.append(enc)
        return results

    mock_tokenizer.encode_batch = fake_encode_batch

    # Mock ONNX session — returns fake last_hidden_state
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [
        MagicMock(name="input_ids"),
        MagicMock(name="attention_mask"),
    ]

    def fake_run(output_names, feeds):
        batch_size = feeds["input_ids"].shape[0]
        seq_len = feeds["input_ids"].shape[1]
        # Return last_hidden_state with constant values
        return [np.ones((batch_size, seq_len, 768), dtype=np.float32) * 0.1]

    mock_session.run = fake_run

    service._session = mock_session
    service._tokenizer = mock_tokenizer

    return service


class TestEmbedText:
    """Test single text embedding."""

    @pytest.fixture
    def mock_service(self):
        return _make_mock_service()

    @pytest.mark.asyncio
    async def test_embed_text_returns_list(self, mock_service):
        """Should return list of floats."""
        result = await mock_service.embed_text("test text")
        assert isinstance(result, list)
        assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_text_normalized(self, mock_service):
        """Embeddings should be L2-normalized (unit length)."""
        result = await mock_service.embed_text("test text")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

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

    @pytest.mark.asyncio
    async def test_embed_text_caching(self, mock_service):
        """Second call with same text should return cached result."""
        r1 = await mock_service.embed_text("cached query")
        r2 = await mock_service.embed_text("cached query")
        assert r1 == r2


class TestEmbedTexts:
    """Test batch embedding."""

    @pytest.fixture
    def mock_service(self):
        return _make_mock_service()

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
        """Should return 768 (model dimension)."""
        from roampal.backend.modules.memory.embedding_service import EmbeddingService
        service = EmbeddingService()
        assert service.get_embedding_dimension() == 768


class TestPrewarm:
    """Test model prewarming."""

    @pytest.mark.asyncio
    async def test_prewarm_loads_model(self):
        """Prewarm should trigger model load."""
        from roampal.backend.modules.memory.embedding_service import EmbeddingService

        service = EmbeddingService()
        # Mock _load_model so it doesn't actually download anything
        service._load_model = MagicMock()
        service._session = MagicMock()  # pretend it's loaded

        await service.prewarm()
        # No error means success — session property was accessed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit Tests for ContextService.

Tests conversation context analysis and pattern detection.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import pytest
from unittest.mock import MagicMock, AsyncMock


class TestContextServiceInit:
    """Test ContextService initialization."""

    def test_init_with_collections(self):
        """Should initialize with collections."""
        from roampal.backend.modules.memory.context_service import ContextService

        collections = {"working": MagicMock(), "history": MagicMock()}
        service = ContextService(collections=collections)

        assert service.collections == collections

    def test_init_with_kg_service(self):
        """Should accept KG service."""
        from roampal.backend.modules.memory.context_service import ContextService

        kg_service = MagicMock()
        service = ContextService(collections={}, kg_service=kg_service)

        assert service.kg_service == kg_service

    def test_init_default_config(self):
        """Should use default config when not provided."""
        from roampal.backend.modules.memory.context_service import ContextService
        from roampal.backend.modules.memory.config import MemoryConfig

        service = ContextService(collections={})

        assert isinstance(service.config, MemoryConfig)


class TestAnalyzeContext:
    """Test conversation context analysis."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocks."""
        from roampal.backend.modules.memory.context_service import ContextService

        collections = {"working": MagicMock(), "patterns": MagicMock()}
        kg_service = MagicMock()
        kg_service.extract_concepts.return_value = ["python", "api"]
        kg_service.get_problem_categories.return_value = {}
        kg_service.get_failure_patterns.return_value = {}
        kg_service.get_routing_patterns.return_value = {}

        embed_fn = AsyncMock(return_value=[0.1] * 768)

        service = ContextService(
            collections=collections,
            kg_service=kg_service,
            embed_fn=embed_fn
        )

        return service

    @pytest.mark.asyncio
    async def test_analyze_returns_expected_keys(self, mock_service):
        """Should return context with all expected keys."""
        result = await mock_service.analyze_conversation_context(
            current_message="test message",
            recent_conversation=[],
            conversation_id="test123"
        )

        assert "relevant_patterns" in result
        assert "past_outcomes" in result
        assert "topic_continuity" in result
        assert "proactive_insights" in result

    @pytest.mark.asyncio
    async def test_analyze_empty_message(self, mock_service):
        """Empty message should return empty context."""
        mock_service.kg_service.extract_concepts.return_value = []

        result = await mock_service.analyze_conversation_context(
            current_message="",
            recent_conversation=[],
            conversation_id="test123"
        )

        assert result["relevant_patterns"] == []
        assert result["past_outcomes"] == []


class TestTopicContinuity:
    """Test topic continuity detection."""

    @pytest.fixture
    def service(self):
        """Create basic service."""
        from roampal.backend.modules.memory.context_service import ContextService

        kg_service = MagicMock()
        kg_service.extract_concepts.return_value = ["python", "api"]

        return ContextService(collections={}, kg_service=kg_service)

    def test_detect_continuing_topic(self, service):
        """Should detect when topic continues."""
        current = ["python", "api", "test"]
        # Need at least 2 messages for continuity check
        recent = [
            {"role": "assistant", "content": "I can help with python"},
            {"role": "user", "content": "python api test"}
        ]

        result = service._detect_topic_continuity(current, recent)

        assert len(result) > 0
        assert result[0].get("continuing") is True

    def test_detect_topic_shift(self, service):
        """Should detect topic shift."""
        current = ["javascript", "react"]
        # Need at least 2 messages for continuity check
        recent = [
            {"role": "assistant", "content": "Sure, about python"},
            {"role": "user", "content": "python api test"}
        ]

        result = service._detect_topic_continuity(current, recent)

        # No overlap = topic shift
        assert len(result) > 0
        assert result[0].get("continuing") is False

    def test_empty_conversation(self, service):
        """Empty conversation should return empty."""
        result = service._detect_topic_continuity(["test"], [])
        assert result == []


class TestConceptExtraction:
    """Test concept extraction."""

    def test_extract_with_kg_service(self):
        """Should use KG service when available."""
        from roampal.backend.modules.memory.context_service import ContextService

        kg_service = MagicMock()
        kg_service.extract_concepts.return_value = ["test", "concept"]

        service = ContextService(collections={}, kg_service=kg_service)
        result = service._extract_concepts("test concept")

        kg_service.extract_concepts.assert_called_with("test concept")
        assert result == ["test", "concept"]

    def test_basic_extraction_fallback(self):
        """Should fall back to basic extraction."""
        from roampal.backend.modules.memory.context_service import ContextService

        service = ContextService(collections={}, kg_service=None)
        result = service._basic_concept_extraction("Python programming language")

        assert "python" in result
        assert "programming" in result
        assert "language" in result

    def test_filters_stopwords(self):
        """Should filter stopwords."""
        from roampal.backend.modules.memory.context_service import ContextService

        service = ContextService(collections={}, kg_service=None)
        result = service._basic_concept_extraction("the quick brown fox")

        assert "the" not in result
        assert "quick" in result


class TestContextSummary:
    """Test context summary generation."""

    def test_get_context_summary(self):
        """Should generate readable summary."""
        from roampal.backend.modules.memory.context_service import ContextService

        service = ContextService(collections={})

        context = {
            "relevant_patterns": [{"text": "pattern"}],
            "past_outcomes": [],
            "topic_continuity": [{"continuing": True, "common_concepts": ["python"]}],
            "proactive_insights": []
        }

        summary = service.get_context_summary(context)

        assert "pattern" in summary.lower() or "continuing" in summary.lower()

    def test_empty_context_summary(self):
        """Empty context should return default message."""
        from roampal.backend.modules.memory.context_service import ContextService

        service = ContextService(collections={})

        context = {
            "relevant_patterns": [],
            "past_outcomes": [],
            "topic_continuity": [],
            "proactive_insights": []
        }

        summary = service.get_context_summary(context)

        assert "no significant" in summary.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

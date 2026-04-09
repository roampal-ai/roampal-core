"""
Unit Tests for ContextService (v0.4.5).

v0.4.5: KG removed. Context service uses basic concept extraction only.
KG-dependent methods return empty results.
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

    def test_init_accepts_kwargs(self):
        """v0.4.5: Should accept and ignore kg_service via **kwargs."""
        from roampal.backend.modules.memory.context_service import ContextService
        service = ContextService(collections={}, kg_service=MagicMock())
        # Should not crash — **kwargs absorbs kg_service

    def test_init_default_config(self):
        """Should use default config when not provided."""
        from roampal.backend.modules.memory.context_service import ContextService
        from roampal.backend.modules.memory.config import MemoryConfig
        service = ContextService(collections={})
        assert isinstance(service.config, MemoryConfig)


class TestAnalyzeContext:
    """Test conversation context analysis."""

    @pytest.fixture
    def service(self):
        from roampal.backend.modules.memory.context_service import ContextService
        collections = {"working": MagicMock(), "patterns": MagicMock()}
        embed_fn = AsyncMock(return_value=[0.1] * 768)
        return ContextService(collections=collections, embed_fn=embed_fn)

    @pytest.mark.asyncio
    async def test_analyze_returns_expected_keys(self, service):
        """Should return context with all expected keys."""
        result = await service.analyze_conversation_context(
            current_message="test message",
            recent_conversation=[],
            conversation_id="test123"
        )
        assert "relevant_patterns" in result
        assert "past_outcomes" in result
        assert "topic_continuity" in result
        assert "proactive_insights" in result

    @pytest.mark.asyncio
    async def test_analyze_returns_empty_patterns(self, service):
        """v0.4.5: KG removed — patterns and outcomes should be empty."""
        result = await service.analyze_conversation_context(
            current_message="python api question",
            recent_conversation=[],
            conversation_id="test123"
        )
        assert result["relevant_patterns"] == []
        assert result["past_outcomes"] == []


class TestTopicContinuity:
    """Test topic continuity detection (KG-independent)."""

    @pytest.fixture
    def service(self):
        from roampal.backend.modules.memory.context_service import ContextService
        return ContextService(collections={})

    def test_detect_continuing_topic(self, service):
        current = ["python", "api", "test"]
        recent = [
            {"role": "assistant", "content": "I can help with python"},
            {"role": "user", "content": "python api test"}
        ]
        result = service._detect_topic_continuity(current, recent)
        assert len(result) > 0
        assert result[0].get("continuing") is True

    def test_detect_topic_shift(self, service):
        current = ["javascript", "react"]
        recent = [
            {"role": "assistant", "content": "Sure, about python"},
            {"role": "user", "content": "python api test"}
        ]
        result = service._detect_topic_continuity(current, recent)
        assert len(result) > 0
        assert result[0].get("continuing") is False

    def test_empty_conversation(self, service):
        result = service._detect_topic_continuity(["test"], [])
        assert result == []


class TestConceptExtraction:
    """Test concept extraction — v0.4.5: always uses basic extraction."""

    def test_extract_uses_basic_extraction(self):
        """v0.4.5: Should always use basic extraction (no KG)."""
        from roampal.backend.modules.memory.context_service import ContextService
        service = ContextService(collections={})
        result = service._extract_concepts("Python programming language")
        assert "python" in result
        assert "programming" in result

    def test_basic_extraction_filters_stopwords(self):
        from roampal.backend.modules.memory.context_service import ContextService
        service = ContextService(collections={})
        result = service._basic_concept_extraction("the quick brown fox")
        assert "the" not in result
        assert "quick" in result


class TestKnownSolutions:
    """v0.4.5: find_known_solutions always returns empty."""

    @pytest.mark.asyncio
    async def test_find_known_solutions_returns_empty(self):
        from roampal.backend.modules.memory.context_service import ContextService
        service = ContextService(collections={})
        result = await service.find_known_solutions("any query")
        assert result == []


class TestContextSummary:
    """Test context summary generation."""

    def test_get_context_summary(self):
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

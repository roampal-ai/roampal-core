"""
Unit Tests for PromotionService.

Tests automatic memory promotion/demotion between collections.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timedelta


class TestPromotionServiceInit:
    """Test PromotionService initialization."""

    def test_init_with_collections(self):
        """Should initialize with collections."""
        from roampal.backend.modules.memory.promotion_service import PromotionService

        collections = {"working": MagicMock(), "history": MagicMock()}
        embed_fn = AsyncMock(return_value=[0.1] * 768)

        service = PromotionService(collections=collections, embed_fn=embed_fn)

        assert service.collections == collections

    def test_init_default_config(self):
        """Should use default config."""
        from roampal.backend.modules.memory.promotion_service import PromotionService
        from roampal.backend.modules.memory.config import MemoryConfig

        service = PromotionService(collections={}, embed_fn=AsyncMock())

        assert isinstance(service.config, MemoryConfig)


class TestPromotionWorkingToHistory:
    """Test working -> history promotion."""

    @pytest.fixture
    def service(self):
        """Create service with mocked collections."""
        from roampal.backend.modules.memory.promotion_service import PromotionService

        working = MagicMock()
        working.get_fragment.return_value = {
            "content": "test content",
            "metadata": {"text": "test content", "score": 0.8, "uses": 3}
        }
        working.delete_vectors = MagicMock()

        history = MagicMock()
        history.upsert_vectors = AsyncMock()

        collections = {"working": working, "history": history, "patterns": MagicMock()}
        embed_fn = AsyncMock(return_value=[0.1] * 768)

        return PromotionService(collections=collections, embed_fn=embed_fn)

    @pytest.mark.asyncio
    async def test_promote_high_score(self, service):
        """High score should trigger promotion."""
        result = await service.handle_promotion(
            doc_id="working_test123",
            collection="working",
            score=0.8,
            uses=3,
            metadata={"text": "test content", "score": 0.8, "uses": 3}
        )

        assert result is not None
        assert result.startswith("history_")
        service.collections["history"].upsert_vectors.assert_called_once()
        service.collections["working"].delete_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_promote_low_score(self, service):
        """Low score should not promote."""
        result = await service.handle_promotion(
            doc_id="working_test123",
            collection="working",
            score=0.3,  # Below threshold
            uses=3,
            metadata={"text": "test", "score": 0.3, "uses": 3}
        )

        assert result is None


class TestPromotionHistoryToPatterns:
    """Test history -> patterns promotion."""

    @pytest.fixture
    def service(self):
        """Create service for history promotion."""
        from roampal.backend.modules.memory.promotion_service import PromotionService

        history = MagicMock()
        history.get_fragment.return_value = {
            "content": "proven pattern",
            "metadata": {"text": "proven pattern", "score": 0.95, "uses": 5}
        }
        history.delete_vectors = MagicMock()

        patterns = MagicMock()
        patterns.upsert_vectors = AsyncMock()

        collections = {"working": MagicMock(), "history": history, "patterns": patterns}
        embed_fn = AsyncMock(return_value=[0.1] * 768)

        return PromotionService(collections=collections, embed_fn=embed_fn)

    @pytest.mark.asyncio
    async def test_promote_to_patterns(self, service):
        """Very high score should promote to patterns."""
        result = await service.handle_promotion(
            doc_id="history_test123",
            collection="history",
            score=0.95,  # Above high_value_threshold
            uses=5,
            metadata={"text": "proven pattern", "score": 0.95, "uses": 5}
        )

        assert result is not None
        assert result.startswith("patterns_")


class TestDemotionPatternsToHistory:
    """Test patterns -> history demotion."""

    @pytest.fixture
    def service(self):
        """Create service for demotion."""
        from roampal.backend.modules.memory.promotion_service import PromotionService

        patterns = MagicMock()
        patterns.get_fragment.return_value = {
            "content": "failing pattern",
            "metadata": {"text": "failing pattern", "score": 0.2}
        }
        patterns.delete_vectors = MagicMock()

        history = MagicMock()
        history.upsert_vectors = AsyncMock()

        collections = {"working": MagicMock(), "history": history, "patterns": patterns}
        embed_fn = AsyncMock(return_value=[0.1] * 768)

        return PromotionService(collections=collections, embed_fn=embed_fn)

    @pytest.mark.asyncio
    async def test_demote_low_score_pattern(self, service):
        """Low score pattern should be demoted."""
        result = await service.handle_promotion(
            doc_id="patterns_test123",
            collection="patterns",
            score=0.2,  # Below demotion threshold
            uses=5,
            metadata={"text": "failing pattern", "score": 0.2}
        )

        assert result is not None
        assert result.startswith("history_")


class TestDeletion:
    """Test deletion of low-scoring memories."""

    @pytest.fixture
    def service(self):
        """Create service for deletion tests."""
        from roampal.backend.modules.memory.promotion_service import PromotionService

        working = MagicMock()
        working.get_fragment.return_value = {
            "content": "bad memory",
            "metadata": {"text": "bad memory", "score": 0.05, "timestamp": datetime.now().isoformat()}
        }
        working.delete_vectors = MagicMock()

        collections = {"working": working, "history": MagicMock(), "patterns": MagicMock()}
        embed_fn = AsyncMock(return_value=[0.1] * 768)

        return PromotionService(collections=collections, embed_fn=embed_fn)

    @pytest.mark.asyncio
    async def test_delete_very_low_score(self, service):
        """Very low score should trigger deletion."""
        await service.handle_promotion(
            doc_id="working_test123",
            collection="working",
            score=0.05,  # Below deletion threshold
            uses=1,
            metadata={"text": "bad memory", "score": 0.05, "timestamp": datetime.now().isoformat()}
        )

        service.collections["working"].delete_vectors.assert_called()


class TestBatchPromotion:
    """Test batch promotion of working memory."""

    @pytest.fixture
    def service(self):
        """Create service for batch promotion."""
        from roampal.backend.modules.memory.promotion_service import PromotionService

        working = MagicMock()
        working.list_all_ids.return_value = ["working_1", "working_2"]
        working.get_fragment.side_effect = [
            {"metadata": {"text": "good memory", "score": 0.8, "uses": 3, "timestamp": datetime.now().isoformat()}},
            {"metadata": {"text": "old memory", "score": 0.3, "uses": 0, "timestamp": (datetime.now() - timedelta(hours=30)).isoformat()}}
        ]
        working.delete_vectors = MagicMock()
        working.collection = True  # Exists

        history = MagicMock()
        history.upsert_vectors = AsyncMock()

        collections = {"working": working, "history": history, "patterns": MagicMock()}
        embed_fn = AsyncMock(return_value=[0.1] * 768)

        return PromotionService(collections=collections, embed_fn=embed_fn)

    @pytest.mark.asyncio
    async def test_batch_promote_valuable(self, service):
        """Should promote valuable memories in batch."""
        count = await service.promote_valuable_working_memory()

        # At least one should be promoted
        assert count >= 1


class TestAgeCalculation:
    """Test age calculation helper."""

    def test_calculate_age_hours(self):
        """Should calculate age in hours."""
        from roampal.backend.modules.memory.promotion_service import PromotionService

        service = PromotionService(collections={}, embed_fn=AsyncMock())

        # 2 hours ago
        two_hours_ago = (datetime.now() - timedelta(hours=2)).isoformat()
        age = service._calculate_age_hours(two_hours_ago)

        assert 1.9 < age < 2.1  # Approximately 2 hours

    def test_invalid_timestamp(self):
        """Invalid timestamp should return 0."""
        from roampal.backend.modules.memory.promotion_service import PromotionService

        service = PromotionService(collections={}, embed_fn=AsyncMock())
        age = service._calculate_age_hours("invalid")

        assert age == 0.0


class TestCleanupOldWorkingMemory:
    """Test cleanup of old working memory."""

    @pytest.fixture
    def service(self):
        """Create service for cleanup tests."""
        from roampal.backend.modules.memory.promotion_service import PromotionService

        working = MagicMock()
        working.list_all_ids.return_value = ["working_old"]
        working.get_fragment.return_value = {
            "metadata": {"timestamp": (datetime.now() - timedelta(hours=30)).isoformat()}
        }
        working.delete_vectors = MagicMock()
        working.collection = True

        collections = {"working": working}
        embed_fn = AsyncMock()

        return PromotionService(collections=collections, embed_fn=embed_fn)

    @pytest.mark.asyncio
    async def test_cleanup_removes_old_items(self, service):
        """Should remove items older than threshold."""
        count = await service.cleanup_old_working_memory(max_age_hours=24.0)

        assert count == 1
        service.collections["working"].delete_vectors.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

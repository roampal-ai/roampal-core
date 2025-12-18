"""
Unit Tests for MemoryBankService

Tests the extracted memory bank operations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import json
import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from roampal.backend.modules.memory.memory_bank_service import MemoryBankService
from roampal.backend.modules.memory.config import MemoryConfig


class TestMemoryBankServiceInit:
    """Test MemoryBankService initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default config."""
        collection = MagicMock()
        service = MemoryBankService(
            collection=collection,
            embed_fn=AsyncMock()
        )
        assert service.config is not None
        assert service.MAX_ITEMS == 500

    def test_init_with_custom_config(self):
        """Should use custom config."""
        config = MemoryConfig(promotion_score_threshold=0.8)
        service = MemoryBankService(
            collection=MagicMock(),
            embed_fn=AsyncMock(),
            config=config
        )
        assert service.config.promotion_score_threshold == 0.8


class TestStore:
    """Test memory storage."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.collection = MagicMock()
        coll.collection.count = MagicMock(return_value=10)
        coll.upsert_vectors = AsyncMock()
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock(return_value=[0.1] * 384)
        )

    @pytest.mark.asyncio
    async def test_store_basic(self, service, mock_collection):
        """Should store memory with correct metadata."""
        doc_id = await service.store(
            text="User prefers dark mode",
            tags=["preference"]
        )

        assert doc_id.startswith("memory_bank_")
        mock_collection.upsert_vectors.assert_called_once()

        call_args = mock_collection.upsert_vectors.call_args
        metadata = call_args[1]["metadatas"][0]

        assert metadata["text"] == "User prefers dark mode"
        assert metadata["status"] == "active"
        assert metadata["score"] == 1.0
        assert json.loads(metadata["tags"]) == ["preference"]

    @pytest.mark.asyncio
    async def test_store_with_importance_confidence(self, service, mock_collection):
        """Should store with custom importance/confidence."""
        await service.store(
            text="Critical info",
            tags=["identity"],
            importance=0.95,
            confidence=0.9
        )

        call_args = mock_collection.upsert_vectors.call_args
        metadata = call_args[1]["metadatas"][0]

        assert metadata["importance"] == 0.95
        assert metadata["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_store_capacity_check(self, service, mock_collection):
        """Should reject when at capacity."""
        mock_collection.collection.count = MagicMock(return_value=500)

        with pytest.raises(ValueError, match="capacity"):
            await service.store("Test", ["test"])


class TestUpdate:
    """Test memory update with archiving."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value={
            "content": "old content",
            "metadata": {
                "text": "old content",
                "tags": '["identity"]',
                "importance": 0.7,
                "confidence": 0.7
            }
        })
        coll.upsert_vectors = AsyncMock()
        coll.collection = MagicMock()
        coll.collection.count = MagicMock(return_value=10)
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock(return_value=[0.1] * 384)
        )

    @pytest.mark.asyncio
    async def test_update_archives_old(self, service, mock_collection):
        """Should archive old version when updating."""
        await service.update(
            doc_id="memory_bank_test123",
            new_text="new content",
            reason="correction"
        )

        # Should have 2 upsert calls: archive + update
        assert mock_collection.upsert_vectors.call_count == 2

        # Check archive call
        archive_call = mock_collection.upsert_vectors.call_args_list[0]
        archive_id = archive_call[1]["ids"][0]
        archive_metadata = archive_call[1]["metadatas"][0]

        assert "archived" in archive_id
        assert archive_metadata["status"] == "archived"
        assert archive_metadata["archive_reason"] == "correction"

    @pytest.mark.asyncio
    async def test_update_preserves_metadata(self, service, mock_collection):
        """Should preserve original metadata fields."""
        await service.update(
            doc_id="memory_bank_test123",
            new_text="new content"
        )

        update_call = mock_collection.upsert_vectors.call_args_list[1]
        metadata = update_call[1]["metadatas"][0]

        assert metadata["importance"] == 0.7
        assert metadata["text"] == "new content"

    @pytest.mark.asyncio
    async def test_update_not_found_creates_new(self, service, mock_collection):
        """Should create new memory if not found."""
        mock_collection.get_fragment = MagicMock(return_value=None)

        doc_id = await service.update(
            doc_id="memory_bank_nonexistent",
            new_text="new memory"
        )

        assert doc_id.startswith("memory_bank_")


class TestArchive:
    """Test memory archiving."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {"status": "active"}
        })
        coll.delete_vectors = MagicMock()
        return coll

    @pytest.fixture
    def mock_search_fn(self):
        """Mock search function that returns matching results."""
        async def search_fn(query, collections, limit=5):
            return [{"id": "memory_bank_test123", "content": query}]
        return search_fn

    @pytest.fixture
    def service(self, mock_collection, mock_search_fn):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock(),
            search_fn=mock_search_fn
        )

    @pytest.mark.asyncio
    async def test_archive_success(self, service, mock_collection):
        """Should delete memory successfully (archive = hard delete)."""
        result = await service.archive(
            content="test content to archive",
            reason="outdated"
        )

        assert result is True
        mock_collection.delete_vectors.assert_called_once_with(["memory_bank_test123"])

    @pytest.mark.asyncio
    async def test_archive_not_found(self, mock_collection):
        """Should return False if not found."""
        # Service with search that returns no results
        async def empty_search(query, collections, limit=5):
            return []
        service = MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock(),
            search_fn=empty_search
        )

        result = await service.archive("nonexistent")
        assert result is False


class TestSearch:
    """Test memory search."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.list_all_ids = MagicMock(return_value=[
            "memory_bank_1", "memory_bank_2", "memory_bank_3"
        ])

        def get_fragment_side_effect(doc_id):
            if doc_id == "memory_bank_1":
                return {
                    "content": "User name is John",
                    "metadata": {"status": "active", "tags": '["identity"]'}
                }
            elif doc_id == "memory_bank_2":
                return {
                    "content": "Prefers dark mode",
                    "metadata": {"status": "active", "tags": '["preference"]'}
                }
            elif doc_id == "memory_bank_3":
                return {
                    "content": "Old info",
                    "metadata": {"status": "archived", "tags": '["identity"]'}
                }
            return None

        coll.get_fragment = MagicMock(side_effect=get_fragment_side_effect)
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    @pytest.mark.asyncio
    async def test_search_excludes_archived_by_default(self, service):
        """Should exclude archived memories by default."""
        results = await service.search()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_includes_archived_when_requested(self, service):
        """Should include archived when requested."""
        results = await service.search(include_archived=True)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_filters_by_tags(self, service):
        """Should filter by tags."""
        results = await service.search(tags=["identity"])
        assert len(results) == 1
        assert results[0]["metadata"]["tags"] == '["identity"]'


class TestRestore:
    """Test memory restoration."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {"status": "archived"}
        })
        coll.update_fragment_metadata = MagicMock()
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    @pytest.mark.asyncio
    async def test_restore_success(self, service, mock_collection):
        """Should restore archived memory."""
        result = await service.restore("memory_bank_test123")

        assert result is True
        call_args = mock_collection.update_fragment_metadata.call_args
        metadata = call_args[0][1]
        assert metadata["status"] == "active"
        assert metadata["restored_by"] == "user"

    @pytest.mark.asyncio
    async def test_restore_not_found(self, service, mock_collection):
        """Should return False if not found."""
        mock_collection.get_fragment = MagicMock(return_value=None)

        result = await service.restore("nonexistent")
        assert result is False


class TestDelete:
    """Test memory deletion."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.delete_vectors = MagicMock()
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    @pytest.mark.asyncio
    async def test_delete_success(self, service, mock_collection):
        """Should delete memory successfully."""
        result = await service.delete("memory_bank_test123")

        assert result is True
        mock_collection.delete_vectors.assert_called_with(["memory_bank_test123"])

    @pytest.mark.asyncio
    async def test_delete_failure(self, service, mock_collection):
        """Should return False on error."""
        mock_collection.delete_vectors = MagicMock(side_effect=Exception("Test error"))

        result = await service.delete("memory_bank_test123")
        assert result is False


class TestListAll:
    """Test listing all memories."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.list_all_ids = MagicMock(return_value=["m1", "m2", "m3"])

        def get_fragment_side_effect(doc_id):
            return {
                "content": f"content_{doc_id}",
                "metadata": {
                    "status": "active" if doc_id != "m3" else "archived",
                    "tags": '["identity"]' if doc_id == "m1" else '["preference"]'
                }
            }

        coll.get_fragment = MagicMock(side_effect=get_fragment_side_effect)
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    def test_list_all_excludes_archived(self, service):
        """Should exclude archived by default."""
        results = service.list_all()
        assert len(results) == 2

    def test_list_all_includes_archived(self, service):
        """Should include archived when requested."""
        results = service.list_all(include_archived=True)
        assert len(results) == 3

    def test_list_all_filters_tags(self, service):
        """Should filter by tags."""
        results = service.list_all(tags=["identity"])
        assert len(results) == 1


class TestStats:
    """Test statistics retrieval."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.list_all_ids = MagicMock(return_value=["m1", "m2", "m3"])

        def get_fragment_side_effect(doc_id):
            if doc_id == "m1":
                return {
                    "content": "identity",
                    "metadata": {
                        "status": "active",
                        "tags": '["identity"]',
                        "importance": 0.9,
                        "confidence": 0.8
                    }
                }
            elif doc_id == "m2":
                return {
                    "content": "preference",
                    "metadata": {
                        "status": "active",
                        "tags": '["preference", "identity"]',
                        "importance": 0.7,
                        "confidence": 0.7
                    }
                }
            elif doc_id == "m3":
                return {
                    "content": "archived",
                    "metadata": {
                        "status": "archived",
                        "tags": '["old"]',
                        "importance": 0.5,
                        "confidence": 0.5
                    }
                }
            return None

        coll.get_fragment = MagicMock(side_effect=get_fragment_side_effect)
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    def test_get_stats(self, service):
        """Should return correct statistics."""
        stats = service.get_stats()

        assert stats["total"] == 3
        assert stats["active"] == 2
        assert stats["archived"] == 1
        assert stats["capacity"] == 500
        assert stats["tag_counts"]["identity"] == 2
        assert stats["tag_counts"]["preference"] == 1
        assert stats["avg_importance"] == 0.8  # (0.9 + 0.7) / 2
        assert stats["avg_confidence"] == 0.75  # (0.8 + 0.7) / 2


class TestIncrementMention:
    """Test mention count tracking."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {"mentioned_count": 5}
        })
        coll.update_fragment_metadata = MagicMock()
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    def test_increment_mention(self, service, mock_collection):
        """Should increment mention count."""
        result = service.increment_mention("memory_bank_test123")

        assert result is True
        call_args = mock_collection.update_fragment_metadata.call_args
        metadata = call_args[0][1]
        assert metadata["mentioned_count"] == 6
        assert "last_mentioned" in metadata

    def test_increment_not_found(self, service, mock_collection):
        """Should return False if not found."""
        mock_collection.get_fragment = MagicMock(return_value=None)

        result = service.increment_mention("nonexistent")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

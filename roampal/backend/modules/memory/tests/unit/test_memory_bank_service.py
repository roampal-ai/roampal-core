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
        coll.collection.get = MagicMock(return_value={
            "ids": ["memory_bank_1"],
            "metadatas": [{"status": "active"}]
        })
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
        mock_collection.collection.get = MagicMock(return_value={
            "ids": [f"memory_bank_{i}" for i in range(500)],
            "metadatas": [{"status": "active"}] * 500
        })
        # v0.5.6: _maybe_cleanup_archived needs total count too; with all active, ratio < 0.5 so no cleanup triggers
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
        coll.collection.get = MagicMock(return_value={
            "ids": ["memory_bank_1"],
            "metadatas": [{"status": "active"}]
        })
        coll.upsert_vectors = AsyncMock()
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock(return_value=[0.1] * 384)
        )

    @pytest.mark.asyncio
    async def test_update_in_place_no_archive(self, service, mock_collection):
        """Should update in place without creating archive copy (v0.2.9+)."""
        await service.update(
            doc_id="memory_bank_test123",
            new_text="new content",
            reason="correction"
        )

        # Should have only 1 upsert call (no archive copy)
        assert mock_collection.upsert_vectors.call_count == 1

        # Check update call
        update_call = mock_collection.upsert_vectors.call_args_list[0]
        update_id = update_call[1]["ids"][0]
        update_metadata = update_call[1]["metadatas"][0]

        assert update_id == "memory_bank_test123"
        assert update_metadata["text"] == "new content"
        assert update_metadata["update_reason"] == "correction"

    @pytest.mark.asyncio
    async def test_update_preserves_metadata(self, service, mock_collection):
        """Should preserve original metadata fields."""
        await service.update(
            doc_id="memory_bank_test123",
            new_text="new content"
        )

        update_call = mock_collection.upsert_vectors.call_args_list[0]
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


class TestUpdateById:
    """v0.5.6: update_by_id is the only update path. No semantic search, no fallback."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value={
            "content": "User prefers dark mode",
            "metadata": {
                "text": "User prefers dark mode",
                "tags": '["preference"]',
                "importance": 0.7,
                "confidence": 0.8,
            }
        })
        coll.upsert_vectors = AsyncMock()
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock(return_value=[0.1] * 384),
        )

    @pytest.mark.asyncio
    async def test_update_by_id_success(self, service, mock_collection):
        """Direct id lookup, content replaced, doc_id returned."""
        doc_id = await service.update_by_id(
            doc_id="memory_bank_match123",
            new_content="User switched to light mode",
        )

        assert doc_id == "memory_bank_match123"
        mock_collection.upsert_vectors.assert_called_once()

        call_args = mock_collection.upsert_vectors.call_args
        assert call_args[1]["ids"] == ["memory_bank_match123"]
        update_metadata = call_args[1]["metadatas"][0]
        assert update_metadata["text"] == "User switched to light mode"
        assert update_metadata["content"] == "User switched to light mode"
        assert update_metadata["update_reason"] == "mcp_update"

    @pytest.mark.asyncio
    async def test_update_by_id_overwrites_provided_metadata(self, service, mock_collection):
        """Provided metadata fields overwrite; omitted ones preserve."""
        await service.update_by_id(
            doc_id="memory_bank_match123",
            new_content="Updated content",
            tags=["identity"],
            importance=0.95,
        )

        call_args = mock_collection.upsert_vectors.call_args
        update_metadata = call_args[1]["metadatas"][0]
        assert json.loads(update_metadata["tags"]) == ["identity"]
        assert update_metadata["importance"] == 0.95
        assert update_metadata["confidence"] == 0.8  # preserved from old

    @pytest.mark.asyncio
    async def test_update_by_id_preserves_metadata_when_not_provided(self, service, mock_collection):
        """When optional kwargs are omitted, existing metadata is kept verbatim."""
        await service.update_by_id(
            doc_id="memory_bank_match123",
            new_content="New text only",
        )

        call_args = mock_collection.upsert_vectors.call_args
        update_metadata = call_args[1]["metadatas"][0]
        assert json.loads(update_metadata["tags"]) == ["preference"]
        assert update_metadata["importance"] == 0.7
        assert update_metadata["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_update_by_id_returns_none_when_not_found(self):
        """Unknown doc_id returns None — no semantic fallback, no new entry."""
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value=None)
        coll.upsert_vectors = AsyncMock()

        service = MemoryBankService(
            collection=coll,
            embed_fn=AsyncMock(return_value=[0.1] * 384),
        )

        result = await service.update_by_id(
            doc_id="memory_bank_deadbeef",
            new_content="should never write",
        )
        assert result is None
        coll.upsert_vectors.assert_not_called()


class TestArchive:
    """Test memory archiving."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {"status": "active"}
        })
        coll.update_fragment_metadata = MagicMock()
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
    async def test_archive_soft_delete(self, service, mock_collection):
        """v0.5.5: Should soft-delete by setting status=archived (not hard delete)."""
        result = await service.archive(
            content="test content to archive",
            reason="outdated"
        )

        assert result is True
        # v0.5.5: archive() no longer calls delete_vectors — it updates metadata
        mock_collection.update_fragment_metadata.assert_called_once()
        call_args = mock_collection.update_fragment_metadata.call_args
        doc_id = call_args[0][0]
        metadata = call_args[0][1]

        assert doc_id == "memory_bank_test123"
        assert metadata["status"] == "archived"
        assert metadata["archive_reason"] == "outdated"
        assert "archived_at" in metadata

    @pytest.mark.asyncio
    async def test_archive_skips_already_archived(self, mock_collection):
        """v0.5.5: Should skip already-archived entries when searching for target."""
        coll = MagicMock()
        coll.get_fragment = MagicMock(return_value={
            "content": "test",
            "metadata": {"status": "active"}
        })
        coll.update_fragment_metadata = MagicMock()

        async def search_returns_archived(query, collections, limit=5):
            return [
                {"id": "memory_bank_old1", "content": query, "metadata": {"status": "archived"}},
                {"id": "memory_bank_active1", "content": query, "metadata": {"status": "active"}}
            ]

        service = MemoryBankService(
            collection=coll,
            embed_fn=AsyncMock(),
            search_fn=search_returns_archived
        )

        result = await service.archive("test content to archive", reason="outdated")
        assert result is True
        call_args = coll.update_fragment_metadata.call_args
        # Should pick the active one, not the archived one
        assert call_args[0][0] == "memory_bank_active1"

    @pytest.mark.asyncio
    async def test_archive_not_found(self, mock_collection):
        """Should return False if not found."""
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


class TestDeletePermanent:
    """v0.5.6: delete_permanent requires force=True to prevent accidental hard-deletes."""

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
    async def test_delete_permanent_success_with_force(self, service, mock_collection):
        """Should delete memory successfully when force=True."""
        result = await service.delete_permanent("memory_bank_test123", force=True)

        assert result is True
        mock_collection.delete_vectors.assert_called_with(["memory_bank_test123"])

    @pytest.mark.asyncio
    async def test_delete_permanent_failure(self, service, mock_collection):
        """Should return False on error."""
        mock_collection.delete_vectors = MagicMock(side_effect=Exception("Test error"))

        result = await service.delete_permanent("memory_bank_test123", force=True)
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_permanent_requires_force_flag(self, service):
        """Calling without force=True must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="force=True"):
            await service.delete_permanent("memory_bank_test123")

    @pytest.mark.asyncio
    async def test_delete_permanent_raises_on_false_force(self, service):
        """Explicitly passing force=False must also raise."""
        with pytest.raises(RuntimeError, match="force=True"):
            await service.delete_permanent("memory_bank_test123", force=False)


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


class TestGetCount:
    """Test capacity count excludes archived entries (v0.5.5)."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.collection.get = MagicMock(return_value={
            "ids": ["memory_bank_1", "memory_bank_2", "memory_bank_3"],
            "metadatas": [
                {"status": "active"},
                {"status": "archived"},
                {"status": "active"}
            ]
        })
        return coll

    @pytest.fixture
    def service(self, mock_collection):
        return MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )

    def test_get_count_excludes_archived(self, service):
        """v0.5.5: Should only count active entries."""
        count = service._get_count()
        assert count == 2

    def test_get_count_all_active(self, mock_collection):
        """Should return correct count when all are active."""
        mock_collection.collection.get = MagicMock(return_value={
            "ids": ["m1", "m2"],
            "metadatas": [
                {"status": "active"},
                {"status": "active"}
            ]
        })
        service = MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )
        assert service._get_count() == 2

    def test_get_count_all_archived(self, mock_collection):
        """Should return 0 when all entries are archived."""
        mock_collection.collection.get = MagicMock(return_value={
            "ids": ["m1", "m2"],
            "metadatas": [
                {"status": "archived"},
                {"status": "archived"}
            ]
        })
        service = MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )
        assert service._get_count() == 0


class TestMaybeCleanupArchived:
    """v0.5.6: _maybe_cleanup_archived auto-cleanup under capacity pressure."""

    @pytest.fixture
    def mock_collection(self):
        coll = MagicMock()
        coll.collection.count = MagicMock(return_value=800)
        coll.list_all_ids = MagicMock(return_value=["a1"])
        coll.get_fragment = MagicMock(return_value={
            "content": "old",
            "metadata": {"status": "archived"},
        })
        coll.delete_vectors = MagicMock()
        return coll

    def test_no_cleanup_below_active_threshold(self, mock_collection):
        """Active count below ACTIVE_THRESHOLD (400) → no cleanup."""
        mock_collection.collection.get = MagicMock(return_value={
            "ids": list(range(300)),
            "metadatas": [{"status": "active"}] * 300,
        })
        service = MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )
        service._maybe_cleanup_archived()
        mock_collection.delete_vectors.assert_not_called()

    def test_no_cleanup_low_archived_ratio(self, mock_collection):
        """Active above threshold but archived ratio < 50% → no cleanup."""
        # active=410 out of total=600 → archived=190, ratio ≈ 0.32 < 0.5
        mock_collection.collection.count = MagicMock(return_value=600)
        mock_collection.collection.get = MagicMock(return_value={
            "ids": list(range(410)),
            "metadatas": [{"status": "active"}] * 410,
        })
        service = MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )
        service._maybe_cleanup_archived()
        mock_collection.delete_vectors.assert_not_called()

    def test_triggers_cleanup_high_archived_ratio(self, mock_collection):
        """Active above threshold and archived ratio >= 50% → cleanup triggers."""
        # active=410 out of total=820 → archived=410, ratio = 0.5 >= 0.5
        mock_collection.collection.count = MagicMock(return_value=820)
        mock_collection.collection.get = MagicMock(return_value={
            "ids": list(range(410)),
            "metadatas": [{"status": "active"}] * 410,
        })
        service = MemoryBankService(
            collection=mock_collection,
            embed_fn=AsyncMock()
        )
        service._maybe_cleanup_archived()
        mock_collection.delete_vectors.assert_called_once()

    def test_no_cleanup_when_total_zero(self):
        """Total count of 0 → no cleanup (division guard)."""
        coll = MagicMock()
        coll.collection.count = MagicMock(return_value=0)
        coll.collection.get = MagicMock(return_value={"ids": [], "metadatas": []})
        service = MemoryBankService(
            collection=coll,
            embed_fn=AsyncMock()
        )
        service._maybe_cleanup_archived()
        coll.delete_vectors.assert_not_called()


class TestCleanupArchivedSweepsPhantoms:
    """v0.5.6 Item 5: cleanup_archived() must sweep phantoms its own delete_vectors leaves behind.

    ChromaDB's delete_vectors() removes the document/metadata but leaves the ID in
    list_all_ids() until the next HNSW rebuild. Those phantom IDs poison dedup
    ($ne archived returns them as candidates with no content). Issue #8 root cause.
    """

    def test_cleanup_archived_invokes_sweep_after_delete(self):
        """After deleting archived IDs, _sweep_phantoms must run to clear leftovers."""
        coll = MagicMock()
        coll.list_all_ids = MagicMock(return_value=["m1", "m2_archived_xyz"])
        coll.get_fragment = MagicMock(return_value={
            "metadata": {"status": "archived"}
        })
        coll.delete_vectors = MagicMock()
        service = MemoryBankService(collection=coll, embed_fn=AsyncMock())

        # Spy on _sweep_phantoms to confirm it's called after delete_vectors
        sweep_calls = []
        original_sweep = service._sweep_phantoms

        def spy_sweep():
            sweep_calls.append(len(coll.delete_vectors.call_args_list))
            return original_sweep()

        service._sweep_phantoms = spy_sweep
        service.cleanup_archived()

        # _sweep_phantoms ran exactly once, after the archived delete_vectors call
        assert len(sweep_calls) == 1
        assert sweep_calls[0] >= 1, "sweep ran before delete_vectors — wrong order"

    def test_cleanup_archived_ends_with_no_phantoms(self):
        """End-to-end: archived delete leaves phantoms, sweep removes them, post-state is clean."""
        ids_state = {"m1", "m2"}
        fragments = {
            "m1": {"metadata": {"status": "archived"}},
            "m2": {"metadata": {"status": "archived"}},
        }

        coll = MagicMock()
        coll.list_all_ids = MagicMock(side_effect=lambda: list(ids_state))

        def get_frag(doc_id):
            return fragments.get(doc_id)
        coll.get_fragment = MagicMock(side_effect=get_frag)

        def delete_vecs(target_ids):
            # Simulate ChromaDB: drop fragments but leave IDs in the index until rebuild
            for tid in target_ids:
                fragments.pop(tid, None)
        coll.delete_vectors = MagicMock(side_effect=delete_vecs)

        service = MemoryBankService(collection=coll, embed_fn=AsyncMock())
        deleted = service.cleanup_archived()

        assert deleted == 2
        # delete_vectors was called twice: once for archived IDs, once for phantoms
        assert coll.delete_vectors.call_count == 2
        first_call = set(coll.delete_vectors.call_args_list[0][0][0])
        second_call = set(coll.delete_vectors.call_args_list[1][0][0])
        assert first_call == {"m1", "m2"}
        # Phantom sweep targets the same IDs ChromaDB still lists despite no fragment
        assert second_call == {"m1", "m2"}

    def test_cleanup_archived_no_phantoms_to_sweep(self):
        """If delete_vectors removes IDs from the index too, sweep is a no-op."""
        ids_state = {"m1"}
        fragments = {"m1": {"metadata": {"status": "archived"}}}

        coll = MagicMock()
        coll.list_all_ids = MagicMock(side_effect=lambda: list(ids_state))
        coll.get_fragment = MagicMock(side_effect=lambda did: fragments.get(did))

        def delete_vecs(target_ids):
            for tid in target_ids:
                fragments.pop(tid, None)
                ids_state.discard(tid)  # well-behaved index drops the ID too
        coll.delete_vectors = MagicMock(side_effect=delete_vecs)

        service = MemoryBankService(collection=coll, embed_fn=AsyncMock())
        deleted = service.cleanup_archived()

        assert deleted == 1
        # Only the archived delete fired; sweep had nothing to do
        assert coll.delete_vectors.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

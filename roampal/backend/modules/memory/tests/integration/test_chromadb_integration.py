"""
Integration Tests for ChromaDBAdapter.

Tests real ChromaDB operations using temporary in-memory storage.
These tests verify the actual vector database behavior.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import pytest
import tempfile
import shutil
from typing import List

# Ensure embedded mode for tests
os.environ["ROAMPAL_USE_SERVER"] = "false"


class TestChromaDBIntegration:
    """Integration tests using real ChromaDB with temp directory."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp(prefix="roampal_test_")
        yield temp_dir
        # Cleanup after test
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    @pytest.fixture
    async def adapter(self, temp_db_path):
        """Create initialized adapter."""
        from roampal.backend.modules.memory.chromadb_adapter import ChromaDBAdapter

        adapter = ChromaDBAdapter(
            persistence_directory=temp_db_path,
            use_server=False
        )
        await adapter.initialize(collection_name="test_collection")
        yield adapter
        await adapter.cleanup()

    @pytest.mark.asyncio
    async def test_upsert_and_query(self, adapter):
        """Should store and retrieve vectors correctly."""
        # Store a vector
        test_vector = [0.1] * 768
        await adapter.upsert_vectors(
            ids=["test_1"],
            vectors=[test_vector],
            metadatas=[{"text": "test content", "score": 0.5}]
        )

        # Query with same vector
        results = await adapter.query_vectors(test_vector, top_k=5)

        assert len(results) == 1
        assert results[0]["id"] == "test_1"
        assert "test content" in results[0]["text"]

    @pytest.mark.asyncio
    async def test_multiple_vectors(self, adapter):
        """Should handle multiple vectors correctly."""
        vectors = [[0.1 * i] * 768 for i in range(1, 4)]
        ids = ["vec_1", "vec_2", "vec_3"]
        metadatas = [
            {"text": "first document", "score": 0.9},
            {"text": "second document", "score": 0.7},
            {"text": "third document", "score": 0.5}
        ]

        await adapter.upsert_vectors(ids=ids, vectors=vectors, metadatas=metadatas)

        count = await adapter.get_collection_count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_delete_vectors(self, adapter):
        """Should delete vectors correctly."""
        # Add vectors
        await adapter.upsert_vectors(
            ids=["del_1", "del_2"],
            vectors=[[0.1] * 768, [0.2] * 768],
            metadatas=[{"text": "a"}, {"text": "b"}]
        )

        # Delete one
        adapter.delete_vectors(["del_1"])

        # Verify
        count = await adapter.get_collection_count()
        assert count == 1

        # Check remaining
        all_ids = adapter.list_all_ids()
        assert "del_1" not in all_ids
        assert "del_2" in all_ids

    @pytest.mark.asyncio
    async def test_get_fragment(self, adapter):
        """Should retrieve specific fragment by ID."""
        await adapter.upsert_vectors(
            ids=["frag_1"],
            vectors=[[0.5] * 768],
            metadatas=[{"text": "fragment content", "score": 0.8}]
        )

        frag = adapter.get_fragment("frag_1")

        assert frag is not None
        assert frag["id"] == "frag_1"
        assert frag["metadata"]["score"] == 0.8

    @pytest.mark.asyncio
    async def test_get_nonexistent_fragment(self, adapter):
        """Should return None for nonexistent fragment."""
        frag = adapter.get_fragment("nonexistent_id")
        assert frag is None

    @pytest.mark.asyncio
    async def test_update_fragment_metadata(self, adapter):
        """Should update metadata correctly."""
        await adapter.upsert_vectors(
            ids=["update_1"],
            vectors=[[0.3] * 768],
            metadatas=[{"text": "original", "score": 0.5}]
        )

        adapter.update_fragment_metadata("update_1", {"score": 0.9, "updated": True})

        frag = adapter.get_fragment("update_1")
        assert frag["metadata"]["score"] == 0.9
        assert frag["metadata"]["updated"] is True

    @pytest.mark.asyncio
    async def test_update_fragment_score(self, adapter):
        """Should update composite score."""
        await adapter.upsert_vectors(
            ids=["score_1"],
            vectors=[[0.4] * 768],
            metadatas=[{"text": "test", "composite_score": 0.5}]
        )

        adapter.update_fragment_score("score_1", 0.95)

        frag = adapter.get_fragment("score_1")
        assert frag["metadata"]["composite_score"] == 0.95

    @pytest.mark.asyncio
    async def test_get_all_vectors(self, adapter):
        """Should return all vectors in collection."""
        await adapter.upsert_vectors(
            ids=["all_1", "all_2"],
            vectors=[[0.1] * 768, [0.2] * 768],
            metadatas=[{"text": "first"}, {"text": "second"}]
        )

        all_vecs = adapter.get_all_vectors()

        assert len(all_vecs) == 2
        ids = [v["id"] for v in all_vecs]
        assert "all_1" in ids
        assert "all_2" in ids

    @pytest.mark.asyncio
    async def test_empty_collection_query(self, adapter):
        """Empty collection should return empty results."""
        results = await adapter.query_vectors([0.1] * 768, top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_list_all_ids(self, adapter):
        """Should list all document IDs."""
        await adapter.upsert_vectors(
            ids=["id_a", "id_b", "id_c"],
            vectors=[[0.1] * 768, [0.2] * 768, [0.3] * 768],
            metadatas=[{"text": "a"}, {"text": "b"}, {"text": "c"}]
        )

        ids = adapter.list_all_ids()

        assert len(ids) == 3
        assert set(ids) == {"id_a", "id_b", "id_c"}

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, adapter):
        """Upsert should update existing document."""
        # Initial insert
        await adapter.upsert_vectors(
            ids=["upsert_1"],
            vectors=[[0.1] * 768],
            metadatas=[{"text": "original", "version": 1}]
        )

        # Overwrite
        await adapter.upsert_vectors(
            ids=["upsert_1"],
            vectors=[[0.2] * 768],
            metadatas=[{"text": "updated", "version": 2}]
        )

        # Should still be 1 document
        count = await adapter.get_collection_count()
        assert count == 1

        # Should have new content
        frag = adapter.get_fragment("upsert_1")
        assert frag["metadata"]["version"] == 2
        assert frag["metadata"]["text"] == "updated"


class TestChromaDBEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp(prefix="roampal_edge_")
        yield temp_dir
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    @pytest.fixture
    async def adapter(self, temp_db_path):
        """Create initialized adapter."""
        from roampal.backend.modules.memory.chromadb_adapter import ChromaDBAdapter

        adapter = ChromaDBAdapter(
            persistence_directory=temp_db_path,
            use_server=False
        )
        await adapter.initialize(collection_name="edge_test")
        yield adapter
        await adapter.cleanup()

    @pytest.mark.asyncio
    async def test_upsert_mismatched_lengths_raises(self, adapter):
        """Should raise on mismatched input lengths."""
        with pytest.raises(ValueError):
            await adapter.upsert_vectors(
                ids=["id_1", "id_2"],
                vectors=[[0.1] * 768],  # Only 1 vector
                metadatas=[{"text": "a"}, {"text": "b"}]
            )

    @pytest.mark.asyncio
    async def test_invalid_query_vector(self, adapter):
        """Invalid query vector should return empty."""
        # None values
        results = await adapter.query_vectors([None] * 768, top_k=5)
        assert results == []

        # Non-numeric
        results = await adapter.query_vectors(["not", "a", "number"], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_with_empty_vector(self, adapter):
        """Empty query vector should return empty."""
        results = await adapter.query_vectors([], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, adapter):
        """Deleting nonexistent ID should not raise."""
        # Should not raise
        adapter.delete_vectors(["nonexistent_id"])

    @pytest.mark.asyncio
    async def test_update_nonexistent_metadata(self, adapter):
        """Updating nonexistent fragment should warn but not raise."""
        # Should not raise
        adapter.update_fragment_metadata("nonexistent", {"key": "value"})


class TestChromaDBPersistence:
    """Test data persistence across adapter restarts."""

    @pytest.fixture
    def persistent_path(self):
        """Create persistent temp directory."""
        temp_dir = tempfile.mkdtemp(prefix="roampal_persist_")
        yield temp_dir
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_data_persists_across_sessions(self, persistent_path):
        """Data should persist after adapter cleanup and reinit."""
        from roampal.backend.modules.memory.chromadb_adapter import ChromaDBAdapter

        # Session 1: Write data
        adapter1 = ChromaDBAdapter(
            persistence_directory=persistent_path,
            use_server=False
        )
        await adapter1.initialize(collection_name="persist_test")

        await adapter1.upsert_vectors(
            ids=["persist_1"],
            vectors=[[0.5] * 768],
            metadatas=[{"text": "persistent data", "score": 0.9}]
        )

        await adapter1.cleanup()

        # Session 2: Read data
        adapter2 = ChromaDBAdapter(
            persistence_directory=persistent_path,
            use_server=False
        )
        await adapter2.initialize(collection_name="persist_test")

        count = await adapter2.get_collection_count()
        assert count == 1

        frag = adapter2.get_fragment("persist_1")
        assert frag is not None
        assert frag["metadata"]["text"] == "persistent data"

        await adapter2.cleanup()


class TestChromaDBMultipleCollections:
    """Test multiple collection support."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp(prefix="roampal_multi_")
        yield temp_dir
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_separate_collections(self, temp_db_path):
        """Different collections should be isolated."""
        from roampal.backend.modules.memory.chromadb_adapter import ChromaDBAdapter

        # Create two adapters with different collections
        adapter_a = ChromaDBAdapter(
            persistence_directory=temp_db_path,
            use_server=False
        )
        await adapter_a.initialize(collection_name="collection_a")

        adapter_b = ChromaDBAdapter(
            persistence_directory=temp_db_path,
            use_server=False
        )
        await adapter_b.initialize(collection_name="collection_b")

        # Add data to each
        await adapter_a.upsert_vectors(
            ids=["a_1"],
            vectors=[[0.1] * 768],
            metadatas=[{"text": "in collection a"}]
        )

        await adapter_b.upsert_vectors(
            ids=["b_1", "b_2"],
            vectors=[[0.2] * 768, [0.3] * 768],
            metadatas=[{"text": "in collection b"}, {"text": "also b"}]
        )

        # Verify isolation
        count_a = await adapter_a.get_collection_count()
        count_b = await adapter_b.get_collection_count()

        assert count_a == 1
        assert count_b == 2

        # Check IDs
        ids_a = adapter_a.list_all_ids()
        ids_b = adapter_b.list_all_ids()

        assert "a_1" in ids_a
        assert "a_1" not in ids_b
        assert "b_1" in ids_b
        assert "b_1" not in ids_a

        await adapter_a.cleanup()
        await adapter_b.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Integration Tests for Archive-Then-Add Cycle (Issue #8 repro).

End-to-end tests using real ChromaDB and UnifiedMemorySystem to verify
that archived entries do not block dedup of new facts. This reproduces
the exact bug Marcus reported: add facts → archive some → add similar → 
all new facts stored successfully.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

import pytest
import tempfile
import shutil


@pytest.fixture(autouse=True)
async def _cancel_warmup_tasks():
    """Cancel UnifiedMemorySystem warmup_ce / warmup_embedding tasks after each
    test. See test_phantom_cleanup_safety.py for the full backstory; same fix.
    """
    yield
    try:
        for task in asyncio.all_tasks():
            if task.get_name() in ("warmup_ce", "warmup_embedding") and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
    except RuntimeError:
        pass


class TestArchiveDedupCycle:
    """End-to-end archive-then-add cycle with real ChromaDB."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary directory for UnifiedMemorySystem storage."""
        temp_dir = tempfile.mkdtemp(prefix="roampal_cycle_")
        yield temp_dir
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    @pytest.fixture
    async def mem_system(self, temp_data_path):
        """Create initialized UnifiedMemorySystem with real ChromaDB."""
        from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem

        mem = UnifiedMemorySystem(data_path=temp_data_path)
        await mem.initialize()
        yield mem

    @pytest.mark.asyncio
    async def test_archive_then_add_similar_fact_succeeds(self, mem_system):
        """v0.5.5 issue #8: archived entries must not block dedup of new facts."""
        # Add 3 distinct facts (semantically different enough to not dedup)
        fact_a = await mem_system.store_memory_bank("User is a senior software engineer at Acme Corp")
        fact_b = await mem_system.store_memory_bank("User prefers Python over JavaScript for backend work")
        fact_c = await mem_system.store_memory_bank("User has two cats named Luna and Milo")

        count = mem_system._memory_bank_service._get_count()
        assert count == 3, f"Expected 3 facts after initial store, got {count}"

        # Archive the first one (soft-delete via delete_memory_bank)
        ok = await mem_system.delete_memory_bank("User is a senior software engineer at Acme Corp")
        assert ok, "Should be able to archive fact A"

        count = mem_system._memory_bank_service._get_count()
        assert count == 2, f"Expected 2 active facts after archive, got {count}"

        # Re-add same content — should succeed because archived entry is filtered from dedup
        new_id_a = await mem_system.store_memory_bank("User is a senior software engineer at Acme Corp")
        assert new_id_a is not None, "Should store fact that matches an archived entry"
        assert new_id_a != fact_a, "Re-added fact should get a new ID"

        count = mem_system._memory_bank_service._get_count()
        assert count == 3, f"Expected 3 active facts after re-add, got {count}"

    @pytest.mark.asyncio
    async def test_active_duplicates_still_dedup(self, mem_system):
        """Active entries should still be deduped normally."""
        id1 = await mem_system.store_memory_bank("User prefers dark mode")
        
        # Attempt to add the exact same fact — should return existing ID
        id2 = await mem_system.store_memory_bank("User prefers dark mode")

        assert id1 == id2, "Identical active facts should be deduped"

        count = mem_system._memory_bank_service._get_count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_archive_all_then_readd(self, mem_system):
        """Archive all entries, then re-add same ones — all should succeed."""
        # Add facts
        id_a = await mem_system.store_memory_bank("User works on distributed systems")
        id_b = await mem_system.store_memory_bank("User enjoys hiking on weekends")
        id_c = await mem_system.store_memory_bank("User is learning Rust programming language")

        assert mem_system._memory_bank_service._get_count() == 3

        # Archive all of them
        ok1 = await mem_system.delete_memory_bank("User works on distributed systems")
        ok2 = await mem_system.delete_memory_bank("User enjoys hiking on weekends")
        ok3 = await mem_system.delete_memory_bank("User is learning Rust programming language")

        assert ok1, "Should archive fact A"
        assert ok2, "Should archive fact B"
        assert ok3, "Should archive fact C"

        count = mem_system._memory_bank_service._get_count()
        assert count == 0, f"Expected 0 active facts after archiving all, got {count}"

        # Re-add same facts — should all succeed
        new_id_a = await mem_system.store_memory_bank("User works on distributed systems")
        new_id_b = await mem_system.store_memory_bank("User enjoys hiking on weekends")
        new_id_c = await mem_system.store_memory_bank("User is learning Rust programming language")

        assert new_id_a and new_id_b and new_id_c, "All re-added facts should succeed"
        count = mem_system._memory_bank_service._get_count()
        assert count == 3, f"Expected 3 active facts after re-add, got {count}"


class TestArchiveDedupCycleSearch:
    """Verify archived entries don't pollute search results after cycle."""

    @pytest.fixture
    def temp_data_path(self):
        temp_dir = tempfile.mkdtemp(prefix="roampal_cycle_search_")
        yield temp_dir
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    @pytest.fixture
    async def mem_system(self, temp_data_path):
        from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem

        mem = UnifiedMemorySystem(data_path=temp_data_path)
        await mem.initialize()
        yield mem

    @pytest.mark.asyncio
    async def test_search_excludes_archived_after_cycle(self, mem_system):
        """Search must exclude archived entries even if they share embedding space."""
        # Add two distinct facts
        id1 = await mem_system.store_memory_bank("User is a data scientist at Google")
        id2 = await mem_system.store_memory_bank("User is a product manager at Facebook")

        assert mem_system._memory_bank_service._get_count() == 2

        # Archive one
        ok = await mem_system.delete_memory_bank("User is a data scientist at Google")
        assert ok
        assert mem_system._memory_bank_service._get_count() == 1

        # Search for a term that would match both via embedding similarity
        results = await mem_system.search(
            query="tech company",
            collections=["memory_bank"],
            limit=10,
        )

        ids_in_results = [r.get("id") for r in results]
        assert id1 not in ids_in_results, "Archived fact should not appear in search"
        # id2 may or may not match depending on embedding quality; just verify archived is excluded


    @pytest.mark.asyncio
    async def test_search_after_archive_and_readd(self, mem_system):
        """After archive→re-add cycle, only the active entry is searchable."""
        # Add a fact
        id1 = await mem_system.store_memory_bank("User is a Python developer")

        # Archive it
        ok = await mem_system.delete_memory_bank("User is a Python developer")
        assert ok

        # Re-add same content
        id2 = await mem_system.store_memory_bank("User is a Python developer")

        assert id1 != id2, "Re-added fact should get new ID"

        # Search — only the re-added entry should appear
        results = await mem_system.search(
            query="Python developer",
            collections=["memory_bank"],
            limit=10,
        )

        ids_in_results = [r.get("id") for r in results]
        assert id1 not in ids_in_results, "Archived original should not appear"
        assert id2 in ids_in_results, "Re-added fact should appear"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

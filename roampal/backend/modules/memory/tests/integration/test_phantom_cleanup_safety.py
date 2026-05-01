"""
Integration tests for v0.5.6 phantom cleanup safety.

Validates that the phantom-sweep + cleanup_archived chain only ever removes
entries it should — never a real, non-archived entry — using real ChromaDB
on an isolated temp data path.

Mirrors the manual live-validation run on 2026-05-01:
  Scenario A: force-delete entries via raw `adapter.collection.delete(ids=...)`,
              run `_sweep_phantoms()`, verify only the deleted IDs are gone
              and all survivors remain intact.
  Scenario B: archive a single entry, run `cleanup_archived()`, verify the
              archived entry is removed and all other entries remain intact.
"""
import asyncio
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))


@pytest.fixture(autouse=True)
async def _cancel_warmup_tasks():
    """Cancel UnifiedMemorySystem warmup_ce / warmup_embedding tasks after each
    test so they don't leak into the next test's event loop or RAM.

    UnifiedMemorySystem.initialize() schedules two background warmup tasks via
    asyncio.create_task that wrap asyncio.to_thread(...) for ONNX model loads.
    Without explicit cancellation these orphan worker threads accumulate across
    tests on Linux CI runners and either deadlock the next initialize() (3.10)
    or push the worker pool past the runner's resource limit so the runner
    sends SIGTERM (3.11/3.12 — what tanked the v0.5.6 release CI). 3.11+
    asyncio.Runner cleans these up correctly when not under load; production
    code only initialize()s once per process so the leak never materializes
    outside the test harness.
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


class TestPhantomCleanupSafety:
    """Sweep + cleanup must never touch a real, non-archived entry."""

    @pytest.fixture
    def temp_data_path(self):
        temp_dir = tempfile.mkdtemp(prefix="roampal_phantom_safety_")
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

    async def _seed(self, mem_system, count: int = 5):
        """Add `count` distinctive marker entries; return list of doc_ids."""
        svc = mem_system._memory_bank_service
        ids = []
        for i in range(count):
            doc_id = await svc.store(
                text=f"PHANTOM_SAFETY_MARKER_{i:02d} - distinctive content",
                tags=["preference"],
                importance=0.5,
                confidence=0.5,
            )
            ids.append(doc_id)
        return ids

    async def test_sweep_after_raw_delete_leaves_survivors_intact(self, mem_system):
        """
        Force-delete 2 of 5 entries via the raw ChromaDB delete path, then run
        _sweep_phantoms(). Survivors must remain present and intact.

        This tests the worst case: a delete path that historically left HNSW
        phantoms (per v0.5.5 release notes). On modern ChromaDB the raw delete
        cleans the index immediately, so the sweep finds 0 phantoms — but the
        critical safety property is that it never removes a real entry.
        """
        svc = mem_system._memory_bank_service
        adapter = svc.collection
        ids = await self._seed(mem_system, count=5)

        assert len(adapter.list_all_ids()) == 5

        victims = ids[:2]
        survivors = ids[2:]

        adapter.collection.delete(ids=victims)

        swept = svc._sweep_phantoms()
        assert swept >= 0  # never negative

        remaining = set(adapter.list_all_ids())
        assert remaining == set(survivors), (
            f"sweep removed unexpected entries. expected={set(survivors)} got={remaining}"
        )

        # Every survivor must still resolve to a non-None fragment with content
        for doc_id in survivors:
            frag = adapter.get_fragment(doc_id)
            assert frag is not None, f"survivor {doc_id} missing after sweep"
            content = frag.get("metadata", {}).get("text") or frag.get("content", "")
            assert "PHANTOM_SAFETY_MARKER" in content, (
                f"survivor {doc_id} content corrupted: {content!r}"
            )

    async def test_cleanup_archived_only_removes_archived(self, mem_system):
        """
        Archive 1 of 5 entries via the official archive() API, then run
        cleanup_archived(). The archived entry must be the only one removed.

        Closes the safety contract: archived entries get permanently removed
        (with a phantom sweep right after, item 5), and active entries are
        never touched.
        """
        svc = mem_system._memory_bank_service
        adapter = svc.collection
        ids = await self._seed(mem_system, count=5)

        archive_target = ids[2]  # somewhere in the middle
        target_frag = adapter.get_fragment(archive_target)
        target_content = (
            target_frag.get("metadata", {}).get("text")
            or target_frag.get("content", "")
        )
        await svc.archive(target_content)

        # Archive is soft — list_all_ids unchanged, but status flipped
        assert len(adapter.list_all_ids()) == 5
        post = adapter.get_fragment(archive_target)
        assert post is not None
        assert post.get("metadata", {}).get("status") == "archived"

        deleted = svc.cleanup_archived()
        assert deleted == 1, f"cleanup_archived removed {deleted}, expected 1"

        remaining = set(adapter.list_all_ids())
        expected = set(ids) - {archive_target}
        assert remaining == expected, (
            f"cleanup_archived touched unexpected entries. "
            f"expected={expected} got={remaining}"
        )

        # Every expected-remaining entry must still be intact
        for doc_id in expected:
            frag = adapter.get_fragment(doc_id)
            assert frag is not None, f"non-archived {doc_id} missing after cleanup"
            content = frag.get("metadata", {}).get("text") or frag.get("content", "")
            assert "PHANTOM_SAFETY_MARKER" in content, (
                f"non-archived {doc_id} content corrupted: {content!r}"
            )

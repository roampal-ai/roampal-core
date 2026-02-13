"""
Integration Test: Action KG Sync (v0.2.0 regression test).

Tests the fix for the "Reading Your Own Writes" bug where:
- UMS.record_action_outcome() writes to self.knowledge_graph
- SearchService reads from _kg_service.knowledge_graph
- Before fix: _kg_service never saw the updates (stale data)
- After fix: _save_kg() syncs to _kg_service via reload_kg()

This test ensures the full write→read flow works end-to-end.
"""

import sys
import os
import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

# Ensure embedded mode for tests
os.environ["ROAMPAL_USE_SERVER"] = "false"


class TestActionKGSync:
    """Integration tests for Action KG write→read sync."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp(prefix="roampal_action_kg_test_")
        yield Path(temp_dir)
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    @pytest.fixture
    async def ums(self, temp_data_path):
        """Create initialized UMS with temp data path."""
        from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem

        ums = UnifiedMemorySystem(data_path=temp_data_path)
        await ums.initialize()
        yield ums
        # Cleanup handled by temp_data_path fixture

    @pytest.mark.asyncio
    async def test_action_outcome_visible_to_kg_service(self, ums):
        """
        Record action outcome via UMS → _kg_service must see it immediately.

        This is the core regression test for the v0.2.0 "Reading Your Own Writes" fix.
        """
        from roampal.backend.modules.memory.unified_memory_system import ActionOutcome

        # 1. Create an action outcome
        action = ActionOutcome(
            action_type="search_memory",
            context_type="coding",
            outcome="worked",
            doc_id="test_doc_123",
            collection="patterns"
        )

        # 2. Record via UMS (writes to self.knowledge_graph, saves, syncs)
        await ums.record_action_outcome(action)

        # 3. Read via _kg_service (what SearchService uses)
        key = "coding|search_memory|patterns"
        kg_service_data = ums._kg_service.knowledge_graph.get("context_action_effectiveness", {})

        # 4. Assert _kg_service sees the update
        assert key in kg_service_data, \
            f"_kg_service didn't see the update! Keys: {list(kg_service_data.keys())}"

        stats = kg_service_data[key]
        assert stats["successes"] == 1
        assert stats["total_uses"] == 1
        assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_multiple_outcomes_accumulate(self, ums):
        """Multiple action outcomes should accumulate correctly."""
        from roampal.backend.modules.memory.unified_memory_system import ActionOutcome

        # Record multiple outcomes
        for i in range(3):
            action = ActionOutcome(
                action_type="score_memories",
                context_type="general",
                outcome="worked" if i < 2 else "failed",
                doc_id=f"doc_{i}",
                collection="history"
            )
            await ums.record_action_outcome(action)

        # Check _kg_service sees all outcomes
        key = "general|score_memories|history"
        stats = ums._kg_service.knowledge_graph["context_action_effectiveness"][key]

        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert stats["total_uses"] == 3
        # success_rate = (2 + 0) / 3 = 0.666...
        assert 0.66 < stats["success_rate"] < 0.67

    @pytest.mark.asyncio
    async def test_routing_patterns_sync(self, ums):
        """_update_kg_routing should also sync to _kg_service."""
        # Record a routing pattern update
        await ums._update_kg_routing("python tutorial", "books", "worked")

        # Check _kg_service sees the routing pattern
        routing_patterns = ums._kg_service.knowledge_graph.get("routing_patterns", {})

        # Should have patterns for extracted concepts
        assert len(routing_patterns) > 0, "No routing patterns synced to _kg_service"

    @pytest.mark.asyncio
    async def test_persistence_across_reload(self, temp_data_path):
        """Action outcomes should persist and be visible after UMS restart."""
        from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem, ActionOutcome

        # Session 1: Record action
        ums1 = UnifiedMemorySystem(data_path=temp_data_path)
        await ums1.initialize()

        action = ActionOutcome(
            action_type="add_to_memory_bank",
            context_type="research",
            outcome="worked",
            doc_id="persist_test",
            collection="memory_bank"
        )
        await ums1.record_action_outcome(action)

        # Session 2: New UMS instance should see the data
        ums2 = UnifiedMemorySystem(data_path=temp_data_path)
        await ums2.initialize()

        key = "research|add_to_memory_bank|memory_bank"
        stats = ums2._kg_service.knowledge_graph["context_action_effectiveness"].get(key)

        assert stats is not None, "Persisted action outcome not visible after restart"
        assert stats["successes"] == 1


class TestSearchServiceReadsFromKGService:
    """Verify SearchService reads from _kg_service (not UMS.knowledge_graph)."""

    @pytest.fixture
    def temp_data_path(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp(prefix="roampal_search_kg_test_")
        yield Path(temp_dir)
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    @pytest.fixture
    async def ums(self, temp_data_path):
        """Create initialized UMS."""
        from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem

        ums = UnifiedMemorySystem(data_path=temp_data_path)
        await ums.initialize()
        yield ums

    @pytest.mark.asyncio
    async def test_kg_service_sees_action_via_direct_access(self, ums):
        """_kg_service.knowledge_graph should see action outcomes after UMS records them."""
        from roampal.backend.modules.memory.unified_memory_system import ActionOutcome

        # Record an action with a specific doc_id
        action = ActionOutcome(
            action_type="search_memory",
            context_type="testing",
            outcome="worked",
            doc_id="search_test_doc",
            collection="patterns"
        )
        await ums.record_action_outcome(action)

        # _kg_service.knowledge_graph should have the action data
        # This is what SearchService reads from
        key = "testing|search_memory|patterns"
        action_data = ums._kg_service.knowledge_graph.get("context_action_effectiveness", {})

        assert key in action_data, f"_kg_service didn't see action! Keys: {list(action_data.keys())}"

        # Check the example has the doc_id
        stats = action_data[key]
        examples = stats.get("examples", [])
        doc_ids = [ex.get("doc_id") for ex in examples]

        assert "search_test_doc" in doc_ids, f"Doc ID not in examples: {doc_ids}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

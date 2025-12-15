"""
Unit Tests for KnowledgeGraphService

Tests the extracted KG logic including the race condition fix.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from roampal.backend.modules.memory.knowledge_graph_service import KnowledgeGraphService
from roampal.backend.modules.memory.config import MemoryConfig


class TestConceptExtraction:
    """Test concept extraction from text."""

    @pytest.fixture
    def temp_paths(self, tmp_path):
        """Create temporary paths for KG files."""
        return {
            "kg_path": tmp_path / "knowledge_graph.json",
            "content_graph_path": tmp_path / "content_graph.json",
            "relationships_path": tmp_path / "relationships.json",
        }

    @pytest.fixture
    def service(self, temp_paths):
        """Create KnowledgeGraphService instance."""
        return KnowledgeGraphService(
            kg_path=temp_paths["kg_path"],
            content_graph_path=temp_paths["content_graph_path"],
            relationships_path=temp_paths["relationships_path"],
        )

    def test_extract_unigrams(self, service):
        """Should extract single words longer than 3 chars."""
        concepts = service.extract_concepts("Python programming language")
        assert "python" in concepts
        assert "programming" in concepts
        assert "language" in concepts

    def test_extract_bigrams(self, service):
        """Should extract two-word phrases."""
        concepts = service.extract_concepts("Python programming")
        bigrams = [c for c in concepts if "_" in c and c.count("_") == 1]
        assert len(bigrams) > 0
        assert "python_programming" in bigrams

    def test_extract_trigrams(self, service):
        """Should extract three-word phrases."""
        concepts = service.extract_concepts("Python programming language rocks")
        trigrams = [c for c in concepts if c.count("_") == 2]
        assert len(trigrams) > 0

    def test_filter_stop_words(self, service):
        """Should filter out stop words."""
        concepts = service.extract_concepts("The quick brown fox is a test")
        assert "the" not in concepts
        assert "is" not in concepts
        assert "quick" in concepts
        assert "brown" in concepts

    def test_filter_short_words(self, service):
        """Should filter out words with 3 or fewer chars."""
        concepts = service.extract_concepts("Go is a fun language")
        unigrams = [c for c in concepts if "_" not in c]
        # "fun" is exactly 3 chars, should be filtered for unigrams (>3 required)
        assert "fun" not in unigrams
        assert "language" in unigrams

    def test_filter_tool_blocklist(self, service):
        """Should filter out MCP tool names and internal terms."""
        concepts = service.extract_concepts("Use search_memory to find memory_bank items")
        assert "search_memory" not in concepts
        assert "memory_bank" not in concepts

    def test_empty_text(self, service):
        """Empty text should return empty list."""
        concepts = service.extract_concepts("")
        assert concepts == []

    def test_case_insensitive(self, service):
        """Extraction should be case-insensitive."""
        concepts = service.extract_concepts("Python PYTHON python")
        python_concepts = [c for c in concepts if "python" in c.lower()]
        # Should have python unigrams, all normalized to lowercase
        assert "python" in concepts


class TestKGLoading:
    """Test KG loading from disk."""

    @pytest.fixture
    def temp_paths(self, tmp_path):
        return {
            "kg_path": tmp_path / "knowledge_graph.json",
            "content_graph_path": tmp_path / "content_graph.json",
            "relationships_path": tmp_path / "relationships.json",
        }

    def test_load_empty_kg(self, temp_paths):
        """Should create default KG when file doesn't exist."""
        service = KnowledgeGraphService(**temp_paths)
        assert "routing_patterns" in service.knowledge_graph
        assert "problem_solutions" in service.knowledge_graph
        assert "context_action_effectiveness" in service.knowledge_graph

    def test_load_existing_kg(self, temp_paths):
        """Should load existing KG from file."""
        # Create a KG file first
        kg_data = {
            "routing_patterns": {"test_concept": {"best_collection": "history"}},
            "success_rates": {},
            "failure_patterns": {},
            "problem_categories": {},
            "problem_solutions": {},
            "solution_patterns": {},
            "context_action_effectiveness": {},
        }
        temp_paths["kg_path"].parent.mkdir(exist_ok=True, parents=True)
        with open(temp_paths["kg_path"], "w") as f:
            json.dump(kg_data, f)

        service = KnowledgeGraphService(**temp_paths)
        assert "test_concept" in service.knowledge_graph["routing_patterns"]

    def test_load_partial_kg_fills_missing_keys(self, temp_paths):
        """Should fill missing keys when loading partial KG."""
        # Create a partial KG file
        kg_data = {"routing_patterns": {}}
        temp_paths["kg_path"].parent.mkdir(exist_ok=True, parents=True)
        with open(temp_paths["kg_path"], "w") as f:
            json.dump(kg_data, f)

        service = KnowledgeGraphService(**temp_paths)
        # Should have all required keys
        assert "problem_solutions" in service.knowledge_graph
        assert "context_action_effectiveness" in service.knowledge_graph


class TestConceptRelationships:
    """Test building concept relationships."""

    @pytest.fixture
    def temp_paths(self, tmp_path):
        return {
            "kg_path": tmp_path / "knowledge_graph.json",
            "content_graph_path": tmp_path / "content_graph.json",
            "relationships_path": tmp_path / "relationships.json",
        }

    @pytest.fixture
    def service(self, temp_paths):
        return KnowledgeGraphService(**temp_paths)

    def test_build_concept_relationships(self, service):
        """Should create relationships between concept pairs."""
        concepts = ["python", "django", "web"]
        service.build_concept_relationships(concepts)

        relationships = service.knowledge_graph.get("relationships", {})
        # Should have 3 relationships: python-django, python-web, django-web
        assert len(relationships) == 3

        # Check relationship keys are sorted
        assert "django|python" in relationships  # Sorted alphabetically
        assert "python|web" in relationships
        assert "django|web" in relationships

    def test_relationship_co_occurrence_increments(self, service):
        """Co-occurrence should increment on repeated builds."""
        concepts = ["python", "django"]
        service.build_concept_relationships(concepts)
        service.build_concept_relationships(concepts)

        rel_key = "django|python"
        assert service.knowledge_graph["relationships"][rel_key]["co_occurrence"] == 2


class TestKGRouting:
    """Test KG routing pattern updates."""

    @pytest.fixture
    def temp_paths(self, tmp_path):
        return {
            "kg_path": tmp_path / "knowledge_graph.json",
            "content_graph_path": tmp_path / "content_graph.json",
            "relationships_path": tmp_path / "relationships.json",
        }

    @pytest.fixture
    def service(self, temp_paths):
        return KnowledgeGraphService(**temp_paths)

    @pytest.mark.asyncio
    async def test_update_kg_routing_creates_pattern(self, service):
        """Should create routing pattern for new concept."""
        await service.update_kg_routing("Python tutorial", "books", "worked")

        patterns = service.knowledge_graph["routing_patterns"]
        assert "python" in patterns
        assert patterns["python"]["best_collection"] == "books"

    @pytest.mark.asyncio
    async def test_update_kg_routing_tracks_outcomes(self, service):
        """Should track success/failure outcomes."""
        await service.update_kg_routing("Python help", "history", "worked")
        await service.update_kg_routing("Python help", "history", "failed")

        stats = service.knowledge_graph["routing_patterns"]["python"]["collections_used"]["history"]
        assert stats["successes"] == 1
        assert stats["failures"] == 1
        assert stats["total"] == 2

    @pytest.mark.asyncio
    async def test_update_kg_routing_updates_best_collection(self, service):
        """Should update best collection based on success rate."""
        # Books has 2 successes
        await service.update_kg_routing("Python docs", "books", "worked")
        await service.update_kg_routing("Python docs", "books", "worked")

        # History has 1 success, 1 failure
        await service.update_kg_routing("Python help", "history", "worked")
        await service.update_kg_routing("Python help", "history", "failed")

        pattern = service.knowledge_graph["routing_patterns"]["python"]
        assert pattern["best_collection"] == "books"  # 100% vs 50%


class TestRaceConditionFix:
    """Test the race condition fix in debounced saves."""

    @pytest.fixture
    def temp_paths(self, tmp_path):
        return {
            "kg_path": tmp_path / "knowledge_graph.json",
            "content_graph_path": tmp_path / "content_graph.json",
            "relationships_path": tmp_path / "relationships.json",
        }

    @pytest.fixture
    def service(self, temp_paths):
        # Use short debounce for testing
        config = MemoryConfig(kg_debounce_seconds=0.1)
        return KnowledgeGraphService(**temp_paths, config=config)

    @pytest.mark.asyncio
    async def test_debounced_save_serializes_access(self, service):
        """Multiple concurrent debounced saves should not race."""
        # Launch multiple concurrent saves
        tasks = [
            asyncio.create_task(service._debounced_save_kg())
            for _ in range(10)
        ]
        await asyncio.gather(*tasks)

        # Wait for debounce to complete
        await asyncio.sleep(0.2)

        # Should have completed without errors
        assert not service._kg_save_pending or service._kg_save_task.done()

    @pytest.mark.asyncio
    async def test_debounced_save_batches_updates(self, service):
        """Multiple updates within debounce window should batch."""
        save_count = 0
        original_save = service._save_kg

        async def counting_save():
            nonlocal save_count
            save_count += 1
            await original_save()

        service._save_kg = counting_save

        # Rapid updates
        for i in range(5):
            await service._debounced_save_kg()

        # Wait for debounce
        await asyncio.sleep(0.2)

        # Should only save once due to batching
        assert save_count == 1

    @pytest.mark.asyncio
    async def test_cleanup_cancels_pending_save(self, service):
        """Cleanup should cancel pending save task."""
        await service._debounced_save_kg()
        assert service._kg_save_task is not None

        await service.cleanup()

        # Task should be cancelled or done
        assert service._kg_save_task.done() or service._kg_save_task.cancelled()


class TestProblemSolutionTracking:
    """Test problem-solution pattern tracking."""

    @pytest.fixture
    def temp_paths(self, tmp_path):
        return {
            "kg_path": tmp_path / "knowledge_graph.json",
            "content_graph_path": tmp_path / "content_graph.json",
            "relationships_path": tmp_path / "relationships.json",
        }

    @pytest.fixture
    def service(self, temp_paths):
        config = MemoryConfig(kg_debounce_seconds=0)  # No debounce for tests
        return KnowledgeGraphService(**temp_paths, config=config)

    @pytest.mark.asyncio
    async def test_track_problem_solution_creates_entry(self, service):
        """Should create problem-solution mapping."""
        metadata = {
            "original_context": "How do I fix Python import errors?",
            "text": "Use sys.path.insert to add the module path",
        }
        await service.track_problem_solution("doc_123", metadata, None)

        assert len(service.knowledge_graph["problem_solutions"]) > 0

    @pytest.mark.asyncio
    async def test_track_problem_solution_increments_on_repeat(self, service):
        """Should increment success_count for existing solutions."""
        metadata = {
            "original_context": "Python import errors fix",
            "text": "Use sys.path.insert",
        }

        await service.track_problem_solution("doc_123", metadata, None)
        await service.track_problem_solution("doc_123", metadata, None)

        # Find the entry
        for sig, solutions in service.knowledge_graph["problem_solutions"].items():
            for sol in solutions:
                if sol["doc_id"] == "doc_123":
                    assert sol["success_count"] == 2
                    return

        pytest.fail("Solution not found")

    @pytest.mark.asyncio
    async def test_find_known_solutions_exact_match(self, service):
        """Should find exact problem-solution matches."""
        # First track a solution
        metadata = {
            "original_context": "Python import errors debugging",
            "text": "Use sys.path.insert to fix",
        }
        await service.track_problem_solution("history_doc_1", metadata, None)

        # Mock get_fragment_fn
        def get_fragment(coll_name, doc_id):
            if doc_id == "history_doc_1":
                return {
                    "id": doc_id,
                    "content": "Use sys.path.insert to fix",
                    "distance": 1.0,
                }
            return None

        # Search for similar problem
        solutions = await service.find_known_solutions(
            "Python import errors debugging",
            get_fragment
        )

        # Should find the tracked solution
        assert len(solutions) > 0
        assert solutions[0]["id"] == "history_doc_1"
        assert solutions[0]["is_known_solution"] is True


class TestKGCleanup:
    """Test KG cleanup operations."""

    @pytest.fixture
    def temp_paths(self, tmp_path):
        return {
            "kg_path": tmp_path / "knowledge_graph.json",
            "content_graph_path": tmp_path / "content_graph.json",
            "relationships_path": tmp_path / "relationships.json",
        }

    @pytest.fixture
    def service(self, temp_paths):
        return KnowledgeGraphService(**temp_paths)

    @pytest.mark.asyncio
    async def test_cleanup_dead_references(self, service):
        """Should remove references to non-existent documents."""
        # Add some problem_solutions with doc_ids
        service.knowledge_graph["problem_solutions"] = {
            "test_problem": [
                {"doc_id": "history_valid", "success_count": 1},
                {"doc_id": "history_invalid", "success_count": 1},
            ]
        }

        # Mock doc_exists - only valid doc exists
        def doc_exists(doc_id):
            return doc_id == "history_valid"

        cleaned = await service.cleanup_dead_references(doc_exists)

        # Should have cleaned 1 reference
        assert cleaned == 1
        solutions = service.knowledge_graph["problem_solutions"]["test_problem"]
        assert len(solutions) == 1
        assert solutions[0]["doc_id"] == "history_valid"

    @pytest.mark.asyncio
    async def test_cleanup_action_kg_for_doc_ids(self, service):
        """Should remove Action KG examples for deleted doc_ids."""
        service.knowledge_graph["context_action_effectiveness"] = {
            "context|action|coll": {
                "successes": 2,
                "failures": 0,
                "examples": [
                    {"doc_id": "doc_1", "text": "example 1"},
                    {"doc_id": "doc_2", "text": "example 2"},
                    {"doc_id": "doc_3", "text": "example 3"},
                ]
            }
        }

        cleaned = await service.cleanup_action_kg_for_doc_ids(["doc_1", "doc_3"])

        assert cleaned == 2
        examples = service.knowledge_graph["context_action_effectiveness"]["context|action|coll"]["examples"]
        assert len(examples) == 1
        assert examples[0]["doc_id"] == "doc_2"


class TestKGEntitiesVisualization:
    """Test KG entity retrieval for visualization."""

    @pytest.fixture
    def temp_paths(self, tmp_path):
        return {
            "kg_path": tmp_path / "knowledge_graph.json",
            "content_graph_path": tmp_path / "content_graph.json",
            "relationships_path": tmp_path / "relationships.json",
        }

    @pytest.fixture
    def service(self, temp_paths):
        return KnowledgeGraphService(**temp_paths)

    @pytest.mark.asyncio
    async def test_get_kg_entities_routing(self, service):
        """Should return routing KG entities."""
        service.knowledge_graph["routing_patterns"] = {
            "python": {
                "collections_used": {"history": {"total": 5}},
                "best_collection": "history",
                "success_rate": 0.8,
            }
        }
        # Save to disk so reload_kg finds it (get_kg_entities calls reload_kg)
        service._save_kg_sync()

        entities = await service.get_kg_entities()

        assert len(entities) > 0
        python_entity = next((e for e in entities if e["entity"] == "python"), None)
        assert python_entity is not None
        assert python_entity["source"] == "routing"
        assert python_entity["success_rate"] == 0.8

    @pytest.mark.asyncio
    async def test_get_kg_entities_with_filter(self, service):
        """Should filter entities by text."""
        service.knowledge_graph["routing_patterns"] = {
            "python": {"collections_used": {"history": {"total": 5}}, "best_collection": "history"},
            "javascript": {"collections_used": {"books": {"total": 3}}, "best_collection": "books"},
        }
        # Save to disk so reload_kg finds it
        service._save_kg_sync()

        entities = await service.get_kg_entities(filter_text="python")

        assert len(entities) == 1
        assert entities[0]["entity"] == "python"

    @pytest.mark.asyncio
    async def test_get_kg_relationships(self, service):
        """Should return merged relationships."""
        service.knowledge_graph["relationships"] = {
            "django|python": {
                "co_occurrence": 10,
                "success_together": 5,
                "failure_together": 1,
            }
        }

        relationships = await service.get_kg_relationships("python")

        assert len(relationships) > 0
        django_rel = next((r for r in relationships if r["related_entity"] == "django"), None)
        assert django_rel is not None
        assert django_rel["strength"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

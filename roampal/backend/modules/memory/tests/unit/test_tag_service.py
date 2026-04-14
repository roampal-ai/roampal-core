"""
Tests for TagService — noun tag extraction and matching for TagCascade retrieval.

v0.4.5: Tags replace KG as routing mechanism.
"""

import json
import pytest
from unittest.mock import MagicMock

from roampal.backend.modules.memory.tag_service import TagService


class TestTagServiceExtract:
    """Test TagService.extract_tags() with LLM fallback."""

    def test_regex_fallback_when_no_llm(self):
        """Without LLM fn, returns [] (no regex fallback in v0.4.9)."""
        service = TagService()
        tags = service.extract_tags("Calvin drove to Boston.")
        assert tags == []  # No LLM, no tags

    def test_llm_extraction_used_when_available(self):
        """LLM fn is called first when provided."""
        llm_fn = MagicMock(return_value=["calvin", "boston", "muscle car"])
        service = TagService(llm_extract_fn=llm_fn)
        tags = service.extract_tags("Calvin drove to Boston in a muscle car.")
        llm_fn.assert_called_once()
        assert "calvin" in tags
        assert "boston" in tags

    def test_llm_failure_returns_empty(self):
        """If LLM raises, returns [] (no regex fallback in v0.4.9)."""
        llm_fn = MagicMock(side_effect=Exception("LLM unavailable"))
        service = TagService(llm_extract_fn=llm_fn)
        tags = service.extract_tags("Calvin drove to Boston.")
        assert tags == []  # No regex fallback

    def test_llm_returns_none_returns_empty(self):
        """If LLM returns None, returns [] (no regex fallback in v0.4.9)."""
        llm_fn = MagicMock(return_value=None)
        service = TagService(llm_extract_fn=llm_fn)
        tags = service.extract_tags("Calvin drove to Boston.")
        assert tags == []  # No regex fallback

    def test_llm_tags_normalized(self):
        """LLM tags are lowercased, deduped, noise filtered."""
        llm_fn = MagicMock(return_value=["Calvin", "BOSTON", "source", "Calvin"])
        service = TagService(llm_extract_fn=llm_fn)
        tags = service.extract_tags("text")
        assert "calvin" in tags
        assert "boston" in tags
        assert "source" not in tags  # noise word
        assert tags.count("calvin") == 1  # deduped

    def test_extracted_tags_registered_in_known_tags(self):
        """Tags are added to known_tags index after extraction."""
        llm_fn = MagicMock(return_value=["calvin", "joanna", "paris"])
        service = TagService(llm_extract_fn=llm_fn)
        service.extract_tags("Calvin met Joanna in Paris.")
        assert "calvin" in service.known_tags or "joanna" in service.known_tags


class TestMatchQueryTags:
    """Test query-to-tag matching."""

    def test_matches_known_tags(self):
        """Matches query words against known tag index."""
        service = TagService()
        service.add_known_tags(["calvin", "boston", "muscle car"])
        matches = service.match_query_tags("What did Calvin do in Boston?")
        assert "calvin" in matches
        assert "boston" in matches

    def test_word_boundary_matching(self):
        """'log' should NOT match 'logan'."""
        service = TagService()
        service.add_known_tags(["log", "logan"])
        matches = service.match_query_tags("Tell me about Logan.")
        assert "logan" in matches
        assert "log" not in matches

    def test_longest_first(self):
        """Matches sorted by length descending (most specific first)."""
        service = TagService()
        service.add_known_tags(["car", "muscle car", "ford mustang"])
        matches = service.match_query_tags("I want a Ford Mustang muscle car.")
        if len(matches) >= 2:
            assert len(matches[0]) >= len(matches[1])

    def test_max_8_matches(self):
        """Capped at 8 matches."""
        service = TagService()
        service.add_known_tags([f"tag{i}" for i in range(20)])
        query = " ".join([f"tag{i}" for i in range(20)])
        matches = service.match_query_tags(query)
        assert len(matches) <= 8

    def test_empty_query(self):
        service = TagService()
        service.add_known_tags(["calvin"])
        assert service.match_query_tags("") == []

    def test_no_known_tags(self):
        service = TagService()
        assert service.match_query_tags("Calvin in Boston") == []


class TestRebuildKnownTags:
    """Test rebuilding known tag index from ChromaDB collections."""

    def test_rebuild_from_collections(self):
        """Rebuilds known_tags from noun_tags metadata in collections."""
        mock_adapter = MagicMock()
        mock_adapter.collection.get.return_value = {
            "metadatas": [
                {"noun_tags": '["calvin", "boston"]'},
                {"noun_tags": '["joanna", "paris"]'},
                {"noun_tags": None},
                {},
            ]
        }
        service = TagService()
        service.rebuild_known_tags({"working": mock_adapter})

        assert "calvin" in service.known_tags
        assert "boston" in service.known_tags
        assert "joanna" in service.known_tags
        assert "paris" in service.known_tags

    def test_rebuild_handles_empty_collections(self):
        """Handles collections with no memories."""
        mock_adapter = MagicMock()
        mock_adapter.collection.get.return_value = {"metadatas": []}
        service = TagService()
        service.rebuild_known_tags({"working": mock_adapter})
        assert service.known_tag_count == 0

    def test_rebuild_handles_bad_json(self):
        """Gracefully handles malformed noun_tags JSON."""
        mock_adapter = MagicMock()
        mock_adapter.collection.get.return_value = {
            "metadatas": [{"noun_tags": "not valid json"}]
        }
        service = TagService()
        service.rebuild_known_tags({"working": mock_adapter})
        # Should not crash, just skip bad entries


class TestKnownTagManagement:
    """Test known tag index operations."""

    def test_add_known_tags(self):
        service = TagService()
        service.add_known_tags(["calvin", "boston"])
        assert "calvin" in service.known_tags
        assert "boston" in service.known_tags

    def test_known_tag_count(self):
        service = TagService()
        service.add_known_tags(["a", "b", "c"])
        assert service.known_tag_count == 3

    def test_known_tags_returns_copy(self):
        """known_tags property returns a copy, not the internal set."""
        service = TagService()
        service.add_known_tags(["calvin"])
        tags = service.known_tags
        tags.add("bogus")
        assert "bogus" not in service.known_tags

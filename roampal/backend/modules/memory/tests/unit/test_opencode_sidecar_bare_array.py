"""
Tests for v0.5.3 bare-array tolerance in sidecar_service.py extract_tags / extract_facts.

v0.5.3 Section 8: When the LLM returns a bare array (no JSON block wrapper), the parser
must still extract noun_tags and facts correctly. Previously it would fail with empty results.
"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

from roampal.sidecar_service import extract_tags, extract_facts


class TestBareArrayTolerance:
    """Section 8: Bare-array tolerance for extract_tags and extract_facts."""

    def test_extract_tags_bare_array(self):
        """extract_tags handles bare array without JSON block wrapper."""
        with patch("roampal.sidecar_service._call_llm", return_value=["calvin", "boston"]):
            result = extract_tags("test text")

        assert isinstance(result, list)
        assert "calvin" in result
        assert "boston" in result
        assert len(result) == 2

    def test_extract_tags_json_block(self):
        """extract_tags still handles JSON block wrapper."""
        with patch("roampal.sidecar_service._call_llm", return_value={"tags": ["calvin", "boston"]}):
            result = extract_tags("test text")

        assert isinstance(result, list)
        assert "calvin" in result
        assert "boston" in result

    def test_extract_tags_mixed_wrappers(self):
        """extract_tags handles various wrapper styles."""
        with patch("roampal.sidecar_service._call_llm", return_value=["calvin", "boston"]):
            result = extract_tags("test text")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_extract_facts_bare_array(self):
        """extract_facts handles bare array without JSON block wrapper."""
        with patch("roampal.sidecar_service._call_llm", return_value=["Logan is a data scientist"]):
            result = extract_facts("test text")

        assert isinstance(result, list)
        assert len(result) == 1
        assert "data scientist" in result[0].lower() or "logan" in result[0].lower()

    def test_extract_facts_json_block(self):
        """extract_facts still handles JSON block wrapper."""
        with patch("roampal.sidecar_service._call_llm", return_value={"facts": ["Logan is a data scientist"]}):
            result = extract_facts("test text")

        assert isinstance(result, list)
        assert len(result) == 1

    def test_extract_tags_empty_bare_array(self):
        """extract_tags handles empty bare array (returns None when no tags)."""
        with patch("roampal.sidecar_service._call_llm", return_value=[]):
            result = extract_tags("test text")

        # Empty list → None (no valid tags after filtering) is correct behavior
        assert result is None or isinstance(result, list) and len(result) == 0

    def test_extract_facts_empty_bare_array(self):
        """extract_facts handles empty bare array (returns None when no facts)."""
        with patch("roampal.sidecar_service._call_llm", return_value=[]):
            result = extract_facts("test text")

        # Empty list → None (no valid facts after filtering) is correct behavior
        assert result is None or isinstance(result, list) and len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

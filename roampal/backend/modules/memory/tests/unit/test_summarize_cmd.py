"""
Tests for `roampal summarize` CLI command (v0.4.6).

Covers:
1. Only memories over max_chars threshold are candidates
2. Tag extraction from summaries via sidecar
3. Skip tag extraction when memory already has noun_tags
4. No fact extraction (removed in v0.4.6)
"""

import argparse
from unittest.mock import patch, MagicMock


def _make_args(**overrides):
    """Build args namespace matching cmd_summarize expectations."""
    defaults = {
        "port": None,
        "max_chars": 300,
        "collection": "working",  # single collection to simplify mocking
        "dry_run": False,
        "limit": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _mock_search_response(memories):
    """Build a mock httpx response for /api/search."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"results": memories}
    return resp


def _mock_update_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"ok": True}
    return resp


def _mock_post_for(memories, update_payloads=None):
    """Return a mock httpx.post that serves search + update."""
    def mock_post(url, **kwargs):
        if "/api/search" in url:
            return _mock_search_response(memories)
        if "/api/memory/update-content" in url:
            if update_payloads is not None:
                update_payloads.append(kwargs.get("json", {}))
            return _mock_update_response()
        if "/api/record-response" in url:
            raise AssertionError("record-response should not be called — fact extraction removed in v0.4.6")
        if "/api/health" in url:
            return MagicMock(status_code=200)
        return MagicMock(status_code=200)
    return mock_post


class TestSummarizeThreshold:
    """Only memories over max_chars are candidates."""

    def test_short_memories_skipped(self):
        """Memories under max_chars are not summarized."""
        from roampal.cli import cmd_summarize

        short_memory = {"content": "Short memory", "id": "working_short", "metadata": {}}
        long_memory = {"content": "x" * 500, "id": "working_long", "metadata": {}}

        call_count = {"summarize": 0}

        def mock_summarize(text):
            call_count["summarize"] += 1
            return "Summarized version"

        with patch("roampal.cli._check_sidecar_configured", return_value=True), \
             patch("roampal.sidecar_service.get_backend_info", return_value="Ollama (test)"), \
             patch("roampal.sidecar_service.summarize_only", side_effect=mock_summarize), \
             patch("roampal.sidecar_service.extract_tags", return_value=["test"]), \
             patch("roampal.cli._is_interactive", return_value=False), \
             patch("httpx.post", side_effect=_mock_post_for([short_memory, long_memory])), \
             patch("httpx.get", return_value=MagicMock(status_code=200)):
            cmd_summarize(_make_args(max_chars=300))

        # Only the long memory should have been summarized
        assert call_count["summarize"] == 1

    def test_already_summarized_skipped(self):
        """Memories with summarized_at metadata are skipped."""
        from roampal.cli import cmd_summarize

        already_done = {
            "content": "x" * 500,
            "id": "working_done",
            "metadata": {"summarized_at": "2026-01-01T00:00:00"}
        }

        call_count = {"summarize": 0}

        def mock_summarize(text):
            call_count["summarize"] += 1
            return "Summarized"

        with patch("roampal.sidecar_service.get_backend_info", return_value="Ollama (test)"), \
             patch("roampal.sidecar_service.summarize_only", side_effect=mock_summarize), \
             patch("roampal.cli._is_interactive", return_value=False), \
             patch("httpx.post", side_effect=_mock_post_for([already_done])), \
             patch("httpx.get", return_value=MagicMock(status_code=200)):
            cmd_summarize(_make_args(max_chars=300))

        assert call_count["summarize"] == 0


class TestTagExtraction:
    """Tag extraction during summarize."""

    def test_extracts_tags_from_summary(self):
        """Tags are extracted from the summary text and sent to update endpoint."""
        from roampal.cli import cmd_summarize

        memory = {"content": "x" * 500, "id": "working_notags", "metadata": {}}

        extract_calls = []

        def mock_extract_tags(text):
            extract_calls.append(text)
            return ["roampal", "python"]

        update_payloads = []

        with patch("roampal.sidecar_service.get_backend_info", return_value="Ollama (test)"), \
             patch("roampal.sidecar_service.summarize_only", return_value="Summary about roampal python project"), \
             patch("roampal.sidecar_service.extract_tags", side_effect=mock_extract_tags), \
             patch("roampal.cli._is_interactive", return_value=False), \
             patch("httpx.post", side_effect=_mock_post_for([memory], update_payloads)), \
             patch("httpx.get", return_value=MagicMock(status_code=200)):
            cmd_summarize(_make_args(max_chars=300))

        # extract_tags called with the summary text
        assert len(extract_calls) == 1
        assert "roampal" in extract_calls[0].lower()

        # Update payload includes the extracted tags
        assert len(update_payloads) == 1
        assert update_payloads[0]["noun_tags"] == ["roampal", "python"]

    def test_skips_tag_extraction_when_tags_exist(self):
        """Memories with existing noun_tags don't get re-extracted."""
        from roampal.cli import cmd_summarize

        memory_with_tags = {
            "content": "x" * 500,
            "id": "working_hastags",
            "metadata": {"noun_tags": '["existing_tag"]'}
        }

        extract_calls = []

        def mock_extract_tags(text):
            extract_calls.append(text)
            return ["new_tag"]

        update_payloads = []

        with patch("roampal.sidecar_service.get_backend_info", return_value="Ollama (test)"), \
             patch("roampal.sidecar_service.summarize_only", return_value="Summary text"), \
             patch("roampal.sidecar_service.extract_tags", side_effect=mock_extract_tags), \
             patch("roampal.cli._is_interactive", return_value=False), \
             patch("httpx.post", side_effect=_mock_post_for([memory_with_tags], update_payloads)), \
             patch("httpx.get", return_value=MagicMock(status_code=200)):
            cmd_summarize(_make_args(max_chars=300))

        # extract_tags should NOT have been called
        assert len(extract_calls) == 0
        # Update should not include noun_tags (existing ones preserved)
        assert len(update_payloads) == 1
        assert "noun_tags" not in update_payloads[0]


class TestNoFactExtraction:
    """Fact extraction removed in v0.4.6."""

    def test_no_record_response_calls(self):
        """record-response endpoint is never called (fact extraction removed)."""
        from roampal.cli import cmd_summarize

        memory = {"content": "x" * 500, "id": "working_nofacts", "metadata": {}}

        # _mock_post_for raises AssertionError if /api/record-response is hit
        with patch("roampal.sidecar_service.get_backend_info", return_value="Ollama (test)"), \
             patch("roampal.sidecar_service.summarize_only", return_value="Summary"), \
             patch("roampal.sidecar_service.extract_tags", return_value=["tag"]), \
             patch("roampal.cli._is_interactive", return_value=False), \
             patch("httpx.post", side_effect=_mock_post_for([memory])), \
             patch("httpx.get", return_value=MagicMock(status_code=200)):
            cmd_summarize(_make_args(max_chars=300))

    def test_no_extract_facts_in_source(self):
        """The summarize code path should not reference extract_facts."""
        import inspect
        from roampal.cli import cmd_summarize
        source = inspect.getsource(cmd_summarize)
        assert "extract_facts" not in source

"""
Tests for MCP server thin HTTP client (mcp/server.py).

v0.3.2: MCP server is a lightweight HTTP proxy — no ChromaDB/PyTorch.
Tests cover:
- Helper functions (_humanize_age, _format_outcomes, _is_temporal_query, _sort_results)
- _api_call HTTP proxy
- Tool handler logic (search_memory, add_to_memory_bank, etc.)
- Server lifecycle (_ensure_server_running, _start_fastapi_server)
"""

import sys
import os
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))

from roampal.mcp.server import (
    _humanize_age,
    _format_outcomes,
    _is_temporal_query,
    _sort_results,
)


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestHumanizeAge:
    """Test _humanize_age timestamp formatting."""

    def test_recent_timestamp(self):
        """Timestamps within seconds show 'now'."""
        ts = datetime.now().isoformat()
        assert _humanize_age(ts) == "now"

    def test_minutes_ago(self):
        """Timestamps minutes ago show 'Xm'."""
        ts = (datetime.now() - timedelta(minutes=15)).isoformat()
        result = _humanize_age(ts)
        assert result.endswith("m")
        assert int(result[:-1]) >= 14  # Allow small timing variance

    def test_hours_ago(self):
        """Timestamps hours ago show 'Xh'."""
        ts = (datetime.now() - timedelta(hours=5)).isoformat()
        result = _humanize_age(ts)
        assert result.endswith("h")

    def test_days_ago(self):
        """Timestamps days ago show 'Xd'."""
        ts = (datetime.now() - timedelta(days=3)).isoformat()
        assert _humanize_age(ts) == "3d"

    def test_months_ago(self):
        """Timestamps months ago show 'Xmo'."""
        ts = (datetime.now() - timedelta(days=60)).isoformat()
        result = _humanize_age(ts)
        assert result.endswith("mo")

    def test_empty_string(self):
        """Empty string returns empty."""
        assert _humanize_age("") == ""

    def test_none_returns_empty(self):
        """None returns empty."""
        assert _humanize_age(None) == ""

    def test_invalid_timestamp(self):
        """Invalid timestamp returns empty."""
        assert _humanize_age("not-a-date") == ""


class TestFormatOutcomes:
    """Test _format_outcomes for outcome history display."""

    def test_all_worked(self):
        """Three worked outcomes show [YYY]."""
        history = json.dumps([
            {"outcome": "worked"}, {"outcome": "worked"}, {"outcome": "worked"}
        ])
        assert _format_outcomes(history) == "[YYY]"

    def test_mixed_outcomes(self):
        """Mixed outcomes show correct symbols."""
        history = json.dumps([
            {"outcome": "worked"}, {"outcome": "failed"}, {"outcome": "partial"}
        ])
        assert _format_outcomes(history) == "[YN~]"

    def test_only_last_three(self):
        """Only last 3 outcomes are shown."""
        history = json.dumps([
            {"outcome": "failed"}, {"outcome": "failed"},
            {"outcome": "worked"}, {"outcome": "worked"}, {"outcome": "partial"}
        ])
        assert _format_outcomes(history) == "[YY~]"

    def test_empty_history(self):
        """Empty history returns empty string."""
        assert _format_outcomes(json.dumps([])) == ""

    def test_none_input(self):
        """None returns empty string."""
        assert _format_outcomes(None) == ""

    def test_empty_string(self):
        """Empty string returns empty."""
        assert _format_outcomes("") == ""

    def test_invalid_json(self):
        """Invalid JSON returns empty."""
        assert _format_outcomes("not json") == ""


class TestIsTemporalQuery:
    """Test temporal query detection for auto-recency sort."""

    def test_detects_recent(self):
        assert _is_temporal_query("show me recent work") is True

    def test_detects_last(self):
        assert _is_temporal_query("what was the last thing we discussed") is True

    def test_detects_yesterday(self):
        assert _is_temporal_query("what did we do yesterday") is True

    def test_detects_earlier(self):
        assert _is_temporal_query("earlier we talked about X") is True

    def test_detects_ago(self):
        assert _is_temporal_query("what did we do 2 days ago") is True

    def test_non_temporal(self):
        assert _is_temporal_query("how does authentication work") is False

    def test_case_insensitive(self):
        assert _is_temporal_query("Show me RECENT changes") is True


class TestSortResults:
    """Test result sorting."""

    def test_sort_by_recency(self):
        """Recency sort orders by timestamp descending."""
        results = [
            {"metadata": {"timestamp": "2025-01-01T00:00:00"}},
            {"metadata": {"timestamp": "2025-01-03T00:00:00"}},
            {"metadata": {"timestamp": "2025-01-02T00:00:00"}},
        ]
        sorted_results = _sort_results(results, "recency")
        assert sorted_results[0]["metadata"]["timestamp"] == "2025-01-03T00:00:00"
        assert sorted_results[2]["metadata"]["timestamp"] == "2025-01-01T00:00:00"

    def test_sort_by_score(self):
        """Score sort orders by score descending."""
        results = [
            {"metadata": {"score": 0.3}},
            {"metadata": {"score": 0.9}},
            {"metadata": {"score": 0.6}},
        ]
        sorted_results = _sort_results(results, "score")
        assert sorted_results[0]["metadata"]["score"] == 0.9
        assert sorted_results[2]["metadata"]["score"] == 0.3

    def test_default_preserves_order(self):
        """Default sort preserves original order."""
        results = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        sorted_results = _sort_results(results, "relevance")
        assert sorted_results == results


# ============================================================================
# _api_call HTTP Proxy Tests
# ============================================================================

class TestApiCall:
    """Test _api_call HTTP proxy helper."""

    @pytest.mark.asyncio
    async def test_post_request(self):
        """POST request sends JSON payload."""
        from roampal.mcp.server import _api_call

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        # httpx is imported inline inside _api_call, so we patch the module in sys.modules
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await _api_call("POST", "/api/search", {"query": "test"})
            assert result == {"success": True}
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_request(self):
        """GET request works without payload."""
        from roampal.mcp.server import _api_call

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await _api_call("GET", "/api/health")
            assert result == {"status": "healthy"}


# ============================================================================
# Session ID Uniqueness
# ============================================================================

class TestMcpSessionId:
    """Test that MCP uses 'default' session ID for injection_map lookup.

    v0.3.2: Changed from random 'mcp_{uuid}' to 'default' to fix multi-session
    scoring mismatch. The injection_map in main.py now tracks which doc_ids
    were injected to which conversation, allowing correct scoring correlation.
    """

    def test_session_id_is_default(self):
        """Session ID is 'default' to trigger injection_map lookup."""
        from roampal.mcp.server import _mcp_session_id
        assert _mcp_session_id == "default"


# ============================================================================
# Port Configuration
# ============================================================================

class TestPortConfig:
    """Test port configuration for dev/prod."""

    def test_prod_port(self):
        from roampal.mcp.server import PROD_PORT
        assert PROD_PORT == 27182

    def test_dev_port(self):
        from roampal.mcp.server import DEV_PORT
        assert DEV_PORT == 27183

    def test_get_port_prod(self):
        """Default port is prod."""
        from roampal.mcp import server
        original = server._dev_mode
        server._dev_mode = False
        try:
            port = server._get_port()
            assert port == 27182
        finally:
            server._dev_mode = original

    def test_get_port_dev(self):
        """Dev mode uses dev port."""
        from roampal.mcp import server
        original = server._dev_mode
        server._dev_mode = True
        try:
            port = server._get_port()
            assert port == 27183
        finally:
            server._dev_mode = original

    def test_get_port_env_override(self):
        """ROAMPAL_PORT env var overrides."""
        from roampal.mcp import server
        with patch.dict(os.environ, {"ROAMPAL_PORT": "9999"}):
            port = server._get_port()
            assert port == 9999


# ============================================================================
# Server Lifecycle
# ============================================================================

class TestServerLifecycle:
    """Test _start_fastapi_server and _ensure_server_running."""

    def test_start_skips_if_already_started(self):
        """_start_fastapi_server is idempotent."""
        from roampal.mcp import server
        original = server._fastapi_started
        server._fastapi_started = True
        try:
            with patch("subprocess.Popen") as mock_popen:
                server._start_fastapi_server()
                mock_popen.assert_not_called()
        finally:
            server._fastapi_started = original

    def test_start_skips_if_port_in_use(self):
        """_start_fastapi_server skips if port is already bound."""
        from roampal.mcp import server
        original = server._fastapi_started
        server._fastapi_started = False
        try:
            with patch.object(server, "_is_port_in_use", return_value=True):
                with patch("subprocess.Popen") as mock_popen:
                    server._start_fastapi_server()
                    mock_popen.assert_not_called()
                    assert server._fastapi_started is True
        finally:
            server._fastapi_started = original

    def test_cleanup_terminates_process(self):
        """_cleanup_fastapi terminates the subprocess."""
        from roampal.mcp import server
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running
        mock_proc.wait.return_value = 0

        original = server._fastapi_process
        server._fastapi_process = mock_proc
        try:
            server._cleanup_fastapi()
            mock_proc.terminate.assert_called_once()
        finally:
            server._fastapi_process = original

    def test_cleanup_noop_when_no_process(self):
        """_cleanup_fastapi is noop when no process."""
        from roampal.mcp import server
        original = server._fastapi_process
        server._fastapi_process = None
        try:
            # Should not raise
            server._cleanup_fastapi()
        finally:
            server._fastapi_process = original


# ============================================================================
# MCP Tool Handler Tests (v0.3.2)
# ============================================================================

class TestToolHandlers:
    """Test all 7 MCP tool handlers dispatched via call_tool.

    These tests capture the call_tool closure from run_mcp_server() by
    mocking the MCP Server class and asyncio.run, then invoke the handler
    directly with mocked _api_call and _ensure_server_running.
    """

    @pytest.fixture
    def tool_handler(self):
        """Capture the call_tool handler from run_mcp_server."""
        import roampal.mcp.server as server_module

        captured = {}

        class FakeServer:
            def __init__(self, name):
                self.name = name

            def list_prompts(self):
                return lambda fn: fn

            def list_resources(self):
                return lambda fn: fn

            def list_tools(self):
                return lambda fn: fn

            def call_tool(self):
                def deco(fn):
                    captured['handler'] = fn
                    return fn
                return deco

            def create_initialization_options(self):
                return {}

            async def run(self, *a, **kw):
                pass

        with patch('roampal.mcp.server.Server', FakeServer), \
             patch('roampal.mcp.server.stdio_server'), \
             patch.object(server_module, '_start_fastapi_server'), \
             patch('asyncio.run'):
            server_module.run_mcp_server(dev=False)

        return captured['handler']

    # ---- search_memory ----

    @pytest.mark.asyncio
    async def test_search_memory_basic(self, tool_handler):
        """search_memory returns formatted results."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={
            "results": [
                {
                    "id": "working_abc",
                    "content": "Test pattern",
                    "collection": "working",
                    "metadata": {"score": 0.8, "created_at": "2025-01-01T00:00:00"}
                }
            ]
        })

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("search_memory", {"query": "test"})

        assert len(result) == 1
        assert "Test pattern" in result[0].text
        assert "working" in result[0].text

    @pytest.mark.asyncio
    async def test_search_memory_empty_results(self, tool_handler):
        """search_memory handles no results."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"results": []})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("search_memory", {"query": "nothing"})

        assert "No results found" in result[0].text

    @pytest.mark.asyncio
    async def test_search_memory_temporal_auto_sort(self, tool_handler):
        """search_memory auto-detects temporal queries for recency sort."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={
            "results": [
                {"id": "a", "content": "old", "collection": "working",
                 "metadata": {"timestamp": "2025-01-01T00:00:00"}},
                {"id": "b", "content": "new", "collection": "working",
                 "metadata": {"timestamp": "2025-06-01T00:00:00"}},
            ]
        })

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("search_memory", {"query": "last thing we did"})

        # sort_by should have been set to "recency" due to "last"
        call_payload = mock_api.call_args[1].get("payload") or mock_api.call_args[0][2]
        assert call_payload.get("sort_by") == "recency"

    @pytest.mark.asyncio
    async def test_search_memory_api_error(self, tool_handler):
        """search_memory handles API errors gracefully."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(side_effect=Exception("Connection refused"))

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("search_memory", {"query": "test"})

        assert "Search error" in result[0].text

    @pytest.mark.asyncio
    async def test_search_memory_with_collections(self, tool_handler):
        """search_memory passes collections filter."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"results": []})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            await tool_handler("search_memory", {
                "query": "test",
                "collections": ["patterns", "memory_bank"],
                "limit": 3
            })

        call_payload = mock_api.call_args[0][2]
        assert call_payload["collections"] == ["patterns", "memory_bank"]
        assert call_payload["limit"] == 3

    # ---- add_to_memory_bank ----

    @pytest.mark.asyncio
    async def test_add_to_memory_bank(self, tool_handler):
        """add_to_memory_bank stores a fact."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"doc_id": "mb_test123"})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("add_to_memory_bank", {
                "content": "User prefers dark mode",
                "tags": ["preference"],
                "importance": 0.8
            })

        assert "mb_test123" in result[0].text
        call_payload = mock_api.call_args[0][2]
        assert call_payload["content"] == "User prefers dark mode"
        assert call_payload["tags"] == ["preference"]

    @pytest.mark.asyncio
    async def test_add_to_memory_bank_with_always_inject(self, tool_handler):
        """add_to_memory_bank passes always_inject flag."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"doc_id": "mb_identity"})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("add_to_memory_bank", {
                "content": "User's name is Logan",
                "tags": ["identity"],
                "always_inject": True
            })

        call_payload = mock_api.call_args[0][2]
        assert call_payload["always_inject"] is True

    # ---- update_memory ----

    @pytest.mark.asyncio
    async def test_update_memory_success(self, tool_handler):
        """update_memory returns success message."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"success": True, "doc_id": "mb_updated"})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("update_memory", {
                "old_content": "old fact",
                "new_content": "corrected fact"
            })

        assert "Updated" in result[0].text
        assert "mb_updated" in result[0].text

    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, tool_handler):
        """update_memory handles not found case."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"success": False})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("update_memory", {
                "old_content": "nonexistent",
                "new_content": "new"
            })

        assert "not found" in result[0].text

    # ---- delete_memory ----

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, tool_handler):
        """delete_memory returns success message."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"success": True})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("delete_memory", {"content": "outdated fact"})

        assert "deleted successfully" in result[0].text

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, tool_handler):
        """delete_memory handles not found case."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"success": False})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("delete_memory", {"content": "nonexistent"})

        assert "not found" in result[0].text

    # ---- score_response ----

    @pytest.mark.asyncio
    async def test_score_response_with_memory_scores(self, tool_handler):
        """score_response sends per-memory scores."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"documents_scored": 3})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("score_response", {
                "outcome": "worked",
                "memory_scores": {
                    "patterns_abc": "worked",
                    "working_def": "partial",
                    "history_ghi": "unknown"
                }
            })

        assert "Scored" in result[0].text
        assert "3 memories" in result[0].text
        call_payload = mock_api.call_args[0][2]
        assert call_payload["outcome"] == "worked"
        assert call_payload["memory_scores"]["patterns_abc"] == "worked"

    @pytest.mark.asyncio
    async def test_score_response_backward_compat_related(self, tool_handler):
        """score_response supports deprecated 'related' parameter."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"documents_scored": 1})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("score_response", {
                "outcome": "failed",
                "memory_scores": {},
                "related": ["doc_123"]
            })

        # When memory_scores is empty but related is provided, related should be used
        # Actually, looking at the code: if memory_scores (truthy), use that.
        # Empty dict is falsy, so related branch kicks in
        call_payload = mock_api.call_args[0][2]
        assert "related" in call_payload

    # ---- record_response ----

    @pytest.mark.asyncio
    async def test_record_response_success(self, tool_handler):
        """record_response stores key takeaway."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"success": True, "doc_id": "working_take1"})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("record_response", {
                "key_takeaway": "User prefers concise responses"
            })

        assert "Recorded" in result[0].text
        assert "concise" in result[0].text

    @pytest.mark.asyncio
    async def test_record_response_empty_takeaway(self, tool_handler):
        """record_response rejects empty key_takeaway."""
        import roampal.mcp.server as server_module

        with patch.object(server_module, '_ensure_server_running', return_value=True):
            result = await tool_handler("record_response", {"key_takeaway": ""})

        assert "Error" in result[0].text
        assert "key_takeaway" in result[0].text

    # ---- get_context_insights ----

    @pytest.mark.asyncio
    async def test_get_context_insights_success(self, tool_handler):
        """get_context_insights returns formatted context."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={
            "user_facts": [{"content": "User is a developer"}],
            "relevant_memories": [
                {"content": "Previous testing work", "collection": "working",
                 "metadata": {"score": 0.8}}
            ],
            "doc_ids": ["working_abc", "patterns_def"]
        })

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api), \
             patch.object(server_module, '_get_update_notice', return_value=""):
            result = await tool_handler("get_context_insights", {"query": "testing"})

        text = result[0].text
        assert "Known Context" in text
        assert "Memory Bank" in text
        assert "developer" in text
        assert "Relevant Memories" in text
        assert "Cached 2 doc_ids" in text

    @pytest.mark.asyncio
    async def test_get_context_insights_empty_query(self, tool_handler):
        """get_context_insights rejects empty query."""
        import roampal.mcp.server as server_module

        with patch.object(server_module, '_ensure_server_running', return_value=True):
            result = await tool_handler("get_context_insights", {"query": ""})

        assert "Error" in result[0].text
        assert "query" in result[0].text

    @pytest.mark.asyncio
    async def test_get_context_insights_no_context(self, tool_handler):
        """get_context_insights handles no matching context."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={
            "user_facts": [],
            "relevant_memories": [],
            "doc_ids": []
        })

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api), \
             patch.object(server_module, '_get_update_notice', return_value=""):
            result = await tool_handler("get_context_insights", {"query": "new topic"})

        assert "No relevant context" in result[0].text

    # ---- unknown tool ----

    @pytest.mark.asyncio
    async def test_unknown_tool(self, tool_handler):
        """Unknown tool name returns error message."""
        import roampal.mcp.server as server_module

        with patch.object(server_module, '_ensure_server_running', return_value=True):
            result = await tool_handler("nonexistent_tool", {})

        assert "Unknown tool" in result[0].text

    # ---- general error handling ----

    @pytest.mark.asyncio
    async def test_tool_general_exception(self, tool_handler):
        """General exception in tool returns error text."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(side_effect=RuntimeError("unexpected crash"))

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("add_to_memory_bank", {"content": "test"})

        assert "Error" in result[0].text
        assert "unexpected crash" in result[0].text


# ============================================================================
# Update Notice Tests
# ============================================================================

class TestUpdateNotice:
    """Test _get_update_notice and _check_for_updates."""

    def test_no_update_returns_empty(self):
        """No update available returns empty string."""
        from roampal.mcp.server import _get_update_notice
        import roampal.mcp.server as server_module

        server_module._update_check_cache = (False, "0.3.2", "0.3.2")
        try:
            result = _get_update_notice()
            assert result == ""
        finally:
            server_module._update_check_cache = None

    def test_update_available_returns_notice(self):
        """Update available returns formatted notice."""
        from roampal.mcp.server import _get_update_notice
        import roampal.mcp.server as server_module

        server_module._update_check_cache = (True, "0.3.1", "0.3.2")
        try:
            result = _get_update_notice()
            assert "Update available" in result
            assert "0.3.2" in result
        finally:
            server_module._update_check_cache = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

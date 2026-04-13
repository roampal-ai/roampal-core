"""
Tests for OpenCode sidecar-only scoring architecture (v0.3.7).

Verifies:
- score_memories tool always registered (6 tools for both platforms)
- record_response registered for both platforms
- Claude Code schema includes exchange_summary + exchange_outcome fields
- Server endpoints still return scoring data for sidecar consumption
"""

import sys
import os
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))


# ============================================================================
# Helper: Capture list_tools from run_mcp_server with platform override
# ============================================================================

async def _get_tools_for_platform(platform: str = ""):
    """Capture list_tools handler and call it with the given platform flag set.

    _is_opencode is a module-level variable read at call time (not closure capture time),
    so we must set it both during run_mcp_server() AND during the list_tools() call.
    """
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
            def deco(fn):
                captured['list_tools'] = fn
                return fn
            return deco
        def call_tool(self):
            return lambda fn: fn
        def create_initialization_options(self):
            return {}
        async def run(self, *a, **kw):
            pass

    original_is_opencode = server_module._is_opencode
    server_module._is_opencode = (platform.lower() == "opencode")

    try:
        with patch('roampal.mcp.server.Server', FakeServer), \
             patch('roampal.mcp.server.stdio_server'), \
             patch.object(server_module, '_start_fastapi_server'), \
             patch('asyncio.run'):
            server_module.run_mcp_server(dev=False)

        # Call list_tools with the flag still set
        tools = await captured['list_tools']()
        return tools
    finally:
        server_module._is_opencode = original_is_opencode


# ============================================================================
# MCP Tool Registration Tests
# ============================================================================

class TestOpenCodeToolRegistration:
    """score_memories always registered (6 tools for both platforms).

    v0.3.7: score_memories is always registered so OpenCode sidecar can call it.
    The model never sees the scoring prompt — sidecar handles scoring silently.
    """

    @pytest.mark.asyncio
    async def test_score_memories_present_for_opencode(self):
        """score_memories IS in tool list for OpenCode (sidecar calls it)."""
        tools = await _get_tools_for_platform("opencode")
        tool_names = [t.name for t in tools]
        assert "score_memories" in tool_names

    @pytest.mark.asyncio
    async def test_score_memories_present_for_claude_code(self):
        """score_memories IS in tool list when platform is unset (Claude Code)."""
        tools = await _get_tools_for_platform("")
        tool_names = [t.name for t in tools]
        assert "score_memories" in tool_names

    @pytest.mark.asyncio
    async def test_record_response_present_for_both(self):
        """record_response is registered for both OpenCode and Claude Code."""
        for platform in ["opencode", ""]:
            tools = await _get_tools_for_platform(platform)
            tool_names = [t.name for t in tools]
            assert "record_response" in tool_names, f"record_response missing for platform='{platform}'"

    @pytest.mark.asyncio
    async def test_both_platforms_have_6_tools(self):
        """Both platforms register all 6 tools."""
        expected = {"search_memory", "add_to_memory_bank", "update_memory", "delete_memory", "score_memories", "record_response"}
        for platform in ["opencode", ""]:
            tools = await _get_tools_for_platform(platform)
            tool_names = [t.name for t in tools]
            assert set(tool_names) == expected, f"Wrong tools for platform='{platform}': {tool_names}"

    @pytest.mark.asyncio
    async def test_claude_code_score_memories_schema(self):
        """Claude Code score_memories schema includes exchange_summary + exchange_outcome."""
        tools = await _get_tools_for_platform("")
        score_tool = next(t for t in tools if t.name == "score_memories")
        schema = score_tool.inputSchema
        assert "memory_scores" in schema["properties"]
        assert "exchange_summary" in schema["properties"]
        assert "exchange_outcome" in schema["properties"]
        assert "memory_scores" in schema["required"]


# ============================================================================
# Server Endpoint Regression Tests
# ============================================================================

class TestServerEndpointRegression:
    """Verify server endpoints still provide data the sidecar needs."""

    @pytest.fixture
    def tool_handler(self):
        """Capture the call_tool handler (Claude Code mode for full coverage)."""
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

    @pytest.mark.asyncio
    async def test_score_memories_still_works_for_claude_code(self, tool_handler):
        """score_memories handler processes requests correctly (Claude Code path)."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"documents_scored": 2})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("score_memories", {
                "memory_scores": {"pat_1": "worked", "hist_2": "partial"},
                "exchange_summary": "User asked about auth, I explained JWT flow",
                "exchange_outcome": "worked"
            })

        assert "Scored" in result[0].text
        call_payload = mock_api.call_args[0][2]
        assert call_payload["outcome"] == "worked"
        assert call_payload["memory_scores"]["pat_1"] == "worked"
        assert call_payload["exchange_summary"] == "User asked about auth, I explained JWT flow"

    @pytest.mark.asyncio
    async def test_record_response_works_for_opencode(self, tool_handler):
        """record_response stores key takeaways (used by both platforms)."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"success": True, "doc_id": "working_take1"})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("record_response", {
                "key_takeaway": "Sidecar handles all scoring in OpenCode"
            })

        assert "Recorded" in result[0].text


# ============================================================================
# Sidecar Scoring Flow Tests (unit-level, mocking HTTP)
# ============================================================================

class TestSidecarScoringFlow:
    """Test that the scoring data pipeline works correctly for sidecar consumption.

    These test the MCP server's record-outcome path which the sidecar plugin calls
    via HTTP after scoreExchangeViaLLM completes.
    """

    @pytest.fixture
    def tool_handler(self):
        """Capture call_tool handler."""
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

    @pytest.mark.asyncio
    async def test_score_memories_per_memory_scores(self, tool_handler):
        """Per-memory scores are forwarded individually via record-outcome."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"documents_scored": 3})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("score_memories", {
                "memory_scores": {
                    "patterns_abc": "worked",
                    "working_def": "unknown",
                    "history_ghi": "failed"
                },
                "exchange_outcome": "partial"
            })

        call_payload = mock_api.call_args[0][2]
        assert call_payload["memory_scores"]["patterns_abc"] == "worked"
        assert call_payload["memory_scores"]["working_def"] == "unknown"
        assert call_payload["memory_scores"]["history_ghi"] == "failed"
        assert call_payload["outcome"] == "partial"

    @pytest.mark.asyncio
    async def test_score_memories_missing_scores_uses_blanket_outcome(self, tool_handler):
        """Empty memory_scores still sends exchange outcome."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"documents_scored": 0})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("score_memories", {
                "memory_scores": {},
                "exchange_outcome": "worked"
            })

        call_payload = mock_api.call_args[0][2]
        assert call_payload["outcome"] == "worked"
        # Empty memory_scores is falsy, so it won't be in payload
        assert "memory_scores" not in call_payload

    @pytest.mark.asyncio
    async def test_score_memories_invalid_outcome_falls_back(self, tool_handler):
        """Missing exchange_outcome falls back to 'unknown'."""
        import roampal.mcp.server as server_module

        mock_api = AsyncMock(return_value={"documents_scored": 1})

        with patch.object(server_module, '_ensure_server_running', return_value=True), \
             patch.object(server_module, '_api_call', mock_api):
            result = await tool_handler("score_memories", {
                "memory_scores": {"pat_1": "worked"}
            })

        call_payload = mock_api.call_args[0][2]
        # No exchange_outcome → falls back to "unknown"
        assert call_payload["outcome"] == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

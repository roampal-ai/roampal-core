"""
Tests for OpenCode plugin (plugins/opencode/roampal.ts).

Since the plugin is TypeScript and the project uses Python test infrastructure,
these tests perform structural validation of the plugin file to ensure:
- Correct exports (RoampalPlugin, default export)
- All required hook handlers present
- Event handlers for all 5 event types
- Self-healing (restartServer) implementation
- Caching architecture (cachedContext Map)
- Split delivery (unshift/push pattern)
- Port configuration matches Python constants
- Session state cleanup on session.deleted
"""

import sys
import os
import re
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))


PLUGIN_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "plugins" / "opencode" / "roampal.ts"


@pytest.fixture
def plugin_source():
    """Read the plugin source file."""
    if not PLUGIN_PATH.exists():
        pytest.skip(f"Plugin file not found: {PLUGIN_PATH}")
    return PLUGIN_PATH.read_text(encoding="utf-8")


# ============================================================================
# Export Structure
# ============================================================================

class TestPluginExports:
    """Verify the plugin has correct TypeScript exports."""

    def test_exports_roampal_plugin(self, plugin_source):
        """Plugin exports RoampalPlugin as named export."""
        assert "export const RoampalPlugin" in plugin_source

    def test_exports_default(self, plugin_source):
        """Plugin has a default export."""
        assert "export default RoampalPlugin" in plugin_source

    def test_imports_plugin_type(self, plugin_source):
        """Plugin imports Plugin type from @opencode-ai/plugin."""
        assert 'import type { Plugin } from "@opencode-ai/plugin"' in plugin_source


# ============================================================================
# Hook Handlers
# ============================================================================

class TestHookHandlers:
    """Verify all required hook handlers are defined."""

    def test_chat_message_handler(self, plugin_source):
        """Plugin has chat.message hook handler."""
        assert '"chat.message"' in plugin_source

    def test_system_transform_handler(self, plugin_source):
        """Plugin has experimental.chat.system.transform hook handler."""
        assert '"experimental.chat.system.transform"' in plugin_source

    def test_event_handler(self, plugin_source):
        """Plugin has event hook handler."""
        # The event handler is defined as `event: async`
        assert re.search(r'event:\s*async', plugin_source)


# ============================================================================
# Event Types
# ============================================================================

class TestEventTypes:
    """Verify all 5 event types are handled."""

    def test_session_created(self, plugin_source):
        """Handles session.created event."""
        assert '"session.created"' in plugin_source

    def test_session_deleted(self, plugin_source):
        """Handles session.deleted event."""
        assert '"session.deleted"' in plugin_source

    def test_session_idle(self, plugin_source):
        """Handles session.idle event."""
        assert '"session.idle"' in plugin_source

    def test_message_updated(self, plugin_source):
        """Handles message.updated event."""
        assert '"message.updated"' in plugin_source

    def test_message_part_updated(self, plugin_source):
        """Handles message.part.updated event."""
        assert '"message.part.updated"' in plugin_source


# ============================================================================
# Event Property Paths (v0.3.2 fix)
# ============================================================================

class TestEventPropertyPaths:
    """Verify correct OpenCode event property paths (v0.3.2 fix)."""

    def test_session_created_uses_properties_info_id(self, plugin_source):
        """session.created reads event.properties.info.id."""
        assert "event.properties?.info?.id" in plugin_source

    def test_message_updated_uses_properties_info(self, plugin_source):
        """message.updated reads event.properties.info.sessionID."""
        assert "event.properties?.info" in plugin_source
        # Should check for role=assistant
        assert 'info.role !== "assistant"' in plugin_source or 'info?.role !== "assistant"' in plugin_source

    def test_message_part_updated_uses_properties_part(self, plugin_source):
        """message.part.updated reads event.properties.part."""
        assert "event.properties?.part" in plugin_source
        # Should filter by type=text
        assert 'part.type !== "text"' in plugin_source

    def test_session_idle_uses_properties_sessionID(self, plugin_source):
        """session.idle reads event.properties.sessionID."""
        assert "event.properties?.sessionID" in plugin_source


# ============================================================================
# Self-Healing (v0.3.2)
# ============================================================================

class TestSelfHealing:
    """Verify self-healing server restart implementation."""

    def test_restart_server_function_exists(self, plugin_source):
        """restartServer() function is defined."""
        assert "async function restartServer()" in plugin_source

    def test_restart_in_progress_guard(self, plugin_source):
        """restartServer has _restartInProgress guard to prevent concurrent restarts."""
        assert "_restartInProgress" in plugin_source

    def test_cross_platform_port_killing(self, plugin_source):
        """restartServer handles both Windows and Unix for port killing."""
        assert "win32" in plugin_source
        assert "taskkill" in plugin_source
        assert "lsof" in plugin_source or "kill" in plugin_source

    def test_health_polling(self, plugin_source):
        """restartServer polls /api/health after starting server."""
        assert "/api/health" in plugin_source

    def test_get_context_retries_on_503(self, plugin_source):
        """getContextFromRoampal retries after 503."""
        # Should have a 503 check in getContextFromRoampal
        assert "503" in plugin_source
        # Should call restartServer on 503
        assert re.search(r'response\.status\s*===\s*503.*restartServer', plugin_source, re.DOTALL)

    def test_store_exchange_retries_on_503(self, plugin_source):
        """storeExchange retries after 503."""
        # Both functions should have self-healing
        # Count occurrences of restartServer() calls
        restart_calls = plugin_source.count("await restartServer()")
        assert restart_calls >= 4  # 2 in getContext (503 + catch) + 2 in storeExchange (503 + catch)

    def test_detached_server_spawn(self, plugin_source):
        """Server is spawned detached so it outlives the plugin."""
        assert "detached: true" in plugin_source


# ============================================================================
# Caching Architecture
# ============================================================================

class TestCachingArchitecture:
    """Verify the two-phase caching architecture for split delivery."""

    def test_cached_context_map_exists(self, plugin_source):
        """cachedContext Map is defined for caching between hooks."""
        assert "cachedContext" in plugin_source
        assert "new Map" in plugin_source

    def test_chat_message_sets_cache(self, plugin_source):
        """chat.message hook caches context for system.transform."""
        assert "cachedContext.set(" in plugin_source

    def test_system_transform_reads_cache(self, plugin_source):
        """system.transform reads from cachedContext."""
        assert "cachedContext.get(" in plugin_source

    def test_session_idle_clears_cache(self, plugin_source):
        """session.idle clears cachedContext after exchange complete."""
        assert "cachedContext.delete(" in plugin_source


# ============================================================================
# Split Delivery (v0.3.2)
# ============================================================================

class TestSplitDelivery:
    """Verify scoring prompt and context injection architecture."""

    def test_scoring_prompt_push(self, plugin_source):
        """Scoring prompt injected into system prompt via push."""
        assert "output.system.push(" in plugin_source

    def test_context_push(self, plugin_source):
        """Memory context injected at END of system prompt via push."""
        assert "output.system.push(" in plugin_source

    def test_does_not_modify_output_parts(self, plugin_source):
        """Neither hook modifies output.parts (would be visible in UI)."""
        # chat.message should NOT write to output.parts
        # Check that there's no output.parts assignment (only read via extractTextFromParts)
        # The extractTextFromParts reads parts but doesn't modify them
        parts_writes = re.findall(r'output\.parts\[.*\]\s*=', plugin_source)
        assert len(parts_writes) == 0, f"Found output.parts writes: {parts_writes}"

    def test_fetches_split_fields(self, plugin_source):
        """Server response includes scoring_prompt and context_only fields."""
        assert "scoring_prompt" in plugin_source
        assert "context_only" in plugin_source


# ============================================================================
# Port Configuration
# ============================================================================

class TestPortConfiguration:
    """Verify port configuration matches Python server constants."""

    def test_prod_port_27182(self, plugin_source):
        """Plugin uses port 27182 for production."""
        assert "27182" in plugin_source

    def test_dev_port_27183(self, plugin_source):
        """Plugin uses port 27183 for development."""
        assert "27183" in plugin_source

    def test_dev_mode_env_var(self, plugin_source):
        """Plugin reads ROAMPAL_DEV env var."""
        assert "ROAMPAL_DEV" in plugin_source


# ============================================================================
# Session State Management
# ============================================================================

class TestSessionStateManagement:
    """Verify session state maps are properly managed."""

    def test_session_state_maps_defined(self, plugin_source):
        """All required state maps are defined."""
        assert "sessionContextMap" in plugin_source
        assert "lastUserMessage" in plugin_source
        assert "assistantMessageIds" in plugin_source
        assert "assistantTextParts" in plugin_source
        assert "cachedContext" in plugin_source

    def test_session_deleted_cleans_all_state(self, plugin_source):
        """session.deleted cleans up all 5 state maps."""
        # Find the session.deleted case block
        deleted_match = re.search(
            r'case\s*"session\.deleted".*?break\s*\}',
            plugin_source,
            re.DOTALL
        )
        assert deleted_match, "session.deleted case not found"
        deleted_block = deleted_match.group(0)

        assert "sessionContextMap.delete" in deleted_block
        assert "lastUserMessage.delete" in deleted_block
        assert "assistantMessageIds.delete" in deleted_block
        assert "assistantTextParts.delete" in deleted_block
        assert "cachedContext.delete" in deleted_block

    def test_session_idle_cleans_assistant_state(self, plugin_source):
        """session.idle clears assistant tracking for next exchange."""
        idle_match = re.search(
            r'case\s*"session\.idle".*?break\s*\}',
            plugin_source,
            re.DOTALL
        )
        assert idle_match, "session.idle case not found"
        idle_block = idle_match.group(0)

        assert "assistantMessageIds.delete" in idle_block
        assert "assistantTextParts.delete" in idle_block

    def test_chat_message_clears_previous_exchange(self, plugin_source):
        """chat.message clears assistant tracking from previous exchange."""
        # The chat.message handler should clear old assistant state
        assert "assistantMessageIds.delete(sessionId)" in plugin_source
        assert "assistantTextParts.delete(sessionId)" in plugin_source


# ============================================================================
# Exchange Capture Flow
# ============================================================================

class TestExchangeCapture:
    """Verify the exchange capture flow: user → assistant → idle → store."""

    def test_stores_user_text(self, plugin_source):
        """chat.message stores user text in lastUserMessage."""
        assert "lastUserMessage.set(" in plugin_source

    def test_tracks_assistant_message_ids(self, plugin_source):
        """message.updated tracks assistant message IDs."""
        assert "assistantMessageIds" in plugin_source

    def test_accumulates_text_parts(self, plugin_source):
        """message.part.updated accumulates text from TextPart.text."""
        assert "assistantTextParts" in plugin_source
        # Should use part.text (not msg.content)
        assert "part.text" in plugin_source

    def test_session_idle_calls_store_exchange(self, plugin_source):
        """session.idle assembles response and calls storeExchange."""
        assert "storeExchange(" in plugin_source
        # Should join text parts
        assert "Array.from(textParts.values()).join" in plugin_source

    def test_sends_to_stop_endpoint(self, plugin_source):
        """storeExchange sends to /api/hooks/stop endpoint."""
        assert "/api/hooks/stop" in plugin_source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

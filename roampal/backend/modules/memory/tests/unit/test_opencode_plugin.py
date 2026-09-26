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

    def test_get_context_503_routes_through_health_gated_restart(self, plugin_source):
        """v0.6.0 review fix 2: a response 503 is a BROKEN server (dead
        embed service / failed profile init — never busy). The 503 branch
        routes through restartServer(), which health-gates: a healthy
        server is untouched (Task 23 F2), a degraded one is replaced; then
        the request is retried once."""
        check_block = re.search(
            r"if \(!response\.ok\) \{.*?\n    \}", plugin_source, re.DOTALL
        )
        assert check_block is not None
        block = check_block.group(0)
        assert "response.status === 503" in block
        assert "restartServer()" in block, (
            "503 branch must route through the health-gated restart "
            "(fix 2: 503 = broken, and only the health gate decides)"
        )
        assert "setTimeout" in block  # brief backoff before the health gate

    def test_get_context_restarts_only_on_connection_failure(self, plugin_source):
        """Connection DOWN -> the single restart path (health-first guard
        inside restartServer makes it a no-op for an answering server)."""
        # The catch (connection failure) branch still restarts.
        catch_block = re.search(
            r"\} catch \(error\) \{.*?restartServer", plugin_source, re.DOTALL
        )
        assert catch_block is not None

    def test_get_context_404_surfaces_server_detail(self, plugin_source):
        """v0.6.0 review fix 8: a profile-404 surfaces the server's
        actionable detail (it names the exact fix command) instead of a
        bare status code — and triggers no restart (404 is deterministic)."""
        check_block = re.search(
            r"if \(!response\.ok\) \{.*?\n    \}", plugin_source, re.DOTALL
        )
        assert check_block is not None
        block = check_block.group(0)
        assert "response.status === 404" in block
        assert "err?.detail" in block
        assert "restartServer" not in block.split("response.status === 404")[1].split("console.error")[0]

    def test_restart_repasses_launch_pin(self, plugin_source):
        """v0.6.0 review fix 5: the plugin re-passes the launch pin on
        respawn (per-port pin file matching profile_manager's write)."""
        fn = re.search(
            r"async function restartServer\(\).*?\n\}", plugin_source, re.DOTALL
        )
        assert fn is not None
        body = fn.group(0)
        assert "server_pin_${port}.txt" in body, "pin file path mismatch"
        assert 'args.push("--profile", pin)' in body

    def test_detached_server_spawn(self, plugin_source):
        """Server is spawned detached so it outlives the plugin."""
        assert "detached: true" in plugin_source

    def test_restart_health_first_never_kills_answering_server(self, plugin_source):
        """Task 23, amended by fix 2: restartServer probes /api/health
        BEFORE any kill — a 200 server returns immediately; any non-503
        HTTP answer is left alone (foreign port holder); only 503 or
        connection failure reaches the kill path."""
        fn = re.search(
            r"async function restartServer\(\).*?\n\}", plugin_source, re.DOTALL
        )
        assert fn is not None
        body = fn.group(0)
        health_guard_pos = body.find('const probe = await fetch(healthUrl')
        netstat_pos = body.find('execSync("netstat -ano"')
        assert health_guard_pos != -1, "health-first guard missing"
        assert netstat_pos != -1
        assert health_guard_pos < netstat_pos, "restart must health-check before any kill"
        assert "if (probe.status !== 503) return true" in body, (
            "fix 2: non-503 HTTP answers must never trigger a kill"
        )

    def test_restart_single_flight_lock(self, plugin_source):
        """Task 23: cross-process lock file (proc_lock contract) around the
        restart — exclusive create + staleness recovery + guaranteed release."""
        fn = re.search(
            r"async function restartServer\(\).*?\n\}", plugin_source, re.DOTALL
        )
        assert fn is not None
        body = fn.group(0)
        assert re.search(
            r"server_restart_\$\{port\}\.lock", body
        ), "lock file missing"
        assert re.search(r"\{ flag: \"wx\" \}", body), "exclusive-create missing"
        # v0.6.0 review fix 6: staleness bound must exceed the restarter's
        # worst-case hold (~26s) — 45s matches the Python token TTL.
        assert "lockStaleMs = 45000" in body, "stale-lock TTL below worst-case hold"
        # fix 6: ownership-checked release — the unlock reads the lock and
        # compares to OUR pid before unlinking.
        assert re.search(
            r"readFileSync\(lockPath[\s\S]*?===\s*String\(process\.pid\)[\s\S]*?unlinkSync\(lockPath\)",
            body,
        ), "release must verify ownership before unlinking"

    def test_lock_path_matches_python_config_dir(self, plugin_source):
        """Task 23 double-check: the plugin's lock base must equal
        profile_manager._config_dir() on every platform, or the
        single-flight silently splits across seams (Windows/macOS
        discrepancy found and fixed during review)."""
        # win32: APPDATA\Roampal
        assert re.search(
            r'\s*join\(process\.env\.APPDATA.*"Roampal"\)', plugin_source
        )
        # darwin: ~/Library/Application Support/Roampal
        assert re.search(
            r'"Library", "Application Support", "Roampal"', plugin_source
        )
        # linux: <xdg-config>/roampal (lowercase, matching profile_manager)
        assert re.search(
            r'join\(process\.env\.XDG_CONFIG_HOME.*"roampal"\)', plugin_source, re.DOTALL
        )


# ============================================================================
# Caching Architecture
# ============================================================================

class TestCwdHeaderBinding:
    """Round 2 Item 6 / Task 19: the plugin honors bindings by sending the
    project directory when its own resolution is empty; the server resolves
    that binding. Plugin priorities stay intact (structural validation)."""

    def test_cached_worktree_tracked(self, plugin_source):
        """refreshProfile stores the resolved worktree for the cwd header."""
        assert "let _cachedWorktree" in plugin_source
        assert "_cachedWorktree = worktree" in plugin_source

    def test_cwd_header_sent_when_resolution_empty(self, plugin_source):
        """roampalHeaders sends X-Roampal-Cwd instead of a bare request.
        v0.6.0 review fix 4: percent-encoded — header values cannot carry
        characters above Latin-1, and Node's fetch throws on raw
        Cyrillic/CJK project paths (non-Latin Windows usernames)."""
        assert 'h["X-Roampal-Cwd"] = encodeURIComponent(_cachedWorktree || process.cwd())' in plugin_source
        assert 'h["X-Roampal-Cwd"] = _cachedWorktree' not in plugin_source

    def test_restart_creates_lock_folder_first(self, plugin_source):
        """v0.6.0 review fix 7: the config dir is mkdir'd BEFORE the
        exclusive-create lock attempt — on a fresh machine the dir does
        not exist yet (on Linux it is a different root than the data dir),
        and every wx write would fail with ENOENT, so the plugin could
        never restart the server. (The Python restarter mkdirs inside
        proc_lock.acquire.)"""
        fn = re.search(
            r"async function restartServer\(\).*?\n\}", plugin_source, re.DOTALL
        )
        assert fn is not None
        body = fn.group(0)
        mkdir_pos = body.find("mkdirSync(configBase")
        lock_pos = body.find('{ flag: "wx" }')
        assert mkdir_pos != -1, "config-dir mkdir missing"
        assert lock_pos != -1
        assert mkdir_pos < lock_pos, "must create the config dir before the lock attempt"

    def test_profile_header_still_wins(self, plugin_source):
        """env / per-project / user-global resolution keeps X-Roampal-Profile."""
        assert 'h["X-Roampal-Profile"] = _cachedProfile' in plugin_source

    def test_resolution_priorities_unchanged(self, plugin_source):
        """Priority 1/2/3 unchanged: env var > project opencode.json > user-global."""
        assert "const envProfile = process.env.ROAMPAL_PROFILE" in plugin_source
        assert 'config?.mcp?.["roampal-core"]?.environment?.ROAMPAL_PROFILE' in plugin_source

    def test_binding_walk_not_duplicated_in_ts(self, plugin_source):
        """No local binding-walk re-implementation — the server owns it."""
        assert "binding_for_cwd" not in plugin_source


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
        # The cleanup is inside the debounced setTimeout callback, not the top-level case.
        # Use a broader regex that captures the full session.idle block including the callback.
        idle_match = re.search(
            r'case\s*"session\.idle".*?case\s*"session\.compacted"',
            plugin_source,
            re.DOTALL
        )
        assert idle_match, "session.idle block not found"
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

    def test_session_idle_captures_exchange(self, plugin_source):
        """session.idle assembles response text (v0.4.8: storeExchange removed, sidecar stores)."""
        # Should join text parts for sidecar scoring
        assert "Array.from(textParts.values()).join" in plugin_source

    def test_sends_to_stop_endpoint(self, plugin_source):
        """storeExchange sends to /api/hooks/stop endpoint."""
        assert "/api/hooks/stop" in plugin_source


# ============================================================================
# v0.5.0: Subagent Filtering (Issue #4)
# ============================================================================

class TestSubagentFiltering:
    """Verify subagent filtering implementation."""

    def test_subagent_sessions_set_defined(self, plugin_source):
        """subagentSessions Set is defined for tracking subagent sessions."""
        assert "const subagentSessions = new Set<string>()" in plugin_source

    def test_primary_sessions_set_defined(self, plugin_source):
        """primarySessions Set tracks sessions where chat.message fired."""
        assert "const primarySessions = new Set<string>()" in plugin_source

    def test_cached_agent_modes_defined(self, plugin_source):
        """cachedAgentModes Map caches agent name-to-mode mapping."""
        assert "let cachedAgentModes = new Map<string, string>()" in plugin_source

    def test_is_subagent_helper_exists(self, plugin_source):
        """isSubagent() helper function is defined."""
        assert "function isSubagent(agentName?: string): boolean" in plugin_source

    def test_is_subagent_checks_mode_first(self, plugin_source):
        """isSubagent() checks cachedAgentModes before name heuristic."""
        # Mode check should come before the toLowerCase() heuristic
        mode_pos = plugin_source.find('cachedAgentModes.get(agentName)')
        heuristic_pos = plugin_source.find('agentName.toLowerCase()')
        assert mode_pos > 0 and heuristic_pos > 0
        assert mode_pos < heuristic_pos, "Mode check should come before name heuristic"

    def test_allow_subagents_config(self, plugin_source):
        """ROAMPAL_ALLOW_SUBAGENTS env var is read from config."""
        assert "ROAMPAL_ALLOW_SUBAGENTS" in plugin_source
        assert "ALLOW_SUBAGENTS" in plugin_source

    def test_chat_message_marks_primary_session(self, plugin_source):
        """chat.message adds session to primarySessions."""
        assert "primarySessions.add(sessionId)" in plugin_source

    def test_session_idle_checks_primary(self, plugin_source):
        """session.idle skips sessions not in primarySessions."""
        assert "primarySessions.has(sid)" in plugin_source

    def test_session_idle_subagent_guard(self, plugin_source):
        """session.idle has explicit subagentSessions check."""
        assert "subagentSessions.has(sid)" in plugin_source

    def test_system_transform_subagent_guard(self, plugin_source):
        """system.transform skips subagent sessions."""
        assert "subagentSessions.has(sessionId)" in plugin_source

    def test_session_deleted_cleans_subagent_state(self, plugin_source):
        """session.deleted cleans up subagent and primary tracking."""
        deleted_match = re.search(
            r'case\s*"session\.deleted".*?break\s*\}',
            plugin_source,
            re.DOTALL
        )
        assert deleted_match, "session.deleted case not found"
        deleted_block = deleted_match.group(0)
        assert "subagentSessions.delete" in deleted_block
        assert "primarySessions.delete" in deleted_block

    def test_agent_api_has_timeout(self, plugin_source):
        """client.app.agents() call is wrapped in Promise.race timeout."""
        assert "Promise.race" in plugin_source

    def test_agent_api_optional_chaining(self, plugin_source):
        """client.app.agents() uses optional chaining for safety."""
        assert "client as any).app?.agents?.()" in plugin_source


# ============================================================================
# v0.5.0: Sidecar Request Body (think:false removal)
# ============================================================================

class TestSidecarRequestBody:
    """Verify think:false was removed from sidecar requests."""

    def test_no_think_false_in_request(self, plugin_source):
        """Request body does not contain think: false field."""
        # Should not have think: false as an actual field (comment references OK)
        # Look for think: false NOT preceded by // (not in a comment)
        think_matches = re.findall(r'^\s+think:\s*false', plugin_source, re.MULTILINE)
        assert len(think_matches) == 0, f"Found think: false in request body: {think_matches}"

    def test_no_think_prefix_still_present(self, plugin_source):
        """/no_think text prefix is still used in messages (safe approach)."""
        assert "/no_think" in plugin_source


# ============================================================================
# v0.5.0: Consecutive Failure Counter
# ============================================================================

class TestConsecutiveFailureCounter:
    """Verify scoringBroken boolean was replaced with failure counter."""

    def test_consecutive_failures_defined(self, plugin_source):
        """consecutiveFailures counter is defined."""
        assert "let consecutiveFailures" in plugin_source

    def test_no_scoring_broken_boolean(self, plugin_source):
        """scoringBroken boolean is no longer used (only in comments)."""
        # Should not have `scoringBroken = ` assignments (only in comments)
        assignments = re.findall(r'^\s+scoringBroken\s*=', plugin_source, re.MULTILINE)
        assert len(assignments) == 0, f"Found scoringBroken assignments: {assignments}"

    def test_counter_resets_on_success(self, plugin_source):
        """consecutiveFailures resets to 0 on scoring success."""
        assert "consecutiveFailures = 0" in plugin_source

    def test_counter_increments_on_failure(self, plugin_source):
        """consecutiveFailures increments on scoring failure."""
        assert "consecutiveFailures++" in plugin_source

    def test_status_tag_uses_threshold(self, plugin_source):
        """Status tag checks consecutiveFailures >= 2 for persistent failure."""
        assert "consecutiveFailures >= 2" in plugin_source

    def test_status_tag_shows_failure_count(self, plugin_source):
        """Status tag includes actual failure count when broken."""
        assert "consecutiveFailures}" in plugin_source or "consecutiveFailures} consecutive" in plugin_source


# ============================================================================
# v0.5.0: Sidecar Input/Output Caps
# ============================================================================

class TestSidecarCaps:
    """Verify sidecar input and output caps."""

    def test_facts_input_cap_8k(self, plugin_source):
        """Facts extraction uses 8K char cap for user and assistant."""
        assert "exchange.user.slice(0, 8000)" in plugin_source
        assert "exchange.assistant.slice(0, 8000)" in plugin_source

    def test_summary_output_cap_300(self, plugin_source):
        """Summary prompt instructs 300 char limit."""
        assert "300 chars" in plugin_source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

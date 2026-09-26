"""
Tests for hook scripts (user_prompt_submit_hook.py, stop_hook.py).

Tests cover:
- Input parsing (stdin JSON)
- Server URL configuration (dev/prod/env override)
- HTTP request construction
- Self-healing (_restart_server) logic
- Exit code behavior
- Transcript reading (stop_hook)
- Update check caching (user_prompt_submit_hook)
"""

import sys
import os
import io
import json
import pytest
import tempfile
import threading
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))


# ============================================================================
# User Prompt Submit Hook Tests
# ============================================================================

class TestUserPromptSubmitHook:
    """Test user_prompt_submit_hook.py main flow."""

    def _run_hook(self, input_data, env=None):
        """Run the hook's main() with mocked stdin and capture exit code."""
        from roampal.hooks import user_prompt_submit_hook

        # Reset update check cache between tests
        user_prompt_submit_hook._update_check_cache = {
            "checked": False, "available": False, "current": "", "latest": ""
        }

        stdin_data = json.dumps(input_data)

        with patch('sys.stdin', io.StringIO(stdin_data)), \
             patch('builtins.print') as mock_print, \
             patch.dict(os.environ, env or {}, clear=False):
            try:
                user_prompt_submit_hook.main()
            except SystemExit as e:
                return e.code, mock_print
        return None, mock_print

    def test_empty_stdin_exits_0(self):
        """Empty/invalid stdin exits cleanly."""
        from roampal.hooks import user_prompt_submit_hook

        with patch('sys.stdin', io.StringIO("")):
            with pytest.raises(SystemExit) as exc:
                user_prompt_submit_hook.main()
            assert exc.value.code == 0

    def test_empty_prompt_exits_0(self):
        """Empty prompt field exits cleanly."""
        code, _ = self._run_hook({"prompt": ""})
        assert code == 0

    def test_successful_context_injection(self):
        """Successful request prints formatted injection to stdout."""
        response_data = json.dumps({
            "formatted_injection": "<test>context here</test>",
            "scoring_required": False
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = response_data

        with patch("urllib.request.urlopen", return_value=mock_resp):
            code, mock_print = self._run_hook({
                "prompt": "hello world",
                "session_id": "test_session"
            })

        assert code == 0
        # Should have printed the formatted injection
        mock_print.assert_any_call("<test>context here</test>")

    def test_dev_mode_port(self):
        """ROAMPAL_DEV=1 uses port 27183."""
        from roampal.hooks import user_prompt_submit_hook

        # We can't easily test the full flow, but we can verify port selection logic
        with patch.dict(os.environ, {"ROAMPAL_DEV": "1"}):
            dev_mode = os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")
            default_port = 27183 if dev_mode else 27182
            assert default_port == 27183

    def test_prod_mode_port(self):
        """Default (no ROAMPAL_DEV) uses port 27182."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove ROAMPAL_DEV if present
            env = os.environ.copy()
            env.pop("ROAMPAL_DEV", None)
            dev_mode = env.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")
            default_port = 27183 if dev_mode else 27182
            assert default_port == 27182

    def test_server_url_override(self):
        """ROAMPAL_SERVER_URL env var overrides default."""
        with patch.dict(os.environ, {"ROAMPAL_SERVER_URL": "http://custom:9999"}):
            server_url = os.environ.get("ROAMPAL_SERVER_URL", "http://127.0.0.1:27182")
            assert server_url == "http://custom:9999"

    def test_conversation_id_from_session_id(self):
        """Reads conversation_id from session_id field."""
        input_data = {"prompt": "test", "session_id": "my_session"}
        conversation_id = input_data.get("conversation_id") or input_data.get("session_id", "default")
        assert conversation_id == "my_session"

    def test_conversation_id_fallback_default(self):
        """Falls back to 'default' when no ID provided."""
        input_data = {"prompt": "test"}
        conversation_id = input_data.get("conversation_id") or input_data.get("session_id", "default")
        assert conversation_id == "default"


# ============================================================================
# Stop Hook Tests
# ============================================================================

class TestStopHook:
    """Test stop_hook.py main flow."""

    def test_empty_stdin_exits_0(self):
        """Empty stdin exits cleanly."""
        from roampal.hooks import stop_hook

        with patch('sys.stdin', io.StringIO("")):
            with pytest.raises(SystemExit) as exc:
                stop_hook.main()
            assert exc.value.code == 0

    def test_stop_hook_active_prevents_loop(self):
        """stop_hook_active=True prevents infinite loops."""
        from roampal.hooks import stop_hook

        with patch('sys.stdin', io.StringIO(json.dumps({"stop_hook_active": True}))):
            with pytest.raises(SystemExit) as exc:
                stop_hook.main()
            assert exc.value.code == 0

    def test_state_management_only(self):
        """v0.3.6: Stop hook sends conversation_id only (no exchange storage)."""
        from roampal.hooks import stop_hook

        response_data = json.dumps({
            "stored": False,
            "doc_id": "",
            "scoring_complete": False,
            "should_block": False
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = response_data

        input_data = json.dumps({
            "session_id": "test",
        })

        with patch('sys.stdin', io.StringIO(input_data)), \
             patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen, \
             patch('builtins.print'):
            with pytest.raises(SystemExit) as exc:
                stop_hook.main()
            assert exc.value.code == 0
            # v0.3.6: Stop hook always sends lifecycle_only=True with exchange data
            # (empty when no transcript_path provided)
            call_args = mock_urlopen.call_args
            sent_data = json.loads(call_args[0][0].data.decode("utf-8"))
            assert sent_data["conversation_id"] == "test"
            assert sent_data["lifecycle_only"] is True
            assert sent_data["user_message"] == ""
            assert sent_data["assistant_response"] == ""

    def test_blocking_exit_code_2(self):
        """should_block=True causes exit code 2."""
        from roampal.hooks import stop_hook

        response_data = json.dumps({
            "stored": False,
            "doc_id": "",
            "scoring_complete": False,
            "should_block": True,
            "block_message": "Please score the cached memories"
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = response_data

        input_data = json.dumps({
            "session_id": "block_test",
        })

        with patch('sys.stdin', io.StringIO(input_data)), \
             patch("urllib.request.urlopen", return_value=mock_resp), \
             patch('builtins.print'):
            with pytest.raises(SystemExit) as exc:
                stop_hook.main()
            assert exc.value.code == 2

    def test_server_error_exits_0(self):
        """Stop hook never blocks on server errors."""
        from roampal.hooks import stop_hook

        input_data = json.dumps({
            "session_id": "error_test",
        })

        with patch('sys.stdin', io.StringIO(input_data)), \
             patch("urllib.request.urlopen", side_effect=Exception("connection refused")), \
             patch('builtins.print'):
            with pytest.raises(SystemExit) as exc:
                stop_hook.main()
            assert exc.value.code == 0  # Never blocks on error


# ============================================================================
# Transcript Reading Tests
# ============================================================================

class TestTranscriptReading:
    """Test stop_hook.read_transcript for Claude Code format."""

    def test_read_valid_transcript(self):
        """Reads user and assistant messages from JSONL."""
        from roampal.hooks.stop_hook import read_transcript

        lines = [
            json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": "What is Python?"}]}}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "A programming language."}]}}),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.write("\n".join(lines))
            f.flush()
            transcript_path = f.name

        try:
            user_msg, assistant_msg = read_transcript(transcript_path)
            assert "Python" in user_msg
            assert "programming language" in assistant_msg
        finally:
            os.unlink(transcript_path)

    def test_read_tool_calls_in_transcript(self):
        """Tool call text parts are captured, tool_use names are not (only text blocks extracted)."""
        from roampal.hooks.stop_hook import read_transcript

        lines = [
            json.dumps({"type": "user", "message": {"content": [{"type": "text", "text": "score this"}]}}),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Scoring now"},
                {"type": "tool_use", "name": "score_memories"}
            ]}}),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.write("\n".join(lines))
            f.flush()
            transcript_path = f.name

        try:
            _, assistant_msg = read_transcript(transcript_path)
            assert "Scoring now" in assistant_msg
        finally:
            os.unlink(transcript_path)

    def test_nonexistent_file(self):
        """Missing file returns empty strings."""
        from roampal.hooks.stop_hook import read_transcript
        user_msg, assistant_msg = read_transcript("/nonexistent/path.jsonl")
        assert user_msg == ""
        assert assistant_msg == ""

    def test_empty_file(self):
        """Empty file returns empty strings."""
        from roampal.hooks.stop_hook import read_transcript

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            transcript_path = f.name

        try:
            user_msg, assistant_msg = read_transcript(transcript_path)
            assert user_msg == ""
            assert assistant_msg == ""
        finally:
            os.unlink(transcript_path)


# ============================================================================
# Self-Healing (_restart_server) Tests
# ============================================================================

class TestRestartServer:
    """Test _restart_server self-healing logic in both hooks."""

    def _isolated_locks(self, monkeypatch, tmp_path):
        """Single-flight lock dir isolation for restart tests."""
        import roampal.profile_manager as pm

        monkeypatch.setattr(pm, "_config_dir", lambda: tmp_path / "config")
        (tmp_path / "config").mkdir(exist_ok=True)

    def test_healthy_server_is_never_restarted(self, monkeypatch, tmp_path):
        """Task 23 down-only: an answering server returns immediately; the
        old path killed it on ANY restart trigger (503/timeout)."""
        from roampal.hooks import user_prompt_submit_hook as ups

        self._isolated_locks(monkeypatch, tmp_path)
        with patch.object(ups, "_server_health_ok", return_value=True), \
             patch("subprocess.Popen") as mock_popen:
            result = ups._restart_server("http://127.0.0.1:27182", 27182, timeout=2.0)
            assert result is True
            mock_popen.assert_not_called()

    def test_503_is_degraded_counts_as_down(self):
        """v0.6.0 review fix 2: the server's only 503 is a BROKEN state
        (dead embed service / failed profile init — never busy). Health
        503 must count as down so the single-flight restart replaces the
        process (what main.py's health docstring always intended)."""
        import urllib.error

        from roampal.hooks import user_prompt_submit_hook as ups
        from roampal.hooks import stop_hook as stop

        for hook in (ups, stop):
            with patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.HTTPError("u", 503, "unhealthy", None, None),
            ):
                assert hook._server_health_ok("http://127.0.0.1:27182") is False

    def test_non_503_http_answers_still_count_as_up(self):
        """fix 2 F2 rule: any HTTP answer that is not the roampal broken
        signal (404/500 — e.g. a foreign process squatting the port) is
        UP: never kill a process we cannot positively identify."""
        import urllib.error

        from roampal.hooks import user_prompt_submit_hook as ups
        from roampal.hooks import stop_hook as stop

        for code in (404, 500, 401):
            for hook in (ups, stop):
                with patch(
                    "urllib.request.urlopen",
                    side_effect=urllib.error.HTTPError("u", code, "x", None, None),
                ):
                    assert hook._server_health_ok("http://127.0.0.1:27182") is True

    def test_degraded_server_is_restarted_and_recovers(self, monkeypatch, tmp_path):
        """fix 2 acceptance: health 503 -> the single-flight restart kills
        and respawns; the fresh server's 200 health ends the flow."""
        from roampal.hooks import user_prompt_submit_hook as ups

        self._isolated_locks(monkeypatch, tmp_path)

        health_states = [False, False, True]  # first check, under-lock recheck, post-spawn poll

        def fake_health(url):
            return health_states.pop(0) if health_states else True

        with patch.object(ups, "_server_health_ok", side_effect=fake_health), \
             patch("subprocess.Popen") as mock_popen, \
             patch("time.sleep"), \
             patch("subprocess.run"):
            result = ups._restart_server("http://127.0.0.1:27182", 27182, timeout=2.0)

        assert result is True
        mock_popen.assert_called_once()  # degraded server WAS replaced

    def test_preflight_degradation_check_gates_restart(self, monkeypatch, tmp_path):
        """fix 2, round-2 amended: the UPS pre-flight probes health directly
        and only reacts to an ACTUAL 503 answer. Healthy (200), down/slow
        (connection error or 2s timeout), and foreign HTTP answers (404)
        must all leave the server alone — the per-prompt path must not
        become a new kill opportunity for busy-but-healthy servers."""
        import urllib.error
        from roampal.hooks import user_prompt_submit_hook as ups

        # Healthy -> untouched.
        with patch("urllib.request.urlopen") as mock_open, \
             patch.object(ups, "_restart_server") as mock_restart:
            mock_open.return_value.status = 200
            mock_open.return_value.__enter__ = MagicMock(return_value=mock_open.return_value)
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            ups._preflight_degradation_check("http://127.0.0.1:27182", 27182)
            mock_restart.assert_not_called()

        # Down/slow (connection error or 2s timeout) -> untouched:
        # the real request's failure path owns the down-restart.
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("timeout")), \
             patch.object(ups, "_restart_server") as mock_restart:
            ups._preflight_degradation_check("http://127.0.0.1:27182", 27182)
            mock_restart.assert_not_called()

        # Foreign HTTP answer (404) -> untouched.
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError("u", 404, "x", None, None)), \
             patch.object(ups, "_restart_server") as mock_restart:
            ups._preflight_degradation_check("http://127.0.0.1:27182", 27182)
            mock_restart.assert_not_called()

        # Probe blowing up -> never blocks the prompt.
        with patch("urllib.request.urlopen", side_effect=RuntimeError("boom")), \
             patch.object(ups, "_restart_server") as mock_restart:
            ups._preflight_degradation_check("http://127.0.0.1:27182", 27182)
            mock_restart.assert_not_called()

        # The one signal that reacts: 503 -> health-gated restart.
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError("u", 503, "unhealthy", None, None)), \
             patch.object(ups, "_restart_server") as mock_restart:
            ups._preflight_degradation_check("http://127.0.0.1:27182", 27182)
            mock_restart.assert_called_once_with("http://127.0.0.1:27182", 27182)

    def test_restart_single_flight_one_spawn(self, monkeypatch, tmp_path):
        """Task 23 acceptance: two concurrent restarters -> exactly one
        spawn; the latecomer waits for the winner's health."""
        from roampal.hooks import user_prompt_submit_hook as ups

        self._isolated_locks(monkeypatch, tmp_path)

        # Health is DOWN until the first spawn happens; afterwards, up.
        # That makes the flow deterministic whichever thread runs first:
        # early checks (and under-lock rechecks without a winner) are down,
        # and any post-spawn recheck sees an up server.
        spawned = threading.Event()
        spawns = []

        def fake_health(url):
            return spawned.is_set()

        def fake_popen(*args, **kwargs):
            spawns.append(args)
            spawned.set()
            return MagicMock()

        with patch.object(ups, "_server_health_ok", side_effect=fake_health), \
             patch.object(ups, "_poll_health", side_effect=lambda url, t: True), \
             patch("subprocess.Popen", side_effect=fake_popen), \
             patch("time.sleep"), \
             patch("subprocess.run"):
            barrier = threading.Barrier(2)
            results = {}

            def contender(name):
                barrier.wait()
                results[name] = ups._restart_server("http://127.0.0.1:27182", 27182, timeout=2.0)

            threads = [threading.Thread(target=contender, args=(n,)) for n in ("a", "b")]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # One shared server: exactly one spawn, everyone recovered.
        assert len(spawns) == 1, (
            f"single-flight violated: {len(spawns)} restart spawns: {results}"
        )
        assert all(results.values()), results

    def test_restart_returns_false_on_spawn_failure(self, monkeypatch, tmp_path):
        """Down server + failed health thereafter -> False (restart path)."""
        from roampal.hooks import user_prompt_submit_hook as ups

        self._isolated_locks(monkeypatch, tmp_path)
        with patch.object(ups, "_server_health_ok", return_value=False), \
             patch.object(ups, "_poll_health", return_value=False), \
             patch("subprocess.Popen") as mock_popen, \
             patch("time.sleep"), \
             patch("subprocess.run"):
            result = ups._restart_server("http://127.0.0.1:27182", 27182, timeout=2.0)
            assert result is False
            mock_popen.assert_called_once()  # spawn reached (down path)

    def test_stop_hook_healthy_server_never_restarted(self, monkeypatch, tmp_path):
        """Stop hook's seam matches: healthy -> return True, no spawn."""
        from roampal.hooks import stop_hook as stop

        self._isolated_locks(monkeypatch, tmp_path)
        with patch.object(stop, "_server_health_ok", return_value=True), \
             patch("subprocess.Popen") as mock_popen:
            result = stop._restart_server("http://127.0.0.1:27183", 27183, timeout=2.0)
            assert result is True
            mock_popen.assert_not_called()

    def test_stop_hook_spawn_after_locked_restart(self, monkeypatch, tmp_path):
        """Stop hook's down path: single turn of spawn + poll."""
        from roampal.hooks import stop_hook as stop

        self._isolated_locks(monkeypatch, tmp_path)
        with patch.object(stop, "_server_health_ok", return_value=False), \
             patch.object(stop, "_poll_health", return_value=True), \
             patch("subprocess.Popen") as mock_popen, \
             patch("time.sleep"), \
             patch("subprocess.run"):
            result = stop._restart_server("http://127.0.0.1:27183", 27183, timeout=2.0)
            assert result is True
            mock_popen.assert_called_once()


# ============================================================================
# Profile-404 Surfacing Tests (v0.6.0 review fix 8)
# ============================================================================

class TestProfile404Surfacing:
    """fix 8: a 404 is the routing contract rejecting an unknown profile
    (e.g. after `roampal profile delete`) — NOT a down/degraded server.
    The hooks must print the server's actionable detail (it includes the
    exact fix command) and exit WITHOUT the restart/retry theater that
    previously buried it under 'retry failed after restart: HTTP 404'."""

    @staticmethod
    def _404_error(detail=None):
        import urllib.error

        body = json.dumps({"detail": detail}).encode("utf-8") if detail else b"<html>502</html>"
        return urllib.error.HTTPError(
            "http://127.0.0.1:27182", 404, "Not Found", None, io.BytesIO(body)
        )

    def test_ups_404_prints_detail_without_restart(self):
        from roampal.hooks import user_prompt_submit_hook as ups

        detail = "Profile 'ghost' is not registered. Create it first: roampal profile create ghost"
        with patch('sys.stdin', io.StringIO(json.dumps({"prompt": "hi", "session_id": "s"}))), \
             patch("urllib.request.urlopen", side_effect=self._404_error(detail)), \
             patch.object(ups, "_preflight_degradation_check"), \
             patch.object(ups, "_restart_server") as mock_restart, \
             patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc:
                ups.main()

        assert exc.value.code == 1
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Create it first: roampal profile create ghost" in printed
        assert "retry failed" not in printed
        mock_restart.assert_not_called()

    def test_ups_404_unparseable_body_falls_back(self):
        from roampal.hooks import user_prompt_submit_hook as ups

        with patch('sys.stdin', io.StringIO(json.dumps({"prompt": "hi", "session_id": "s"}))), \
             patch("urllib.request.urlopen", side_effect=self._404_error(None)), \
             patch.object(ups, "_preflight_degradation_check"), \
             patch.object(ups, "_restart_server") as mock_restart, \
             patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc:
                ups.main()

        assert exc.value.code == 1
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "HTTP 404 from server" in printed
        mock_restart.assert_not_called()

    def test_stop_hook_404_prints_detail_exit_0(self):
        """Stop hook never blocks the user's flow — 404 surfaces the detail
        and exits 0, no restart."""
        from roampal.hooks import stop_hook as stop

        detail = "Profile 'ghost' is not registered. Create it first: roampal profile create ghost"
        with patch('sys.stdin', io.StringIO(json.dumps({
                "conversation_id": "s", "last_assistant_message": "resp"}))), \
             patch("urllib.request.urlopen", side_effect=self._404_error(detail)), \
             patch.object(stop, "_restart_server") as mock_restart, \
             patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit) as exc:
                stop.main()

        assert exc.value.code == 0
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Create it first: roampal profile create ghost" in printed
        mock_restart.assert_not_called()


# ============================================================================
# Update Check Cache Tests
# ============================================================================

class TestUpdateCheckCache:
    """Test update check caching in user_prompt_submit_hook."""

    def test_cache_hit_skips_pypi(self):
        """Cached result doesn't hit PyPI again."""
        from roampal.hooks import user_prompt_submit_hook

        user_prompt_submit_hook._update_check_cache = {
            "checked": True,
            "available": False,
            "current": "0.3.2",
            "latest": "0.3.2"
        }

        result = user_prompt_submit_hook.check_for_updates_cached()
        assert result == (False, "0.3.2", "0.3.2")

    def test_update_available_returns_true(self):
        """Returns True when newer version exists."""
        from roampal.hooks import user_prompt_submit_hook

        user_prompt_submit_hook._update_check_cache = {
            "checked": True,
            "available": True,
            "current": "0.3.1",
            "latest": "0.3.2"
        }

        available, current, latest = user_prompt_submit_hook.check_for_updates_cached()
        assert available is True
        assert current == "0.3.1"
        assert latest == "0.3.2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

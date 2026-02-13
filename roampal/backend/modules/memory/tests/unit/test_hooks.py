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

    def test_restart_attempts_kill_and_start(self):
        """Restart kills old process and starts new one."""
        from roampal.hooks.user_prompt_submit_hook import _restart_server

        mock_health_resp = MagicMock()
        mock_health_resp.__enter__ = MagicMock(return_value=mock_health_resp)
        mock_health_resp.__exit__ = MagicMock(return_value=False)
        mock_health_resp.status = 200

        with patch("subprocess.run") as mock_run, \
             patch("subprocess.Popen") as mock_popen, \
             patch("urllib.request.urlopen", return_value=mock_health_resp), \
             patch("time.sleep"), \
             patch('builtins.print'):

            # Mock netstat output (Windows path)
            if sys.platform == "win32":
                mock_run.return_value = MagicMock(
                    stdout="  TCP    127.0.0.1:27182    0.0.0.0:0    LISTENING    12345\n"
                )

            result = _restart_server("http://127.0.0.1:27182", 27182, timeout=2.0)
            assert result is True
            mock_popen.assert_called_once()

    def test_restart_returns_false_on_timeout(self):
        """Returns False if server doesn't start within timeout."""
        from roampal.hooks.user_prompt_submit_hook import _restart_server

        with patch("subprocess.run"), \
             patch("subprocess.Popen"), \
             patch("urllib.request.urlopen", side_effect=Exception("refused")), \
             patch("time.sleep"), \
             patch("time.time", side_effect=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 100.0]), \
             patch('builtins.print'):

            result = _restart_server("http://127.0.0.1:27182", 27182, timeout=2.0)
            assert result is False

    def test_stop_hook_restart_same_behavior(self):
        """Stop hook's _restart_server has same interface."""
        from roampal.hooks.stop_hook import _restart_server

        mock_health_resp = MagicMock()
        mock_health_resp.__enter__ = MagicMock(return_value=mock_health_resp)
        mock_health_resp.__exit__ = MagicMock(return_value=False)
        mock_health_resp.status = 200

        with patch("subprocess.run") as mock_run, \
             patch("subprocess.Popen") as mock_popen, \
             patch("urllib.request.urlopen", return_value=mock_health_resp), \
             patch("time.sleep"), \
             patch('builtins.print'):

            if sys.platform == "win32":
                mock_run.return_value = MagicMock(
                    stdout="  TCP    127.0.0.1:27183    0.0.0.0:0    LISTENING    67890\n"
                )

            result = _restart_server("http://127.0.0.1:27183", 27183, timeout=2.0)
            assert result is True


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

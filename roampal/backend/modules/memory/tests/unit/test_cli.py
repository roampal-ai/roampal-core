"""
Tests for CLI commands (cli.py).

Tests cover:
- configure_opencode(): MCP config, plugin install, idempotency, force
- configure_claude_code(): Settings, hooks, MCP, permissions
- configure_cursor(): MCP config, hooks config
- is_dev_mode() / get_port() helpers
- PYTHONPATH computation for OpenCode MCP
- cmd_init() auto-detection and explicit flags
"""

import sys
import os
import json
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')))


# ============================================================================
# Helper Tests
# ============================================================================

class TestIsDevMode:
    """Test is_dev_mode() single source of truth."""

    def test_default_is_prod(self):
        from roampal.cli import is_dev_mode
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("ROAMPAL_DEV", None)
            with patch.dict(os.environ, env, clear=True):
                assert is_dev_mode() is False

    def test_env_var_enables_dev(self):
        from roampal.cli import is_dev_mode
        with patch.dict(os.environ, {"ROAMPAL_DEV": "1"}):
            assert is_dev_mode() is True

    def test_args_flag_enables_dev(self):
        from roampal.cli import is_dev_mode
        args = MagicMock()
        args.dev = True
        assert is_dev_mode(args) is True

    def test_args_no_dev_uses_env(self):
        from roampal.cli import is_dev_mode
        args = MagicMock()
        args.dev = False
        with patch.dict(os.environ, {"ROAMPAL_DEV": "true"}):
            assert is_dev_mode(args) is True


class TestGetPort:
    """Test get_port() for dev/prod."""

    def test_prod_port(self):
        from roampal.cli import get_port
        args = MagicMock()
        args.dev = False
        with patch.dict(os.environ, {}, clear=False):
            env = os.environ.copy()
            env.pop("ROAMPAL_DEV", None)
            with patch.dict(os.environ, env, clear=True):
                assert get_port(args) == 27182

    def test_dev_port(self):
        from roampal.cli import get_port
        args = MagicMock()
        args.dev = True
        assert get_port(args) == 27183


class TestGetDataDir:
    """Test get_data_dir() for platform-specific paths."""

    def test_prod_data_dir(self):
        from roampal.cli import get_data_dir
        result = get_data_dir(dev=False)
        assert isinstance(result, Path)
        assert "Roampal" in str(result) or "roampal" in str(result)
        assert "DEV" not in str(result) and "dev" not in str(result).split(os.sep)[-2:]

    def test_dev_data_dir(self):
        from roampal.cli import get_data_dir
        result = get_data_dir(dev=True)
        assert isinstance(result, Path)
        # Dev dir should have DEV or dev suffix
        path_str = str(result)
        assert "DEV" in path_str or "dev" in path_str.lower()


# ============================================================================
# _build_hook_command() Tests (v0.3.2 - Bug 8 fix)
# ============================================================================

class TestBuildHookCommand:
    """Test _build_hook_command() centralized hook command builder.

    v0.3.2: This helper ensures all platforms handle ROAMPAL_DEV consistently.
    Previously Claude Code forgot to wrap commands while Cursor did.
    """

    def test_prod_mode_no_env_wrapper(self):
        """In prod mode (is_dev=False), command is plain without env wrapper."""
        from roampal.cli import _build_hook_command
        cmd = _build_hook_command("user_prompt_submit_hook", is_dev=False)
        assert "ROAMPAL_DEV" not in cmd
        assert "roampal.hooks.user_prompt_submit_hook" in cmd

    def test_dev_mode_has_env_wrapper(self):
        """In dev mode (is_dev=True), command includes ROAMPAL_DEV=1."""
        from roampal.cli import _build_hook_command
        cmd = _build_hook_command("user_prompt_submit_hook", is_dev=True)
        assert "ROAMPAL_DEV" in cmd
        assert "roampal.hooks.user_prompt_submit_hook" in cmd

    def test_dev_mode_windows_format(self):
        """On Windows dev mode, uses cmd /c 'set ROAMPAL_DEV=1 && ...' format."""
        from roampal.cli import _build_hook_command
        with patch("sys.platform", "win32"):
            # Re-import to get fresh module with patched platform
            import importlib
            import roampal.cli
            importlib.reload(roampal.cli)
            cmd = roampal.cli._build_hook_command("stop_hook", is_dev=True)
            # Should have Windows-style env set
            if sys.platform == "win32":
                assert "cmd /c" in cmd or "set ROAMPAL_DEV=1" in cmd

    def test_dev_mode_unix_format(self):
        """On Unix dev mode, uses 'ROAMPAL_DEV=1 python ...' format."""
        from roampal.cli import _build_hook_command
        # Unix format: ROAMPAL_DEV=1 at start of command
        cmd = _build_hook_command("stop_hook", is_dev=True)
        if sys.platform != "win32":
            assert cmd.startswith("ROAMPAL_DEV=1 ")

    def test_both_hooks_get_env(self):
        """Both user_prompt_submit_hook and stop_hook get env wrapper in dev mode."""
        from roampal.cli import _build_hook_command
        submit_cmd = _build_hook_command("user_prompt_submit_hook", is_dev=True)
        stop_cmd = _build_hook_command("stop_hook", is_dev=True)
        assert "ROAMPAL_DEV" in submit_cmd
        assert "ROAMPAL_DEV" in stop_cmd


# ============================================================================
# configure_opencode() Tests
# ============================================================================

class TestConfigureOpencode:
    """Test configure_opencode() function."""

    @pytest.fixture
    def opencode_env(self, tmp_path):
        """Set up a temporary OpenCode config environment."""
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)

        # Create a fake plugin source
        plugin_source_dir = tmp_path / "plugins" / "opencode"
        plugin_source_dir.mkdir(parents=True)
        plugin_source = plugin_source_dir / "roampal.ts"
        plugin_source.write_text("export const RoampalPlugin = {}")

        return config_dir, plugin_source

    def test_creates_mcp_config(self, tmp_path):
        """configure_opencode creates opencode.json with MCP config."""
        from roampal.cli import configure_opencode

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)

        # Create plugin source so the function can copy it
        import roampal.cli
        plugin_source = Path(roampal.cli.__file__).parent / "plugins" / "opencode" / "roampal.ts"

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("builtins.print"):
            # Patch the config_dir derivation for Windows
            if sys.platform == "win32":
                with patch.object(Path, "home", return_value=tmp_path):
                    configure_opencode(is_dev=False, force=True)
            else:
                configure_opencode(is_dev=False, force=True)

        config_file = config_dir / "opencode.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
            assert "mcp" in config
            assert "roampal-core" in config["mcp"]
            mcp_config = config["mcp"]["roampal-core"]
            assert mcp_config["type"] == "local"
            assert "-m" in mcp_config["command"]
            assert "roampal.mcp.server" in mcp_config["command"]

    def test_pythonpath_in_environment(self, tmp_path):
        """configure_opencode includes PYTHONPATH for module resolution."""
        from roampal.cli import configure_opencode

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("builtins.print"):
            if sys.platform == "win32":
                with patch.object(Path, "home", return_value=tmp_path):
                    configure_opencode(is_dev=False, force=True)
            else:
                configure_opencode(is_dev=False, force=True)

        config_file = config_dir / "opencode.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
            env = config["mcp"]["roampal-core"].get("environment", {})
            assert "PYTHONPATH" in env
            # PYTHONPATH should point to roampal's parent (so `import roampal` works)
            assert os.path.isabs(env["PYTHONPATH"])

    def test_dev_mode_adds_env_var(self, tmp_path):
        """configure_opencode with is_dev=True adds ROAMPAL_DEV to env."""
        from roampal.cli import configure_opencode

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("builtins.print"):
            if sys.platform == "win32":
                with patch.object(Path, "home", return_value=tmp_path):
                    configure_opencode(is_dev=True, force=True)
            else:
                configure_opencode(is_dev=True, force=True)

        config_file = config_dir / "opencode.json"
        if config_file.exists():
            config = json.loads(config_file.read_text())
            env = config["mcp"]["roampal-core"].get("environment", {})
            assert env.get("ROAMPAL_DEV") == "1"

    def test_idempotency_skips_matching_config(self, tmp_path, capsys):
        """configure_opencode skips write when config already matches."""
        from roampal.cli import configure_opencode
        import roampal.cli

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        plugin_dir = config_dir / "plugin"
        plugin_dir.mkdir()

        # Pre-create matching config (must include ROAMPAL_PLATFORM for v0.3.6+)
        roampal_root = str(Path(roampal.cli.__file__).parent.parent.resolve())
        config = {
            "mcp": {
                "roampal-core": {
                    "type": "local",
                    "command": [sys.executable, "-m", "roampal.mcp.server"],
                    "enabled": True,
                    "environment": {"PYTHONPATH": roampal_root, "ROAMPAL_PLATFORM": "opencode"}
                }
            }
        }
        config_file = config_dir / "opencode.json"
        config_file.write_text(json.dumps(config))

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("builtins.print") as mock_print:
            if sys.platform == "win32":
                with patch.object(Path, "home", return_value=tmp_path):
                    configure_opencode(is_dev=False, force=False)
            else:
                configure_opencode(is_dev=False, force=False)

        # Should print "[OK] already configured"
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "already configured" in printed or "[OK]" in printed

    def test_plugin_install(self, tmp_path):
        """configure_opencode copies plugin to plugin directory."""
        from roampal.cli import configure_opencode
        import roampal.cli

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)

        # Check if plugin source exists
        plugin_source = Path(roampal.cli.__file__).parent / "plugins" / "opencode" / "roampal.ts"

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("builtins.print"):
            if sys.platform == "win32":
                with patch.object(Path, "home", return_value=tmp_path):
                    configure_opencode(is_dev=False, force=True)
            else:
                configure_opencode(is_dev=False, force=True)

        plugin_dest = config_dir / "plugins" / "roampal.ts"
        if plugin_source.exists():
            assert plugin_dest.exists()
            # Content should match source
            assert plugin_dest.read_text(encoding="utf-8") == plugin_source.read_text(encoding="utf-8")

# ============================================================================
# configure_claude_code() Tests
# ============================================================================

class TestConfigureClaudeCode:
    """Test configure_claude_code() function."""

    def test_creates_settings_with_hooks(self, tmp_path):
        """configure_claude_code creates settings.json with hooks."""
        from roampal.cli import configure_claude_code

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.validate_roampal_importable", return_value=True), \
             patch("builtins.print"):
            configure_claude_code(claude_dir, is_dev=False)

        settings = json.loads((claude_dir / "settings.json").read_text())
        assert "hooks" in settings
        assert "UserPromptSubmit" in settings["hooks"]
        assert "Stop" in settings["hooks"]

    def test_adds_mcp_permissions(self, tmp_path):
        """configure_claude_code auto-allows roampal MCP tool permissions."""
        from roampal.cli import configure_claude_code

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.validate_roampal_importable", return_value=True), \
             patch("builtins.print"):
            configure_claude_code(claude_dir, is_dev=False)

        settings = json.loads((claude_dir / "settings.json").read_text())
        allow_list = settings.get("permissions", {}).get("allow", [])
        assert "mcp__roampal-core__search_memory" in allow_list
        assert "mcp__roampal-core__record_response" in allow_list
        assert "mcp__roampal-core__score_memories" in allow_list

    def test_creates_mcp_in_claude_json(self, tmp_path):
        """configure_claude_code writes MCP server to ~/.claude.json."""
        from roampal.cli import configure_claude_code

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        claude_json_path = tmp_path / ".claude.json"

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.validate_roampal_importable", return_value=True), \
             patch("builtins.print"):
            configure_claude_code(claude_dir, is_dev=False)

        assert claude_json_path.exists()
        config = json.loads(claude_json_path.read_text())
        assert "mcpServers" in config
        assert "roampal-core" in config["mcpServers"]
        server_config = config["mcpServers"]["roampal-core"]
        assert server_config["type"] == "stdio"
        assert "-m" in server_config["args"]

    def test_dev_mode_adds_env(self, tmp_path):
        """configure_claude_code with is_dev adds ROAMPAL_DEV to env sections."""
        from roampal.cli import configure_claude_code

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.validate_roampal_importable", return_value=True), \
             patch("builtins.print"):
            configure_claude_code(claude_dir, is_dev=True)

        settings = json.loads((claude_dir / "settings.json").read_text())
        assert settings.get("env", {}).get("ROAMPAL_DEV") == "1"

    def test_dev_mode_hooks_have_env_in_command(self, tmp_path):
        """v0.3.2 Bug 8 fix: Claude Code hooks get ROAMPAL_DEV in command string.

        Previously hooks only got the top-level env section which doesn't propagate
        to subprocess commands. Now the hook commands themselves include the env var.
        """
        from roampal.cli import configure_claude_code

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.validate_roampal_importable", return_value=True), \
             patch("builtins.print"):
            configure_claude_code(claude_dir, is_dev=True)

        settings = json.loads((claude_dir / "settings.json").read_text())

        # Get the UserPromptSubmit hook command
        submit_hooks = settings["hooks"]["UserPromptSubmit"][0]["hooks"]
        submit_cmd = submit_hooks[0]["command"]

        # Get the Stop hook command
        stop_hooks = settings["hooks"]["Stop"][0]["hooks"]
        stop_cmd = stop_hooks[0]["command"]

        # Both commands should have ROAMPAL_DEV in them
        assert "ROAMPAL_DEV" in submit_cmd, f"UserPromptSubmit hook missing ROAMPAL_DEV: {submit_cmd}"
        assert "ROAMPAL_DEV" in stop_cmd, f"Stop hook missing ROAMPAL_DEV: {stop_cmd}"

    def test_prod_mode_hooks_no_env_in_command(self, tmp_path):
        """In prod mode, hooks should NOT have ROAMPAL_DEV in command."""
        from roampal.cli import configure_claude_code

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.validate_roampal_importable", return_value=True), \
             patch("builtins.print"):
            configure_claude_code(claude_dir, is_dev=False)

        settings = json.loads((claude_dir / "settings.json").read_text())

        # Get the UserPromptSubmit hook command
        submit_hooks = settings["hooks"]["UserPromptSubmit"][0]["hooks"]
        submit_cmd = submit_hooks[0]["command"]

        # In prod mode, command should NOT have ROAMPAL_DEV
        assert "ROAMPAL_DEV" not in submit_cmd, f"Prod hook should not have ROAMPAL_DEV: {submit_cmd}"


# ============================================================================
# configure_cursor() Tests
# ============================================================================

class TestConfigureCursor:
    """Test configure_cursor() function."""

    def test_creates_mcp_json(self, tmp_path):
        """configure_cursor creates mcp.json with roampal-core server."""
        from roampal.cli import configure_cursor

        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()

        with patch("builtins.print"):
            configure_cursor(cursor_dir, is_dev=False)

        mcp_config = json.loads((cursor_dir / "mcp.json").read_text())
        assert "mcpServers" in mcp_config
        assert "roampal-core" in mcp_config["mcpServers"]

    def test_creates_hooks_json(self, tmp_path):
        """configure_cursor creates hooks.json with beforeSubmitPrompt and stop."""
        from roampal.cli import configure_cursor

        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()

        with patch("builtins.print"):
            configure_cursor(cursor_dir, is_dev=False)

        hooks_config = json.loads((cursor_dir / "hooks.json").read_text())
        assert hooks_config["version"] == 1
        assert "beforeSubmitPrompt" in hooks_config["hooks"]
        assert "stop" in hooks_config["hooks"]

    def test_dev_mode_hooks(self, tmp_path):
        """configure_cursor with is_dev includes ROAMPAL_DEV in hook commands."""
        from roampal.cli import configure_cursor

        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()

        with patch("builtins.print"):
            configure_cursor(cursor_dir, is_dev=True)

        hooks_config = json.loads((cursor_dir / "hooks.json").read_text())
        submit_cmd = hooks_config["hooks"]["beforeSubmitPrompt"][0]["command"]
        assert "ROAMPAL_DEV" in submit_cmd


# ============================================================================
# cmd_init() Auto-Detection Tests
# ============================================================================

class TestCmdInit:
    """Test cmd_init() auto-detection and explicit flags."""

    def test_explicit_opencode_flag(self, tmp_path):
        """--opencode flag configures OpenCode."""
        from roampal.cli import cmd_init

        opencode_dir = tmp_path / ".config" / "opencode"

        args = MagicMock()
        args.claude_code = False
        args.cursor = False
        args.opencode = True
        args.dev = False
        args.force = True

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.configure_opencode") as mock_configure, \
             patch("roampal.cli.collect_email"), \
             patch("roampal.cli.print_banner"), \
             patch("roampal.cli.print_update_notice"), \
             patch("builtins.print"):
            cmd_init(args)

        mock_configure.assert_called_once()

    def test_explicit_claude_code_flag(self, tmp_path):
        """--claude-code flag configures Claude Code."""
        from roampal.cli import cmd_init

        claude_dir = tmp_path / ".claude"

        args = MagicMock()
        args.claude_code = True
        args.cursor = False
        args.opencode = False
        args.dev = False
        args.force = True

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.configure_claude_code") as mock_configure, \
             patch("roampal.cli.collect_email"), \
             patch("roampal.cli.print_banner"), \
             patch("roampal.cli.print_update_notice"), \
             patch("builtins.print"):
            cmd_init(args)

        mock_configure.assert_called_once()

    def test_auto_detect_multiple_tools(self, tmp_path):
        """Auto-detects multiple installed tools."""
        from roampal.cli import cmd_init

        # Create directories for both tools
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".config" / "opencode").mkdir(parents=True)

        args = MagicMock()
        args.claude_code = False
        args.cursor = False
        args.opencode = False
        args.dev = False
        args.force = True

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("shutil.which", return_value=None), \
             patch("roampal.cli.configure_claude_code") as mock_cc, \
             patch("roampal.cli.configure_opencode") as mock_oc, \
             patch("roampal.cli.collect_email"), \
             patch("roampal.cli.print_banner"), \
             patch("roampal.cli.print_update_notice"), \
             patch("builtins.print"):
            cmd_init(args)

        mock_cc.assert_called_once()
        mock_oc.assert_called_once()

    def test_no_tools_detected_shows_message(self, tmp_path):
        """No detected tools shows help message."""
        from roampal.cli import cmd_init

        args = MagicMock()
        args.claude_code = False
        args.cursor = False
        args.opencode = False
        args.dev = False
        args.force = False

        # Must mock shutil.which AND env vars to fully isolate from real system
        fake_env = {"APPDATA": str(tmp_path / "appdata"), "LOCALAPPDATA": str(tmp_path / "localappdata")}
        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("shutil.which", return_value=None), \
             patch.dict(os.environ, fake_env, clear=False), \
             patch("roampal.cli.print_banner"), \
             patch("roampal.cli.print_update_notice"), \
             patch("builtins.print") as mock_print:
            cmd_init(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "No AI coding tools detected" in printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

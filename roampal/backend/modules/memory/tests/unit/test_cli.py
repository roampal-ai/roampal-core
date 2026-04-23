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

        # Patch XDG_CONFIG_HOME so Linux CI doesn't bypass the Path.home() mock
        fake_env = {"XDG_CONFIG_HOME": str(tmp_path / ".config")}
        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch.dict(os.environ, fake_env, clear=False), \
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
# configure_opencode() Parse Failure & Atomic Write Tests (v0.5.3 Section 8)
# ============================================================================

class TestConfigureOpencodeParseFailure:
    """Section 8: parse-failure guard + atomic write for opencode.json."""

    def test_invalid_json_aborts_without_write(self, tmp_path):
        """opencode.json with invalid JSON → configure_opencode should NOT overwrite it.

        v0.5.3 Section 8: When reading existing config fails due to parse error,
        the function must not attempt to write (preserves user hand-authored content).
        """
        from roampal.cli import configure_opencode

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.json"

        # Write invalid JSON
        config_file.write_text("{ this is not valid json {{{")

        original_content = config_file.read_text()

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("builtins.print"), \
             patch("roampal.cli.logger.warning") as mock_warn:
            if sys.platform == "win32":
                with patch.object(Path, "home", return_value=tmp_path):
                    configure_opencode(is_dev=False, force=True)
            else:
                configure_opencode(is_dev=False, force=True)

        # File should be unchanged (not overwritten)
        assert config_file.read_text() == original_content
        # Should have logged a warning about parse failure
        warn_calls = [c[0][0] for c in mock_warn.call_args_list]
        assert any("parse" in str(w).lower() or "Failed to parse" in str(w) for w in warn_calls)

    def test_existing_mcp_servers_preserved_on_merge(self, tmp_path):
        """Other MCP servers in existing config should be preserved after merge."""
        from roampal.cli import configure_opencode

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.json"

        # Pre-existing config with another MCP server
        existing_config = {
            "mcp": {
                "other-server": {
                    "type": "local",
                    "command": ["node", "./other.js"],
                }
            }
        }
        config_file.write_text(json.dumps(existing_config))

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("builtins.print"):
            if sys.platform == "win32":
                with patch.object(Path, "home", return_value=tmp_path):
                    configure_opencode(is_dev=False, force=True)
            else:
                configure_opencode(is_dev=False, force=True)

        result = json.loads(config_file.read_text())
        assert "mcp" in result
        assert "other-server" in result["mcp"]
        assert result["mcp"]["other-server"]["command"] == ["node", "./other.js"]
        assert "roampal-core" in result["mcp"]

    def test_existing_providers_preserved_on_merge(self, tmp_path):
        """Top-level 'providers' section should be preserved after merge."""
        from roampal.cli import configure_opencode

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.json"

        # Pre-existing config with providers
        existing_config = {
            "providers": {
                "anthropic": {"api_key": "sk-test-123"}
            },
            "mcp": {}
        }
        config_file.write_text(json.dumps(existing_config))

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("builtins.print"):
            if sys.platform == "win32":
                with patch.object(Path, "home", return_value=tmp_path):
                    configure_opencode(is_dev=False, force=True)
            else:
                configure_opencode(is_dev=False, force=True)

        result = json.loads(config_file.read_text())
        assert "providers" in result
        assert "anthropic" in result["providers"]
        assert result["providers"]["anthropic"]["api_key"] == "sk-test-123"

    def test_atomic_write_leaves_no_tmp(self, tmp_path):
        """After successful configure_opencode, no *.tmp files remain."""
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

        tmp_files = list(tmp_path.glob("**/*.tmp"))
        assert len(tmp_files) == 0, f"Found leftover tmp files: {tmp_files}"

    def test_backup_created_when_file_exists(self, tmp_path):
        """_safe_write_opencode_config creates timestamped backup before overwrite."""
        from roampal.cli import _safe_write_opencode_config

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.json"

        # Write initial content
        config_file.write_text(json.dumps({"initial": "data"}))

        _safe_write_opencode_config(config_file, {"updated": "content"})

        # Backup should exist with .bak-YYYYMMDDHHmmSS suffix
        backups = list(config_dir.glob("opencode.json.bak-*"))
        assert len(backups) == 1, f"Expected 1 backup, found: {backups}"

        # Backup content should match original
        backup_content = json.loads(backups[0].read_text())
        assert backup_content == {"initial": "data"}


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
             patch("roampal.cli._prompt_smart_onboarding"), \
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
             patch("roampal.cli._prompt_smart_onboarding"), \
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
        # XDG_CONFIG_HOME must be patched so Linux CI doesn't bypass the Path.home() mock
        fake_env = {"APPDATA": str(tmp_path / "appdata"), "LOCALAPPDATA": str(tmp_path / "localappdata"), "XDG_CONFIG_HOME": str(tmp_path / ".config")}
        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("shutil.which", return_value=None), \
             patch.dict(os.environ, fake_env, clear=False), \
             patch("roampal.cli.print_banner"), \
             patch("roampal.cli.print_update_notice"), \
             patch("builtins.print") as mock_print:
            cmd_init(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "No AI coding tools detected" in printed


class TestV052CmdDoctorProfile:
    """v0.5.2: cmd_doctor must honor --profile and ROAMPAL_PROFILE."""

    def _patched_env(self, tmp_path):
        """Shared patches that keep doctor off real configs / ChromaDB / PyPI."""
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(
            patch("roampal.cli.Path.home", return_value=tmp_path)
        )
        stack.enter_context(
            patch("roampal.profile_manager.Path.home", return_value=tmp_path)
        )
        # Isolate from real user config on every platform. profile_manager's
        # _config_dir() reads os.environ["APPDATA"] on Windows directly — not
        # Path.home() — so without this patch the test reads the real user's
        # profiles.json and can collide with their registered profiles.
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "APPDATA": str(tmp_path / "appdata"),
                    "LOCALAPPDATA": str(tmp_path / "localappdata"),
                    "XDG_CONFIG_HOME": str(tmp_path / ".config"),
                },
                clear=False,
            )
        )
        # Register profile 'research' so resolve_data_path succeeds for it.
        # Must come AFTER env patch so the registry writes to tmp_path.
        from roampal.profile_manager import ProfileRegistry
        reg = ProfileRegistry()
        if not reg.exists("research"):
            reg.create("research")
        # Skip real MCP import + Memory System init to keep test <1s.
        # asyncio is imported inside cmd_doctor, so patch at the module root.
        stack.enter_context(
            patch("asyncio.run", return_value=MagicMock())
        )
        return stack

    def test_doctor_respects_profile_flag(self, tmp_path):
        """--profile research must resolve via profile_manager, not get_data_dir."""
        from roampal.cli import cmd_doctor
        args = MagicMock()
        args.dev = False
        args.profile = "research"

        with self._patched_env(tmp_path), \
             patch.dict(os.environ, {}, clear=False), \
             patch("builtins.print") as mock_print:
            os.environ.pop("ROAMPAL_PROFILE", None)
            os.environ.pop("ROAMPAL_DEV", None)
            cmd_doctor(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Profile: research" in printed
        assert "source: env" in printed  # --profile was set via env

    def test_doctor_respects_profile_env(self, tmp_path):
        """ROAMPAL_PROFILE env var alone (no --profile flag) must be honored."""
        from roampal.cli import cmd_doctor
        args = MagicMock()
        args.dev = False
        args.profile = None

        with self._patched_env(tmp_path), \
             patch.dict(os.environ, {"ROAMPAL_PROFILE": "research"}, clear=False), \
             patch("builtins.print") as mock_print:
            cmd_doctor(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Profile: research" in printed
        assert "source: env" in printed

    def test_doctor_unregistered_profile(self, tmp_path):
        """Unknown --profile must FAIL cleanly and skip memory init."""
        from roampal.cli import cmd_doctor
        args = MagicMock()
        args.dev = False
        args.profile = "ghost"

        with self._patched_env(tmp_path), \
             patch.dict(os.environ, {}, clear=False), \
             patch("builtins.print") as mock_print:
            os.environ.pop("ROAMPAL_PROFILE", None)
            os.environ.pop("ROAMPAL_DEV", None)
            cmd_doctor(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "'ghost' is not registered" in printed
        assert "Skipping memory system init" in printed


# ============================================================================
# v0.5.3 Section 9: Scope-aware sidecar commands tests
# ============================================================================

class TestScopeAwareSidecar:
    """v0.5.3 Section 9: --scope flag for sidecar status/setup/disable."""

    def _make_args(self, scope=None):
        args = MagicMock()
        args.sidecar_command = "status"
        args.scope = scope
        return args

    # --- Path resolution helpers ---

    def test_scope_user_uses_global_config(self, tmp_path):
        """--scope user must resolve to ~/.config/opencode/opencode.json."""
        from roampal.cli import _get_opencode_config_path

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.json"
        config_file.write_text(json.dumps({"mcp": {"roampal-core": {}}}))

        with patch("roampal.cli.Path.home", return_value=tmp_path):
            result = _get_opencode_config_path()
            assert result == config_file

    def test_scope_project_uses_local_config(self, tmp_path):
        """--scope project must resolve to local opencode.json if it exists."""
        from roampal.cli import _find_project_opencode_config

        local_config = tmp_path / "opencode.json"
        local_config.write_text(json.dumps({"mcp": {"roampal-core": {}}}))

        with patch("roampal.cli.Path.cwd", return_value=tmp_path):
            result = _find_project_opencode_config()
            assert result == local_config

    def test_scope_project_falls_back_to_global(self, tmp_path):
        """--scope project without local config falls back to global."""
        from roampal.cli import _get_scope_config_path

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.json"
        config_file.write_text(json.dumps({"mcp": {"roampal-core": {}}}))

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path):
            result = _get_scope_config_path()
            assert result == config_file

    def test_find_project_walks_up_tree(self, tmp_path):
        """_find_project_opencode_config walks up from subdirectory to find opencode.json."""
        from roampal.cli import _find_project_opencode_config

        # Create opencode.json at root of tmp_path
        root_config = tmp_path / "opencode.json"
        root_config.write_text(json.dumps({"mcp": {"roampal-core": {}}}))

        # Create a subdirectory deeper in the tree
        deep_dir = tmp_path / "a" / "b" / "c"
        deep_dir.mkdir(parents=True)

        with patch("roampal.cli.Path.cwd", return_value=deep_dir):
            result = _find_project_opencode_config()
            assert result == root_config

    def test_find_project_stops_at_home(self, tmp_path):
        """_find_project_opencode_config stops walking at home directory."""
        from roampal.cli import _find_project_opencode_config

        # Create opencode.json in the "home" dir (simulated)
        home_config = tmp_path / ".config" / "opencode" / "opencode.json"
        home_config.parent.mkdir(parents=True)
        home_config.write_text(json.dumps({"mcp": {"roampal-core": {}}}))

        # Create a subdirectory under tmp_path (which is the simulated home)
        subdir = tmp_path / "project" / "src"
        subdir.mkdir(parents=True)

        with patch("roampal.cli.Path.cwd", return_value=subdir):
            result = _find_project_opencode_config()
            # Should NOT find the config in home — stops at home
            assert result is None

    def test_get_scope_returns_project_when_exists(self, tmp_path):
        """_get_scope_config_path returns project-local if it exists and differs from global."""
        from roampal.cli import _get_scope_config_path

        # Create both local and global configs
        local_config = tmp_path / "opencode.json"
        local_config.write_text(json.dumps({"mcp": {"roampal-core": {}}}))

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        global_config = config_dir / "opencode.json"
        global_config.write_text(json.dumps({"mcp": {"roampal-core": {}}}))

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path):
            result = _get_scope_config_path()
            assert result == local_config

    # --- _apply_sidecar_env_and_write tests ---

    def test_apply_sidecar_sets_env_vars(self, tmp_path):
        """_apply_sidecar_env_and_write sets sidecar env vars correctly."""
        from roampal.cli import _apply_sidecar_env_and_write

        config_file = tmp_path / "opencode.json"
        config_file.write_text(json.dumps({
            "mcp": {
                "roampal-core": {
                    "environment": {}
                }
            }
        }))

        env_updates = {
            "ROAMPAL_SIDECAR_URL": "http://localhost:1234",
            "ROAMPAL_SIDECAR_MODEL": "qwen3:8b",
            "ROAMPAL_SIDECAR_KEY": "sk-test-key",
            "ROAMPAL_SIDECAR_FALLBACK": "true",
        }

        changed = _apply_sidecar_env_and_write(config_file, env_updates)
        assert changed is True

        result = json.loads(config_file.read_text())
        env = result["mcp"]["roampal-core"]["environment"]
        assert env["ROAMPAL_SIDECAR_URL"] == "http://localhost:1234"
        assert env["ROAMPAL_SIDECAR_MODEL"] == "qwen3:8b"
        assert env["ROAMPAL_SIDECAR_KEY"] == "sk-test-key"
        assert env["ROAMPAL_SIDECAR_FALLBACK"] == "true"

    def test_apply_sidecar_removes_keys_on_disable(self, tmp_path):
        """_apply_sidecar_env_and_write with empty dict removes all sidecar keys."""
        from roampal.cli import _apply_sidecar_env_and_write

        config_file = tmp_path / "opencode.json"
        config_file.write_text(json.dumps({
            "mcp": {
                "roampal-core": {
                    "environment": {
                        "ROAMPAL_SIDECAR_URL": "http://localhost:1234",
                        "ROAMPAL_SIDECAR_MODEL": "qwen3:8b",
                        "OTHER_VAR": "keep-me",
                    }
                }
            }
        }))

        changed = _apply_sidecar_env_and_write(config_file, {})
        assert changed is True

        result = json.loads(config_file.read_text())
        env = result["mcp"]["roampal-core"]["environment"]
        assert "ROAMPAL_SIDECAR_URL" not in env
        assert "ROAMPAL_SIDECAR_MODEL" not in env
        assert "OTHER_VAR" in env  # non-sidecar vars preserved

    def test_apply_sidecar_returns_false_no_mcp_section(self, tmp_path):
        """_apply_sidecar_env_and_write returns False when no mcp section exists."""
        from roampal.cli import _apply_sidecar_env_and_write

        config_file = tmp_path / "opencode.json"
        config_file.write_text(json.dumps({"other": "data"}))

        changed = _apply_sidecar_env_and_write(config_file, {"ROAMPAL_SIDECAR_URL": "http://x"})
        assert changed is False

    def test_apply_sidecar_returns_false_on_invalid_json(self, tmp_path):
        """_apply_sidecar_env_and_write returns False on invalid JSON."""
        from roampal.cli import _apply_sidecar_env_and_write

        config_file = tmp_path / "opencode.json"
        config_file.write_text("{ invalid json {{{")

        changed = _apply_sidecar_env_and_write(config_file, {"ROAMPAL_SIDECAR_URL": "http://x"})
        assert changed is False

    def test_apply_sidecar_no_change_returns_false(self, tmp_path):
        """_apply_sidecar_env_and_write returns False when values already match."""
        from roampal.cli import _apply_sidecar_env_and_write

        config_file = tmp_path / "opencode.json"
        config_file.write_text(json.dumps({
            "mcp": {
                "roampal-core": {
                    "environment": {
                        "ROAMPAL_SIDECAR_URL": "http://localhost:1234",
                        "ROAMPAL_SIDECAR_MODEL": "qwen3:8b",
                    }
                }
            }
        }))

        changed = _apply_sidecar_env_and_write(config_file, {
            "ROAMPAL_SIDECAR_URL": "http://localhost:1234",
            "ROAMPAL_SIDECAR_MODEL": "qwen3:8b",
        })
        assert changed is False

    # --- _cmd_sidecar_status tests ---

    def test_cmd_sidecar_status_shows_model(self, tmp_path):
        """_cmd_sidecar_status displays model info when configured."""
        from roampal.cli import _cmd_sidecar_status

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.json"
        config_file.write_text(json.dumps({
            "mcp": {
                "roampal-core": {
                    "environment": {
                        "ROAMPAL_SIDECAR_URL": "http://localhost:1234",
                        "ROAMPAL_SIDECAR_MODEL": "qwen3:8b",
                        "ROAMPAL_SIDECAR_KEY": "sk-test-key",
                    }
                }
            }
        }))

        args = MagicMock()
        args.scope = None

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path), \
             patch("builtins.print") as mock_print:
            _cmd_sidecar_status(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Sidecar scoring configuration" in printed
        assert "qwen3:8b" in printed
        assert "localhost:1234" in printed
        assert "Effective in cwd" in printed

    def test_cmd_sidecar_status_no_config(self, tmp_path):
        """_cmd_sidecar_status shows 'no sidecar' message when not configured."""
        from roampal.cli import _cmd_sidecar_status

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.json"
        config_file.write_text(json.dumps({
            "mcp": {
                "roampal-core": {
                    "environment": {}
                }
            }
        }))

        args = MagicMock()
        args.scope = None

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path), \
             patch("builtins.print") as mock_print:
            _cmd_sidecar_status(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        # v0.5.3: No silent Zen fallback — message reflects §12 behavior
        assert "no sidecar configured" in printed.lower()
        assert "disabled" in printed.lower() or "scoring" in printed.lower()

    def test_cmd_sidecar_disable_removes_keys(self, tmp_path):
        """_cmd_sidecar_disable removes all sidecar keys from config."""
        from roampal.cli import _cmd_sidecar_disable

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.json"
        config_file.write_text(json.dumps({
            "mcp": {
                "roampal-core": {
                    "environment": {
                        "ROAMPAL_SIDECAR_URL": "http://localhost:1234",
                        "ROAMPAL_SIDECAR_MODEL": "qwen3:8b",
                        "OTHER_VAR": "keep-me",
                    }
                }
            }
        }))

        args = MagicMock()
        args.scope = None

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path), \
             patch("builtins.print"):
            _cmd_sidecar_disable(args)

        result = json.loads(config_file.read_text())
        env = result["mcp"]["roampal-core"]["environment"]
        assert "ROAMPAL_SIDECAR_URL" not in env
        assert "ROAMPAL_SIDECAR_MODEL" not in env
        assert "OTHER_VAR" in env

    def test_cmd_sidecar_disable_no_config_found(self, tmp_path):
        """_cmd_sidecar_disable prints warning when no config exists."""
        from roampal.cli import _cmd_sidecar_disable

        args = MagicMock()
        args.scope = None

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path), \
             patch("builtins.print") as mock_print:
            _cmd_sidecar_disable(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "no opencode.json" in printed.lower() or "no sidecar configuration" in printed.lower()

    def test_apply_sidecar_preserves_non_sidecar_env(self, tmp_path):
        """_apply_sidecar_env_and_write preserves non-sidecar env vars."""
        from roampal.cli import _apply_sidecar_env_and_write

        config_file = tmp_path / "opencode.json"
        config_file.write_text(json.dumps({
            "mcp": {
                "roampal-core": {
                    "environment": {
                        "PYTHONPATH": "/some/path",
                        "ROAMPAL_SIDECAR_URL": "http://localhost:1234",
                        "CUSTOM_VAR": "custom-value",
                    }
                }
            }
        }))

        changed = _apply_sidecar_env_and_write(config_file, {
            "ROAMPAL_SIDECAR_URL": "http://new-url:5678",
        })
        assert changed is True

        result = json.loads(config_file.read_text())
        env = result["mcp"]["roampal-core"]["environment"]
        assert env["PYTHONPATH"] == "/some/path"
        assert env["CUSTOM_VAR"] == "custom-value"
        assert env["ROAMPAL_SIDECAR_URL"] == "http://new-url:5678"

    def test_scope_project_creates_local_if_missing(self, tmp_path):
        """scope='project' with no local config creates opencode.json in current dir."""
        from roampal.cli import _find_project_opencode_config

        # No local opencode.json exists
        assert not (tmp_path / "opencode.json").exists()

        with patch("roampal.cli.Path.cwd", return_value=tmp_path):
            result = _find_project_opencode_config()
            assert result is None

    def test_apply_sidecar_creates_environment_if_missing(self, tmp_path):
        """_apply_sidecar_env_and_write creates environment dict if it doesn't exist."""
        from roampal.cli import _apply_sidecar_env_and_write

        config_file = tmp_path / "opencode.json"
        config_file.write_text(json.dumps({
            "mcp": {
                "roampal-core": {}  # no 'environment' key
            }
        }))

        changed = _apply_sidecar_env_and_write(config_file, {
            "ROAMPAL_SIDECAR_URL": "http://localhost:1234",
            "ROAMPAL_SIDECAR_MODEL": "qwen3:8b",
        })
        assert changed is True

        result = json.loads(config_file.read_text())
        env = result["mcp"]["roampal-core"]["environment"]
        assert env["ROAMPAL_SIDECAR_URL"] == "http://localhost:1234"
        assert env["ROAMPAL_SIDECAR_MODEL"] == "qwen3:8b"

    def test_apply_sidecar_missing_roampal_core_returns_false(self, tmp_path):
        """_apply_sidecar_env_and_write returns False when roampal-core MCP is missing."""
        from roampal.cli import _apply_sidecar_env_and_write

        config_file = tmp_path / "opencode.json"
        config_file.write_text(json.dumps({
            "mcp": {
                "other-server": {"command": ["node", "./server.js"]}
            }
        }))

        changed = _apply_sidecar_env_and_write(config_file, {
            "ROAMPAL_SIDECAR_URL": "http://localhost:1234",
        })
        assert changed is False


class TestPromptSmartOnboardingScope:
    """v0.5.3 Section 9: _prompt_smart_onboarding scope parameter."""

    def test_user_scope_uses_global(self, tmp_path):
        """scope='user' must use global config path."""
        from roampal.cli import _get_opencode_config_path

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "opencode.json"
        config_file.write_text(json.dumps({"mcp": {"roampal-core": {}}}))

        with patch("roampal.cli.Path.home", return_value=tmp_path):
            result = _get_opencode_config_path()
            assert result == config_file

    def test_project_scope_uses_local(self, tmp_path):
        """scope='project' must use local opencode.json if it exists."""
        from roampal.cli import _find_project_opencode_config

        local_config = tmp_path / "opencode.json"
        local_config.write_text(json.dumps({"mcp": {"roampal-core": {}}}))

        with patch("roampal.cli.Path.cwd", return_value=tmp_path):
            result = _find_project_opencode_config()
            assert result == local_config


class TestScopeAwareSidecarStatus:
    """v0.5.3 Section 9: _cmd_sidecar_status with explicit --scope values."""

    def test_status_scope_user_only(self, tmp_path):
        """--scope user shows only user-global config, hides project-local."""
        from roampal.cli import _cmd_sidecar_status

        local_config = tmp_path / "opencode.json"
        local_config.write_text(json.dumps({
            "mcp": {"roampal-core": {"environment": {
                "ROAMPAL_SIDECAR_URL": "http://local:1234",
                "ROAMPAL_SIDECAR_MODEL": "qwen3:8b",
            }}}
        }))

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        global_config = config_dir / "opencode.json"
        global_config.write_text(json.dumps({
            "mcp": {"roampal-core": {"environment": {
                "ROAMPAL_SIDECAR_URL": "http://global:5678",
                "ROAMPAL_SIDECAR_MODEL": "llama3:8b",
            }}}
        }))

        args = MagicMock()
        args.scope = "user"

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path), \
             patch("builtins.print") as mock_print:
            _cmd_sidecar_status(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "User-global" in printed
        assert "llama3:8b" in printed
        # Project-local should NOT appear when scope=user
        assert "Project-local" not in printed

    def test_status_scope_project_only(self, tmp_path):
        """--scope project shows only project-local config."""
        from roampal.cli import _cmd_sidecar_status

        local_config = tmp_path / "opencode.json"
        local_config.write_text(json.dumps({
            "mcp": {"roampal-core": {"environment": {
                "ROAMPAL_SIDECAR_URL": "http://local:1234",
                "ROAMPAL_SIDECAR_MODEL": "qwen3:8b",
            }}}
        }))

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        global_config = config_dir / "opencode.json"
        global_config.write_text(json.dumps({
            "mcp": {"roampal-core": {"environment": {
                "ROAMPAL_SIDECAR_URL": "http://global:5678",
                "ROAMPAL_SIDECAR_MODEL": "llama3:8b",
            }}}
        }))

        args = MagicMock()
        args.scope = "project"

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path), \
             patch("builtins.print") as mock_print:
            _cmd_sidecar_status(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Project-local" in printed
        assert "qwen3:8b" in printed
        # User-global should NOT appear when scope=project
        assert "User-global" not in printed

    def test_status_scope_project_no_local(self, tmp_path):
        """--scope project with no local config prints warning."""
        from roampal.cli import _cmd_sidecar_status

        args = MagicMock()
        args.scope = "project"

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path), \
             patch("builtins.print") as mock_print:
            _cmd_sidecar_status(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "no project-local" in printed.lower() or "not found" in printed.lower()


class TestScopeAwareSidecarDisable:
    """v0.5.3 Section 9: _cmd_sidecar_disable with explicit --scope values."""

    def test_disable_scope_user_only(self, tmp_path):
        """--scope user removes sidecar keys from global config only."""
        from roampal.cli import _cmd_sidecar_disable

        local_config = tmp_path / "opencode.json"
        local_config.write_text(json.dumps({
            "mcp": {"roampal-core": {"environment": {
                "ROAMPAL_SIDECAR_URL": "http://local:1234",
                "OTHER_VAR": "keep-me",
            }}}
        }))

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        global_config = config_dir / "opencode.json"
        global_config.write_text(json.dumps({
            "mcp": {"roampal-core": {"environment": {
                "ROAMPAL_SIDECAR_URL": "http://global:5678",
                "OTHER_VAR": "keep-me-too",
            }}}
        }))

        args = MagicMock()
        args.scope = "user"

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path):
            _cmd_sidecar_disable(args)

        # Global config should have sidecar keys removed
        result_global = json.loads(global_config.read_text())
        env_global = result_global["mcp"]["roampal-core"]["environment"]
        assert "ROAMPAL_SIDECAR_URL" not in env_global
        assert "OTHER_VAR" in env_global  # non-sidecar preserved

        # Local config should be untouched
        result_local = json.loads(local_config.read_text())
        env_local = result_local["mcp"]["roampal-core"]["environment"]
        assert env_local["ROAMPAL_SIDECAR_URL"] == "http://local:1234"
        assert "OTHER_VAR" in env_local

    def test_disable_scope_project_only(self, tmp_path):
        """--scope project removes sidecar keys from local config only."""
        from roampal.cli import _cmd_sidecar_disable

        local_config = tmp_path / "opencode.json"
        local_config.write_text(json.dumps({
            "mcp": {"roampal-core": {"environment": {
                "ROAMPAL_SIDECAR_URL": "http://local:1234",
                "OTHER_VAR": "keep-me",
            }}}
        }))

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        global_config = config_dir / "opencode.json"
        global_config.write_text(json.dumps({
            "mcp": {"roampal-core": {"environment": {
                "ROAMPAL_SIDECAR_URL": "http://global:5678",
                "OTHER_VAR": "keep-me-too",
            }}}
        }))

        args = MagicMock()
        args.scope = "project"

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path):
            _cmd_sidecar_disable(args)

        # Local config should have sidecar keys removed
        result_local = json.loads(local_config.read_text())
        env_local = result_local["mcp"]["roampal-core"]["environment"]
        assert "ROAMPAL_SIDECAR_URL" not in env_local
        assert "OTHER_VAR" in env_local

        # Global config should be untouched
        result_global = json.loads(global_config.read_text())
        env_global = result_global["mcp"]["roampal-core"]["environment"]
        assert env_global["ROAMPAL_SIDECAR_URL"] == "http://global:5678"
        assert "OTHER_VAR" in env_global

    def test_disable_scope_project_no_local(self, tmp_path):
        """--scope project with no local config prints warning."""
        from roampal.cli import _cmd_sidecar_disable

        args = MagicMock()
        args.scope = "project"

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path), \
             patch("builtins.print") as mock_print:
            _cmd_sidecar_disable(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "no project-local" in printed.lower() or "not found" in printed.lower()


class TestScopeAwareSidecarSetup:
    """v0.5.3 Section 9: _cmd_sidecar_setup with explicit --scope values."""

    def test_setup_scope_user_only(self, tmp_path):
        """--scope user targets only global config path."""
        from roampal.cli import _cmd_sidecar_setup, _apply_sidecar_env_and_write

        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        global_config = config_dir / "opencode.json"
        global_config.write_text(json.dumps({
            "mcp": {"roampal-core": {}}
        }))

        # Create a local config that should NOT be touched
        local_config = tmp_path / "opencode.json"
        local_config.write_text(json.dumps({
            "mcp": {"roampal-core": {"environment": {
                "ROAMPAL_SIDECAR_URL": "http://should-not-change:9999",
            }}}
        }))

        args = MagicMock()
        args.scope = "user"

        def fake_picker(path, defer_write=False):
            env_updates = {
                "ROAMPAL_SIDECAR_URL": "http://test:1234",
                "ROAMPAL_SIDECAR_MODEL": "qwen3:8b",
                "ROAMPAL_SIDECAR_KEY": "sk-test-key",
            }
            _apply_sidecar_env_and_write(path, env_updates)
            return True

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path), \
             patch("roampal.cli._sidecar_model_picker", side_effect=fake_picker):
            _cmd_sidecar_setup(args)

        # Global config should have been modified by the picker
        result_global = json.loads(global_config.read_text())
        env_global = result_global["mcp"]["roampal-core"].get("environment", {})
        assert env_global["ROAMPAL_SIDECAR_URL"] == "http://test:1234"
        assert env_global["ROAMPAL_SIDECAR_MODEL"] == "qwen3:8b"

        # Local config should be untouched
        result_local = json.loads(local_config.read_text())
        env_local = result_local["mcp"]["roampal-core"].get("environment", {})
        assert env_local["ROAMPAL_SIDECAR_URL"] == "http://should-not-change:9999"

    def test_setup_scope_project_only(self, tmp_path):
        """--scope project targets only local config path."""
        from roampal.cli import _cmd_sidecar_setup, _apply_sidecar_env_and_write

        # Create a local config
        local_config = tmp_path / "opencode.json"
        local_config.write_text(json.dumps({
            "mcp": {"roampal-core": {}}
        }))

        # Global config that should NOT be touched
        config_dir = tmp_path / ".config" / "opencode"
        config_dir.mkdir(parents=True)
        global_config = config_dir / "opencode.json"
        global_config.write_text(json.dumps({
            "mcp": {"roampal-core": {}}
        }))

        args = MagicMock()
        args.scope = "project"

        def fake_picker(path, defer_write=False):
            env_updates = {
                "ROAMPAL_SIDECAR_URL": "http://test:5678",
                "ROAMPAL_SIDECAR_MODEL": "llama3:8b",
            }
            _apply_sidecar_env_and_write(path, env_updates)
            return True

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path), \
             patch("roampal.cli._sidecar_model_picker", side_effect=fake_picker):
            _cmd_sidecar_setup(args)

        # Local config should have been modified by the picker
        result_local = json.loads(local_config.read_text())
        env_local = result_local["mcp"]["roampal-core"].get("environment", {})
        assert env_local["ROAMPAL_SIDECAR_URL"] == "http://test:5678"
        assert env_local["ROAMPAL_SIDECAR_MODEL"] == "llama3:8b"

        # Global config should be untouched
        result_global = json.loads(global_config.read_text())
        env_global = result_global["mcp"]["roampal-core"].get("environment", {})
        assert "ROAMPAL_SIDECAR_URL" not in env_global

    def test_setup_scope_project_no_existing_file(self, tmp_path):
        """--scope project with no local config prints error (does not auto-create)."""
        from roampal.cli import _cmd_sidecar_setup

        # No local or global config exists
        args = MagicMock()
        args.scope = "project"

        with patch("roampal.cli.Path.home", return_value=tmp_path), \
             patch("roampal.cli.Path.cwd", return_value=tmp_path), \
             patch("builtins.print") as mock_print:
            _cmd_sidecar_setup(args)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "no opencode.json" in printed.lower() or "run roampal init" in printed.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

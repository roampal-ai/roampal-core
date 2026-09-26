"""
Client-seam wiring tests (v0.6.0 Item 3 / Task 14).

All three client entry points resolve the profile through THE one helper
(profile_manager.profile_header_value -> active_profile_name's full
precedence walk); env var still beats cwd binding.

Seam rows (Test Plan B):
- stdio MCP (_get_mcp_profile_name, unit-shaped): cwd inside a bound dir
  -> seam returns the bound name; env -> env; default -> None.
- Claude Code/Cursor hook context path (cmd_context invocation shape):
  httpx POST carries X-Roampal-Profile resolved from cwd binding.
- opencode plugin: no new behavior (fill-through comment in roampal.ts);
  the fall-through is exercised here only via the server-side default DNS.

Env/config isolation mirrors test_profile_bindings.py (temp registry via
patched _registry_path; chdir only where the seam itself uses real cwd).
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
)

import json
from io import StringIO
from unittest.mock import patch, MagicMock

import roampal.profile_manager as pm
from roampal.profile_manager import profile_header_value


def _seed(path, profiles=None, bindings=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(profiles or {})
    if bindings:
        payload.update(bindings)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _isolated(monkeypatch, tmp_path, profiles=None, bindings=None):
    """Temp APPDATA-equivalent registry + env, returns the registry path."""
    reg_path = tmp_path / "roampal" / "profiles.json"
    _seed(reg_path, profiles=profiles, bindings=bindings)
    monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
    monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
    monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
    return reg_path


class TestMcpSeam:
    def test_seam_is_the_one_helper(self, monkeypatch, tmp_path):
        """_get_mcp_profile_name is the helper verbatim (Round 2 Task 18):
        default -> the literal "default" header value, named -> the name."""
        import roampal.mcp.server as srv

        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        monkeypatch.setattr(pm, "_registry_path", lambda: tmp_path / "roampal" / "profiles.json")
        # cwd = repo root, no bindings -> explicit "default"
        monkeypatch.chdir(tmp_path)
        assert srv._get_mcp_profile_name() == "default"

    def test_mcp_seam_resolves_server_pin_as_late_tier(self, monkeypatch, tmp_path):
        """fix 5: with nothing else configured, MCP names the SERVER's
        launch pin (per-port pin file) instead of "default" — matching the
        hooks, so `roampal start --profile X` reaches MCP clients too."""
        import roampal.mcp.server as srv

        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        monkeypatch.setattr(pm, "_registry_path", lambda: tmp_path / "roampal" / "profiles.json")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(pm, "_config_dir", lambda: tmp_path / "config")
        monkeypatch.setattr(srv, "_dev_mode", False)
        pm.write_server_pin(27182, "work")
        monkeypatch.delenv("ROAMPAL_PORT", raising=False)
        assert srv._get_mcp_profile_name() == "work"

        # Unpinned -> the literal "default" (Task 18 unchanged).
        pm.clear_server_pin(27182)
        assert srv._get_mcp_profile_name() == "default"

    def test_mcp_seam_explicit_default_env_beats_pin(self, monkeypatch, tmp_path):
        """Round-2 parity: an explicit ROAMPAL_PROFILE (even the literal
        "default") short-circuits BEFORE the pin tier in the MCP, exactly
        as in the hooks — pre-fix the MCP swapped in the pin while the
        hooks returned the env's "default" verbatim."""
        import roampal.mcp.server as srv

        monkeypatch.setenv("ROAMPAL_PROFILE", "default")
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        monkeypatch.setattr(pm, "_registry_path", lambda: tmp_path / "roampal" / "profiles.json")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(pm, "_config_dir", lambda: tmp_path / "config")
        monkeypatch.setattr(srv, "_dev_mode", False)
        monkeypatch.delenv("ROAMPAL_PORT", raising=False)
        pm.write_server_pin(27182, "work")
        assert srv._get_mcp_profile_name() == "default"

    def test_mcp_seam_resolves_cwd_binding(self, monkeypatch, tmp_path):
        """Matrix row (integration-shaped): stdio MCP launched with cwd
        inside a bound dir -> resolved profile equals the binding."""
        import roampal.mcp.server as srv

        proj = tmp_path / "proj"
        proj.mkdir()
        _seed(
            tmp_path / "roampal" / "profiles.json",
            profiles={"work": None},
            bindings={"bindings": {str(proj): "work"}},
        )
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        monkeypatch.setattr(pm, "_registry_path", lambda: tmp_path / "roampal" / "profiles.json")
        monkeypatch.chdir(proj)
        assert srv._get_mcp_profile_name() == "work"

    def test_mcp_seam_env_beats_binding(self, monkeypatch, tmp_path):
        import roampal.mcp.server as srv

        proj = tmp_path / "proj"
        proj.mkdir()
        _seed(
            tmp_path / "roampal" / "profiles.json",
            profiles={"work": None, "other": None},
            bindings={"bindings": {str(proj): "work"}},
        )
        monkeypatch.setenv("ROAMPAL_PROFILE", "other")
        monkeypatch.setattr(pm, "_registry_path", lambda: tmp_path / "roampal" / "profiles.json")
        monkeypatch.chdir(proj)
        assert srv._get_mcp_profile_name() == "other"

    def test_mcp_re_resolves_per_call_binding_flip(self, monkeypatch, tmp_path):
        """Round 2 Task 20: no per-process cache — flipping a binding (or
        unbinding) between two MCP tool calls takes effect immediately."""
        import json as _json

        import roampal.mcp.server as srv

        reg_path = tmp_path / "roampal" / "profiles.json"
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        reg_path.write_text(_json.dumps({
            "work_a": None, "other": None,
            "bindings": {str(tmp_path): "work_a"},
        }), encoding="utf-8")
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.chdir(tmp_path)

        assert srv._get_mcp_profile_name() == "work_a"

        # Unbind mid-"session": the SAME process, next tool call.
        reg_path.write_text(_json.dumps({
            "work_a": None, "other": None,
        }), encoding="utf-8")
        assert srv._get_mcp_profile_name() == "default"

        # Re-bind the same directory to a different profile; next call flips.
        reg_path.write_text(_json.dumps({
            "work_a": None, "other": None,
            "bindings": {str(tmp_path): "other"},
        }), encoding="utf-8")
        assert srv._get_mcp_profile_name() == "other"

    def test_helper_value_contract(self, monkeypatch, tmp_path):
        """profile_header_value contract (Round 2 Task 18): clients ALWAYS
        name their profile — default resolves to the literal "default", a
        full header value, never None."""
        _seed(tmp_path / "roampal" / "profiles.json")
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        monkeypatch.setattr(pm, "_registry_path", lambda: tmp_path / "roampal" / "profiles.json")
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)
        assert profile_header_value() == "default"
        monkeypatch.setenv("ROAMPAL_PROFILE", "work")
        assert profile_header_value() == "work"


class TestHookHeaderSeam:
    """Round 2 Task 18: the hook's own precedence, matching what the MCP
    process sees — process env > project MCP-config env > binding > use >
    default. Exercises user_prompt_submit_hook's seam (identical copy in
    stop_hook; divergence would need EQUALS-side updates is a deliberate
    maintenance note, so only one is wired here)."""

    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        monkeypatch.setattr(pm, "_registry_path", lambda: tmp_path / "roampal" / "profiles.json")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        return tmp_path / "roampal" / "profiles.json"

    def test_hook_sends_explicit_default(self, monkeypatch, tmp_path):
        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        # no .mcp.json ancestors; real cwd (tmp) has no claude.json project
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "default"

    def test_hook_resolves_server_pin_as_late_tier(self, monkeypatch, tmp_path):
        """fix 5: with nothing else configured, the hook names the SERVER's
        launch pin (per-port pin file) instead of sending "default" — how
        `roampal start --profile X` reaches header-sending clients."""
        import roampal.profile_manager as pm

        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.delenv("ROAMPAL_DEV", raising=False)
        monkeypatch.setattr(pm, "_config_dir", lambda: tmp_path / "config")
        pm.write_server_pin(27182, "work")
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "work"

    def test_hook_pin_tier_respects_dev_port(self, monkeypatch, tmp_path):
        """fix 5: the pin file is per-port — dev hooks (27183) read the dev
        pin, prod hooks (27182) the prod pin."""
        import roampal.profile_manager as pm

        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.setattr(pm, "_config_dir", lambda: tmp_path / "config")
        pm.write_server_pin(27182, "prod-pin")
        pm.write_server_pin(27183, "dev-pin")

        from roampal.hooks import user_prompt_submit_hook as ups

        monkeypatch.delenv("ROAMPAL_DEV", raising=False)
        assert ups._roampal_headers()["X-Roampal-Profile"] == "prod-pin"
        monkeypatch.setenv("ROAMPAL_DEV", "1")
        assert ups._roampal_headers()["X-Roampal-Profile"] == "dev-pin"

    def test_hook_explicit_default_env_beats_pin(self, monkeypatch, tmp_path):
        """Round-2 parity: an env explicitly set to "default" is terminal —
        the pin tier is unreachable (hooks already did this; the MCP now
        matches)."""
        import roampal.profile_manager as pm

        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.setenv("ROAMPAL_PROFILE", "default")
        monkeypatch.delenv("ROAMPAL_DEV", raising=False)
        monkeypatch.setattr(pm, "_config_dir", lambda: tmp_path / "config")
        pm.write_server_pin(27182, "work")
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "default"

    def test_hook_pin_tier_loses_to_binding(self, monkeypatch, tmp_path):
        """fix 5 ordering: the pin is a LATE tier - env, config env, and
        bindings all beat it. Here a cwd binding wins over the pin."""
        import roampal.profile_manager as pm

        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.delenv("ROAMPAL_DEV", raising=False)
        monkeypatch.setattr(pm, "_config_dir", lambda: tmp_path / "config")
        pm.write_server_pin(27182, "work")
        bound = tmp_path / "boundproj"
        bound.mkdir()
        reg_path = tmp_path / "roampal" / "profiles.json"
        reg_path.write_text(json.dumps({
            "work": None, "other": None,
            "bindings": {str(bound): "other"},
        }), encoding="utf-8")
        monkeypatch.chdir(bound)
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "other"

    def test_hook_config_env_from_mcp_json(self, monkeypatch, tmp_path):
        """(a) project scope: .mcp.json env reaches the hook."""
        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"roampal-core": {"env": {"ROAMPAL_PROFILE": "work"}}}}),
            encoding="utf-8",
        )
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "work"

    def test_hook_config_env_from_ancestor_mcp_json(self, monkeypatch, tmp_path):
        """(a') cwd ancestry: a subproject inherits the bound project root's
        .mcp.json env (init writes it at the project root)."""
        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        root = tmp_path / "projroot"
        root.mkdir()
        (root / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"roampal-core": {"env": {"ROAMPAL_PROFILE": "work"}}}}),
            encoding="utf-8",
        )
        sub = root / "sub" / "nested"
        sub.mkdir(parents=True)
        monkeypatch.chdir(sub)
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "work"
    def test_hook_config_env_from_claude_json_local_scope(self, monkeypatch, tmp_path):
        """(b) Claude Code local scope: ~/.claude.json
        projects.<cwd>.mcpServers[roampal*].env.ROAMPAL_PROFILE.

        v0.6.0 review fix 3 regression: Claude Code writes the key with
        FORWARD slashes on Windows ('C:/proj'), not the native form
        ('C:\\proj') that a naive projects.get(str(cwd)) lookup uses — the
        seed here uses Claude Code's actual form."""
        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        cwd = tmp_path / "localproj"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        claude_code_key = str(cwd.resolve()).replace("\\", "/")
        (tmp_path / ".claude.json").write_text(
            json.dumps({
                "projects": {
                    claude_code_key: {
                        "mcpServers": {
                            "roampal-core": {"env": {"ROAMPAL_PROFILE": "work"}}
                        }
                    }
                }
            }),
            encoding="utf-8",
        )
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "work"

    def test_hook_config_env_from_claude_json_native_key(self, monkeypatch, tmp_path):
        """fix 3 tolerance, both directions: the native separator form
        (backslashes on Windows) is also matched — real files carry both
        styles side by side."""
        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        cwd = tmp_path / "nativeproj"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        (tmp_path / ".claude.json").write_text(
            json.dumps({
                "projects": {
                    str(cwd.resolve()): {
                        "mcpServers": {
                            "roampal-core": {"env": {"ROAMPAL_PROFILE": "work"}}
                        }
                    }
                }
            }),
            encoding="utf-8",
        )
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "work"

    def test_hook_config_env_from_claude_json_case_insensitive_drive(self, monkeypatch, tmp_path):
        """fix 3: Windows drive-letter case varies in the wild
        ('c:/proj' vs 'C:/proj') — matched case-insensitively there."""
        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        cwd = tmp_path / "caseproj"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        key = str(cwd.resolve()).replace("\\", "/")
        if os.name == "nt":
            key = key[0].lower() + key[1:]
        (tmp_path / ".claude.json").write_text(
            json.dumps({
                "projects": {
                    key: {
                        "mcpServers": {
                            "roampal-core": {"env": {"ROAMPAL_PROFILE": "work"}}
                        }
                    }
                }
            }),
            encoding="utf-8",
        )
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "work"

    def test_hook_config_env_from_claude_json_user_scope(self, monkeypatch, tmp_path):
        """(c) fix 3: the USER-scope mcpServers.*.env (root-level
        mcpServers in ~/.claude.json — where init writes the server
        config) reaches the hook, so hooks and MCP tools can't end up on
        different profiles."""
        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        cwd = tmp_path / "userscopeproj"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        (tmp_path / ".claude.json").write_text(
            json.dumps({
                "mcpServers": {
                    "roampal-core": {"env": {"ROAMPAL_PROFILE": "global"}}
                }
            }),
            encoding="utf-8",
        )
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "global"

    def test_hook_config_local_scope_beats_user_scope(self, monkeypatch, tmp_path):
        """fix 3 precedence: per-project local scope overrides user scope."""
        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        cwd = tmp_path / "precedenceproj"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        (tmp_path / ".claude.json").write_text(
            json.dumps({
                "mcpServers": {
                    "roampal-core": {"env": {"ROAMPAL_PROFILE": "global"}}
                },
                "projects": {
                    str(cwd.resolve()).replace("\\", "/"): {
                        "mcpServers": {
                            "roampal-core": {"env": {"ROAMPAL_PROFILE": "local"}}
                        }
                    }
                }
            }),
            encoding="utf-8",
        )
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "local"

    def test_hook_falls_through_to_cwd_binding(self, monkeypatch, tmp_path):
        """With no process env and no MCP-config env, the hook falls through
        to profile_header_value() — cwd binding included."""
        proj = tmp_path / "hookproj"
        proj.mkdir()
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        _seed(
            tmp_path / "roampal" / "profiles.json",
            profiles={"work": None},
            bindings={"bindings": {str(proj): "work"}},
        )
        monkeypatch.setattr(pm, "_registry_path", lambda: tmp_path / "roampal" / "profiles.json")
        monkeypatch.setenv("APPDATA", str(tmp_path))
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.chdir(proj)
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "work"

    def test_hook_process_env_beats_config_env(self, monkeypatch, tmp_path):
        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.setenv("ROAMPAL_PROFILE", "shell-value")
        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"roampal-core": {"env": {"ROAMPAL_PROFILE": "work"}}}}),
            encoding="utf-8",
        )
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "shell-value"

    def test_hook_non_roampal_servers_ignored(self, monkeypatch, tmp_path):
        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"other-tool": {"env": {"ROAMPAL_PROFILE": "wrong"}}}}),
            encoding="utf-8",
        )
        from roampal.hooks import user_prompt_submit_hook as ups

        assert ups._roampal_headers()["X-Roampal-Profile"] == "default"

    def test_stop_hook_seam_is_identical(self, monkeypatch, tmp_path):
        """stop_hook carries the same resolution — one row proving parity."""
        reg = self._isolate(monkeypatch, tmp_path)
        _seed(reg)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        (tmp_path / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"roampal-core": {"env": {"ROAMPAL_PROFILE": "work"}}}}),
            encoding="utf-8",
        )
        from roampal.hooks import stop_hook

        assert stop_hook._roampal_headers()["X-Roampal-Profile"] == "work"


class TestHookContextSeam:
    def test_cmd_context_attaches_profile_header(self, monkeypatch, tmp_path):
        """Plan B row: hook context path — same cwd walk, one test through
        the cmd_context invocation shape."""
        proj = tmp_path / "proj"
        proj.mkdir()
        _seed(
            tmp_path / "roampal" / "profiles.json",
            profiles={"work": None},
            bindings={"bindings": {str(proj): "work"}},
        )
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        monkeypatch.setattr(pm, "_registry_path", lambda: tmp_path / "roampal" / "profiles.json")
        monkeypatch.chdir(proj)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}

        captured = {}
        import roampal.cli.memory_cmds as mc

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["headers"] = headers
            return mock_response

        with patch("httpx.post") as mock_post:
            mock_post.side_effect = fake_post
            cmd_context_sut = getattr(__import__("roampal.cli", fromlist=["cmd_context"]), "cmd_context")
            args = MagicMock()
            args.recent_exchanges = True
            args.port = None
            args.dev = False
            with patch("sys.stdout", StringIO()):
                cmd_context_sut(args)
        assert captured["headers"] == {"X-Roampal-Profile": "work"}

    def test_cmd_context_explicit_default_without_binding(self, monkeypatch, tmp_path):
        """Round 2 Task 18: with no binding and no env, the hook path sends
        an EXPLICIT "default" — the server never resolves a profile for a
        client (F1)."""
        _seed(tmp_path / "roampal" / "profiles.json")
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        monkeypatch.chdir(tmp_path)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        captured = {}

        import roampal.cli.memory_cmds as mc

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["headers"] = headers or {}
            return mock_response

        with patch("httpx.post") as mock_post:
            mock_post.side_effect = fake_post
            cmd_context_sut = getattr(__import__("roampal.cli", fromlist=["cmd_context"]), "cmd_context")
            args = MagicMock()
            args.recent_exchanges = True
            args.port = None
            args.dev = False
            with patch("sys.stdout", StringIO()):
                cmd_context_sut(args)
        assert captured["headers"] == {"X-Roampal-Profile": "default"}

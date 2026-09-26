"""
Tests for _resolve_profile_name (v0.5.4 per-request profile resolver).

Round 2 Item 6 (v0.6.0 Task 17) contract:
  1. X-Roampal-Profile header — clients ALWAYS name their profile
     (explicit "default" included)
  2. profile pinned at launch (start_server(profile=X) — `roampal start
     --profile X`), over the persisted fallback only
  3. persisted `profile use` -> "default"

The server DELIBERATELY never resolves from its own cwd binding or a
spawner's ROAMPAL_PROFILE env (audit finding F1): neither leaks into
headerless resolution here. (Directory-cwd resolution arrives in Task 19
as an explicit X-Roampal-Cwd header — env/binding stay authoritative
client-side via profile_header_value().)
"""

import os
import argparse
import io
import sys
import json
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

import roampal.server.main as srv
from roampal.server.main import _resolve_profile_name


def _request_with_header(value):
    """Build a minimal Request mock with the given X-Roampal-Profile header value."""
    req = MagicMock()
    req.headers = {"X-Roampal-Profile": value} if value is not None else {}
    # MagicMock dict-style get for headers
    req.headers = MagicMock()
    if value is not None:
        req.headers.get = MagicMock(return_value=value)
    else:
        req.headers.get = MagicMock(return_value=None)
    return req


def _restore_pinned():
    srv._STARTUP_PINNED_PROFILE = None


@pytest.fixture(autouse=True)
def _reset_pinned_profile():
    """start_server mutates the module-global pin; tests must not leak it."""
    before = srv._STARTUP_PINNED_PROFILE
    srv._STARTUP_PINNED_PROFILE = None
    yield
    srv._STARTUP_PINNED_PROFILE = before


class TestProfileResolution:
    def test_header_returns_header_value(self):
        """Any non-empty X-Roampal-Profile header wins over pin/file."""
        req = _request_with_header("ghost")
        assert _resolve_profile_name(req) == "ghost"

    def test_header_research(self):
        """Different header value routes to a different profile."""
        req = _request_with_header("research")
        assert _resolve_profile_name(req) == "research"

    def test_no_header_falls_back_to_persisted_use(self):
        """Headerless -> persisted `profile use` (binding-free, env-free)."""
        req = _request_with_header(None)
        with patch(
            "roampal.profile_manager.persisted_profile_fallback",
            return_value="qr",
        ):
            assert _resolve_profile_name(req) == "qr"

    def test_no_header_no_use_returns_string_default(self):
        """No pin, no persisted `use` -> 'default'."""
        req = _request_with_header(None)
        with patch(
            "roampal.profile_manager.persisted_profile_fallback",
            return_value="default",
        ):
            assert _resolve_profile_name(req) == "default"

    def test_header_empty_string_falls_through(self):
        """Whitespace/empty header falls through — no empty profile bucket."""
        for value in ("", "   ", None):
            req = _request_with_header(value)
            with patch(
                "roampal.profile_manager.persisted_profile_fallback",
                return_value="research",
            ):
                assert _resolve_profile_name(req) == "research"

    def test_header_beats_pinned_profile(self):
        """Header beats the launch pin even when both exist."""
        srv._STARTUP_PINNED_PROFILE = "pinned"
        req = _request_with_header("ghost")
        with patch(
            "roampal.profile_manager.persisted_profile_fallback",
            return_value="research",
        ):
            assert _resolve_profile_name(req) == "ghost"

    def test_leaked_env_is_ignored(self, monkeypatch):
        """F1, env half: a spawner's ROAMPAL_PROFILE never resolves here."""
        monkeypatch.setenv("ROAMPAL_PROFILE", "work-a")
        # binding data present too (cwd half) — active_profile_name must
        # not be consulted at all: the resolver imports only the fallback.
        with patch("roampal.profile_manager.active_profile_name") as apn, \
             patch(
                 "roampal.profile_manager.persisted_profile_fallback",
                 return_value="default",
             ):
            req = _request_with_header(None)
            assert _resolve_profile_name(req) == "default"
        apn.assert_not_called()

    def test_persisted_use_beats_pin_headerless(self):
        """F1 launch-pin, round 2: `profile use X` beats `roampal start
        --profile Y` — the server checks the pin last, matching the
        client walk. The pin only applies when use resolves to default."""
        srv._STARTUP_PINNED_PROFILE = "pinned"
        req = _request_with_header(None)
        with patch(
            "roampal.profile_manager.persisted_profile_fallback",
            return_value="other-use",
        ):
            assert _resolve_profile_name(req) == "other-use"
        srv._STARTUP_PINNED_PROFILE = None

    def test_start_server_records_pin_from_flag(self):
        """start_server(profile=X) sets the module pin; bare start clears it."""
        from types import SimpleNamespace

        from unittest.mock import MagicMock, patch as mock_patch

        fake_server = MagicMock()
        fake_server.run = lambda: None
        fake_app = MagicMock()
        fake_app.state = SimpleNamespace()

        with mock_patch.object(srv.uvicorn, "Config", MagicMock()), \
             mock_patch.object(srv.uvicorn, "Server", MagicMock(return_value=fake_server)), \
             mock_patch.object(srv, "create_app", lambda: fake_app), \
             mock_patch.dict(os.environ, {"ROAMPAL_DEV": "0"}, clear=False), \
             mock_patch("roampal.profile_manager.persisted_profile_fallback",
                   return_value="default"), \
             mock_patch("roampal.profile_manager.resolve_data_path",
                   return_value="/tmp/data"):
            srv.start_server(host="127.0.0.1", port=27182, profile="work")
            assert srv._STARTUP_PINNED_PROFILE == "work"
            srv.start_server(host="127.0.0.1", port=27182, profile=None)
            assert srv._STARTUP_PINNED_PROFILE is None
            srv.start_server(host="127.0.0.1", port=27182, profile="   ")
            assert srv._STARTUP_PINNED_PROFILE is None

    def test_persisted_profile_fallback_ignores_env(self, monkeypatch):
        """The fallback helper itself must stay env-free and binding-free."""
        monkeypatch.setenv("ROAMPAL_PROFILE", "ghost")
        with patch("roampal.profile_manager.read_active_profile_file",
                   return_value=None), \
             patch("roampal.profile_manager.binding_for_cwd") as walk:
            from roampal.profile_manager import persisted_profile_fallback
            assert persisted_profile_fallback() == "default"
        walk.assert_not_called()


def _seed_binding_registry(monkeypatch, tmp_path, bindings):
    """Seed a temp registry with only a bindings map; patch its path."""
    import json as _json

    import roampal.profile_manager as pm

    reg_path = tmp_path / "roampal" / "profiles.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    payload.update(bindings)
    reg_path.write_text(_json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
    return reg_path


class TestCwdHeaderResolution:
    """Round 2 Item 6 / Task 19: the OpenCode cwd-header branch.

    Order: profile header (any, incl. "default") > launch pin >
    X-Roampal-Cwd binding > use > default. fix 5 adds the pin as a late
    tier in the CLIENTS' own resolution (hooks/MCP), so a session with
    nothing configured explicitly names the pinned profile.
    """

    def test_cwd_header_resolves_binding(self, monkeypatch, tmp_path):
        _seed_binding_registry(
            monkeypatch, tmp_path,
            {"bindings": {str(tmp_path / "projC"): "proj-c"}},
        )
        srv._STARTUP_PINNED_PROFILE = None
        req = MagicMock()
        req.headers.get = MagicMock(side_effect=lambda k: None if k != "X-Roampal-Cwd" else str(tmp_path / "projC"))
        assert _resolve_profile_name(req) == "proj-c"

    def test_cwd_header_ancestry_innermost_wins(self, monkeypatch, tmp_path):
        _seed_binding_registry(
            monkeypatch, tmp_path,
            {"bindings": {
                str(tmp_path / "outer"): "outer-profile",
                str(tmp_path / "outer" / "inner"): "inner-profile",
            }},
        )
        srv._STARTUP_PINNED_PROFILE = None
        req = MagicMock()
        req.headers.get = MagicMock(return_value=None)
        req.headers.get = MagicMock(side_effect=lambda k: None if k != "X-Roampal-Cwd" else str(tmp_path / "outer" / "inner"))
        assert _resolve_profile_name(req) == "inner-profile"

    def test_cwd_header_unbound_falls_to_use(self, monkeypatch, tmp_path):
        """No binding match -> persisted `use` (not a guess)."""
        _seed_binding_registry(monkeypatch, tmp_path, {})
        srv._STARTUP_PINNED_PROFILE = None
        req = MagicMock()
        req.headers.get = MagicMock(
            side_effect=lambda k: None if k != "X-Roampal-Cwd" else str(tmp_path)
        )
        with patch("roampal.profile_manager.persisted_profile_fallback",
                   return_value="main"):
            assert _resolve_profile_name(req) == "main"

    def test_profile_header_beats_cwd_header(self, monkeypatch, tmp_path):
        _seed_binding_registry(
            monkeypatch, tmp_path,
            {"bindings": {str(tmp_path / "projC"): "proj-c"}},
        )
        srv._STARTUP_PINNED_PROFILE = None
        req = MagicMock()
        values = {"X-Roampal-Profile": "explicit", "X-Roampal-Cwd": str(tmp_path / "projC")}
        req.headers.get = MagicMock(side_effect=lambda k: values.get(k))
        assert _resolve_profile_name(req) == "explicit"

    def test_cwd_binding_beats_pin(self, monkeypatch, tmp_path):
        """Round 2: the server checks the pin LAST, matching the client
        walk — a bound folder beats the pin, so a pinned server never
        steals a directory-bound client (the Claude/OpenCode disagreement
        the reviewer found: CC hook sent the binding, OpenCode's
        cwd-header landed on the pin)."""
        _seed_binding_registry(
            monkeypatch, tmp_path,
            {"bindings": {str(tmp_path / "projC"): "proj-c"}},
        )
        srv._STARTUP_PINNED_PROFILE = "pinned"
        req = MagicMock()
        req.headers.get = MagicMock(side_effect=lambda k: None if k != "X-Roampal-Cwd" else str(tmp_path / "projC"))
        assert _resolve_profile_name(req) == "proj-c"

    def test_persisted_use_beats_pin(self, monkeypatch, tmp_path):
        """Round 2: `profile use X` beats the launch pin — same order the
        clients use (binding > use > pin). The pin governs ONLY sessions
        that configured nothing anywhere."""
        srv._STARTUP_PINNED_PROFILE = "pinned"
        req = MagicMock()
        req.headers.get = MagicMock(return_value=None)
        with patch("roampal.profile_manager.persisted_profile_fallback",
                   return_value="other-use"):
            assert _resolve_profile_name(req) == "other-use"

    def test_pin_is_last_exit_before_default(self, monkeypatch, tmp_path):
        """fix 5: the pin applies exactly when the client walk found
        NOTHING (no binding, no `use`) — the last exit before "default",
        matching the hooks' and MCP's walk (env > config > binding > use >
        pin > default)."""
        _seed_binding_registry(monkeypatch, tmp_path, {})
        srv._STARTUP_PINNED_PROFILE = "pinned"
        req = MagicMock()
        req.headers.get = MagicMock(return_value=None)
        with patch("roampal.profile_manager.persisted_profile_fallback",
                   return_value="default"):
            assert _resolve_profile_name(req) == "pinned"
        srv._STARTUP_PINNED_PROFILE = None

    def test_named_profile_header_beats_pin(self, monkeypatch, tmp_path):
        """fix 5 documentation: an explicitly named profile beats the
        launch pin; the literal "default" header ALSO beats it — which is
        exactly why fix 5 adds the pin tier to the CLIENTS' own resolution
        (hooks/MCP resolve the server's pin file and name the profile
        explicitly instead of sending "default")."""
        _seed_binding_registry(monkeypatch, tmp_path, {})
        srv._STARTUP_PINNED_PROFILE = "pinned"
        for header_value in ("work", "default"):
            req = MagicMock()
            req.headers.get = MagicMock(
                side_effect=lambda k, v=header_value: None if k != "X-Roampal-Profile" else v
            )
            assert _resolve_profile_name(req) == header_value
        srv._STARTUP_PINNED_PROFILE = None

    def test_cwd_header_whitespace_ignored(self, monkeypatch, tmp_path):
        _seed_binding_registry(monkeypatch, tmp_path, {})
        srv._STARTUP_PINNED_PROFILE = None
        req = MagicMock()
        req.headers.get = MagicMock(side_effect=lambda k: None if k != "X-Roampal-Cwd" else "   ")
        with patch("roampal.profile_manager.persisted_profile_fallback",
                   return_value="main"):
            assert _resolve_profile_name(req) == "main"

    def test_cwd_header_percent_encoded_decoded(self, monkeypatch, tmp_path):
        """v0.6.0 review fix 4: the OpenCode plugin percent-encodes the cwd
        header (header values cannot carry characters above Latin-1 —
        Node's fetch throws on raw Cyrillic/CJK paths). The server decodes
        before binding; a percent-encoded non-Latin path resolves."""
        cyrillic_dir = tmp_path / "Проекты" / "проект-альфа"
        _seed_binding_registry(
            monkeypatch, tmp_path,
            {"bindings": {str(cyrillic_dir): "cyrillic-bound"}},
        )
        srv._STARTUP_PINNED_PROFILE = None
        from urllib.parse import quote

        encoded = quote(str(cyrillic_dir))
        assert encoded != str(cyrillic_dir)  # actually encoded, not raw
        req = MagicMock()
        req.headers.get = MagicMock(
            side_effect=lambda k: None if k != "X-Roampal-Cwd" else encoded
        )
        assert _resolve_profile_name(req) == "cyrillic-bound"

    def test_cwd_header_raw_ascii_unchanged_by_unquote(self, monkeypatch, tmp_path):
        """fix 4 back-compat: plain paths (hooks, pre-fix plugin versions)
        pass through unquote byte-identical — no behavior change."""
        _seed_binding_registry(
            monkeypatch, tmp_path,
            {"bindings": {str(tmp_path / "projC"): "proj-c"}},
        )
        srv._STARTUP_PINNED_PROFILE = None
        req = MagicMock()
        req.headers.get = MagicMock(side_effect=lambda k: None if k != "X-Roampal-Cwd" else str(tmp_path / "projC"))
        assert _resolve_profile_name(req) == "proj-c"


class TestServerPinFile:
    """Review fix 5: the launch pin persists across respawn via a per-port
    pin file (profile_manager.read_server_pin), so `roampal start --profile
    X` keeps its routing identity through idle retirement and re-pins."""

    def test_pin_round_trip_and_tolerant_reads(self, tmp_path, monkeypatch):
        import roampal.profile_manager as pm

        monkeypatch.setattr(pm, "_config_dir", lambda: tmp_path / "config")
        assert pm.read_server_pin(27182) is None  # missing file
        pm.write_server_pin(27182, "work")
        assert pm.read_server_pin(27182) == "work"
        pm.clear_server_pin(27182)
        assert pm.read_server_pin(27182) is None
        pm.clear_server_pin(27182)  # idempotent
        # unreadable/garbage counts as unpinned
        (tmp_path / "config").mkdir(exist_ok=True)
        (tmp_path / "config" / "server_pin_27182.txt").write_text("   \n", encoding="utf-8")
        assert pm.read_server_pin(27182) is None

    def test_shutdown_pin_cleanup_explicit_vs_idle(self, tmp_path, monkeypatch):
        """Explicit shutdown clears the pin; idle retirement preserves it
        (the respawn re-passes --profile). The flag always resets."""
        import roampal.profile_manager as pm
        from types import SimpleNamespace

        monkeypatch.setattr(pm, "_config_dir", lambda: tmp_path / "config")
        pm.write_server_pin(27182, "work")
        state = SimpleNamespace(roampal_port=27182)

        srv._retiring_idle = False
        srv._shutdown_pin_cleanup(state)
        assert pm.read_server_pin(27182) is None

        pm.write_server_pin(27182, "work")
        srv._retiring_idle = True
        srv._shutdown_pin_cleanup(state)
        assert pm.read_server_pin(27182) == "work"
        assert srv._retiring_idle is False  # flag resets for next shutdown

    def test_shutdown_pin_cleanup_without_port_is_safe(self, tmp_path, monkeypatch):
        """No port on app.state (test harnesses) -> no-op, no raise."""
        from types import SimpleNamespace

        srv._retiring_idle = False
        srv._shutdown_pin_cleanup(SimpleNamespace())  # must not raise
        assert srv._retiring_idle is False

    def test_pin_file_records_pid_visible_to_show(self, tmp_path, monkeypatch):
        """Round 2: the pin file records the launching pid, and `profile
        show` surfaces the pinned server — a pin whose owner was hard-
        killed used to linger invisibly."""
        import roampal.profile_manager as pm
        from roampal.cli.profile_cmds import cmd_profile
        import argparse
        import os as _os

        monkeypatch.setattr(pm, "_config_dir", lambda: tmp_path / "config")
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        monkeypatch.setattr(pm, "_registry_path", lambda: tmp_path / "config" / "profiles.json")
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.delenv("ROAMPAL_DEV", raising=False)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)

        pm.write_server_pin(27182, "work")
        (tmp_path / "config" / "profiles.json").write_text(json.dumps({}), encoding="utf-8")

        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(argparse.Namespace(profile_command="show", name=None))
        out = buf.getvalue()
        assert code == 0
        assert "roampal start --profile work" in out, out
        assert "roampal stop" in out

"""
Profile bind/unbind CLI command tests (v0.6.0 Item 3, Task 13 acceptance).

Exercises cmd_profile() directly with parser-Namespace-like args against a
temp APPDATA-equivalent registry (no os.chdir — every test passes --path).
Acceptance rows covered:
- bind errors cleanly on an unregistered profile: registry untouched, exit 1
- bind registered: binding lands in the registry, cwd ancestry resolves
- unbind: removes; idempotent-absent -> exit 0, registry unchanged;
  name verification mismatch -> exit 1
- show: reports which binding is active in cwd and why (matchedancestor)
- list: bindings section renders; absent bindings -> legacy bytes
"""

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
)

import argparse
import inspect
import json
import os
from io import StringIO
from unittest.mock import patch

import roampal.profile_manager as pm
from roampal.profile_manager import ProfileRegistry


def _register(path, profiles=None, bindings=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(profiles or {})
    if bindings:
        payload.update(bindings)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _profile_reg_path(tmp_path):
    return tmp_path / "roampal" / "profiles.json"


def _patched_reg(monkeypatch, tmp_path):
    """Point the module-level registry reader at a temp config dir."""
    reg_path = _profile_reg_path(tmp_path)
    monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
    return reg_path


def _run_cmd_profile(monkeypatch, tmp_path, argv_tail):
    """Invoke cmd_profile with an argparse-namespace built by real parser wiring.

    Returns (exit_code, output, reg_path).
    """
    from roampal.cli.profile_cmds import cmd_profile

    reg_path = _patched_reg(monkeypatch, tmp_path)
    # Build args exactly as the CLI parser: parent 'profile' + tail
    args = argparse.Namespace()
    args.profile_command = argv_tail[0]
    if argv_tail[0] == "bind":
        args.name = argv_tail[1] if len(argv_tail) > 1 else None
        args.path = argv_tail[2][0] if len(argv_tail) > 2 else None
    elif argv_tail[0] == "unbind":
        args.name = argv_tail[1] if len(argv_tail) > 1 else None
        args.path = argv_tail[2] if len(argv_tail) > 2 and argv_tail[1] == "--path" else None
    else:
        # generic: name/path only where each sub expects them
        args.name = argv_tail[1] if len(argv_tail) > 1 else None
        args.path = argv_tail[1 + 1] if len(argv_tail) > 2 and argv_tail[1] != "--path" else None

    buf = StringIO()
    with patch("sys.stdout", buf):
        code = cmd_profile(args)
    return code, buf.getvalue(), reg_path


class TestSwitchNeverKillsServer:
    """Round 2 Item 7 / Task 24: `profile switch` == `use` + a note — the
    per-request routing contract makes killing the server pointless AND
    harmful (it interrupted every other session)."""

    def test_switch_persists_profile_without_touching_server(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"work": None, "other": None})
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)

        from roampal.cli.profile_cmds import cmd_profile
        import roampal.cli.server as srv_mod

        args = argparse.Namespace(
            profile_command="switch", name="work", path=None
        )
        buf = StringIO()
        with patch("sys.stdout", buf), \
             patch.object(srv_mod, "_stop_server_on_port") as mock_stop:
            code = cmd_profile(args)
        assert code == 0
        out = buf.getvalue()
        assert "Active profile set to" in out
        assert "left alone" in out
        assert "Stopping any running server" not in out, (
            "switch must not stop the shared server (other sessions uninterrupted)"
        )
        mock_stop.assert_not_called(), "switch must not kill any server"

        # The persisted profile really moved.
        active_file = pm._active_profile_file()
        assert active_file.read_text(encoding="utf-8").strip() == "work"


class TestCliPolishF6:
    """Round 2 F6 polish items (Task 32): the crash + silent-shell list."""

    def test_dir_of_cli_package_works(self):
        """F6: dir(roampal.cli) raised TypeError (stray module-level
        __dir__ called unbound). Now a real class method reporting the
        package mirror plus every group/impl module's names."""
        import roampal.cli as pkg

        listing = dir(pkg)
        assert isinstance(listing, list)
        assert "cmd_context" in listing      # group-module name
        assert "configure_opencode" in listing  # impl name (setup.py)
        assert "GREEN" in listing            # mirrored constant
        assert "cmd_context" in listing

    def test_show_source_label_has_no_task_marker(self, tmp_path, monkeypatch):
        from roampal.cli._common import GREEN

        reg_path = _profile_reg_path(tmp_path)
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        # a persisted-use + a binding in cwd so show resolves binding source
        reg_path.write_text(
            json.dumps({"work": None, "bindings": {str(tmp_path): "work"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)

        from roampal.cli.profile_cmds import cmd_profile

        args = argparse.Namespace(profile_command="show", name=None, path=None)
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        assert "directory binding for this cwd" in buf.getvalue()
        assert "(Task 13)" not in buf.getvalue(), "internal task numbers must not reach users"

    def test_bind_default_allowed(self, tmp_path, monkeypatch):
        """F6: binding literal `default` is valid (default is always
        resolvable) — no create hint, registry gains the binding."""
        code, out, reg_path = _run_cmd_profile(monkeypatch, tmp_path, ["bind", "default"])
        assert code == 0, out
        assert "Bound profile 'default'" in out
        data = json.loads(reg_path.read_text(encoding="utf-8"))
        assert list(data["bindings"].values()) == ["default"]

    def test_list_default_binding_not_marked_unregistered(self, tmp_path, monkeypatch):
        """`profile list` must not annotate a literal-default binding as
        [profile not registered!] — default never lives in the registry."""
        from roampal.cli.profile_cmds import cmd_profile

        reg_path = _profile_reg_path(tmp_path)
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        reg = pm.ProfileRegistry(registry_path=reg_path)
        reg.bind("default", str(tmp_path))

        args = argparse.Namespace(profile_command="list", name=None, path=None)
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        out = buf.getvalue()
        assert "Directory bindings (1):" in out
        assert "[profile not registered!]" not in out

    def test_create_bindings_rejected_registry_untouched(self, tmp_path, monkeypatch):
        """F6: `profile create bindings` was accepted then silently vanished
        on reload — the name is RESERVED now, and nothing is written."""
        code, out, reg_path = _run_cmd_profile(monkeypatch, tmp_path, ["create", "bindings"])
        assert code == 1, out
        assert "reserved" in out
        assert not reg_path.exists(), "rejected create must not touch the registry"

    def test_register_bindings_rejected(self, tmp_path, monkeypatch):
        """register must reject it the same as create."""
        from roampal.cli.profile_cmds import cmd_profile

        reg_path = _profile_reg_path(tmp_path)
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        args = argparse.Namespace(
            profile_command="register", name="bindings", path=str(tmp_path)
        )
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 1
        assert "reserved" in buf.getvalue()
        assert not reg_path.exists()

    def test_legacy_bindings_profile_warns_on_load(self, tmp_path, caplog):
        """F6: pre-reservation registries could carry a REAL profile named
        'bindings' — the reserved key swallowed it on reload. Loader warns
        when the entry has profile-entry shape (path string / None)."""
        reg_path = _profile_reg_path(tmp_path)
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        reg_path.write_text(
            json.dumps({"work": None, "bindings": None}),  # legacy profile shape
            encoding="utf-8",
        )
        import logging

        with caplog.at_level(logging.WARNING, logger="roampal.profile_manager"):
            pm.ProfileRegistry(registry_path=reg_path)
        assert any("legacy" in r.message.lower() and "bindings" in r.message.lower()
                   for r in caplog.records), [r.message for r in caplog.records]

    def test_use_warns_when_cwd_binding_shadows(self, tmp_path, monkeypatch):
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"work": None, "other": None})
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)
        # programmatic bind of cwd to a DIFFERENT profile
        reg = pm.ProfileRegistry(registry_path=reg_path)
        reg.bind("work", str(tmp_path))

        from roampal.cli.profile_cmds import cmd_profile

        args = argparse.Namespace(profile_command="use", name="other", path=None)
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        out = buf.getvalue()
        assert "DIRECTORY-BOUND to 'work'" in out
        assert "roampal profile unbind" in out

    def test_switch_warns_when_cwd_binding_shadows(self, tmp_path, monkeypatch):
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"work": None, "other": None})
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)
        reg = pm.ProfileRegistry(registry_path=reg_path)
        reg.bind("work", str(tmp_path))

        from roampal.cli.profile_cmds import cmd_profile
        import roampal.cli.server as srv_mod

        args = argparse.Namespace(profile_command="switch", name="other", path=None)
        buf = StringIO()
        with patch("sys.stdout", buf), patch.object(srv_mod, "_stop_server_on_port"):
            code = cmd_profile(args)
        assert code == 0
        out = buf.getvalue()
        assert "Active profile set to" in out
        assert "DIRECTORY-BOUND to 'work'" in out

    def test_use_no_warning_without_binding(self, tmp_path, monkeypatch):
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"other": None})
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.chdir(tmp_path)

        from roampal.cli.profile_cmds import cmd_profile

        args = argparse.Namespace(profile_command="use", name="other", path=None)
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        assert "DIRECTORY-BOUND" not in buf.getvalue()
        assert "directory-bound" not in buf.getvalue()

    def test_use_default_also_warns_under_binding(self, tmp_path, monkeypatch):
        """The reverted-to-default path of `use` warns too — the binding
        outranks even the default resolved state in that cwd."""
        from roampal.cli.profile_cmds import cmd_profile

        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"other": None})
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.chdir(tmp_path)
        reg = pm.ProfileRegistry(registry_path=reg_path)
        reg.bind("other", str(tmp_path))

        args = argparse.Namespace(profile_command="use", name="default", path=None)
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        out = buf.getvalue()
        # either branch text ("Reverted ..." or "already default")
        assert "default" in out
        assert "directory-bound to" in out.lower()

    def test_dead_references_removed(self):
        """F6: the binding_for_cwd reg_bindings alias and commands.py's
        stale import_module are gone."""
        pm_source = inspect.getsource(pm.binding_for_cwd)
        assert "reg_bindings" not in pm_source
        from roampal.cli import commands as cmod

        assert not hasattr(cmod, "import_module"), (
            "commands.py carries dead import_module (F6): remove or use it"
        )

    def test_switch_unregistered_exits_1(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"work": None})
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)

        from roampal.cli.profile_cmds import cmd_profile

        args = argparse.Namespace(
            profile_command="switch", name="ghosttown", path=None
        )
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 1
        assert "not registered" in buf.getvalue()


class TestBindUnbindCommands:
    def test_bind_unregistered_exits_1_registry_untouched(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"alpha": None})
        before = json.loads(reg_path.read_text(encoding="utf-8"))

        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        from roampal.cli.profile_cmds import cmd_profile

        args = argparse.Namespace(profile_command="bind", name="ghosttown", path=str(tmp_path / "proj"))
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 1, "(exit = 0 would mean the clean-error row failed)"
        assert "not registered" in buf.getvalue()
        assert json.loads(reg_path.read_text(encoding="utf-8")) == before

    def test_bind_registered_writes_binding(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"work": None})
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)

        from roampal.cli.profile_cmds import cmd_profile

        proj = tmp_path / "proj"
        proj.mkdir()
        args = argparse.Namespace(profile_command="bind", name="work", path=str(proj))
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        out = buf.getvalue()
        assert "Bound profile" in out and "work" in out
        raw = json.loads(reg_path.read_text(encoding="utf-8"))
        assert len(raw["bindings"]) == 1
        bound_name = list(raw["bindings"].values())[0]
        assert bound_name == "work"
        # cwd ancestry resolution picks it up
        reg2 = ProfileRegistry(registry_path=reg_path)
        assert pm.active_profile_name(cwd=proj / "sub", registry=reg2) == "work"

    def test_unbind_removes(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"work": None})
        reg = ProfileRegistry(registry_path=reg_path)
        reg.bind("work", str(tmp_path / "proj"))
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)

        from roampal.cli.profile_cmds import cmd_profile

        buf = StringIO()
        args = argparse.Namespace(
            profile_command="unbind", name=None, path=str(tmp_path / "proj")
        )
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        assert "Unbound" in buf.getvalue()
        snapshot = json.loads(reg_path.read_text(encoding="utf-8"))
        assert "bindings" not in snapshot

    def test_unbind_absent_second_call(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"work": None})
        reg = ProfileRegistry(registry_path=reg_path)
        reg.bind("work", str(tmp_path / "proj"))
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)

        from roampal.cli.profile_cmds import cmd_profile

        args2 = argparse.Namespace(profile_command="unbind", name=None, path=str(tmp_path / "proj"))
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args2)
        assert code == 0
        assert "Unbound" in buf.getvalue()
        snapshot = json.loads(reg_path.read_text(encoding="utf-8"))

        args3 = argparse.Namespace(profile_command="unbind", name=None, path=str(tmp_path / "proj"))
        buf3 = StringIO()
        with patch("sys.stdout", buf3):
            code3 = cmd_profile(args3)
        assert code3 == 0
        assert json.loads(reg_path.read_text(encoding="utf-8")) == snapshot
        assert "No binding" in buf3.getvalue()

    def test_unbind_name_verification_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"work": None})
        reg = ProfileRegistry(registry_path=reg_path)
        reg.bind("work", str(tmp_path / "proj"))
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)

        from roampal.cli.profile_cmds import cmd_profile

        args = argparse.Namespace(
            profile_command="unbind", name="other", path=str(tmp_path / "proj")
        )
        before = json.loads(reg_path.read_text(encoding="utf-8"))
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 1
        assert "bound to" in buf.getvalue()
        assert json.loads(reg_path.read_text(encoding="utf-8")) == before


class TestShowListExtension:
    def test_show_reports_binding_and_why(self, tmp_path, monkeypatch):
        """show gains: binding source label + matched ancestor + why-line."""
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"work": None})
        reg = ProfileRegistry(registry_path=reg_path)
        proj = tmp_path / "proj"
        proj.mkdir()
        reg.bind("work", str(proj))
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)
        # The CLI show path resolves cwd inherently (no injection seam);
        # pytest's chdir monkeypatch auto-restores after the test.
        monkeypatch.chdir(proj)

        from roampal.cli.profile_cmds import cmd_profile

        args = argparse.Namespace(profile_command="show", name=None, path=None)
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        out = buf.getvalue()
        assert "directory binding" in out  # source label
        assert "Bound directory: " in out
        assert "cwd ancestry match" in out  # the why
        assert str(proj).lower() in out.lower()

    def test_show_without_bindings_is_legacy_shaped(self, tmp_path, monkeypatch):
        """No bindings in the registry -> show output has NO binding lines
        (byte-golden compatibility)."""
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"alpha": None})
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        # No persisted active profile on this machine env
        monkeypatch.delenv("ROAMPAL_DATA_PATH", raising=False)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: None)

        from roampal.cli.profile_cmds import cmd_profile

        args = argparse.Namespace(profile_command="show", name=None, path=None)
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        out = buf.getvalue()
        assert "Bound directory" not in out
        assert "directory binding" not in out

    def test_list_shows_bindings_inline(self, tmp_path, monkeypatch):
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"work": None})
        reg = ProfileRegistry(registry_path=reg_path)
        proj = tmp_path / "proj"
        proj.mkdir()
        reg.bind("work", str(proj))
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)

        from roampal.cli.profile_cmds import cmd_profile

        args = argparse.Namespace(profile_command="list", name=None, path=None)
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        out = buf.getvalue()
        assert "Directory bindings (1):" in out
        assert "-> work" in out

    def test_list_no_bindings_legacy_output(self, tmp_path, monkeypatch, capsys):
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"alpha": None})
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)

        from roampal.cli.profile_cmds import cmd_profile

        args = argparse.Namespace(profile_command="list", name=None, path=None)
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        assert "Directory bindings" not in buf.getvalue()

    def test_list_warns_on_unregistered_binding_target(self, tmp_path, monkeypatch):
        """`profile list` warns when a binding points at an unregistered
        profile. v0.6.0 review fix 8: delete() now cascades bindings, so
        this state can only arise from hand-edited/legacy registries —
        seeded directly here instead of via delete."""
        proj = tmp_path / "proj"
        proj.mkdir()
        from roampal.profile_manager import _norm_binding_dir

        key = _norm_binding_dir(str(proj))
        # "work" is NOT registered — dangling binding, seeded directly.
        _register(
            _profile_reg_path(tmp_path),
            profiles={"keeper": None},
            bindings={"bindings": {key: "work"}},
        )
        monkeypatch.setattr(pm, "_registry_path", lambda: _profile_reg_path(tmp_path))
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)

        from roampal.cli.profile_cmds import cmd_profile

        args = argparse.Namespace(profile_command="list", name=None, path=None)
        buf = StringIO()
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        assert "[profile not registered!]" in buf.getvalue()


class TestDeleteReportsWhereUnboundFoldersGo:
    """Smoke-test fix (2026-09-26): `profile delete` told users its unbound
    folders "fall back to the default profile" — wrong whenever a
    `profile use` is set (they resolve to it). It now names each folder's
    actual profile and why."""

    def _delete_work(self, tmp_path, monkeypatch, persisted):
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)
        reg_path = _profile_reg_path(tmp_path)
        _register(reg_path, profiles={"work": None, "main": None})
        proj = tmp_path / "proj"
        proj.mkdir()
        ProfileRegistry(registry_path=reg_path).bind("work", str(proj))
        monkeypatch.setattr(pm, "_registry_path", lambda: reg_path)
        monkeypatch.setattr(pm, "read_active_profile_file", lambda: persisted)

        from roampal.cli.profile_cmds import cmd_profile

        buf = StringIO()
        args = argparse.Namespace(profile_command="delete", name="work", destroy_data=False)
        with patch("sys.stdout", buf):
            code = cmd_profile(args)
        assert code == 0
        return buf.getvalue()

    def test_unbound_folder_reports_profile_use(self, tmp_path, monkeypatch):
        out = self._delete_work(tmp_path, monkeypatch, persisted="main")
        assert "-> main (profile use)" in out
        assert "fall back to the default" not in out

    def test_unbound_folder_reports_default_when_nothing_set(self, tmp_path, monkeypatch):
        out = self._delete_work(tmp_path, monkeypatch, persisted=None)
        assert "-> default (default)" in out

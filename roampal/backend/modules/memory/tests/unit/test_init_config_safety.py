"""Task 40: `roampal init` never wipes or clutters a user's config files.

Covers the rule in the user's words: we never wipe people's settings, ever —
we only adjust Roampal's own entries.

- Unreadable config file → stop, change nothing, explain (byte-identical file,
  no .bak- created); `cmd_init` returns 1 and still configures other tools.
- Encoding: files are read as UTF-8 (BOM tolerated) — the pre-Task-40 locale
  (cp1252) read is what turned non-ASCII files into "corrupt" and wiped them.
- Hooks: for every managed event the user's own hooks survive in place; only
  Roampal-owned commands (`roampal.hooks.` / `roampal context`) are replaced.
- `~/.claude.json`: only mcpServers["roampal-core"] is touched; projects,
  oauthAccount and other servers survive.
- Backups: every overwrite is backed up first, pruned to the newest 3, and
  pruning only ever matches `<exact filename>.bak-<14 digits>`.
- Project .mcp.json: default scope never creates one and never touches one
  without a roampal-core entry; --scope project creates it.

All tests run against a temp HOME (patched pathlib.Path.home, the same
mechanism test_cli.py uses) and a temp cwd, so nothing in this file can touch
the developer's real configs.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _run_in_temp_cwd(tmp_path, monkeypatch):
    """configure_claude_code can write a project .mcp.json into Path.cwd();
    run every test from a temp dir (same fixture pattern as test_cli.py)."""
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)


@pytest.fixture(autouse=True)
def _fake_home(tmp_path, monkeypatch):
    """Redirect Path.home() to tmp_path for the whole test (same global-class
    patch mechanism test_cli.py relies on), and clear XDG_CONFIG_HOME so
    OpenCode paths stay inside tmp_path on Linux CI too."""
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    with patch("roampal.cli.Path.home", return_value=tmp_path):
        yield tmp_path


def _configure_claude(tmp_path, scope=None, **kwargs):
    """Run configure_claude_code against the fake home; print stays live so
    capsys can assert on the abort message."""
    from roampal.cli import configure_claude_code

    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(exist_ok=True)
    with patch("roampal.cli.validate_roampal_importable", return_value=True):
        return configure_claude_code(claude_dir, is_dev=False, scope=scope, **kwargs)


def _roampal_owned(hook_list):
    """Count hook entries whose command is Roampal-owned (the Task 40
    predicate: contains `roampal.hooks.` or `roampal context`)."""
    return [
        h
        for h in hook_list
        if isinstance(h, dict)
        and isinstance(h.get("command"), str)
        and ("roampal.hooks." in h["command"] or "roampal context" in h["command"])
    ]


# ============================================================================
# Rule B: unreadable file → stop, change nothing, explain
# ============================================================================


class TestUnreadableFileAborts:
    def test_unreadable_claude_json_byte_identical_no_backup(self, tmp_path, capsys):
        """~/.claude.json with invalid JSON is left byte-identical; the abort
        message is printed; no .bak- is created for it."""
        claude_json_path = tmp_path / ".claude.json"
        broken = '{ "mcpServers": { "roampal-core": {'
        claude_json_path.write_text(broken, encoding="utf-8")

        result = _configure_claude(tmp_path)

        assert result is False  # tool skipped
        assert claude_json_path.read_text(encoding="utf-8") == broken
        assert not list(tmp_path.glob(".claude.json.bak-*"))
        out = capsys.readouterr().out
        assert str(claude_json_path) in out
        assert "Roampal did not change this file" in out
        assert "roampal init" in out
        # The tool is skipped entirely: settings.json was not created either.
        assert not (tmp_path / ".claude" / "settings.json").exists()

    def test_unreadable_settings_json_aborts_before_any_write(self, tmp_path, capsys):
        """Corrupt ~/.claude/settings.json → nothing is written anywhere:
        no fresh settings.json overwrite, no ~/.claude.json either."""
        settings_path = tmp_path / ".claude" / "settings.json"
        broken = "{ broken hooks: ["
        settings_path.parent.mkdir(exist_ok=True)
        settings_path.write_text(broken, encoding="utf-8")

        result = _configure_claude(tmp_path)

        assert result is False
        assert settings_path.read_text(encoding="utf-8") == broken
        assert not list((tmp_path / ".claude").glob("settings.json.bak-*"))
        assert not (tmp_path / ".claude.json").exists()
        out = capsys.readouterr().out
        assert "Roampal did not change this file" in out

    def test_non_object_json_is_not_a_config(self, tmp_path, capsys):
        """A valid-JSON-but-not-an-object file ([] or a string) is not a
        config we can merge into — same abort, byte-identical."""
        claude_json_path = tmp_path / ".claude.json"
        claude_json_path.write_text("[]", encoding="utf-8")

        result = _configure_claude(tmp_path)

        assert result is False
        assert claude_json_path.read_text(encoding="utf-8") == "[]"
        out = capsys.readouterr().out
        assert "not a JSON object" in out

    def test_valid_utf8_with_nonascii_parses_and_is_kept(self, tmp_path):
        """The encoding fix: a VALID UTF-8 file with non-ASCII content must
        parse (the old cp1252 read made it look corrupt and then wiped it)."""
        claude_json_path = tmp_path / ".claude.json"
        data = {
            "mcpServers": {"other-server": {"command": "x"}},
            "env": {"NOTE": "café ümlaut — déjà vu"},
        }
        claude_json_path.write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

        result = _configure_claude(tmp_path)

        assert result is True
        after = json.loads(claude_json_path.read_text(encoding="utf-8"))
        assert after["env"]["NOTE"] == "café ümlaut — déjà vu"
        assert "other-server" in after["mcpServers"]
        assert "roampal-core" in after["mcpServers"]

    def test_utf8_bom_file_parses(self, tmp_path):
        """A BOM-prefixed config (Windows editors) must parse, not abort."""
        claude_json_path = tmp_path / ".claude.json"
        data = {"mcpServers": {"other-server": {"command": "x"}}}
        claude_json_path.write_bytes(
            json.dumps(data, indent=2).encode("utf-8-sig")
        )

        result = _configure_claude(tmp_path)

        assert result is True
        after = json.loads(claude_json_path.read_text(encoding="utf-8-sig"))
        assert "roampal-core" in after["mcpServers"]
        assert "other-server" in after["mcpServers"]


# ============================================================================
# Rule C: only Roampal's own entries change
# ============================================================================


class TestUserHooksSurvive:
    def _seed_settings(self, tmp_path):
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(exist_ok=True)
        settings = {
            "hooks": {
                # The user's own Stop hook
                "Stop": [
                    {
                        "hooks": [
                            {"type": "command", "command": "echo my-own-stop-hook"}
                        ]
                    }
                ],
                # A mixed group: user command + an OLD roampal command
                # (pre-0.6.0 `-I` launch form)
                "UserPromptSubmit": [
                    {
                        "hooks": [
                            {"type": "command", "command": "echo user-submit-hook"},
                            {
                                "type": "command",
                                "command": "python -I -m roampal.hooks.user_prompt_submit_hook",
                            },
                        ]
                    }
                ],
                # An event Roampal does not manage — must stay untouched
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "echo user-pre-hook"}]}
                ],
            }
        }
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        return settings_path, settings

    def test_user_hook_survives_and_old_roampal_hook_replaced(self, tmp_path):
        settings_path, _ = self._seed_settings(tmp_path)

        result = _configure_claude(tmp_path)
        assert result is True

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        hooks = settings["hooks"]

        # The user's own Stop hook survives, in place, before Roampal's group.
        stop = hooks["Stop"]
        assert stop[0]["hooks"][0]["command"] == "echo my-own-stop-hook"
        owned = _roampal_owned(
            [h for group in stop for h in group.get("hooks", [])]
        )
        assert len(owned) == 1  # exactly one fresh Roampal command

        # The mixed group keeps the user's command; the old Roampal one is
        # gone from inside it; Roampal's fresh group was appended.
        submit = hooks["UserPromptSubmit"]
        assert submit[0]["hooks"][0]["command"] == "echo user-submit-hook"
        owned = _roampal_owned(
            [h for group in submit for h in group.get("hooks", [])]
        )
        assert len(owned) == 1
        assert "-I" not in owned[0]["command"]

        # SessionStart: exactly the three matcher groups Roampal manages.
        session = hooks["SessionStart"]
        matchers = [g.get("matcher") for g in session]
        assert matchers == ["compact", "startup", "clear"]
        for group in session:
            assert len(_roampal_owned(group["hooks"])) == 1

        # An event Roampal never manages is untouched.
        assert hooks["PreToolUse"] == [
            {"hooks": [{"type": "command", "command": "echo user-pre-hook"}]}
        ]

    def test_group_becomes_empty_when_only_roampal_commands(self, tmp_path):
        """A group holding ONLY an old Roampal command is dropped, not left
        as an empty husk."""
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(exist_ok=True)
        settings = {
            "hooks": {
                "Stop": [
                    {
                        "matcher": "mine",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python -I -m roampal.hooks.stop_hook",
                            }
                        ],
                    }
                ]
            }
        }
        settings_path.write_text(json.dumps(settings), encoding="utf-8")

        _configure_claude(tmp_path)

        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        stop_groups = settings["hooks"]["Stop"]
        # The all-Roampal group is gone; the fresh Roampal group is there.
        assert len(stop_groups) == 1
        assert "matcher" not in stop_groups[0]
        assert len(_roampal_owned(stop_groups[0]["hooks"])) == 1


class TestClaudeJsonOnlyRoampalEntryChanges:
    def test_other_servers_and_unrelated_keys_survive(self, tmp_path):
        claude_json_path = tmp_path / ".claude.json"
        claude_json_path.write_text(
            json.dumps(
                {
                    "mcpServers": {"other-server": {"command": "x", "args": ["a"]}},
                    "projects": {"/some/project": {"allowedTools": ["Bash"]}},
                    "oauthAccount": {"emailAddress": "user@example.com"},
                    "numStartups": 7,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        assert _configure_claude(tmp_path) is True

        after = json.loads(claude_json_path.read_text(encoding="utf-8"))
        assert after["mcpServers"]["other-server"] == {"command": "x", "args": ["a"]}
        assert after["projects"] == {"/some/project": {"allowedTools": ["Bash"]}}
        assert after["oauthAccount"] == {"emailAddress": "user@example.com"}
        assert after["numStartups"] == 7
        assert "roampal-core" in after["mcpServers"]


# ============================================================================
# Rule A: backups — newest 3 kept, nothing else ever pruned
# ============================================================================


class TestBackupsAndIdempotency:
    def test_init_twice_identical_bytes_and_pruned_backups(self, tmp_path):
        claude_json_path = tmp_path / ".claude.json"
        settings_path = tmp_path / ".claude" / "settings.json"

        # Seed user content + decoy files that pruning must never touch.
        settings_path.parent.mkdir(exist_ok=True)
        settings_path.write_text(
            json.dumps(
                {"hooks": {"Stop": [
                    {"hooks": [{"type": "command", "command": "echo my-own-stop-hook"}]}
                ]}}
            ),
            encoding="utf-8",
        )
        claude_json_path.write_text(
            json.dumps({"mcpServers": {"other-server": {"command": "x"}}}),
            encoding="utf-8",
        )
        for i in range(1, 7):  # six stale backups from the old unbounded era
            (tmp_path / ".claude" / f"settings.json.bak-2020010100000{i}").write_text(
                "old", encoding="utf-8"
            )
        (tmp_path / ".claude" / "settings.json.bak-manual").write_text(
            "manual", encoding="utf-8"
        )
        (tmp_path / ".claude" / "notes.txt").write_text("keep me", encoding="utf-8")

        assert _configure_claude(tmp_path) is True
        first_settings = settings_path.read_bytes()
        first_claude = claude_json_path.read_bytes()

        assert _configure_claude(tmp_path) is True
        assert settings_path.read_bytes() == first_settings
        assert claude_json_path.read_bytes() == first_claude

        backups = []
        backup_prefix = "settings.json.bak-"
        for p in (tmp_path / ".claude").iterdir():
            if p.name.startswith(backup_prefix):
                stamped = p.name[len(backup_prefix):]
                if len(stamped) == 14 and stamped.isdigit():
                    backups.append(p)
        assert len(backups) <= 3
        # Pruning never deletes a non-matching file.
        assert (tmp_path / ".claude" / "settings.json.bak-manual").read_text(
            encoding="utf-8"
        ) == "manual"
        assert (tmp_path / ".claude" / "notes.txt").read_text(
            encoding="utf-8"
        ) == "keep me"
        # The user's own hook is still there after two runs.
        hooks = json.loads(settings_path.read_text(encoding="utf-8"))["hooks"]
        assert hooks["Stop"][0]["hooks"][0]["command"] == "echo my-own-stop-hook"


# ============================================================================
# Rule D: project .mcp.json only when asked for
# ============================================================================


class TestProjectMcpJsonScope:
    def test_default_scope_does_not_create(self, tmp_path):
        result = _configure_claude(tmp_path, scope=None)
        assert result is True
        assert not (Path.cwd() / ".mcp.json").exists()

    def test_default_scope_upgrades_existing_roampal_entry(self, tmp_path):
        cwd_mcp = Path.cwd() / ".mcp.json"
        cwd_mcp.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "other-server": {"command": "other"},
                        "roampal-core": {"stale": True},
                    }
                }
            ),
            encoding="utf-8",
        )

        assert _configure_claude(tmp_path, scope=None) is True

        updated = json.loads(cwd_mcp.read_text(encoding="utf-8"))
        assert "stale" not in updated["mcpServers"]["roampal-core"]
        assert "command" in updated["mcpServers"]["roampal-core"]
        assert "other-server" in updated["mcpServers"]

    def test_default_scope_leaves_foreign_file_alone(self, tmp_path):
        cwd_mcp = Path.cwd() / ".mcp.json"
        content = '{"mcpServers": {"mine": {"command": "x"}}}'
        cwd_mcp.write_text(content, encoding="utf-8")

        assert _configure_claude(tmp_path, scope=None) is True
        assert cwd_mcp.read_text(encoding="utf-8") == content

    def test_project_scope_creates(self, tmp_path):
        assert _configure_claude(tmp_path, scope="project") is True
        created = json.loads((Path.cwd() / ".mcp.json").read_text(encoding="utf-8"))
        assert "roampal-core" in created["mcpServers"]

    def test_user_scope_ignores_corrupt_project_file(self, tmp_path):
        """--scope user never touches the project .mcp.json, so a corrupt one
        there must not skip the tool (regression for the read-phase guard)."""
        corrupt = Path.cwd() / ".mcp.json"
        corrupt.write_text("{ not json", encoding="utf-8")

        result = _configure_claude(tmp_path, scope="user")

        assert result is True
        assert corrupt.read_text(encoding="utf-8") == "{ not json"
        assert "roampal-core" in json.loads(
            (tmp_path / ".claude.json").read_text(encoding="utf-8")
        )["mcpServers"]


# ============================================================================
# Latent crashes on hand-broken files
# ============================================================================


class TestMalformedShapes:
    def test_non_dict_mcp_servers_does_not_crash(self, tmp_path):
        """~/.claude.json with `mcpServers: []` must not crash init (the old
        code AttributeError'd on .get); roampal-core is added, the rest of the
        file survives."""
        claude_json_path = tmp_path / ".claude.json"
        claude_json_path.write_text(
            json.dumps({"mcpServers": [], "numStartups": 3}), encoding="utf-8"
        )

        result = _configure_claude(tmp_path)

        assert result is True
        after = json.loads(claude_json_path.read_text(encoding="utf-8"))
        assert after["numStartups"] == 3
        assert "roampal-core" in after["mcpServers"]

    def test_import_validation_failure_is_not_a_config_skip(self, tmp_path):
        """roampal not importable (broken install) is NOT 'a config file could
        not be read' — it must not flip cmd_init's exit code or message
        (regression: the path used to bare-return None into the skip logic)."""
        claude_json_path = tmp_path / ".claude.json"
        claude_json_path.write_text(
            json.dumps({"mcpServers": {}}), encoding="utf-8"
        )

        from roampal.cli import configure_claude_code

        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir(exist_ok=True)
        with patch("roampal.cli.validate_roampal_importable", return_value=False):
            result = configure_claude_code(claude_dir, is_dev=False, scope="user")

        assert result is True
        # No MCP entry written (validation gate held), file untouched.
        after = json.loads(claude_json_path.read_text(encoding="utf-8"))
        assert "roampal-core" not in after["mcpServers"]


# ============================================================================
# Cursor
# ============================================================================


class TestCursorUserHooksSurvive:
    def test_user_stop_hook_survives(self, tmp_path):
        from roampal.cli import configure_cursor

        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir(exist_ok=True)
        hooks_path = cursor_dir / "hooks.json"
        hooks_path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "hooks": {
                        "stop": [{"command": "echo my-own-cursor-stop"}],
                        # An event Roampal does not manage
                        "afterSubmitPrompt": [{"command": "echo other-event"}],
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch("builtins.print"):
            result = configure_cursor(cursor_dir, is_dev=False)

        assert result is True
        hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
        assert hooks["version"] == 2  # user's version value is not clobbered
        stop_entries = hooks["hooks"]["stop"]
        commands = [e["command"] for e in stop_entries if "command" in e]
        assert "echo my-own-cursor-stop" in commands
        assert len(_roampal_owned(stop_entries)) == 1
        assert hooks["hooks"]["afterSubmitPrompt"] == [
            {"command": "echo other-event"}
        ]
        assert len(_roampal_owned(hooks["hooks"]["beforeSubmitPrompt"])) == 1

    def test_corrupt_mcp_json_aborts_and_keeps_hooks_untouched(self, tmp_path, capsys):
        from roampal.cli import configure_cursor

        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir(exist_ok=True)
        mcp_path = cursor_dir / "mcp.json"
        broken = "{ mcpServers: ["
        mcp_path.write_text(broken, encoding="utf-8")
        hooks_path = cursor_dir / "hooks.json"
        hooks_path.write_text(
            json.dumps({"version": 1, "hooks": {"stop": [{"command": "echo mine"}]}}),
            encoding="utf-8",
        )

        result = configure_cursor(cursor_dir, is_dev=False)

        assert result is False
        assert mcp_path.read_text(encoding="utf-8") == broken
        # Rule B: the tool is skipped — hooks.json not touched either.
        hooks = json.loads(hooks_path.read_text(encoding="utf-8"))
        assert hooks["hooks"]["stop"] == [{"command": "echo mine"}]
        out = capsys.readouterr().out
        assert "Roampal did not change this file" in out


# ============================================================================
# cmd_init: exit 1 on a skipped tool, others still configured
# ============================================================================


class TestCmdInitSkipExit:
    def test_corrupt_cursor_file_returns_1_and_configures_claude(self, tmp_path):
        from roampal.cli import cmd_init

        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir(exist_ok=True)
        mcp_path = cursor_dir / "mcp.json"
        broken = "{ not json"
        mcp_path.write_text(broken, encoding="utf-8")

        claude_json_path = tmp_path / ".claude.json"

        args = type("Args", (), {})()
        args.claude_code = True
        args.cursor = True
        args.opencode = False
        args.dev = False
        args.force = False
        args.scope = None

        with patch("roampal.cli.validate_roampal_importable", return_value=True), \
             patch("roampal.cli.get_data_dir", return_value=tmp_path / "data"), \
             patch("roampal.cli.collect_email"), \
             patch("roampal.cli.print_banner"), \
             patch("roampal.cli.print_update_notice"), \
             patch("builtins.print"):
            exit_code = cmd_init(args)

        assert exit_code == 1
        # Cursor's corrupt file is untouched...
        assert mcp_path.read_text(encoding="utf-8") == broken
        # ...while Claude Code was still configured.
        configured = json.loads(claude_json_path.read_text(encoding="utf-8"))
        assert "roampal-core" in configured["mcpServers"]

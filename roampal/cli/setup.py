"""Setup group (v0.6.0 Task 5): init + per-tool configuration.

Moved verbatim from the pre-refactor roampal/cli.py as part of the Task 5
split (cmd_init, validate_roampal_importable, configure_*,
_install_plugin_file, _verify_plugin_install_targets, _build_hook_command).
Cross-impl deps on helpers relocated by Tasks 5-9: is_dev_mode and
get_data_dir stay in the impl; _prompt_smart_onboarding,
_find_project_opencode_config, _safe_write_opencode_config live in
roampal.cli.sidecar and _get_opencode_config_path in roampal.cli._common
(G3 fan-in votes) since Task 9. Module-level imports are safe: the package
__init__ resolves the whole chain before dispatch (no cycles - sidecar
imports nothing from here).
"""

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from typing import List
import typing  # noqa: F401  (typing.List forward-ref used by moved verify helper)
from pathlib import Path

# Task 40: every config file init touches is read/written through these —
# a parse error must ABORT (leaving the file byte-identical), never "start
# fresh", and every overwrite is backed up first.
from roampal.utils.safe_config import (
    ConfigReadError,
    read_json_config,
    write_json_config,
)

from roampal.cli._common import BLUE, BOLD, GREEN, RED, RESET, YELLOW, logger
from roampal.cli.update_check import (
    print_banner,
    print_update_notice,
    collect_email,
)

# Cross-group dependencies after Tasks 9-10: the onboarding/config helpers
# the sidecar task owns come from roampal.cli.sidecar; the config-path
# predicate and the mode/port/data helpers came to _common with the G3
# decision and the Task 10 helper relocation.
# Each is safe as a module-level import (see docstring note).
from roampal.cli.sidecar import (
    _prompt_smart_onboarding,
    configured_sidecar_for_scope,
    _find_project_opencode_config,
    _safe_write_opencode_config,
)
from roampal.cli._common import (  # noqa: F401 (get_port: configure wiring)
    _is_interactive,
    get_port,
    _get_opencode_config_path,
    is_dev_mode,
    get_data_dir,
)


def _impl():
    import importlib

    return importlib.import_module("roampal.cli._monolith_impl")


def _build_hook_command(module: str, is_dev: bool) -> str:
    """
    Build hook command with proper ROAMPAL_DEV env var for any platform.

    v0.3.2: Centralized hook command builder ensures all platforms handle
    dev mode consistently. Previously Claude Code forgot to wrap commands
    while Cursor did - this function is the single source of truth.

    Args:
        module: Hook module name (e.g., 'user_prompt_submit_hook', 'stop_hook')
        is_dev: If True, wraps command to set ROAMPAL_DEV=1 env var

    Returns:
        Command string ready for hook config
    """
    # Forward slashes avoid bash escape mangling on Windows (Claude Code 2.1.x)
    # C:\roampal-core\.venv\Scripts\python.exe → C:/roampal-core/.venv/Scripts/python.exe
    python_exe = sys.executable.replace("\\", "/")
    # Task 36: keep the project dir (where Claude Code / OpenCode run hooks)
    # off the hook process's sys.path, so a project containing a local
    # `roampal/` directory cannot shadow the installed package.
    # v0.6.0 review fix 1: -E [-P] instead of -I — -I's implied -s hides USER
    # site-packages, breaking Store-Python / pip --user installs.
    flags = " ".join(_spawn_isolation_flags())
    base = f"{python_exe} {flags} -m roampal.hooks.{module}"

    if not is_dev:
        return base

    # Wrap command to set env var - hooks run as subprocesses that don't
    # inherit MCP's env block, so we must set the var explicitly
    if sys.platform == "win32":
        return f'cmd /c "set ROAMPAL_DEV=1 && {base}"'
    else:
        return f"ROAMPAL_DEV=1 {base}"


def cmd_init(args):
    """Initialize Roampal for the current environment."""
    print_banner()
    print_update_notice()
    print(f"{BOLD}Initializing Roampal...{RESET}\n")

    # Detect environment
    home = Path.home()
    claude_code_dir = home / ".claude"
    cursor_dir = home / ".cursor"

    # OpenCode config location (XDG Base Directory spec)
    if sys.platform == "win32":
        opencode_dir = home / ".config" / "opencode"
    else:
        xdg_config = os.environ.get("XDG_CONFIG_HOME", str(home / ".config"))
        opencode_dir = Path(xdg_config) / "opencode"

    # Check for explicit flags (--claude-code, --cursor, --opencode)
    explicit_claude = getattr(args, "claude_code", False)
    explicit_cursor = getattr(args, "cursor", False)
    explicit_opencode = getattr(args, "opencode", False)

    if explicit_claude or explicit_cursor or explicit_opencode:
        # User specified explicit tools - use those
        detected = []
        if explicit_claude:
            if claude_code_dir.exists():
                detected.append("claude-code")
            else:
                # Create the directory if it doesn't exist
                claude_code_dir.mkdir(parents=True, exist_ok=True)
                detected.append("claude-code")
                print(f"{YELLOW}Created ~/.claude directory{RESET}")
        if explicit_cursor:
            if cursor_dir.exists():
                detected.append("cursor")
            else:
                cursor_dir.mkdir(parents=True, exist_ok=True)
                detected.append("cursor")
                print(f"{YELLOW}Created ~/.cursor directory{RESET}")
        if explicit_opencode:
            if opencode_dir.exists():
                detected.append("opencode")
            else:
                opencode_dir.mkdir(parents=True, exist_ok=True)
                detected.append("opencode")
                print(f"{YELLOW}Created {opencode_dir} directory{RESET}")
    else:
        # Auto-detect installed tools
        # Check config dirs, PATH binaries, AND platform-specific install locations
        # (fresh installs may not have config dirs yet)
        detected = []

        # Claude Code: config dir OR binary in PATH
        if claude_code_dir.exists() or shutil.which("claude"):
            detected.append("claude-code")

        # Cursor: config dir OR binary in PATH OR platform-specific install
        cursor_found = cursor_dir.exists() or shutil.which("cursor")
        if not cursor_found:
            if sys.platform == "darwin":
                cursor_found = Path("/Applications/Cursor.app").exists()
            elif sys.platform == "win32":
                localappdata = os.environ.get("LOCALAPPDATA", "")
                if (
                    localappdata
                    and (Path(localappdata) / "Programs" / "cursor").exists()
                ):
                    cursor_found = True
        if cursor_found:
            detected.append("cursor")

        # OpenCode: config dir OR binary in PATH OR platform-specific install
        opencode_found = opencode_dir.exists() or shutil.which("opencode")
        if not opencode_found:
            if sys.platform == "win32":
                # Windows: Electron installer puts files in LOCALAPPDATA
                for env_var in ["LOCALAPPDATA", "APPDATA"]:
                    d = os.environ.get(env_var, "")
                    if d and (Path(d) / "opencode").exists():
                        opencode_found = True
                        break
            elif sys.platform == "darwin":
                # macOS: .app bundle or Homebrew
                opencode_found = Path("/Applications/OpenCode.app").exists()
        if opencode_found:
            detected.append("opencode")

    if not detected:
        print(f"{YELLOW}No AI coding tools detected.{RESET}")
        print("Roampal works with:")
        print("  - Claude Code (https://claude.com/claude-code)")
        print("  - Cursor (https://cursor.sh)")
        print("  - OpenCode (https://opencode.ai)")
        print(
            "\nInstall one of these tools first, or use --claude-code / --cursor / --opencode to force setup."
        )
        return 1

    print(f"{GREEN}Configuring: {', '.join(detected)}{RESET}\n")

    # v0.5.3 Section 8: Get scope parameter for config placement
    init_scope = getattr(args, "scope", None)

    # Configure each detected tool
    # Task 40: a tool whose config file cannot be read is skipped entirely
    # (nothing written, message printed) and `cmd_init` exits 1.
    is_dev = is_dev_mode(args)
    force = getattr(args, "force", False)
    skipped_tools = []
    for tool in detected:
        if tool == "claude-code":
            configured = configure_claude_code(
                claude_code_dir, is_dev=is_dev, force=force, scope=init_scope
            )
        elif tool == "cursor":
            configured = configure_cursor(cursor_dir, is_dev=is_dev, force=force)
        elif tool == "opencode":
            configured = configure_opencode(is_dev=is_dev, force=force, scope=init_scope)
        else:
            configured = True
        if not configured:
            skipped_tools.append(tool)

    if skipped_tools:
        print(
            f"\n{RED}Skipped: {', '.join(skipped_tools)} — a config file could not be read.{RESET}"
        )
        print(
            f"  {YELLOW}Roampal did not change that file. Fix the JSON (or restore a "
            f".bak- copy next to it) and run `roampal init` again.{RESET}"
        )
        return 1

    # Sidecar setup for OpenCode users (v0.5.3: scope-aware config path)
    if "opencode" in detected:
        # --no-input / non-TTY: nobody can answer the model picker, so skip it
        # (it used to print the menu and block on input()).
        current = configured_sidecar_for_scope(init_scope)
        if _is_interactive():
            _prompt_smart_onboarding(force=force, scope=init_scope)
        elif current:
            # Task 44: nothing to choose — say what stays in place.
            print(
                f"\n{BOLD}Memory scoring:{RESET} keeping {current}. "
                f"Change it any time: {BLUE}roampal sidecar setup{RESET}"
            )
        else:
            print(
                f"\n{BOLD}Memory scoring setup skipped{RESET} (non-interactive). "
                f"Run {BLUE}roampal sidecar setup{RESET} to choose a scoring model."
            )
            print(
                f"  Without a menu: {BLUE}roampal sidecar setup --list{RESET} shows every "
                f"choice and the one command that selects it. Scoring stays off (no "
                f"exchange text leaves this machine) until one is chosen."
            )

    # Create data directory
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"{GREEN}Created data directory: {data_dir}{RESET}")

    # Build next steps based on configured tools
    next_steps = []
    if "claude-code" in detected:
        next_steps.append(f"  {BLUE}Restart Claude Code{RESET} and start chatting!")
    if "cursor" in detected:
        next_steps.append(f"  {BLUE}Restart Cursor{RESET} and start chatting!")
    if "opencode" in detected:
        next_steps.append(
            f"  Run {BLUE}opencode{RESET} and start chatting! (server auto-starts on first message)"
        )

    print(f"\n{GREEN}{BOLD}Roampal initialized successfully!{RESET}\n")

    # Offer email signup (optional, non-blocking)
    # --force resets the email marker so user gets re-prompted
    if force:
        email_marker = get_data_dir() / ".email_asked"
        if email_marker.exists():
            try:
                email_marker.unlink()
            except Exception:
                pass
    collect_email(detected)

    print(f"""{BOLD}Next steps:{RESET}
{chr(10).join(next_steps)}

{BOLD}How it works:{RESET}
  - Relevant memories are injected into your AI's context automatically
  - The AI learns what works and what doesn't via outcome scoring
  - You type normally; the AI sees your message + relevant context from past sessions

{BOLD}Optional commands:{RESET}
  - {BLUE}roampal ingest myfile.pdf{RESET} - Add documents to memory
  - {BLUE}roampal stats{RESET} - Show memory statistics
  - {BLUE}roampal status{RESET} - Check server status""")

    if "opencode" in detected:
        print(f"""  - {BLUE}roampal sidecar status{RESET}   - Check scoring model configuration
  - {BLUE}roampal sidecar setup{RESET}    - Change scoring model""")

    print(f"""

{BOLD}Feedback & Support:{RESET}
  - Discord: https://discord.com/invite/F87za86R3v
  - Issues:  https://github.com/roampal-ai/roampal-core/issues
""")


def _spawn_isolation_flags() -> list:
    """v0.6.0 review fix 1: flags for every command init writes.

    ``-E`` ignores PYTHONPATH/PYTHONHOME (cwd shadowing is handled by -P or
    the spawn cwd). ``-P`` (3.11+ only) keeps the project cwd — where Claude
    Code / Cursor launch hooks and MCP stdio servers — off sys.path, so a
    local ``roampal/`` directory cannot shadow the installed package.

    -I is NOT used: its implied -s hides USER site-packages, silently
    breaking Microsoft Store Python (always user site), ``pip install
    --user``, and non-writable system site-packages. The flags are chosen
    from the interpreter running init — the same one the written commands
    execute — so the -P guard is version-exact.
    """
    from roampal.profile_manager import spawn_isolation_flags

    return spawn_isolation_flags()


def _mcp_args() -> list:
    """The args list init writes for the roampal MCP stdio server."""
    return [*_spawn_isolation_flags(), "-m", "roampal.mcp.server"]


def validate_roampal_importable(python_exe: str) -> bool:
    """Validate that roampal can be imported from the given python executable.

    Task 36: validates under the SAME conditions the spawned processes run
    in: site-packages must be the provider, not the caller's cwd or
    PYTHONPATH. Otherwise init could bless a setup whose MCP/hook commands
    then fail at runtime. v0.6.0 review fix 1: validates under
    _spawn_isolation_flags() (no -I), so user-site installs validate — and
    the written commands import — correctly.
    """
    try:
        result = subprocess.run(
            [python_exe, *_spawn_isolation_flags(), "-c", "import roampal"],
            capture_output=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def _print_config_read_error(exc: ConfigReadError) -> None:
    """Task 40, rule B: unreadable config file → explain, change nothing."""
    print(f"  {RED}[ERROR] Cannot read {exc.path}:{RESET}")
    print(f"    {exc.reason}")
    print(
        f"  {YELLOW}Roampal did not change this file. Fix the JSON (or restore a "
        f".bak- copy next to it) and run `roampal init` again.{RESET}"
    )


def _is_roampal_hook_command(command) -> bool:
    """True when a hook command is Roampal-owned (Task 40, rule C).

    Matches the current `-E -P -m roampal.hooks.*` form, the older `-I` and
    bare `-m` forms, the `ROAMPAL_DEV=1 ...` / `cmd /c "set ROAMPAL_DEV=1 ..."`
    prefixes, and the `roampal context --recent-exchanges` SessionStart command.
    """
    if not isinstance(command, str):
        return False
    return "roampal.hooks." in command or "roampal context" in command


def _strip_roampal_hook_commands(entries: list) -> list:
    """Keep every non-Roampal entry, in place; drop only Roampal-owned ones."""
    return [
        entry
        for entry in entries
        if not (
            isinstance(entry, dict) and _is_roampal_hook_command(entry.get("command"))
        )
    ]


def _merge_claude_hook_events(existing_hooks: dict, roampal_hooks: dict) -> dict:
    """Task 40, rule C — merge Roampal's hooks into the user's Claude Code
    settings.json WITHOUT replacing the user's entries.

    For each event Roampal manages: keep every existing group in place,
    remove only Roampal-owned hook commands from inside groups, drop a group
    only if it ends up with no hooks at all, then add Roampal's groups at the
    end. Events Roampal does not manage, and groups/hooks of shapes we do not
    recognize, are never touched (we only adjust Roampal's own entries).
    Idempotent: run twice → identical bytes.
    """
    merged = dict(existing_hooks)
    for event, roampal_groups in roampal_hooks.items():
        existing_groups = existing_hooks.get(event, [])
        if not isinstance(existing_groups, list):
            continue  # unrecognized shape — never wipe it
        kept_groups = []
        for group in existing_groups:
            if isinstance(group, dict) and isinstance(group.get("hooks"), list):
                kept_entries = _strip_roampal_hook_commands(group["hooks"])
                if not kept_entries:
                    continue  # group was all-Roampal → dropped
                group = {**group, "hooks": kept_entries}
            kept_groups.append(group)
        merged[event] = kept_groups + list(roampal_groups)
    return merged


def _merge_cursor_hook_events(existing_hooks: dict, roampal_hooks: dict) -> dict:
    """Task 40, rule C — same merge for Cursor's flat hook-entry lists."""
    merged = dict(existing_hooks)
    for event, roampal_entries in roampal_hooks.items():
        existing_entries = existing_hooks.get(event, [])
        if not isinstance(existing_entries, list):
            continue  # unrecognized shape — never wipe it
        merged[event] = _strip_roampal_hook_commands(existing_entries) + list(
            roampal_entries
        )
    return merged


def configure_claude_code(
    claude_dir: Path, is_dev: bool = False, force: bool = False, scope=None
):
    """Configure Claude Code hooks, MCP, and permissions.

    Task 40: existing files are never wiped or cluttered. Every file is read
    through `read_json_config` — a parse failure ABORTS this tool (prints and
    leaves every file byte-identical; `cmd_init` exits 1) instead of starting
    fresh — and every write goes through `write_json_config` (timestamped
    backup, atomic, backups pruned to the newest 3). Hooks merge per-event:
    the user's own hooks and events are kept; only Roampal-owned commands are
    replaced.

    Args:
        claude_dir: Path to ~/.claude directory
        is_dev: If True, adds ROAMPAL_DEV=1 to env sections
        force: If True, overwrite existing config even if different
        scope: `roampal init --scope`. "user" writes only the global config
            (~/.claude.json); "project"/"both" also write/update the project
            .mcp.json in the cwd; default (None) never CREATES the project
            file — it only updates one that already has a roampal-core entry.

    Returns:
        True when this tool was configured (or nothing needed changing),
        False when an unreadable config file made init skip it.
    """
    print(f"{BOLD}Configuring Claude Code...{RESET}")

    # Ensure directory exists (may not if detected via PATH on fresh install)
    claude_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # READ PHASE (Task 40): every file this function may change is read
    # BEFORE any write. One unreadable file aborts the whole tool — no
    # partial writes, no backups, files stay byte-for-byte identical.
    # ========================================================================
    settings_path = claude_dir / "settings.json"
    try:
        settings = read_json_config(settings_path)
    except ConfigReadError as e:
        _print_config_read_error(e)
        return False

    claude_json_path = Path.home() / ".claude.json"
    try:
        claude_json = read_json_config(claude_json_path)
    except ConfigReadError as e:
        _print_config_read_error(e)
        return False

    # Project .mcp.json: read up front whenever this run would touch it —
    # scope "project"/"both" always do; default scope only to check whether
    # an existing file already carries roampal-core (the upgrade path).
    # --scope user never touches the project file, so it must not read it
    # either: a corrupt one there must not block configuring the tool.
    local_mcp_path = Path.cwd() / ".mcp.json"
    project_writes = scope in ("project", "both")
    existing_local_mcp = None
    try:
        if scope != "user" and (project_writes or local_mcp_path.exists()):
            existing_local_mcp = read_json_config(local_mcp_path)
    except ConfigReadError as e:
        _print_config_read_error(e)
        return False

    # =========================================================================
    # settings.json — env, hooks (merge), permissions
    # =========================================================================
    # PRESERVE existing env section (critical for DEV/PROD isolation)
    existing_env = settings.get("env")
    if existing_env:
        settings["env"] = existing_env  # Keep what user had
    elif is_dev:
        settings["env"] = {"ROAMPAL_DEV": "1"}  # Add for --dev

    # Configure hooks - Claude Code expects nested format with type/command
    # v0.3.2: Use _build_hook_command() to ensure ROAMPAL_DEV is passed to hooks
    # Previously hooks didn't get the env var, causing split-brain between MCP (dev) and hooks (prod)
    submit_cmd = _build_hook_command("user_prompt_submit_hook", is_dev)
    stop_cmd = _build_hook_command("stop_hook", is_dev)

    # v0.3.6: Sidecar scoring moved server-side (no more --from-hook in Stop hook)
    context_cmd = "roampal context --recent-exchanges"
    if is_dev:
        context_cmd = f"ROAMPAL_DEV=1 {context_cmd}"

    # v0.4.2.1 → Task 40: MERGE hooks instead of replacing — keep the user's
    # own hooks (and the event lists for events Roampal doesn't manage);
    # only Roampal-owned commands are removed and re-added.
    roampal_hooks = {
        "UserPromptSubmit": [{"hooks": [{"type": "command", "command": submit_cmd}]}],
        "Stop": [{"hooks": [{"type": "command", "command": stop_cmd}]}],
        "SessionStart": [
            {
                "matcher": "compact",
                "hooks": [{"type": "command", "command": context_cmd}],
            },
            {
                "matcher": "startup",
                "hooks": [{"type": "command", "command": context_cmd}],
            },
            # v0.6.0 Task 30: `clear` sessions also rebuild from scratch —
            # the injection record must reset (session id from stdin).
            {
                "matcher": "clear",
                "hooks": [{"type": "command", "command": context_cmd}],
            },
        ],
    }
    existing_hooks = settings.get("hooks", {})
    if not isinstance(existing_hooks, dict):
        existing_hooks = {}
    settings["hooks"] = _merge_claude_hook_events(existing_hooks, roampal_hooks)

    # Configure permissions to auto-allow roampal MCP tools
    # This prevents the user from being spammed with permission prompts
    if "permissions" not in settings:
        settings["permissions"] = {}
    if "allow" not in settings["permissions"]:
        settings["permissions"]["allow"] = []

    # Add roampal MCP tools to allow list (using roampal-core server name)
    roampal_perms = [
        "mcp__roampal-core__search_memory",
        "mcp__roampal-core__add_to_memory_bank",
        "mcp__roampal-core__update_memory",
        "mcp__roampal-core__delete_memory",
        "mcp__roampal-core__record_response",
        "mcp__roampal-core__score_memories",
    ]

    for perm in roampal_perms:
        if perm not in settings["permissions"]["allow"]:
            settings["permissions"]["allow"].append(perm)

    write_json_config(settings_path, settings)
    print(f"  {GREEN}Wrote settings: {settings_path}{RESET}")
    print(f"  {GREEN}  - UserPromptSubmit hook (injects scoring + memories){RESET}")
    print(f"  {GREEN}  - Stop hook (enforces scoring + sidecar summarization){RESET}")
    print(f"  {GREEN}  - SessionStart hook (compaction recovery){RESET}")
    print(f"  {GREEN}  - Auto-allowed MCP permissions{RESET}")

    # =========================================================================
    # MCP Configuration - Write to ~/.claude.json (USER SCOPE - GLOBAL)
    # =========================================================================
    # Claude Code reads MCP servers from:
    #   - ~/.claude.json (root-level mcpServers = user scope, global)
    #   - ~/.claude.json (projects.{path}.mcpServers = local scope, per-project)
    #   - .mcp.json in project root (project scope, shared)
    #
    # Previously we wrote to ~/.claude/.mcp.json which is NOT a valid location.
    # Fixed in v0.2.5 to write to ~/.claude.json root-level mcpServers.
    #
    # Task 40: ~/.claude.json (login, per-project history, every MCP server)
    # was read in the read phase above — a parse failure aborted this tool.
    # Only mcpServers["roampal-core"] is ever touched here.
    # =========================================================================

    # Task 36: keep the project cwd (where Claude Code launches this MCP
    # process) off sys.path. v0.6.0 review fix 1: -E [-P] instead of -I —
    # -I's implied -s hides USER site-packages, breaking Store-Python /
    # pip --user installs.
    roampal_server_config = {
        "type": "stdio",
        "command": sys.executable,
        "args": _mcp_args(),
        "env": {"ROAMPAL_DEV": "1"} if is_dev else {},
    }

    # =========================================================================
    # IDEMPOTENCY CHECK: Skip if already correctly configured
    # =========================================================================
    # A non-dict mcpServers (hand-broken file) must not crash init: treat it
    # as empty and rebuild the key — only Roampal's entry is affected.
    claude_mcp_servers = claude_json.get("mcpServers")
    if not isinstance(claude_mcp_servers, dict):
        claude_mcp_servers = {}
        claude_json["mcpServers"] = claude_mcp_servers
    existing_config = claude_mcp_servers.get("roampal-core", {})
    if existing_config:
        # Check if args match (the key identifier).
        # v0.6.0 review fix 1: compare against the CURRENTLY written shape —
        # pre-v0.6.0 args (bare "-m", or Task 36's "-I") take the update
        # path below and migrate to the -E [-P] shape.
        if existing_config.get("args") == _mcp_args():
            # Check if env matches too
            existing_env = existing_config.get("env", {})
            expected_env = {"ROAMPAL_DEV": "1"} if is_dev else {}
            if existing_env == expected_env:
                print(
                    f"  {GREEN}[OK] roampal-core already configured correctly in {claude_json_path}{RESET}"
                )
                # Skip to migration check, don't write
            else:
                # v0.4.2.1: Always update roampal-owned MCP config when it differs
                claude_json["mcpServers"]["roampal-core"] = roampal_server_config
                write_json_config(claude_json_path, claude_json)
                print(f"  {GREEN}Updated MCP server in: {claude_json_path}{RESET}")
        else:
            # Different args — update to current version
            claude_json["mcpServers"]["roampal-core"] = roampal_server_config
            write_json_config(claude_json_path, claude_json)
            print(f"  {GREEN}Updated MCP server in: {claude_json_path}{RESET}")
    else:
        # No existing config - validate and write
        # =========================================================================
        # VALIDATION: Ensure roampal is importable before writing config
        # =========================================================================
        if not validate_roampal_importable(sys.executable):
            print(f"  {RED}Error: roampal not importable from {sys.executable}{RESET}")
            print(
                f"  {YELLOW}Make sure you're running from the correct virtual environment{RESET}"
            )
            print(f"  {YELLOW}Try: pip install roampal{RESET}")
            # Task 40: this is NOT a config-safety skip — the config file was
            # readable; the install is broken. Return True so cmd_init does
            # not report "a config file could not be read" and exit 1 (the
            # pre-Task-40 behavior: print the error and keep going).
            return True

        # Add roampal-core to root-level mcpServers (user scope = global)
        if "mcpServers" not in claude_json:
            claude_json["mcpServers"] = {}

        claude_json["mcpServers"]["roampal-core"] = roampal_server_config

        # Write back
        try:
            write_json_config(claude_json_path, claude_json)
            print(f"  {GREEN}Added MCP server to: {claude_json_path}{RESET}")
            print(
                f"  {GREEN}  - User scope (works globally across all projects){RESET}"
            )
        except Exception as e:
            print(f"  {RED}Error writing {claude_json_path}: {e}{RESET}")
            print(
                f"  {YELLOW}You may need to run: claude mcp add roampal-core python -- -m roampal.mcp.server{RESET}"
            )

    # =========================================================================
    # Migration: Clean up old broken config at ~/.claude/.mcp.json
    # =========================================================================
    # Task 40: read through the safe reader. The old file is a v0.2.5-era
    # leftover Claude Code never reads, so an unreadable one just skips the
    # cleanup (file untouched, same as the pre-Task-40 `except: pass`) —
    # it must not block configuring the tool.
    old_mcp_config_path = claude_dir / ".mcp.json"
    if old_mcp_config_path.exists():
        try:
            old_config = read_json_config(old_mcp_config_path)
        except ConfigReadError as e:
            print(f"  {YELLOW}Cannot read old config {old_mcp_config_path}: {e.reason}{RESET}")
            print(
                f"  {YELLOW}Roampal did not change this file; skipping the old-location cleanup.{RESET}"
            )
        else:
            if "mcpServers" in old_config and "roampal-core" in old_config.get(
                "mcpServers", {}
            ):
                # Remove roampal-core from the old location
                try:
                    del old_config["mcpServers"]["roampal-core"]
                    if old_config["mcpServers"]:
                        # Other servers exist, keep the file
                        write_json_config(old_mcp_config_path, old_config)
                    elif old_config:
                        # mcpServers emptied but the file still holds the
                        # user's other keys — keep the file (Task 40: never
                        # wipe).
                        write_json_config(old_mcp_config_path, old_config)
                    else:
                        # Only roampal was there, remove the file
                        old_mcp_config_path.unlink()
                    print(
                        f"  {GREEN}Migrated from old config: {old_mcp_config_path}{RESET}"
                    )
                except Exception as e:
                    print(
                        f"  {YELLOW}Could not update old config {old_mcp_config_path}: {e}{RESET}"
                    )
                    print(
                        f"  {YELLOW}Continuing — the old location is not read by Claude Code.{RESET}"
                    )

    # =========================================================================
    # Project-level .mcp.json (Task 40, rule D)
    # =========================================================================
    # --scope project|both: create or update it (as before). Default (no
    # --scope): NEVER create it — the user-scope entry in ~/.claude.json
    # already covers Claude Code, and a stray .mcp.json inside a git repo
    # invites accidental commits. Default only UPDATES an existing file that
    # already carries a roampal-core entry (the upgrade path for existing
    # users); a file without one is left untouched. --scope user: never touch.
    if scope == "user":
        print(f"  {GREEN}Claude Code configured!{RESET}\n")
        return True

    roampal_local_server = {
        "command": sys.executable,
        "args": _mcp_args(),
        "env": {"ROAMPAL_DEV": "1"} if is_dev else {},
    }

    if existing_local_mcp is None:
        # No project file and default scope: do not create one.
        print(f"  {GREEN}Claude Code configured!{RESET}\n")
        return True

    already_ours = (
        isinstance(existing_local_mcp.get("mcpServers"), dict)
        and "roampal-core" in existing_local_mcp["mcpServers"]
    )
    if scope is None and not already_ours:
        # Existing file without roampal-core — it is the user's; leave it.
        print(
            f"  {YELLOW}Found {local_mcp_path} without a roampal-core entry; "
            f"left untouched (pass --scope project to add Roampal to it).{RESET}"
        )
        print(f"  {GREEN}Claude Code configured!{RESET}\n")
        return True

    servers = existing_local_mcp.get("mcpServers")
    if not isinstance(servers, dict):
        # Malformed mcpServers value — rebuild the section fresh (the file
        # itself stays valid JSON; this only ever hits hand-broken files).
        existing_local_mcp["mcpServers"] = {}
    existing_local_mcp["mcpServers"]["roampal-core"] = roampal_local_server

    write_json_config(local_mcp_path, existing_local_mcp)
    if already_ours:
        print(f"  {GREEN}Updated project MCP config: {local_mcp_path}{RESET}")
    else:
        print(f"  {GREEN}Created project MCP config: {local_mcp_path}{RESET}")

    print(f"  {GREEN}Claude Code configured!{RESET}\n")
    return True


def configure_cursor(cursor_dir: Path, is_dev: bool = False, force: bool = False):
    """Configure Cursor MCP and hooks.

    Task 40: mcp.json and hooks.json are read through `read_json_config` — a
    parse failure aborts this tool (prints and leaves every file
    byte-identical; `cmd_init` exits 1). Hooks merge per-event: the user's own
    hook entries are kept in place; only Roampal-owned commands are removed
    and re-added. Every write goes through `write_json_config`.

    Args:
        cursor_dir: Path to ~/.cursor directory
        is_dev: If True, adds ROAMPAL_DEV=1 to env section
        force: If True, overwrite existing config even if different

    Returns:
        True when this tool was configured, False when an unreadable config
        file made init skip it.
    """
    print(f"{BOLD}Configuring Cursor...{RESET}")

    # Ensure directory exists (may not if detected via PATH on fresh install)
    cursor_dir.mkdir(parents=True, exist_ok=True)

    # Cursor uses ~/.cursor/mcp.json
    mcp_config_path = cursor_dir / "mcp.json"
    expected_env = {"ROAMPAL_DEV": "1"} if is_dev else {}
    # Task 36: Cursor spawns MCP with the project cwd; a local `roampal/`
    # directory there must not shadow the installed package.
    # v0.6.0 review fix 1: -E [-P] instead of -I — -I's implied -s hides
    # USER site-packages, breaking Store-Python / pip --user installs.
    roampal_server_config = {
        "command": sys.executable,
        "args": _mcp_args(),
        "env": expected_env,
    }

    # Task 40 read phase: both files before either write.
    try:
        existing_mcp = read_json_config(mcp_config_path)
    except ConfigReadError as e:
        _print_config_read_error(e)
        return False

    mcp_needs_write = True
    if existing_mcp:
        mcp_servers = existing_mcp.get("mcpServers")
        mcp_servers = mcp_servers if isinstance(mcp_servers, dict) else {}
        existing_config = mcp_servers.get("roampal-core", {})

        if existing_config:
            # Check if config matches.
            # v0.6.0 review fix 1: compare against the CURRENTLY written
            # shape — pre-v0.6.0 args migrate via the update path.
            if (
                existing_config.get("args") == _mcp_args()
                and existing_config.get("env", {}) == expected_env
            ):
                print(
                    f"  {GREEN}[OK] roampal-core already configured correctly in {mcp_config_path}{RESET}"
                )
                mcp_needs_write = False
            else:
                # v0.4.2.1: Always update roampal-owned MCP config when it differs
                mcp_servers["roampal-core"] = roampal_server_config
                print(f"  {GREEN}Updated MCP config: {mcp_config_path}{RESET}")
        else:
            # No roampal-core entry - add it
            mcp_servers["roampal-core"] = roampal_server_config
            existing_mcp["mcpServers"] = mcp_servers
    else:
        existing_mcp["mcpServers"] = {"roampal-core": roampal_server_config}

    if mcp_needs_write:
        write_json_config(mcp_config_path, existing_mcp)
        print(f"  {GREEN}Created MCP config: {mcp_config_path}{RESET}")

    # Cursor 1.7+ supports hooks - create ~/.cursor/hooks.json
    hooks_config_path = cursor_dir / "hooks.json"

    # v0.3.2: Use centralized _build_hook_command() for consistency
    submit_cmd = _build_hook_command("user_prompt_submit_hook", is_dev)
    stop_cmd = _build_hook_command("stop_hook", is_dev)

    expected_hooks = {
        "beforeSubmitPrompt": [{"command": submit_cmd}],
        "stop": [{"command": stop_cmd}],
    }

    # Task 40 read phase (part 2): hooks.json before any Cursor write.
    try:
        hooks_existing = read_json_config(hooks_config_path)
    except ConfigReadError as e:
        _print_config_read_error(e)
        return False

    hooks_existing_dict = hooks_existing.get("hooks")
    existing_hooks = (
        hooks_existing_dict if isinstance(hooks_existing_dict, dict) else {}
    )
    # Task 40, rule C: keep the user's own entries in place for managed
    # events; remove only Roampal-owned commands, then add Roampal's.
    merged_hooks = _merge_cursor_hook_events(existing_hooks, expected_hooks)
    hooks_changed = merged_hooks != existing_hooks

    if not hooks_existing or hooks_changed:
        if "version" not in hooks_existing:
            hooks_existing["version"] = 1
        hooks_existing["hooks"] = merged_hooks
        write_json_config(hooks_config_path, hooks_existing)
        print(f"  {GREEN}Wrote hooks config: {hooks_config_path}{RESET}")
        print(
            f"  {GREEN}  - beforeSubmitPrompt hook (injects scoring + memories){RESET}"
        )
        print(f"  {GREEN}  - stop hook (enforces record_response){RESET}")
    else:
        print(
            f"  {GREEN}[OK] Hooks already configured correctly in {hooks_config_path}{RESET}"
        )

    print(f"  {GREEN}Cursor configured!{RESET}\n")
    return True


def _install_plugin_file(plugin_source: Path, plugin_dest: Path):
    """Install plugin file with post-copy verification and fallback methods.

    v0.5.5.2: Windows shutil.copy can silently fail due to OneDrive sync,
    antivirus interference, or Controlled Folder Access. This function verifies
    the copy succeeded and falls back to manual read/write if needed.
    """
    plugin_dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Primary method: shutil.copy
        shutil.copy(str(plugin_source), str(plugin_dest))
    except (OSError, PermissionError) as e:
        print(f"  {RED}Failed to install plugin: {e}{RESET}")
        print(f"  {YELLOW}Possible causes:{RESET}")
        print(
            f"  {YELLOW}  - OpenCode Desktop is running and holds a file lock{RESET}"
        )
        print(
            f"  {YELLOW}  - Read-only attribute on existing {plugin_dest.name}{RESET}"
        )
        print(f"  {YELLOW}  - Antivirus / Controlled Folder Access blocking write{RESET}")
        print(f"  {YELLOW}  - OneDrive sync quarantining the destination{RESET}")
        print(f"  {YELLOW}If those don't apply, copy manually:{RESET}")
        print(f"  {YELLOW}  cp {plugin_source} {plugin_dest}{RESET}")
        return

    # Verify the copy actually succeeded
    if not plugin_dest.exists():
        print(
            f"  {RED}Plugin install failed: file disappeared after copy{RESET}"
        )
        print(f"  {YELLOW}Possible causes:{RESET}")
        print(f"  {YELLOW}  - Antivirus quarantined the destination{RESET}")
        print(f"  {YELLOW}  - OneDrive sync still processing{RESET}")
        print(
            f"  {YELLOW}  - Controlled Folder Access blocked the write{RESET}"
        )
        print(f"  {YELLOW}Copy manually:{RESET}")
        print(f"  {YELLOW}  cp {plugin_source} {plugin_dest}{RESET}")
        return

    try:
        dest_size = plugin_dest.stat().st_size
        source_size = plugin_source.stat().st_size
        if dest_size == 0 or dest_size != source_size:
            # Copy appeared to succeed but file is wrong size - retry with manual method
            src_content = plugin_source.read_bytes()
            plugin_dest.write_bytes(src_content)
            new_size = plugin_dest.stat().st_size
            if new_size != len(src_content):
                print(
                    f"  {RED}Plugin install failed: file size mismatch after fallback copy (expected {len(src_content)}, got {new_size}){RESET}"
                )
                return
    except Exception as e:
        logger.warning(f"Failed to verify plugin file: {e}")

    print(f"  {GREEN}Installed plugin: {plugin_dest}{RESET}")


def _verify_plugin_install_targets(plugin_source: Path, targets: list[Path]) -> None:
    """Hash every install target against source. Warn loudly on drift.

    The dual-path install introduced in v0.5.5.2 succeeds as long as ONE
    destination got the new bytes. If the other silently retained stale content
    (lock, antivirus, OneDrive), OpenCode may load from that stale path with no
    warning — exactly the symptom Marcus reported in issue #11.

    This function detects the divergence after both writes and tells the user
    which path is stale plus the exact command to repair it.
    """
    try:
        src_bytes = plugin_source.read_bytes()
    except OSError as e:
        logger.warning(f"Plugin verification skipped — could not read source: {e}")
        return
    src_hash = hashlib.sha256(src_bytes).hexdigest()

    fresh: List[Path] = []
    stale: List[Path] = []
    for target in targets:
        try:
            target_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError:
            stale.append(target)
            continue
        (fresh if target_hash == src_hash else stale).append(target)

    if not stale:
        return

    print()
    print(
        f"  {RED}WARNING: Plugin install left stale content at {len(stale)} of {len(targets)} location(s).{RESET}"
    )
    print(
        f"  {YELLOW}OpenCode may load whichever path it discovers first — if that's a stale one, you'll silently run an old plugin.{RESET}"
    )
    for path in stale:
        print(f"  {RED}  STALE:  {path}{RESET}")
    for path in fresh:
        print(f"  {GREEN}  FRESH:  {path}{RESET}")
    if fresh:
        # Suggest the simplest repair: copy from a known-fresh destination.
        donor = fresh[0]
        for stale_path in stale:
            print(f"  {YELLOW}  repair: cp {donor} {stale_path}{RESET}")
    else:
        # Both stale (rare) — point them at the source.
        for stale_path in stale:
            print(f"  {YELLOW}  repair: cp {plugin_source} {stale_path}{RESET}")
    print(
        f"  {YELLOW}  Then fully restart OpenCode Desktop (kill all processes) so it reloads the plugin.{RESET}"
    )
    print()


def configure_opencode(is_dev: bool = False, force: bool = False, scope: str | None = None):
    """Configure OpenCode MCP and plugin.

    v0.5.3 Section 8: Scope-aware writes + atomic config updates.

    Args:
        is_dev: If True, adds ROAMPAL_DEV=1 to env section
        force: If True, overwrite existing config even if different
        scope: 'user' = user-global only, 'project' = project-local only, None = auto-detect
    """
    print(f"{BOLD}Configuring OpenCode...{RESET}")

    # OpenCode config locations (XDG Base Directory spec)
    if sys.platform == "win32":
        user_config_dir = Path.home() / ".config" / "opencode"
    else:
        xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        user_config_dir = Path(xdg_config) / "opencode"

    user_config_file = user_config_dir / "opencode.json"
    plugin_dir = user_config_dir / "plugins"

    # Create directories if they don't exist
    user_config_dir.mkdir(parents=True, exist_ok=True)
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Configure MCP server in opencode.json (scope-aware)
    # =========================================================================
    # v0.6.0 round 2: PYTHONPATH REMOVED from the OpenCode MCP env. Two
    # reasons: (a) it is dead weight — the written command runs with -E,
    # which ignores PYTHONPATH; (b) the CLI split moved this file one
    # folder deeper, so `Path(__file__).parent.parent` (the pre-split
    # package-root computation) now yields the roampal PACKAGE dir — and
    # without -E that path would make `import mcp` load roampal's own
    # mcp/ directory instead of the real MCP package. Removing it fixes
    # both the stale path and the shadowing hazard.
    expected_env = {"ROAMPAL_PLATFORM": "opencode"}
    if is_dev:
        expected_env["ROAMPAL_DEV"] = "1"
    roampal_mcp_config = {
        "type": "local",
        # Task 36: opencode spawns this MCP process with the project cwd;
        # a local `roampal/` directory there must not shadow the install.
        # v0.6.0 review fix 1: -E [-P] instead of -I — -I's implied -s hides
        # USER site-packages, breaking Store-Python / pip --user installs.
        "command": [sys.executable, *_mcp_args()],
        "enabled": True,
    }
    if expected_env:
        roampal_mcp_config["environment"] = expected_env

    # Scope-aware config selection (v0.5.3 Section 8)
    project_config = _find_project_opencode_config()

    if scope == "user":
        config_path = user_config_file
    elif scope == "project":
        if project_config and project_config != user_config_file:
            config_path = project_config
        else:
            # No project-local config exists — create one in current directory
            config_path = Path("opencode.json")
    else:
        # Auto-detect (default)
        if force or not user_config_file.exists():
            config_path = user_config_file
        elif project_config and project_config != user_config_file:
            config_path = project_config
        else:
            config_path = user_config_file

    # Load existing config or create new
    config = {}
    mcp_needs_write = True
    parse_failed = False

    # Task 40: read through the safe reader — utf-8-sig, and a parse failure
    # raises ConfigReadError instead of risking a "fresh" overwrite.
    if config_path.exists():
        try:
            config = read_json_config(config_path)
            mcp_section = config.get("mcp")
            mcp_section = mcp_section if isinstance(mcp_section, dict) else {}
            existing_mcp = mcp_section.get("roampal-core", {})

            if existing_mcp:
                # Check if roampal's base config matches (ignore sidecar vars)
                existing_cmd = existing_mcp.get("command", [])
                existing_env = existing_mcp.get("environment", {})
                # v0.4.2.2: Compare only roampal base keys, not sidecar vars
                base_keys_match = all(
                    existing_env.get(k) == v for k, v in expected_env.items()
                )

                if existing_cmd == roampal_mcp_config["command"] and base_keys_match:
                    print(
                        f"  {GREEN}[OK] roampal-core MCP already configured correctly{RESET}"
                    )
                    mcp_needs_write = False
                # else: mcp_needs_write stays True, update applied below
        except ConfigReadError as e:
            logger.warning(f"Failed to parse existing opencode.json: {e}")
            parse_failed = True
            parse_error = e

    if parse_failed:
        print(f"  {RED}[ERROR] Cannot parse existing opencode.json:{RESET}")
        print(f"    {config_path}")
        print(f"    {parse_error}")
        print(f"    Fix the JSON or back up + delete to regenerate.")
        print(
            f"  {YELLOW}Roampal did not change this file. Fix the JSON (or restore a "
            f".bak- copy next to it) and run `roampal init` again.{RESET}"
        )
        print(f"  {YELLOW}Skipping MCP write — file left untouched.{RESET}")
        mcp_needs_write = False

    if mcp_needs_write:
        if not isinstance(config.get("mcp"), dict):
            config["mcp"] = {}
        # v0.4.2.1: Merge environment — preserve sidecar vars (ROAMPAL_SIDECAR_*)
        existing_mcp_env = (
            config.get("mcp", {}).get("roampal-core", {}).get("environment", {})
        )
        merged_env = {**existing_mcp_env, **expected_env}
        # v0.6.0 round 2: strip the pre-split PYTHONPATH entry — moved-file
        # path was wrong, and -E ignores it anyway. Re-running init cleans
        # old configs.
        merged_env.pop("PYTHONPATH", None)
        roampal_mcp_config["environment"] = merged_env
        config["mcp"]["roampal-core"] = roampal_mcp_config
        _safe_write_opencode_config(config_path, config)
        print(f"  {GREEN}Created MCP config: {config_path}{RESET}")

    # =========================================================================
    # 2. Install TypeScript plugin (user-global only — never project-local)
    # =========================================================================
    plugin_file = plugin_dir / "roampal.ts"
    # v0.6.0 Task 3: two parents up — __file__ is roampal/cli/_monolith_impl.py,
    # so plugins/ lives at roampal/plugins (the pre-refactor Path(__file__).parent
    # from the cli.py module would now point inside the cli package).
    plugin_source = Path(__file__).parent.parent / "plugins" / "opencode" / "roampal.ts"

    plugin_needs_write = True

    if not force and plugin_file.exists():
        # Check if plugin content matches
        try:
            existing_content = plugin_file.read_text(encoding="utf-8")
            source_content = (
                plugin_source.read_text(encoding="utf-8")
                if plugin_source.exists()
                else ""
            )

            if existing_content == source_content:
                print(f"  {GREEN}[OK] roampal plugin already installed{RESET}")
                plugin_needs_write = False
            # v0.4.2.1: Always update roampal-owned plugin when source differs
        except Exception as e:
            logger.warning(f"Failed to compare plugin files: {e}")

    if plugin_needs_write:
        if plugin_source.exists():
            install_targets: list[Path] = [plugin_file]
            _install_plugin_file(plugin_source, plugin_file)

            # v0.5.5.2: On Windows, also expose the plugin at %APPDATA%\opencode\plugins
            # because different OpenCode builds resolve plugins from different paths.
            # v0.5.6: Hardlink the alt path to the canonical .config copy when possible.
            # Same inode → divergence is structurally impossible; both paths read the
            # same bytes, so a successful canonical write is automatically reflected
            # at the alt path with no second write to fail. Fall back to a real copy
            # if hardlink creation fails (cross-volume, OneDrive reparse-point quirks,
            # antivirus, filesystem that doesn't support hardlinks). The post-copy
            # hash verification below still runs in either case.
            if sys.platform == "win32":
                appdata = os.environ.get("APPDATA", "")
                if appdata:
                    alt_plugin_dir = Path(appdata) / "opencode" / "plugins"
                    alt_plugin_file = alt_plugin_dir / "roampal.ts"
                    if alt_plugin_file != plugin_file:
                        alt_plugin_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            if alt_plugin_file.exists() or alt_plugin_file.is_symlink():
                                alt_plugin_file.unlink()
                            os.link(str(plugin_file), str(alt_plugin_file))
                        except OSError:
                            # Hardlink unsupported here — fall back to a real copy.
                            _install_plugin_file(plugin_source, alt_plugin_file)
                        install_targets.append(alt_plugin_file)
                else:
                    print(
                        f"  {YELLOW}Skipped %APPDATA% fallback install — APPDATA env var is unset.{RESET}"
                    )
                    print(
                        f"  {YELLOW}If OpenCode can't find the plugin, set APPDATA and re-run, or copy manually.{RESET}"
                    )

            # v0.5.6: Cross-target verification. Catches the case where one
            # destination silently retained stale content (e.g. transient lock,
            # antivirus, OneDrive sync) while another succeeded — the bug Marcus
            # originally reported (issue #11) that v0.5.5.2's dual-path partially
            # masked. OpenCode loads from whichever destination it discovers
            # first; if that one is stale, the user runs old plugin code with no
            # warning. Hash all written destinations and warn loudly on drift.
            _verify_plugin_install_targets(plugin_source, install_targets)
        else:
            print(f"  {RED}Plugin source not found: {plugin_source}{RESET}")
            print(f"  {YELLOW}You may need to reinstall roampal{RESET}")

    # =========================================================================
    # 3. Remind user about shared server
    # =========================================================================
    shared_port = 27183 if is_dev else 27182
    print(f"  {GREEN}OpenCode configured!{RESET}")
    print(f"  {GREEN}  Server port: {shared_port}{RESET}")
    print(
        f"  {YELLOW}Note: Server auto-starts on first message. To stop: roampal stop{RESET}"
    )
    print()

    # Task 40: the MCP part was skipped when opencode.json was unreadable —
    # the tool counts as skipped so `cmd_init` can exit 1.
    return not parse_failed



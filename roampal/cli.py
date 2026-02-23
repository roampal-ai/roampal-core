"""
Roampal CLI - One command install for AI coding tools

Usage:
    pip install roampal
    roampal init          # Configure Claude Code / Cursor / OpenCode
    roampal start         # Start the memory server
    roampal stop          # Stop the memory server
    roampal status        # Check server status
    roampal doctor        # Diagnose installation issues
"""

import argparse
import json
import logging
import os
import platform
import sys
import shutil
import subprocess
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def _should_color() -> bool:
    """Check if terminal output should use ANSI colors.

    Respects NO_COLOR (https://no-color.org), TERM=dumb, and non-TTY stdout.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        if not sys.stdout.isatty():
            return False
    except Exception:
        return False
    return True


# ANSI colors (disabled when NO_COLOR set, TERM=dumb, or stdout is not a TTY)
if _should_color():
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
else:
    GREEN = YELLOW = RED = BLUE = RESET = BOLD = ""

# Non-interactive mode flag — set by --no-input or non-TTY stdin
_NO_INPUT = False


def _is_interactive() -> bool:
    """Check if we can prompt the user for input.

    Returns False when --no-input is set, stdin is not a TTY (piped/CI),
    or stdin is unavailable.
    """
    if _NO_INPUT:
        return False
    try:
        return sys.stdin.isatty()
    except Exception:
        return False

# Port configuration - DEV and PROD use different ports to avoid collision
PROD_PORT = 27182
DEV_PORT = 27183

# Email signup webhook (Google Apps Script → Google Sheet)
# Deploy Apps Script and paste the web app URL here
SIGNUP_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxnj6GN8mNtq_xn6vRLwMLSH6VL7397Vcx-9Me8pX_BP7Yt2oob6utsZ7pCke6rcfsY/exec"


def is_dev_mode(args=None) -> bool:
    """
    SINGLE SOURCE OF TRUTH for DEV mode detection.

    Checks (in order):
    1. args.dev flag (if args provided)
    2. ROAMPAL_DEV environment variable

    ALL commands MUST use this. Never check args.dev directly.
    See ARCHITECTURE.md 'Dev Mode Implementation' for details.
    """
    if args is not None and getattr(args, 'dev', False):
        return True
    return os.environ.get('ROAMPAL_DEV', '').lower() in ('1', 'true', 'yes')


def get_port(args=None) -> int:
    """Get port based on DEV/PROD mode. Respects explicit --port override."""
    if args and hasattr(args, 'port') and isinstance(getattr(args, 'port', None), int):
        return args.port
    return DEV_PORT if is_dev_mode(args) else PROD_PORT


def get_data_dir(dev: bool = False) -> Path:
    """Get the data directory path. DEV uses separate directory to avoid collision with PROD."""
    if dev:
        # DEV mode uses separate directory
        if os.name == 'nt':  # Windows
            appdata = os.environ.get('APPDATA', str(Path.home()))
            return Path(appdata) / "Roampal_DEV" / "data"
        elif sys.platform == 'darwin':  # macOS
            return Path.home() / "Library" / "Application Support" / "Roampal_DEV" / "data"
        else:  # Linux
            return Path.home() / ".local" / "share" / "roampal_dev" / "data"
    else:
        # PROD mode
        if os.name == 'nt':  # Windows
            appdata = os.environ.get('APPDATA', str(Path.home()))
            return Path(appdata) / "Roampal" / "data"
        elif sys.platform == 'darwin':  # macOS
            return Path.home() / "Library" / "Application Support" / "Roampal" / "data"
        else:  # Linux
            return Path.home() / ".local" / "share" / "roampal" / "data"


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
    base = f'{python_exe} -m roampal.hooks.{module}'

    if not is_dev:
        return base

    # Wrap command to set env var - hooks run as subprocesses that don't
    # inherit MCP's env block, so we must set the var explicitly
    if sys.platform == "win32":
        return f'cmd /c "set ROAMPAL_DEV=1 && {base}"'
    else:
        return f'ROAMPAL_DEV=1 {base}'


def print_banner():
    """Print Roampal banner."""
    print(f"""
{BLUE}{BOLD}+---------------------------------------------------+
|                   ROAMPAL                         |
|    Outcome-Based Memory for AI Coding Tools       |
+---------------------------------------------------+{RESET}
""")


def check_for_updates() -> tuple:
    """Check if a newer version is available on PyPI.

    Caches the result for 24 hours to avoid hitting PyPI on every command.

    Returns:
        tuple: (update_available: bool, current_version: str, latest_version: str)
    """
    try:
        from roampal import __version__
    except Exception:
        return (False, "unknown", "unknown")

    # Check cache first (stored in data dir)
    import time
    cache_file = get_data_dir() / ".update_cache"
    try:
        if cache_file.exists():
            cache_data = json.loads(cache_file.read_text())
            cache_age = time.time() - cache_data.get("timestamp", 0)
            if cache_age < 86400 and cache_data.get("current") == __version__:
                return (cache_data["update_available"], __version__, cache_data["latest"])
    except Exception:
        pass  # Corrupted cache, re-check

    # Fresh check from PyPI
    try:
        import urllib.request

        url = "https://pypi.org/pypi/roampal/json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
            latest = data.get("info", {}).get("version", __version__)

            current_parts = [int(x) for x in __version__.split(".")]
            latest_parts = [int(x) for x in latest.split(".")]
            update_available = latest_parts > current_parts

            # Cache result
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({
                    "timestamp": time.time(),
                    "current": __version__,
                    "latest": latest,
                    "update_available": update_available,
                }))
            except Exception:
                pass

            return (update_available, __version__, latest)
    except Exception:
        return (False, __version__, __version__)


def print_update_notice():
    """Print update notice if newer version available. Non-blocking."""
    update_available, current, latest = check_for_updates()
    if update_available:
        print(f"{YELLOW}[!] Update available: {latest} (you have {current}){RESET}")
        print(f"    Run: pip install --upgrade roampal && roampal init --force\n")


def collect_email(detected_tools: list):
    """Optionally collect user email for updates. Non-blocking, skippable.

    Marker file stores 'version:status' (e.g. '0.3.2:provided' or '0.3.2:skipped').
    - First install (no marker) → ask
    - Re-run same version → skip regardless
    - Update + previously provided email → skip (we already have it)
    - Update + previously skipped → ask again (one more chance)
    """
    if not SIGNUP_WEBHOOK_URL:
        return  # Webhook not configured yet

    from roampal import __version__
    data_dir = get_data_dir()
    marker = data_dir / ".email_asked"
    if marker.exists():
        try:
            marker_data = marker.read_text().strip()
            if ":" in marker_data:
                asked_version, status = marker_data.rsplit(":", 1)
            else:
                # Legacy marker (just version) — treat as skipped
                asked_version, status = marker_data, "skipped"
            if asked_version == __version__:
                return  # Already asked for this version
            if status == "provided":
                # They already gave email on a previous version, don't nag
                _write_email_marker(marker, __version__, "provided")
                return
        except Exception:
            pass  # Corrupted marker, ask again

    if not _is_interactive():
        _write_email_marker(marker, __version__, "skipped")
        return

    print(f"{BOLD}Stay in the loop?{RESET}")
    print(f"  Get notified about updates and new features.")
    print(f"  {YELLOW}(Optional - press Enter to skip){RESET}")

    try:
        email = input(f"\n  Email: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        _write_email_marker(marker, __version__, "skipped")
        return

    if not email or "@" not in email:
        if email:
            print(f"  {YELLOW}Doesn't look like an email, skipping.{RESET}")
        else:
            print()  # Blank line after empty Enter
        _write_email_marker(marker, __version__, "skipped")
        return

    # Send to webhook (fire-and-forget, don't block on failure)
    # Uses httpx because Google Apps Script redirects break urllib
    try:
        import httpx
        payload = {
            "email": email,
            "platform": ", ".join(detected_tools),
            "version": __version__,
            "os": platform.system()
        }
        httpx.post(SIGNUP_WEBHOOK_URL, json=payload, follow_redirects=True, timeout=5.0)
        print(f"  {GREEN}Thanks! We'll keep you posted.{RESET}\n")
    except Exception:
        # Silently fail - don't let signup issues block init
        print(f"  {GREEN}Thanks! We'll keep you posted.{RESET}\n")

    _write_email_marker(marker, __version__, "provided")


def _write_email_marker(marker: Path, version: str, status: str = "skipped"):
    """Write version:status to marker file.

    Status is 'provided' (gave email) or 'skipped' (pressed Enter).
    """
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{version}:{status}")
    except Exception:
        pass  # Non-critical


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
    explicit_claude = getattr(args, 'claude_code', False)
    explicit_cursor = getattr(args, 'cursor', False)
    explicit_opencode = getattr(args, 'opencode', False)

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
                if localappdata and (Path(localappdata) / "Programs" / "cursor").exists():
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
        print("\nInstall one of these tools first, or use --claude-code / --cursor / --opencode to force setup.")
        return 1

    print(f"{GREEN}Configuring: {', '.join(detected)}{RESET}\n")

    # Configure each detected tool
    is_dev = is_dev_mode(args)
    force = getattr(args, 'force', False)
    for tool in detected:
        if tool == "claude-code":
            configure_claude_code(claude_code_dir, is_dev=is_dev, force=force)
        elif tool == "cursor":
            configure_cursor(cursor_dir, is_dev=is_dev, force=force)
        elif tool == "opencode":
            configure_opencode(is_dev=is_dev, force=force)

    # Sidecar setup for OpenCode users
    if "opencode" in detected:
        _prompt_sidecar_setup(force=force)

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
        next_steps.append(f"  Run {BLUE}opencode{RESET} and start chatting! (server auto-starts on first message)")

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


def validate_roampal_importable(python_exe: str) -> bool:
    """Validate that roampal can be imported from the given python executable."""
    try:
        result = subprocess.run(
            [python_exe, "-c", "import roampal"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def configure_claude_code(claude_dir: Path, is_dev: bool = False, force: bool = False):
    """Configure Claude Code hooks, MCP, and permissions.

    Args:
        claude_dir: Path to ~/.claude directory
        is_dev: If True, adds ROAMPAL_DEV=1 to env sections
        force: If True, overwrite existing config even if different
    """
    print(f"{BOLD}Configuring Claude Code...{RESET}")

    # Ensure directory exists (may not if detected via PATH on fresh install)
    claude_dir.mkdir(parents=True, exist_ok=True)

    # Create settings.json with hooks and permissions
    settings_path = claude_dir / "settings.json"

    # Load existing settings or create new
    settings = {}
    existing_env = None
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            # PRESERVE existing env section (critical for DEV/PROD isolation)
            existing_env = settings.get("env")
        except Exception as e:
            logger.warning(f"Failed to parse existing settings.json: {e}")

    # Handle env section: preserve existing OR add if --dev
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

    settings["hooks"] = {
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": submit_cmd
                    }
                ]
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": stop_cmd
                    }
                ]
            }
        ],
        "SessionStart": [
            {
                "matcher": "compact",
                "hooks": [{
                    "type": "command",
                    "command": context_cmd
                }]
            },
            {
                "matcher": "startup",
                "hooks": [{
                    "type": "command",
                    "command": context_cmd
                }]
            }
        ]
    }

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
        "mcp__roampal-core__score_memories"
    ]

    for perm in roampal_perms:
        if perm not in settings["permissions"]["allow"]:
            settings["permissions"]["allow"].append(perm)

    settings_path.write_text(json.dumps(settings, indent=2))
    print(f"  {GREEN}Created settings: {settings_path}{RESET}")
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
    # =========================================================================

    claude_json_path = Path.home() / ".claude.json"
    roampal_server_config = {
        "type": "stdio",
        "command": sys.executable,
        "args": ["-m", "roampal.mcp.server"],
        "env": {"ROAMPAL_DEV": "1"} if is_dev else {}
    }

    # Load existing ~/.claude.json or create minimal structure
    claude_json = {}
    if claude_json_path.exists():
        try:
            claude_json = json.loads(claude_json_path.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"  {YELLOW}Warning: Could not read {claude_json_path}: {e}{RESET}")
            print(f"  {YELLOW}Creating new config...{RESET}")

    # =========================================================================
    # IDEMPOTENCY CHECK: Skip if already correctly configured
    # =========================================================================
    existing_config = claude_json.get("mcpServers", {}).get("roampal-core", {})
    if existing_config:
        # Check if args match (the key identifier)
        if existing_config.get("args") == ["-m", "roampal.mcp.server"]:
            # Check if env matches too
            existing_env = existing_config.get("env", {})
            expected_env = {"ROAMPAL_DEV": "1"} if is_dev else {}
            if existing_env == expected_env:
                print(f"  {GREEN}[OK] roampal-core already configured correctly in {claude_json_path}{RESET}")
                # Skip to migration check, don't write
            else:
                # Env differs (e.g., dev vs prod)
                if not force:
                    print(f"  {YELLOW}roampal-core config differs (env mismatch):{RESET}")
                    print(f"    Current env: {existing_env}")
                    print(f"    New env:     {expected_env}")
                    print(f"  {YELLOW}Use --force to overwrite{RESET}")
                else:
                    # Force overwrite
                    claude_json["mcpServers"]["roampal-core"] = roampal_server_config
                    claude_json_path.write_text(json.dumps(claude_json, indent=2), encoding='utf-8')
                    print(f"  {GREEN}Updated MCP server in: {claude_json_path} (forced){RESET}")
        else:
            # Different args - this is weird, warn and require force
            if not force:
                print(f"  {YELLOW}roampal-core config differs:{RESET}")
                print(f"    Current args: {existing_config.get('args')}")
                print(f"    New args:     {roampal_server_config['args']}")
                print(f"  {YELLOW}Use --force to overwrite{RESET}")
            else:
                claude_json["mcpServers"]["roampal-core"] = roampal_server_config
                claude_json_path.write_text(json.dumps(claude_json, indent=2), encoding='utf-8')
                print(f"  {GREEN}Updated MCP server in: {claude_json_path} (forced){RESET}")
    else:
        # No existing config - validate and write
        # =========================================================================
        # VALIDATION: Ensure roampal is importable before writing config
        # =========================================================================
        if not validate_roampal_importable(sys.executable):
            print(f"  {RED}Error: roampal not importable from {sys.executable}{RESET}")
            print(f"  {YELLOW}Make sure you're running from the correct virtual environment{RESET}")
            print(f"  {YELLOW}Try: pip install roampal{RESET}")
            return

        # Add roampal-core to root-level mcpServers (user scope = global)
        if "mcpServers" not in claude_json:
            claude_json["mcpServers"] = {}

        claude_json["mcpServers"]["roampal-core"] = roampal_server_config

        # Write back
        try:
            claude_json_path.write_text(json.dumps(claude_json, indent=2), encoding='utf-8')
            print(f"  {GREEN}Added MCP server to: {claude_json_path}{RESET}")
            print(f"  {GREEN}  - User scope (works globally across all projects){RESET}")
        except Exception as e:
            print(f"  {RED}Error writing {claude_json_path}: {e}{RESET}")
            print(f"  {YELLOW}You may need to run: claude mcp add roampal-core python -- -m roampal.mcp.server{RESET}")

    # =========================================================================
    # Migration: Clean up old broken config at ~/.claude/.mcp.json
    # =========================================================================
    old_mcp_config_path = claude_dir / ".mcp.json"
    if old_mcp_config_path.exists():
        try:
            old_config = json.loads(old_mcp_config_path.read_text())
            if "mcpServers" in old_config and "roampal-core" in old_config.get("mcpServers", {}):
                # Remove roampal-core from the old location
                del old_config["mcpServers"]["roampal-core"]
                if old_config["mcpServers"]:
                    # Other servers exist, keep the file
                    old_mcp_config_path.write_text(json.dumps(old_config, indent=2))
                else:
                    # Only roampal was there, remove the file
                    old_mcp_config_path.unlink()
                print(f"  {GREEN}Migrated from old config: {old_mcp_config_path}{RESET}")
        except Exception:
            pass  # Old file might be malformed, ignore

    # =========================================================================
    # Also create project-level .mcp.json (optional, for project scope)
    # =========================================================================
    # This is valid for project-scoped MCP but requires user approval
    local_mcp_path = Path.cwd() / ".mcp.json"
    local_mcp_config = {
        "mcpServers": {
            "roampal-core": {
                "command": sys.executable,
                "args": ["-m", "roampal.mcp.server"],
                "env": {"ROAMPAL_DEV": "1"} if is_dev else {}
            }
        }
    }

    # Merge with existing local config if present
    if local_mcp_path.exists():
        try:
            existing = json.loads(local_mcp_path.read_text())
            if "mcpServers" in existing:
                existing["mcpServers"]["roampal-core"] = local_mcp_config["mcpServers"]["roampal-core"]
            else:
                existing["mcpServers"] = local_mcp_config["mcpServers"]
            local_mcp_config = existing
        except Exception as e:
            logger.warning(f"Failed to parse existing local MCP config: {e}")

    local_mcp_path.write_text(json.dumps(local_mcp_config, indent=2))
    print(f"  {GREEN}Created project MCP config: {local_mcp_path}{RESET}")

    print(f"  {GREEN}Claude Code configured!{RESET}\n")


def configure_cursor(cursor_dir: Path, is_dev: bool = False, force: bool = False):
    """Configure Cursor MCP and hooks.

    Args:
        cursor_dir: Path to ~/.cursor directory
        is_dev: If True, adds ROAMPAL_DEV=1 to env section
        force: If True, overwrite existing config even if different
    """
    print(f"{BOLD}Configuring Cursor...{RESET}")

    # Ensure directory exists (may not if detected via PATH on fresh install)
    cursor_dir.mkdir(parents=True, exist_ok=True)

    # Cursor uses ~/.cursor/mcp.json
    mcp_config_path = cursor_dir / "mcp.json"
    expected_env = {"ROAMPAL_DEV": "1"} if is_dev else {}
    roampal_server_config = {
        "command": sys.executable,
        "args": ["-m", "roampal.mcp.server"],
        "env": expected_env
    }

    mcp_config = {"mcpServers": {"roampal-core": roampal_server_config}}
    mcp_needs_write = True

    # Check existing MCP config
    if mcp_config_path.exists():
        try:
            existing = json.loads(mcp_config_path.read_text())
            existing_config = existing.get("mcpServers", {}).get("roampal-core", {})

            if existing_config:
                # Check if config matches
                if (existing_config.get("args") == ["-m", "roampal.mcp.server"] and
                    existing_config.get("env", {}) == expected_env):
                    print(f"  {GREEN}[OK] roampal-core already configured correctly in {mcp_config_path}{RESET}")
                    mcp_needs_write = False
                else:
                    # Config differs
                    if not force:
                        print(f"  {YELLOW}roampal-core config differs:{RESET}")
                        print(f"    Current: args={existing_config.get('args')}, env={existing_config.get('env', {})}")
                        print(f"    New:     args={roampal_server_config['args']}, env={expected_env}")
                        print(f"  {YELLOW}Use --force to overwrite{RESET}")
                        mcp_needs_write = False
                    else:
                        # Force - merge with existing
                        existing["mcpServers"]["roampal-core"] = roampal_server_config
                        mcp_config = existing
                        print(f"  {GREEN}Updated MCP config: {mcp_config_path} (forced){RESET}")
            else:
                # No roampal-core entry - add it
                if "mcpServers" in existing:
                    existing["mcpServers"]["roampal-core"] = roampal_server_config
                else:
                    existing["mcpServers"] = {"roampal-core": roampal_server_config}
                mcp_config = existing
        except Exception as e:
            logger.warning(f"Failed to parse existing MCP config: {e}")

    if mcp_needs_write:
        mcp_config_path.write_text(json.dumps(mcp_config, indent=2))
        if not mcp_config_path.exists() or "forced" not in str(mcp_needs_write):
            print(f"  {GREEN}Created MCP config: {mcp_config_path}{RESET}")

    # Cursor 1.7+ supports hooks - create ~/.cursor/hooks.json
    hooks_config_path = cursor_dir / "hooks.json"

    # v0.3.2: Use centralized _build_hook_command() for consistency
    submit_cmd = _build_hook_command("user_prompt_submit_hook", is_dev)
    stop_cmd = _build_hook_command("stop_hook", is_dev)

    expected_hooks = {
        "beforeSubmitPrompt": [{"command": submit_cmd}],
        "stop": [{"command": stop_cmd}]
    }
    hooks_config = {"version": 1, "hooks": expected_hooks}
    hooks_needs_write = True

    # Check existing hooks config
    if hooks_config_path.exists():
        try:
            existing = json.loads(hooks_config_path.read_text())
            existing_hooks = existing.get("hooks", {})

            # Check if roampal hooks already match
            existing_submit = existing_hooks.get("beforeSubmitPrompt", [])
            existing_stop = existing_hooks.get("stop", [])

            submit_matches = (existing_submit == expected_hooks["beforeSubmitPrompt"])
            stop_matches = (existing_stop == expected_hooks["stop"])

            if submit_matches and stop_matches:
                print(f"  {GREEN}[OK] Hooks already configured correctly in {hooks_config_path}{RESET}")
                hooks_needs_write = False
            elif existing_submit or existing_stop:
                # Hooks exist but differ
                if not force:
                    print(f"  {YELLOW}Cursor hooks config differs:{RESET}")
                    if not submit_matches:
                        print(f"    beforeSubmitPrompt: current differs from expected")
                    if not stop_matches:
                        print(f"    stop: current differs from expected")
                    print(f"  {YELLOW}Use --force to overwrite{RESET}")
                    hooks_needs_write = False
                else:
                    # Force - merge with existing
                    existing["hooks"]["beforeSubmitPrompt"] = expected_hooks["beforeSubmitPrompt"]
                    existing["hooks"]["stop"] = expected_hooks["stop"]
                    if "version" not in existing:
                        existing["version"] = 1
                    hooks_config = existing
                    print(f"  {GREEN}Updated hooks config: {hooks_config_path} (forced){RESET}")
            else:
                # No existing roampal hooks - add them
                if "hooks" in existing:
                    existing["hooks"]["beforeSubmitPrompt"] = expected_hooks["beforeSubmitPrompt"]
                    existing["hooks"]["stop"] = expected_hooks["stop"]
                else:
                    existing["hooks"] = expected_hooks
                if "version" not in existing:
                    existing["version"] = 1
                hooks_config = existing
        except Exception as e:
            logger.warning(f"Failed to parse existing hooks config: {e}")

    if hooks_needs_write:
        hooks_config_path.write_text(json.dumps(hooks_config, indent=2))
        print(f"  {GREEN}Created hooks config: {hooks_config_path}{RESET}")
        print(f"  {GREEN}  - beforeSubmitPrompt hook (injects scoring + memories){RESET}")
        print(f"  {GREEN}  - stop hook (enforces record_response){RESET}")

    print(f"  {GREEN}Cursor configured!{RESET}\n")


def configure_opencode(is_dev: bool = False, force: bool = False):
    """Configure OpenCode MCP and plugin.

    Args:
        is_dev: If True, adds ROAMPAL_DEV=1 to env section
        force: If True, overwrite existing config even if different
    """
    print(f"{BOLD}Configuring OpenCode...{RESET}")

    # OpenCode config locations (XDG Base Directory spec)
    if sys.platform == "win32":
        config_dir = Path.home() / ".config" / "opencode"
    else:
        xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        config_dir = Path(xdg_config) / "opencode"

    config_file = config_dir / "opencode.json"
    plugin_dir = config_dir / "plugins"

    # Create directories if they don't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Configure MCP server in opencode.json
    # =========================================================================
    # Both Claude Code and OpenCode share the same server (27182/27183)
    # ChromaDB doesn't support concurrent access, so no port isolation
    # Conversation IDs provide session isolation instead
    # Ensure PYTHONPATH includes roampal's parent dir so the module is found
    # regardless of OpenCode's cwd (which may differ from the install location)
    roampal_root = str(Path(__file__).parent.parent.resolve())
    expected_env = {"PYTHONPATH": roampal_root, "ROAMPAL_PLATFORM": "opencode"}
    if is_dev:
        expected_env["ROAMPAL_DEV"] = "1"
    roampal_mcp_config = {
        "type": "local",
        "command": [sys.executable, "-m", "roampal.mcp.server"],
        "enabled": True,
    }
    if expected_env:
        roampal_mcp_config["environment"] = expected_env

    # Load existing config or create new
    config = {}
    mcp_needs_write = True

    if config_file.exists():
        try:
            config = json.loads(config_file.read_text())
            existing_mcp = config.get("mcp", {}).get("roampal-core", {})

            if existing_mcp:
                # Check if config matches
                existing_cmd = existing_mcp.get("command", [])
                existing_env = existing_mcp.get("environment", {})

                if (existing_cmd == roampal_mcp_config["command"] and
                    existing_env == expected_env):
                    print(f"  {GREEN}[OK] roampal-core MCP already configured correctly{RESET}")
                    mcp_needs_write = False
                elif not force:
                    print(f"  {YELLOW}roampal-core MCP config differs:{RESET}")
                    print(f"    Current: command={existing_cmd}")
                    print(f"    New:     command={roampal_mcp_config['command']}")
                    print(f"  {YELLOW}Use --force to overwrite{RESET}")
                    mcp_needs_write = False
        except Exception as e:
            logger.warning(f"Failed to parse existing opencode.json: {e}")

    if mcp_needs_write:
        if "mcp" not in config:
            config["mcp"] = {}
        config["mcp"]["roampal-core"] = roampal_mcp_config
        config_file.write_text(json.dumps(config, indent=2))
        print(f"  {GREEN}Created MCP config: {config_file}{RESET}")

    # =========================================================================
    # 2. Install TypeScript plugin
    # =========================================================================
    plugin_file = plugin_dir / "roampal.ts"
    plugin_source = Path(__file__).parent / "plugins" / "opencode" / "roampal.ts"

    plugin_needs_write = True

    if plugin_file.exists():
        # Check if plugin content matches
        try:
            existing_content = plugin_file.read_text(encoding="utf-8")
            source_content = plugin_source.read_text(encoding="utf-8") if plugin_source.exists() else ""

            if existing_content == source_content:
                print(f"  {GREEN}[OK] roampal plugin already installed{RESET}")
                plugin_needs_write = False
            elif not force:
                print(f"  {YELLOW}roampal plugin differs from source{RESET}")
                print(f"  {YELLOW}Use --force to overwrite{RESET}")
                plugin_needs_write = False
        except Exception as e:
            logger.warning(f"Failed to compare plugin files: {e}")

    if plugin_needs_write:
        if plugin_source.exists():
            shutil.copy(plugin_source, plugin_file)
            print(f"  {GREEN}Installed plugin: {plugin_file}{RESET}")
        else:
            print(f"  {RED}Plugin source not found: {plugin_source}{RESET}")
            print(f"  {YELLOW}You may need to reinstall roampal{RESET}")

    # =========================================================================
    # 3. Remind user about shared server
    # =========================================================================
    shared_port = 27183 if is_dev else 27182
    print(f"  {GREEN}OpenCode configured!{RESET}")
    print(f"  {GREEN}  Server port: {shared_port}{RESET}")
    print(f"  {YELLOW}Note: Server auto-starts on first message. To stop: roampal stop{RESET}")
    print()


def cmd_start(args):
    """Start the Roampal server."""

    # Determine port based on mode (DEV=27183, PROD=27182)
    # User can override with --port
    is_dev = is_dev_mode(args)
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = args.port if args.port != PROD_PORT else default_port  # Use default unless explicitly overridden

    # Handle dev mode - uses Roampal_DEV folder
    if is_dev:
        os.environ["ROAMPAL_DEV"] = "1"
        data_path = get_data_dir(dev=True)
        print(f"{YELLOW}DEV MODE{RESET} - Isolated from production")
        print(f"  Data path: {data_path}")
        print(f"  Port: {port} (PROD uses {PROD_PORT})\n")
    else:
        data_path = get_data_dir(dev=False)
        print(f"{GREEN}PROD MODE{RESET}")
        print(f"  Data path: {data_path}")
        print(f"  Port: {port}\n")

    print(f"{BOLD}Starting Roampal server...{RESET}\n")

    host = args.host or "127.0.0.1"

    print(f"Server: http://{host}:{port}")
    print(f"Hooks endpoint: http://{host}:{port}/api/hooks/get-context")
    print(f"Health check: http://{host}:{port}/api/health")
    print(f"\nPress Ctrl+C to stop.\n")

    # Import and start server
    from roampal.server.main import start_server
    start_server(host=host, port=port)


def cmd_stop(args):
    """Stop the Roampal server."""

    is_dev = is_dev_mode(args)
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = args.port if args.port else default_port

    print(f"{BOLD}Stopping Roampal server on port {port}...{RESET}\n")

    killed = False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split("\n"):
                if f"127.0.0.1:{port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    if pid:
                        subprocess.run(
                            ["taskkill", "/pid", pid, "/f"],
                            capture_output=True, timeout=5
                        )
                        print(f"  {GREEN}Killed server process (PID {pid}){RESET}")
                        killed = True
                    break
        except Exception as e:
            print(f"  {RED}Error finding server process: {e}{RESET}")
    else:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5
            )
            pid = result.stdout.strip().split("\n")[0] if result.stdout.strip() else ""
            if pid:
                subprocess.run(
                    ["kill", "-9", pid],
                    capture_output=True, timeout=5
                )
                print(f"  {GREEN}Killed server process (PID {pid}){RESET}")
                killed = True
        except Exception as e:
            print(f"  {RED}Error finding server process: {e}{RESET}")

    if not killed:
        print(f"  {YELLOW}No server found on port {port}{RESET}")
        return 1
    else:
        print(f"\n{GREEN}Server stopped.{RESET}")
        return 0


def cmd_status(args):
    """Check Roampal server status and MCP configuration."""
    import httpx

    json_mode = getattr(args, 'json_output', False)

    if not json_mode:
        print_update_notice()

    host = args.host or "127.0.0.1"
    is_dev = is_dev_mode(args)
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = args.port if args.port and args.port != PROD_PORT else default_port
    url = f"http://{host}:{port}/api/health"

    # Collect MCP config info
    mcp_status = "not_found"
    mcp_detail = {}
    claude_json_path = Path.home() / ".claude.json"
    if claude_json_path.exists():
        try:
            claude_json = json.loads(claude_json_path.read_text(encoding='utf-8'))
            roampal_config = claude_json.get("mcpServers", {}).get("roampal-core", {})
            if roampal_config:
                mcp_status = "configured"
                mcp_detail = {
                    "path": str(claude_json_path),
                    "command": roampal_config.get("command", "N/A"),
                    "args": roampal_config.get("args", []),
                }
                if roampal_config.get("env"):
                    mcp_detail["env"] = roampal_config["env"]
            else:
                mcp_status = "not_configured"
        except Exception as e:
            mcp_status = "error"
            mcp_detail = {"error": str(e)}

    # Collect server info
    server_status = "unknown"
    server_detail = {}
    try:
        response = httpx.get(url, timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            server_status = "running"
            server_detail = {
                "port": port,
                "mode": "dev" if is_dev else "prod",
                "memory_initialized": data.get("memory_initialized", False),
                "timestamp": data.get("timestamp", "N/A"),
            }
        else:
            server_status = "error"
            server_detail = {"status_code": response.status_code}
    except httpx.ConnectError:
        server_status = "stopped"
        server_detail = {"port": port, "mode": "dev" if is_dev else "prod"}
    except Exception as e:
        server_status = "error"
        server_detail = {"error": str(e)}

    if json_mode:
        result = {
            "mcp": {"status": mcp_status, **mcp_detail},
            "server": {"status": server_status, **server_detail},
        }
        print(json.dumps(result, indent=2))
        return 0 if server_status == "running" else 1

    # Human-readable output
    mode_str = f"{YELLOW}DEV{RESET}" if is_dev else f"{GREEN}PROD{RESET}"

    print(f"{BOLD}MCP Configuration:{RESET}")
    if mcp_status == "configured":
        print(f"  Location: {GREEN}{mcp_detail['path']}{RESET} (user scope)")
        print(f"  Command:  {mcp_detail['command']} {' '.join(mcp_detail.get('args', []))}")
        if mcp_detail.get("env"):
            print(f"  Env:      {mcp_detail['env']}")
        print(f"  Status:   {GREEN}[OK] Configured{RESET}")
    elif mcp_status == "not_configured":
        print(f"  Status:   {YELLOW}Not configured{RESET}")
        print(f"  Run: roampal init")
    elif mcp_status == "error":
        print(f"  {RED}Error reading config: {mcp_detail.get('error')}{RESET}")
    else:
        print(f"  Status:   {YELLOW}~/.claude.json not found{RESET}")
        print(f"  Run: roampal init")

    print()

    print(f"{BOLD}Server Status:{RESET}")
    if server_status == "running":
        print(f"  Mode: {mode_str}")
        print(f"  Status: {GREEN}RUNNING{RESET}")
        print(f"  Port: {port}")
        print(f"  Memory initialized: {server_detail.get('memory_initialized', False)}")
        print(f"  Timestamp: {server_detail.get('timestamp', 'N/A')}")
        return 0
    elif server_status == "stopped":
        print(f"  Mode: {mode_str}")
        print(f"  Status: {YELLOW}NOT RUNNING{RESET}")
        start_cmd = "roampal start --dev" if is_dev else "roampal start"
        print(f"\n  Start with: {start_cmd}")
        return 1
    else:
        print(f"  {RED}Error: {server_detail.get('error', server_detail.get('status_code', 'unknown'))}{RESET}")
        return 1


def cmd_stats(args):
    """Show memory statistics."""
    import httpx

    json_mode = getattr(args, 'json_output', False)

    if not json_mode:
        print_update_notice()

    host = args.host or "127.0.0.1"
    is_dev = is_dev_mode(args)
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = args.port if args.port and args.port != PROD_PORT else default_port
    url = f"http://{host}:{port}/api/stats"

    try:
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()

            if json_mode:
                data["port"] = port
                data["mode"] = "dev" if is_dev else "prod"
                print(json.dumps(data, indent=2))
                return 0

            mode_str = f"{YELLOW}DEV{RESET}" if is_dev else f"{GREEN}PROD{RESET}"
            print(f"{BOLD}Memory Statistics ({mode_str}):{RESET}\n")
            print(f"Data path: {data.get('data_path', 'N/A')}")
            print(f"Port: {port}")
            print(f"\nCollections:")
            for name, info in data.get("collections", {}).items():
                count = info.get("count", 0)
                print(f"  {name}: {count} items")
            return 0
        else:
            if json_mode:
                print(json.dumps({"error": f"HTTP {response.status_code}"}, indent=2))
            else:
                print(f"{RED}Error getting stats: {response.status_code}{RESET}")
            return 1
    except httpx.ConnectError:
        if json_mode:
            print(json.dumps({"error": "server_not_running"}, indent=2))
        else:
            start_cmd = "roampal start --dev" if is_dev else "roampal start"
            print(f"{YELLOW}Server not running. Start with: {start_cmd}{RESET}")
        return 1
    except Exception as e:
        if json_mode:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"{RED}Error: {e}{RESET}")
        return 1


def cmd_ingest(args):
    """Ingest a document into the books collection."""
    import asyncio
    import httpx

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"{RED}File not found: {file_path}{RESET}")
        return

    # Read file content
    print(f"{BOLD}Ingesting:{RESET} {file_path.name}")

    try:
        # Detect file type and read content
        suffix = file_path.suffix.lower()
        content = None
        title = args.title or file_path.stem

        if suffix == '.txt':
            content = file_path.read_text(encoding='utf-8')
        elif suffix == '.md':
            content = file_path.read_text(encoding='utf-8')
        elif suffix == '.pdf':
            # Try to use pypdf if available
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                content = ""
                for page in reader.pages:
                    content += page.extract_text() + "\n"
                print(f"  Extracted {len(reader.pages)} pages from PDF")
            except ImportError:
                print(f"{RED}PDF support requires pypdf: pip install pypdf{RESET}")
                return
        else:
            # Try to read as text
            try:
                content = file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                print(f"{RED}Cannot read file as text. Supported: .txt, .md, .pdf{RESET}")
                return

        if not content or len(content.strip()) == 0:
            print(f"{YELLOW}File is empty or could not be read{RESET}")
            return

        print(f"  Content length: {len(content):,} characters")

        is_dev = is_dev_mode(args)
        data_path = None
        if is_dev:
            data_path = str(get_data_dir(dev=True))
            print(f"  {YELLOW}DEV MODE{RESET} - Using: {data_path}")

        # Try to use running server first (data immediately searchable)
        # Use dev port if --dev flag
        host = "127.0.0.1"
        port = DEV_PORT if is_dev else PROD_PORT
        server_url = f"http://{host}:{port}/api/ingest"

        try:
            response = httpx.post(
                server_url,
                json={
                    "content": content,
                    "title": title,
                    "source": str(file_path),
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap
                },
                timeout=300.0  # 5 min timeout for very large files
            )

            if response.status_code == 200:
                data = response.json()
                print(f"\n{GREEN}Success!{RESET} Stored '{title}' in {data['chunks']} chunks")
                print(f"  (via running server - immediately searchable)")
                print(f"\nThe document is now searchable via:")
                print(f"  - search_memory(query, collections=['books'])")
                print(f"  - Automatic context injection via hooks")
                return
            else:
                print(f"  {YELLOW}Server error, falling back to direct storage...{RESET}")

        except httpx.ConnectError:
            print(f"  {YELLOW}Server not running, using direct storage...{RESET}")
            print(f"  {YELLOW}(Restart server for immediate searchability){RESET}")

        # Fallback: Store directly (requires server restart to be searchable)
        async def do_ingest():
            from roampal.backend.modules.memory import UnifiedMemorySystem

            mem = UnifiedMemorySystem(data_path=data_path)
            await mem.initialize()

            doc_ids = await mem.store_book(
                content=content,
                title=title,
                source=str(file_path),
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )

            return doc_ids

        doc_ids = asyncio.run(do_ingest())

        print(f"\n{GREEN}Success!{RESET} Stored '{title}' in {len(doc_ids)} chunks")
        print(f"\nThe document is now searchable via:")
        print(f"  - search_memory(query, collections=['books'])")
        print(f"  - Automatic context injection via hooks")
        print(f"\n{YELLOW}Note: Restart 'roampal start' for immediate searchability{RESET}")

    except Exception as e:
        print(f"{RED}Error ingesting file: {e}{RESET}")
        raise


def cmd_remove(args):
    """Remove a book from the books collection."""
    import asyncio
    import httpx

    title = args.title
    print(f"{BOLD}Removing book:{RESET} {title}\n")

    # v0.2.0: Handle --dev flag by setting env var
    if getattr(args, 'dev', False):
        os.environ["ROAMPAL_DEV"] = "1"
        print(f"  {YELLOW}DEV mode{RESET}")

    # Try running server first
    host = "127.0.0.1"
    port = get_port(args)
    server_url = f"http://{host}:{port}/api/remove-book"

    try:
        response = httpx.post(
            server_url,
            json={"title": title},
            timeout=30.0
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("removed", 0) > 0:
                print(f"{GREEN}Success!{RESET} Removed '{title}' ({data['removed']} chunks)")
                if data.get("cleaned_kg_refs", 0) > 0:
                    print(f"  Cleaned {data['cleaned_kg_refs']} Action KG references")
            else:
                print(f"{YELLOW}No book found with title '{title}'{RESET}")
            return
        else:
            print(f"  {YELLOW}Server error, falling back to direct removal...{RESET}")

    except httpx.ConnectError:
        print(f"  {YELLOW}Server not running, using direct removal...{RESET}")

    # Fallback: Remove directly
    async def do_remove():
        from roampal.backend.modules.memory import UnifiedMemorySystem

        mem = UnifiedMemorySystem()
        await mem.initialize()
        return await mem.remove_book(title)

    result = asyncio.run(do_remove())

    if result.get("removed", 0) > 0:
        print(f"\n{GREEN}Success!{RESET} Removed '{title}' ({result['removed']} chunks)")
        if result.get("cleaned_kg_refs", 0) > 0:
            print(f"  Cleaned {result['cleaned_kg_refs']} Action KG references")
    else:
        print(f"{YELLOW}No book found with title '{title}'{RESET}")


def cmd_doctor(args):
    """Diagnose Roampal installation and configuration."""
    import asyncio

    print(f"{BOLD}Roampal Doctor - Diagnostics{RESET}\n")

    is_dev = is_dev_mode(args)
    mode_str = f"{YELLOW}DEV{RESET}" if is_dev else f"{GREEN}PROD{RESET}"
    print(f"Mode: {mode_str}\n")

    checks_passed = 0
    checks_failed = 0
    checks_warned = 0

    def check_pass(msg):
        nonlocal checks_passed
        checks_passed += 1
        print(f"  {GREEN}[OK]{RESET} {msg}")

    def check_fail(msg):
        nonlocal checks_failed
        checks_failed += 1
        print(f"  {RED}[FAIL]{RESET} {msg}")

    def check_warn(msg):
        nonlocal checks_warned
        checks_warned += 1
        print(f"  {YELLOW}[WARN]{RESET} {msg}")

    # 1. Check Python version
    print(f"{BOLD}Python Environment:{RESET}")
    py_version = sys.version_info
    if py_version >= (3, 10):
        check_pass(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        check_fail(f"Python {py_version.major}.{py_version.minor} (need 3.10+)")

    check_pass(f"Executable: {sys.executable}")

    # 2. Check roampal version
    print(f"\n{BOLD}Roampal Version:{RESET}")
    try:
        from roampal import __version__
        check_pass(f"roampal v{__version__}")
    except ImportError as e:
        check_fail(f"Cannot import roampal: {e}")

    # 3. Check config files
    print(f"\n{BOLD}Configuration Files:{RESET}")
    home = Path.home()

    # Claude Code configs
    claude_dir = home / ".claude"
    if claude_dir.exists():
        check_pass(f"~/.claude exists")

        settings_path = claude_dir / "settings.json"
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text())
                if "hooks" in settings:
                    check_pass("settings.json has hooks configured")
                else:
                    check_warn("settings.json missing hooks (run 'roampal init')")
            except json.JSONDecodeError as e:
                check_fail(f"settings.json invalid JSON: {e}")
        else:
            check_warn("settings.json not found (run 'roampal init')")

        # Check ~/.claude.json for user-scope MCP (v0.2.5+)
        claude_json_path = home / ".claude.json"
        if claude_json_path.exists():
            try:
                claude_json = json.loads(claude_json_path.read_text())
                if "mcpServers" in claude_json and "roampal-core" in claude_json["mcpServers"]:
                    check_pass("~/.claude.json has roampal-core server (user scope)")
                else:
                    check_warn("~/.claude.json missing roampal-core (run 'roampal init')")
            except json.JSONDecodeError as e:
                check_fail(f"~/.claude.json invalid JSON: {e}")
        else:
            check_warn("~/.claude.json not found (run 'roampal init')")

        # Check for old broken config location
        old_mcp_path = claude_dir / ".mcp.json"
        if old_mcp_path.exists():
            try:
                old_mcp = json.loads(old_mcp_path.read_text())
                if "mcpServers" in old_mcp and "roampal-core" in old_mcp["mcpServers"]:
                    check_warn("Old config at ~/.claude/.mcp.json (run 'roampal init' to migrate)")
            except Exception as e:
                logger.debug(f"Could not parse old .mcp.json: {e}")
    else:
        check_warn("~/.claude not found (Claude Code not installed?)")

    # Cursor configs
    cursor_dir = home / ".cursor"
    if cursor_dir.exists():
        check_pass("~/.cursor exists")

        mcp_path = cursor_dir / "mcp.json"
        if mcp_path.exists():
            try:
                mcp = json.loads(mcp_path.read_text())
                if "mcpServers" in mcp and "roampal-core" in mcp["mcpServers"]:
                    check_pass("mcp.json has roampal-core server")
                else:
                    check_warn("mcp.json missing roampal-core (run 'roampal init --cursor')")
            except json.JSONDecodeError as e:
                check_fail(f"mcp.json invalid JSON: {e}")
        else:
            check_warn("mcp.json not found (run 'roampal init --cursor')")

        # Cursor hooks (1.7+)
        hooks_path = cursor_dir / "hooks.json"
        if hooks_path.exists():
            try:
                hooks = json.loads(hooks_path.read_text())
                if "hooks" in hooks and "beforeSubmitPrompt" in hooks["hooks"]:
                    check_pass("hooks.json has beforeSubmitPrompt configured")
                else:
                    check_warn("hooks.json missing beforeSubmitPrompt (run 'roampal init --cursor')")
                if "hooks" in hooks and "stop" in hooks["hooks"]:
                    check_pass("hooks.json has stop hook configured")
                else:
                    check_warn("hooks.json missing stop hook (run 'roampal init --cursor')")
            except json.JSONDecodeError as e:
                check_fail(f"hooks.json invalid JSON: {e}")
        else:
            check_warn("hooks.json not found (run 'roampal init --cursor' for Cursor 1.7+)")

    # 4. Check data directory
    print(f"\n{BOLD}Data Directory:{RESET}")
    data_dir = get_data_dir(dev=is_dev)
    if data_dir.exists():
        check_pass(f"Data directory exists: {data_dir}")
        chromadb_path = data_dir / "chromadb"
        if chromadb_path.exists():
            check_pass("ChromaDB directory exists")
        else:
            check_warn("ChromaDB not initialized yet (first use will create it)")
    else:
        check_warn(f"Data directory not created: {data_dir}")

    # 5. Check MCP server can start and list tools
    print(f"\n{BOLD}MCP Server:{RESET}")
    try:
        # Import the server module - this validates the code compiles
        import roampal.mcp.server as mcp_module
        check_pass("MCP server module loads")

        # Check that tools are defined (this catches syntax errors like false/False)
        # The actual list_tools is a decorated async function, so we check the TOOLS dict
        if hasattr(mcp_module, 'TOOLS'):
            tool_count = len(mcp_module.TOOLS)
            check_pass(f"Tools defined: {tool_count}")
        else:
            # Try to find tools another way - check if the server starts
            check_pass("Server module valid (tools loaded at runtime)")

    except SyntaxError as e:
        check_fail(f"MCP server syntax error: {e}")
    except NameError as e:
        check_fail(f"MCP server name error: {e}")
    except Exception as e:
        check_fail(f"MCP server import failed: {e}")

    # 6. Check memory system initialization
    print(f"\n{BOLD}Memory System:{RESET}")
    try:
        async def test_memory():
            from roampal.backend.modules.memory import UnifiedMemorySystem

            data_path = str(get_data_dir(dev=is_dev)) if is_dev else None
            mem = UnifiedMemorySystem(data_path=data_path)
            await mem.initialize()
            return mem

        mem = asyncio.run(test_memory())
        check_pass("Memory system initializes")

        # Check collections
        if hasattr(mem, 'collections') and mem.collections:
            collection_names = list(mem.collections.keys())
            check_pass(f"Collections: {', '.join(collection_names)}")
        else:
            check_warn("No collections found")

    except Exception as e:
        check_fail(f"Memory system failed: {e}")

    # 7. Check dependencies
    print(f"\n{BOLD}Dependencies:{RESET}")
    deps = [
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
        ("mcp", "mcp"),
        ("httpx", "httpx"),
        ("fastapi", "fastapi"),
    ]

    for import_name, display_name in deps:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "?")
            check_pass(f"{display_name} v{version}")
        except ImportError:
            check_fail(f"{display_name} not installed")

    # Summary
    print(f"\n{BOLD}{'='*50}{RESET}")
    total = checks_passed + checks_failed + checks_warned
    if checks_failed == 0:
        print(f"{GREEN}All checks passed!{RESET} ({checks_passed}/{total})")
        if checks_warned > 0:
            print(f"{YELLOW}{checks_warned} warnings{RESET}")
        return 0
    else:
        print(f"{RED}{checks_failed} checks failed{RESET}, {checks_passed} passed, {checks_warned} warnings")
        print(f"\nRun {BLUE}roampal init{RESET} to fix configuration issues.")
        return 1


def cmd_books(args):
    """List all books in the books collection."""
    import asyncio
    import httpx

    print(f"{BOLD}Books in memory:{RESET}\n")

    # Try running server first
    host = "127.0.0.1"
    port = get_port()
    server_url = f"http://{host}:{port}/api/books"

    books = None

    try:
        response = httpx.get(server_url, timeout=10.0)
        if response.status_code == 200:
            books = response.json().get("books", [])
    except httpx.ConnectError:
        pass

    # Fallback: List directly
    if books is None:
        async def do_list():
            from roampal.backend.modules.memory import UnifiedMemorySystem

            mem = UnifiedMemorySystem()
            await mem.initialize()
            return await mem.list_books()

        books = asyncio.run(do_list())

    if not books:
        print(f"{YELLOW}No books found.{RESET}")
        print(f"\nAdd books with: roampal ingest <file>")
        return

    for book in books:
        print(f"  {GREEN}{book['title']}{RESET}")
        print(f"    Source: {book.get('source', 'unknown')}")
        print(f"    Chunks: {book.get('chunk_count', 0)}")
        if book.get('created_at'):
            print(f"    Added: {book['created_at'][:10]}")
        print()


def cmd_score(args):
    """Score the last exchange using the sidecar (Haiku via claude CLI)."""
    import hashlib
    import httpx

    port = get_port(args)
    base_url = f"http://127.0.0.1:{port}"

    # Lock file to prevent concurrent sidecar processes from racing
    lock_dir = Path.home() / ".cache" / "roampal"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / "sidecar.lock"

    try:
        # Check if another sidecar is already running
        if lock_file.exists():
            try:
                lock_data = json.loads(lock_file.read_text())
                lock_pid = lock_data.get("pid", 0)
                lock_time = lock_data.get("time", 0)
                import time as _time
                # If lock is older than 3 minutes, it's stale — ignore it
                if _time.time() - lock_time < 180:
                    # Check if the PID is still alive
                    try:
                        os.kill(lock_pid, 0)
                        logger.debug(f"Sidecar already running (pid={lock_pid}), skipping")
                        return
                    except (OSError, ProcessLookupError):
                        pass  # Process dead, lock is stale
            except (json.JSONDecodeError, ValueError):
                pass  # Corrupted lock file, proceed

        # Acquire lock
        import time as _time
        lock_file.write_text(json.dumps({"pid": os.getpid(), "time": _time.time()}))
    except Exception:
        pass  # If locking fails, continue anyway — dedup check is the safety net

    try:
        _cmd_score_inner(args, base_url, port)
    finally:
        # Release lock
        try:
            lock_file.unlink(missing_ok=True)
        except Exception:
            pass


def _cmd_score_inner(args, base_url, port):
    """Inner scoring logic, called under lock."""
    import hashlib
    import httpx

    if args.from_hook:
        # Read Claude Code Stop hook event data from stdin
        import sys
        try:
            hook_data = json.loads(sys.stdin.read())
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to read hook data from stdin: {e}")
            return

        transcript_path = hook_data.get("transcript_path", "")
        session_id = hook_data.get("session_id", "default")

        if not transcript_path:
            logger.error("No transcript_path in hook data")
            return

        # Parse transcript for last user + assistant messages
        user_msg, assistant_msg, followup = _parse_last_exchange(transcript_path)
        if not user_msg or not assistant_msg:
            logger.debug("No complete exchange found in transcript")
            return
    else:
        # Background sidecar or manual mode
        transcript_arg = getattr(args, 'transcript', None)
        session_id = getattr(args, 'session_id', None) or "default"
        try:
            if transcript_arg and Path(transcript_arg).exists():
                transcript_path = transcript_arg
            else:
                transcript_path = _find_latest_transcript(session_id if session_id != "default" else None)
            if not transcript_path:
                logger.debug("No Claude Code transcript found")
                return
            if not getattr(args, 'from_hook', False) and session_id == "default":
                print(f"{YELLOW}Scoring from transcript: {Path(transcript_path).name}{RESET}")
            user_msg, assistant_msg, followup = _parse_last_exchange(transcript_path)
            if not user_msg or not assistant_msg:
                logger.debug("No complete exchange found in transcript")
                return
        except Exception as e:
            logger.error(f"Score error: {e}")
            return

    # Fingerprint check — skip if this exchange was already summarized
    fingerprint = hashlib.md5(f"{user_msg[:200]}:{assistant_msg[:200]}".encode()).hexdigest()[:12]

    try:
        check_resp = httpx.post(
            f"{base_url}/api/search",
            json={
                "query": "",
                "collections": ["working"],
                "limit": 1,
                "sort_by": "recency",
                "metadata_filters": {"exchange_fingerprint": fingerprint}
            },
            timeout=5.0
        )
        if check_resp.status_code == 200 and check_resp.json().get("count", 0) > 0:
            logger.debug(f"Exchange already summarized (fingerprint={fingerprint}), skipping")
            return
    except Exception:
        pass  # If check fails, proceed with scoring

    # Call sidecar to summarize + score
    from roampal.sidecar_service import summarize_and_score

    result = summarize_and_score(user_msg, assistant_msg, followup=followup)
    if not result:
        logger.error("Sidecar failed to summarize/score exchange")
        return

    summary = result.get("summary", "")
    outcome = result.get("outcome", "unknown")

    # Store the summary as a working memory via the server
    try:
        store_resp = httpx.post(
            f"{base_url}/api/hooks/stop",
            json={
                "conversation_id": session_id,
                "user_message": user_msg[:200],
                "assistant_response": summary,
                "metadata": {
                    "memory_type": "exchange_summary",
                    "sidecar_outcome": outcome,
                    "exchange_fingerprint": fingerprint,
                    "original_user_msg_length": len(user_msg),
                    "original_assistant_msg_length": len(assistant_msg)
                }
            },
            timeout=10.0
        )

        if store_resp.status_code == 200:
            store_data = store_resp.json()
            doc_id = store_data.get("doc_id", "")

            # Score the exchange if we got a doc_id
            if doc_id and outcome != "unknown":
                httpx.post(
                    f"{base_url}/api/record-outcome",
                    json={
                        "conversation_id": session_id,
                        "outcome": outcome,
                        "memory_scores": {doc_id: outcome}
                    },
                    timeout=10.0
                )

            if not args.from_hook:
                print(f"{GREEN}Scored: {outcome}{RESET}")
                print(f"Summary: {summary}")
        else:
            logger.error(f"Failed to store exchange: {store_resp.status_code}")

    except httpx.ConnectError:
        logger.error("Roampal server not running")
    except Exception as e:
        logger.error(f"Error storing exchange: {e}")


def _parse_last_exchange(transcript_path: str) -> tuple:
    """Parse a Claude Code JSONL transcript for the last COMPLETE user+assistant exchange.

    Returns (user_msg, assistant_msg, followup):
    - user_msg: The user message that the assistant was responding to
    - assistant_msg: The assistant's full response (all text blocks from that turn)
    - followup: The user's next message (for scoring context), may be empty

    Uses a turn-based approach: each user string message starts a new turn.
    All assistant text blocks between two user messages belong to one turn.

    Handles:
    - Real user messages: type="user", content is a plain string
    - Tool results: type="user", content is a list → SKIP (not a turn boundary)
    - Real assistant text: type="assistant", content has type="text" entries
    - Tool calls/thinking: type="assistant", no substantive text → SKIP
    - Noise types: progress, system, file-history-snapshot, queue-operation → SKIP
    - System-reminder/hook injection tags in user messages → STRIPPED
    """
    import re

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Build turns: each turn = (user_text, assistant_text)
        # A new turn starts when we see a real user message (string content)
        turns = []  # List of {"user": str, "assistant": str}
        current_turn = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                msg_type = entry.get("type", "")

                if msg_type not in ("user", "assistant"):
                    continue

                content = entry.get("message", {}).get("content", [])

                if msg_type == "user" and isinstance(content, str):
                    text = content.strip()
                    if not text:
                        continue
                    # Strip hook injection tags
                    text = re.sub(r'<system-reminder>.*?</system-reminder>', '', text, flags=re.DOTALL).strip()
                    text = re.sub(r'<task-notification>.*?</task-notification>', '', text, flags=re.DOTALL).strip()
                    text = re.sub(r'<roampal-[^>]*>.*?</roampal-[^>]*>', '', text, flags=re.DOTALL).strip()
                    if not text or len(text) < 2:
                        continue

                    # Save previous turn and start a new one
                    if current_turn is not None:
                        turns.append(current_turn)
                    current_turn = {"user": text, "assistant": ""}

                elif msg_type == "assistant" and isinstance(content, list) and current_turn is not None:
                    # Accumulate all substantive text for this turn
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            t = p.get("text", "").strip()
                            if t and t not in ("\n\n", "\n"):
                                if current_turn["assistant"]:
                                    current_turn["assistant"] += " " + t
                                else:
                                    current_turn["assistant"] = t

            except (json.JSONDecodeError, KeyError):
                continue

        # Don't forget the last in-progress turn
        if current_turn is not None:
            turns.append(current_turn)

        if not turns:
            return "", "", ""

        # Find the last turn with a substantive assistant response
        last_complete = None
        followup_turn = None

        for i in range(len(turns) - 1, -1, -1):
            if turns[i]["assistant"] and len(turns[i]["assistant"]) >= 10:
                last_complete = turns[i]
                # If there's a turn after this one, it's the follow-up
                if i + 1 < len(turns):
                    followup_turn = turns[i + 1]
                break

        if not last_complete:
            return "", "", ""

        user_msg = last_complete["user"]
        assistant_msg = last_complete["assistant"]
        followup = followup_turn["user"] if followup_turn else ""

        return user_msg, assistant_msg, followup

    except Exception as e:
        logger.error(f"Failed to parse transcript: {e}")
        return "", "", ""


def _find_latest_transcript(session_id: str = None) -> str:
    """Find a Claude Code transcript file.

    If session_id is provided, find the exact transcript for that session.
    Otherwise fall back to most recent (WARNING: may pick up sidecar transcripts).
    """
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        return ""

    if session_id:
        # Session ID IS the transcript filename in Claude Code
        matches = list(claude_dir.rglob(f"{session_id}.jsonl"))
        if matches:
            return str(matches[0])
        logger.warning(f"No transcript found for session {session_id}")

    # Fallback: most recent, but skip tiny files (likely sidecar/haiku transcripts)
    transcripts = sorted(claude_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    for t in transcripts:
        # Skip files under ~5KB — real sessions are much larger, haiku transcripts are tiny
        if t.stat().st_size > 5000:
            return str(t)
    return str(transcripts[0]) if transcripts else ""


def cmd_summarize(args):
    """Summarize existing long memories using the sidecar."""
    import httpx

    port = get_port(args)
    base_url = f"http://127.0.0.1:{port}"
    max_chars = args.max_chars
    collections_to_scan = [args.collection] if args.collection else ["working", "history"]
    dry_run = args.dry_run

    print(f"{BOLD}Summarizing memories over {max_chars} characters{RESET}")
    if dry_run:
        print(f"{YELLOW}DRY RUN -- no changes will be made{RESET}")

    from roampal.sidecar_service import get_backend_info
    backend = get_backend_info()
    print(f"Backend: {GREEN}{backend}{RESET}")

    if backend == "none available":
        print(f"\n{RED}No summarization backend available.{RESET}")
        print()
        print(f"  {BOLD}What was checked:{RESET}")
        print(f"    ROAMPAL_SUMMARIZE_MODEL  {RED}not set{RESET}  (opt-in to main model)")
        print(f"    ROAMPAL_SIDECAR_URL      {RED}not set{RESET}  (custom OpenAI-compatible API)")
        print(f"    ANTHROPIC_API_KEY         {RED}not set{RESET}  (Haiku direct)")
        print(f"    Zen free models           {RED}unavailable{RESET} (CLI only works inside OpenCode)")
        print(f"    Ollama                    {RED}not running{RESET} (http://localhost:11434)")
        print(f"    LM Studio                 {RED}not running{RESET} (http://localhost:1234)")
        print(f"    claude CLI                {RED}not found{RESET}")
        print()
        print(f"  {BOLD}Options (pick one):{RESET}")
        print(f"    1. {GREEN}Install Ollama{RESET} (recommended, free, local, ~14s/memory)")
        print(f"       ollama.com -> install -> ollama pull llama3.2:3b")
        print(f"    2. {GREEN}Set ANTHROPIC_API_KEY{RESET} (~$0.001/memory via Haiku)")
        print(f"       export ANTHROPIC_API_KEY=sk-ant-...")
        print(f"    3. {GREEN}Set ROAMPAL_SUMMARIZE_MODEL{RESET} (use your main model)")
        print(f"       export ROAMPAL_SUMMARIZE_MODEL=claude-sonnet-4-5-20250929")
        print(f"       export ANTHROPIC_API_KEY=sk-ant-...")
        print(f"    4. {GREEN}Set ROAMPAL_SIDECAR_URL{RESET} (any OpenAI-compatible endpoint)")
        print(f"       export ROAMPAL_SIDECAR_URL=https://api.groq.com/openai/v1")
        print(f"       export ROAMPAL_SIDECAR_KEY=your-key")
        print(f"       export ROAMPAL_SIDECAR_MODEL=llama-3.3-70b-versatile")
        print()
        print(f"  {YELLOW}Note:{RESET} OpenCode users don't need this -- memories auto-summarize")
        print(f"  during normal use via the plugin (1 per exchange, zero config).")
        return
    elif "claude -p" in backend:
        print(f"{YELLOW}Warning: claude -p is slow (~30-60s/memory) and unreliable (~40% success rate).{RESET}")
        print(f"For better results, install Ollama (ollama.com) or set ROAMPAL_SIDECAR_URL.{RESET}")

    # Small model disclaimer
    if "Ollama" in backend or "LM Studio" in backend:
        print(f"{YELLOW}Note: Local models may struggle with very long memories (>5000 chars).{RESET}")
        print(f"Those will be skipped if summarization fails. Use a larger model or API for best results.{RESET}")

    print()

    # Check server is running
    try:
        httpx.get(f"{base_url}/api/health", timeout=5.0)
    except Exception:
        print(f"{RED}Server not running. Start with: roampal start{RESET}")
        return

    from roampal.sidecar_service import summarize_only

    # Scan all collections first to count available memories
    all_candidates = []  # list of (coll_name, mem) tuples
    for coll_name in collections_to_scan:
        try:
            resp = httpx.post(
                f"{base_url}/api/search",
                json={
                    "query": "",
                    "collections": [coll_name],
                    "limit": 500,
                    "sort_by": "recency"
                },
                timeout=30.0
            )

            if resp.status_code != 200:
                print(f"{RED}Failed to search {coll_name}: {resp.status_code}{RESET}")
                continue

            results = resp.json().get("results", [])
            for r in results:
                content = r.get("content", "")
                metadata = r.get("metadata", {})
                # Skip already-summarized memories (have summarized_at timestamp)
                if metadata.get("summarized_at"):
                    continue
                if len(content) > max_chars:
                    all_candidates.append((coll_name, r))

        except Exception as e:
            print(f"{RED}Error scanning {coll_name}: {e}{RESET}")

    if not all_candidates:
        print(f"{GREEN}No memories need summarization.{RESET}")
        return

    # Show count and prompt for limit
    print(f"{BOLD}{len(all_candidates)} memories available for summarization{RESET}")

    # Group by collection for display
    by_coll = {}
    for coll_name, mem in all_candidates:
        by_coll.setdefault(coll_name, []).append(mem)
    for coll_name, mems in by_coll.items():
        print(f"  {coll_name}: {len(mems)} memories over {max_chars} chars")

    # Determine batch limit
    batch_limit = args.limit
    if batch_limit is not None and batch_limit <= 0:
        print(f"Nothing to do (limit={batch_limit}).")
        return
    if batch_limit is None and not dry_run:
        if _is_interactive():
            # Interactive: ask user how many
            print()
            try:
                user_input = input(f"How many to summarize? (Enter for all {len(all_candidates)}, or a number): ").strip()
                if user_input:
                    batch_limit = int(user_input)
                    if batch_limit <= 0:
                        print(f"Cancelled.")
                        return
                    print(f"Summarizing {batch_limit} memories...")
                else:
                    print(f"Summarizing all {len(all_candidates)} memories...")
            except (ValueError, EOFError):
                print(f"Summarizing all {len(all_candidates)} memories...")
        else:
            # Non-interactive: summarize all
            print(f"Summarizing all {len(all_candidates)} memories...")

    print()

    total_summarized = 0
    total_skipped = 0
    batch_count = 0

    for i, (coll_name, mem) in enumerate(all_candidates):
        if batch_limit is not None and batch_count >= batch_limit:
            remaining = len(all_candidates) - i
            print(f"  {YELLOW}Batch limit reached ({batch_limit}). {remaining} remaining -- run again to continue.{RESET}")
            break

        content = mem.get("content", "")
        doc_id = mem.get("id", mem.get("doc_id", ""))

        if dry_run:
            print(f"  [{i+1}/{len(all_candidates)}] {doc_id}: {len(content)} chars -> would summarize")
            if i < 3:  # Show sample previews for first 3
                print(f"    {YELLOW}(generating preview...){RESET}")
                try:
                    summary = summarize_only(content)
                    if summary:
                        print(f"    Preview: {summary[:100]}...")
                except Exception as e:
                    print(f"    {YELLOW}Preview failed: {e}{RESET}")
            total_summarized += 1
            batch_count += 1
            continue

        # Summarize
        summary = summarize_only(content)
        if not summary:
            print(f"  {YELLOW}[{i+1}] Failed to summarize {doc_id} ({len(content)} chars), skipping{RESET}")
            total_skipped += 1
            continue

        # Enforce summary length — if LLM returned >max_chars, truncate to prevent re-summarization loop
        if len(summary) > max_chars:
            summary = summary[:max_chars - 20] + "... [truncated]"

        # Update the memory with the summary
        try:
            update_resp = httpx.post(
                f"{base_url}/api/memory/update-content",
                json={
                    "doc_id": doc_id,
                    "collection": coll_name,
                    "new_content": summary
                },
                timeout=10.0
            )

            if update_resp.status_code == 200:
                total_summarized += 1
                batch_count += 1
                print(f"  [{i+1}/{len(all_candidates)}] {doc_id}: {len(content)} -> {len(summary)} chars")
            else:
                total_skipped += 1
                print(f"  {YELLOW}[{i+1}] Failed to update {doc_id}: {update_resp.status_code}{RESET}")
        except Exception as e:
            total_skipped += 1
            print(f"  {RED}[{i+1}] Error updating {doc_id}: {e}{RESET}")

    print()
    if dry_run:
        print(f"{YELLOW}DRY RUN: Would summarize {total_summarized} memories{RESET}")
    else:
        print(f"{GREEN}Summarized {total_summarized} memories{RESET}")
        if total_skipped > 0:
            print(f"{YELLOW}Skipped {total_skipped} (failed or too long for model){RESET}")


def cmd_context(args):
    """Output recent exchange context for platform hooks."""
    import httpx

    port = get_port(args)
    base_url = f"http://127.0.0.1:{port}"

    if args.recent_exchanges:
        try:
            # Search for recent exchange summaries
            resp = httpx.post(
                f"{base_url}/api/search",
                json={
                    "query": "",
                    "collections": ["working"],
                    "limit": 4,
                    "sort_by": "recency",
                    "metadata_filters": {"memory_type": "exchange_summary"}
                },
                timeout=10.0
            )

            if resp.status_code != 200:
                return

            results = resp.json().get("results", [])
            if not results:
                return

            # Format output for platform hook injection
            print("RECENT EXCHANGES (last 4):")
            for i, r in enumerate(results, 1):
                content = r.get("content", "")
                metadata = r.get("metadata", {})
                recency = metadata.get("recency", "")
                time_str = f"[{recency}] " if recency else ""
                # Truncate to keep output compact
                summary = content[:200] if content else "No content"
                print(f"{i}. {time_str}{summary}")

        except httpx.ConnectError:
            pass  # Server not running, fail silently for hooks
        except Exception:
            pass  # Fail silently for hooks


def _get_opencode_config_path() -> Path:
    """Get the opencode.json config path."""
    if sys.platform == "win32":
        config_dir = Path.home() / ".config" / "opencode"
    else:
        xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        config_dir = Path(xdg_config) / "opencode"
    return config_dir / "opencode.json"


def _detect_ollama_models() -> list:
    """Check if Ollama is running and return available models."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            models = []
            # Families that are embedding/vision-only models, not chat-capable
            embed_families = {"nomic-bert", "bert", "clip", "all-minilm"}
            for m in data.get("models", []):
                name = m.get("name", "")
                family = (m.get("details") or {}).get("family", "").lower()
                # Skip embedding models — they can't generate text for scoring
                if family in embed_families or "embed" in name.lower():
                    continue
                size_bytes = m.get("size", 0)
                size_gb = round(size_bytes / (1024**3), 1) if size_bytes else 0
                models.append({"name": name, "size_gb": size_gb, "source": "ollama"})
            return models
    except Exception:
        return []


def _detect_local_servers() -> list:
    """Probe known default ports for running OpenAI-compatible local inference servers.

    Uses concurrent.futures to probe all ports in parallel (~2s total).
    Ollama is excluded here (handled by _detect_ollama_models with richer metadata).

    Returns:
        List of dicts: [{name, port, server_label, source: "local"}]
    """
    import concurrent.futures

    # Known local inference server ports (Ollama excluded — handled separately)
    LOCAL_SERVERS = [
        (1234, "LM Studio"),
        (8080, "LocalAI / llama.cpp"),
        (1337, "Jan.ai"),
        (8000, "vLLM"),
        (5000, "text-generation-webui"),
        (4891, "GPT4All"),
        (5001, "KoboldCpp"),
    ]

    def _probe_port(port: int, label: str) -> list:
        """Probe a single port for /v1/models endpoint."""
        try:
            url = f"http://localhost:{port}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                results = []
                for m in data.get("data", []):
                    model_id = m.get("id", "")
                    if model_id:
                        results.append({
                            "name": model_id,
                            "port": port,
                            "server_label": label,
                            "source": "local",
                        })
                return results
        except Exception:
            return []

    # Also check OLLAMA_HOST for non-standard Ollama port (but use /v1/models, not /api/tags)
    ollama_host = os.environ.get("OLLAMA_HOST", "")
    extra_probes = []
    if ollama_host:
        # Parse host:port from OLLAMA_HOST (e.g. "http://192.168.1.5:11434" or "localhost:9999")
        host_str = ollama_host.replace("http://", "").replace("https://", "").rstrip("/")
        if ":" in host_str:
            try:
                port = int(host_str.split(":")[-1])
                if port != 11434:  # Skip default — _detect_ollama_models handles it
                    extra_probes.append((port, f"Ollama ({host_str})"))
            except ValueError:
                pass

    all_servers = LOCAL_SERVERS + extra_probes
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(all_servers)) as executor:
        futures = {
            executor.submit(_probe_port, port, label): (port, label)
            for port, label in all_servers
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                models = future.result(timeout=3)
                results.extend(models)
            except Exception:
                pass

    return results


def _detect_api_models(config: dict) -> list:
    """Extract configured API models from opencode.json providers.

    Returns URL and key alongside model info so sidecar setup can write them
    to opencode.json without re-scanning the config.
    """
    models = []
    providers = config.get("provider", {})
    for provider_id, provider_cfg in providers.items():
        base_url = (provider_cfg.get("options") or {}).get("baseURL", "")
        api_key = (provider_cfg.get("options") or {}).get("apiKey", "")
        if not base_url:
            continue
        # Skip Zen proxy and Ollama (handled separately)
        if "opencode.ai" in base_url or "localhost:11434" in base_url:
            continue
        provider_name = provider_cfg.get("name", provider_id)
        for model_id, model_cfg in (provider_cfg.get("models") or {}).items():
            model_name = model_cfg.get("name", model_id) if isinstance(model_cfg, dict) else model_id
            models.append({
                "name": model_id,
                "display": f"{model_name} ({provider_name})",
                "source": "api",
                "has_key": bool(api_key),
                "base_url": base_url,
                "api_key": api_key,
            })
    return models


def _prompt_custom_endpoint(config: dict, config_path: Path):
    """Interactive prompt for custom sidecar API endpoint.

    Collects URL, API key, and model name, then writes them to
    opencode.json MCP environment variables.
    """
    print(f"\n{BOLD}Custom scoring endpoint{RESET}")
    print(f"  Any OpenAI-compatible /chat/completions endpoint works.")
    print(f"  Examples: Groq, Together, OpenRouter, Ollama, LM Studio\n")

    try:
        url = input(f"  URL (e.g. https://api.groq.com/openai/v1): ").strip()
        if not url:
            print(f"  {YELLOW}Cancelled.{RESET}")
            return False

        import getpass
        api_key = getpass.getpass("  API Key (optional for local, Enter to skip): ").strip()

        model = input(f"  Model name (e.g. llama-3.3-70b-versatile): ").strip()
        if not model:
            print(f"  {YELLOW}Model name required.{RESET}")
            return False
    except (EOFError, KeyboardInterrupt):
        print(f"\n  {YELLOW}Cancelled.{RESET}")
        return False

    # Write env vars to opencode.json MCP environment
    if "mcp" not in config:
        config["mcp"] = {}
    if "roampal-core" not in config["mcp"]:
        print(f"  {RED}roampal-core MCP not configured. Run {BLUE}roampal init --opencode{RESET} first.{RESET}")
        return False
    if "environment" not in config["mcp"]["roampal-core"]:
        config["mcp"]["roampal-core"]["environment"] = {}

    env = config["mcp"]["roampal-core"]["environment"]
    env["ROAMPAL_SIDECAR_URL"] = url
    if api_key:
        env["ROAMPAL_SIDECAR_KEY"] = api_key
    env["ROAMPAL_SIDECAR_MODEL"] = model
    config_path.write_text(json.dumps(config, indent=2))

    print(f"\n  {GREEN}Custom sidecar configured!{RESET}")
    print(f"    URL:   {url}")
    print(f"    Model: {model}")
    if api_key:
        print(f"    Key:   {'*' * min(len(api_key), 8)}...")
    return True


def _write_sidecar_model(config: dict, config_path: Path, chosen: dict) -> bool:
    """Write sidecar URL/KEY/MODEL to opencode.json for the chosen model.

    Determines the OpenAI-compatible endpoint URL from the model's source:
    - Ollama: http://localhost:11434/v1
    - Local server: http://localhost:{port}/v1
    - API: base_url from provider config

    Removes ROAMPAL_SIDECAR_FALLBACK (legacy) — URL/MODEL is all the plugin needs.
    """
    if "mcp" not in config:
        config["mcp"] = {}
    if "roampal-core" not in config["mcp"]:
        return False
    if "environment" not in config["mcp"]["roampal-core"]:
        config["mcp"]["roampal-core"]["environment"] = {}

    env = config["mcp"]["roampal-core"]["environment"]
    source = chosen.get("source", "")
    model_name = chosen.get("name", "")

    if source == "ollama":
        env["ROAMPAL_SIDECAR_URL"] = "http://localhost:11434/v1"
        env["ROAMPAL_SIDECAR_MODEL"] = model_name
        env.pop("ROAMPAL_SIDECAR_KEY", None)
    elif source == "local":
        port = chosen.get("port", 8080)
        env["ROAMPAL_SIDECAR_URL"] = f"http://localhost:{port}/v1"
        env["ROAMPAL_SIDECAR_MODEL"] = model_name
        env.pop("ROAMPAL_SIDECAR_KEY", None)
    elif source == "api":
        env["ROAMPAL_SIDECAR_URL"] = chosen.get("base_url", "")
        env["ROAMPAL_SIDECAR_MODEL"] = model_name
        api_key = chosen.get("api_key", "")
        if api_key:
            env["ROAMPAL_SIDECAR_KEY"] = api_key
        else:
            env.pop("ROAMPAL_SIDECAR_KEY", None)
    else:
        return False

    # Clean up legacy flag — URL/MODEL is all the plugin needs
    env.pop("ROAMPAL_SIDECAR_FALLBACK", None)

    config_path.write_text(json.dumps(config, indent=2))
    return True


def _prompt_sidecar_setup(force: bool = False):
    """Interactive sidecar model selection during init.

    Shows available scoring models (Zen, local, API, custom) and lets
    the user choose. Called during init when OpenCode is detected.

    Args:
        force: If True, always show prompt even if already configured.
    """
    config_path = _get_opencode_config_path()
    if not config_path.exists():
        return  # OpenCode not configured yet

    config = json.loads(config_path.read_text())
    mcp_env = config.get("mcp", {}).get("roampal-core", {}).get("environment", {})

    # Idempotency: skip if already configured (unless force)
    has_model = bool(mcp_env.get("ROAMPAL_SIDECAR_URL"))
    if not force and has_model:
        model = mcp_env.get("ROAMPAL_SIDECAR_MODEL", "unknown")
        url = mcp_env.get("ROAMPAL_SIDECAR_URL", "")
        print(f"  {GREEN}[OK] Sidecar scorer: {model} ({url}){RESET}")
        return

    # Non-interactive: skip prompts, use Zen default
    if not _is_interactive():
        print(f"  {GREEN}[OK] Sidecar scorer: free community models (default){RESET}")
        return

    # Detect available models
    ollama_models = _detect_ollama_models()
    local_models = _detect_local_servers()
    api_models = _detect_api_models(config)

    # Display the sidecar explanation box (ASCII-safe for Windows cp1252)
    print(f"""
{BLUE}+-----------------------------------------------------------+
|  Memory Scoring                                           |
|                                                           |
|  Roampal learns what works by scoring your exchanges      |
|  in the background. This uses a small AI model -- it      |
|  doesn't need to be smart or expensive. A cheap local     |
|  model works great for this.                              |
+-----------------------------------------------------------+{RESET}

{BOLD}Available scoring models:{RESET}
""")

    # Build numbered menu
    all_options = []  # Each entry: {display, ...model_data}

    # Option 1: Zen (always)
    print(f"  {BOLD}[1]{RESET} Free community models (default, no setup needed)")
    print(f"      Best-effort, may be unreliable")
    all_options.append({"choice": "zen"})

    # Local models: Ollama
    if ollama_models or local_models:
        print(f"\n  {GREEN}Local models detected:{RESET}")

    if ollama_models:
        for m in ollama_models:
            idx = len(all_options) + 1
            size_str = f", {m['size_gb']} GB" if m.get('size_gb') else ""
            print(f"  {BOLD}[{idx}]{RESET} {m['name']} (Ollama{size_str})")
            all_options.append({**m, "choice": "local"})

    # Local models: other servers
    if local_models:
        for m in local_models:
            idx = len(all_options) + 1
            print(f"  {BOLD}[{idx}]{RESET} {m['name']} ({m['server_label']}, port {m['port']})")
            all_options.append({**m, "choice": "local"})

    # API models from config
    if api_models:
        print(f"\n  {GREEN}From your config:{RESET}")
        for m in api_models:
            idx = len(all_options) + 1
            key_str = ", API key ok" if m.get('has_key') else ""
            print(f"  {BOLD}[{idx}]{RESET} {m['display']}{key_str}")
            all_options.append({**m, "choice": "api"})

    # Custom option
    print(f"\n  {BOLD}[C]{RESET} Use a custom API endpoint (bring your own key)")

    # Prompt
    try:
        raw = input(f"\nChoose [1-{len(all_options)}/C] or press Enter for default: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(f"\n  Using free community models (default).")
        return

    if not raw or raw == "1":
        # Zen default — nothing to configure
        print(f"\n  {GREEN}Using free community models.{RESET} No configuration needed.")
        return

    if raw == "c":
        _prompt_custom_endpoint(config, config_path)
        return

    # Numeric choice
    try:
        idx = int(raw) - 1
        if idx < 0 or idx >= len(all_options):
            print(f"  {RED}Invalid choice. Using default (free community models).{RESET}")
            return
    except ValueError:
        print(f"  {RED}Invalid choice. Using default (free community models).{RESET}")
        return

    chosen = all_options[idx]

    if chosen["choice"] == "zen":
        print(f"\n  {GREEN}Using free community models.{RESET} No configuration needed.")
        return

    # Write the chosen model's URL/KEY/MODEL to opencode.json
    if not _write_sidecar_model(config, config_path, chosen):
        print(f"  {RED}Failed to write sidecar config. Run {BLUE}roampal init --opencode{RESET} first.{RESET}")
        return

    display_name = chosen.get("display", chosen.get("name", "unknown"))
    url = config.get("mcp", {}).get("roampal-core", {}).get("environment", {}).get("ROAMPAL_SIDECAR_URL", "")
    print(f"\n  {GREEN}Sidecar configured!{RESET}")
    print(f"    Model: {display_name}")
    print(f"    URL:   {url}")


def cmd_sidecar(args):
    """Configure sidecar scoring model."""
    subcommand = args.sidecar_command or "setup"

    if subcommand == "status":
        _cmd_sidecar_status(args)
    elif subcommand == "setup":
        _cmd_sidecar_setup(args)
    elif subcommand == "disable":
        _cmd_sidecar_disable(args)
    else:
        print(f"{RED}Unknown sidecar command: {subcommand}{RESET}")


def _cmd_sidecar_status(args):
    """Show current sidecar configuration."""
    config_path = _get_opencode_config_path()
    if not config_path.exists():
        print(f"{YELLOW}No opencode.json found at {config_path}{RESET}")
        return

    config = json.loads(config_path.read_text())
    mcp_env = config.get("mcp", {}).get("roampal-core", {}).get("environment", {})

    custom_url = mcp_env.get("ROAMPAL_SIDECAR_URL", "")
    custom_model = mcp_env.get("ROAMPAL_SIDECAR_MODEL", "")

    print(f"{BOLD}Sidecar scoring status:{RESET}")
    if custom_url:
        print(f"  {GREEN}Scoring model: {custom_model or 'default model'}{RESET}")
        print(f"  URL: {custom_url}")
        if mcp_env.get("ROAMPAL_SIDECAR_KEY"):
            print(f"  Key: {'*' * 8}...")
    else:
        print(f"  {YELLOW}No sidecar configured - using Zen (free, best-effort) only{RESET}")
        print(f"  Run {BLUE}roampal sidecar setup{RESET} to configure a reliable scorer")


def _cmd_sidecar_setup(args):
    """Auto-detect and configure a sidecar scorer.

    Expanded in v0.3.7 to match init-time setup: detects Ollama, local
    inference servers (LM Studio, LocalAI, etc.), API models from config,
    and offers a custom endpoint option.
    """
    config_path = _get_opencode_config_path()
    if not config_path.exists():
        print(f"{RED}No opencode.json found at {config_path}{RESET}")
        print(f"Run {BLUE}roampal init --opencode{RESET} first")
        return

    config = json.loads(config_path.read_text())

    # Detect available models
    print(f"{BOLD}Detecting available models...{RESET}")
    ollama_models = _detect_ollama_models()
    local_models = _detect_local_servers()
    api_models = _detect_api_models(config)

    # Build numbered menu: Zen → local → API → custom
    all_options = []

    print(f"\n{BOLD}Available scoring models:{RESET}\n")

    # Option 1: Zen (always)
    print(f"  {BOLD}[1]{RESET} Free community models (default, no setup needed)")
    print(f"      Best-effort, may be unreliable")
    all_options.append({"choice": "zen"})

    # Local models: Ollama
    if ollama_models or local_models:
        print(f"\n  {GREEN}Local models detected:{RESET}")

    if ollama_models:
        for m in ollama_models:
            idx = len(all_options) + 1
            size_str = f", {m['size_gb']} GB" if m.get('size_gb') else ""
            print(f"  {BOLD}[{idx}]{RESET} {m['name']} (Ollama{size_str})")
            all_options.append({**m, "choice": "local"})

    # Local models: other servers
    if local_models:
        for m in local_models:
            idx = len(all_options) + 1
            print(f"  {BOLD}[{idx}]{RESET} {m['name']} ({m['server_label']}, port {m['port']})")
            all_options.append({**m, "choice": "local"})

    # API models from config
    if api_models:
        print(f"\n  {GREEN}From your config:{RESET}")
        for m in api_models:
            idx = len(all_options) + 1
            key_str = ", API key ok" if m.get('has_key') else ""
            print(f"  {BOLD}[{idx}]{RESET} {m['display']}{key_str}")
            all_options.append({**m, "choice": "api"})

    # Custom option
    print(f"\n  {BOLD}[C]{RESET} Use a custom API endpoint (bring your own key)")

    # Auto mode or non-interactive — pick first non-Zen model
    if getattr(args, 'auto', False) or not _is_interactive():
        non_zen = [o for o in all_options if o.get("choice") != "zen"]
        if non_zen:
            choice = non_zen[0]
            display_name = choice.get("display", choice.get("name", "unknown"))
            print(f"\n{GREEN}Auto-selected: {display_name}{RESET}")
        else:
            print(f"\n{YELLOW}No models found. Using free community models.{RESET}")
            return
    else:
        # Interactive — ask user
        try:
            raw = input(f"\nChoose [1-{len(all_options)}/C] or press Enter for default: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Cancelled. Using free community models.{RESET}")
            return

        if not raw or raw == "1":
            # Clear existing sidecar config if resetting to Zen
            mcp_env = config.get("mcp", {}).get("roampal-core", {}).get("environment", {})
            changed = False
            for key in ["ROAMPAL_SIDECAR_FALLBACK", "ROAMPAL_SIDECAR_URL", "ROAMPAL_SIDECAR_KEY", "ROAMPAL_SIDECAR_MODEL"]:
                if key in mcp_env:
                    del mcp_env[key]
                    changed = True
            if changed:
                config_path.write_text(json.dumps(config, indent=2))
            print(f"\n{GREEN}Using free community models.{RESET} No configuration needed.")
            print(f"\n{YELLOW}Restart OpenCode to activate.{RESET}")
            return

        if raw == "c":
            if _prompt_custom_endpoint(config, config_path):
                print(f"\n{YELLOW}Restart OpenCode to activate.{RESET}")
            return

        try:
            idx = int(raw) - 1
            if idx < 0 or idx >= len(all_options):
                print(f"{RED}Invalid choice.{RESET}")
                return
            choice = all_options[idx]
        except ValueError:
            print(f"{RED}Invalid choice.{RESET}")
            return

        if choice["choice"] == "zen":
            print(f"\n{GREEN}Using free community models.{RESET} No configuration needed.")
            return

    # Write the chosen model's URL/KEY/MODEL to opencode.json
    if not _write_sidecar_model(config, config_path, choice):
        print(f"{RED}roampal-core MCP not configured. Run {BLUE}roampal init --opencode{RESET} first.{RESET}")
        return

    display_name = choice.get("display", choice.get("name", "unknown"))
    url = config.get("mcp", {}).get("roampal-core", {}).get("environment", {}).get("ROAMPAL_SIDECAR_URL", "")
    print(f"\n{GREEN}Sidecar configured!{RESET}")
    print(f"  Model: {display_name}")
    print(f"  URL:   {url}")
    print(f"  Config: {config_path}")
    print(f"\n{YELLOW}Restart OpenCode to activate.{RESET}")


def _cmd_sidecar_disable(args):
    """Remove sidecar fallback configuration."""
    config_path = _get_opencode_config_path()
    if not config_path.exists():
        print(f"{YELLOW}No opencode.json found.{RESET}")
        return

    config = json.loads(config_path.read_text())
    mcp_env = config.get("mcp", {}).get("roampal-core", {}).get("environment", {})

    sidecar_keys = ["ROAMPAL_SIDECAR_FALLBACK", "ROAMPAL_SIDECAR_URL", "ROAMPAL_SIDECAR_KEY", "ROAMPAL_SIDECAR_MODEL"]
    found = any(k in mcp_env for k in sidecar_keys)
    if not found:
        print(f"{YELLOW}No sidecar configuration found.{RESET}")
        return

    for key in sidecar_keys:
        mcp_env.pop(key, None)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"{GREEN}Sidecar configuration removed. Reverted to free community models.{RESET}")
    print(f"{YELLOW}Restart OpenCode to take effect.{RESET}")


def main():
    """Main CLI entry point."""
    from roampal import __version__

    parser = argparse.ArgumentParser(
        prog="roampal",
        description="roampal - Persistent memory for AI coding assistants",
        allow_abbrev=False,
        epilog="""commands:
  Setup:
    init              Initialize for Claude Code / Cursor / OpenCode
    doctor            Diagnose installation and configuration

  Server:
    start             Start the memory server
    stop              Stop the memory server
    status            Check server status (--json for scripting)
    stats             Show memory statistics (--json for scripting)

  Memory:
    ingest <file>     Add documents to books collection
    books             List all ingested books
    remove <title>    Remove a book by title
    summarize         Summarize long memories (retroactive cleanup)

  Scoring (OpenCode):
    sidecar status    Check scoring model configuration
    sidecar setup     Configure scoring model
    sidecar disable   Remove scoring model configuration

  Advanced:
    score             Score the last exchange (manual/testing)
    context           Output recent exchange context

examples:
  roampal init --claude-code    Set up for Claude Code
  roampal init --no-input       Non-interactive setup (CI/scripts)
  roampal status --json         Machine-readable status
  roampal stats --json          Machine-readable statistics""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"roampal {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize Roampal for Claude Code / Cursor / OpenCode")
    init_parser.add_argument("--dev", action="store_true", help="Initialize for DEV mode (separate data directory)")
    init_parser.add_argument("--claude-code", action="store_true", help="Configure Claude Code only (skip auto-detect)")
    init_parser.add_argument("--cursor", action="store_true", help="Configure Cursor only (skip auto-detect)")
    init_parser.add_argument("--opencode", action="store_true", help="Configure OpenCode only (skip auto-detect)")
    init_parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing config")
    init_parser.add_argument("--no-input", action="store_true", help="Non-interactive mode (skip all prompts, use defaults)")

    # start command
    start_parser = subparsers.add_parser("start", help="Start the memory server")
    start_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    start_parser.add_argument("--port", type=int, default=27182, help="Server port")
    start_parser.add_argument("--dev", action="store_true", help="Dev mode - use separate data directory")

    # stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the memory server")
    stop_parser.add_argument("--port", type=int, default=None, help="Server port (default: 27182 prod, 27183 dev)")
    stop_parser.add_argument("--dev", action="store_true", help="Stop dev server")

    # status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    status_parser.add_argument("--port", type=int, default=None, help="Server port (default: 27182 prod, 27183 dev)")
    status_parser.add_argument("--dev", action="store_true", help="Check dev server status")
    status_parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON (for scripting)")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    stats_parser.add_argument("--port", type=int, default=None, help="Server port (default: 27182 prod, 27183 dev)")
    stats_parser.add_argument("--dev", action="store_true", help="Show dev server stats")
    stats_parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON (for scripting)")

    # ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document into the books collection")
    ingest_parser.add_argument("file", help="File to ingest (.txt, .md, .pdf)")
    ingest_parser.add_argument("--title", help="Document title (defaults to filename)")
    ingest_parser.add_argument("--chunk-size", type=int, default=1000, help="Characters per chunk (default: 1000)")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks (default: 200)")
    ingest_parser.add_argument("--dev", action="store_true", help="Dev mode - use separate data directory")

    # remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a book from the books collection")
    remove_parser.add_argument("title", help="Title of the book to remove")
    remove_parser.add_argument("--dev", action="store_true", help="Use dev server")

    # books command
    books_parser = subparsers.add_parser("books", help="List all books in memory")

    # score command (v0.3.6)
    score_parser = subparsers.add_parser("score", help="Score the last exchange using sidecar (Haiku)")
    score_parser.add_argument("--from-hook", action="store_true", help="Read Stop hook event data from stdin")
    score_parser.add_argument("--session-id", type=str, default=None, help="Claude Code session ID (transcript filename)")
    score_parser.add_argument("--transcript", type=str, default=None, help="Direct path to transcript file")
    score_parser.add_argument("--last", action="store_true", help="Score the most recent exchange manually")
    score_parser.add_argument("--dev", action="store_true", help="Use dev server")
    score_parser.add_argument("--port", type=int, default=None, help="Server port")

    # summarize command (v0.3.6)
    summarize_parser = subparsers.add_parser("summarize", help="Summarize existing long memories")
    summarize_parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    summarize_parser.add_argument("--max-chars", type=int, default=400, help="Summarize memories over this length (default: 400)")
    summarize_parser.add_argument("--collection", choices=["working", "history"], help="Target specific collection")
    summarize_parser.add_argument("--limit", type=int, default=None, help="Max memories to summarize (for batching)")
    summarize_parser.add_argument("--dev", action="store_true", help="Use dev server")
    summarize_parser.add_argument("--port", type=int, default=None, help="Server port")

    # context command (v0.3.6)
    context_parser = subparsers.add_parser("context", help="Output recent exchange context")
    context_parser.add_argument("--recent-exchanges", action="store_true", help="Output last 4 exchange summaries")
    context_parser.add_argument("--dev", action="store_true", help="Use dev server")
    context_parser.add_argument("--port", type=int, default=None, help="Server port")

    # sidecar command (v0.3.7)
    sidecar_parser = subparsers.add_parser("sidecar", help="Configure sidecar scoring model")
    sidecar_sub = sidecar_parser.add_subparsers(dest="sidecar_command")
    sidecar_setup = sidecar_sub.add_parser("setup", help="Auto-detect and configure a scorer")
    sidecar_setup.add_argument("--auto", action="store_true", help="Auto-select best model (no prompts)")
    sidecar_sub.add_parser("status", help="Show current sidecar configuration")
    sidecar_sub.add_parser("disable", help="Remove sidecar fallback")

    # doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Diagnose installation and configuration")
    doctor_parser.add_argument("--dev", action="store_true", help="Check dev mode configuration")

    # help command (alias for --help)
    subparsers.add_parser("help", help="Show this help message")

    args = parser.parse_args()

    # Set global non-interactive flag from --no-input
    global _NO_INPUT
    if getattr(args, 'no_input', False):
        _NO_INPUT = True

    # Dispatch commands — functions return exit codes (0=success, 1=failure)
    # Functions that don't return a value are treated as success (0)
    exit_code = 0
    if args.command == "init":
        exit_code = cmd_init(args) or 0
    elif args.command == "start":
        cmd_start(args)
    elif args.command == "stop":
        exit_code = cmd_stop(args) or 0
    elif args.command == "status":
        exit_code = cmd_status(args) or 0
    elif args.command == "stats":
        exit_code = cmd_stats(args) or 0
    elif args.command == "ingest":
        exit_code = cmd_ingest(args) or 0
    elif args.command == "remove":
        exit_code = cmd_remove(args) or 0
    elif args.command == "books":
        cmd_books(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    elif args.command == "context":
        cmd_context(args)
    elif args.command == "sidecar":
        cmd_sidecar(args)
    elif args.command == "doctor":
        exit_code = cmd_doctor(args) or 0
    elif args.command == "help":
        parser.print_help()
    else:
        parser.print_help()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

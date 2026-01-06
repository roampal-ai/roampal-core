"""
Roampal CLI - One command install for AI coding tools

Usage:
    pip install roampal
    roampal init          # Configure Claude Code / Cursor
    roampal doctor        # Diagnose installation issues
    roampal start         # Start the memory server
    roampal status        # Check server status
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Port configuration - DEV and PROD use different ports to avoid collision
PROD_PORT = 27182
DEV_PORT = 27183


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
    """Get port based on DEV/PROD mode. Uses is_dev_mode() for consistency."""
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


def print_banner():
    """Print Roampal banner."""
    print(f"""
{BLUE}{BOLD}+---------------------------------------------------+
|                   ROAMPAL                         |
|     Persistent Memory for AI Coding Tools         |
+---------------------------------------------------+{RESET}
""")


def check_for_updates() -> tuple:
    """Check if a newer version is available on PyPI.

    Returns:
        tuple: (update_available: bool, current_version: str, latest_version: str)
    """
    try:
        import urllib.request
        from roampal import __version__

        url = "https://pypi.org/pypi/roampal/json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
            latest = data.get("info", {}).get("version", __version__)

            # Compare versions (simple tuple comparison works for semver)
            current_parts = [int(x) for x in __version__.split(".")]
            latest_parts = [int(x) for x in latest.split(".")]
            update_available = latest_parts > current_parts

            return (update_available, __version__, latest)
    except Exception:
        # Fail silently - don't block CLI on network issues
        try:
            from roampal import __version__
            return (False, __version__, __version__)
        except Exception:
            return (False, "unknown", "unknown")


def print_update_notice():
    """Print update notice if newer version available. Non-blocking."""
    update_available, current, latest = check_for_updates()
    if update_available:
        print(f"{YELLOW}[!] Update available: {latest} (you have {current}){RESET}")
        print(f"    Run: pip install --upgrade roampal\n")


def cmd_init(args):
    """Initialize Roampal for the current environment."""
    print_banner()
    print_update_notice()
    print(f"{BOLD}Initializing Roampal...{RESET}\n")

    # Detect environment
    home = Path.home()
    claude_code_dir = home / ".claude"
    cursor_dir = home / ".cursor"

    # Check for explicit flags (--claude-code, --cursor)
    explicit_claude = getattr(args, 'claude_code', False)
    explicit_cursor = getattr(args, 'cursor', False)

    if explicit_claude or explicit_cursor:
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
    else:
        # Auto-detect installed tools
        detected = []
        if claude_code_dir.exists():
            detected.append("claude-code")
        if cursor_dir.exists():
            detected.append("cursor")

    if not detected:
        print(f"{YELLOW}No AI coding tools detected.{RESET}")
        print("Roampal works with:")
        print("  - Claude Code (https://claude.com/claude-code)")
        print("  - Cursor (https://cursor.sh)")
        print("\nInstall one of these tools first, or use --claude-code / --cursor to force setup.")
        return

    print(f"{GREEN}Configuring: {', '.join(detected)}{RESET}\n")

    # Configure each detected tool
    is_dev = is_dev_mode(args)
    force = getattr(args, 'force', False)
    for tool in detected:
        if tool == "claude-code":
            configure_claude_code(claude_code_dir, is_dev=is_dev, force=force)
        elif tool == "cursor":
            configure_cursor(cursor_dir, is_dev=is_dev, force=force)

    # Create data directory
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"{GREEN}Created data directory: {data_dir}{RESET}")

    print(f"""
{GREEN}{BOLD}Roampal initialized successfully!{RESET}

{BOLD}Next step:{RESET}
  {BLUE}Restart Claude Code{RESET} and start chatting!
  The MCP server auto-starts - no manual server needed.

{BOLD}How it works:{RESET}
  - Hooks inject relevant memories into your context automatically
  - The AI learns what works and what doesn't via outcome scoring
  - You see your original message; the AI sees your message + context + scoring prompt

{BOLD}Optional commands:{RESET}
  - {BLUE}roampal ingest myfile.pdf{RESET} - Add documents to memory
  - {BLUE}roampal stats{RESET} - Show memory statistics
  - {BLUE}roampal status{RESET} - Check server status
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
        except:
            pass

    # Handle env section: preserve existing OR add if --dev
    if existing_env:
        settings["env"] = existing_env  # Keep what user had
    elif is_dev:
        settings["env"] = {"ROAMPAL_DEV": "1"}  # Add for --dev

    # Configure hooks - Claude Code expects nested format with type/command
    python_exe = sys.executable.replace("\\", "\\\\")  # Escape backslashes for JSON
    settings["hooks"] = {
        "UserPromptSubmit": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{python_exe} -m roampal.hooks.user_prompt_submit_hook"
                    }
                ]
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"{python_exe} -m roampal.hooks.stop_hook"
                    }
                ]
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
        "mcp__roampal-core__get_context_insights",
        "mcp__roampal-core__record_response",
        "mcp__roampal-core__score_response"
    ]

    for perm in roampal_perms:
        if perm not in settings["permissions"]["allow"]:
            settings["permissions"]["allow"].append(perm)

    settings_path.write_text(json.dumps(settings, indent=2))
    print(f"  {GREEN}Created settings: {settings_path}{RESET}")
    print(f"  {GREEN}  - UserPromptSubmit hook (injects scoring + memories){RESET}")
    print(f"  {GREEN}  - Stop hook (enforces record_response){RESET}")
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
        except:
            pass

    local_mcp_path.write_text(json.dumps(local_mcp_config, indent=2))
    print(f"  {GREEN}Created project MCP config: {local_mcp_path}{RESET}")

    print(f"  {GREEN}Claude Code configured!{RESET}\n")


def configure_cursor(cursor_dir: Path, is_dev: bool = False, force: bool = False):
    """Configure Cursor MCP and hooks.

    Args:
        cursor_dir: Path to ~/.cursor directory
        is_dev: If True, adds ROAMPAL_DEV=1 to env section
        force: If True, overwrite existing config (reserved for future use)
    """
    # Note: force parameter reserved for future idempotency implementation
    print(f"{BOLD}Configuring Cursor...{RESET}")

    # Cursor uses ~/.cursor/mcp.json
    mcp_config_path = cursor_dir / "mcp.json"
    mcp_config = {
        "mcpServers": {
            "roampal-core": {
                "command": sys.executable,
                "args": ["-m", "roampal.mcp.server"],
                "env": {"ROAMPAL_DEV": "1"} if is_dev else {}
            }
        }
    }

    # Merge with existing if present
    if mcp_config_path.exists():
        try:
            existing = json.loads(mcp_config_path.read_text())
            if "mcpServers" in existing:
                existing["mcpServers"]["roampal-core"] = mcp_config["mcpServers"]["roampal-core"]
            else:
                existing["mcpServers"] = mcp_config["mcpServers"]
            mcp_config = existing
        except:
            pass

    mcp_config_path.write_text(json.dumps(mcp_config, indent=2))
    print(f"  {GREEN}Created MCP config: {mcp_config_path}{RESET}")

    # Cursor 1.7+ supports hooks - create ~/.cursor/hooks.json
    hooks_config_path = cursor_dir / "hooks.json"
    python_exe = sys.executable

    # Build hook commands - need to set ROAMPAL_DEV env var if in dev mode
    if is_dev:
        # On Windows, use cmd /c to set env var before running
        if sys.platform == "win32":
            submit_cmd = f'cmd /c "set ROAMPAL_DEV=1 && {python_exe} -m roampal.hooks.user_prompt_submit_hook"'
            stop_cmd = f'cmd /c "set ROAMPAL_DEV=1 && {python_exe} -m roampal.hooks.stop_hook"'
        else:
            submit_cmd = f'ROAMPAL_DEV=1 {python_exe} -m roampal.hooks.user_prompt_submit_hook'
            stop_cmd = f'ROAMPAL_DEV=1 {python_exe} -m roampal.hooks.stop_hook'
    else:
        submit_cmd = f'{python_exe} -m roampal.hooks.user_prompt_submit_hook'
        stop_cmd = f'{python_exe} -m roampal.hooks.stop_hook'

    hooks_config = {
        "version": 1,
        "hooks": {
            "beforeSubmitPrompt": [
                {"command": submit_cmd}
            ],
            "stop": [
                {"command": stop_cmd}
            ]
        }
    }

    # Merge with existing hooks if present
    if hooks_config_path.exists():
        try:
            existing = json.loads(hooks_config_path.read_text())
            if "hooks" in existing:
                existing["hooks"]["beforeSubmitPrompt"] = hooks_config["hooks"]["beforeSubmitPrompt"]
                existing["hooks"]["stop"] = hooks_config["hooks"]["stop"]
            else:
                existing["hooks"] = hooks_config["hooks"]
            if "version" not in existing:
                existing["version"] = 1
            hooks_config = existing
        except:
            pass

    hooks_config_path.write_text(json.dumps(hooks_config, indent=2))
    print(f"  {GREEN}Created hooks config: {hooks_config_path}{RESET}")
    print(f"  {GREEN}  - beforeSubmitPrompt hook (injects scoring + memories){RESET}")
    print(f"  {GREEN}  - stop hook (enforces record_response){RESET}")

    print(f"  {GREEN}Cursor configured!{RESET}\n")


def cmd_start(args):
    """Start the Roampal server."""
    print_banner()

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


def cmd_status(args):
    """Check Roampal server status and MCP configuration."""
    print_banner()
    print_update_notice()

    # =========================================================================
    # MCP Configuration Status
    # =========================================================================
    print(f"{BOLD}MCP Configuration:{RESET}")

    claude_json_path = Path.home() / ".claude.json"
    if claude_json_path.exists():
        try:
            claude_json = json.loads(claude_json_path.read_text(encoding='utf-8'))
            roampal_config = claude_json.get("mcpServers", {}).get("roampal-core", {})
            if roampal_config:
                command = roampal_config.get("command", "N/A")
                args_list = roampal_config.get("args", [])
                env = roampal_config.get("env", {})

                print(f"  Location: {GREEN}{claude_json_path}{RESET} (user scope)")
                print(f"  Command:  {command} {' '.join(args_list)}")
                if env:
                    print(f"  Env:      {env}")
                print(f"  Status:   {GREEN}[OK] Configured{RESET}")
            else:
                print(f"  Status:   {YELLOW}Not configured{RESET}")
                print(f"  Run: roampal init")
        except Exception as e:
            print(f"  {RED}Error reading config: {e}{RESET}")
    else:
        print(f"  Status:   {YELLOW}~/.claude.json not found{RESET}")
        print(f"  Run: roampal init")

    print()  # Empty line separator

    # =========================================================================
    # Server Status
    # =========================================================================
    import httpx

    host = args.host or "127.0.0.1"
    # Use dev port if --dev flag, otherwise prod port (unless explicitly overridden)
    is_dev = is_dev_mode(args)
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = args.port if args.port and args.port != PROD_PORT else default_port

    mode_str = f"{YELLOW}DEV{RESET}" if is_dev else f"{GREEN}PROD{RESET}"
    url = f"http://{host}:{port}/api/health"

    print(f"{BOLD}Server Status:{RESET}")
    try:
        response = httpx.get(url, timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            print(f"  Mode: {mode_str}")
            print(f"  Status: {GREEN}RUNNING{RESET}")
            print(f"  Port: {port}")
            print(f"  Memory initialized: {data.get('memory_initialized', False)}")
            print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
        else:
            print(f"  {RED}Server returned error: {response.status_code}{RESET}")
    except httpx.ConnectError:
        print(f"  Mode: {mode_str}")
        print(f"  Status: {YELLOW}NOT RUNNING{RESET}")
        start_cmd = "roampal start --dev" if is_dev else "roampal start"
        print(f"\n  Start with: {start_cmd}")
    except Exception as e:
        print(f"  {RED}Error checking status: {e}{RESET}")


def cmd_stats(args):
    """Show memory statistics."""
    print_banner()
    print_update_notice()

    import httpx

    host = args.host or "127.0.0.1"
    # Use dev port if --dev flag
    is_dev = is_dev_mode(args)
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = args.port if args.port and args.port != PROD_PORT else default_port

    mode_str = f"{YELLOW}DEV{RESET}" if is_dev else f"{GREEN}PROD{RESET}"
    url = f"http://{host}:{port}/api/stats"

    try:
        response = httpx.get(url, timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            print(f"{BOLD}Memory Statistics ({mode_str}):{RESET}\n")
            print(f"Data path: {data.get('data_path', 'N/A')}")
            print(f"Port: {port}")
            print(f"\nCollections:")
            for name, info in data.get("collections", {}).items():
                count = info.get("count", 0)
                print(f"  {name}: {count} items")
        else:
            print(f"{RED}Error getting stats: {response.status_code}{RESET}")
    except httpx.ConnectError:
        start_cmd = "roampal start --dev" if is_dev else "roampal start"
        print(f"{YELLOW}Server not running. Start with: {start_cmd}{RESET}")
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")


def cmd_ingest(args):
    """Ingest a document into the books collection."""
    import asyncio
    import httpx

    print_banner()

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

    print_banner()
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

    print_banner()
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
            except:
                pass
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
    else:
        print(f"{RED}{checks_failed} checks failed{RESET}, {checks_passed} passed, {checks_warned} warnings")
        print(f"\nRun {BLUE}roampal init{RESET} to fix configuration issues.")


def cmd_books(args):
    """List all books in the books collection."""
    import asyncio
    import httpx

    print_banner()
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Roampal - Persistent Memory for AI Coding Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize Roampal for Claude Code / Cursor")
    init_parser.add_argument("--dev", action="store_true", help="Initialize for DEV mode (separate data directory)")
    init_parser.add_argument("--claude-code", action="store_true", help="Configure Claude Code only (skip auto-detect)")
    init_parser.add_argument("--cursor", action="store_true", help="Configure Cursor only (skip auto-detect)")
    init_parser.add_argument("--force", "-f", action="store_true", help="Force overwrite existing config")

    # start command
    start_parser = subparsers.add_parser("start", help="Start the memory server")
    start_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    start_parser.add_argument("--port", type=int, default=27182, help="Server port")
    start_parser.add_argument("--dev", action="store_true", help="Dev mode - use separate data directory")

    # status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    status_parser.add_argument("--port", type=int, default=None, help="Server port (default: 27182 prod, 27183 dev)")
    status_parser.add_argument("--dev", action="store_true", help="Check dev server status")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.add_argument("--host", default="127.0.0.1", help="Server host")
    stats_parser.add_argument("--port", type=int, default=None, help="Server port (default: 27182 prod, 27183 dev)")
    stats_parser.add_argument("--dev", action="store_true", help="Show dev server stats")

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

    # doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Diagnose installation and configuration")
    doctor_parser.add_argument("--dev", action="store_true", help="Check dev mode configuration")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "start":
        cmd_start(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "books":
        cmd_books(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

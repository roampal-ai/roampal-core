"""Server control group (v0.6.0 Task 6): start/stop/status/stats.

Moved verbatim from the pre-refactor roampal/cli.py as part of the Task 6
split (cmd_start, _stop_server_on_port, cmd_stop, cmd_status, cmd_stats).
Cross-impl deps (is_dev_mode, get_data_dir, get_port?, PROD_PORT/DEV_PORT)
import module-level from the impl — safe: the package __init__ loads the
impl before group modules, so no cycle. `print_update_notice` lives in
update_check (Task 5). `_debug context` helpers used by score/summarize stay
in the impl until their move.
"""

import json
import os  # cmd_start lock/PID scrubbing uses os.link/kills
import subprocess  # _stop_server_on_port's netstat/taskkill calls
import sys
import time
import urllib.request
from pathlib import Path

# `import httpx` stays function-local (as it was in the monolith).
import httpx  # noqa: F401  — cmd_status/cmd_stats bind it at module import too

from roampal.cli._common import (
    BLUE,
    BOLD,
    GREEN,
    RED,
    RESET,
    YELLOW,
    is_dev_mode,
    get_data_dir,
    profile_headers,
    PROD_PORT,
    DEV_PORT,
    logger,
)
from roampal.cli.update_check import print_update_notice


def _impl():
    import importlib

    return importlib.import_module("roampal.cli._monolith_impl")


def cmd_start(args):
    """Start the Roampal server."""

    # Determine port based on mode (DEV=27183, PROD=27182)
    # User can override with --port
    is_dev = is_dev_mode(args)
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = (
        args.port if args.port != PROD_PORT else default_port
    )  # Use default unless explicitly overridden

    # v0.5.1: Named profile support. --profile overrides ROAMPAL_PROFILE env.
    profile_name = getattr(args, "profile", None)
    if profile_name:
        os.environ["ROAMPAL_PROFILE"] = profile_name

    # v0.6.0 review fix 5: a bare start clears any previous launch pin
    # (the user chose an unpinned server). Writing the pin for --profile
    # happens AFTER profile validation below — an aborted start must not
    # leave a pin pointing at an unregistered profile.
    if not profile_name:
        from roampal.profile_manager import clear_server_pin

        clear_server_pin(port)

    # Handle dev mode - uses Roampal_DEV folder
    if is_dev:
        os.environ["ROAMPAL_DEV"] = "1"
        data_path = get_data_dir(dev=True)
        print(f"{YELLOW}DEV MODE{RESET} - Isolated from production")
        print(f"  Data path: {data_path}")
        print(f"  Port: {port} (PROD uses {PROD_PORT})\n")
    else:
        if profile_name and profile_name != "default":
            # v0.5.1: resolve path via profile registry for display
            from roampal.profile_manager import ProfileRegistry, ProfileNotFoundError
            try:
                data_path = ProfileRegistry().resolve(profile_name)
            except ProfileNotFoundError:
                print(f"{RED}Error:{RESET} profile {profile_name!r} is not registered.")
                print(f"Create it first: {BLUE}roampal profile create {profile_name}{RESET}")
                return 1
            print(f"{GREEN}PROD MODE{RESET} - profile: {profile_name}")
            print(f"  Data path: {data_path}")
            print(f"  Port: {port}\n")
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

    # Round 2 Item 6 / Task 17: the server ignores ROAMPAL_PROFILE env
    # (profile routing contract — headerless requests resolve the pin or
    # persisted `use` -> default, never a leaked env). The flag pins
    # explicitly.
    # v0.6.0 review fix 5: persist the launch pin per port so a respawn
    # (idle retirement -> auto-restart) re-passes --profile instead of
    # silently dropping it. Written here — after profile validation — so
    # an aborted start never leaves a pin behind.
    if profile_name:
        from roampal.profile_manager import write_server_pin

        write_server_pin(port, profile_name)

    from roampal.server.main import start_server

    # v0.6.0 review fix 9: a foreground server the user launched explicitly
    # is exempt from Task 22's idle self-retirement — it stays up until
    # Ctrl+C / roampal stop instead of exiting after 30 idle minutes.
    start_server(host=host, port=port, profile=profile_name, idle_retire=False)


def _stop_server_on_port(port: int, *, verbose: bool = True) -> bool:
    """Kill the server listening on `port`. Returns True if something was killed.

    v0.5.1: extracted from cmd_stop so 'profile switch' can reuse the logic.
    """
    killed = False
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split("\n"):
                if f"127.0.0.1:{port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    if pid and pid.isdigit():
                        subprocess.run(
                            ["taskkill", "/pid", pid, "/f"],
                            capture_output=True,
                            timeout=5,
                        )
                        if verbose:
                            print(f"  {GREEN}Killed server process (PID {pid}){RESET}")
                        killed = True
                    break
        except Exception as e:
            if verbose:
                print(f"  {RED}Error finding server process: {e}{RESET}")
    else:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5
            )
            pid = result.stdout.strip().split("\n")[0] if result.stdout.strip() else ""
            if pid and pid.isdigit():
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                if verbose:
                    print(f"  {GREEN}Killed server process (PID {pid}){RESET}")
                killed = True
        except Exception as e:
            if verbose:
                print(f"  {RED}Error finding server process: {e}{RESET}")
    return killed


def cmd_stop(args):
    """Stop the Roampal server."""

    is_dev = is_dev_mode(args)
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = args.port if args.port else default_port

    # v0.6.0 review fix 5: an explicit stop ends the pinned server's
    # identity — later auto-started shared servers stay unpinned.
    from roampal.profile_manager import clear_server_pin

    clear_server_pin(port)

    print(f"{BOLD}Stopping Roampal server on port {port}...{RESET}\n")

    killed = _stop_server_on_port(port, verbose=True)

    if not killed:
        print(f"  {YELLOW}No server found on port {port}{RESET}")
        return 1
    else:
        print(f"\n{GREEN}Server stopped.{RESET}")
        return 0


def cmd_status(args):
    """Check Roampal server status and MCP configuration."""
    import httpx

    json_mode = getattr(args, "json_output", False)

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
            claude_json = json.loads(claude_json_path.read_text(encoding="utf-8"))
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
    except (httpx.ConnectError, httpx.ConnectTimeout):
        # Task 27: a connection TIMEOUT is as conclusive as a refusal — an
        # address:port where ping hangs is a stopped server too; reporting
        # "error" (generic) instead of the actionable "stopped" was a bug
        # (nod to the golden that recorded "timed out"). Deliberate:
        # status_json_stopped.golden regenerated on this product change.
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
        print(
            f"  Command:  {mcp_detail['command']} {' '.join(mcp_detail.get('args', []))}"
        )
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
        print(
            f"  {RED}Error: {server_detail.get('error', server_detail.get('status_code', 'unknown'))}{RESET}"
        )
        return 1


def cmd_stats(args):
    """Show memory statistics."""
    import httpx

    json_mode = getattr(args, "json_output", False)

    if not json_mode:
        print_update_notice()

    host = args.host or "127.0.0.1"
    is_dev = is_dev_mode(args)
    default_port = DEV_PORT if is_dev else PROD_PORT
    port = args.port if args.port and args.port != PROD_PORT else default_port
    url = f"http://{host}:{port}/api/stats"

    try:
        response = httpx.get(url, headers=profile_headers(), timeout=5.0)
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
                removed = info.get("removed", 0)
                suffix = f" ({removed} removed, hidden)" if removed else ""
                print(f"  {name}: {count} items{suffix}")
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



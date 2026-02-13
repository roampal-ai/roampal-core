#!/usr/bin/env python3
"""
Roampal UserPromptSubmit Hook

Called by Claude Code / Cursor BEFORE the LLM sees the user's message.
This hook:
1. Checks if previous exchange needs scoring
2. Injects scoring prompt if needed
3. Injects relevant memories as context

Usage (Claude Code - .claude/settings.json):
{
  "hooks": {
    "UserPromptSubmit": ["python", "-m", "roampal.hooks.user_prompt_submit_hook"]
  }
}

Usage (Cursor 1.7+ - .cursor/hooks.json):
{
  "version": 1,
  "hooks": {
    "beforeSubmitPrompt": [{"command": "python -m roampal.hooks.user_prompt_submit_hook"}]
  }
}

Environment variables:
- ROAMPAL_DEV: Set to "1" to use dev port 27183 (default: prod port 27182)
- ROAMPAL_SERVER_URL: Override server URL (takes precedence over ROAMPAL_DEV)

Reads from stdin:
- JSON with user_message

Outputs to stdout:
- Modified user message with injected context (prepended)

Exit codes:
- 0: Success
- 1: Error (but don't break the flow)
"""

import sys
import json
import os
import subprocess
import time
import urllib.request
import urllib.error

# Update check cache to avoid hitting PyPI on every message
_update_check_cache = {"checked": False, "available": False, "current": "", "latest": ""}


def _restart_server(server_url: str, port: int, timeout: float = 15.0) -> bool:
    """
    v0.3.2: Self-healing server restart for hooks.

    If FastAPI is down or unhealthy (503), kill the old process and start fresh.
    Same pattern as _ensure_server_running() in server.py but standalone (no roampal imports).
    """
    # 1. Kill whatever is on the port (may be zombie with corrupted PyTorch state)
    try:
        if sys.platform == "win32":
            # netstat to find PID, taskkill to end it
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if f"127.0.0.1:{port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    if pid.isdigit():
                        subprocess.run(
                            ["taskkill", "/pid", pid, "/f"],
                            capture_output=True, timeout=5
                        )
                        print(f"Roampal: killed stale server process {pid}", file=sys.stderr)
                    break
        else:
            # Unix: lsof + kill
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                pid = result.stdout.strip().split('\n')[0]
                if pid.isdigit():
                    subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                    print(f"Roampal: killed stale server process {pid}", file=sys.stderr)
    except Exception:
        pass  # Best effort — if we can't kill, the new server will fail to bind and we'll exit

    time.sleep(1)  # Let port release

    # 2. Start fresh server
    try:
        env = os.environ.copy()
        dev_mode = env.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")

        cmd = [sys.executable, "-m", "roampal.server.main", "--port", str(port)]
        if dev_mode:
            cmd.append("--dev")

        subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        print(f"Roampal: starting fresh server on port {port}", file=sys.stderr)
    except Exception as e:
        print(f"Roampal: failed to start server: {e}", file=sys.stderr)
        return False

    # 3. Poll for health
    health_url = f"http://127.0.0.1:{port}/api/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print("Roampal: server restarted successfully", file=sys.stderr)
                    return True
        except Exception:
            pass
        time.sleep(1)

    print("Roampal: server restart timed out", file=sys.stderr)
    return False


def check_for_updates_cached() -> tuple:
    """Check if newer version available (cached to avoid repeated PyPI calls)."""
    global _update_check_cache

    if _update_check_cache["checked"]:
        return (_update_check_cache["available"],
                _update_check_cache["current"],
                _update_check_cache["latest"])

    try:
        from roampal import __version__

        url = "https://pypi.org/pypi/roampal/json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
            latest = data.get("info", {}).get("version", __version__)

            current_parts = [int(x) for x in __version__.split(".")]
            latest_parts = [int(x) for x in latest.split(".")]
            update_available = latest_parts > current_parts

            _update_check_cache["checked"] = True
            _update_check_cache["available"] = update_available
            _update_check_cache["current"] = __version__
            _update_check_cache["latest"] = latest

            return (update_available, __version__, latest)
    except Exception:
        _update_check_cache["checked"] = True
        return (False, "", "")


# v0.3.6: _fire_sidecar_background() removed — main LLM handles summarization
# via score_memories (Claude Code) or sidecar on session.idle (OpenCode)


# Fix Windows encoding issues with unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def main():
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # No input - pass through
        sys.exit(0)

    # Claude Code sends "prompt" field
    user_message = input_data.get("prompt", input_data.get("user_message", input_data.get("query", "")))

    if not user_message:
        sys.exit(0)

    # Get conversation_id - support both Claude Code (session_id) and Cursor (conversation_id)
    # This ensures completion state is tracked consistently across hooks
    conversation_id = input_data.get("conversation_id") or input_data.get("session_id", "default")

    # Call Roampal server for context
    # Respect ROAMPAL_DEV env var for port selection
    dev_mode = os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")
    default_port = 27183 if dev_mode else 27182
    server_url = os.environ.get("ROAMPAL_SERVER_URL", f"http://127.0.0.1:{default_port}")

    try:
        request_data = json.dumps({
            "query": user_message,
            "conversation_id": conversation_id
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{server_url}/api/hooks/get-context",
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))

        # Get the formatted injection
        formatted_injection = result.get("formatted_injection", "")

        if formatted_injection:
            # Print context to stdout - Claude Code adds this to conversation
            print(formatted_injection)

        # Check for updates (cached - only hits PyPI once per session)
        update_available, current, latest = check_for_updates_cached()
        if update_available:
            print(f"\n<roampal-update-available>Roampal update: {current} -> {latest}. Run: pip install --upgrade roampal && roampal init --force</roampal-update-available>")

        # v0.3.6: Sidecar summarization moved server-side (asyncio.create_task in get-context)
        # No more fire-and-forget subprocess — server handles it

        # Exit 0 = success, stdout added as context

        sys.exit(0)

    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        # v0.3.2: Self-healing — restart server and retry once
        is_503 = isinstance(e, urllib.error.HTTPError) and e.code == 503
        is_down = isinstance(e, urllib.error.URLError)

        if is_503 or is_down:
            reason = "unhealthy (embedding corruption)" if is_503 else "unavailable"
            print(f"Roampal server {reason}, attempting restart...", file=sys.stderr)

            if _restart_server(server_url, default_port):
                # Retry the original request
                try:
                    retry_req = urllib.request.Request(
                        f"{server_url}/api/hooks/get-context",
                        data=request_data,
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(retry_req, timeout=5) as response:
                        result = json.loads(response.read().decode("utf-8"))

                    formatted_injection = result.get("formatted_injection", "")
                    if formatted_injection:
                        print(formatted_injection)
                    sys.exit(0)
                except Exception as retry_err:
                    print(f"Roampal: retry failed after restart: {retry_err}", file=sys.stderr)

        elif isinstance(e, urllib.error.HTTPError):
            print(f"Roampal server error: HTTP {e.code}", file=sys.stderr)

        sys.exit(1)
    except Exception as e:
        print(f"Roampal hook error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

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
import urllib.request
import urllib.error

# Update check cache to avoid hitting PyPI on every message
_update_check_cache = {"checked": False, "available": False, "current": "", "latest": ""}


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
            print(f"\n<roampal-update-available>Roampal update: {current} -> {latest}. Run: pip install --upgrade roampal</roampal-update-available>")

        # Exit 0 = success, stdout added as context

        sys.exit(0)

    except urllib.error.HTTPError as e:
        # v0.3.0: Server returned error (e.g. 503 for embedding corruption)
        # Exit non-zero so user sees the error instead of silent failure
        if e.code == 503:
            print(f"Roampal server unhealthy (embedding error). Will auto-restart on next MCP tool call.", file=sys.stderr)
        else:
            print(f"Roampal server error: HTTP {e.code}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        # v0.3.0: Server not running - exit non-zero so user knows
        print(f"Roampal server unavailable: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # v0.3.0: Other error - exit non-zero for visibility
        print(f"Roampal hook error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Roampal UserPromptSubmit Hook

Called by Claude Code BEFORE the LLM sees the user's message.
This hook:
1. Checks if previous exchange needs scoring
2. Injects scoring prompt if needed
3. Injects relevant memories as context

Usage (in .claude/settings.json):
{
  "hooks": {
    "UserPromptSubmit": ["python", "-m", "roampal.hooks.user_prompt_submit_hook"]
  }
}

Environment variables:
- ROAMPAL_SERVER_URL: URL of Roampal server (default: http://127.0.0.1:27182)

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

    # Get session_id from Claude Code input - this matches what Stop hook uses
    # This ensures completion state is tracked consistently across hooks
    conversation_id = input_data.get("session_id", "default")

    # Call Roampal server for context
    server_url = os.environ.get("ROAMPAL_SERVER_URL", "http://127.0.0.1:27182")

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

        # Exit 0 = success, stdout added as context

        sys.exit(0)

    except urllib.error.URLError as e:
        # Server not running - no context to inject
        print(f"Roampal server unavailable: {e}", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        # Other error - no context to inject
        print(f"Roampal hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()

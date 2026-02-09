#!/usr/bin/env python3
"""
Roampal Stop Hook

Called by Claude Code / Cursor AFTER the LLM responds.
This hook:
1. Stores the exchange for later scoring
2. Checks if record_response() was called
3. BLOCKS (exit 2) if scoring was required but not done

Usage (Claude Code - .claude/settings.json):
{
  "hooks": {
    "Stop": [{"type": "command", "command": "python -m roampal.hooks.stop_hook"}]
  }
}

Usage (Cursor 1.7+ - .cursor/hooks.json):
{
  "version": 1,
  "hooks": {
    "stop": [{"command": "python -m roampal.hooks.stop_hook"}]
  }
}

Environment variables:
- ROAMPAL_DEV: Set to "1" to use dev port 27183 (default: prod port 27182)
- ROAMPAL_SERVER_URL: Override server URL (takes precedence over ROAMPAL_DEV)

Reads from stdin (Claude Code format):
- session_id: Conversation session ID
- transcript_path: Path to JSONL file with conversation history
- stop_hook_active: Boolean to prevent infinite loops

Exit codes:
- 0: Success, continue
- 2: Block - record_response() not called, inject message back to LLM
"""

import sys
import json
import os
import subprocess
import time
import urllib.request
import urllib.error


def _restart_server(server_url: str, port: int, timeout: float = 15.0) -> bool:
    """
    v0.3.2: Self-healing server restart for hooks.

    If FastAPI is down or unhealthy (503), kill the old process and start fresh.
    Same pattern as _ensure_server_running() in server.py but standalone (no roampal imports).
    """
    # 1. Kill whatever is on the port
    try:
        if sys.platform == "win32":
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
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                pid = result.stdout.strip().split('\n')[0]
                if pid.isdigit():
                    subprocess.run(["kill", "-9", pid], capture_output=True, timeout=5)
                    print(f"Roampal: killed stale server process {pid}", file=sys.stderr)
    except Exception:
        pass

    time.sleep(1)

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


def read_transcript(transcript_path: str) -> tuple[str, str, str]:
    """
    Read the transcript JSONL file and extract last user message,
    assistant response, and full transcript text.

    Claude Code transcript format:
    - type: "user" or "assistant" (top level)
    - message: { role: "user"|"assistant", content: [...] }
    """
    user_message = ""
    assistant_response = ""
    transcript_lines = []

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)

                    # Claude Code uses "type" at top level, not "role"
                    entry_type = entry.get("type", "")

                    # Content can be in message.content or directly in entry
                    message = entry.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", "")
                    else:
                        content = entry.get("content", "")

                    # Handle content that might be a list of content blocks
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                                elif block.get("type") == "tool_use":
                                    # Include tool calls in transcript for record_response detection
                                    tool_name = block.get("name", "")
                                    text_parts.append(f"[Tool: {tool_name}]")
                            elif isinstance(block, str):
                                text_parts.append(block)
                        content = "\n".join(text_parts)

                    if entry_type == "user":
                        user_message = content if content else user_message
                        if content:
                            transcript_lines.append(f"User: {content}")
                    elif entry_type == "assistant":
                        assistant_response = content if content else assistant_response
                        if content:
                            transcript_lines.append(f"Assistant: {content}")
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading transcript: {e}", file=sys.stderr)

    return user_message, assistant_response, "\n\n".join(transcript_lines)


def main():
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # No input or invalid JSON - just exit cleanly
        sys.exit(0)

    # Check if this is already a stop hook continuation (prevent infinite loops)
    if input_data.get("stop_hook_active", False):
        sys.exit(0)

    # Extract conversation data - support both Claude Code (session_id) and Cursor (conversation_id)
    conversation_id = input_data.get("conversation_id") or input_data.get("session_id", os.environ.get("ROAMPAL_CONVERSATION_ID", "default"))
    transcript_path = input_data.get("transcript_path", "")

    # Read the transcript file to get actual messages
    if transcript_path and os.path.exists(transcript_path):
        user_message, assistant_response, transcript = read_transcript(transcript_path)
    else:
        # Fallback for direct input (testing)
        user_message = input_data.get("user_message", "")
        assistant_response = input_data.get("assistant_response", "")
        transcript = input_data.get("transcript", "")

    # If no messages, nothing to do
    if not user_message and not assistant_response:
        print(f"Stop hook: no messages found for {conversation_id}, transcript_path={transcript_path}", file=sys.stderr)
        sys.exit(0)

    # Call Roampal server
    # Respect ROAMPAL_DEV env var for port selection
    dev_mode = os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")
    default_port = 27183 if dev_mode else 27182
    server_url = os.environ.get("ROAMPAL_SERVER_URL", f"http://127.0.0.1:{default_port}")

    try:
        request_data = json.dumps({
            "conversation_id": conversation_id,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "transcript": transcript
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{server_url}/api/hooks/stop",
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))

        # Check if we should block
        if result.get("should_block"):
            # Output the block message to stderr - exit code 2 shows stderr to Claude
            block_message = result.get("block_message", "")
            if block_message:
                print(block_message, file=sys.stderr)

            # Exit code 2 = block stopping, shows stderr to Claude
            sys.exit(2)

        # Success - exchange stored
        sys.exit(0)

    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        # v0.3.2: Self-healing — restart server and retry once
        is_503 = isinstance(e, urllib.error.HTTPError) and e.code == 503
        is_down = isinstance(e, urllib.error.URLError)

        if is_503 or is_down:
            reason = "unhealthy (embedding corruption)" if is_503 else "unavailable"
            print(f"Roampal server {reason}, attempting restart...", file=sys.stderr)

            if _restart_server(server_url, default_port):
                try:
                    retry_req = urllib.request.Request(
                        f"{server_url}/api/hooks/stop",
                        data=request_data,
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(retry_req, timeout=5) as response:
                        result = json.loads(response.read().decode("utf-8"))

                    if result.get("should_block"):
                        block_message = result.get("block_message", "")
                        if block_message:
                            print(block_message, file=sys.stderr)
                        sys.exit(2)
                    sys.exit(0)
                except Exception as retry_err:
                    print(f"Roampal: retry failed after restart: {retry_err}", file=sys.stderr)

        elif isinstance(e, urllib.error.HTTPError):
            print(f"Roampal server error: HTTP {e.code}", file=sys.stderr)

        # Stop hook never blocks on error — don't break the user's flow
        sys.exit(0)
    except Exception as e:
        print(f"Roampal hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()

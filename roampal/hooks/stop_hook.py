#!/usr/bin/env python3
"""
Roampal Stop Hook

Called by Claude Code AFTER the LLM responds.
This hook:
1. Stores the exchange for later scoring
2. Checks if record_response() was called
3. BLOCKS (exit 2) if scoring was required but not done

Usage (in .claude/settings.json):
{
  "hooks": {
    "Stop": [{"type": "command", "command": "python -m roampal.hooks.stop_hook"}]
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
import urllib.request
import urllib.error


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

    # Extract conversation data - Claude Code sends transcript_path, not raw content
    conversation_id = input_data.get("session_id", os.environ.get("ROAMPAL_CONVERSATION_ID", "default"))
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

    except urllib.error.URLError as e:
        # Server not running - log but don't block
        print(f"Roampal server unavailable: {e}", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        # Other error - log but don't block
        print(f"Roampal hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()

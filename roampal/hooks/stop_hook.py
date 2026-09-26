#!/usr/bin/env python3
"""
Roampal Stop Hook

Called by Claude Code / Cursor AFTER the LLM responds.
Handles two responsibilities:
1. Exchange tracking: Reads transcript and sends exchange data to server
   with lifecycle_only=True. Server stores in session JSONL (for scoring
   prompt generation via get_previous_exchange) but skips ChromaDB storage.
2. Turn lifecycle: Signals the server that the assistant turn is complete
   so the next UserPromptSubmit can inject scoring prompts.

v0.3.6: ChromaDB exchange storage moved to main LLM (via score_memories tool).
Stop hook handles JSONL lifecycle tracking + state management only.

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
- transcript_path: Path to conversation transcript JSONL
- stop_hook_active: Boolean to prevent infinite loops

Exit codes:
- 0: Success, continue
- 2: Block - score_memories() not called, inject message back to LLM
"""

import sys
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def _env_from_mcp_servers(servers) -> str:
    """Task 18: read ROAMPAL_PROFILE from roampal server entries in an
    mcpServers dict (whatever key name the servers are registered under)."""
    try:
        for server_name, server_cfg in servers.items():
            if "roampal" not in str(server_name).lower():
                continue
            server_env = server_cfg.get("env", {}) or {}
            val = server_env.get("ROAMPAL_PROFILE", "")
            if isinstance(val, str) and val.strip():
                return val.strip()
    except (TypeError, ValueError):
        pass
    return ""


def _claude_json_projects_entry(projects, cwd: Path) -> dict:
    """v0.6.0 review fix 3: Claude Code does NOT normalize project keys to
    the platform's native form — real ~/.claude.json files carry both
    'C:/proj' and 'C:\\proj' styles side by side (observed on Windows).
    A native-only lookup misses every forward-slash entry, so per-project
    ROAMPAL_PROFILE env was silently unreadable on Windows. Match
    tolerantly: exact native form first, then separator-flipped, then
    case-insensitive on Windows (drive-letter case varies too). POSIX
    lookups stay case-sensitive."""
    def norm(p: str) -> str:
        p = p.replace("\\", "/")
        return p.lower() if os.name == "nt" else p

    native = str(cwd)
    for candidate in (native, native.replace("\\", "/")):
        entry = projects.get(candidate)
        if isinstance(entry, dict):
            return entry
    target = norm(native)
    for key, entry in projects.items():
        if isinstance(entry, dict) and norm(key) == target:
            return entry
    return {}


def _profile_from_project_config() -> str:
    """Task 18: the project's MCP-config ROAMPAL_PROFILE, which the hook's
    own process env never sees (Claude Code merges .mcp.json env into the
    MCP process env only, not the hook env).

    v0.6.0 review fix 3: also reads the USER-scope mcpServers.*.env
    (root-level mcpServers in ~/.claude.json) — the place `roampal init`
    writes the server config — so hooks and MCP tools resolve the same
    profile from the same source. Precedence: project .mcp.json >
    per-project local scope > user scope > (caller falls through to
    bindings)."""
    def env_from(path) -> str:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            servers = data.get("mcpServers", {})
            if isinstance(servers, dict):
                return _env_from_mcp_servers(servers)
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        return ""

    # (a) .mcp.json in the cwd ancestry (project scope)
    current = Path(os.getcwd()).resolve()
    while True:
        candidate = current / ".mcp.json"
        if candidate.is_file():
            found = env_from(candidate)
            if found:
                return found
        if current.parent == current:
            break
        current = current.parent

    # (b) + (c) ~/.claude.json local scope (projects.<cwd>.mcpServers)
    # and user scope (root-level mcpServers). fix 3: (b)'s key match is
    # separator/case tolerant; (c) is new.
    claude_json = Path.home() / ".claude.json"
    if claude_json.is_file():
        try:
            data = json.loads(claude_json.read_text(encoding="utf-8"))
            projects = data.get("projects", {})
            if isinstance(projects, dict):
                project = _claude_json_projects_entry(
                    projects, Path(os.getcwd()).resolve()
                )
                if project:
                    servers = project.get("mcpServers", {})
                    if isinstance(servers, dict):
                        found = _env_from_mcp_servers(servers)
                        if found:
                            return found
            # (c) user scope: root-level mcpServers.*.env (fix 3)
            servers = data.get("mcpServers", {})
            if isinstance(servers, dict):
                found = _env_from_mcp_servers(servers)
                if found:
                    return found
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return ""


def _hook_default_port() -> int:
    """The server port this hook targets (matches main()'s selection)."""
    dev_mode = os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")
    return 27183 if dev_mode else 27182


def _hook_profile_name() -> str:
    """Task 18 hook precedence (mine, matching what the MCP process sees):
    process env > project MCP-config env > binding > `use` > default.
    The last two arrive via profile_header_value()'s full walk.

    v0.6.0 review fix 5: one more tier before "default" — the SERVER's
    launch pin (`roampal start --profile X`, per-port pin file). A session
    with nothing configured explicitly names the pinned profile instead of
    sending "default", so the pin reaches header-sending clients. F1 is
    intact: the CLIENT walks everything (now including the pin) and names
    its choice; the server never guesses."""
    env = os.environ.get("ROAMPAL_PROFILE", "").strip()
    if env:
        return env
    try:
        conf_profile = _profile_from_project_config()
        if conf_profile:
            return conf_profile
    except Exception:
        pass  # best effort — never block the hook on config parsing
    try:
        from roampal.profile_manager import profile_header_value, read_server_pin

        resolved = profile_header_value()
        if resolved == "default":
            pin = read_server_pin(_hook_default_port())
            if pin:
                return pin
        return resolved
    except Exception:
        return "default"


def _roampal_headers() -> dict:
    """v0.5.4: Build headers with X-Roampal-Profile so FastAPI hits the right
    profile instead of falling back to its own active_profile_name() (which
    only sees the FastAPI process's startup env, not this hook's per-invocation env).

    v0.6.0 Task 14: resolves through profile_manager's ONE helper
    (profile_header_value -> active_profile_name full precedence incl. cwd
    bindings); None (default) -> no header, FastAPI fallback unchanged.

    Round 2 Item 6 / Task 18: the hook ALWAYS names its profile — an
    explicit "default" replaces the bare request, so the server never
    guesses for it (F1).
    """
    profile = _hook_profile_name()
    return {"Content-Type": "application/json", "X-Roampal-Profile": profile}
import urllib.error


def _server_health_ok(server_url: str) -> bool:
    """Task 23, amended by v0.6.0 review fix 2: TRUE only when the server
    answers /api/health AND reports itself healthy (200).

    The server's ONLY 503s are broken states — a dead shared embed service
    (server/main.py health_check) or a failed profile init — never a
    'busy' signal (the server has no queue). A 503 therefore counts as
    DOWN for restart purposes: the single-flight restart replaces the
    process, which re-creates the embedder (what the health docstring
    always meant by 'allowing auto-restart'). Any OTHER HTTP answer
    (401/404/500 — e.g. a foreign process squatting the port) still counts
    as UP: never kill a process we cannot positively identify as a broken
    roampal (F2)."""
    try:
        req = urllib.request.Request(f"{server_url}/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        return e.code != 503
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _restart_server(server_url: str, port: int, timeout: float = 15.0) -> bool:
    """
    v0.3.2: Self-healing server restart for hooks.

    Round 2 Item 7 / Task 23 contract, amended by v0.6.0 review fix 2:
    - HEALTH-GATED: a server answering /api/health 200 is never restarted
      (the old path killed a healthy busy server on any trigger — F2
      incident 2026-09-18, PID 2348). A 503 health answer means the server
      reports itself BROKEN (dead embed service) — it IS replaced.
    - SINGLE-FLIGHT: a cross-process lock file (proc_lock, per port) makes
      concurrent restarters (CC hooks, MCP, the plugin) produce exactly
      one server; latecomers wait for the winner's health.
    """
    try:
        from roampal.utils import proc_lock
    except Exception:
        proc_lock = None  # best effort — restart without single-flight

    # 0. Down only — an answering server is never touched.
    if _server_health_ok(server_url):
        return True

    if proc_lock is None:
        return _spawn_fresh_server(server_url, port)

    lock = proc_lock.acquire(port)
    if lock is None:
        # Someone else is mid-restart; wait for THEIR health, don't spawn twice.
        print("Roampal: restart already in progress by another client", file=sys.stderr)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if _server_health_ok(server_url):
                print("Roampal: server restarted by another client", file=sys.stderr)
                return True
            time.sleep(1)
        return False

    try:
        # Re-check under the lock: the other restarter likely just finished.
        if _server_health_ok(server_url):
            return True

        _spawn_fresh_server(server_url, port)
        return _poll_health(server_url, timeout)
    finally:
        proc_lock.release(port)


def _spawn_fresh_server(server_url: str, port: int) -> bool:
    """Best-effort cleanup of a stale port holder + spawn a fresh server.

    Kept from the v0.3.2 flow but ONLY reachable on the DOWN path (Task 23)
    — the health check in _restart_server already gates this function to
    servers that answer nothing."""
    try:
        if sys.platform == "win32":
            # netstat to find PID, taskkill to end it
            result = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=5
            )
            port_str = f"127.0.0.1:{port}"
            for line in result.stdout.splitlines():
                if port_str in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    if pid.isdigit():
                        subprocess.run(
                            ["taskkill", "/pid", pid, "/f"],
                            capture_output=True, timeout=5
                        )
                        print(f"Roampal: restarting server...", file=sys.stderr)
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
                    print(f"Roampal: restarting server...", file=sys.stderr)
    except Exception:
        pass  # Best effort — if we can't kill, the new server will fail to bind and we'll exit

    time.sleep(1)  # Let port release

    # 2. Start fresh server
    try:
        env = os.environ.copy()
        dev_mode = env.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")
        # Round 2 Item 6 / Task 17: the shared server never reads
        # ROAMPAL_PROFILE (headerless requests route via persisted `use`
        # -> default); strip this hook process's env so one project's
        # shell cannot reach the server process.
        env.pop("ROAMPAL_PROFILE", None)

        # Task 36: the hook process's cwd (the project dir Claude Code runs
        # hooks in) must not reach the child's sys.path — cwd is pinned to
        # the neutral data dir for that reason. v0.6.0 review fix 1: -E
        # (PYTHONPATH/PYTHONHOME ignored) instead of -I — -I's implied -s
        # hides USER site-packages, breaking Store-Python/pip --user installs.
        # -P (3.11+) additionally keeps the spawn cwd off sys.path.
        from roampal.profile_manager import spawn_isolation_flags, read_server_pin

        cmd = [sys.executable, *spawn_isolation_flags(), "-m", "roampal.server.main", "--port", str(port)]
        # v0.6.0 review fix 5: a respawn of a pinned server re-passes the
        # launch pin (per-port pin file), preserving its routing identity.
        _pin = read_server_pin(port)
        if _pin:
            cmd += ["--profile", _pin]
        if dev_mode:
            cmd.append("--dev")

        from roampal.profile_manager import neutral_spawn_dir
        subprocess.Popen(
            cmd, env=env, cwd=str(neutral_spawn_dir()),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        print(f"Roampal: starting fresh server on port {port}", file=sys.stderr)
    except Exception as e:
        print(f"Roampal: failed to start server: {e}", file=sys.stderr)
        return False

    return True


def _poll_health(server_url: str, timeout: float) -> bool:
    """Task 23: wait for the freshly spawned server to answer health."""
    start = time.time()
    while time.time() - start < timeout:
        if _server_health_ok(server_url):
            print("Roampal: server restarted successfully", file=sys.stderr)
            return True
        time.sleep(1)

    print("Roampal: server restart timed out", file=sys.stderr)
    return False


def read_transcript(transcript_path: str) -> tuple[str, str]:
    """
    Read the transcript JSONL file and extract last user message and assistant response.

    Claude Code transcript format:
    - type: "user" or "assistant" (top level)
    - message: { role: "user"|"assistant", content: [...] }
    """
    user_message = ""
    assistant_response = ""

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
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif isinstance(block, str):
                                text_parts.append(block)
                        content = "\n".join(text_parts)

                    if entry_type == "user":
                        user_message = content if content else user_message
                    elif entry_type == "assistant":
                        assistant_response = content if content else assistant_response
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading transcript: {e}", file=sys.stderr)

    return user_message, assistant_response


_DIAG_LOG = os.path.join(os.path.expanduser("~"), ".claude", "stop_hook_diag.log")


def _diag(msg: str):
    """Diagnostic logging to file + stderr."""
    line = f"[roampal-stop-diag] {msg}"
    print(line, file=sys.stderr)
    try:
        with open(_DIAG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass


def main():
    # Breadcrumb: touch a file to prove this hook was invoked at all
    try:
        breadcrumb = os.path.join(os.path.expanduser("~"), ".claude", "stop_hook_breadcrumb.txt")
        with open(breadcrumb, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} stop_hook invoked\n")
    except Exception:
        pass

    # Read hook input from stdin
    try:
        raw_input = sys.stdin.read()
        _diag(f"stdin length: {len(raw_input)} chars")
        if not raw_input.strip():
            _diag("stdin was empty — exiting")
            sys.exit(0)
        input_data = json.loads(raw_input)
        _diag(f"stdin keys: {list(input_data.keys())}")
    except (json.JSONDecodeError, Exception) as e:
        _diag(f"stdin parse error: {e}")
        sys.exit(0)

    # Check if this is already a stop hook continuation (prevent infinite loops)
    if input_data.get("stop_hook_active", False):
        _diag("stop_hook_active=True — exiting to prevent loop")
        sys.exit(0)

    # Extract conversation ID - support both Claude Code (session_id) and Cursor (conversation_id)
    conversation_id = input_data.get("conversation_id") or input_data.get("session_id", os.environ.get("ROAMPAL_CONVERSATION_ID", "default"))
    _diag(f"conversation_id={conversation_id}")

    # v0.4.0: Read last_assistant_message from input if available (Claude Code v2.1+)
    # Fall back to transcript parsing for older versions
    user_message = ""
    assistant_response = input_data.get("last_assistant_message", "")
    if assistant_response:
        _diag(f"got last_assistant_message from input ({len(assistant_response)} chars)")
        # Try to get user message from transcript for completeness
        transcript_path = input_data.get("transcript_path", "")
        if transcript_path and os.path.exists(transcript_path):
            user_msg_from_transcript, _ = read_transcript(transcript_path)
            user_message = user_msg_from_transcript
    else:
        # Legacy path: parse transcript file
        transcript_path = input_data.get("transcript_path", "")
        if transcript_path and os.path.exists(transcript_path):
            user_message, assistant_response = read_transcript(transcript_path)
            _diag(f"parsed user_message length: {len(user_message)}")
            _diag(f"parsed assistant_response length: {len(assistant_response)}")

    # Call Roampal server
    dev_mode = os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")
    default_port = 27183 if dev_mode else 27182
    server_url = os.environ.get("ROAMPAL_SERVER_URL", f"http://127.0.0.1:{default_port}")

    try:
        request_data = json.dumps({
            "conversation_id": conversation_id,
            "user_message": user_message,
            "assistant_response": assistant_response,
            "lifecycle_only": True,  # v0.3.6: track in JSONL but skip ChromaDB storage
        }).encode("utf-8")

        _diag(f"POSTing to {server_url}/api/hooks/stop ({len(request_data)} bytes)")

        req = urllib.request.Request(
            f"{server_url}/api/hooks/stop",
            data=request_data,
            headers=_roampal_headers(),
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))

        _diag(f"server response: {json.dumps(result)[:500]}")

        # Check if we should block
        if result.get("should_block"):
            block_message = result.get("block_message", "")
            if block_message:
                print(block_message, file=sys.stderr)

            _diag("BLOCKING — exit code 2")
            sys.exit(2)

        _diag("SUCCESS — state updated, exit 0")
        sys.exit(0)

    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        _diag(f"server error: {type(e).__name__}: {e}")
        # v0.6.0 review fix 8: a 404 is NOT a down/degraded server — it is
        # the routing contract rejecting an unknown profile (the request's
        # X-Roampal-Profile names a deleted/unregistered profile). Print
        # the server's actionable detail (it includes the exact fix
        # command) and bail: restarting or retrying cannot help. The stop
        # hook never blocks the user's flow on errors — exit 0.
        if isinstance(e, urllib.error.HTTPError) and e.code == 404:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("detail", "")
            except Exception:
                detail = ""
            print(f"Roampal: {detail or 'HTTP 404 from server'}", file=sys.stderr)
            sys.exit(0)

        # Round 2 Item 7 / Task 23, amended by v0.6.0 review fix 2:
        # - 503: the server is UP but BROKEN (dead embed service / failed
        #   profile init — the server has no 'busy' 503). Route through the
        #   health-gated single-flight restart: a healthy server is left
        #   untouched (F2), a degraded one is replaced, then retry once.
        # - down: restart exactly once (single-flight, down-only), retry once.
        is_503 = isinstance(e, urllib.error.HTTPError) and e.code == 503
        is_down = isinstance(e, urllib.error.URLError)

        should_retry = False
        if is_503:
            print("Roampal: server degraded (503) — health-gated restart check", file=sys.stderr)
            time.sleep(1.0)
            should_retry = _restart_server(server_url, default_port)
        elif is_down:
            reason = "unavailable"
            print(f"Roampal server {reason}, attempting restart...", file=sys.stderr)
            should_retry = _restart_server(server_url, default_port)

        if should_retry:
            try:
                retry_req = urllib.request.Request(
                    f"{server_url}/api/hooks/stop",
                    data=request_data,
                    headers=_roampal_headers(),
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
        _diag(f"unexpected error: {type(e).__name__}: {e}")
        print(f"Roampal hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()

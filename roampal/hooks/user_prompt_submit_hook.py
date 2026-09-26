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
from pathlib import Path

# Update check cache to avoid hitting PyPI on every message
_update_check_cache = {"checked": False, "available": False, "current": "", "latest": ""}


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


def check_for_updates_cached() -> tuple:
    """Check if newer version available (cached to avoid repeated PyPI calls)."""
    global _update_check_cache

    if _update_check_cache["checked"]:
        return (_update_check_cache["available"],
                _update_check_cache["current"],
                _update_check_cache["latest"])

    try:
        from importlib.metadata import version as _pkg_version
        __version__ = _pkg_version("roampal")

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


def _ack_injection(server_url: str, result: dict) -> None:
    """v0.6.0 Task 30: confirm delivery of this turn's memory block.

    The server holds the surfaced memories as pending until this ack; only
    then do they count as shown (pointer lines on later turns). Called right
    before the zero exit, after the block is printed. Best effort: if the ack
    fails, the memories simply show in full again next turn."""
    token = (result or {}).get("injection_token") or ""
    if not token:
        return
    try:
        req = urllib.request.Request(
            f"{server_url}/api/hooks/injection-ack",
            data=json.dumps({"injection_token": token}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0):
            pass
    except Exception:
        pass  # never fail the prompt over the ack


def _preflight_degradation_check(server_url: str, port: int) -> None:
    """v0.6.0 review fix 2, amended by round 2: catch a DEGRADED server
    before it eats the prompt — but ONLY on the server's own 503 answer.

    The server's only 503 is a broken state (dead embed service / failed
    profile init) — never busy. A degraded server keeps ANSWERING
    get-context (with empty memories), so failure-driven restart never
    fires on its own; this probe catches it. A healthy server costs one
    ~2ms localhost GET and is never touched (F2). Crucially, a BUSY or
    slow server (health times out within 2s) is LEFT ALONE here — this
    path runs on every prompt, so timing out would turn it into a new,
    per-prompt kill opportunity for healthy-but-loaded servers (round-2
    finding). Slow-but-answering is not broken."""
    try:
        req = urllib.request.Request(f"{server_url}/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=2.0):
            return  # healthy up — nothing to do
    except urllib.error.HTTPError as e:
        if e.code != 503:
            return  # foreign/other HTTP answer — not our signal (F2)
        print("Roampal: server reports degraded (503) — restarting", file=sys.stderr)
        _restart_server(server_url, port)
    except (urllib.error.URLError, TimeoutError, OSError):
        pass  # down/slow — the real request's failure path handles restart
    except Exception:
        pass  # best effort — never block the prompt on the probe


def main():
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # No input - pass through
        sys.exit(0)

    # Claude Code sends "prompt" field; Cursor sends "user_message"
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

    # v0.6.0 review fix 2: a 503-ing (degraded) server keeps answering
    # requests, so only a proactive health probe catches it — see
    # _preflight_degradation_check.
    _preflight_degradation_check(server_url, default_port)

    try:
        # v0.6.0 Task 30: Claude Code hook output is APPEND-ONLY — repeats
        # sit in old context until compaction. This hook therefore opts its
        # surface into the server's injection-replay dedup (pointer lines
        # for unchanged repeats). OpenCode's rebuilt-per-request surface
        # deliberately does NOT set this flag (a pointer line there would
        # hide the memory, not dedup it).
        request_data = json.dumps({
            "query": user_message,
            "conversation_id": conversation_id,
            "dedup_injections": True,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{server_url}/api/hooks/get-context",
            data=request_data,
            headers=_roampal_headers(),
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=10) as response:
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

        # Exit 0 = success, stdout added as context. Ack LAST: Claude Code
        # only keeps stdout from a zero exit, so nothing above may fail after it.
        _ack_injection(server_url, result)
        sys.exit(0)

    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        # v0.6.0 review fix 8: a 404 is NOT a down/degraded server — it is
        # the routing contract rejecting an unknown profile (the request's
        # X-Roampal-Profile names a deleted/unregistered profile). Print
        # the server's actionable detail (it includes the exact fix
        # command) and bail: restarting or retrying cannot help.
        if isinstance(e, urllib.error.HTTPError) and e.code == 404:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("detail", "")
            except Exception:
                detail = ""
            print(f"Roampal: {detail or 'HTTP 404 from server'}", file=sys.stderr)
            sys.exit(1)

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
            print(f"Roampal: server restarting, please wait...", file=sys.stderr)
            should_retry = _restart_server(server_url, default_port)

        if should_retry:
            # Retry the original request
            try:
                retry_req = urllib.request.Request(
                    f"{server_url}/api/hooks/get-context",
                    data=request_data,
                    headers=_roampal_headers(),
                    method="POST"
                )
                with urllib.request.urlopen(retry_req, timeout=10) as response:
                    result = json.loads(response.read().decode("utf-8"))

                formatted_injection = result.get("formatted_injection", "")
                if formatted_injection:
                    print(formatted_injection)
                _ack_injection(server_url, result)
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

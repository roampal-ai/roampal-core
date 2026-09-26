"""Sidecar group (v0.6.0 Task 9): detect/setup/status/disable/test pipeline.

Largest moved group (~1,113 lines): _safe_write_opencode_config,
_find_project_opencode_config, _apply_sidecar_env_and_write, the model
detectors (_detect_*), _recommend_model, the prompt/picker flow,
_prompt_smart_onboarding, cmd_sidecar, and _cmd_sidecar_* handlers. The two
config-location predicates from the same monolith span
(_check_sidecar_configured, _get_opencode_config_path) went to _common.py
per the G3 decision instead. All helpers moved verbatim from the
pre-refactor roampal/cli.py; cross-group consumers (setup.py, memory_cmds,
scoring) retarget their imports here / to _common in this task.
"""

import contextlib
import json
import os
import sys
import urllib.request
from pathlib import Path

from roampal.cli._common import (
    BLUE,
    BOLD,
    GREEN,
    RED,
    RESET,
    YELLOW,
    _check_sidecar_configured,
    _get_opencode_config_path,
)
from roampal.utils.safe_config import ConfigReadError, read_json_config, write_json_config


# ============================================================================
# Section 8 & 9 helpers — atomic writes, scope-awareness, backup
# ============================================================================

def _safe_write_opencode_config(path: Path, config: dict) -> None:
    """Write opencode.json atomically with timestamped backup (Task 40).

    v0.5.3 Section 8/9: All writes to opencode.json route through this helper.
    Replaces inline `config_path.write_text(json.dumps(config, indent=2))`
    everywhere in cli.py.

    v0.6.0 Task 40: delegates to `roampal.utils.safe_config.write_json_config`
    — same atomic temp-file + os.replace write, same `<name>.bak-<timestamp>`
    backup, plus crash-safe behavior and backup pruning to the newest 3
    (0.5.3-0.5.9 piles grew unbounded — 17 on the dev machine).
    """
    write_json_config(path, config)


def _find_project_opencode_config(cwd: Path | None = None) -> Path | None:
    """Walk up from `cwd` to find the first opencode.json in ancestry.

    v0.5.3 Section 9: Scope-aware sidecar commands need to know whether a
    project-local config exists before deciding where to write.

    Returns the path if found, or None (never returns user-global).
    Walks up to root but stops at home directory for user-global.
    """
    if cwd is None:
        cwd = Path.cwd()

    current = cwd.resolve()
    # Stop walking when we reach the home directory — that's where user-global lives
    home = Path.home().resolve()

    while True:
        candidate = current / "opencode.json"
        if candidate.exists():
            return candidate
        if current == home or current.parent == current:
            break
        current = current.parent

    return None


def _apply_sidecar_env_and_write(
    path: Path, env_updates: dict[str, str | None]
) -> bool:
    """Apply sidecar environment updates to a config file and write atomically.

    v0.5.3 Section 9: Shared helper for setup (env_updates has values) and
    disable (env_updates is empty = clear keys).

    Args:
        path: opencode.json path to modify
        env_updates: dict of {key: value} — set value, or None to remove key

    Returns True if any change was made.
    """
    sidecar_keys = [
        "ROAMPAL_SIDECAR_FALLBACK",
        "ROAMPAL_SIDECAR_URL",
        "ROAMPAL_SIDECAR_KEY",
        "ROAMPAL_SIDECAR_MODEL",
        "ROAMPAL_SIDECAR_PRIORITY",
    ]

    # Task 40: safe reader — a missing file returns {} (falls into the
    # no-roampal-core check below), an unreadable one aborts without writing.
    try:
        config = read_json_config(path)
    except ConfigReadError as e:
        print(f"  {RED}[ERROR] Cannot parse {path}:{RESET} {e.reason}")
        print(f"    Fix the JSON or back up + delete to regenerate.")
        return False

    if "mcp" not in config or "roampal-core" not in config["mcp"]:
        return False

    mcp_env = config.setdefault("mcp", {}).setdefault("roampal-core", {}).setdefault(
        "environment", {}
    )

    changed = False
    for key, value in env_updates.items():
        if value is None:
            # Remove the key (disable path)
            if key in mcp_env:
                mcp_env.pop(key)
                changed = True
        else:
            # Set the key (setup path)
            if mcp_env.get(key) != value:
                mcp_env[key] = value
                changed = True

    if not env_updates and sidecar_keys:
        # Disable: clear all sidecar keys even if updates dict is empty
        for key in sidecar_keys:
            if key in mcp_env:
                mcp_env.pop(key)
                changed = True

    if changed:
        _safe_write_opencode_config(path, config)

    return changed


_DEFAULT_GO_MODELS = [
    "glm-5.1",
    "qwen3.5-plus",
    "deepseek-v4-flash",
]


def _detect_opencode_go() -> dict | None:
    """Detect an OpenCode Go subscription via auth.json.

    Returns {url, key, models} if detected, or None.
    """
    parents = [Path.home() / ".local" / "share", Path.home() / ".config"]
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        parents.append(Path(appdata))
    for parent in parents:
        auth_path = parent / "opencode" / "auth.json"
        if not auth_path.exists():
            continue
        try:
            auth = json.loads(auth_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        entry = auth.get("opencode-go")
        if not isinstance(entry, dict) or entry.get("type") != "api" or not entry.get("key"):
            continue
        key = entry["key"].strip()
        if not key:
            continue
        url = "https://opencode.ai/zen/go/v1"
        fetched = _list_opencode_go_models(url, key)
        if not fetched:
            print(f"  {YELLOW}Note: Using default Go model list (API fetch failed){RESET}")
        models = fetched or _DEFAULT_GO_MODELS
        return {"url": url, "key": key, "models": models}
    return None


def _list_opencode_go_models(url: str, key: str) -> list[str] | None:
    """Hit /models on the Go endpoint to list available models.

    Filters out non-OpenAI-compatible models (MiniMax uses Anthropic's
    /v1/messages endpoint, which our sidecar OpenAI client cannot call).
    """
    import ssl

    try:
        req = urllib.request.Request(
            f"{url}/models",
            headers={
                "Authorization": f"Bearer {key}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
        )
        with urllib.request.urlopen(req, timeout=5, context=ssl.create_default_context()) as resp:
            data = json.loads(resp.read().decode())
            ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
            # MiniMax models route through Anthropic /v1/messages, not OpenAI
            # /chat/completions — exclude them so the sidecar doesn't 403/400.
            ids = [m for m in ids if not m.startswith("minimax-")]
            return ids or None
    except Exception as e:
        # Redact key from error — str(e) may contain the bearer token bytes
        err_msg = str(e).replace(key, "sk-**redacted**") if len(key) > 4 else str(e)
        print(f"  {YELLOW}Note: Failed to fetch Go models from API ({type(e).__name__}: {err_msg}){RESET}")
        return None


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
                        results.append(
                            {
                                "name": model_id,
                                "port": port,
                                "server_label": label,
                                "source": "local",
                            }
                        )
                return results
        except Exception:
            return []

    # Also check OLLAMA_HOST for non-standard Ollama port (but use /v1/models, not /api/tags)
    ollama_host = os.environ.get("OLLAMA_HOST", "")
    extra_probes = []
    if ollama_host:
        # Parse host:port from OLLAMA_HOST (e.g. "http://192.168.1.5:11434" or "localhost:9999")
        host_str = (
            ollama_host.replace("http://", "").replace("https://", "").rstrip("/")
        )
        if ":" in host_str:
            try:
                port = int(host_str.split(":")[-1])
                if port != 11434:  # Skip default — _detect_ollama_models handles it
                    extra_probes.append((port, f"Ollama ({host_str})"))
            except ValueError:
                pass

    all_servers = LOCAL_SERVERS + extra_probes
    results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(all_servers)
    ) as executor:
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


def _recommend_model(existing_models: list) -> dict:
    """Pick the best sidecar model from detected models.

    Priority: smallest local Ollama > other local > API.
    Sidecar only needs basic JSON output, so smaller is better.
    """
    # Prefer smallest Ollama model (free, fast, private)
    ollama_models = [m for m in existing_models if m.get("source") == "ollama"]
    if ollama_models:
        sorted_models = sorted(ollama_models, key=lambda m: m.get("size_gb", 999))
        return {"model": sorted_models[0]}

    # Other local servers
    local_models = [m for m in existing_models if m.get("source") == "local"]
    if local_models:
        return {"model": local_models[0]}

    # API models (costs money)
    api_models = [
        m for m in existing_models if m.get("source") == "api" and m.get("has_key")
    ]
    if api_models:
        return {"model": api_models[0]}

    return {}


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
            model_name = (
                model_cfg.get("name", model_id)
                if isinstance(model_cfg, dict)
                else model_id
            )
            models.append(
                {
                    "name": model_id,
                    "display": f"{model_name} ({provider_name})",
                    "source": "api",
                    "has_key": bool(api_key),
                    "base_url": base_url,
                    "api_key": api_key,
                }
            )
    return models


def _build_sidecar_env_updates(chosen: dict) -> dict[str, str | None]:
    """Build env updates dict from a chosen model.

    v0.5.3 Section 9: Returns {key: value} pairs for _apply_sidecar_env_and_write().
    """
    source = chosen.get("source", "")
    model_name = chosen.get("name", "")
    updates: dict[str, str | None] = {}

    if source == "ollama":
        updates["ROAMPAL_SIDECAR_URL"] = "http://localhost:11434/v1"
        updates["ROAMPAL_SIDECAR_MODEL"] = model_name
        updates["ROAMPAL_SIDECAR_KEY"] = None  # remove key
    elif source == "local":
        port = chosen.get("port", 8080)
        updates["ROAMPAL_SIDECAR_URL"] = f"http://localhost:{port}/v1"
        updates["ROAMPAL_SIDECAR_MODEL"] = model_name
        updates["ROAMPAL_SIDECAR_KEY"] = None  # remove key
    elif source == "api":
        base_url = chosen.get("base_url", "")
        api_key = chosen.get("api_key", "")
        updates["ROAMPAL_SIDECAR_URL"] = base_url
        updates["ROAMPAL_SIDECAR_MODEL"] = model_name
        if api_key:
            updates["ROAMPAL_SIDECAR_KEY"] = api_key
        else:
            updates["ROAMPAL_SIDECAR_KEY"] = None  # remove key
    else:
        return {}

    # Clean up legacy flag — URL/MODEL is all the plugin needs
    updates["ROAMPAL_SIDECAR_FALLBACK"] = None  # remove legacy
    # v0.6.0 Task 39: a chosen model replaces an earlier Zen opt-in.
    updates["ROAMPAL_SIDECAR_PRIORITY"] = None
    return updates


def _prompt_custom_endpoint(config_path: Path) -> bool:
    """Interactive prompt for custom sidecar API endpoint.

    v0.5.3 Section 9: Uses _apply_sidecar_env_and_write() instead of direct write.
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

        api_key = getpass.getpass(
            "  API Key (optional for local, Enter to skip): "
        ).strip()

        model = input(f"  Model name (e.g. llama-3.3-70b-versatile): ").strip()
        if not model:
            print(f"  {YELLOW}Model name required.{RESET}")
            return False
    except (EOFError, KeyboardInterrupt):
        print(f"\n  {YELLOW}Cancelled.{RESET}")
        return False

    # Build env updates and write atomically via shared helper
    updates = {
        "ROAMPAL_SIDECAR_URL": url,
        "ROAMPAL_SIDECAR_MODEL": model,
        "ROAMPAL_SIDECAR_KEY": api_key if api_key else None,
        "ROAMPAL_SIDECAR_FALLBACK": None,  # remove legacy
    }

    changed = _apply_sidecar_env_and_write(config_path, updates)
    if not changed:
        print(
            f"  {RED}roampal-core MCP not configured. Run {BLUE}roampal init --opencode{RESET} first.{RESET}"
        )
        return False

    print(f"\n  {GREEN}Custom sidecar configured!{RESET}")
    print(f"    URL:   {url}")
    print(f"    Model: {model}")
    if api_key:
        print(f"    Key:   {'*' * min(len(api_key), 8)}...")
    return True


def _collect_sidecar_options(config: dict) -> list:
    """Every detected scoring backend, in picker order, as (label, model).

    Shared by the interactive picker and the non-interactive flags (Task 39)
    so the two can never offer different choices:
      1. Ollama models, smallest first
      2. other local servers (LM Studio etc.)
      3. API models from opencode.json providers that have a key (paid)
    Embedding-only models are skipped — they can't produce summaries.
    """
    options = []
    for m in sorted(_detect_ollama_models(), key=lambda m: m.get("size_gb", 999)):
        name = m.get("name", "")
        if "embed" in name.lower():
            continue
        size = m.get("size_gb", 0)
        size_str = f", {size}GB" if size else ""
        options.append((f"{name} (Ollama{size_str}, free)", m))
    for m in _detect_local_servers():
        name = m.get("name", "")
        if "embed" in name.lower():
            continue
        options.append((f"{name} ({m.get('server_label', 'local')}, free)", m))
    for m in _detect_api_models(config):
        if m.get("has_key"):
            options.append((f"{m.get('display', '?')} (API, costs money)", m))
    return options


def _describe_configured_sidecar(config: dict) -> str | None:
    """Human description of the scoring model already recorded in an
    opencode.json dict, or None when nothing is configured (Task 44)."""
    env = (config.get("mcp", {}).get("roampal-core", {}) or {}).get("environment", {}) or {}
    url = env.get("ROAMPAL_SIDECAR_URL", "")
    model = env.get("ROAMPAL_SIDECAR_MODEL", "")
    if url and model:
        from urllib.parse import urlparse

        return f"{model} @ {urlparse(url).netloc or url}"
    if _zen_opted_in({"priority": env.get("ROAMPAL_SIDECAR_PRIORITY", "")}):
        return "free Zen cloud models (opencode.ai)"
    return None


def _print_skip_message(current: str | None) -> None:
    """Skip writes nothing, so say what that actually leaves in place."""
    if current:
        print(f"\n{YELLOW}Skipped. Your current scoring model stays: {current}.{RESET}")
        print(f"  Change it any time: {BLUE}roampal sidecar setup{RESET}")
        return
    print(
        f"\n{YELLOW}Skipping sidecar setup. Scoring, summaries, and fact"
        f" extraction are disabled.{RESET}"
    )
    print(f"  Retrieval from existing memories still works.")
    print(
        f"  Run {BLUE}roampal sidecar setup{RESET} when you're ready to enable scoring."
    )


def _sidecar_model_picker(
    config_path: Path, defer_write: bool = False
) -> dict | None:
    """Unified sidecar model selection — used by both init and sidecar setup.

    v0.5.3 Section 9: `defer_write=True` skips writing to disk; caller handles
    scope-aware writes via _apply_sidecar_env_and_write().

    Args:
        config_path: path to the opencode.json file
        defer_write: if True, return model info without modifying the file
    """
    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"\n{RED}Cannot read {config_path}{RESET}")
        return None

    current = _describe_configured_sidecar(config)
    if current:
        print(f"\n  Current scoring model: {BOLD}{current}{RESET}")

    print(f"\n{BOLD}Scanning for available models...{RESET}")

    options = _collect_sidecar_options(config)  # list of (label, model_dict)

    if not options:
        # Nothing detected — show install guidance
        print(f"\n  {YELLOW}No local models or API keys detected.{RESET}")
        print(f"\n  You can use local models from Ollama, LM Studio, or similar.")
        print(f"  Any small model works — the sidecar just produces JSON summaries.\n")
        print(f"    Ollama:     https://ollama.com → ollama pull qwen3:8b")
        print(f"    LM Studio:  https://lmstudio.ai")
        print(f"    Then run:   roampal sidecar setup\n")

        print(f"  {BOLD}[1]{RESET} Configure custom API (Groq, DeepSeek, etc.)")

        go = _detect_opencode_go()
        if go:
            print(
                f"  {BOLD}[2]{RESET} Use OpenCode Go (detected) {GREEN}— your subscription, your quota{RESET}"
            )
            print(
                f"      {YELLOW}Reliable scoring via Go's API. Each scored exchange consumes Go credits.{RESET}"
            )

        zen_idx = "3" if go else "2"
        skip_idx = "4" if go else "3"
        print(
            f"  {BOLD}[{zen_idx}]{RESET} Use free Zen cloud models {YELLOW}(rate-limited, may be flaky — data sent to opencode.ai){RESET}"
        )
        print(
            f"      {YELLOW}Note: OpenCode Go subscribers also use this path — Go quota is not consumed.{RESET}"
        )
        print(
            f"  {BOLD}[{skip_idx}]{RESET} Skip — no scoring, no summaries, no fact extraction (retrieval still works)"
        )

        try:
            choice = input(f"\nChoose [1-{skip_idx}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Cancelled.{RESET}")
            return None

        if choice == "1":
            result = _prompt_custom_endpoint(config_path)
            if defer_write and result:
                # Custom endpoint was set — return it for deferred write
                try:
                    cfg = json.loads(config_path.read_text())
                    env = cfg.get("mcp", {}).get("roampal-core", {}).get(
                        "environment", {}
                    )
                    if env.get("ROAMPAL_SIDECAR_URL"):
                        return {
                            "url": env["ROAMPAL_SIDECAR_URL"],
                            "model": env.get("ROAMPAL_SIDECAR_MODEL", ""),
                        }
                except Exception:
                    pass
            return result

        elif go and choice == "2":
            # OpenCode Go model picker
            print(f"\n{BOLD}Available OpenCode Go models for scoring:{RESET}")
            for i, m in enumerate(go["models"], 1):
                print(f"  {BOLD}[{i}]{RESET} {m}")
            try:
                m_choice = input(f"\nChoose [1-{len(go['models'])}]: ").strip()
                m_idx = int(m_choice) - 1
                if not (0 <= m_idx < len(go["models"])):
                    raise ValueError
            except (ValueError, EOFError, KeyboardInterrupt):
                print(f"\n{YELLOW}Cancelled.{RESET}")
                return None

            chosen_model = go["models"][m_idx]
            print(f"\n{GREEN}Configuring: OpenCode Go ({chosen_model}){RESET}")
            print(f"  {YELLOW}Note: each scored exchange consumes Go credits.{RESET}")

            updates = {
                "ROAMPAL_SIDECAR_URL": go["url"],
                "ROAMPAL_SIDECAR_KEY": go["key"],
                "ROAMPAL_SIDECAR_MODEL": chosen_model,
                "ROAMPAL_SIDECAR_FALLBACK": None,
                "ROAMPAL_SIDECAR_PRIORITY": None,
            }
            if defer_write:
                return {
                    "ROAMPAL_SIDECAR_URL": go["url"],
                    "ROAMPAL_SIDECAR_KEY": go["key"],
                    "ROAMPAL_SIDECAR_MODEL": chosen_model,
                    "ROAMPAL_SIDECAR_FALLBACK": None,
                    "ROAMPAL_SIDECAR_PRIORITY": None,
                }
            _apply_sidecar_env_and_write(config_path, updates)
            return True

        elif choice == zen_idx:
            # v0.5.3: Explicit Zen opt-in — writes ROAMPAL_SIDECAR_PRIORITY=zen
            # so the user has clearly chosen the cloud fallback. Previously
            # this path wrote no config and relied on a hidden default cascade.
            print(f"\n{YELLOW}Using free Zen cloud models.{RESET}")
            print(
                f"  {BOLD}Note:{RESET} Zen is rate-limited and occasionally unreachable."
            )
            print(
                f"  Exchange data is sent to opencode.ai/zen for scoring."
            )
            print(
                f"  If Zen fails, OpenCode exchanges will {BOLD}NOT{RESET} be stored or scored"
            )
            print(
                f"  until you configure a dedicated model via {BLUE}roampal sidecar setup{RESET}."
            )
            if defer_write:
                return {"url": "zen", "model": "zen"}
            else:
                updates = {
                    "ROAMPAL_SIDECAR_URL": None,
                    "ROAMPAL_SIDECAR_KEY": None,
                    "ROAMPAL_SIDECAR_MODEL": None,
                    "ROAMPAL_SIDECAR_FALLBACK": None,
                    "ROAMPAL_SIDECAR_PRIORITY": "zen",
                }
                _apply_sidecar_env_and_write(config_path, updates)
            return False

        else:
            # Skip — writes nothing (an existing model stays).
            _print_skip_message(current)
            return None

    # --- Models found: show numbered list ---
    print(f"\n{BOLD}Available scoring models:{RESET}")
    print(f"  (Sidecar only needs a small model for JSON summaries)\n")

    for i, (label, _model) in enumerate(options, 1):
        print(f"  {BOLD}[{i}]{RESET} {label}")

    go = _detect_opencode_go()
    custom_idx = len(options) + (2 if go else 1)
    free_idx = custom_idx + 1
    cancel_idx = free_idx + 1

    print(f"")
    if go:
        print(
            f"  {BOLD}[{custom_idx - 1}]{RESET} Use OpenCode Go (detected) {GREEN}— your subscription, your quota{RESET}"
        )
        print(
            f"      {YELLOW}Reliable scoring via Go's API. Each scored exchange consumes Go credits.{RESET}"
        )
    print(f"  {BOLD}[{custom_idx}]{RESET} Configure custom API endpoint")
    print(
        f"  {BOLD}[{free_idx}]{RESET} Use free Zen cloud models {YELLOW}(rate-limited, may be flaky — data sent to opencode.ai){RESET}"
    )
    print(
        f"      {YELLOW}Note: OpenCode Go subscribers also use this path — Go quota is not consumed.{RESET}"
    )
    print(
        f"  {BOLD}[{cancel_idx}]{RESET} Skip — no scoring, no summaries, no fact extraction (retrieval still works)"
    )

    try:
        choice = input(f"\nChoose [1-{cancel_idx}]: ").strip()
        choice_num = int(choice)
    except (ValueError, EOFError, KeyboardInterrupt):
        # v0.5.3: No silent fallback. Invalid or no input = cancel.
        print(
            f"\n{YELLOW}Cancelled. Run 'roampal sidecar setup' when ready.{RESET}"
        )
        return None

    if 1 <= choice_num <= len(options):
        label, chosen = options[choice_num - 1]
        print(f"\n{GREEN}Configuring: {label}{RESET}")

        env_updates = _build_sidecar_env_updates(chosen)

        if defer_write:
            return {"url": env_updates.get("ROAMPAL_SIDECAR_URL"), "model": chosen.get("name")}

        changed = _apply_sidecar_env_and_write(config_path, env_updates)
        if changed:
            print(f"{YELLOW}Restart OpenCode to activate.{RESET}")
        return changed

    elif go and choice_num == custom_idx - 1:
        # OpenCode Go model picker
        print(f"\n{BOLD}Available OpenCode Go models for scoring:{RESET}")
        for i, m in enumerate(go["models"], 1):
            print(f"  {BOLD}[{i}]{RESET} {m}")
        try:
            m_choice = input(f"\nChoose [1-{len(go['models'])}]: ").strip()
            m_idx = int(m_choice) - 1
            if not (0 <= m_idx < len(go["models"])):
                raise ValueError
        except (ValueError, EOFError, KeyboardInterrupt):
            print(f"\n{YELLOW}Cancelled.{RESET}")
            return None

        chosen_model = go["models"][m_idx]
        print(f"\n{GREEN}Configuring: OpenCode Go ({chosen_model}){RESET}")
        print(f"  {YELLOW}Note: each scored exchange consumes Go credits.{RESET}")

        updates = {
            "ROAMPAL_SIDECAR_URL": go["url"],
            "ROAMPAL_SIDECAR_KEY": go["key"],
            "ROAMPAL_SIDECAR_MODEL": chosen_model,
            "ROAMPAL_SIDECAR_FALLBACK": None,
            "ROAMPAL_SIDECAR_PRIORITY": None,
        }
        if defer_write:
            return {
                "ROAMPAL_SIDECAR_URL": go["url"],
                "ROAMPAL_SIDECAR_KEY": go["key"],
                "ROAMPAL_SIDECAR_MODEL": chosen_model,
                "ROAMPAL_SIDECAR_FALLBACK": None,
                "ROAMPAL_SIDECAR_PRIORITY": None,
            }
        _apply_sidecar_env_and_write(config_path, updates)
        return True

    elif choice_num == custom_idx:
        result = _prompt_custom_endpoint(config_path)
        if defer_write and result:
            try:
                cfg = json.loads(config_path.read_text())
                env = cfg.get("mcp", {}).get("roampal-core", {}).get(
                    "environment", {}
                )
                if env.get("ROAMPAL_SIDECAR_URL"):
                    return {
                        "url": env["ROAMPAL_SIDECAR_URL"],
                        "model": env.get("ROAMPAL_SIDECAR_MODEL", ""),
                    }
            except Exception:
                pass
        return result

    elif choice_num == free_idx:
        # v0.5.3: Explicit Zen opt-in — writes ROAMPAL_SIDECAR_PRIORITY=zen.
        print(f"\n{YELLOW}Using free Zen cloud models.{RESET}")
        print(
            f"  {BOLD}Note:{RESET} Zen is rate-limited and occasionally unreachable."
        )
        print(
            f"  Exchange data is sent to opencode.ai/zen for scoring."
        )
        print(
            f"  If Zen fails, OpenCode exchanges will {BOLD}NOT{RESET} be stored or scored"
        )
        print(
            f"  until you configure a dedicated model via {BLUE}roampal sidecar setup{RESET}."
        )
        if defer_write:
            return {"url": "zen", "model": "zen"}
        else:
            updates = {
                "ROAMPAL_SIDECAR_URL": None,
                "ROAMPAL_SIDECAR_KEY": None,
                "ROAMPAL_SIDECAR_MODEL": None,
                "ROAMPAL_SIDECAR_FALLBACK": None,
                "ROAMPAL_SIDECAR_PRIORITY": "zen",
            }
            _apply_sidecar_env_and_write(config_path, updates)
        return False

    elif choice_num == cancel_idx:
        _print_skip_message(current)
        return None

    else:
        print(f"\n{YELLOW}Invalid choice. Run 'roampal sidecar setup' to try again.{RESET}")
        return None


def _onboarding_config_path(scope: str | None) -> Path:
    """The opencode.json init's scoring setup reads and writes for `scope`."""
    if scope == "user":
        return _get_opencode_config_path()
    if scope == "project":
        project_config = _find_project_opencode_config()
        if not project_config or project_config == _get_opencode_config_path():
            # No project-local config — create one in current directory
            return Path("opencode.json")
        return project_config
    return _get_scope_config_path()


def configured_sidecar_for_scope(scope: str | None) -> str | None:
    """Task 44: description of the scoring model already configured for this
    init scope, or None (unreadable/missing config counts as none)."""
    try:
        return _describe_configured_sidecar(
            json.loads(_onboarding_config_path(scope).read_text())
        )
    except (json.JSONDecodeError, OSError, TypeError, AttributeError):
        return None


def _prompt_smart_onboarding(force: bool = False, scope: str | None = None):
    """Sidecar model selection during init. Delegates to unified picker.

    v0.5.3 Section 9: Uses scope-aware config path.
    """
    config_path = _onboarding_config_path(scope)

    if not config_path.exists():
        print(f"{RED}No opencode.json found{RESET}")
        return

    # Task 44: an existing scoring model is kept unless the user asks to
    # change it — `init --force` (the documented upgrade path) used to drop
    # every upgrading user into the full menu with no hint of the current one.
    try:
        current = _describe_configured_sidecar(json.loads(config_path.read_text()))
    except (json.JSONDecodeError, OSError):
        current = None
    if current:
        print(f"\n{BOLD}Memory scoring:{RESET} {current} is already configured.")
        try:
            answer = input("  Keep it? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("n", "no"):
            print(f"  {GREEN}Kept.{RESET} Change it any time: {BLUE}roampal sidecar setup{RESET}")
            return

    print(f"\n{BOLD}Memory scoring setup:{RESET}")
    print(f"  Roampal learns what works by scoring exchanges in the background.")
    print(f"  This requires a small AI model — it doesn't need to be smart.")

    result = _sidecar_model_picker(config_path, defer_write=False)
    if isinstance(result, dict):
        # User selected a model during init — write it now (defer was False above)
        pass  # Already written by _sidecar_model_picker when defer_write=False


def cmd_sidecar(args):
    """Configure sidecar scoring model."""
    subcommand = args.sidecar_command or "setup"

    # v0.4.9.3: Load sidecar config from opencode.json before any sidecar command
    # This ensures CLI commands use the same config as the MCP server
    if subcommand in ["test", "status"]:
        _check_sidecar_configured()

    # v0.6.0 Task 39: return the subcommand's exit code (setup returns 1 on
    # errors) so scripts and LLM-driven installs can tell whether it worked.
    if subcommand == "status":
        return _cmd_sidecar_status(args)
    elif subcommand == "setup":
        return _cmd_sidecar_setup(args)
    elif subcommand == "disable":
        return _cmd_sidecar_disable(args)
    elif subcommand == "test":
        return _cmd_sidecar_test(args)
    else:
        print(f"{RED}Unknown sidecar command: {subcommand}{RESET}")
        return 1


def _get_scope_config_path() -> Path | None:
    """Get the scope-aware config path for sidecar commands.

    v0.5.3 Section 9: Returns project-local if exists, otherwise user-global.
    """
    project_config = _find_project_opencode_config()
    if sys.platform == "win32":
        user_config_dir = Path.home() / ".config" / "opencode"
    else:
        xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        user_config_dir = Path(xdg_config) / "opencode"
    user_config_file = user_config_dir / "opencode.json"

    if project_config and project_config != user_config_file:
        return project_config
    return user_config_file


def _zen_opted_in(sc: dict | None) -> bool:
    """True when this scope recorded the explicit Zen opt-in (Task 38)."""
    if not sc or "_parse_error" in sc:
        return False
    return "zen" in [p.strip().lower() for p in (sc.get("priority") or "").split(",")]


def _cmd_sidecar_status(args):
    """Show current sidecar configuration across user-global + project-local scopes.

    v0.5.3 Section 9.3: Report both scopes, flag shadowing when project
    overrides user-global, and print the effective resolution for cwd.
    """
    scope = getattr(args, "scope", None)

    def _read_sidecar(path: Path | None) -> dict | None:
        if not path or not path.exists():
            return None
        try:
            cfg = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            return {"_parse_error": str(e)}
        env = cfg.get("mcp", {}).get("roampal-core", {}).get("environment", {})
        return {
            "url": env.get("ROAMPAL_SIDECAR_URL", ""),
            "model": env.get("ROAMPAL_SIDECAR_MODEL", ""),
            "has_key": bool(env.get("ROAMPAL_SIDECAR_KEY")),
            "priority": env.get("ROAMPAL_SIDECAR_PRIORITY", ""),
        }

    user_path = _get_opencode_config_path()
    project_path = _find_project_opencode_config()
    # Don't double-report the same file
    if project_path and project_path.resolve() == user_path.resolve():
        project_path = None

    # Scope filter lets user narrow the report
    show_user = scope in (None, "user")
    show_project = scope in (None, "project")
    if scope == "project" and not project_path:
        print(f"{YELLOW}No project-local opencode.json found in cwd ancestry.{RESET}")
        return

    user_sc = _read_sidecar(user_path) if show_user else None
    project_sc = _read_sidecar(project_path) if show_project else None

    print(f"{BOLD}Sidecar scoring configuration:{RESET}")

    if show_user:
        if user_sc is None:
            print(f"  {YELLOW}User-global:   not found{RESET}  ({user_path})")
        elif "_parse_error" in user_sc:
            print(f"  {RED}User-global:   JSON parse error — {user_sc['_parse_error']}{RESET}")
            print(f"                 {user_path}")
        elif user_sc["url"]:
            label = user_sc["model"] or "(no model)"
            print(f"  {GREEN}User-global:   {label} @ {user_sc['url']}{RESET}")
            print(f"                 {user_path}")
            if user_sc["has_key"]:
                print(f"                 Key: {'*' * 8}...")
        elif _zen_opted_in(user_sc):
            print(f"  {YELLOW}User-global:   Zen free cloud models (opted in; data sent to opencode.ai){RESET}")
            print(f"                 {user_path}")
        else:
            print(f"  {YELLOW}User-global:   no sidecar configured{RESET}")
            print(f"                 {user_path}")

    if show_project:
        if project_path is None:
            print(f"  {GREEN}Project-local: none found in cwd ancestry{RESET}")
        elif project_sc is None or "_parse_error" in (project_sc or {}):
            msg = project_sc["_parse_error"] if project_sc else "missing"
            print(f"  {RED}Project-local: JSON parse error — {msg}{RESET}")
            print(f"                 {project_path}")
        elif project_sc["url"]:
            label = project_sc["model"] or "(no model)"
            override_note = (
                f"  {YELLOW}⚠ OVERRIDES user-global in this directory{RESET}"
                if user_sc and user_sc.get("url")
                else ""
            )
            print(f"  {YELLOW}Project-local: {label} @ {project_sc['url']}{RESET}{override_note}")
            print(f"                 {project_path}")
        else:
            print(f"  {GREEN}Project-local: found, no sidecar override{RESET}")
            print(f"                 {project_path}")

    # Effective-in-cwd resolution (project wins if it sets a url)
    effective = project_sc if (project_sc and project_sc.get("url")) else user_sc
    if effective and effective.get("url"):
        print()
        print(
            f"  {BOLD}Effective in cwd:{RESET} "
            f"{effective.get('model') or '(no model)'} @ {effective['url']}"
        )
    elif _zen_opted_in(project_sc) or _zen_opted_in(user_sc):
        print()
        print(
            f"  {BOLD}Effective in cwd:{RESET} Zen free cloud models "
            f"(opted in; exchange text is sent to opencode.ai)"
        )
    else:
        print()
        print(
            f"  {YELLOW}No sidecar configured — scoring, summaries, and fact extraction are disabled.{RESET}"
        )
        print(
            f"  Retrieval from existing memories still works. Run "
            f"{BLUE}roampal sidecar setup{RESET} to enable scoring."
        )


# ============================================================================
# Non-interactive setup (v0.6.0 Task 39): lets an LLM or a script configure
# scoring without the typed menu. `--list` shows every choice (with where the
# data goes); one flag records the user's choice. Keys are never printed.
# ============================================================================

_ZEN_OPT_IN_UPDATES = {
    "ROAMPAL_SIDECAR_URL": None,
    "ROAMPAL_SIDECAR_KEY": None,
    "ROAMPAL_SIDECAR_MODEL": None,
    "ROAMPAL_SIDECAR_FALLBACK": None,
    "ROAMPAL_SIDECAR_PRIORITY": "zen",
}


def _host_of(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).hostname or url


def _flag_on(args, flag: str) -> bool:
    """argparse sets store_true flags to True/False; anything else (e.g. a
    mock args object) is treated as not requested."""
    return getattr(args, flag, None) is True


def _flag_value(args, flag: str):
    """A string option's value, or None when unset or not a real string."""
    value = getattr(args, flag, None)
    return value if isinstance(value, str) and value.strip() else None


def _sidecar_noninteractive_requested(args) -> bool:
    return any(_flag_on(args, f) for f in ("list", "zen", "auto")) or any(
        _flag_value(args, f) for f in ("model", "url", "go")
    )


def _sidecar_choice_listing(config: dict) -> dict:
    """Every scoring choice as data: what it is, whether exchange text leaves
    the machine, and the exact command that selects it. Never includes keys."""
    # Detectors print progress notes; keep stdout clean for --json.
    with contextlib.redirect_stdout(sys.stderr):
        options = _collect_sidecar_options(config)
        go = _detect_opencode_go()

    local = [m for _, m in options if m.get("source") in ("ollama", "local")]
    recommended = _recommend_model(local).get("model") if local else None

    choices = []
    for label, m in options:
        if m.get("source") in ("ollama", "local"):
            choices.append({
                "kind": "local",
                "name": m["name"],
                "label": label,
                "provider": "Ollama" if m.get("source") == "ollama" else m.get("server_label", "local"),
                "data_leaves_machine": False,
                "cost": "free",
                "command": f"roampal sidecar setup --model {m['name']}",
            })
        else:
            choices.append({
                "kind": "api",
                "name": m["name"],
                "label": label,
                "provider": m.get("display", ""),
                "data_leaves_machine": True,
                "sends_data_to": _host_of(m.get("base_url", "")),
                "cost": "paid",
                "command": f"roampal sidecar setup --model {m['name']}",
            })
    if go:
        choices.append({
            "kind": "go",
            "models": list(go["models"]),
            "data_leaves_machine": True,
            "sends_data_to": "opencode.ai",
            "cost": "uses OpenCode Go credits",
            "command": "roampal sidecar setup --go <model>",
        })
    choices.append({
        "kind": "custom",
        "data_leaves_machine": True,
        "cost": "depends on the provider",
        "command": "roampal sidecar setup --url <base_url> --model <name> [--key-env <ENV_VAR>]",
    })
    choices.append({
        "kind": "zen",
        "data_leaves_machine": True,
        "sends_data_to": "opencode.ai",
        "cost": "free, rate-limited",
        "command": "roampal sidecar setup --zen",
    })
    choices.append({
        "kind": "off",
        "data_leaves_machine": False,
        "command": "roampal sidecar disable",
    })

    env = config.get("mcp", {}).get("roampal-core", {}).get("environment", {}) or {}
    priority = [p.strip().lower() for p in (env.get("ROAMPAL_SIDECAR_PRIORITY") or "").split(",")]
    return {
        "current": {
            "model": env.get("ROAMPAL_SIDECAR_MODEL") or None,
            "url": env.get("ROAMPAL_SIDECAR_URL") or None,
            "zen_opt_in": "zen" in priority,
        },
        "recommended_local": recommended["name"] if recommended else None,
        "note": "Ask the user before choosing: the choice decides where exchange text is sent.",
        "options": choices,
    }


def _print_sidecar_choices(listing: dict) -> None:
    print(f"{BOLD}Scoring model choices{RESET} (nothing is sent anywhere until one is chosen):\n")
    for c in listing["options"]:
        if c["kind"] == "local":
            where = "stays on this machine"
        elif c["kind"] == "off":
            where = "scoring off; retrieval still works"
        else:
            where = f"data sent to {c.get('sends_data_to', 'the provider')}"
        what = c.get("label") or {
            "go": f"OpenCode Go ({', '.join(c.get('models', []))})",
            "custom": "Custom OpenAI-compatible endpoint",
            "zen": "Free Zen cloud models",
            "off": "No scoring",
        }.get(c["kind"], c["kind"])
        print(f"  {what}  [{where}]")
        print(f"      {BLUE}{c['command']}{RESET}")
    if listing.get("recommended_local"):
        print(
            f"\nRecommended local model: {listing['recommended_local']} "
            f"({BLUE}roampal sidecar setup --auto{RESET})"
        )


def _sidecar_noninteractive_updates(args, config: dict):
    """Turn one non-interactive choice into env updates.

    Returns (updates, description) on success, (None, error) otherwise.
    """
    url = _flag_value(args, "url")
    model = _flag_value(args, "model")
    modes = [f for f in ("zen", "auto") if _flag_on(args, f)]
    if _flag_value(args, "go"):
        modes.append("go")
    if url:
        modes.append("url")
    elif model:
        modes.append("model")
    if len(modes) != 1:
        return None, "Choose exactly one of --model, --url with --model, --go, --zen or --auto."
    mode = modes[0]

    if mode == "zen":
        return dict(_ZEN_OPT_IN_UPDATES), (
            "free Zen cloud models (explicit opt-in; exchange text is sent to opencode.ai)"
        )

    if mode == "url":
        if not model:
            return None, "--url needs --model <name>."
        key = None
        key_env = _flag_value(args, "key_env")
        if key_env:
            key = (os.environ.get(key_env) or "").strip()
            if not key:
                return None, f"Environment variable {key_env} is empty or not set."
        return {
            "ROAMPAL_SIDECAR_URL": url,
            "ROAMPAL_SIDECAR_MODEL": model,
            "ROAMPAL_SIDECAR_KEY": key,
            "ROAMPAL_SIDECAR_FALLBACK": None,
            "ROAMPAL_SIDECAR_PRIORITY": None,
        }, f"{model} @ {url}"

    if mode == "go":
        with contextlib.redirect_stdout(sys.stderr):
            go = _detect_opencode_go()
        if not go:
            return None, "OpenCode Go not detected (no opencode-go login in OpenCode's auth.json)."
        go_model = _flag_value(args, "go")
        if go_model not in go["models"]:
            return None, f"Unknown OpenCode Go model {go_model!r}. Available: {', '.join(go['models'])}"
        return {
            "ROAMPAL_SIDECAR_URL": go["url"],
            "ROAMPAL_SIDECAR_KEY": go["key"],
            "ROAMPAL_SIDECAR_MODEL": go_model,
            "ROAMPAL_SIDECAR_FALLBACK": None,
            "ROAMPAL_SIDECAR_PRIORITY": None,
        }, f"OpenCode Go ({go_model}; uses Go credits)"

    with contextlib.redirect_stdout(sys.stderr):
        options = _collect_sidecar_options(config)
    if mode == "auto":
        # Never picks a cloud or paid backend — local only.
        local = [m for _, m in options if m.get("source") in ("ollama", "local")]
        chosen = _recommend_model(local).get("model") if local else None
        if not chosen:
            return None, "No local model detected (Ollama, LM Studio, ...)."
    else:
        matches = [m for _, m in options if m.get("name") == model]
        if not matches:
            return None, f"No detected model named {model!r}."
        chosen = matches[0]
    label = next(lbl for lbl, m in options if m is chosen)
    return _build_sidecar_env_updates(chosen), label


def _cmd_sidecar_setup(args):
    """Configure sidecar scorer. Delegates to unified model picker.

    v0.5.3 Section 9: Scope-aware config path + atomic writes via shared helpers.
    Supports --scope {user|project|both}.
    """
    scope = getattr(args, "scope", None)

    # Resolve target paths based on scope
    user_path = _get_opencode_config_path()
    project_config = _find_project_opencode_config()

    targets: list[Path] = []
    if scope == "user":
        targets = [user_path]
    elif scope == "project":
        if project_config and project_config != user_path:
            targets = [project_config]
        else:
            # No project-local config — create one in current directory
            targets = [Path("opencode.json")]
    elif scope == "both":
        targets = [user_path]
        if project_config and project_config != user_path:
            targets.append(project_config)
        else:
            print(
                f"{YELLOW}--scope both requested but no project-local opencode.json "
                f"found in cwd ancestry. Writing user-global only.{RESET}"
            )
    else:
        # Auto-detect
        targets = [_get_scope_config_path()]

    # Run picker once on the primary target (first in list); apply env to all
    primary = targets[0]
    if not primary or not primary.exists():
        print(f"{RED}No opencode.json found at {primary}.{RESET}")
        print(f"  Run {BLUE}roampal init --opencode{RESET} to create the config.")
        return 1

    try:
        config = json.loads(primary.read_text())
    except json.JSONDecodeError as e:
        print(
            f"  {RED}[ERROR] Cannot parse {primary}:{RESET}\n"
            f"    {e}\n"
            f"    Fix the JSON or back up + delete to regenerate.\n"
        )
        return 1

    if "roampal-core" not in config.get("mcp", {}):
        print(f"{RED}roampal-core not configured yet at {primary}.{RESET}")
        print(f"  Run {BLUE}roampal init --opencode{RESET} first.")
        return 1

    # v0.6.0 Task 39: non-interactive choice (for LLM-driven installs/scripts).
    if _sidecar_noninteractive_requested(args):
        if _flag_on(args, "list"):
            listing = _sidecar_choice_listing(config)
            if _flag_on(args, "json"):
                print(json.dumps(listing, indent=2))
            else:
                _print_sidecar_choices(listing)
            return 0
        updates, description = _sidecar_noninteractive_updates(args, config)
        if updates is None:
            print(f"{RED}Error:{RESET} {description}")
            print(f"  See every choice: {BLUE}roampal sidecar setup --list{RESET}")
            return 1
        for target in targets:
            _apply_sidecar_env_and_write(target, updates)
        print(f"{GREEN}Scoring model set: {description}{RESET}")
        print(f"{YELLOW}Restart OpenCode to activate.{RESET}")
        return 0

    if len(targets) == 1:
        # Single target — picker writes directly
        _sidecar_model_picker(primary, defer_write=False)
    else:
        # Multiple targets (scope=both) — picker returns env, apply to all
        result = _sidecar_model_picker(primary, defer_write=True)
        if not isinstance(result, dict):
            return  # cancelled
        for target in targets:
            _apply_sidecar_env_and_write(target, result)
        print(
            f"{GREEN}Sidecar updated in user-global AND project-local "
            f"({len(targets)} files).{RESET}"
        )
        print(f"{YELLOW}Restart OpenCode for changes to take effect.{RESET}")


def _cmd_sidecar_disable(args):
    """Remove sidecar configuration."""
    scope = getattr(args, "scope", None)

    if scope == "user":
        config_path = _get_opencode_config_path()
    elif scope == "project":
        project_config = _find_project_opencode_config()
        if not project_config or project_config == _get_opencode_config_path():
            print(f"{YELLOW}No project-local sidecar configuration found.{RESET}")
            return
        config_path = project_config
    else:
        config_path = _get_scope_config_path()

    if not config_path or not config_path.exists():
        print(f"{YELLOW}No opencode.json found.{RESET}")
        return

    # Clear all sidecar keys via shared helper (empty updates dict)
    changed = _apply_sidecar_env_and_write(config_path, {})
    if not changed:
        print(f"{YELLOW}No sidecar configuration found.{RESET}")
        return

    print(f"{YELLOW}Sidecar configuration removed.{RESET}")
    print(
        f"  Scoring, summaries, and fact extraction are now disabled."
    )
    print(f"  Retrieval from existing memories still works.")
    print(
        f"  Run {BLUE}roampal sidecar setup{RESET} when you want to re-enable scoring."
    )
    print(f"{YELLOW}Restart OpenCode to take effect.{RESET}")


def _cmd_sidecar_test(args):
    """Test sidecar scoring with a sample exchange and validate response format."""
    from roampal.sidecar_service import get_backend_info, test_sidecar_scoring

    print(f"{BOLD}Testing sidecar scoring...{RESET}")
    backend = get_backend_info()
    print(f"Backend: {GREEN}{backend}{RESET}")

    if backend == "none available":
        print(f"\n{RED}No sidecar backend available.{RESET}")
        print(f"Run {BLUE}roampal sidecar setup{RESET} to configure one.")
        return

    print()
    print(f"Sending test exchange:")
    print(f'  User: "I\'m working on a Python project called Roampal"')
    print(f'  Assistant: "I\'ll help with your Roampal project"')
    print(f'  Follow-up: "Thanks, that\'s exactly right"')
    print()

    result = test_sidecar_scoring()

    if result.get("error"):
        print(f"{RED}Error: {result['error']}{RESET}")
        return

    fields = result.get("fields", {})
    all_pass = True

    for field_name, field_result in fields.items():
        ok = field_result["ok"]
        value = field_result["value"]
        if ok:
            print(f"  {field_name}: {GREEN}OK{RESET} ({value})")
        else:
            all_pass = False
            print(f"  {field_name}: {RED}FAIL{RESET} ({value})")

    print()
    if all_pass:
        print(f"{GREEN}Sidecar is working correctly.{RESET}")
    else:
        print(f"{RED}Sidecar response is missing required fields.{RESET}")
        print(f"The scoring model may need to be changed — try a larger model.")
        print(f"Run {BLUE}roampal sidecar setup{RESET} to reconfigure.")

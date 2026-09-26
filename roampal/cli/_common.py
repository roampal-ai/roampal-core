"""Shared CLI state + terminal utilities (v0.6.0 Task 3/5).

Holds cross-module state and shared output plumbing that the monolith kept
as module-level globals:
- `_NO_INPUT` flag behind get/set accessors (was a leaking global in cli.py)
- ANSI color constants + `_should_color` (was defined per-module; now the
  single definition shared by impl and every group module)
- `logger` — ONE shared logging object for every roampal.cli module, so
  tests patching `roampal.cli.logger.<method>` observe logs emitted from any
  group module (identity, not name, is what the patch pins).

Group helpers land here per their move task (Task 6 owns
`_stop_server_on_port`, etc.) — this file deliberately contains no command
logic so the scaffold stays behavior-identical.
"""

import json
import logging
import os
import sys
from pathlib import Path

_NO_INPUT = False  # module-private state; access via set_no_input()/no_input()


def set_no_input(value) -> None:
    global _NO_INPUT
    _NO_INPUT = bool(value)


def no_input() -> bool:
    return _NO_INPUT


# Shared logger across the whole roampal.cli package (impl + group modules).
logger = logging.getLogger("roampal.cli")


def _should_color() -> bool:
    """Check if terminal output should use ANSI colors.

    Respects NO_COLOR (https://no-color.org), TERM=dumb, and non-TTY stdout.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        if not sys.stdout.isatty():
            return False
    except Exception:
        return False
    return True


# ANSI colors (disabled when NO_COLOR set, TERM=dumb, or stdout is not a TTY)
if _should_color():
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
else:
    GREEN = YELLOW = RED = BLUE = RESET = BOLD = ""


# ============================================================================
# Config-location predicates (v0.6.0 Task 9 / G3): the shared sidecar
# config fan-in from the monolith (called from memory_cmds, scoring,
# setup, and the sidecar group). Verbatim from pre-refactor cli.py.
# ============================================================================


def _get_opencode_config_path() -> Path:
    """Get the opencode.json config path."""
    if sys.platform == "win32":
        config_dir = Path.home() / ".config" / "opencode"
    else:
        xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        config_dir = Path(xdg_config) / "opencode"
    return config_dir / "opencode.json"


def _check_sidecar_configured() -> bool:
    """Load sidecar config from opencode.json into env vars and verify it's set.

    CLI commands don't inherit the MCP process env vars, so we read them
    from the config file and inject them into the current process.
    No cascade — if the user configured a specific model, use ONLY that.
    """
    import roampal.sidecar_service as svc

    # Already configured in env (e.g. user set manually)
    if svc.CUSTOM_URL and svc.CUSTOM_MODEL:
        return True

    # Try reading from opencode.json
    config_path = _get_opencode_config_path()
    if config_path and config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            env = config.get("mcp", {}).get("roampal-core", {}).get("environment", {})
            url = env.get("ROAMPAL_SIDECAR_URL", "")
            model = env.get("ROAMPAL_SIDECAR_MODEL", "")
            key = env.get("ROAMPAL_SIDECAR_KEY", "")

            if url and model:
                # Inject into current process + reload sidecar module vars
                os.environ["ROAMPAL_SIDECAR_URL"] = url
                os.environ["ROAMPAL_SIDECAR_MODEL"] = model
                if key:
                    os.environ["ROAMPAL_SIDECAR_KEY"] = key

                # Update the module-level vars so _call_custom uses them
                svc.CUSTOM_URL = svc._validate_sidecar_url(url)
                svc.CUSTOM_MODEL = model
                svc.CUSTOM_KEY = key
                return True
        except Exception:
            pass

    # Nothing configured
    print(f"{RED}No scoring model configured.{RESET}")
    print(f"  This command needs a model to process your memories.\n")
    print(f"  Run {BLUE}roampal sidecar setup{RESET} to configure one.")
    return False


# ============================================================================
# Shared CLI state helpers (v0.6.0 Task 10): interactive/dev/port/data
# predicates from the monolith, verbatim. `_no_input_flag` is the monolith's
# import alias for this module's own no_input() accessor - kept so the
# moved bodies stay byte-identical.
# ============================================================================

_no_input_flag = no_input


def _is_interactive() -> bool:
    """Check if we can prompt the user for input.

    Returns False when --no-input is set, stdin is not a TTY (piped/CI),
    or stdin is unavailable.
    """
    if _no_input_flag():
        return False
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


# Port configuration - DEV and PROD use different ports to avoid collision
PROD_PORT = 27182
DEV_PORT = 27183


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
    if args is not None and getattr(args, "dev", False):
        return True
    return os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")


def get_port(args=None) -> int:
    """Get port based on DEV/PROD mode. Respects explicit --port override."""
    if args and hasattr(args, "port") and isinstance(getattr(args, "port", None), int):
        return args.port
    return DEV_PORT if is_dev_mode(args) else PROD_PORT


def profile_headers() -> dict:
    """X-Roampal-Profile for every CLI request that touches profile data.

    v0.6.0 smoke-test fix: the shared server resolves a headerless request
    to persisted `use` -> default (Task 17), so a CLI command run inside a
    bound folder must name its profile or it reads/writes the wrong one.
    Resolved through the one client helper (env > config-env > binding >
    use > pin > default), same as `context` and `score`."""
    from roampal.profile_manager import profile_header_value

    return {"X-Roampal-Profile": profile_header_value()}


def get_data_dir(dev: bool = False) -> Path:
    """Get the data directory path. DEV uses separate directory to avoid collision with PROD."""
    if dev:
        # DEV mode uses separate directory
        if os.name == "nt":  # Windows
            appdata = os.environ.get("APPDATA", str(Path.home()))
            return Path(appdata) / "Roampal_DEV" / "data"
        elif sys.platform == "darwin":  # macOS
            return (
                Path.home() / "Library" / "Application Support" / "Roampal_DEV" / "data"
            )
        else:  # Linux — v0.4.1: respect XDG_DATA_HOME
            xdg_data = os.environ.get(
                "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
            )
            return Path(xdg_data) / "roampal_dev" / "data"
    else:
        # PROD mode
        if os.name == "nt":  # Windows
            appdata = os.environ.get("APPDATA", str(Path.home()))
            return Path(appdata) / "Roampal" / "data"
        elif sys.platform == "darwin":  # macOS
            return Path.home() / "Library" / "Application Support" / "Roampal" / "data"
        else:  # Linux — v0.4.1: respect XDG_DATA_HOME
            xdg_data = os.environ.get(
                "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
            )
            return Path(xdg_data) / "roampal" / "data"

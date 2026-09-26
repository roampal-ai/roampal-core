"""Banner / update-check / email cluster (v0.6.0 Task 5).

Moved verbatim from the pre-refactor roampal/cli.py as part of the Task 5
setup-group split. These are display + one-time email-capture functions;
cross-deps on the impl module (get_data_dir, _is_interactive) are lazy
wrapper functions here to avoid import cycles while the split is in
progress.
"""

import json
import platform  # collect_email payload: os=platform.system() (was impl-level in monolith)
import sys
import time
import urllib.request
from pathlib import Path

from roampal.cli._common import BLUE, BOLD, GREEN, RESET, YELLOW, logger


def get_data_dir(dev: bool = False):
    from roampal.cli._common import get_data_dir as _impl_get_data_dir

    return _impl_get_data_dir(dev=dev)


def _is_interactive() -> bool:
    from roampal.cli._common import _is_interactive as _impl_is_interactive

    return _impl_is_interactive()

# Email signup webhook (Google Apps Script → Google Sheet)
# Deploy Apps Script and paste the web app URL here
SIGNUP_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxnj6GN8mNtq_xn6vRLwMLSH6VL7397Vcx-9Me8pX_BP7Yt2oob6utsZ7pCke6rcfsY/exec"

def print_banner():
    """Print Roampal banner."""
    print(f"""
{BLUE}{BOLD}+---------------------------------------------------+
|                   ROAMPAL                         |
|    Outcome-Based Memory for AI Coding Tools       |
+---------------------------------------------------+{RESET}
""")


def check_for_updates() -> tuple:
    """Check if a newer version is available on PyPI.

    Caches the result for 24 hours to avoid hitting PyPI on every command.

    Returns:
        tuple: (update_available: bool, current_version: str, latest_version: str)
    """
    try:
        from roampal import __version__
    except Exception:
        return (False, "unknown", "unknown")

    # Check cache first (stored in data dir)
    import time

    cache_file = get_data_dir() / ".update_cache"
    try:
        if cache_file.exists():
            cache_data = json.loads(cache_file.read_text())
            cache_age = time.time() - cache_data.get("timestamp", 0)
            if cache_age < 86400 and cache_data.get("current") == __version__:
                return (
                    cache_data["update_available"],
                    __version__,
                    cache_data["latest"],
                )
    except Exception:
        pass  # Corrupted cache, re-check

    # Fresh check from PyPI
    try:
        import urllib.request

        url = "https://pypi.org/pypi/roampal/json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
            latest = data.get("info", {}).get("version", __version__)

            current_parts = [int(x) for x in __version__.split(".")]
            latest_parts = [int(x) for x in latest.split(".")]
            update_available = latest_parts > current_parts

            # Cache result
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(
                    json.dumps(
                        {
                            "timestamp": time.time(),
                            "current": __version__,
                            "latest": latest,
                            "update_available": update_available,
                        }
                    )
                )
            except Exception:
                pass

            return (update_available, __version__, latest)
    except Exception:
        return (False, __version__, __version__)


def print_update_notice():
    """Print update notice if newer version available. Non-blocking."""
    update_available, current, latest = check_for_updates()
    if update_available:
        print(f"{YELLOW}[!] Update available: {latest} (you have {current}){RESET}")
        print(f"    Run: pip install --upgrade roampal && roampal init\n")


def collect_email(detected_tools: list):
    """Optionally collect user email for updates. Non-blocking, skippable.

    Marker file stores 'version:status' (e.g. '0.3.2:provided' or '0.3.2:skipped').
    - First install (no marker) → ask
    - Re-run same version → skip regardless
    - Update + previously provided email → skip (we already have it)
    - Update + previously skipped → ask again (one more chance)
    """
    if not SIGNUP_WEBHOOK_URL:
        return  # Webhook not configured yet

    from roampal import __version__

    data_dir = get_data_dir()
    marker = data_dir / ".email_asked"
    if marker.exists():
        try:
            marker_data = marker.read_text().strip()
            if ":" in marker_data:
                asked_version, status = marker_data.rsplit(":", 1)
            else:
                # Legacy marker (just version) — treat as skipped
                asked_version, status = marker_data, "skipped"
            if asked_version == __version__:
                return  # Already asked for this version
            if status == "provided":
                # They already gave email on a previous version, don't nag
                _write_email_marker(marker, __version__, "provided")
                return
        except Exception:
            pass  # Corrupted marker, ask again

    if not _is_interactive():
        _write_email_marker(marker, __version__, "skipped")
        return

    print(f"{BOLD}Stay in the loop?{RESET}")
    print(f"  Get notified about updates and new features.")
    print(f"  {YELLOW}(Optional - press Enter to skip){RESET}")

    try:
        email = input(f"\n  Email: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        _write_email_marker(marker, __version__, "skipped")
        return

    if not email or "@" not in email:
        if email:
            print(f"  {YELLOW}Doesn't look like an email, skipping.{RESET}")
        else:
            print()  # Blank line after empty Enter
        _write_email_marker(marker, __version__, "skipped")
        return

    # Send to webhook (fire-and-forget, don't block on failure)
    # Uses httpx because Google Apps Script redirects break urllib
    try:
        import httpx

        payload = {
            "email": email,
            "platform": ", ".join(detected_tools),
            "version": __version__,
            "os": platform.system(),
        }
        httpx.post(SIGNUP_WEBHOOK_URL, json=payload, follow_redirects=True, timeout=5.0)
        print(f"  {GREEN}Thanks! We'll keep you posted.{RESET}\n")
    except Exception:
        # Silently fail - don't let signup issues block init
        print(f"  {GREEN}Thanks! We'll keep you posted.{RESET}\n")

    _write_email_marker(marker, __version__, "provided")


def _write_email_marker(marker: Path, version: str, status: str = "skipped"):
    """Write version:status to marker file.

    Status is 'provided' (gave email) or 'skipped' (pressed Enter).
    """
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{version}:{status}")
    except Exception:
        pass  # Non-critical

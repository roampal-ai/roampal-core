"""
Sidecar Service — Cheap LLM for exchange summarization + scoring.

v0.3.6: Auto-detects best backend:
  1. ANTHROPIC_API_KEY set → direct Haiku API (~3s, ~$0.001/call)
  2. No API key → Zen free models (~5s, $0) — same as OpenCode sidecar
  3. Zen unavailable → Ollama / LM Studio local (if running)
  4. Nothing works → fails gracefully (no silent subprocess spawning)

Two operations:
1. summarize_and_score() — Summarize exchange + score outcome (used by stop hook)
2. summarize_only() — Summarize long memory content (used by `roampal summarize`)
"""

import json
import logging
import os
import re
import ssl
import time
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# v0.5.3 Section 12: No default sidecar config — empty strings mean disabled until user explicitly opts in via ROAMPAL_SIDECAR_URL / ROAMPAL_SIDECAR_MODEL env vars or `roampal sidecar setup`.

_raw_custom_url = os.environ.get("ROAMPAL_SIDECAR_URL", "")
CUSTOM_KEY = os.environ.get("ROAMPAL_SIDECAR_KEY", "")
CUSTOM_MODEL = os.environ.get("ROAMPAL_SIDECAR_MODEL", "")

# v0.4.9: Validate custom URL scheme to prevent SSRF (file://, internal IPs)
def _validate_sidecar_url(url: str) -> str:
    """Validate ROAMPAL_SIDECAR_URL is a safe HTTP(S) URL."""
    if not url:
        return ""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.warning(
                f"ROAMPAL_SIDECAR_URL has unsupported scheme '{parsed.scheme}' — ignoring"
            )
            return ""
        return url
    except Exception:
        logger.warning("ROAMPAL_SIDECAR_URL is malformed — ignoring")
        return ""

CUSTOM_URL = _validate_sidecar_url(_raw_custom_url)

# Zen free models — OpenAI-compatible API, no auth required
ZEN_URL = "https://opencode.ai/zen/v1"
ZEN_KEY = "public"
ZEN_MODELS = ["glm-4.7-free", "kimi-k2.5-free", "gpt-5-nano"]

# Anthropic API
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# v0.3.6: User-configurable summarization model (opt-in to expensive main model)
# When set, summarize_only() uses this model via Anthropic API instead of the sidecar chain
SUMMARIZE_MODEL = os.environ.get("ROAMPAL_SUMMARIZE_MODEL", "")

# Ollama default (override with ROAMPAL_OLLAMA_URL)
OLLAMA_URL = os.environ.get("ROAMPAL_OLLAMA_URL", "http://localhost:11434")

# v0.4.9: SSL context for external HTTPS calls (Anthropic, Zen, custom endpoints)
# Verifies certificates to prevent MITM attacks on API key-bearing requests.
_ssl_context = ssl.create_default_context()


def _is_localhost(url: str) -> bool:
    """Check if URL points to localhost (no SSL needed)."""
    from urllib.parse import urlparse

    try:
        host = urlparse(url).hostname or ""
        return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    except Exception:
        return False


# v0.4.9: Backend health tracking for robust sidecar
@dataclass
class BackendHealth:
    """Tracks health status of a backend."""

    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    failure_count: int = 0
    consecutive_failures: int = 0
    total_calls: int = 0
    avg_response_time: float = 0.0

    def record_success(self, response_time: float):
        self.last_success = datetime.now()
        self.consecutive_failures = 0
        self.total_calls += 1
        # Simple moving average
        self.avg_response_time = (
            self.avg_response_time * (self.total_calls - 1) + response_time
        ) / self.total_calls

    def record_failure(self):
        self.last_failure = datetime.now()
        self.failure_count += 1
        self.consecutive_failures += 1
        self.total_calls += 1

    def is_healthy(
        self, max_consecutive_failures: int = 3, cooldown_minutes: int = 5
    ) -> bool:
        """Check if backend should be used."""
        if self.consecutive_failures >= max_consecutive_failures:
            if self.last_failure and datetime.now() - self.last_failure < timedelta(
                minutes=cooldown_minutes
            ):
                return False
        return True

    def get_score(self) -> float:
        """Calculate health score (0.0-1.0). Higher is better."""
        if self.total_calls == 0:
            return 0.5  # Neutral score for untested backends

        success_rate = 1.0 - (self.failure_count / self.total_calls)
        recency_bonus = 0.0
        if self.last_success:
            hours_since_success = (
                datetime.now() - self.last_success
            ).total_seconds() / 3600
            recency_bonus = max(0, 0.3 * (1.0 - min(hours_since_success, 24) / 24))

        return (success_rate * 0.7) + (recency_bonus * 0.3)


# Global health tracking
_backend_health: Dict[str, BackendHealth] = {
    "custom": BackendHealth(),
    "zen": BackendHealth(),
    "ollama": BackendHealth(),
    "lmstudio": BackendHealth(),
    "haiku": BackendHealth(),
}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text that may contain markdown fences or other wrapping."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences (```json ... ```)
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object in the text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Backend 0: Custom OpenAI-compatible endpoint (user-configured)
# ---------------------------------------------------------------------------


def _call_custom(prompt: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """Call user-configured OpenAI-compatible endpoint."""
    if not CUSTOM_URL or not CUSTOM_MODEL:
        return None

    data = json.dumps(
        {
            "model": CUSTOM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are part of a memory system. Return ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 8000,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if CUSTOM_KEY:
        headers["Authorization"] = f"Bearer {CUSTOM_KEY}"

    req = urllib.request.Request(
        f"{CUSTOM_URL}/chat/completions", data=data, headers=headers, method="POST"
    )

    # Use SSL verification for external endpoints; skip for localhost
    _ctx = _ssl_context if not _is_localhost(CUSTOM_URL) else None
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ctx) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not text:
                text = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("reasoning_content", "")
                )
            if text:
                parsed = _extract_json(text)
                if parsed:
                    logger.debug(f"Custom ({CUSTOM_MODEL}) succeeded")
                    return parsed
            return None
    except Exception as e:
        logger.warning(f"Custom endpoint failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Backend 1: Direct Anthropic API (requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


def _call_haiku(prompt: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """Direct Haiku API call. Fast (~3s), requires ANTHROPIC_API_KEY."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    data = json.dumps(
        {
            "model": HAIKU_MODEL,
            "max_tokens": 8000,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            if text:
                return _extract_json(text)
            logger.warning("Haiku API returned no text content")
            return None
    except urllib.error.HTTPError as e:
        logger.warning(f"Haiku API error: HTTP {e.code}")
        return None
    except Exception as e:
        logger.warning(f"Haiku API failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Backend 2: Zen free models (no auth, OpenAI-compatible)
# ---------------------------------------------------------------------------


def _call_zen(prompt: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Call Zen free models. Free, ~5s, no auth required."""
    for model_id in ZEN_MODELS:
        try:
            data = json.dumps(
                {
                    "model": model_id,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are part of a memory system. Return ONLY valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 8000,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                f"{ZEN_URL}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {ZEN_KEY}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = (
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                # Some reasoning models put content in reasoning_content
                if not text:
                    text = (
                        result.get("choices", [{}])[0]
                        .get("message", {})
                        .get("reasoning_content", "")
                    )
                if text:
                    parsed = _extract_json(text)
                    if parsed:
                        logger.debug(f"Zen ({model_id}) succeeded")
                        return parsed
                    logger.debug(f"Zen ({model_id}) returned non-JSON: {text[:100]}")
                else:
                    logger.debug(f"Zen ({model_id}) returned empty")

        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.debug(f"Zen ({model_id}) rate limited, trying next")
                continue
            logger.debug(f"Zen ({model_id}) HTTP {e.code}, trying next")
        except Exception as e:
            logger.debug(f"Zen ({model_id}) failed: {e}, trying next")
            continue

    logger.debug("All Zen models failed")
    return None


# ---------------------------------------------------------------------------
# Backend 3: Local model server (Ollama, LM Studio, or any OpenAI-compatible)
# ---------------------------------------------------------------------------

# LM Studio default port
LMSTUDIO_URL = os.environ.get("ROAMPAL_LMSTUDIO_URL", "http://localhost:1234")


def _call_local(prompt: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """
    Call a local model server. Tries Ollama first, then LM Studio.
    Both are free, no network, no auth.
    """
    # Try Ollama first (native API)
    result = _call_ollama_native(prompt, timeout)
    if result:
        return result

    # Try LM Studio / any local OpenAI-compatible server
    result = _call_lmstudio(prompt, timeout)
    if result:
        return result

    return None


def _call_ollama_native(prompt: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """Call Ollama via native API."""
    try:
        check_req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(check_req, timeout=3) as resp:
            models_data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in models_data.get("models", [])]
            if not models:
                return None
    except Exception:
        return None

    # Pick smallest available model (prefer small/fast for summarization)
    preferred = ["qwen2.5:7b", "qwen2.5:3b", "llama3.2:3b", "gemma2:2b"]
    model = None
    for pref in preferred:
        for avail in models:
            if pref in avail:
                model = avail
                break
        if model:
            break
    if not model:
        model = models[0]

    try:
        # v0.4.9: Add system prompt for JSON response (matches Zen/custom endpoints)
        full_prompt = f"""You are part of a memory system. Return ONLY valid JSON.

{prompt}"""

        data = json.dumps(
            {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"num_predict": 500},
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = result.get("response", "")
            if text:
                parsed = _extract_json(text)
                if parsed:
                    logger.debug(f"Ollama ({model}) succeeded")
                    return parsed
            return None
    except Exception:
        return None


def _call_lmstudio(prompt: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """Call LM Studio or any OpenAI-compatible local server on port 1234."""
    # Check if LM Studio is running
    try:
        check_req = urllib.request.Request(f"{LMSTUDIO_URL}/v1/models", method="GET")
        with urllib.request.urlopen(check_req, timeout=3) as resp:
            models_data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id", "") for m in models_data.get("data", [])]
            if not models:
                return None
            model = models[0]
    except Exception:
        return None

    try:
        data = json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are part of a memory system. Return ONLY valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 8000,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{LMSTUDIO_URL}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if text:
                parsed = _extract_json(text)
                if parsed:
                    logger.debug(f"LM Studio ({model}) succeeded")
                    return parsed
            return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Unified call — tries backends in order
# ---------------------------------------------------------------------------


def _call_llm(prompt: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """
    Call the best available LLM backend with intelligent selection.

    v0.4.9: Robust backend selection with health tracking.

    IMPORTANT: If user has configured custom endpoint (ROAMPAL_SIDECAR_URL),
    we use ONLY that endpoint. No silent fallbacks to Zen/Ollama/LM Studio.

    For users without custom configuration, we use smart cascade:
    zen → ollama → lmstudio (haiku excluded unless configured)

    Returns JSON dict or None on failure.
    """
    start_time = time.time()

    # v0.4.9: RESPECT USER CONFIGURATION
    # If user explicitly configured custom endpoint, use ONLY that
    if CUSTOM_URL and CUSTOM_MODEL:
        logger.debug("Using user-configured custom endpoint (no fallbacks)")
        custom_timeout = timeout if _is_localhost(CUSTOM_URL) else min(15, timeout)
        # Retry up to 3 times — local models occasionally return empty responses
        for attempt in range(3):
            try:
                result = _call_custom(prompt, timeout=custom_timeout)
                if result:
                    _backend_health["custom"].record_success(time.time() - start_time)
                    return result
                if attempt < 2:
                    logger.debug(f"Custom endpoint returned empty (attempt {attempt + 1}/3), retrying")
            except Exception as e:
                if attempt < 2:
                    logger.debug(f"Custom endpoint error (attempt {attempt + 1}/3): {e}")
                else:
                    logger.error(f"Custom endpoint failed after 3 attempts: {e}")
        _backend_health["custom"].record_failure()
        logger.error("Custom endpoint failed after 3 attempts.")
        return None

    # No custom configured - use smart cascade for users who haven't set up anything
    # Get user-configured priority or use default
    priority_env = os.environ.get("ROAMPAL_SIDECAR_PRIORITY", "")
    if priority_env:
        # Parse comma-separated list: "zen,ollama,lmstudio"
        priority_order = [p.strip() for p in priority_env.split(",")]
    else:
        # v0.5.3: No default cascade. If the user hasn't configured a sidecar
        # via ROAMPAL_SIDECAR_URL or ROAMPAL_SIDECAR_PRIORITY, scoring is
        # disabled — no network calls, no localhost probing. Previous
        # versions silently sent exchange text to opencode.ai Zen and to
        # localhost Ollama/LM Studio without opt-in. Users opt in explicitly
        # by running `roampal sidecar setup` or setting the env vars.
        priority_order = []

    # Filter to healthy backends first
    healthy_backends = []
    unhealthy_backends = []

    for backend in priority_order:
        # Special case: Haiku requires ANTHROPIC_API_KEY
        if backend == "haiku" and not os.environ.get("ANTHROPIC_API_KEY"):
            logger.debug("Skipping haiku - no ANTHROPIC_API_KEY")
            continue

        health = _backend_health.get(backend)
        if health and health.is_healthy():
            healthy_backends.append(backend)
        else:
            unhealthy_backends.append(backend)

    # Try healthy backends in priority order
    for backend in healthy_backends:
        try:
            result = None
            backend_start = time.time()

            if backend == "custom":
                result = _call_custom(prompt, timeout=min(10, timeout))
            elif backend == "haiku":
                result = _call_haiku(prompt, timeout=min(8, timeout))
            elif backend == "zen":
                result = _call_zen(prompt, timeout=min(5, timeout))
            elif backend == "ollama":
                result = _call_ollama_native(prompt, timeout=min(12, timeout))
            elif backend == "lmstudio":
                result = _call_lmstudio(prompt, timeout=min(10, timeout))

            if result:
                response_time = time.time() - backend_start
                _backend_health[backend].record_success(response_time)
                total_time = time.time() - start_time
                logger.debug(
                    f"Sidecar succeeded with {backend} in {total_time:.1f}s (backend: {response_time:.1f}s)"
                )
                return result

        except Exception as e:
            _backend_health[backend].record_failure()
            logger.debug(f"Backend {backend} failed: {e}")
            continue

    # If no healthy backends worked, try unhealthy ones as last resort
    logger.warning(
        "No healthy backends succeeded, trying unhealthy backends as last resort"
    )
    for backend in unhealthy_backends:
        try:
            result = None
            if backend == "custom":
                # Custom = user-configured. Give full timeout, not a penalty.
                custom_t = timeout if _is_localhost(CUSTOM_URL) else min(15, timeout)
                result = _call_custom(prompt, timeout=custom_t)
            elif backend == "haiku":
                result = _call_haiku(prompt, timeout=min(10, timeout))
            elif backend == "zen":
                result = _call_zen(prompt, timeout=min(5, timeout))
            elif backend == "ollama":
                result = _call_ollama_native(prompt, timeout=min(15, timeout))
            elif backend == "lmstudio":
                result = _call_lmstudio(prompt, timeout=min(10, timeout))

            if result:
                response_time = time.time() - start_time
                _backend_health[backend].record_success(response_time)
                logger.warning(
                    f"Sidecar succeeded with previously unhealthy backend {backend} in {response_time:.1f}s"
                )
                return result

        except Exception:
            continue

    total_time = time.time() - start_time
    logger.error(f"All sidecar backends failed in {total_time:.1f}s")

    # Log health status for debugging
    health_status = []
    for backend, health in _backend_health.items():
        if health.total_calls > 0:
            health_status.append(
                f"{backend}: {health.get_score():.2f} ({health.consecutive_failures} fails)"
            )
    if health_status:
        logger.debug(f"Backend health: {', '.join(health_status)}")

    return None


def get_sidecar_status() -> Dict[str, Any]:
    """
    Get detailed sidecar status including backend health.

    v0.4.9: Returns health scores, availability, and configuration.
    """
    status = {
        "configured": {
            "custom_url": bool(CUSTOM_URL),
            "custom_model": bool(CUSTOM_MODEL),
            "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "summarize_model": bool(SUMMARIZE_MODEL),
            "priority_set": bool(os.environ.get("ROAMPAL_SIDECAR_PRIORITY")),
        },
        "health": {},
        "available_backends": [],
    }

    # Check each backend's health
    for backend, health in _backend_health.items():
        status["health"][backend] = {
            "score": health.get_score(),
            "total_calls": health.total_calls,
            "failure_count": health.failure_count,
            "consecutive_failures": health.consecutive_failures,
            "last_success": health.last_success.isoformat()
            if health.last_success
            else None,
            "last_failure": health.last_failure.isoformat()
            if health.last_failure
            else None,
            "avg_response_time": health.avg_response_time,
            "is_healthy": health.is_healthy(),
        }

        # Quick availability check for critical backends
        if backend == "custom" and CUSTOM_URL and CUSTOM_MODEL:
            status["available_backends"].append("custom")
        elif backend == "haiku" and os.environ.get("ANTHROPIC_API_KEY"):
            status["available_backends"].append("haiku")
        elif backend == "ollama":
            # Quick Ollama check
            try:
                import urllib.request

                req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    models_data = json.loads(resp.read().decode("utf-8"))
                    if models_data.get("models"):
                        status["available_backends"].append("ollama")
            except Exception:
                pass

    return status


def reset_sidecar_health(backend: Optional[str] = None):
    """
    Reset health tracking for one or all backends.

    Useful when changing configuration or debugging.
    """
    global _backend_health

    if backend:
        if backend in _backend_health:
            _backend_health[backend] = BackendHealth()
            logger.info(f"Reset health for backend: {backend}")
        else:
            logger.warning(f"Unknown backend: {backend}")
    else:
        _backend_health = {k: BackendHealth() for k in _backend_health}
        logger.info("Reset health for all backends")


def get_backend_info() -> str:
    """Return which backend will be used (for CLI display)."""
    if SUMMARIZE_MODEL and os.environ.get("ANTHROPIC_API_KEY"):
        return f"Anthropic API ({SUMMARIZE_MODEL}) via ROAMPAL_SUMMARIZE_MODEL"
    if CUSTOM_URL and CUSTOM_MODEL:
        return f"Custom ({CUSTOM_MODEL} @ {CUSTOM_URL})"
    # v0.4.5: Haiku removed from auto-detect — must be configured explicitly
    # Quick Zen check (chat endpoint — /models returns 403 outside OpenCode)
    try:
        test_data = json.dumps(
            {
                "model": ZEN_MODELS[0],
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{ZEN_URL}/chat/completions",
            data=test_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {ZEN_KEY}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5, context=_ssl_context):
            return "Zen free models"
    except Exception:
        pass
    # Quick Ollama check
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2):
            return "Ollama (local)"
    except Exception:
        pass
    # Quick LM Studio check
    try:
        req = urllib.request.Request(f"{LMSTUDIO_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id", "") for m in data.get("data", [])]
            if models:
                return f"LM Studio ({models[0]})"
    except Exception:
        pass
    return "none available"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def summarize_and_score(
    user_msg: str, assistant_msg: str, followup: str = "", timeout: int = 30
) -> Optional[Dict[str, str]]:
    """
    Summarize an exchange and score its outcome.

    Returns:
        {"summary": "...", "outcome": "worked|failed|partial|unknown"}
        or None on failure
    """
    prompt = f"""You are part of a memory system for an AI assistant. You ARE the main AI in this system — write as if you are making a note for your future self.

The user said:
"{user_msg}"

You responded:
"{assistant_msg}"

The user then followed up with:
"{followup}"

Respond with ONLY a JSON object:
{{"summary": "<~300 chars>", "outcome": "<worked|failed|partial|unknown>"}}

Summary: Write a first-person note to your future self (~300 chars). Capture the important details from the exchange — what the user needed, what you did, and any key specifics worth remembering.

Outcome: Based on the user's follow-up:
- worked: user confirmed, moved on, or was satisfied
- failed: user corrected you, got frustrated, or asked to redo
- partial: helped but incomplete or needed adjustment
- unknown: no clear signal"""

    return _call_llm(prompt, timeout=timeout)


def _call_anthropic_model(
    prompt: str, model: str, timeout: int = 30
) -> Optional[Dict[str, Any]]:
    """Call Anthropic API with a specific model. Requires ANTHROPIC_API_KEY."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    data = json.dumps(
        {
            "model": model,
            "max_tokens": 8000,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    text += block.get("text", "")
            if text:
                return _extract_json(text)
            logger.warning(f"Anthropic API ({model}) returned no text content")
            return None
    except urllib.error.HTTPError as e:
        logger.warning(f"Anthropic API ({model}) error: HTTP {e.code}")
        return None
    except Exception as e:
        logger.warning(f"Anthropic API ({model}) failed: {e}")
        return None


def summarize_only(content: str) -> Optional[str]:
    """
    Summarize a memory's content (for retroactive summarization).

    v0.3.6: Supports ROAMPAL_SUMMARIZE_MODEL env var for main-model override.
    When set + ANTHROPIC_API_KEY available, uses that model instead of the sidecar chain.

    Returns:
        Summary string (<300 chars) or None on failure
    """
    prompt = f"""You are part of a memory system for an AI assistant. You ARE the main AI — write a first-person note to your future self (~300 chars). Capture the important details from this exchange.

Exchange:
"{content}"

Respond with ONLY a JSON object:
{{"summary": "<~300 chars>"}}"""

    # v0.3.6: Main-model override — user explicitly opted in to expensive model
    if SUMMARIZE_MODEL and os.environ.get("ANTHROPIC_API_KEY"):
        logger.info(
            f"Using ROAMPAL_SUMMARIZE_MODEL={SUMMARIZE_MODEL} for summarization"
        )
        result = _call_anthropic_model(prompt, SUMMARIZE_MODEL)
        if isinstance(result, dict):
            return result.get("summary")
        logger.warning(
            f"ROAMPAL_SUMMARIZE_MODEL ({SUMMARIZE_MODEL}) failed, falling back to sidecar chain"
        )

    # v0.5.3: Type guard — _call_llm returns whatever _extract_json parses,
    # which can be a list/string from a 3B-class model that ignores the
    # {"summary": "..."} schema. Without the guard, .get() crashes.
    result = _call_llm(prompt, timeout=30)
    return result.get("summary") if isinstance(result, dict) else None


def extract_tags(text: str, timeout: int = 20) -> Optional[List[str]]:
    """
    Extract key noun tags from text using sidecar LLM.

    v0.4.5: Used for ongoing tag extraction on new memories.
    Prompt proven at 67.8% accuracy in LoCoMo benchmark (EntityRouted strategy).

    Returns:
        List of lowercase tag strings (max 8), or None on failure
    """
    prompt = (
        "Extract the key TOPIC nouns from this text — people's names, places, objects, "
        "and specific things the text is actually about. "
        'Return ONLY a JSON object like: {"tags": ["calvin", "muscle car", "boston"]}\n'
        "Rules:\n"
        "- Use actual names, not pronouns (skip 'he', 'she', 'they', 'user', 'assistant')\n"
        "- Keep each tag as a short noun phrase (1-3 words)\n"
        "- Include both proper nouns and important common nouns\n"
        "- Skip meta-words about the conversation itself: source, answer, details, accuracy, "
        "response, question, topic, context, information, correction, update, memory\n"
        "- Skip generic verbs/actions: said, told, mentioned, discussed, talked, asked\n"
        "- Focus on WHO and WHAT the text is about, not how it was communicated\n"
        f'- Maximum 8 tags\n\nText: "{text[:2000]}"'
    )

    result = _call_llm(prompt)
    if not result:
        return None

    tags = result if isinstance(result, list) else result.get("tags")
    if not isinstance(tags, list):
        return None

    skip = {
        "he",
        "she",
        "they",
        "it",
        "user",
        "assistant",
        "the user",
        "the assistant",
        "i",
        "you",
        "we",
        "them",
        "his",
        "her",
    }
    cleaned = [
        t.lower().strip()
        for t in tags
        if isinstance(t, str) and t.lower().strip() not in skip and len(t.strip()) >= 2
    ]
    return cleaned[:8] if cleaned else None


def extract_facts(text: str, timeout: int = 30) -> Optional[List[str]]:
    """
    Extract atomic facts from text using sidecar LLM.

    v0.4.5: Used for fact extraction alongside summarization.
    Prompt matches benchmark (runner.py:175-196).

    Returns:
        List of fact strings (max 150 chars each), or None on failure
    """
    prompt = (
        "Extract key facts worth remembering from this text. Rules:\n"
        "- Include WHO or WHAT each fact is about — names, projects, topics\n"
        "- Combine related details into ONE rich fact rather than many fragments\n"
        "- Include specifics: dates, versions, preferences, decisions, reasons\n"
        "- ONE fact per line, max 150 characters\n"
        "- Skip vague feelings, pleasantries, or generic observations\n"
        "\n"
        'GOOD: "The auth service uses JWT with 24h expiry, needs refresh token rotation added"\n'
        'GOOD: "User prefers TypeScript over JavaScript and uses Zod for validation"\n'
        'BAD: "They discussed something" (no specifics)\n'
        'BAD: "It was helpful" (no content)\n'
        "\n"
        'Return ONLY a JSON object: {"facts": ["fact 1", "fact 2"]}\n'
        'If no useful facts, return: {"facts": []}\n'
        f'\nText: "{text[:8000]}"'
    )

    result = _call_llm(prompt, timeout=timeout)
    if not result:
        return None

    facts = result if isinstance(result, list) else result.get("facts")
    if not isinstance(facts, list):
        return None

    cleaned = [
        f.strip().lstrip("•-*0123456789. ")
        for f in facts
        if isinstance(f, str) and len(f.strip()) > 10
    ]
    return cleaned[:10] if cleaned else None


def test_sidecar_scoring() -> Dict[str, Any]:
    """
    Test sidecar with a full scoring prompt and validate response format.

    v0.4.5: Used by `roampal sidecar test` to verify the sidecar
    returns all required fields (summary, outcome, noun_tags, facts).

    Returns:
        Dict with test results: {passed: bool, fields: {field: {ok, value}}, raw: str}
    """
    prompt = """The user said:
"I'm working on a Python project called Roampal that adds persistent memory to AI coding tools"

You responded:
"I'll help with your Roampal project. It sounds like a memory system for AI assistants."

The user then followed up with:
"Thanks, that's exactly right"

Respond with ONLY a JSON object:
{ "summary": "<~300 chars>", "outcome": "<worked|failed|partial|unknown>", "noun_tags": ["<lowercase nouns>"], "facts": ["<atomic fact 1>", "<atomic fact 2>"] }

SUMMARY (under 300 chars): Capture what happened AND what changed.
OUTCOME: Based on the user's follow-up (worked/failed/partial/unknown).
NOUN_TAGS: Key topic nouns — lowercase, 1-3 words each, max 8.
FACTS: Key facts worth remembering — WHO/WHAT, specifics, max 150 chars each."""

    result = _call_llm(prompt)

    fields = {}
    if result is None:
        return {"passed": False, "fields": {}, "error": "All backends failed"}

    # v0.5.3 Section 3: 3B-class models sometimes return a bare list
    # instead of the schema-wrapped object. The full scoring schema
    # (summary/outcome/noun_tags/facts) can't be validated from a list,
    # so report a clear error rather than crashing on result.get(...).
    if isinstance(result, list):
        return {
            "passed": False,
            "fields": {},
            "error": "Sidecar returned a bare JSON array; expected an object with summary/outcome/noun_tags/facts. Try a larger model.",
        }

    # Validate summary
    summary = result.get("summary", "")
    fields["summary"] = {
        "ok": isinstance(summary, str) and len(summary) > 0,
        "value": summary[:100] if summary else "(missing)",
    }

    # Validate outcome
    outcome = result.get("outcome", "")
    fields["outcome"] = {
        "ok": outcome in ("worked", "failed", "partial", "unknown"),
        "value": outcome or "(missing)",
    }

    # Validate noun_tags
    noun_tags = result.get("noun_tags", [])
    fields["noun_tags"] = {
        "ok": isinstance(noun_tags, list) and len(noun_tags) > 0,
        "value": noun_tags if isinstance(noun_tags, list) else "(missing)",
    }

    # Validate facts (bare-list case handled by early-return above)
    facts = result.get("facts", [])
    fields["facts"] = {
        "ok": isinstance(facts, list) and len(facts) > 0,
        "value": facts if isinstance(facts, list) else "(missing)",
    }

    passed = all(f["ok"] for f in fields.values())
    return {"passed": passed, "fields": fields}

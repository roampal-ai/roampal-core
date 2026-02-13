"""
Sidecar Service — Cheap LLM for exchange summarization + scoring.

v0.3.6: Auto-detects best backend:
  1. ANTHROPIC_API_KEY set → direct Haiku API (~3s, ~$0.001/call)
  2. No API key → Zen free models (~5s, $0) — same as OpenCode sidecar
  3. Zen unavailable → Ollama local (if running)
  4. Nothing works → claude -p fallback (slow, ~30-60s)

Two operations:
1. summarize_and_score() — Summarize exchange + score outcome (used by stop hook)
2. summarize_only() — Summarize long memory content (used by `roampal summarize`)
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# User-configurable sidecar (env vars override defaults)
# Set ROAMPAL_SIDECAR_URL + ROAMPAL_SIDECAR_KEY + ROAMPAL_SIDECAR_MODEL
# to use any OpenAI-compatible endpoint (Groq, Together, OpenRouter, etc.)
CUSTOM_URL = os.environ.get("ROAMPAL_SIDECAR_URL", "")
CUSTOM_KEY = os.environ.get("ROAMPAL_SIDECAR_KEY", "")
CUSTOM_MODEL = os.environ.get("ROAMPAL_SIDECAR_MODEL", "")

# Zen free models — OpenAI-compatible API, no auth required
ZEN_URL = "https://opencode.ai/zen/v1"
ZEN_KEY = "public"
ZEN_MODELS = ["minimax-m2.5-free", "kimi-k2.5-free", "glm-4.7-free"]

# Anthropic API
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# v0.3.6: User-configurable summarization model (opt-in to expensive main model)
# When set, summarize_only() uses this model via Anthropic API instead of the sidecar chain
SUMMARIZE_MODEL = os.environ.get("ROAMPAL_SUMMARIZE_MODEL", "")

# Ollama default (override with ROAMPAL_OLLAMA_URL)
OLLAMA_URL = os.environ.get("ROAMPAL_OLLAMA_URL", "http://localhost:11434")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text that may contain markdown fences or other wrapping."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences (```json ... ```)
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object in the text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
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

    data = json.dumps({
        "model": CUSTOM_MODEL,
        "messages": [
            {"role": "system", "content": "You are part of a memory system. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if CUSTOM_KEY:
        headers["Authorization"] = f"Bearer {CUSTOM_KEY}"

    req = urllib.request.Request(
        f"{CUSTOM_URL}/chat/completions",
        data=data,
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
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

    data = json.dumps({
        "model": HAIKU_MODEL,
        "max_tokens": 800,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
            data = json.dumps({
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are part of a memory system. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 800
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{ZEN_URL}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {ZEN_KEY}"
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                text = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
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
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 500}
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
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
        data = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are part of a memory system. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 800
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{LMSTUDIO_URL}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            text = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if text:
                parsed = _extract_json(text)
                if parsed:
                    logger.debug(f"LM Studio ({model}) succeeded")
                    return parsed
            return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backend 4: claude -p fallback (slow, but works with any Claude Code auth)
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
    """
    Call Haiku via claude CLI. Slow (~30-60s) but works with any auth.
    Last resort fallback.
    """
    claude_path = shutil.which("claude")
    if not claude_path:
        logger.debug("claude CLI not found in PATH")
        return None

    cmd = [
        claude_path, "-p",
        "--model", "haiku",
        "--output-format", "json",
        "--max-turns", "1",
        "--max-budget-usd", "0.05",
    ]

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=(sys.platform == "win32"),
        )

        if result.returncode != 0:
            logger.warning(f"Claude CLI error (exit {result.returncode}): {result.stderr[:200]}")
            return None

        output = result.stdout.strip()
        if not output:
            logger.warning("Claude CLI returned empty output")
            return None

        envelope = _extract_json(output)
        if envelope and isinstance(envelope, dict) and "result" in envelope:
            inner = envelope["result"]
            if isinstance(inner, str):
                parsed = _extract_json(inner)
                if parsed:
                    return parsed
            elif isinstance(inner, dict):
                return inner

        if envelope:
            return envelope

        logger.warning(f"Could not parse Claude CLI output: {output[:200]}")
        return None

    except subprocess.TimeoutExpired:
        logger.warning(f"Claude CLI timed out after {timeout}s")
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Claude CLI failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Unified call — tries backends in order
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Call the best available LLM backend.

    Fallback chain:
    0. ROAMPAL_SIDECAR_URL configured → custom endpoint (user's choice)
    1. ANTHROPIC_API_KEY → direct Haiku API (~3s, ~$0.001)
    2. Zen free models (~5s, $0)
    3. Ollama / LM Studio local (~5-14s, $0)
    4. claude -p CLI (~30-60s, uses existing auth)
    """
    # 0. Custom endpoint (user-configured — takes priority over everything)
    result = _call_custom(prompt)
    if result:
        return result

    # 1. Direct API (fastest, cheapest per-call)
    result = _call_haiku(prompt)
    if result:
        return result

    # 2. Zen free models (free, no auth)
    result = _call_zen(prompt)
    if result:
        return result

    # 3. Ollama / LM Studio local (free, no network)
    result = _call_local(prompt)
    if result:
        return result

    # 4. claude -p (last resort)
    result = _call_claude(prompt)
    if result:
        return result

    logger.error("All sidecar backends failed")
    return None


def get_backend_info() -> str:
    """Return which backend will be used (for CLI display)."""
    if SUMMARIZE_MODEL and os.environ.get("ANTHROPIC_API_KEY"):
        return f"Anthropic API ({SUMMARIZE_MODEL}) via ROAMPAL_SUMMARIZE_MODEL"
    if CUSTOM_URL and CUSTOM_MODEL:
        return f"Custom ({CUSTOM_MODEL} @ {CUSTOM_URL})"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "Haiku API (direct)"
    # Quick Zen check (chat endpoint — /models returns 403 outside OpenCode)
    try:
        test_data = json.dumps({
            "model": ZEN_MODELS[0],
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{ZEN_URL}/chat/completions",
            data=test_data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {ZEN_KEY}"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5):
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
    if shutil.which("claude"):
        return "claude -p (slow)"
    return "none available"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_and_score(
    user_msg: str,
    assistant_msg: str,
    followup: str = "",
    timeout: int = 30
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

    return _call_llm(prompt)


def _call_anthropic_model(prompt: str, model: str, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """Call Anthropic API with a specific model. Requires ANTHROPIC_API_KEY."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    data = json.dumps({
        "model": model,
        "max_tokens": 800,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
        logger.info(f"Using ROAMPAL_SUMMARIZE_MODEL={SUMMARIZE_MODEL} for summarization")
        result = _call_anthropic_model(prompt, SUMMARIZE_MODEL)
        if result:
            return result.get("summary")
        logger.warning(f"ROAMPAL_SUMMARIZE_MODEL ({SUMMARIZE_MODEL}) failed, falling back to sidecar chain")

    result = _call_llm(prompt)
    return result.get("summary") if result else None

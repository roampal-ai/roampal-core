"""
Roampal FastAPI Server

Provides:
- /api/hooks/get-context: Called by UserPromptSubmit hook (injects scoring prompt)
- /api/hooks/stop: Called by Stop hook (stores exchange, enforces record_response)
- /api/health: Health check endpoint
- MCP server (optional, for power users)

Hook Enforcement Flow:
1. UserPromptSubmit hook calls /api/hooks/get-context
   - If previous exchange unscored, injects scoring prompt
   - Adds relevant memories to context
2. LLM MUST call record_response(outcome) to score previous exchange
3. Stop hook calls /api/hooks/stop
   - Stores current exchange with doc_id
   - Can block (exit 2) if record_response wasn't called
"""

import logging
import json
import asyncio
import os
import sys
import time
from pathlib import Path

# Handle pythonw.exe (GUI subsystem) where stdout/stderr are None.
# This happens when the plugin spawns the server via pythonw to avoid console windows.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# Fix Windows encoding issues with unicode characters (emojis, box drawing, etc.)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass  # Ignore if already reconfigured or in test environment
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import memory system and session manager
from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("roampal")
from roampal.backend.modules.memory import UnifiedMemorySystem
from roampal.backend.modules.memory.scoring_service import wilson_score_lower
from roampal.backend.modules.memory.unified_memory_system import ActionOutcome

# v0.4.5: ContentGraph removed (KG deleted)
from roampal.hooks import SessionManager

logger = logging.getLogger(__name__)

# Per-profile registry (replaces singleton — v0.5.4 profile binding fix)
_memory_by_profile: Dict[str, UnifiedMemorySystem] = {}
_session_manager_by_profile: Dict[str, SessionManager] = {}
_init_lock: asyncio.Lock = asyncio.Lock()

# Shared embedding service — one ONNX model for all profiles (avoids ~420MB x N)
_shared_embed_service: Any = None

# Search result cache for outcome scoring (session_id -> doc_ids)
_search_cache: Dict[str, Dict[str, Any]] = {}

# v0.3.2: Injection map - tracks which doc_ids were injected to which conversation
# This enables robust multi-session scoring by matching doc_ids to their source conversation
# instead of relying on "most recent unscored" heuristics
# Format: {doc_id: {"conversation_id": str, "injected_at": str, "exchange_doc_id": str}}
_injection_map: Dict[str, Dict[str, Any]] = {}

# Port constants for dev/prod isolation
PROD_PORT = 27182
DEV_PORT = 27183

# Cache for update check (re-checks after TTL expires)
_update_check_cache: Optional[tuple] = None
_update_check_time: float = 0.0

# Cache TTL: 30 minutes (entries older than this are evicted)
_CACHE_TTL_SECONDS = 30 * 60


def _evict_stale_entries():
    """Evict stale entries from _search_cache and _injection_map (30-minute TTL)."""
    now = datetime.now()
    stale_search_keys = []
    for key, entry in _search_cache.items():
        ts = entry.get("timestamp", "")
        if ts:
            try:
                entry_time = datetime.fromisoformat(ts)
                if (now - entry_time).total_seconds() > _CACHE_TTL_SECONDS:
                    stale_search_keys.append(key)
            except (ValueError, TypeError):
                stale_search_keys.append(key)
    for key in stale_search_keys:
        del _search_cache[key]

    stale_injection_keys = []
    for key, entry in _injection_map.items():
        ts = entry.get("injected_at", "")
        if ts:
            try:
                entry_time = datetime.fromisoformat(ts)
                if (now - entry_time).total_seconds() > _CACHE_TTL_SECONDS:
                    stale_injection_keys.append(key)
            except (ValueError, TypeError):
                stale_injection_keys.append(key)
    for key in stale_injection_keys:
        del _injection_map[key]

    evicted = len(stale_search_keys) + len(stale_injection_keys)
    if evicted:
        logger.info(
            f"Cache eviction: {len(stale_search_keys)} search + {len(stale_injection_keys)} injection entries"
        )


# Cold start tag priorities - one fact per category (v0.2.7)
TAG_PRIORITIES = [
    "identity",
    "preference",
    "goal",
    "project",
    "system_mastery",
    "agent_growth",
]


def _first_sentence(text: str, max_chars: int = 300) -> str:
    """
    Extract first sentence from text, capped at max_chars.

    v0.2.7: Used for cold start truncation - prevents massive facts
    from overwhelming context. Full facts still available via search.
    Bumped from 150 to 300 chars for better summary context.
    """
    if not text:
        return ""
    # Find first sentence ending
    for end_char in [". ", ".\n", "!", "?"]:
        idx = text.find(end_char)
        if idx > 0:
            first = text[: idx + 1].strip()
            if len(first) <= max_chars:
                return first
            break
    # No sentence ending found or sentence too long - truncate
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rsplit(" ", 1)[0] + "..."


def _get_installed_version() -> str:
    """Get the installed version from pip package metadata.

    Uses importlib.metadata which reads pip metadata directly — always correct
    regardless of namespace conflicts (e.g., roampal-cli co-installed).
    Falls back to reading __init__.py from disk for long-running server upgrades.
    """
    try:
        from importlib.metadata import version as _pkg_ver

        return _pkg_ver("roampal")
    except Exception:
        pass
    # Fallback: read __init__.py from disk (handles mid-session pip upgrades)
    try:
        init_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "__init__.py"
        )
        with open(init_path, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=", 1)[1].strip().strip("\"'")
    except Exception:
        pass
    return __version__


def _check_for_updates() -> tuple:
    """Check if a newer version is available on PyPI.

    Re-checks after _CACHE_TTL_SECONDS expires so that a pip upgrade
    is picked up without requiring a server restart.

    Returns:
        tuple: (update_available: bool, current_version: str, latest_version: str)
    """
    global _update_check_cache, _update_check_time

    # Return cached result if still fresh
    if (
        _update_check_cache is not None
        and (time.time() - _update_check_time) < _CACHE_TTL_SECONDS
    ):
        return _update_check_cache

    current = _get_installed_version()

    try:
        import urllib.request

        url = "https://pypi.org/pypi/roampal/json"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        with urllib.request.urlopen(req, timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
            latest = data.get("info", {}).get("version", current)

            current_parts = [int(x) for x in current.split(".")]
            latest_parts = [int(x) for x in latest.split(".")]
            update_available = latest_parts > current_parts

            _update_check_cache = (update_available, current, latest)
            _update_check_time = time.time()
            return _update_check_cache
    except Exception:
        _update_check_cache = (False, current, current)
        _update_check_time = time.time()
        return _update_check_cache


def _get_update_injection() -> Optional[str]:
    """Get update notice for hook injection if newer version available.

    Returns directive for Claude to tell the user about the update.
    """
    update_available, current, latest = _check_for_updates()
    if update_available:
        return f"""<roampal-update>
A newer version of roampal is available: {latest} (user has {current}).
IMPORTANT: Briefly mention this to the user. Say something like:
"Quick note: roampal {latest} is available. Run `pip install --upgrade roampal && roampal init --force` to update."
Only mention once per conversation.
</roampal-update>
"""
    return None


async def _build_cold_start_profile(mem: UnifiedMemorySystem) -> Optional[str]:
    """
    Build the cold start user profile injection.

    v0.2.7: Lean but rich - 10 facts max with balanced coverage:
    1. Always-inject memories (identity core)
    2. One fact per tag category (identity, preference, goal, project, system_mastery, agent_growth)
    3. (Future) Content KG entities - read path exists but entity extraction not yet wired up

    Args:
        mem: UnifiedMemorySystem for the active profile.

    Returns:
        Formatted user profile string, or None if no facts exist.
    """
    if not mem:
        return None

    try:
        # Get all facts from memory_bank, sorted by quality (importance, confidence)
        all_memory_bank = mem._memory_bank_service.list_all(include_archived=False)

        # v0.3.7: Proven facts (3+ uses) sort by Wilson; cold facts by quality
        def _cold_start_sort_key(f):
            meta = f.get("metadata", {})
            uses = int(meta.get("uses", 0))
            if uses >= 3:
                success_count = float(meta.get("success_count", 0.0))
                return 2.0 + wilson_score_lower(success_count, uses)
            return float(meta.get("importance", 0.5)) * float(
                meta.get("confidence", 0.5)
            )

        sorted_facts = sorted(all_memory_bank, key=_cold_start_sort_key, reverse=True)

        # Pick HIGHEST QUALITY fact for EACH tag category (one per tag)
        all_facts = []
        seen_tags = set()
        for fact in sorted_facts:
            tags_raw = fact.get("metadata", {}).get("tags", [])
            if isinstance(tags_raw, str):
                try:
                    tags = json.loads(tags_raw) if tags_raw else []
                except (json.JSONDecodeError, ValueError, TypeError):
                    tags = []
            else:
                tags = tags_raw or []

            # Find which priority tag this fact matches (if any)
            for tag in TAG_PRIORITIES:
                if tag in tags and tag not in seen_tags:
                    all_facts.append(fact)
                    seen_tags.add(tag)
                    break  # One fact per tag

            # Stop once we have one fact per tag category
            if len(seen_tags) == len(TAG_PRIORITIES):
                break

        # Check if user has NO identity at all
        identity_content = []
        for fact in all_facts:
            content = (
                fact.get("text")
                or fact.get("content")
                or fact.get("metadata", {}).get("content", "")
            )
            tags_raw = fact.get("metadata", {}).get("tags", [])
            if isinstance(tags_raw, str):
                try:
                    tags = json.loads(tags_raw) if tags_raw else []
                except (json.JSONDecodeError, ValueError, TypeError):
                    tags = []
            else:
                tags = tags_raw or []
            if "identity" in tags:
                identity_content.append(content)

        if not identity_content:
            # Check if they have history (existing user) or not (truly new)
            has_history = await mem.search(
                query="", collections=["history", "patterns"], limit=1
            )

            if has_history:
                return """<roampal-identity-missing>
You've been working with this user but don't have their identity stored yet.

When natural, ask their name to personalize future sessions. Store with:
  add_to_memory_bank(content="User's name is [NAME]", tags=["identity"])
</roampal-identity-missing>
"""
            else:
                return """<roampal-new-user>
NEW USER: You don't have any stored information about this user yet.

Consider naturally asking for their name and what they're working on. Store with:
  add_to_memory_bank(content="User's name is [NAME]", tags=["identity"])
  add_to_memory_bank(content="Working on [PROJECT]", tags=["project"])

Keep it conversational - don't interrogate. A simple "I don't think we've met - what's your name?" works.
</roampal-new-user>
"""

        # Build narrative profile - one line per category (compact ~200 chars)
        tag_labels = {
            "identity": "Identity",
            "preference": "Preference",
            "goal": "Goal",
            "project": "Project",
            "system_mastery": "System Mastery",
            "agent_growth": "Agent Growth",
        }

        # Group facts by their primary tag, keeping only FIRST fact per category
        category_facts = {}
        for fact in all_facts:
            content = (
                fact.get("text")
                or fact.get("content")
                or fact.get("metadata", {}).get("content", "")
            )
            tags_raw = fact.get("metadata", {}).get("tags", [])
            if isinstance(tags_raw, str):
                try:
                    tags = json.loads(tags_raw) if tags_raw else []
                except (json.JSONDecodeError, ValueError, TypeError):
                    tags = []
            else:
                tags = tags_raw or []

            for tag in TAG_PRIORITIES:
                if tag in tags and tag not in category_facts:
                    category_facts[tag] = content
                    break

        # Build compact narrative
        profile_parts = ["<roampal-user-profile>"]
        for tag in TAG_PRIORITIES:
            if tag in category_facts:
                profile_parts.append(
                    f"{tag_labels[tag]}: {_first_sentence(category_facts[tag])}"
                )
        profile_parts.append("</roampal-user-profile>")

        logger.info(
            f"Cold start: {len(all_facts)} facts, {len(category_facts)} categories"
        )
        return "\n".join(profile_parts)

    except Exception as e:
        logger.error(f"Error building cold start profile: {e}")
        return None


# ==================== Request/Response Models ====================


class GetContextRequest(BaseModel):
    """Request for hook context injection."""

    query: str
    conversation_id: Optional[str] = None
    recent_messages: Optional[List[Dict[str, Any]]] = None


class GetContextResponse(BaseModel):
    """Response with context to inject."""

    formatted_injection: str
    user_facts: List[Dict[str, Any]]
    relevant_memories: List[Dict[str, Any]]
    context_summary: str
    scoring_required: bool = False  # True if previous exchange needs scoring
    # v0.3.2: Split fields for OpenCode plugin — scoring in user message, context in system prompt
    scoring_prompt: str = ""  # Just the scoring block (for prepending to user message)
    scoring_prompt_simple: str = ""  # Simplified scoring prompt for non-Claude models (no XML tags, plain language)
    context_only: str = (
        ""  # Just the memory context without scoring (for system prompt)
    )
    # v0.3.2: Raw scoring data for independent LLM scoring call (OpenCode plugin)
    scoring_exchange: Optional[Dict[str, str]] = (
        None  # {"user": "...", "assistant": "..."} previous exchange
    )
    scoring_memories: Optional[List[Dict[str, str]]] = (
        None  # [{"id": "doc_id", "content": "full memory content"}, ...]
    )


class StopHookRequest(BaseModel):
    """Request from Stop hook after LLM responds."""

    conversation_id: str
    user_message: str = ""  # v0.3.6: Optional — Claude Code sends empty (state-only)
    assistant_response: str = (
        ""  # v0.3.6: Optional — Claude Code sends empty (state-only)
    )
    transcript: Optional[str] = (
        None  # Full transcript to check for record_response call
    )
    metadata: Optional[Dict[str, Any]] = (
        None  # v0.3.6: Extra metadata (e.g., memory_type: "exchange_summary")
    )
    noun_tags: Optional[List[str]] = (
        None  # v0.4.5: content nouns for TagCascade retrieval
    )
    lifecycle_only: bool = (
        False  # v0.3.6: Track in session JSONL but skip ChromaDB storage
    )


class StopHookResponse(BaseModel):
    """Response to Stop hook."""

    stored: bool
    doc_id: str
    scoring_complete: bool  # Did the LLM call record_response?
    should_block: bool  # Should hook block with exit code 2?
    block_message: Optional[str] = None


class SearchRequest(BaseModel):
    """Request for searching memory."""

    query: Optional[str] = Field("", max_length=2000)
    days_back: Optional[int] = Field(None, ge=1, le=365)
    id: Optional[str] = Field(None, max_length=200)
    conversation_id: Optional[str] = Field(None, max_length=200)
    collections: Optional[List[str]] = None
    limit: int = Field(10, ge=1, le=500)
    metadata_filters: Optional[Dict[str, Any]] = None
    sort_by: Optional[str] = Field(None, pattern="^(relevance|recency|score)$")


class MemoryBankAddRequest(BaseModel):
    """Request to add to memory bank."""

    content: str
    tags: Optional[List[str]] = None
    noun_tags: Optional[List[str]] = (
        None  # v0.4.5: content nouns for TagCascade retrieval
    )
    importance: float = 0.7
    confidence: float = 0.7
    always_inject: bool = False


class MemoryBankUpdateRequest(BaseModel):
    """Request to update memory bank."""

    old_content: str
    new_content: str


class RecordOutcomeRequest(BaseModel):
    """Request to record outcome for scoring."""

    conversation_id: str
    outcome: str  # worked, failed, partial, unknown
    related: Optional[List[str]] = None  # DEPRECATED: use memory_scores instead
    memory_scores: Optional[Dict[str, str]] = (
        None  # v0.2.8: Per-memory scoring (doc_id -> outcome)
    )
    exchange_summary: Optional[str] = None  # v0.3.6: ~300 char summary from main LLM
    exchange_outcome: Optional[str] = None  # v0.3.6: backward compat alias for outcome
    noun_tags: Optional[List[str]] = (
        None  # v0.4.5: content nouns for TagCascade retrieval
    )
    facts: Optional[List[str]] = None  # v0.4.5: atomic facts from exchange


class RecordResponseRequest(BaseModel):
    """Request to record a key takeaway (MCP tool proxy)."""

    key_takeaway: str
    conversation_id: str
    noun_tags: Optional[List[str]] = (
        None  # v0.4.5: content nouns for TagCascade retrieval
    )


class UpdateContentRequest(BaseModel):
    """Request to update a memory's content (v0.3.6 summarization)."""

    doc_id: str
    collection: str
    new_content: str
    noun_tags: Optional[List[str]] = None  # v0.4.5: content nouns for TagCascade


# ==================== Lifecycle ====================


def _hydrate_sidecar_from_opencode_config():
    """v0.5.4: The FastAPI server is launched independently of the OpenCode
    plugin process, so it doesn't inherit ROAMPAL_SIDECAR_URL/MODEL/KEY env
    vars set in opencode.json's mcp.roampal-core.environment block. Without
    these, sidecar_service.CUSTOM_URL is empty and server-side helpers like
    extract_tags() (used by /api/hooks/stop and /record-outcome to populate
    noun_tags when the plugin doesn't send them) silently fail — every
    OpenCode-stored memory ended up without noun_tags despite v0.5.3 §11
    claiming server-side fallback.

    Mirror cli.py:_check_sidecar_configured: read the env block from the
    user-global opencode.json, write missing vars into os.environ, and
    re-bind sidecar_service module globals (since they were already set at
    import time).
    """
    if os.environ.get("ROAMPAL_SIDECAR_URL") and os.environ.get("ROAMPAL_SIDECAR_MODEL"):
        return  # already configured via real env, nothing to hydrate

    if sys.platform == "win32":
        config_dir = Path.home() / ".config" / "opencode"
    else:
        xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        config_dir = Path(xdg_config) / "opencode"
    config_path = config_dir / "opencode.json"
    if not config_path.exists():
        return

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Could not read sidecar config from {config_path}: {e}")
        return

    env_block = config.get("mcp", {}).get("roampal-core", {}).get("environment", {})
    url = env_block.get("ROAMPAL_SIDECAR_URL", "")
    model = env_block.get("ROAMPAL_SIDECAR_MODEL", "")
    key = env_block.get("ROAMPAL_SIDECAR_KEY", "")
    if not (url and model):
        return

    os.environ.setdefault("ROAMPAL_SIDECAR_URL", url)
    os.environ.setdefault("ROAMPAL_SIDECAR_MODEL", model)
    if key:
        os.environ.setdefault("ROAMPAL_SIDECAR_KEY", key)

    # Re-bind module globals — sidecar_service read these at import time.
    import roampal.sidecar_service as svc
    svc.CUSTOM_URL = url
    svc.CUSTOM_MODEL = model
    if key:
        svc.CUSTOM_KEY = key
    logger.info(
        f"Hydrated sidecar config from opencode.json: model={model} url={url}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage memory system lifecycle."""
    logger.info("Starting Roampal server...")

    # v0.5.4: Pull sidecar config from opencode.json before any handler runs
    # so server-side extract_tags() can reach the sidecar.
    _hydrate_sidecar_from_opencode_config()

    # v0.5.4: Resolve profile at startup for banner only — actual memory init
    # is lazy per-request. This avoids binding a single profile to the process.
    from roampal.profile_manager import (
        active_profile_name,
        active_profile_source,
        DEFAULT_PROFILE,
    )

    resolved_name = active_profile_name()
    if resolved_name != DEFAULT_PROFILE:
        logger.info(
            f"Active profile at startup: {resolved_name} (source: {active_profile_source()})"
        )

    # v0.5.4: Shared embedding service — one ONNX model for all profiles.
    global _shared_embed_service
    try:
        from roampal.backend.modules.memory.embedding_service import EmbeddingService
        _shared_embed_service = EmbeddingService()
        await _shared_embed_service.prewarm()
    except Exception as e:
        logger.warning(f"Embedding service unavailable: {e}")
        _shared_embed_service = None

    yield

    # v0.4.1: Explicit cleanup of ChromaDB adapters (replaces __del__ destructor)
    logger.info("Shutting down Roampal server...")
    for mem in _memory_by_profile.values():
        try:
            for name, adapter in getattr(mem, "collections", {}).items():
                if hasattr(adapter, "cleanup"):
                    await adapter.cleanup()
                    logger.debug(f"Cleaned up {name} adapter")
        except Exception as e:
            logger.warning(f"Error during adapter cleanup: {e}")


# ==================== Per-Request Profile Resolution (v0.5.4) ====================

def _resolve_profile_name(request: Request) -> str:
    """Resolve the profile name for a request.

    Priority: X-Roampal-Profile header > ROAMPAL_PROFILE env > active_profile_name().
    Returns a normalized key suitable for _memory_by_profile lookup.
    """
    header = request.headers.get("X-Roampal-Profile")
    if header:
        return header

    from roampal.profile_manager import active_profile_name, DEFAULT_PROFILE
    resolved = active_profile_name()
    return resolved if resolved != DEFAULT_PROFILE else "default"


async def get_memory_for_request(request: Request) -> UnifiedMemorySystem:
    """Get (or lazily create) the UnifiedMemorySystem for this request's profile."""
    profile_name = _resolve_profile_name(request)

    # Handle bogus/unknown profile names gracefully
    if not profile_name or not profile_name.strip():
        logger.warning("Empty profile name in request, falling back to default")
        profile_name = "default"

    if profile_name not in _memory_by_profile:
        async with _init_lock:
            if profile_name not in _memory_by_profile:
                try:
                    data_path = os.environ.get("ROAMPAL_DATA_PATH")
                    umem = UnifiedMemorySystem(
                        data_path=data_path,
                        profile_name=None if profile_name == "default" else profile_name,
                        embed_service=_shared_embed_service,
                    )
                    await umem.initialize()

                    # Wire TagService with sidecar-backed LLM extractor (per-profile)
                    if hasattr(umem, "_tag_service") and umem._tag_service:
                        from roampal.utils.sidecar_tag_wrapper import make_llm_tag_extractor
                        umem._tag_service.set_llm_extract_fn(make_llm_tag_extractor())

                    # v0.2.9: Cleanup legacy archived memories (per-profile, first access)
                    if umem._memory_bank_service:
                        cleaned = umem._memory_bank_service.cleanup_archived()
                        if cleaned > 0:
                            logger.info(f"v0.2.9 migration [{profile_name}]: cleaned up {cleaned} archived memories")

                    _memory_by_profile[profile_name] = umem
                    _session_manager_by_profile[profile_name] = SessionManager(umem.data_path)
                except ImportError as e:
                    logger.error(
                        f"Missing dependency for profile '{profile_name}': {e}\n"
                        "Fix: pip install roampal  (needs: onnxruntime, tokenizers, huggingface-hub)"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=f"Failed to initialize profile '{profile_name}': {e}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to initialize profile '{profile_name}': {e}\n"
                        "Common fixes:\n"
                        "  - First run? Ensure you have internet for model download (~420MB)\n"
                        "  - ChromaDB error? Check disk space and permissions on data directory\n"
                        "  - Run 'roampal doctor' for diagnostics"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail=f"Failed to initialize profile '{profile_name}': {e}"
                    )

    return _memory_by_profile[profile_name]


async def get_session_manager_for_request(request: Request) -> SessionManager:
    """Get (or lazily create) the SessionManager for this request's profile."""
    mem = await get_memory_for_request(request)
    profile_name = _resolve_profile_name(request)
    return _session_manager_by_profile.get(profile_name, None)


async def get_profile_context(request: Request) -> Tuple[UnifiedMemorySystem, SessionManager]:
    """Get (memory, session_manager) pair for this request's profile."""
    mem = await get_memory_for_request(request)
    profile_name = _resolve_profile_name(request)
    return mem, _session_manager_by_profile.get(profile_name, None)


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Roampal",
        description="Persistent memory for AI coding tools",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS — localhost only (v0.3.5: tightened from allow_origins=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # ==================== Hook Endpoints ====================

    @app.post("/api/hooks/get-context", response_model=GetContextResponse)
    async def get_context(request: GetContextRequest, main_req: Request = None):
        """
        Called by UserPromptSubmit hook BEFORE the LLM sees the message.

        Returns:
        1. Cold start user profile (on first message of session)
        2. Scoring prompt if previous exchange needs scoring AND assistant completed
        3. Relevant memories from search
        4. User facts from memory_bank
        """
        _memory, _session_manager = await get_profile_context(main_req)

        try:
            # Evict stale cache entries (30-minute TTL)
            _evict_stale_entries()

            formatted_parts = []
            scoring_required = False
            scoring_prompt_text = (
                ""  # v0.3.2: track separately for OpenCode split delivery
            )
            scoring_exchange_data = (
                None  # v0.3.2: raw exchange data for independent LLM scoring
            )
            scoring_memories_data = (
                None  # v0.3.2: raw surfaced memory data for independent LLM scoring
            )
            scoring_prompt_simple_text = (
                ""  # v0.3.2: simplified scoring for non-Claude models
            )
            context_parts = []  # v0.3.2: non-scoring context parts for split delivery
            conversation_id = request.conversation_id or "default"

            # 0. Check for cold start (first message of session)
            is_cold_start = _session_manager.is_first_message(conversation_id)

            if is_cold_start:
                # Check for updates on cold start only (once per conversation)
                update_injection = _get_update_injection()
                if update_injection:
                    formatted_parts.append(update_injection)
                    context_parts.append(update_injection)

                # Dump full user profile on first message
                cold_start_profile = await _build_cold_start_profile(_memory)
                if cold_start_profile:
                    formatted_parts.append(cold_start_profile)
                    context_parts.append(cold_start_profile)
                    logger.info(
                        f"Cold start: injected user profile for {conversation_id}"
                    )

                # Mark first message as seen
                _session_manager.mark_first_message_seen(conversation_id)

            # v0.2.7: Get context for BOTH cold start and regular messages
            # Cold start uses it for KNOWN CONTEXT (recent work), non-cold start also uses for scoring
            context = await _memory.get_context_for_injection(
                query=request.query,
                conversation_id=conversation_id,
                recent_conversation=request.recent_messages,
            )

            # v0.2.7: On cold start, append KNOWN CONTEXT after profile (recent work context)
            if is_cold_start and context.get("formatted_injection"):
                formatted_parts.append(context["formatted_injection"])
                context_parts.append(context["formatted_injection"])
                logger.info(f"Cold start: added KNOWN CONTEXT for {conversation_id}")

            # 2. Check if assistant completed a response (vs user interrupting mid-work)
            assistant_completed = _session_manager.check_and_clear_completed(
                conversation_id
            )

            # Only inject scoring prompt if:
            # - Assistant completed their previous response (not mid-work interruption)
            # - There's an unscored exchange to score
            # - NOT a cold start (no previous exchange to score on first message)
            if assistant_completed and not is_cold_start:
                previous = await _session_manager.get_previous_exchange(conversation_id)

                if previous and not previous.get("scored", False):
                    # Build list of surfaced memories for selective scoring
                    surfaced_memories = context.get("relevant_memories", [])

                    # Inject scoring prompt with surfaced memories
                    scoring_prompt = _session_manager.build_scoring_prompt(
                        previous_exchange=previous,
                        current_user_message=request.query,
                        surfaced_memories=surfaced_memories
                        if surfaced_memories
                        else None,
                    )
                    scoring_prompt_simple = (
                        _session_manager.build_scoring_prompt_simple(
                            previous_exchange=previous,
                            current_user_message=request.query,
                            surfaced_memories=surfaced_memories
                            if surfaced_memories
                            else None,
                        )
                    )
                    formatted_parts.append(scoring_prompt)
                    scoring_prompt_text = (
                        scoring_prompt  # v0.3.2: track for split delivery
                    )
                    scoring_prompt_simple_text = (
                        scoring_prompt_simple  # v0.3.2: simplified for non-Claude
                    )
                    scoring_required = True
                    # v0.3.2: Raw data for independent LLM scoring (OpenCode plugin)
                    scoring_exchange_data = {
                        "user": previous.get("user", ""),
                        "assistant": previous.get("assistant", ""),
                    }
                    scoring_memories_data = []
                    if surfaced_memories:
                        for mem in surfaced_memories:
                            mem_id = mem.get("id", mem.get("doc_id", "unknown"))
                            content = mem.get("content", mem.get("text", ""))
                            scoring_memories_data.append(
                                {
                                    "id": mem_id,
                                    "content": content,
                                    "content_hint": content[:60]
                                    if content
                                    else "",  # v0.3.5: brief hint for SCORING REFERENCE
                                }
                            )
                    # Track that we injected scoring prompt (for Stop hook to check)
                    _session_manager.set_scoring_required(conversation_id, True)
                    logger.info(
                        f"Injecting scoring prompt for conversation {conversation_id} with {len(surfaced_memories)} memories"
                    )
                else:
                    # No unscored exchange, but assistant did complete
                    _session_manager.set_scoring_required(conversation_id, False)

                # v0.3.6: Exchange summarization handled by main LLM via score_memories
                # (no background sidecar needed — see Change 9 platform-split architecture)
            else:
                # User interrupted mid-work OR cold start - no scoring needed
                _session_manager.set_scoring_required(conversation_id, False)
                if not is_cold_start:
                    logger.info(
                        f"Skipping scoring - user interrupted mid-work for {conversation_id}"
                    )

            # 3. Add memory context after scoring prompt (only if not cold start)
            # v0.2.7: Cold start already added KNOWN CONTEXT above, don't duplicate
            if not is_cold_start and context.get("formatted_injection"):
                formatted_parts.append(context["formatted_injection"])
                context_parts.append(context["formatted_injection"])

            # 4. Cache doc_ids for outcome scoring via record_response
            # This ensures hook-injected memories can be scored later
            injected_doc_ids = context.get("doc_ids", [])
            timestamp = datetime.now().isoformat()
            if injected_doc_ids:
                _search_cache[conversation_id] = {
                    "doc_ids": injected_doc_ids,
                    "query": request.query,
                    "source": "hook_injection",
                    "timestamp": timestamp,
                }
                logger.info(
                    f"Cached {len(injected_doc_ids)} doc_ids from hook injection for {conversation_id}"
                )

                # v0.3.2: Populate injection map for robust multi-session scoring
                # Each doc_id maps back to the conversation that received it
                # This enables matching by doc_id instead of "most recent unscored" heuristics
                for doc_id in injected_doc_ids:
                    _injection_map[doc_id] = {
                        "conversation_id": conversation_id,
                        "injected_at": timestamp,
                        "query": request.query,
                    }
                logger.info(
                    f"Added {len(injected_doc_ids)} doc_ids to injection map for {conversation_id}"
                )

            return GetContextResponse(
                formatted_injection="\n".join(formatted_parts),
                user_facts=context.get("user_facts", []),
                relevant_memories=context.get("relevant_memories", []),
                context_summary=context.get("context_summary", ""),
                scoring_required=scoring_required,
                # v0.3.2: Split fields for OpenCode — scoring in user message, context in system prompt
                scoring_prompt=scoring_prompt_text,
                scoring_prompt_simple=scoring_prompt_simple_text,
                context_only="\n".join(context_parts) if context_parts else "",
                scoring_exchange=scoring_exchange_data,
                scoring_memories=scoring_memories_data,
            )

        except Exception as e:
            import traceback

            logger.error(f"Error getting context: {e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/hooks/check-scored")
    async def check_scored(conversation_id: str = "", request: Request = None):
        """Check if score_memories was already called for this conversation this turn.
        Used by OpenCode plugin to skip sidecar if main LLM already scored."""
        try:
            _session_manager = await get_session_manager_for_request(request)
        except Exception:
            return {"scored": False}
        if not _session_manager:
            return {"scored": False}
        scored = _session_manager.was_scored_this_turn(conversation_id)
        return {"scored": scored}

    @app.post("/api/hooks/stop", response_model=StopHookResponse)
    async def stop_hook(request: StopHookRequest, main_req: Request = None):
        """
        Called by Stop hook AFTER the LLM responds.

        v0.3.6: Two modes via lifecycle_only flag:
        - Claude Code (lifecycle_only=True): JSONL exchange tracking + state management.
          Exchanges stored in session JSONL for scoring prompt generation, but NOT in ChromaDB.
          Main LLM stores summaries directly via score_memories tool.
        - OpenCode (lifecycle_only=False): Full exchange storage in both JSONL and ChromaDB.
          Sidecar summarizes later via session.idle event.

        Always handles turn lifecycle: checks scoring compliance, marks turn complete.
        """
        _memory, _session_manager = await get_profile_context(main_req)

        try:
            conversation_id = request.conversation_id or "default"

            # Store exchange based on mode:
            # - lifecycle_only=True (Claude Code): Store in session JSONL only (for scoring prompts)
            #   Main LLM stores summaries in ChromaDB via score_memories tool.
            # - lifecycle_only=False (OpenCode): Store in both JSONL and ChromaDB working memory
            #   Sidecar summarizes later via session.idle event.
            user_msg = (request.user_message or "").strip()
            assistant_msg = (request.assistant_response or "").strip()
            doc_id = ""

            if user_msg and assistant_msg:
                if request.lifecycle_only:
                    # Claude Code path: JSONL only (for get_previous_exchange scoring lifecycle)
                    # Skip ChromaDB — main LLM stores summaries via score_memories
                    await _session_manager.store_exchange(
                        conversation_id=conversation_id,
                        user_message=request.user_message,
                        assistant_response=request.assistant_response,
                        doc_id="",  # No ChromaDB doc — main LLM handles storage
                    )
                    logger.info(
                        f"Lifecycle-only exchange stored in JSONL for {conversation_id}"
                    )
                else:
                    # OpenCode path: Full storage in both ChromaDB and JSONL
                    content = f"User: {user_msg}\n\nAssistant: {assistant_msg}"
                    store_metadata = {
                        "turn_type": "exchange",
                        "timestamp": datetime.now().isoformat(),
                    }
                    if request.metadata:
                        store_metadata.update(request.metadata)

                    # v0.5.3 Section 11: Extract tags from summary when plugin doesn't provide noun_tags
                    extracted_tags = None
                    if not request.noun_tags and store_metadata.get("memory_type") == "exchange_summary":
                        try:
                            from roampal.sidecar_service import extract_tags as sidecar_extract_tags

                            extracted_tags = sidecar_extract_tags(assistant_msg)
                            if extracted_tags:
                                logger.info(
                                    f"Extracted {len(extracted_tags)} tags from exchange via sidecar"
                                )
                        except Exception as tag_err:
                            logger.warning(f"Failed to extract tags from exchange (non-fatal): {tag_err}")

                    effective_noun_tags = request.noun_tags or extracted_tags

                    doc_id = await _memory.store_working(
                        content=content,
                        conversation_id=conversation_id,
                        metadata=store_metadata,
                        noun_tags=effective_noun_tags,
                    )

                    await _session_manager.store_exchange(
                        conversation_id=conversation_id,
                        user_message=request.user_message,
                        assistant_response=request.assistant_response,
                        doc_id=doc_id,
                    )
                    logger.info(f"Full exchange {doc_id} stored for {conversation_id}")
            else:
                logger.info(
                    f"State-only stop hook for {conversation_id} (no exchange data)"
                )

            # === State management (always runs) ===

            # IMPORTANT: Check scoring flags BEFORE set_completed() resets them
            scoring_was_required = _session_manager.was_scoring_required(
                conversation_id
            )

            # Race condition fix: If scoring was required, wait briefly for record_response
            # MCP tool call to complete. The tool might be in-flight when Stop hook fires.
            scored_this_turn = _session_manager.was_scored_this_turn(conversation_id)

            if scoring_was_required and not scored_this_turn:
                # Wait up to 500ms with 50ms intervals for the MCP tool to complete
                for _ in range(10):
                    await asyncio.sleep(0.05)  # 50ms
                    scored_this_turn = _session_manager.was_scored_this_turn(
                        conversation_id
                    )
                    if scored_this_turn:
                        logger.info(
                            f"Race condition resolved: record_response completed after {(_ + 1) * 50}ms"
                        )
                        break

            # v0.3.2: Session ID mismatch fallback - OpenCode plugin hooks use ses_xxx
            # but MCP tools use mcp_xxx. Check if any MCP session was scored this turn.
            if scoring_was_required and not scored_this_turn:
                state = _session_manager._load_completion_state()
                for sid, sdata in state.items():
                    if sid.startswith("mcp_") and sdata.get("scored_this_turn"):
                        scored_this_turn = True
                        _session_manager.set_scored_this_turn(sid, False)
                        logger.info(
                            f"Session ID mismatch resolved: MCP {sid} scored for plugin {conversation_id}"
                        )
                        break

            # Mark assistant as completed - this signals UserPromptSubmit that
            # scoring is needed on the NEXT user message
            _session_manager.set_completed(conversation_id)
            logger.info(f"Marked assistant as completed for {conversation_id}")

            # Determine blocking based on scoring state
            scoring_complete = False
            should_block = False
            block_message = None

            if scored_this_turn:
                scoring_complete = True
                logger.info(
                    f"score_memories was called this turn for {conversation_id}"
                )
            elif scoring_was_required:
                # SOFT ENFORCE: Log warning but don't block
                should_block = False
                block_message = None
                logger.warning(
                    f"Soft enforce: score_memories not called for {conversation_id}"
                )
            else:
                logger.info(
                    f"No scoring required this turn for {conversation_id} - not blocking"
                )

            # v0.4.8: autoSummarize removed — caused Ollama contention with sidecar scoring.

            return StopHookResponse(
                stored=bool(doc_id),
                doc_id=doc_id,
                scoring_complete=scoring_complete,
                should_block=should_block,
                block_message=block_message,
            )

        except Exception as e:
            logger.error(f"Error in stop hook: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Memory API Endpoints ====================

    @app.post("/api/search")
    async def search_memory(request: SearchRequest, main_req: Request = None):
        """Search across memory collections."""
        _memory = await get_memory_for_request(main_req)

        try:
            # Direct ID lookup — bypass search entirely
            if request.id:
                doc = _memory.get_by_id(request.id)
                results = [doc] if doc else []
                return {"query": request.id, "count": len(results), "results": results}

            # Convert days_back to date filter
            metadata_filters = request.metadata_filters or {}
            if request.days_back:
                from datetime import timedelta

                cutoff = (
                    datetime.now() - timedelta(days=request.days_back)
                ).isoformat()
                metadata_filters["created_at"] = {"$gte": cutoff}

            results = await _memory.search(
                query=request.query or "",
                collections=request.collections,
                limit=request.limit,
                metadata_filters=metadata_filters if metadata_filters else None,
                sort_by=request.sort_by,
            )

            # Cache doc_ids for outcome scoring
            if request.conversation_id:
                doc_ids = [r.get("id") for r in results if r.get("id")]
                _search_cache[request.conversation_id] = {
                    "doc_ids": doc_ids,
                    "query": request.query,
                    "timestamp": datetime.now().isoformat(),
                }

            return {"query": request.query, "count": len(results), "results": results}

        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/memory-bank/add")
    async def add_to_memory_bank(request: MemoryBankAddRequest, main_req: Request = None):
        """Add a fact to memory bank."""
        _memory = await get_memory_for_request(main_req)

        MAX_MEMORY_CHARS = 2000
        content = request.content
        if content and len(content) > MAX_MEMORY_CHARS:
            content = content[:MAX_MEMORY_CHARS]
            logger.warning(
                f"Memory content truncated from {len(request.content)} to {MAX_MEMORY_CHARS} chars (safety cap)"
            )

        try:
            doc_id = await _memory.store_memory_bank(
                text=content,
                tags=request.tags,
                noun_tags=request.noun_tags,
                importance=request.importance,
                confidence=request.confidence,
                always_inject=request.always_inject,
            )

            return {"success": True, "doc_id": doc_id}

        except Exception as e:
            logger.error(f"Error adding to memory bank: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/memory-bank/update")
    async def update_memory_bank(request: MemoryBankUpdateRequest, main_req: Request = None):
        """Update a memory bank entry."""
        _memory = await get_memory_for_request(main_req)

        MAX_MEMORY_CHARS = 2000
        new_content = request.new_content
        if new_content and len(new_content) > MAX_MEMORY_CHARS:
            new_content = new_content[:MAX_MEMORY_CHARS]
            logger.warning(
                f"Updated memory content truncated from {len(request.new_content)} to {MAX_MEMORY_CHARS} chars (safety cap)"
            )

        try:
            doc_id = await _memory.update_memory_bank(
                old_content=request.old_content, new_content=new_content
            )

            return {"success": doc_id is not None, "doc_id": doc_id}

        except Exception as e:
            logger.error(f"Error updating memory bank: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/memory-bank/archive")
    async def delete_memory_bank(request: Dict[str, str], main_req: Request = None):
        """Archive a memory bank entry."""
        _memory = await get_memory_for_request(main_req)

        content = request.get("content", "")
        if not content:
            raise HTTPException(status_code=400, detail="Content required")

        try:
            success = await _memory.delete_memory_bank(content)
            return {"success": success}

        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/ingest")
    async def ingest_document(request: Dict[str, Any], main_req: Request = None):
        """
        Ingest a document into the books collection.

        Called by CLI when server is running, so data is immediately searchable.

        Request body:
            content: Document text
            title: Document title
            source: Source file path
            chunk_size: Characters per chunk (default 1000)
            chunk_overlap: Overlap between chunks (default 200)
        """
        _memory = await get_memory_for_request(main_req)

        content = request.get("content", "")
        title = request.get("title", "Untitled")
        source = request.get("source", "unknown")
        chunk_size = request.get("chunk_size", 1000)
        chunk_overlap = request.get("chunk_overlap", 200)

        if not content:
            raise HTTPException(status_code=400, detail="Content required")

        # v0.3.5: Enforce size limit at API layer (store_book also checks, but fail fast)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Content too large ({len(content)} bytes, max {max_size})",
            )

        try:
            doc_ids = await _memory.store_book(
                content=content,
                title=title,
                source=source,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            logger.info(f"Ingested '{title}' in {len(doc_ids)} chunks")

            return {
                "success": True,
                "title": title,
                "chunks": len(doc_ids),
                "doc_ids": doc_ids,
            }

        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/books")
    async def list_books(request: Request = None):
        """List all ingested books."""
        _memory = await get_memory_for_request(request)

        try:
            books = await _memory.list_books()
            return {"success": True, "books": books}
        except Exception as e:
            logger.error(f"Error listing books: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/remove-book")
    async def remove_book(request_body: Dict[str, Any], request: Request = None):
        """Remove a book by title."""
        _memory = await get_memory_for_request(request)

        title = request_body.get("title", "")
        if not title:
            raise HTTPException(status_code=400, detail="Title required")

        try:
            result = await _memory.remove_book(title)
            return {
                "success": result.get("removed", 0) > 0,
                "removed": result.get("removed", 0),
                "title": title,
                "message": result.get("message", ""),
            }
        except Exception as e:
            logger.error(f"Error removing book: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== MCP Tool Proxy Endpoints ====================

    @app.post("/api/record-response")
    async def record_response_endpoint(request: RecordResponseRequest, main_req: Request = None):
        """Record a key takeaway in working memory (MCP tool proxy)."""
        _memory = await get_memory_for_request(main_req)

        MAX_MEMORY_CHARS = 2000
        takeaway = request.key_takeaway
        if takeaway and len(takeaway) > MAX_MEMORY_CHARS:
            takeaway = takeaway[:MAX_MEMORY_CHARS]
            logger.warning(
                f"Key takeaway truncated from {len(request.key_takeaway)} to {MAX_MEMORY_CHARS} chars (safety cap)"
            )

        try:
            doc_id = await _memory.store_working(
                content=f"Key takeaway: {takeaway}",
                conversation_id=request.conversation_id,
                metadata={
                    "type": "key_takeaway",
                    "timestamp": datetime.now().isoformat(),
                },
                initial_score=0.7,
                noun_tags=request.noun_tags,
            )
            logger.info(
                f"Recorded takeaway (score=0.7): {request.key_takeaway[:50]}..."
            )
            return {"success": True, "doc_id": doc_id}

        except Exception as e:
            logger.error(f"Error recording response: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/record-outcome")
    async def record_outcome(request: RecordOutcomeRequest, main_req: Request = None):
        """
        Record outcome for learning.

        Called by the score_memories MCP tool.
        Scores:
        1. Most recent unscored exchange (across ALL sessions - handles MCP/hook ID mismatch)
        2. Cached search results (from _search_cache)
        """
        _memory = await get_memory_for_request(main_req)
        _session_manager = await get_session_manager_for_request(main_req)

        try:
            conversation_id = request.conversation_id or "default"
            doc_ids_scored = []

            # Track that score_memories was called this turn (for Stop hook blocking)
            _session_manager.set_scored_this_turn(conversation_id, True)

            # v0.3.6: Resolve the plugin session ID via injection_map BEFORE outcome check.
            # The MCP session ID (conversation_id) differs from the plugin session (ses_xxx).
            # We need to mark the plugin session as scored so check-scored returns true.
            exchange_conv_id = conversation_id
            resolved_via = None
            if request.memory_scores and _injection_map:
                for doc_id in request.memory_scores.keys():
                    injection = _injection_map.get(doc_id)
                    if injection:
                        exchange_conv_id = injection["conversation_id"]
                        _session_manager.set_scored_this_turn(exchange_conv_id, True)
                        resolved_via = f"injection_map (doc_id={doc_id})"
                        logger.info(
                            f"Resolved conversation {exchange_conv_id} via {resolved_via}"
                        )
                        break

            # Fallback: if injection_map didn't resolve, check _last_exchange_cache
            # for most recent unscored exchange (handles empty memory_scores case)
            if not resolved_via and _session_manager:
                best_ts = ""
                for cid, exc in _session_manager._last_exchange_cache.items():
                    if cid != conversation_id and not exc.get("scored", False):
                        ts = exc.get("timestamp", "")
                        if ts > best_ts:
                            best_ts = ts
                            exchange_conv_id = cid
                            resolved_via = "last_exchange_cache (most recent unscored)"
                if resolved_via:
                    _session_manager.set_scored_this_turn(exchange_conv_id, True)
                    logger.info(
                        f"Resolved conversation {exchange_conv_id} via {resolved_via}"
                    )

            # Look up the previous exchange doc_id (needed for summary replacement on OpenCode)
            # Claude Code: exchange_doc_id will be None (stop hook doesn't store exchanges)
            exchange_doc_id = None
            previous = _session_manager._last_exchange_cache.get(exchange_conv_id)
            if not previous:
                previous = _session_manager._last_exchange_cache.get(conversation_id)
                if previous:
                    exchange_conv_id = conversation_id

            if not previous:
                # Last resort: Find the MOST RECENT unscored exchange across all sessions
                best_ts = ""
                for cid, exc in _session_manager._last_exchange_cache.items():
                    if not exc.get("scored", False):
                        exc_ts = exc.get("timestamp", "")
                        if exc_ts > best_ts:
                            best_ts = exc_ts
                            previous = exc
                            exchange_conv_id = cid
                if previous:
                    logger.warning(
                        f"Used 'most recent unscored' fallback - resolved to {exchange_conv_id}"
                    )

            if (
                previous
                and previous.get("doc_id")
                and not previous.get("scored", False)
            ):
                exchange_doc_id = previous["doc_id"]
                # Score the exchange with the outcome (skip for "unknown" — no signal)
                if request.outcome in ["worked", "failed", "partial"]:
                    await _memory.record_outcome(
                        doc_ids=[exchange_doc_id], outcome=request.outcome
                    )
                await _session_manager.mark_scored(
                    exchange_conv_id, exchange_doc_id, request.outcome
                )
                logger.info(
                    f"Scored exchange {exchange_doc_id} with outcome={request.outcome}"
                )

            # v0.2.3: Skip session file scan - doc_ids are in _search_cache or request.related
            # Old approach scanned ALL session files (O(n×m) I/O) - now O(1) cache lookup

            # Score cached search results
            # v0.3.2: Try resolved exchange_conv_id first (from injection_map), then original conversation_id
            cached = _search_cache.get(exchange_conv_id, {})
            cached_doc_ids = cached.get("doc_ids", [])
            cache_key_used = exchange_conv_id

            # Fallback to original conversation_id if exchange_conv_id didn't have cache
            if not cached_doc_ids and exchange_conv_id != conversation_id:
                cached = _search_cache.get(conversation_id, {})
                cached_doc_ids = cached.get("doc_ids", [])
                cache_key_used = conversation_id

            # If no cache for this ID, find the most recent cache entry
            # This handles MCP using "default" while hook caches under real session_id
            if not cached_doc_ids and _search_cache:
                most_recent_key = max(
                    _search_cache.keys(),
                    key=lambda k: _search_cache[k].get("timestamp", ""),
                )
                cached = _search_cache.get(most_recent_key, {})
                cached_doc_ids = cached.get("doc_ids", [])
                cache_key_used = most_recent_key
                if cached_doc_ids:
                    logger.info(
                        f"Using cache from session {most_recent_key} (MCP used {conversation_id})"
                    )

            # v0.2.8: Per-memory scoring - process each memory with individual outcome
            # v0.4.4: Score all memories concurrently
            if request.memory_scores:
                outcome_tasks = []
                for doc_id, mem_outcome in request.memory_scores.items():
                    if mem_outcome in ["worked", "failed", "partial", "unknown"]:
                        outcome_tasks.append(
                            _memory.record_outcome(
                                doc_ids=[doc_id], outcome=mem_outcome
                            )
                        )
                        doc_ids_scored.append(doc_id)
                if outcome_tasks:
                    await asyncio.gather(*outcome_tasks)
                logger.info(
                    f"Per-memory scoring: {len(doc_ids_scored)} memories with individual outcomes"
                )

            # DEPRECATED: related param (backward compat)
            elif request.related is not None:
                doc_ids_scored.extend(request.related)
                logger.info(
                    f"Direct scoring (deprecated): {len(request.related)} doc_ids from related param"
                )
                if doc_ids_scored and request.outcome in [
                    "worked",
                    "failed",
                    "partial",
                ]:
                    await _memory.record_outcome(
                        doc_ids=doc_ids_scored, outcome=request.outcome
                    )

            # Fallback: score all cached with exchange outcome
            elif cached_doc_ids:
                doc_ids_scored.extend(cached_doc_ids)
                logger.info(f"Cache scoring: {len(cached_doc_ids)} doc_ids")
                if doc_ids_scored and request.outcome in [
                    "worked",
                    "failed",
                    "partial",
                ]:
                    await _memory.record_outcome(
                        doc_ids=doc_ids_scored, outcome=request.outcome
                    )

            # v0.3.6: Store exchange summary
            # Claude Code: main LLM stores summary directly (no prior exchange to replace)
            # OpenCode: replace the full exchange stored by stop hook with summary
            # v0.5.3: Extract tags from content when plugin doesn't provide noun_tags
            summary_stored = False
            # v0.5.4: Initialize before the summary block so the facts loop
            # below always has a defined value to read. Without this, requests
            # carrying facts but no exchange_summary hit a NameError on
            # `effective_summary_tags` and every fact fails to store.
            effective_summary_tags = request.noun_tags or None
            if request.exchange_summary and _memory:
                try:
                    # v0.5.4: Route tag extraction through TagService (Desktop-aligned)
                    # instead of calling sidecar_extract_tags directly. Extract once at
                    # exchange level (preserves Qwen's 1-call/request perf) and reuse the
                    # result for both summary storage and the facts loop below.
                    extracted_tags = None
                    if not request.noun_tags and hasattr(_memory, "_tag_service") and _memory._tag_service:
                        try:
                            extracted_tags = await _memory._tag_service.extract_tags_async(
                                request.exchange_summary
                            )
                            if extracted_tags:
                                logger.info(
                                    f"Extracted {len(extracted_tags)} tags from exchange summary via TagService"
                                )
                        except Exception as tag_err:
                            logger.warning(f"Failed to extract tags from summary (non-fatal): {tag_err}")

                    effective_summary_tags = request.noun_tags or extracted_tags

                    if exchange_doc_id:
                        # OpenCode path: replace existing full exchange with summary
                        adapter = _memory.collections.get("working")
                        if adapter:
                            doc = adapter.get_fragment(exchange_doc_id)
                            if doc:
                                metadata = doc.get("metadata", {})
                                metadata["text"] = request.exchange_summary
                                metadata["content"] = request.exchange_summary
                                metadata["memory_type"] = "exchange_summary"
                                metadata["exchange_outcome"] = request.outcome
                                metadata["summarized_at"] = datetime.now().isoformat()
                                metadata["original_length"] = len(
                                    doc.get("content", "")
                                )
                                # v0.4.5: noun_tags for TagCascade retrieval
                                if effective_summary_tags:
                                    import json as _json

                                    metadata["noun_tags"] = _json.dumps(
                                        effective_summary_tags
                                    )

                                embedding = await _memory._embedding_service.embed_text(
                                    request.exchange_summary
                                )
                                await adapter.upsert_vectors(
                                    ids=[exchange_doc_id],
                                    vectors=[embedding],
                                    metadatas=[metadata],
                                )
                                summary_stored = True
                                logger.info(
                                    f"Replaced exchange {exchange_doc_id} with summary ({len(request.exchange_summary)} chars, was {metadata.get('original_length', '?')})"
                                )
                    else:
                        # Claude Code path: store summary directly as new working memory
                        summary_doc_id = await _memory.store_working(
                            content=request.exchange_summary,
                            conversation_id=conversation_id,
                            metadata={
                                "memory_type": "exchange_summary",
                                "exchange_outcome": request.outcome or "unknown",
                                "summarized_at": datetime.now().isoformat(),
                                "turn_type": "exchange",
                            },
                            noun_tags=effective_summary_tags,
                        )
                        summary_stored = True
                        logger.info(
                            f"Stored exchange summary {summary_doc_id} directly ({len(request.exchange_summary)} chars)"
                        )
                except Exception as e:
                    logger.error(f"Failed to store exchange summary: {e}")

            # v0.4.5: Store atomic facts as separate working memories
            # v0.4.8: Restore noun_tags on facts — benchmark proves tagged facts
            # improve TagCascade retrieval.
            # v0.5.4: Reuse exchange-level effective_summary_tags for every fact
            # (1 sidecar call total per request, Qwen's perf rework). When neither the
            # plugin nor exchange-level extraction produced tags, store_working's
            # auto-extract via TagService kicks in as the last fallback.
            if request.facts and _memory:
                for fact_text in request.facts:
                    if not fact_text or len(fact_text.strip()) < 10:
                        continue
                    try:
                        fact_tags = (
                            list(effective_summary_tags)
                            if effective_summary_tags
                            else None
                        )
                        await _memory.store_working(
                            content=fact_text,
                            conversation_id=conversation_id,
                            metadata={
                                "memory_type": "fact",
                                "timestamp": datetime.now().isoformat(),
                            },
                            noun_tags=request.noun_tags or fact_tags,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store fact: {e}")
                if request.facts:
                    logger.info(
                        f"Stored {len([f for f in request.facts if f and len(f.strip()) >= 10])} atomic facts"
                    )

            # Log final result and trigger background updates
            if doc_ids_scored:
                logger.info(f"Scored {len(doc_ids_scored)} documents")

                # ========== FAST PATH COMPLETE (v0.2.3) ==========
                # Score recorded. v0.4.5: Action KG and routing updates removed
                # (deferred-task path was dead — methods deleted in v0.4.5).

            # Clear search cache for the key we used
            if cache_key_used in _search_cache:
                del _search_cache[cache_key_used]

            # v0.3.2: Clean up injection_map for scored doc_ids
            # This prevents the map from growing indefinitely
            for doc_id in doc_ids_scored:
                if doc_id in _injection_map:
                    del _injection_map[doc_id]

            return {
                "success": True,
                "outcome": request.outcome,
                "documents_scored": len(doc_ids_scored),
                "summary_stored": summary_stored,
            }

        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Content Update (v0.3.6 Summarization) ====================

    @app.post("/api/memory/update-content")
    async def update_memory_content(request: UpdateContentRequest, main_req: Request = None):
        """
        Update a memory's content and re-embed it.

        v0.3.6: Used by `roampal summarize` to replace long memories with summaries.
        """
        _memory = await get_memory_for_request(main_req)

        try:
            collection = request.collection
            if collection not in _memory.collections:
                raise HTTPException(
                    status_code=400, detail=f"Unknown collection: {collection}"
                )

            adapter = _memory.collections[collection]
            doc = adapter.get_fragment(request.doc_id)
            if not doc:
                raise HTTPException(
                    status_code=404, detail=f"Document not found: {request.doc_id}"
                )

            # Update metadata with new content
            metadata = doc.get("metadata", {})
            metadata["text"] = request.new_content
            metadata["content"] = request.new_content
            metadata["summarized_at"] = datetime.now().isoformat()
            metadata["original_length"] = len(doc.get("content", ""))
            # v0.4.5: noun_tags for TagCascade retrieval
            if request.noun_tags:
                metadata["noun_tags"] = json.dumps(request.noun_tags)

            # Re-embed with new content
            embedding = await _memory._embedding_service.embed_text(request.new_content)
            await adapter.upsert_vectors(
                ids=[request.doc_id], vectors=[embedding], metadatas=[metadata]
            )

            logger.info(
                f"Updated content for {request.doc_id}: {metadata['original_length']} -> {len(request.new_content)} chars"
            )
            return {
                "success": True,
                "doc_id": request.doc_id,
                "new_length": len(request.new_content),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating content: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # v0.4.8: autoSummarize removed — caused Ollama contention with sidecar scoring.
    # Sidecar produces proper summaries; no oversized memories are created.

    # Retag endpoint — re-extract tags on existing memories via LLM
    VALID_RETAG_COLLECTIONS = {"working", "history", "patterns", "memory_bank", "all"}

    class RetagRequest(BaseModel):
        collection: str = Field(
            default="all",
            description="Collection to retag: working, history, patterns, memory_bank, or all",
        )
        limit: Optional[int] = Field(
            default=None,
            description="Maximum number of memories to process (max 5000)",
            le=5000,
            ge=1,
        )
        dry_run: bool = Field(
            default=False, description="Preview changes without modifying"
        )
        model: Optional[str] = Field(
            default=None,
            description="Specific model to use (default: sidecar)",
            max_length=100,
        )

    class RetagResponse(BaseModel):
        processed: int = Field(description="Number of memories processed")
        tags_added: int = Field(description="Number of new LLM tags added")
        errors: int = Field(description="Number of errors encountered")
        sample_updates: List[Dict[str, Any]] = Field(
            default_factory=list, description="Sample of updated memories"
        )

    @app.post("/api/retag", response_model=RetagResponse)
    async def retag_memories(request: RetagRequest, main_req: Request = None):
        """
        Re-extract tags on existing memories using the sidecar LLM.

        Reads memories, extracts fresh tags, replaces existing ones.
        Use for ongoing cleanup or to improve tag quality over time.
        """
        _memory = await get_memory_for_request(main_req)

        # v0.4.9: Validate collection name against whitelist
        if request.collection not in VALID_RETAG_COLLECTIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid collection '{request.collection}'. Must be one of: {', '.join(sorted(VALID_RETAG_COLLECTIONS))}",
            )

        try:
            # Determine collections to process
            collections_to_process = []
            if request.collection == "all":
                collections_to_process = [
                    "working",
                    "history",
                    "patterns",
                    "memory_bank",
                ]
            else:
                collections_to_process = [request.collection]

            processed = 0
            tags_added = 0
            errors = 0
            sample_updates = []

            logger.info(
                f"Starting retag operation: collections={collections_to_process}, limit={request.limit}, dry_run={request.dry_run}"
            )

            for collection_name in collections_to_process:
                if collection_name not in _memory.collections:
                    logger.warning(f"Skipping unknown collection: {collection_name}")
                    continue

                adapter = _memory.collections[collection_name]

                # Get memories from collection via ChromaDB .get()
                try:
                    limit = request.limit or 5000
                    result = adapter.collection.get(
                        include=["documents", "metadatas"],
                        limit=limit,
                    )
                    # Build list of dicts from ChromaDB result
                    memories = []
                    ids = result.get("ids", [])
                    docs = result.get("documents", [])
                    metas = result.get("metadatas", [])
                    for i in range(len(ids)):
                        memories.append({
                            "id": ids[i],
                            "content": docs[i] if i < len(docs) else "",
                            "metadata": metas[i] if i < len(metas) else {},
                        })
                except Exception as e:
                    logger.error(
                        f"Failed to fetch memories from {collection_name}: {e}"
                    )
                    errors += 1
                    continue

                logger.info(
                    f"Processing {len(memories)} memories from {collection_name}"
                )

                for memory in memories:
                    try:
                        processed += 1
                        doc_id = memory.get("id", "")
                        content = memory.get("content", "") or memory.get("text", "")
                        metadata = memory.get("metadata", {})

                        if not content:
                            continue

                        # Get existing tags
                        existing_tags = []
                        if "noun_tags" in metadata:
                            try:
                                existing_tags = json.loads(metadata["noun_tags"])
                            except (json.JSONDecodeError, TypeError):
                                existing_tags = []

                        # Extract LLM tags — call sidecar directly
                        # (retag is user-initiated, always uses sidecar regardless of platform)
                        try:
                            from roampal.sidecar_service import extract_tags as sidecar_extract_tags
                            llm_tags = sidecar_extract_tags(content)
                        except ImportError:
                            llm_tags = None
                        if not llm_tags:
                            # LLM extraction failed
                            continue

                        # Replace tags with fresh LLM-extracted tags
                        # Only update if tags are different
                        if set(llm_tags) != set(existing_tags):
                            tags_added += len(llm_tags)

                            if not request.dry_run:
                                # Update metadata with LLM tags (REPLACE old tags)
                                metadata["noun_tags"] = json.dumps(llm_tags)
                                metadata["retagged_at"] = datetime.now().isoformat()
                                metadata["retag_model"] = request.model or "sidecar"

                                # Update in database
                                await adapter.upsert_vectors(
                                    ids=[doc_id],
                                    vectors=[
                                        memory.get("embedding")
                                    ],  # Keep existing embedding
                                    metadatas=[metadata],
                                )

                            # Add to sample (first 5)
                            if len(sample_updates) < 5:
                                sample_updates.append(
                                    {
                                        "doc_id": doc_id,
                                        "collection": collection_name,
                                        "old_tags": existing_tags[:5],
                                        "new_tags": llm_tags[:5],
                                    }
                                )

                    except Exception as e:
                        logger.error(f"Error processing memory {doc_id}: {e}")
                        errors += 1

                    # Check limit
                    if request.limit and processed >= request.limit:
                        break

                if request.limit and processed >= request.limit:
                    break

            logger.info(
                f"Retag complete: processed={processed}, tags_added={tags_added}, errors={errors}"
            )

            return RetagResponse(
                processed=processed,
                tags_added=tags_added,
                errors=errors,
                sample_updates=sample_updates,
            )

        except Exception as e:
            logger.error(f"Retag operation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==================== Health/Status Endpoints ====================

    @app.get("/api/health")
    async def health_check():
        """
        Health check endpoint.

        v0.3.0: Actually tests embedding functionality to catch PyTorch state corruption.
        Returns 503 if embeddings are broken, allowing auto-restart.
        v0.5.4: Checks any initialized profile (singleton removed).
        """
        embedding_ok = False
        embedding_error = None

        for mem in _memory_by_profile.values():
            if mem.initialized and mem._embedding_service:
                try:
                    test_vector = await mem._embedding_service.embed_text("health check")
                    if len(test_vector) > 0:
                        embedding_ok = True
                    break
                except Exception as e:
                    embedding_error = str(e)

        if not embedding_ok:
            raise HTTPException(
                status_code=503,
                detail=f"Embedding service unhealthy: {embedding_error or 'not initialized'}",
            )

        return {
            "status": "healthy",
            "memory_initialized": len(_memory_by_profile) > 0,
            "session_manager_ready": len(_session_manager_by_profile) > 0,
            "profiles_loaded": len(_memory_by_profile),
            "embedding_ok": embedding_ok,
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/api/stats")
    async def get_stats(request: Request = None):
        """Get memory system statistics."""
        _memory = await get_memory_for_request(request)

        return _memory.get_stats()

    return app


def start_server(host: str = "127.0.0.1", port: int = None, dev: bool = False):
    """
    Start the Roampal server with dev/prod isolation.

    Args:
        host: Server host
        port: Server port (auto-determined from dev mode if not specified)
        dev: Run in dev mode (port 27183, Roampal_DEV data)
    """
    # Determine mode and port
    dev_mode = dev or os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")

    # Set env var so lifespan() can read it
    if dev_mode:
        os.environ["ROAMPAL_DEV"] = "1"

    if port is None:
        port = DEV_PORT if dev_mode else PROD_PORT

    # v0.5.2: Resolve the actual data path for the banner so it reflects the
    # active profile + any ROAMPAL_DATA_PATH override. Previous versions printed
    # an unexpanded %APPDATA%/Roampal/data literal that was wrong under named
    # profiles.
    from roampal.profile_manager import (
        DEFAULT_PROFILE,
        ProfileNotFoundError,
        active_profile_name,
        active_profile_source,
        resolve_data_path,
    )

    profile_name = active_profile_name()
    try:
        resolved_path = resolve_data_path(profile_name)
    except ProfileNotFoundError:
        resolved_path = f"<profile {profile_name!r} not registered>"

    # Startup banner (ASCII-safe for Windows cp1252)
    mode_str = "DEV" if dev_mode else "PROD"
    banner_lines = [
        "===================================================",
        f"  ROAMPAL SERVER - {mode_str} MODE",
        f"  Port: {port}",
        f"  Data: {resolved_path}",
    ]
    if profile_name != DEFAULT_PROFILE:
        banner_lines.append(
            f"  Profile: {profile_name} ({active_profile_source()})"
        )
    banner_lines.append("===================================================")
    print("\n" + "\n".join(banner_lines) + "\n")

    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Roampal FastAPI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 27182 prod, 27183 dev)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in dev mode (port 27183, Roampal_DEV data)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    start_server(host=args.host, port=args.port, dev=args.dev)

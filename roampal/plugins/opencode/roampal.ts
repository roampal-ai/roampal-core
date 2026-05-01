/**
 * Roampal Plugin for OpenCode
 *
 * Provides persistent memory capabilities through:
 * 1. Context fetch via chat.message hook (caches scoring prompt + context)
 * 2a. Memory context via experimental.chat.system.transform (system prompt, invisible in UI)
 * 2b. Scoring prompt via system.transform when sidecar is broken (system prompt, invisible in UI)
 * 3. Exchange capture via event hook (session lifecycle + message parts)
 * 4. MCP tools for memory operations (configured separately in opencode.json)
 *
 * Both Claude Code and OpenCode share the same server (27182/27183) to avoid ChromaDB locking.
 *
 * v0.3.6: Sidecar now summarizes + scores (matching Claude Code sidecar_service.py).
 *         Prompt updated to first-person voice. Summary stored as exchange_summary working memory.
 *         Compaction recovery via experimental.session.compacting hook + session.idle cleanup.
 * v0.3.4: Fix #1 — scoring prompt injection uses deep clone instead of in-place mutation.
 *         OpenCode holds refs to original message objects for UI rendering; mutating them
 *         caused garbled text in the input box. Clone goes to LLM, original stays in UI.
 *         Also: sidecar scoring deferred from chat.message to session.idle to prevent
 *         double-scoring memories (sidecar only runs if main LLM didn't call score_memories).
 * v0.3.2: Fixed event handler to use correct OpenCode event structure (event.properties.*)
 *         Added message.part.updated handling for assistant text capture
 */

import type { Plugin } from "@opencode-ai/plugin"

// ============================================================================
// Configuration
// ============================================================================

// Roampal HTTP hook server endpoint
// Both Claude Code and OpenCode share the same server to avoid ChromaDB locking issues
// ChromaDB doesn't support concurrent access from multiple processes
// Port scheme: 27182 (prod), 27183 (dev)
const ROAMPAL_DEV = process.env.ROAMPAL_DEV === "1"
const ROAMPAL_PORT = ROAMPAL_DEV ? 27183 : 27182
const ROAMPAL_API_URL = `http://127.0.0.1:${ROAMPAL_PORT}/api`  // v0.4.0: single base URL constant
const ROAMPAL_HOOK_URL = `${ROAMPAL_API_URL}/hooks`

// Debug logging to file (console.log leaks into OpenCode UI on some platforms)
import { appendFileSync, readFileSync, statSync, writeFileSync } from "fs"
import { join } from "path"
const DEBUG_LOG = join(process.env.APPDATA || process.env.HOME || ".", "roampal_plugin_debug.log")
const DEBUG_LOG_MAX_BYTES = 1024 * 1024  // 1 MB — rotate by truncating to last 256KB
function debugLog(msg: string) {
  try {
    appendFileSync(DEBUG_LOG, `[${new Date().toISOString()}] ${msg}\n`)
    // Rotate: if log exceeds 1MB, keep only the last 256KB
    try {
      const size = statSync(DEBUG_LOG).size
      if (size > DEBUG_LOG_MAX_BYTES) {
        const tail = readFileSync(DEBUG_LOG, "utf-8").slice(-256 * 1024)
        writeFileSync(DEBUG_LOG, `[LOG ROTATED at ${new Date().toISOString()}]\n${tail}`)
      }
    } catch { /* best effort rotation */ }
  } catch { /* best effort */ }
}

// v0.5.4: Profile resolution for X-Roampal-Profile header on every FastAPI call.
// Mirrors MCP server-side _get_mcp_profile_name() so OpenCode hits the right
// profile instead of defaulting to active_profile_name() on the server.
//
// v0.5.5.1: Desktop project-switch fix (#10).
// The plugin receives a client from OpenCode that has project.current().
// This queries Desktop's HTTP API (localhost:4096) for the currently active
// project's worktree directory — which updates when the user switches projects
// in the UI. We cache the resolved profile and refresh it on every chat.message
// (before any API calls), so the correct profile is used for the exchange.
let _pluginClient: any = null
let _cachedProfile: string = ""

function resolveRoampalProfile(worktree?: string): string {
  // Priority 1: explicit env var (CLI: $env:ROAMPAL_PROFILE = "..."; opencode)
  const envProfile = process.env.ROAMPAL_PROFILE
  if (envProfile && envProfile !== "default") return envProfile

  // Priority 2: project opencode.json (Desktop per-project)
  // Uses the worktree from client.project.current() when available (Desktop),
  // falls back to process.cwd() (CLI / older Desktop without project.current).
  try {
    const projectDir = worktree || process.cwd()
    const projectConfig = join(projectDir, "opencode.json")
    statSync(projectConfig)
    const config = JSON.parse(readFileSync(projectConfig, "utf-8"))
    const projectProfile = config?.mcp?.["roampal-core"]?.environment?.ROAMPAL_PROFILE
    if (projectProfile && projectProfile !== "default") return projectProfile
  } catch { /* no project config or unreadable */ }

  // Priority 3: user-global opencode.json
  try {
    const home = process.env.USERPROFILE || process.env.HOME || ""
    const xdgConfig = process.env.XDG_CONFIG_HOME || (home ? join(home, ".config") : "")
    if (xdgConfig) {
      const globalConfig = join(xdgConfig, "opencode", "opencode.json")
      statSync(globalConfig)
      const config = JSON.parse(readFileSync(globalConfig, "utf-8"))
      const globalProfile = config?.mcp?.["roampal-core"]?.environment?.ROAMPAL_PROFILE
      if (globalProfile && globalProfile !== "default") return globalProfile
    }
  } catch { /* no global config or unreadable */ }

  return ""
}

async function _withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return await Promise.race<T>([
    p,
    new Promise<T>((_, reject) => setTimeout(() => reject(new Error("timeout")), ms)),
  ])
}

// Resolve the active project's worktree.
// Primary: client.session.get({path: {id: sessionID}}) — the OpenCode SDK uses
// URL templating, so the session ID goes under `path`. The session's directory
// reflects the project the message was sent from, even when the Desktop plugin
// is a singleton across project switches. Verified against Desktop 1.14.29.
// Fallback: client.project.current() (works on CLI, returns workspace root on
// Desktop — only useful when there's no session yet, e.g. plugin load).
async function _resolveActiveWorktree(sessionID?: string): Promise<string> {
  if (sessionID && _pluginClient?.session?.get) {
    try {
      const resp = await _withTimeout(
        (_pluginClient as any).session.get({ path: { id: sessionID } }),
        2000,
      )
      const data = (resp as any)?.data ?? resp
      if (!(resp as any)?.error && !data?.error) {
        const dir = data?.directory || data?.worktree
        if (dir && dir !== "/" && dir !== "\\") return dir as string
      }
    } catch (err) {
      debugLog(`[v0.5.6] session.get failed (${err})`)
    }
  }

  if (_pluginClient?.project?.current) {
    try {
      const resp = await _withTimeout((_pluginClient as any).project.current(), 2000)
      const data = (resp as any)?.data ?? resp
      const wt = data?.worktree
      if (wt && wt !== "/" && wt !== "\\") return wt as string
    } catch (err) {
      debugLog(`[v0.5.6] project.current() failed (${err})`)
    }
  }

  return ""
}

async function refreshProfile(sessionID?: string): Promise<void> {
  const worktree = await _resolveActiveWorktree(sessionID)
  if (worktree) {
    _cachedProfile = resolveRoampalProfile(worktree)
    debugLog(`[v0.5.6] refreshProfile: worktree=${worktree} → profile=${_cachedProfile || "(none)"}`)
    return
  }
  _cachedProfile = resolveRoampalProfile()
  debugLog(`[v0.5.6] refreshProfile: no worktree from APIs, fell back to cwd → profile=${_cachedProfile || "(none)"}`)
}

let _roampalProfileResolved = false

function roampalHeaders(): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" }
  if (!_roampalProfileResolved) {
    debugLog(`[v0.5.6] Resolved profile: ${_cachedProfile || "(default — no header)"}`)
    _roampalProfileResolved = true
  }
  if (_cachedProfile) {
    h["X-Roampal-Profile"] = _cachedProfile
  }
  return h
}

// ============================================================================
// Session State Management
// ============================================================================

interface SessionContext {
  sessionId: string
  userPrompt: string
}

const sessionContextMap = new Map<string, SessionContext>()

// Last user message per session, set by chat.message for system.transform to use
const lastUserMessage = new Map<string, string>()

// Track which messageIDs are assistant messages (per session)
const assistantMessageIds = new Map<string, Set<string>>()

// Track accumulated text parts from assistant messages (per session)
// Map<sessionID, Map<partID, fullText>>
const assistantTextParts = new Map<string, Map<string, string>>()

// v0.4.9.4: Subagent filtering (Issue #4).
// Sessions belonging to subagents are tracked here so hooks without agent info
// (session.idle, message.part.updated) can skip them. Populated in chat.message
// where input.agent is available; checked in session.idle where it's not.
const subagentSessions = new Set<string>()

// v0.5.0: Track sessions where chat.message fired (primary sessions).
// OpenCode does NOT fire chat.message for subagent sessions — only event hooks
// (message.part.updated, session.idle) fire. If a session reaches session.idle
// but was never seen by chat.message, it's a subagent.
const primarySessions = new Set<string>()

// Cached agent definitions from client.app.agents() — populated at plugin load.
// Maps agent name → mode ("primary" | "subagent").
let cachedAgentModes = new Map<string, string>()

// v0.4.2: Debounce session.idle to avoid acting on subagent completions.
// OpenCode fires session.idle when subagents finish (issue #13334), which can
// store incomplete exchanges and corrupt the scoring loop. Debounce waits 1.5s
// after idle fires; if new text parts arrive, the timer resets.
const idleTimers = new Map<string, ReturnType<typeof setTimeout>>()

// v0.3.7: mainLLMScored removed — sidecar is sole scorer in OpenCode

// v0.3.7: pendingScoringPrompt removed — scoring prompt injected via system.transform when sidecar broken

// v0.5.6: Async scoring queue — only one LLM call at a time to avoid 429 pile-ups,
// but concurrent callers are queued (not dropped). Each waiter gets its own resolved value.
type _ScoringQueueItem = {
  sessionId: string
  currentUserMessage: string
  exchange: { user: string; assistant: string }
  memories: Array<{ id: string; content: string }> | null
  resolve: (ok: boolean) => void
}
let scoringQueueRunning = false
const scoringQueue: _ScoringQueueItem[] = []

// User-configurable sidecar — reads from opencode.json MCP environment at load time.
// These env vars are stored in opencode.json > mcp > roampal-core > environment by `roampal sidecar setup`.
// OpenCode passes MCP env vars to MCP server subprocesses but NOT to plugins, so we read
// the config file directly instead of process.env.
// Fallback: process.env still checked for manual/testing overrides.
function _loadSidecarConfig(): { url: string; key: string; model: string; disabled: boolean; allowSubagents: boolean } {
  try {
    // v0.4.0: Respect XDG_CONFIG_HOME on Linux
    const configDir = process.env.XDG_CONFIG_HOME || join(
      process.env.USERPROFILE || process.env.HOME || ".",
      ".config"
    )
    const configPath = join(configDir, "opencode", "opencode.json")
    const raw = readFileSync(configPath, "utf-8")
    const config = JSON.parse(raw)
    const env = config?.mcp?.["roampal-core"]?.environment || {}
    return {
      url: env.ROAMPAL_SIDECAR_URL || process.env.ROAMPAL_SIDECAR_URL || "",
      key: env.ROAMPAL_SIDECAR_KEY || process.env.ROAMPAL_SIDECAR_KEY || "",
      model: env.ROAMPAL_SIDECAR_MODEL || process.env.ROAMPAL_SIDECAR_MODEL || "",
      disabled: (env.ROAMPAL_SIDECAR_DISABLED === "true") || (process.env.ROAMPAL_SIDECAR_DISABLED === "true"),
      allowSubagents: (env.ROAMPAL_ALLOW_SUBAGENTS === "1") || (process.env.ROAMPAL_ALLOW_SUBAGENTS === "1"),
    }
  } catch {
    // Config not found — fall back to process.env only
    return {
      url: process.env.ROAMPAL_SIDECAR_URL || "",
      key: process.env.ROAMPAL_SIDECAR_KEY || "",
      model: process.env.ROAMPAL_SIDECAR_MODEL || "",
      disabled: process.env.ROAMPAL_SIDECAR_DISABLED === "true",
      allowSubagents: process.env.ROAMPAL_ALLOW_SUBAGENTS === "1",
    }
  }
}
const _sidecarCfg = _loadSidecarConfig()
const CUSTOM_SIDECAR_URL = _sidecarCfg.url
const CUSTOM_SIDECAR_KEY = _sidecarCfg.key
const CUSTOM_SIDECAR_MODEL = _sidecarCfg.model
const SIDECAR_DISABLED = _sidecarCfg.disabled
const ALLOW_SUBAGENTS = _sidecarCfg.allowSubagents
debugLog(`Sidecar config loaded: custom=${CUSTOM_SIDECAR_URL ? `${CUSTOM_SIDECAR_MODEL} via ${CUSTOM_SIDECAR_URL}` : "none (zen)"}, disabled=${SIDECAR_DISABLED}, allowSubagents=${ALLOW_SUBAGENTS}`)

// Zen proxy — default for scoring to save API credits (even for paid users).
// Defaults hardcoded; dynamically updated if "opencode" provider seen in chat.params.
const ZEN_FALLBACK_URL = "https://opencode.ai/zen/v1"
const ZEN_FALLBACK_KEY = "public"
let zenBaseURL = ZEN_FALLBACK_URL
let zenApiKey = ZEN_FALLBACK_KEY

// Free models on Zen, tried in order for scoring. If a model returns 404
// (removed by OpenCode update), the next is tried automatically.
// Verified models: glm-4.7-free (reasoning), kimi-k2.5-free, gpt-5-nano (reasoning)
const ZEN_SCORING_MODELS = ["glm-4.7-free", "kimi-k2.5-free", "gpt-5-nano"]

// Scoring timeout budgets.
// v0.3.7: Zen and fallback get SEPARATE budgets so dead Zen models can never starve fallback.
// Zen models: fast-fail (5s timeout, 1 attempt, circuit breaker skips dead ones).
// Fallback: generous budget with exponential backoff (user's model IS working — just rate-limited).
const ZEN_BUDGET_MS = 15000          // Max 15s for all Zen models combined
const FALLBACK_BUDGET_MS = 45000     // Fallback gets its own 45s budget (covers Ollama cold start)
const ZEN_REQUEST_TIMEOUT_MS = 5000  // Per-request timeout for Zen models
const FALLBACK_REQUEST_TIMEOUT_MS = 30000  // Per-request timeout for fallback (Ollama cold load = 15-20s)
// Circuit breaker: skip models that failed recently (avoids burning timeout on dead models).
// Map of model name → timestamp of last failure. Cleared after CIRCUIT_BREAKER_COOLDOWN_MS.
const modelCircuitBreaker = new Map<string, number>()
const CIRCUIT_BREAKER_COOLDOWN_MS = 120000  // 2 minutes — recover quickly from transient failures

// v0.5.6: Per-session deferred retry — multiple sessions can each have a pending score
// without overwriting each other. Each entry tracks its own retry budget.
type _PendingScoringEntry = {
  userMessage: string
  exchange: { user: string; assistant: string }
  memories: Array<{ id: string; content: string }> | null
  retryAttempts: number
}
const pendingScoringQueue = new Map<string, _PendingScoringEntry>()
const PENDING_SCORING_MAX_RETRIES = 3   // initial attempt + 2 retries

// v0.5.6 Fix G: Failed summary writes get their own deferred queue.
// When sidecar scoring succeeds but the FastAPI POST to /stop fails (TimeoutError,
// non-2xx status, network blip), the summary text was previously dropped without
// retry. Now it lands here and the background drainer retries the write — which
// is dedup-protected by exchange_fingerprint, so an already-stored summary
// (FastAPI completed despite client disconnect) returns a cheap skip.
type _PendingSummaryEntry = {
  exchange: { user: string; assistant: string }
  summary: string
  outcome: string
  fingerprint: string
  retryAttempts: number
}
const pendingSummaryQueue = new Map<string, _PendingSummaryEntry>()
const PENDING_SUMMARY_MAX_RETRIES = 3

// v0.5.6 Fix F: Background drainer for pendingScoringQueue.
// Item 19's session.idle-only retry meant queued items waited for the next user
// message before retrying. If the user went idle, the queue stalled — and a new
// user message added its own scoring task on top of stuck ones, compounding load.
// Background interval drains the queue independently of user activity.
const BACKGROUND_DRAIN_INTERVAL_MS = 30000   // 30s — matches sidecar timeout cadence
let backgroundDrainTimer: ReturnType<typeof setInterval> | null = null
let backgroundDrainRunning = false

async function drainPendingScoringQueue(source: string): Promise<void> {
  if (backgroundDrainRunning) return
  if (SIDECAR_DISABLED || pendingScoringQueue.size === 0) return
  backgroundDrainRunning = true
  try {
    const entries = Array.from(pendingScoringQueue.entries())
    for (const [pendingSid, payload] of entries) {
      payload.retryAttempts++
      await refreshProfile(pendingSid)
      debugLog(`${source}: Retrying deferred scoring for ${pendingSid} (attempt ${payload.retryAttempts}/${PENDING_SCORING_MAX_RETRIES})`)
      try {
        const ok = await scoreExchangeViaLLM(pendingSid, payload.userMessage, payload.exchange, payload.memories)
        if (ok) {
          pendingScoringQueue.delete(pendingSid)
          debugLog(`${source}: Deferred scoring succeeded for ${pendingSid}`)
        } else if (payload.retryAttempts >= PENDING_SCORING_MAX_RETRIES) {
          consecutiveFailures++
          pendingScoringQueue.delete(pendingSid)
          debugLog(`${source}: Deferred scoring dropped after ${PENDING_SCORING_MAX_RETRIES} attempts — consecutiveFailures=${consecutiveFailures}`)
        } else {
          debugLog(`${source}: Deferred scoring failed for ${pendingSid}, will retry next drain`)
        }
      } catch (err) {
        if (payload.retryAttempts >= PENDING_SCORING_MAX_RETRIES) {
          consecutiveFailures++
          pendingScoringQueue.delete(pendingSid)
          debugLog(`${source}: Deferred scoring error, dropped after ${PENDING_SCORING_MAX_RETRIES} attempts — consecutiveFailures=${consecutiveFailures}: ${err}`)
        } else {
          debugLog(`${source}: Deferred scoring error for ${pendingSid}, will retry next drain: ${err}`)
        }
      }
    }
  } finally {
    backgroundDrainRunning = false
  }
}

// v0.5.6 Fix G: Compute the same fingerprint that gates dedup at /stop. Inlined
// here so retry attempts produce the byte-identical fingerprint the original
// attempt used — letting an already-stored summary short-circuit cleanly.
function _summaryFingerprint(exchange: { user: string; assistant: string }): string {
  const fpInput = `${exchange.user.slice(0, 200)}:${exchange.assistant.slice(0, 200)}`
  let fpHash = 5381
  for (let i = 0; i < fpInput.length; i++) {
    fpHash = ((fpHash << 5) + fpHash + fpInput.charCodeAt(i)) | 0
  }
  return Math.abs(fpHash).toString(16).padStart(8, '0').slice(0, 12)
}

// v0.5.6 Fix G: Single-shot summary write. Returns true on success or dedup-skip,
// false on any failure (timeout, non-2xx, network error). Caller decides whether
// to enqueue for retry.
async function tryStoreSummary(
  sessionId: string,
  exchange: { user: string; assistant: string },
  summary: string,
  outcome: string,
  fingerprint: string
): Promise<boolean> {
  // Dedup check — skip if an entry with this fingerprint already exists.
  // Best-effort: if the dedup query itself fails, proceed with the store
  // (better duplicate than lost), and fall through to the same dedup-by-fingerprint
  // metadata lookup the previous attempt may have missed.
  try {
    const dedupResp = await fetch(`${ROAMPAL_API_URL}/search`, {
      method: "POST",
      headers: roampalHeaders(),
      body: JSON.stringify({
        query: "",
        collections: ["working"],
        limit: 1,
        sort_by: "recency",
        metadata_filters: { exchange_fingerprint: fingerprint }
      }),
      signal: AbortSignal.timeout(3000)
    })
    if (dedupResp.ok) {
      const dedupData = await dedupResp.json() as { count?: number }
      if ((dedupData.count || 0) > 0) {
        debugLog(`tryStoreSummary: SKIP — already exists (fingerprint=${fingerprint})`)
        return true
      }
    }
  } catch {
    // proceed to store
  }

  let summaryResp: Response
  try {
    summaryResp = await fetch(`${ROAMPAL_HOOK_URL}/stop`, {
      method: "POST",
      headers: roampalHeaders(),
      body: JSON.stringify({
        conversation_id: sessionId,
        user_message: exchange.user.slice(0, 200),
        assistant_response: summary,
        metadata: {
          memory_type: "exchange_summary",
          sidecar_outcome: outcome,
          exchange_fingerprint: fingerprint,
          original_user_msg_length: exchange.user.length,
          original_assistant_msg_length: exchange.assistant.length
        }
      }),
      signal: AbortSignal.timeout(30000)
    })
  } catch (err) {
    debugLog(`tryStoreSummary: network/timeout error: ${err}`)
    return false
  }

  if (!summaryResp.ok) {
    debugLog(`tryStoreSummary: failed status=${summaryResp.status}`)
    return false
  }

  let docId = ""
  try {
    const storeData = await summaryResp.json() as { doc_id?: string }
    docId = storeData.doc_id || ""
  } catch {
    // Body was 2xx but couldn't parse — treat as success (data landed)
  }
  debugLog(`tryStoreSummary: SUMMARY stored: ${summary.length} chars, doc_id=${docId}`)

  // Score the summary itself with the exchange outcome (best-effort).
  if (docId && outcome !== "unknown") {
    try {
      await fetch(`${ROAMPAL_API_URL}/record-outcome`, {
        method: "POST",
        headers: roampalHeaders(),
        body: JSON.stringify({
          conversation_id: sessionId,
          outcome,
          memory_scores: { [docId]: outcome }
        }),
        signal: AbortSignal.timeout(3000)
      })
    } catch {
      // Best effort — summary is stored regardless
    }
  }
  return true
}

// v0.5.6 Fix G: Drain pendingSummaryQueue. Same backoff/retry semantics as Fix F's
// scoring drain. Shares the backgroundDrainRunning guard so the two drainers don't
// race each other (and don't double-pump on consecutive 30s ticks).
async function drainPendingSummaryQueue(source: string): Promise<void> {
  if (SIDECAR_DISABLED || pendingSummaryQueue.size === 0) return
  const entries = Array.from(pendingSummaryQueue.entries())
  for (const [pendingSid, payload] of entries) {
    payload.retryAttempts++
    await refreshProfile(pendingSid)
    debugLog(`${source}: Retrying deferred summary for ${pendingSid} (attempt ${payload.retryAttempts}/${PENDING_SUMMARY_MAX_RETRIES})`)
    try {
      const ok = await tryStoreSummary(pendingSid, payload.exchange, payload.summary, payload.outcome, payload.fingerprint)
      if (ok) {
        pendingSummaryQueue.delete(pendingSid)
        debugLog(`${source}: Deferred summary stored for ${pendingSid}`)
      } else if (payload.retryAttempts >= PENDING_SUMMARY_MAX_RETRIES) {
        pendingSummaryQueue.delete(pendingSid)
        debugLog(`${source}: Deferred summary dropped after ${PENDING_SUMMARY_MAX_RETRIES} attempts for ${pendingSid}`)
      } else {
        debugLog(`${source}: Deferred summary failed for ${pendingSid}, will retry next drain`)
      }
    } catch (err) {
      if (payload.retryAttempts >= PENDING_SUMMARY_MAX_RETRIES) {
        pendingSummaryQueue.delete(pendingSid)
        debugLog(`${source}: Deferred summary error, dropped after ${PENDING_SUMMARY_MAX_RETRIES} attempts for ${pendingSid}: ${err}`)
      } else {
        debugLog(`${source}: Deferred summary error for ${pendingSid}, will retry next drain: ${err}`)
      }
    }
  }
}

// Pending scoring data — set in chat.message, consumed in session.idle.
// Sidecar scoring is deferred to session.idle so we can check if the main LLM
// already scored (prevents double-scoring memories — GitHub issue #1 follow-up).
const pendingScoringData = new Map<string, {
  userMessage: string
  exchange: { user: string; assistant: string }
  memories: Array<{ id: string; content: string }> | null
}>()

// v0.4.7: Post-compaction flag with generation counter.
// compactionGen increments on each compaction; session.idle only clears
// flags that predate the current generation (set before this idle ran).
// This prevents session.idle from clearing a flag set by compaction
// in the same turn (idle fires BEFORE compaction in that turn).
let compactionGen = 0
const includeRecentOnNextTurn = new Map<string, { flag: boolean; gen: number }>()

// Cached context from chat.message for system.transform to use (avoids double-fetch)
// Map<sessionID, { contextOnly, scoringRequired, scoringPromptSimple, scoringExchange, scoringMemories, timestamp }>
const cachedContext = new Map<string, {
  contextOnly: string
  scoringRequired: boolean
  scoringPromptSimple: string
  scoringExchange: { user: string; assistant: string } | null
  scoringMemories: Array<{ id: string; content: string }> | null
  recentExchanges: string | null  // v0.3.7: pre-fetched in chat.message to avoid system.transform race
  timestamp: number
}>()

// ============================================================================
// Provider Capture for Independent LLM Scoring
// ============================================================================

// Captured provider details from chat.params hook — used for independent scoring calls
let capturedProvider: {
  baseURL: string
  apiKey: string
  modelID: string
  providerID: string
  sdk: string  // e.g. "@ai-sdk/openai-compatible", "@ai-sdk/openai", "@ai-sdk/anthropic"
} | null = null

// ============================================================================
// Scoring Status State
// ============================================================================

// v0.5.0: Consecutive failure counter replaces boolean scoringBroken.
// Tracks how many exchanges in a row have failed scoring. Resets to 0 on success.
// Status tag shows "unavailable" at 2+ consecutive failures (1 could be transient).
// Retries happen every exchange regardless of count — never permanently gives up.
let consecutiveFailures = SIDECAR_DISABLED ? 2 : 0
let lastScorerLabel = ""  // e.g. "zen(glm-4.7-free)" or "fallback(deepseek-chat)"

// Per-session onboarding flag — inject first-run message once per session.
const sessionOnboarded = new Set<string>()

// ============================================================================
// Local Model Discovery from OpenCode Config
// ============================================================================

interface ScoringTarget {
  url: string
  key: string
  model: string
  label: string
  isFallback?: boolean
}

// v0.3.7: Scoring only uses the SPECIFIC model the user chose in `roampal sidecar setup`.
// The setup command writes ROAMPAL_SIDECAR_URL + ROAMPAL_SIDECAR_MODEL to opencode.json.
// We do NOT scan the full provider config — that would silently use paid API keys
// the user configured for chat, not scoring. No surprise API charges.

// ============================================================================
// Self-Healing: Server Auto-Restart
// ============================================================================

let _restartInProgress = false

async function restartServer(): Promise<boolean> {
  if (_restartInProgress) return false
  _restartInProgress = true

  try {
    const { execSync, spawn } = await import("child_process")
    const port = ROAMPAL_PORT

    // 1. Kill whatever is on the port
    try {
      if (process.platform === "win32") {
        const result = execSync("netstat -ano", { timeout: 5000, windowsHide: true }).toString()
        for (const line of result.split("\n")) {
          if (line.includes(`127.0.0.1:${port}`) && line.includes("LISTENING")) {
            const pid = line.trim().split(/\s+/).pop()
            if (pid && /^\d+$/.test(pid)) {
              execSync(`taskkill /pid ${pid} /f`, { timeout: 5000, windowsHide: true })
              debugLog(`Killed stale server process ${pid}`)
            }
            break
          }
        }
      } else {
        const result = execSync(`lsof -ti :${port}`, { timeout: 5000 }).toString().trim()
        if (result) {
          const pid = result.split("\n")[0]
          if (/^\d+$/.test(pid)) {
            execSync(`kill -9 ${pid}`, { timeout: 5000 })
            debugLog(`Killed stale server process ${pid}`)
          }
        }
      }
    } catch {
      // Best effort
    }

    await new Promise(resolve => setTimeout(resolve, 1000))

    // 2. Start fresh server
    const devMode = ROAMPAL_DEV
    const args = ["-m", "roampal.server.main", "--port", String(port)]
    if (devMode) args.push("--dev")

    if (process.platform === "win32") {
      // v0.3.7: python.exe is a Windows console app. Node's detached+windowsHide are
      // mutually exclusive at the Windows API level (CREATE_NO_WINDOW silently ignored
      // when DETACHED_PROCESS is set). Two approaches:
      //
      // 1. pythonw.exe (preferred): GUI subsystem app → no console window at all.
      //    Server handles None stdout/stderr (redirects to devnull).
      //    detached: true works cleanly since it's already a GUI app.
      //
      // 2. WScript.Shell fallback: Creates a temp VBS script with window style 0 (hidden).
      //    May produce a brief console flash on some systems.
      const { execSync: execSyncLocal } = await import("child_process")
      let usePythonw = false
      try {
        execSyncLocal("where pythonw", { timeout: 2000, windowsHide: true, stdio: "pipe" })
        usePythonw = true
      } catch {
        // pythonw not available
      }

      if (usePythonw) {
        spawn("pythonw", args, { detached: true, stdio: "ignore" }).unref()
        debugLog("Server started via pythonw (no console window)")
      } else {
        const { writeFileSync, unlinkSync } = await import("fs")
        const { tmpdir } = await import("os")
        const { join } = await import("path")
        const vbsPath = join(tmpdir(), `roampal_start_${Date.now()}.vbs`)
        const pyCmd = `python ${args.join(" ")}`
        writeFileSync(vbsPath, `CreateObject("WScript.Shell").Run "${pyCmd}", 0, False`)
        spawn("wscript", [vbsPath], { detached: true, stdio: "ignore" }).unref()
        setTimeout(() => { try { unlinkSync(vbsPath) } catch {} }, 5000)
        debugLog("Server started via WScript.Shell (VBS fallback)")
      }
    } else {
      // Unix: detached so server survives parent exit
      spawn("python3", args, { detached: true, stdio: "ignore" }).unref()
    }

    debugLog(`Starting fresh server on port ${port}`)

    // 3. Poll for health
    const healthUrl = `http://127.0.0.1:${port}/api/health`
    const start = Date.now()
    const timeout = 15000

    while (Date.now() - start < timeout) {
      try {
        const resp = await fetch(healthUrl, { signal: AbortSignal.timeout(2000) })
        if (resp.ok) {
          debugLog("Server restarted successfully")
          return true
        }
      } catch {
        // Not ready yet
      }
      await new Promise(resolve => setTimeout(resolve, 1000))
    }

    console.error("[roampal] Server restart timed out")
    return false
  } catch (error) {
    console.error("[roampal] Failed to restart server:", error)
    return false
  } finally {
    _restartInProgress = false
  }
}

// ============================================================================
// HTTP Client Functions
// ============================================================================

interface ContextResponse {
  formatted_injection: string
  scoring_required: boolean
  scoring_prompt_simple: string  // model-agnostic scoring prompt for fallback when sidecar is broken (no XML tags, works with any model)
  context_only: string
  // v0.3.2: Raw scoring data for sidecar scoring
  scoring_exchange: { user: string; assistant: string } | null
  scoring_memories: Array<{ id: string; content: string }> | null
}

async function getContextFromRoampal(
  sessionId: string,
  userPrompt: string
): Promise<{
  contextOnly: string
  injection: string
  scoringRequired: boolean
  scoringPromptSimple: string
  scoringExchange: { user: string; assistant: string } | null
  scoringMemories: Array<{ id: string; content: string }> | null
}> {
  try {
    const response = await fetch(`${ROAMPAL_HOOK_URL}/get-context`, {
      method: "POST",
      headers: roampalHeaders(),
      body: JSON.stringify({
        query: userPrompt,
        conversation_id: sessionId
      })
    })

    if (!response.ok) {
      // v0.3.2: Self-healing on 503 (embedding corruption)
      if (response.status === 503) {
        console.error("[roampal] Server unhealthy (503), attempting restart...")
        if (await restartServer()) {
          const retry = await fetch(`${ROAMPAL_HOOK_URL}/get-context`, {
            method: "POST",
            headers: roampalHeaders(),
            body: JSON.stringify({ query: userPrompt, conversation_id: sessionId })
          })
          if (retry.ok) {
            const data: ContextResponse = await retry.json()
            return {
              contextOnly: data.context_only || "",
              injection: data.formatted_injection || "",
              scoringRequired: data.scoring_required || false,
              scoringPromptSimple: data.scoring_prompt_simple || "",
              scoringExchange: data.scoring_exchange || null,
              scoringMemories: data.scoring_memories || null
            }
          }
        }
      }
      console.error(`[roampal] Hook server returned ${response.status}`)
      return { contextOnly: "", injection: "", scoringRequired: false, scoringPromptSimple: "", scoringExchange: null, scoringMemories: null }
    }

    const data: ContextResponse = await response.json()
    return {
      contextOnly: data.context_only || "",
      injection: data.formatted_injection || "",
      scoringRequired: data.scoring_required || false,
      scoringPromptSimple: data.scoring_prompt_simple || "",
      scoringExchange: data.scoring_exchange || null,
      scoringMemories: data.scoring_memories || null
    }
  } catch (error) {
    // v0.3.2: Self-healing on connection failure
    console.error("[roampal] Hook server unavailable, attempting restart...")
    if (await restartServer()) {
      try {
        const retry = await fetch(`${ROAMPAL_HOOK_URL}/get-context`, {
          method: "POST",
          headers: roampalHeaders(),
          body: JSON.stringify({ query: userPrompt, conversation_id: sessionId })
        })
        if (retry.ok) {
          const data: ContextResponse = await retry.json()
          return {
            contextOnly: data.context_only || "",
            injection: data.formatted_injection || "",
            scoringRequired: data.scoring_required || false,
            scoringPromptSimple: data.scoring_prompt_simple || "",
            scoringExchange: data.scoring_exchange || null,
            scoringMemories: data.scoring_memories || null
          }
        }
      } catch {
        // Retry also failed
      }
    }
    return { contextOnly: "", injection: "", scoringRequired: false, scoringPromptSimple: "", scoringExchange: null, scoringMemories: null }
  }
}

// v0.4.8: storeExchange removed — sidecar stores summaries via /stop.
// If sidecar fails, nothing is stored (no raw transcript pollution).

// ============================================================================
// Helper Functions
// ============================================================================

function extractTextFromParts(parts: any[]): string {
  if (!Array.isArray(parts)) return ""
  const textPart = parts.find((p: any) => p.type === "text")
  return textPart?.text || ""
}

/**
 * v0.4.9.4: Detect subagent by mode (Issue #4).
 * Primary: uses cachedAgentModes from client.app.agents() (authoritative).
 * Fallback: name-based heuristic for when the API isn't available.
 */
function isSubagent(agentName?: string): boolean {
  if (!agentName) return false
  // Mode-based: authoritative if agent list was cached
  const mode = cachedAgentModes.get(agentName)
  if (mode) return mode === "subagent"
  // Fallback heuristic: catch common subagent naming patterns
  const lower = agentName.toLowerCase()
  return lower.includes("subagent") || lower.startsWith("task-") || lower.startsWith("task_")
}

// ============================================================================
// Independent LLM Scoring
// ============================================================================

/**
 * Make an independent LLM call to summarize + score the previous exchange.
 * Always uses Zen free models — saves paid users' API credits.
 * Falls through model list if one is unavailable (resilient to Zen updates).
 *
 * v0.3.7: Sole scorer in OpenCode — main LLM never sees scoring prompt.
 * Produces summary, exchange outcome, AND per-memory scores in one call.
 * The summary is stored as a working memory with memory_type: "exchange_summary" metadata.
 */
// v0.5.6: Public async-queue wrapper — callers get a result even if another scoring call is in flight.
async function scoreExchangeViaLLM(
  sessionId: string,
  currentUserMessage: string,
  exchange: { user: string; assistant: string },
  memories: Array<{ id: string; content: string }> | null
): Promise<boolean> {
  if (scoringQueueRunning) {
    debugLog(`scoreExchange QUEUED — ${scoringQueue.length + 1} waiting`)
    return new Promise<boolean>((resolve) => {
      scoringQueue.push({ sessionId, currentUserMessage, exchange, memories, resolve })
    })
  }

  scoringQueueRunning = true
  try {
    const result = await _scoreExchangeViaLLM(sessionId, currentUserMessage, exchange, memories)
    return result
  } finally {
    // Drain queue sequentially. Each waiter gets its own resolved value.
    while (scoringQueue.length > 0) {
      const next = scoringQueue.shift()!
      try {
        const ok = await _scoreExchangeViaLLM(
          next.sessionId, next.currentUserMessage, next.exchange, next.memories
        )
        next.resolve(ok)
      } catch (err) {
        debugLog(`scoreExchange queued-call error: ${err}`)
        next.resolve(false)
      }
    }
    scoringQueueRunning = false
  }
}

// Private worker — actual scoring logic. Called by queue wrapper or directly.
async function _scoreExchangeViaLLM(
  sessionId: string,
  currentUserMessage: string,
  exchange: { user: string; assistant: string },
  memories: Array<{ id: string; content: string }> | null
): Promise<boolean> {
    // v0.3.6: First-person prompt matching Claude Code sidecar (sidecar_service.py).
    // Asks for summary, outcome, and per-memory scores in one call.
    // Per-memory scoring uses relevance heuristic — sidecar can judge topic relevance
    // but not actual usage, so irrelevant memories get "unknown" instead of inheriting
    // unearned exchange outcomes. Conservative: defaults to "unknown" when unsure.
    // v0.5.5: Wrap memories in delimiters with explicit "do not summarize" hint.
    // Small scoring models (qwen3:1.7b et al.) treat undelimited memory blocks as
    // content to summarize, producing summaries that lift verbatim text from prior
    // memories. Once those contaminated summaries land back in the memory bank,
    // they get re-injected next turn, creating a self-reinforcing pollution loop.
    // Delimiters + an explicit instruction keep the model's summary scope tight
    // to the exchange itself.
    const memorySection = memories?.length
      ? `\n<memories_to_score>\nThese are stored memories — score them below. DO NOT summarize, quote, or reference their content in the "summary" field.\n${memories.map(m => `- ${m.id}: "${m.content}"`).join("\n")}\n</memories_to_score>\n`
      : ""

    const memoryScoreSection = memories?.length
      ? `\n"memory_scores": { ${memories.map(m => `"${m.id}": "<score>"`).join(", ")} }`
      : ""

    const memoryInstructions = memories?.length
      ? `
MEMORY SCORES: For each memory, judge based on topic relevance and exchange outcome.
1. Memory is NOT about the topic discussed → "unknown"
2. Memory IS about the topic AND outcome is "worked" → "worked"
3. Memory IS about the topic AND outcome is "failed":
   - Your response echoed/relied on info from this memory → "failed"
   - The failure seems unrelated to this memory's content → "unknown"
4. Memory IS about the topic AND outcome is "partial" → "partial"
5. Your response contradicts what the memory says and the exchange worked → "unknown"
6. Memory contains good advice/instructions the response IGNORED (didn't follow) → "unknown" not "failed". "failed" means the memory's content was WRONG and caused a bad response, not that the model failed to follow good advice.
7. When in doubt → "unknown"`
      : ""

    // v0.4.8: Call 1 — score_exchange (summary + outcome + memory_scores only).
    // Matches benchmark sidecar_score(). Tags and facts extracted separately.
    // v0.5.1: Cap exchange fields at 8K chars to match extract_facts; safety net
    // against a massive exchange blowing the scoring model's context window.
    const scoringPrompt = `<exchange_to_summarize>
USER: "${exchange.user.slice(0, 8000)}"
ASSISTANT: "${exchange.assistant.slice(0, 8000)}"
USER_FOLLOW_UP: "${currentUserMessage.slice(0, 8000)}"
</exchange_to_summarize>
${memorySection}
Respond with ONLY a JSON object:
{ "exchange_summary": "<around 300 chars, 1-2 sentences if possible>", "exchange_outcome": "<worked|failed|partial|unknown>"${memoryScoreSection} }

SUMMARY (around 300 chars, 1-2 sentences if possible): Capture what happened AND what changed. Summaries provide context and continuity — the story behind the facts.
- Include names, topics, and the flow of the conversation
- Note corrections, decisions, and new information alongside the context
- Help future retrieval understand WHY something matters, not just WHAT
BAD: "User and assistant had a conversation" (empty, no content)
GOOD: "User corrected the baking temp from 375F to 350F while adapting the recipe for a convection oven — first attempt burned the edges"
GOOD: "Discussed switching API from REST to GraphQL after mobile team reported nested query issues with the current setup"
GOOD: "User shared that their daughter Emma starts kindergarten in September, worried about the bus route — asked for tips on easing the transition"

OUTCOME: Based on the user's follow-up:
- worked: user confirmed, moved on, or was satisfied
- failed: user corrected you, got frustrated, or asked to redo
- partial: helped but incomplete or needed adjustment
- unknown: no clear signal
${memoryInstructions}`

    // Build model queue. Simple and explicit — only models the user chose:
    //   - User ran `roampal sidecar setup` → chose a model → written as SIDECAR_URL/MODEL
    //   - Default (no setup): Zen free models only
    // NEVER scan the full provider config — that would silently use paid API keys.
    const targets: ScoringTarget[] = []
    const mainModel = capturedProvider?.modelID || ""

    // 1. User's chosen scoring model (set by `roampal sidecar setup`)
    //    Written as ROAMPAL_SIDECAR_URL + ROAMPAL_SIDECAR_MODEL in opencode.json.
    //    This is the ONLY model the user explicitly chose for scoring.
    if (CUSTOM_SIDECAR_URL && CUSTOM_SIDECAR_MODEL) {
      targets.push({
        url: CUSTOM_SIDECAR_URL,
        key: CUSTOM_SIDECAR_KEY,
        model: CUSTOM_SIDECAR_MODEL,
        label: `sidecar(${CUSTOM_SIDECAR_MODEL})`,
        isFallback: true  // gets fallback budget (25s)
      })
    }

      // 2. Zen free models (best-effort free tier)
      //    Skipped when user configured a scoring model — they chose for a reason.
      //    Zen routes through OpenCode's proxy which may log data.
      if (!CUSTOM_SIDECAR_URL) {
        const zenModels = ZEN_SCORING_MODELS.filter(m => m !== mainModel)
        if (zenModels.length === 0) zenModels.push(...ZEN_SCORING_MODELS)
        for (const m of zenModels) {
          targets.push({ url: zenBaseURL, key: zenApiKey, model: m, label: `zen(${m})` })
        }
      }

    // Try each target in order. Skip to next on 404/500 (model removed or broken).
    // Retry same target on 429 (rate limit, temporary).
    // v0.3.7: Zen and fallback get SEPARATE budgets. Dead Zen models can never starve fallback.
    const zenStartTime = Date.now()
    let fallbackStartTime = 0

    for (const target of targets) {
      // Separate budget tracking: Zen and fallback have independent clocks.
      const phaseStart = target.isFallback ? fallbackStartTime : zenStartTime
      const phaseBudget = target.isFallback ? FALLBACK_BUDGET_MS : ZEN_BUDGET_MS

      if (!target.isFallback && (Date.now() - zenStartTime) > ZEN_BUDGET_MS) {
        debugLog(`scoreExchange ZEN BUDGET expired — skipping remaining Zen models`)
        continue  // skip to fallback
      }

      // Circuit breaker: skip Zen models that failed recently.
      // NEVER skip the fallback — the user is actively chatting with it, it works.
      if (!target.isFallback) {
        const lastFailure = modelCircuitBreaker.get(target.model)
        if (lastFailure && (Date.now() - lastFailure) < CIRCUIT_BREAKER_COOLDOWN_MS) {
          debugLog(`scoreExchange SKIP ${target.label} — circuit breaker (failed ${Math.round((Date.now() - lastFailure) / 1000)}s ago)`)
          continue
        }
      }

      // Fallback: reset clock and wait for rate limit to cool down.
      // The main conversation just finished, so the rate limit window needs time to reset.
      if (target.isFallback) {
        fallbackStartTime = Date.now()
        const cooldown = 5000
        debugLog(`scoreExchange fallback cooldown — waiting ${cooldown}ms for rate limit reset`)
        await new Promise(resolve => setTimeout(resolve, cooldown))
      }

      debugLog(`scoreExchange trying ${target.label} via ${target.url}`)
      const requestTimeout = target.isFallback ? FALLBACK_REQUEST_TIMEOUT_MS : ZEN_REQUEST_TIMEOUT_MS
      // Fallback gets 4 attempts with exponential backoff (3s, 4.5s, 7s, 10s).
      // Zen models get 1 attempt — fail fast, circuit breaker, move on.
      const maxAttempts = target.isFallback ? 4 : 1

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        if (Date.now() - (target.isFallback ? fallbackStartTime : zenStartTime) > phaseBudget) break

        try {
          // All targets use OpenAI-compatible /chat/completions
          // max_tokens 4000 (v0.5.3.1): reasoning models (qwen3, qwen3.6, glm)
          // burn 500-2000 tokens thinking before producing JSON answer. 2000
          // was too low for verbose summaries on qwen3.6 — output got truncated
          // mid-string with no closing brace, causing parse failures.
          const headers: Record<string, string> = {
            "Content-Type": "application/json",
            "User-Agent": "roampal-sidecar/1.0",
          }
          if (target.key) headers["Authorization"] = `Bearer ${target.key}`

          // v0.4.9.4: `think: false` removed entirely.
          // Was Ollama-specific but broke OpenAI/Azure/Groq/etc (400 on unknown fields).
          // The /no_think text prefix in messages already handles reasoning model suppression
          // for all providers — no API-level field needed.
          const resp = await fetch(`${target.url}/chat/completions`, {
            method: "POST",
            headers,
            body: JSON.stringify({
              model: target.model,
              messages: [
                // /no_think disables qwen3's thinking mode (avoids burning all tokens on chain-of-thought).
                // Other models ignore it as a harmless prefix. Scoring is simple — doesn't need reasoning.
                { role: "system", content: "/no_think\nYou are part of a memory system. Return ONLY valid JSON with exchange_summary, exchange_outcome, and memory_scores fields. No other text. Be concise." },
                { role: "user", content: "/no_think\n" + scoringPrompt }
              ],
              max_tokens: 4000,
              temperature: 0,
            }),
            signal: AbortSignal.timeout(requestTimeout)
          })

          if (resp.status === 429) {
            const retryAfter = resp.headers.get("retry-after")
            // Fallback: exponential backoff (3s, 4.5s, 7s, 10s) — rate limit needs time.
            // Zen: no retry on 429 — fail fast, move to next model (maxAttempts=1).
            const baseDelay = target.isFallback ? 3000 * Math.pow(1.5, attempt) : 2000
            const serverDelay = retryAfter ? parseInt(retryAfter) * 1000 : baseDelay
            // Cap at 5s per retry — never honor server's retry-after if it would blow the budget.
            const remaining = phaseBudget - (Date.now() - (target.isFallback ? fallbackStartTime : zenStartTime))
            const delay = Math.min(serverDelay, 5000, remaining - requestTimeout)
            if (delay <= 0) {
              debugLog(`scoreExchange 429 on ${target.label} — no budget left for retry, moving on`)
              break
            }
            debugLog(`scoreExchange 429 on ${target.label} — retry ${attempt + 1}/${maxAttempts} in ${Math.round(delay)}ms`)
            await new Promise(resolve => setTimeout(resolve, delay))
            continue  // retry same target
          }

          if (resp.status === 404 || resp.status >= 500) {
            debugLog(`scoreExchange ${target.label} returned ${resp.status} — trying next`)
            modelCircuitBreaker.set(target.model, Date.now())
            break
          }

          if (!resp.ok) {
            debugLog(`scoreExchange ${target.label} returned ${resp.status}`)
            modelCircuitBreaker.set(target.model, Date.now())
            break  // unexpected error — try next target
          }

          const data = await resp.json()
          debugLog(`scoreExchange raw (${target.label}): ${JSON.stringify(data).slice(0, 500)}`)

          // Handle standard content + reasoning models:
          // - Standard models: answer in content
          // - GLM: answer in reasoning_content
          // - Qwen3 via Ollama: thinking in reasoning, answer in content (or ALL in reasoning if max_tokens hit)
          // Strategy: try content first, then search all fields for JSON.
          const msg = data.choices?.[0]?.message || {}
          const contentText = msg.content || ""
          const reasoningText = msg.reasoning_content || msg.reasoning || ""

          // Robust JSON extraction:
          //   1. Strip markdown code fences
          //   2. Try parsing the whole stripped text directly (handles nested
          //      objects like memory_scores, and strings containing { or })
          //   3. Fallback to greedy match (whole outer object even with nesting)
          //   4. Fallback to non-greedy multi-match validated last-to-first
          // Tries contentText first (standard models), then reasoningText
          // (thinking models). v0.5.3.1: was non-greedy single-match, which
          // broke on responses containing nested objects (memory_scores) or
          // strings with literal `}` — JSON.parse would fail with Expected '}'.
          function extractJson(raw: string): unknown | null {
            if (!raw) return null
            const stripped = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/, "").trim()
            try { return JSON.parse(stripped) } catch { /* fall through */ }
            const greedy = stripped.match(/\{[\s\S]*\}/)
            if (greedy) {
              try { return JSON.parse(greedy[0]) } catch { /* fall through */ }
            }
            const all = [...stripped.matchAll(/\{[\s\S]*?\}/g)]
            for (let i = all.length - 1; i >= 0; i--) {
              try { return JSON.parse(all[i][0]) } catch { /* try next */ }
            }
            return null
          }

          let result: any = extractJson(contentText)
          if (!result && reasoningText) {
            result = extractJson(reasoningText)
            if (result) debugLog(`scoreExchange: extracted JSON from reasoning field`)
          }
          if (!result) {
            debugLog(`scoreExchange no JSON from ${target.label}: content=${contentText.slice(0, 80)} reasoning=${reasoningText.slice(0, 80)}`)
            break  // bad output — try next model
          }
          const outcome = result.exchange_outcome
          if (!["worked", "failed", "partial", "unknown"].includes(outcome)) {
            debugLog(`scoreExchange invalid outcome from ${target.label}: ${outcome}`)
            break  // bad output — try next model
          }

          const summary = typeof result.exchange_summary === "string" ? result.exchange_summary : ""
          // v0.4.8: noun_tags and facts are NOT in Call 1. Tags extracted server-side at store time.
          // Facts extracted in Call 2 below.

          // v0.3.6: Per-memory scoring from sidecar when available.
          // Falls back to blanket exchange outcome if model didn't return memory_scores.
          const memoryScores: Record<string, string> = {}
          if (memories?.length) {
            const llmScores = result.memory_scores && typeof result.memory_scores === "object"
              ? result.memory_scores as Record<string, string>
              : null

            let perMemoryCount = 0
            for (const mem of memories) {
              const llmScore = llmScores?.[mem.id]
              if (llmScore && ["worked", "failed", "partial", "unknown"].includes(llmScore)) {
                memoryScores[mem.id] = llmScore
                perMemoryCount++
              } else {
                memoryScores[mem.id] = outcome  // fallback to blanket
              }
            }

            if (perMemoryCount > 0) {
              debugLog(`scoreExchange per-memory: ${perMemoryCount}/${memories.length} scored individually`)
              debugLog(`scoreExchange per-memory scores: ${JSON.stringify(memoryScores)}`)
            }
          }

          // Send scoring result to roampal server (all outcomes including unknown)
          if (Object.keys(memoryScores).length > 0) {
            const scoreResp = await fetch(`${ROAMPAL_API_URL}/record-outcome`, {
              method: "POST",
              headers: roampalHeaders(),
              body: JSON.stringify({
                conversation_id: sessionId,
                outcome,
                memory_scores: memoryScores,
                exchange_summary: summary || undefined
                // v0.4.8: noun_tags removed — server extracts tags at store time.
                // Facts sent separately via Call 2 below.
              })
            })

            if (scoreResp.ok) {
              debugLog(`scoreExchange SUCCESS: ${outcome} via ${target.label}, ${Object.keys(memoryScores).length} memories`)
              consecutiveFailures = 0
              lastScorerLabel = target.label
            } else {
              debugLog(`scoreExchange post failed: ${scoreResp.status}`)
            }
          } else {
            debugLog(`scoreExchange SUCCESS: ${outcome} via ${target.label} (no memory scoring — no memories surfaced)`)
            consecutiveFailures = 0
            lastScorerLabel = target.label
          }

          // v0.3.6: Store exchange summary as working memory (matches Claude Code sidecar)
          // v0.5.6 Fix G: Summary write goes through tryStoreSummary(). On failure
          // (timeout, non-2xx, network blip) the payload is queued for the
          // background drainer to retry — same self-heal pattern as Fix F's
          // pendingScoringQueue, just for the summary-store side of the pipeline.
          if (summary) {
            const fingerprint = _summaryFingerprint(exchange)
            const ok = await tryStoreSummary(sessionId, exchange, summary, outcome, fingerprint)
            if (!ok) {
              pendingSummaryQueue.set(sessionId, {
                exchange,
                summary,
                outcome,
                fingerprint,
                retryAttempts: 0
              })
              debugLog(`scoreExchange: summary store failed — queued for deferred retry (session ${sessionId})`)
            }
          }

          // Clear circuit breaker on success — model is healthy again
          modelCircuitBreaker.delete(target.model)

          // v0.4.8: Call 2 — extract_facts (separate LLM call, 30s timeout).
          // Matches benchmark sidecar_extract_facts(). Tags handled server-side at store time.
          // v0.5.4: 3-attempt retry loop matches summary path. Earlier the call was
          // single-shot — the plugin debug log showed "FACTS error (non-fatal):
          // SyntaxError: JSON Parse error: Unexpected token" and "TimeoutError"
          // failures going unrecovered. Same backoff family as scoreExchange's loop.
          try {
            const factsPrompt = `Extract key facts worth remembering from this exchange. Rules:
- Include WHO or WHAT each fact is about — names, projects, topics
- Combine related details into ONE rich fact rather than many fragments
- Include specifics: dates, versions, preferences, decisions, reasons
- Capture what can be inferred, not just what was explicitly said
- ONE fact per line, max 150 characters
- Skip vague feelings, pleasantries, or generic observations

GOOD: "The auth service uses JWT with 24h expiry, needs refresh token rotation added"
GOOD: "User prefers TypeScript over JavaScript and uses Zod for validation"
GOOD: "Chapter 3 draft needs more dialogue per editor feedback, focus on protagonist's childhood"
GOOD: "Lakers won 112-108 on March 5, LeBron scored 34 — user's favorite player"
GOOD: "Sourdough starter day 4, feeds every 12h at room temp, first bake planned for Saturday"
GOOD: "User's daughter Emma starts kindergarten in September, worried about the bus route"
BAD: "They discussed something" (no specifics)
BAD: "It was helpful" (no content)
BAD: "The user asked a question" (meta, not a fact)

User: ${exchange.user.slice(0, 8000)}
Assistant: ${exchange.assistant.slice(0, 8000)}

Return ONLY a JSON object: {"facts": ["fact 1", "fact 2"]}
If no useful facts, return: {"facts": []}`

            const factsHeaders: Record<string, string> = {
              "Content-Type": "application/json",
              "User-Agent": "roampal-sidecar/1.0",
            }
            if (target.key) factsHeaders["Authorization"] = `Bearer ${target.key}`

            // v0.5.4: 3 attempts for fallback (custom sidecar like LM Studio); 1 for Zen.
            // Matches scoreExchange retry policy. Target health is already verified at
            // this point (summary call just succeeded), so retries are about transient
            // response failures, not target failover.
            const factsMaxAttempts = target.isFallback ? 3 : 1
            let extractedFacts: string[] | null = null

            for (let factsAttempt = 0; factsAttempt < factsMaxAttempts; factsAttempt++) {
              try {
                const factsResp = await fetch(`${target.url}/chat/completions`, {
                  method: "POST",
                  headers: factsHeaders,
                  body: JSON.stringify({
                    model: target.model,
                    messages: [
                      { role: "system", content: "/no_think\nExtract key facts. Return ONLY valid JSON with a facts array. No other text." },
                      { role: "user", content: "/no_think\n" + factsPrompt }
                    ],
                    max_tokens: 4000,
                    temperature: 0,
                  }),
                  signal: AbortSignal.timeout(30000)
                })

                if (!factsResp.ok) {
                  // 4xx/5xx — retry once after a small delay; same target.
                  if (factsAttempt < factsMaxAttempts - 1) {
                    const delay = 1000 * Math.pow(2, factsAttempt)  // 1s, 2s
                    debugLog(`scoreExchange FACTS HTTP ${factsResp.status} on ${target.label} — retry ${factsAttempt + 1}/${factsMaxAttempts} in ${delay}ms`)
                    await new Promise(resolve => setTimeout(resolve, delay))
                    continue
                  }
                  debugLog(`scoreExchange FACTS HTTP ${factsResp.status} on ${target.label} — exhausted retries`)
                  break
                }

                const factsData = await factsResp.json()
                const factsMsg = factsData.choices?.[0]?.message || {}
                const factsContent = factsMsg.content || factsMsg.reasoning_content || factsMsg.reasoning || ""
                // v0.5.3.1: Use the same robust extractJson helper as the
                // scoring path. Handles backtick fences, nested objects, and
                // strings containing literal { or }. Was a non-greedy single
                // match that broke on any complex output (Unrecognized token,
                // Expected '}', Unexpected token errors).
                const factsResult: any = extractJson(factsContent)

                if (factsResult && Array.isArray(factsResult.facts)) {
                  extractedFacts = factsResult.facts.filter(
                    (f: unknown) => typeof f === "string" && (f as string).length > 10
                  )
                  break  // success — exit retry loop
                }

                // Parse failure — retry. Local models occasionally return malformed JSON.
                if (factsAttempt < factsMaxAttempts - 1) {
                  const delay = 1000 * Math.pow(2, factsAttempt)
                  debugLog(`scoreExchange FACTS parse failure on ${target.label} — retry ${factsAttempt + 1}/${factsMaxAttempts} in ${delay}ms`)
                  await new Promise(resolve => setTimeout(resolve, delay))
                  continue
                }
                debugLog(`scoreExchange FACTS parse failure on ${target.label} — exhausted retries`)
              } catch (attemptErr) {
                // Timeout or fetch exception — retry.
                if (factsAttempt < factsMaxAttempts - 1) {
                  const delay = 1000 * Math.pow(2, factsAttempt)
                  debugLog(`scoreExchange FACTS error on ${target.label} (attempt ${factsAttempt + 1}/${factsMaxAttempts}): ${attemptErr} — retry in ${delay}ms`)
                  await new Promise(resolve => setTimeout(resolve, delay))
                  continue
                }
                debugLog(`scoreExchange FACTS error on ${target.label} — exhausted retries: ${attemptErr}`)
              }
            }

            if (extractedFacts && extractedFacts.length > 0) {
              // Send facts to server (noun_tags handled by store_working at store time)
              await fetch(`${ROAMPAL_API_URL}/record-outcome`, {
                method: "POST",
                headers: roampalHeaders(),
                body: JSON.stringify({
                  conversation_id: sessionId,
                  outcome: "unknown",
                  memory_scores: {},
                  facts: extractedFacts
                })
              })
              debugLog(`scoreExchange FACTS: ${extractedFacts.length} facts extracted via ${target.label}`)
            }
          } catch (factsErr) {
            debugLog(`scoreExchange FACTS error (non-fatal): ${factsErr}`)
          }

          return true  // Done — exit all loops

        } catch (error) {
          debugLog(`scoreExchange error on ${target.label} (attempt ${attempt}): ${error}`)
          // Timeout errors trip circuit breaker immediately — no point retrying a dead model
          if (error instanceof Error && error.name === "TimeoutError") {
            modelCircuitBreaker.set(target.model, Date.now())
            debugLog(`scoreExchange circuit breaker TRIPPED for ${target.model}`)
            break
          }
          if (attempt >= maxAttempts - 1) break  // exhausted retries — try next model
        }
      }
    }

    debugLog(`scoreExchange FAILED — all models exhausted`)
    // Don't increment consecutiveFailures here — only after deferred retry also fails (session.idle)
    return false
}

// v0.4.8: autoSummarize removed — caused Ollama contention with sidecar scoring.
// Sidecar produces proper summaries; no oversized memories are created.

// ============================================================================
// Plugin Export
// ============================================================================

export const RoampalPlugin: Plugin = async ({ client }) => {
  _pluginClient = client
  debugLog(`Plugin loaded (${ROAMPAL_DEV ? "DEV" : "PROD"} mode, port ${ROAMPAL_PORT})${CUSTOM_SIDECAR_URL ? `, custom sidecar: ${CUSTOM_SIDECAR_MODEL}@${CUSTOM_SIDECAR_URL}` : ""}`)

  // v0.5.5.1: Resolve initial profile via client.project.current() (Desktop) or cwd (CLI).
  // Stored in _cachedProfile and re-resolved in chat.message on every exchange.
  await refreshProfile()

  // v0.5.6 Fix F + G: Start background drainer for pendingScoringQueue and pendingSummaryQueue.
  // Without this, deferred retries only fire on user activity (session.idle).
  if (!SIDECAR_DISABLED && backgroundDrainTimer === null) {
    backgroundDrainTimer = setInterval(() => {
      drainPendingScoringQueue("background.drain").catch(err => {
        debugLog(`background.drain: unexpected scoring error: ${err}`)
      })
      drainPendingSummaryQueue("background.drain").catch(err => {
        debugLog(`background.drain: unexpected summary error: ${err}`)
      })
    }, BACKGROUND_DRAIN_INTERVAL_MS)
    debugLog(`Background queue drainer started (interval=${BACKGROUND_DRAIN_INTERVAL_MS}ms)`)
  }

  // Discover available Zen free models at startup — replaces hardcoded list.
  // If discovery fails, the hardcoded fallback list is used.
  try {
    const resp = await fetch(`${ZEN_FALLBACK_URL}/models`, {
      headers: {
        "Authorization": `Bearer ${ZEN_FALLBACK_KEY}`,
        "User-Agent": "roampal-sidecar/1.0",
      },
      signal: AbortSignal.timeout(3000)
    })
    if (resp.ok) {
      const data = await resp.json()
      const freeModels = (data.data || [])
        .map((m: any) => m.id)
        .filter((id: string) => id && id.includes("free"))
      if (freeModels.length > 0) {
        ZEN_SCORING_MODELS.length = 0
        ZEN_SCORING_MODELS.push(...freeModels)
        debugLog(`Plugin load: discovered ${freeModels.length} Zen free models: ${freeModels.join(", ")}`)
      }
    }
  } catch {
    debugLog(`Plugin load: Zen model discovery failed, using hardcoded list`)
  }

  // v0.4.9.4: Cache agent definitions for subagent filtering (Issue #4).
  // client.app.agents() returns agent list with name + mode ("primary" | "subagent").
  // Cached once at load; refreshed on cache miss in chat.message.
  // Wrapped in timeout — some OpenCode versions may not have this API, and awaiting
  // a non-existent method can hang indefinitely, blocking plugin initialization.
  try {
    const agentPromise = (client as any).app?.agents?.()
    if (agentPromise && typeof agentPromise.then === "function") {
      const agents = await Promise.race([
        agentPromise,
        new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), 3000))
      ])
      if (Array.isArray(agents)) {
        for (const a of agents as any[]) {
          if (a.name && a.mode) cachedAgentModes.set(a.name, a.mode)
        }
        debugLog(`Plugin load: cached ${cachedAgentModes.size} agent modes: ${[...cachedAgentModes.entries()].map(([n, m]) => `${n}=${m}`).join(", ")}`)
      }
    } else {
      debugLog(`Plugin load: client.app.agents not available — subagent filtering will use name heuristic`)
    }
  } catch (err) {
    debugLog(`Plugin load: client.app.agents() failed (${err}) — subagent filtering will use name heuristic`)
  }

  return {
    // ========================================================================
    // Hook 0: Capture provider details for independent scoring
    //
    // chat.params fires on every LLM call. We capture the provider's API key,
    // base URL, and model ID so we can make independent scoring calls later.
    // ========================================================================
    "chat.params": async (
      input: { provider: any; model?: any; agent?: any },
      output: { temperature?: number; topP?: number; topK?: number; options?: Record<string, any> }
    ) => {
      try {
        const provider = input.provider
        if (!provider) return

        // Get API key: try provider.key first, then env vars, then options.apiKey
        let apiKey = provider.key || ""
        if (!apiKey && provider.env?.length) {
          for (const envVar of provider.env) {
            if (process.env[envVar]) {
              apiKey = process.env[envVar]!
              break
            }
          }
        }

        // Get baseURL: provider.options.baseURL, then model.api.url (OpenCode Zen)
        let baseURL = provider.options?.baseURL || input.model?.api?.url || ""
        const modelID = input.model?.id || ""
        const providerID = provider.id || ""

        // OpenCode Zen free provider: apiKey is "public", managed by Zen proxy
        if (!apiKey && provider.options?.apiKey) {
          apiKey = provider.options.apiKey  // e.g. "public"
        }

        if (apiKey && baseURL) {
          const sdk = input.model?.api?.npm || ""
          // Prefer openai-compatible models — they work with /chat/completions.
          // gpt-5-nano (@ai-sdk/openai) returns 0 completion tokens via /chat/completions on Zen.
          // Only overwrite if: no provider yet, OR this model is openai-compatible.
          const isCompatible = sdk.includes("openai-compatible")
          if (!capturedProvider || isCompatible) {
            capturedProvider = { apiKey, baseURL, modelID, providerID, sdk }
            debugLog(`CAPTURED provider: ${providerID} (${modelID}) sdk=${sdk}`)
          }
          // Capture Zen proxy URL for scoring (used by ALL users, even paid)
          if (providerID === "opencode") {
            zenBaseURL = baseURL
            zenApiKey = apiKey || "public"
            debugLog(`CAPTURED Zen: ${zenBaseURL}`)
          }
        }
      } catch (err) {
        debugLog(`chat.params ERROR: ${err}`)
      }
    },

    // ========================================================================
    // Hook 1: Capture user message + fetch context
    //
    // chat.message fires FIRST (before system.transform). We:
    // 1. Store user text for exchange tracking
    // 2. Fetch context from server (includes split scoring_prompt + context_only)
    // 3. Cache for system.transform + messages.transform
    // 4. Cache scoring data for session.idle (sidecar scoring is DEFERRED)
    // NOTE: Do NOT modify output.parts here — it breaks the hook chain.
    // Scoring prompt injection is in system.transform (invisible in UI).
    // ========================================================================
    "chat.message": async (
      input: { sessionID: string; agent?: string; messageID?: string },
      output: { message: Record<string, unknown>; parts: Array<{ type: string; text?: string }> }
    ) => {
      const sessionId = input.sessionID
      const userText = extractTextFromParts(output.parts)

      if (!userText) return

      // v0.5.5.1: Re-resolve profile on every exchange. Desktop plugin is a
      // singleton — process.cwd() and process.env never change across project
      // switches. session.get(sessionID) knows which project this message came
      // from; that's our authoritative source for the active project's worktree.
      await refreshProfile(sessionId)

      // v0.4.9.4: Subagent filtering (Issue #4).
      // Skip context fetch + exchange tracking for subagent sessions.
      // Cache miss: refresh agent list in case agents were added after plugin load.
      if (!ALLOW_SUBAGENTS && input.agent) {
        if (!cachedAgentModes.has(input.agent)) {
          try {
            const agentPromise = (client as any).app?.agents?.()
            if (agentPromise && typeof agentPromise.then === "function") {
              const agents = await Promise.race([
                agentPromise,
                new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), 3000))
              ])
              if (Array.isArray(agents)) {
                cachedAgentModes = new Map()
                for (const a of agents as any[]) {
                  if (a.name && a.mode) cachedAgentModes.set(a.name, a.mode)
                }
                debugLog(`chat.message: refreshed agent cache (${cachedAgentModes.size} agents)`)
              }
            }
          } catch { /* best effort — isSubagent falls back to name heuristic */ }
        }
        if (isSubagent(input.agent)) {
          subagentSessions.add(sessionId)
          debugLog(`chat.message: skipping subagent "${input.agent}" (session ${sessionId})`)
          return
        }
      }

      // v0.5.0: Mark this as a primary session (chat.message fired).
      // Subagent sessions never get chat.message — used by session.idle to detect them.
      primarySessions.add(sessionId)

      // Store user text for exchange tracking
      lastUserMessage.set(sessionId, userText)

      // Store for exchange tracking
      let ctx = sessionContextMap.get(sessionId)
      if (!ctx) {
        ctx = { sessionId, userPrompt: "" }
        sessionContextMap.set(sessionId, ctx)
      }
      ctx.userPrompt = userText

      // Clear assistant tracking from previous exchange
      assistantMessageIds.delete(sessionId)
      assistantTextParts.delete(sessionId)

      // Fetch context from server — context + raw scoring data for sidecar
      const { contextOnly, scoringRequired, scoringPromptSimple, scoringExchange, scoringMemories } = await getContextFromRoampal(sessionId, userText)

      // v0.3.7: Pre-fetch recent exchanges here (runs ONCE per user message)
      // to avoid race in system.transform (OpenCode fires it concurrently for
      // main chat + title generator — async fetch resolves for wrong model).
      // v0.4.7: Fetch most recent exchange summaries (no semantic query — pure recency).
      // Empty query triggers _search_all path which now includes _add_recency_metadata.
      // Search all three collections since summaries can be promoted across tiers.
      let recentExchanges: string | null = null
      const recEntry = includeRecentOnNextTurn.get(sessionId)
      if (recEntry?.flag) {
        includeRecentOnNextTurn.delete(sessionId)
        try {
          const recentResp = await fetch(`${ROAMPAL_API_URL}/search`, {
            method: "POST",
            headers: roampalHeaders(),
            body: JSON.stringify({
              query: "",
              collections: ["working", "history", "patterns"],
              limit: 4,
              sort_by: "recency",
              metadata_filters: { memory_type: "exchange_summary" }
            }),
            // v0.5.4: 5s -> 30s. Background scoreExchange path; same rationale as the
            // summary store bump above.
            signal: AbortSignal.timeout(30000)
          })
          if (recentResp.ok) {
            const data = await recentResp.json() as { results: Array<{ id?: string; text?: string; content?: string; collection?: string; wilson_score?: number; uses?: number; metadata: Record<string, any> }> }
            if (data.results?.length > 0) {
              const lines = data.results
                .map((r) => {
                  const body = r.text || r.content || r.metadata?.text || r.metadata?.content || ""
                  const docId = r.id || ""
                  const collection = r.collection || r.metadata?.collection || "working"
                  const recency = r.metadata?.recency || ""
                  const wilson = r.wilson_score || r.metadata?.wilson_score || 0
                  const uses = r.uses || r.metadata?.uses || 0
                  const lastOutcome = r.metadata?.last_outcome || ""
                  // Match _format_mem() from unified_memory_system.py
                  const tagParts: string[] = []
                  if (recency) tagParts.push(recency)
                  tagParts.push(collection)
                  if (uses > 0) {
                    tagParts.push(`wilson:${Math.round(wilson * 100)}%`)
                    tagParts.push(`used:${uses}x`)
                    if (lastOutcome) tagParts.push(`last:${lastOutcome}`)
                  }
                  const idStr = docId ? ` [id:${docId}]` : ""
                  return `• ${body.slice(0, 200)}${idStr} (${tagParts.join(", ")})`
                })
                .join("\n")
              recentExchanges = `RECENT EXCHANGES (last ${data.results.length}):\n${lines}`
              debugLog(`chat.message: Pre-fetched ${data.results.length} recent exchanges (cold start or compaction recovery):\n${recentExchanges}`)
            }
          }
        } catch (err) {
          debugLog(`chat.message: Failed to pre-fetch recent exchanges: ${err}`)
        }
      }

      // Cache for system.transform + sidecar scoring
      debugLog(`chat.message: scoringRequired=${scoringRequired}, contextLen=${contextOnly?.length || 0}, recentExchanges=${recentExchanges ? 'yes' : 'no'}`)
      cachedContext.set(sessionId, { contextOnly, scoringRequired, scoringPromptSimple, scoringExchange, scoringMemories, recentExchanges, timestamp: Date.now() })

      // Cache scoring data for session.idle — sidecar scoring is DEFERRED until
      // after the main LLM responds. Sidecar is the sole scorer (v0.4.1).
      if (scoringRequired && scoringExchange) {
        pendingScoringData.set(sessionId, {
          userMessage: userText,
          exchange: scoringExchange,
          memories: scoringMemories
        })
        debugLog(`chat.message: Cached scoring data for session.idle (deferred to avoid double-scoring)`)
      }

      // Sidecar fallback: scoring prompt injection handled by system.transform
      // (modifying output.parts here breaks OpenCode's hook chain).
    },

    // ========================================================================
    // Hook 2a: Inject memory context into system prompt (invisible in UI)
    //
    // Uses cached data from chat.message to avoid double-fetching.
    // v0.3.7: contextOnly injection + scoring status tag.
    // When sidecar is broken: IMPORTANT tag suggests `roampal sidecar setup` + scoring prompt injected.
    // ========================================================================
    "experimental.chat.system.transform": async (
      input: { sessionID?: string; model?: any },
      output: { system: string[] }
    ) => {
      const sessionId = input.sessionID
      if (!sessionId) return

      // v0.4.9.4: Skip context injection for subagent sessions (Issue #4).
      if (subagentSessions.has(sessionId)) {
        debugLog(`system.transform: skipping subagent session ${sessionId}`)
        return
      }

      // Inject scoring status tag.
      const cached = cachedContext.get(sessionId)
      
      // v0.5.0: Status tag based on consecutive failure count.
      // 0 = healthy, 1 = transient (silent), 2+ = persistent failure (tell user).
      let scoringStatusTag = ""

      if (consecutiveFailures >= 2) {
        scoringStatusTag = `[roampal scoring: failed (${consecutiveFailures} consecutive failures). Check sidecar config or run "roampal sidecar setup".]`
      } else if (lastScorerLabel) {
        scoringStatusTag = `[roampal scoring: ok via ${lastScorerLabel}]`
      } else {
        scoringStatusTag = `[roampal scoring: initializing]`
      }
      
      if (scoringStatusTag) {
        output.system.push(scoringStatusTag)
      }

      // First-run onboarding — inject once per session to inform model about scoring setup.
      if (!sessionOnboarded.has(sessionId)) {
        sessionOnboarded.add(sessionId)
        const onboardingMsg = CUSTOM_SIDECAR_URL
          ? `[roampal memory active | scoring: ${CUSTOM_SIDECAR_MODEL} via ${CUSTOM_SIDECAR_URL}]`
          : `[roampal memory active | scoring: zen (free, best-effort) | For reliable scoring run: roampal sidecar setup]`
        output.system.push(onboardingMsg)
        debugLog(`system.transform: Injected onboarding message for session ${sessionId}`)
      }

      // v0.3.7: Merge context + recent exchanges into ONE system message.
      // Recent exchanges go AFTER KNOWN CONTEXT — models pay more attention
      // to the end of system messages (recency bias). Putting them last ensures
      // they're not overlooked even by models that skim earlier content.
      if (cached) {
        let fullContext = ""
        if (cached.contextOnly) {
          fullContext += cached.contextOnly
        }
        if (cached.recentExchanges) {
          fullContext += (fullContext ? "\n\n" : "") + cached.recentExchanges
          debugLog(`system.transform: Including recent exchanges in context (cold start/compaction recovery)`)
        }
        if (fullContext) {
          output.system.push(fullContext)
          debugLog(`system.transform: Injected ${fullContext.length} chars context into system prompt (merged)`)
        }

        // v0.4.1: Scoring is sidecar-only on OpenCode. No main LLM fallback.
        // score_memories tool is hidden from MCP, so no scoring prompt needed.
        // DEBUG: dump full system prompt content being sent to model
        debugLog(`system.transform FINAL (cached path): ${output.system.length} system entries, total ${output.system.reduce((a, s) => a + s.length, 0)} chars`)
        for (let i = 0; i < output.system.length; i++) {
          debugLog(`  system[${i}] (${output.system[i].length} chars): ${output.system[i].slice(0, 300)}${output.system[i].length > 300 ? "..." : ""}`)
        }
        return
      }

      // Fallback: if chat.message didn't fire
      const userText = lastUserMessage.get(sessionId) || ""
      const { contextOnly } = await getContextFromRoampal(sessionId, userText)

      if (contextOnly) {
        output.system.push(contextOnly)
        debugLog(`system.transform: Injected ${contextOnly.length} chars context into system prompt (fresh fetch)`)
      }
      // DEBUG: dump full system prompt content being sent to model
      debugLog(`system.transform FINAL (fallback path): ${output.system.length} system entries, total ${output.system.reduce((a, s) => a + s.length, 0)} chars`)
      for (let i = 0; i < output.system.length; i++) {
        debugLog(`  system[${i}] (${output.system[i].length} chars): ${output.system[i].slice(0, 300)}${output.system[i].length > 300 ? "..." : ""}`)
      }
    },

    // ========================================================================
    // Hook 2b: messages.transform — no-op (preserved for future transforms)
    //
    // v0.3.7: Scoring handled entirely by sidecar. No prompt injection needed.
    // The deep clone approach (v0.3.4) is preserved here as a no-op for
    // reference — may be useful for future non-scoring user message transforms.
    // ========================================================================
    "experimental.chat.messages.transform": async (
      input: { sessionID?: string },
      output: { messages: Array<{ info: any; parts: Array<{ type: string; text?: string; [key: string]: any }> }> }
    ) => {
      const sessionId = input.sessionID
      debugLog(`messages.transform: sessionID=${sessionId}, messageCount=${output.messages?.length}, pendingPrompt=0 (moved to system.transform)`)
      // Scoring prompt injection now handled by system.transform
    },

    // ========================================================================
    // Hook 2c: Inject recent exchanges into compaction prompt
    //
    // v0.4.2: Moved from event switch to top-level hook. The SDK defines
    // experimental.session.compacting as a top-level hook with (input, output)
    // signature, NOT an event. Fires BEFORE compaction so injected context
    // is included in the model's compacted summary.
    // ========================================================================
    "experimental.session.compacting": async (
      input: { sessionID?: string },
      output: { context?: string[]; prompt?: string }
    ) => {
      const sid = input.sessionID
      if (!sid) return
      debugLog(`experimental.session.compacting: Injecting recent exchanges into compaction for ${sid}`)
      try {
        // v0.4.7: Fetch most recent exchange summaries (no semantic query — pure recency).
        // Empty query triggers _search_all path which now includes _add_recency_metadata.
        // Search all three collections since summaries can be promoted across tiers.
        const recentResp = await fetch(`${ROAMPAL_API_URL}/search`, {
          method: "POST",
          headers: roampalHeaders(),
          body: JSON.stringify({
            query: "",
            collections: ["working", "history", "patterns"],
            limit: 4,
            sort_by: "recency",
            metadata_filters: { memory_type: "exchange_summary" }
          }),
          // v0.5.4: 5s -> 30s. Background scoreExchange path; same rationale as above.
          signal: AbortSignal.timeout(30000)
        })
        if (recentResp.ok) {
          const data = await recentResp.json() as { results: Array<{ id?: string; text?: string; content?: string; collection?: string; wilson_score?: number; uses?: number; metadata: Record<string, any> }> }
          if (data.results?.length > 0) {
            const recentText = data.results
              .map((r) => {
                const body = r.text || r.content || r.metadata?.text || r.metadata?.content || ""
                const docId = r.id || ""
                const collection = r.collection || r.metadata?.collection || "working"
                const recency = r.metadata?.recency || ""
                const wilson = r.wilson_score || r.metadata?.wilson_score || 0
                const uses = r.uses || r.metadata?.uses || 0
                const lastOutcome = r.metadata?.last_outcome || ""
                const tagParts: string[] = []
                if (recency) tagParts.push(recency)
                tagParts.push(collection)
                if (uses > 0) {
                  tagParts.push(`wilson:${Math.round(wilson * 100)}%`)
                  tagParts.push(`used:${uses}x`)
                  if (lastOutcome) tagParts.push(`last:${lastOutcome}`)
                }
                const idStr = docId ? ` [id:${docId}]` : ""
                return `• ${body.slice(0, 200)}${idStr} (${tagParts.join(", ")})`
              })
              .join("\n")
            if (output.context && Array.isArray(output.context)) {
              output.context.push(`<recent-exchanges>\n${recentText}\n</recent-exchanges>`)
              debugLog(`experimental.session.compacting: Injected ${data.results.length} recent exchanges`)
            } else {
              debugLog(`experimental.session.compacting: output.context not an array (type=${Array.isArray(output.context)}), skipping injection`)
            }
          }
        }
      } catch (err) {
        debugLog(`experimental.session.compacting: Failed to fetch recent exchanges: ${err}`)
      }
      // Recovery flag is set by session.compacted (fires after compaction completes).
      // This hook only handles injection into the compaction summary.
    },

    // ========================================================================
    // Hook 2d: Detect MCP score_memories tool completion
    //
    // v0.4.2: Replaces HTTP round-trip (GET /api/hooks/check-scored) with
    // in-plugin detection. When score_memories fires, set scoredThisTurn so
    // sidecar doesn't double-score on session.idle.
    // ========================================================================
    "tool.execute.after": async (
      input: { tool?: string; sessionID?: string; callID?: string },
      output: { title?: string; output?: string; metadata?: any }
    ) => {
      const toolName = input.tool || ""
      if (toolName.includes("score_memories") || toolName.includes("score_response")) {
        const sid = input.sessionID || ""
        debugLog(`tool.execute.after: Detected scoring tool (${toolName}) for session ${sid}`)
        // Clear pending scoring data — main LLM already scored this exchange
        if (sid && pendingScoringData.has(sid)) {
          pendingScoringData.delete(sid)
          debugLog(`tool.execute.after: Cleared pending sidecar scoring for ${sid} (main LLM scored)`)
        }
      }
    },

    // ========================================================================
    // Hook 3: Handle session lifecycle and message events
    //
    // OpenCode event structure (v1.1.42+):
    //   event.type = "session.created" | "message.updated" | "message.part.updated" |
    //               "session.idle" | "session.deleted" | ...
    //   event.properties = type-specific data
    //
    // Exchange capture flow:
    //   1. chat.message → stores user text
    //   2. message.updated (role=assistant) → tracks assistant messageID
    //   3. message.part.updated (type=text) → accumulates assistant text parts
    //   4. session.idle → sends exchange to /api/hooks/stop
    // ========================================================================
    event: async ({ event }: { event: any }) => {
      const eventType = event?.type
      if (!eventType) return

      switch (eventType) {
        case "session.created": {
          const sid = event.properties?.info?.id
          if (!sid) break
          sessionContextMap.set(sid, { sessionId: sid, userPrompt: "" })
          // Cold start: inject recent exchanges on first message so model
          // has continuity with previous sessions.
          includeRecentOnNextTurn.set(sid, { flag: true, gen: compactionGen })
          debugLog(`Session created: ${sid} (cold start flag set, gen=${compactionGen})`)
          break
        }

        case "message.updated": {
          // Track assistant message IDs so we can filter their text parts
          const info = event.properties?.info
          if (!info?.sessionID || info.role !== "assistant") break

          const sid = info.sessionID
          if (!assistantMessageIds.has(sid)) {
            assistantMessageIds.set(sid, new Set())
          }
          assistantMessageIds.get(sid)!.add(info.id)
          break
        }

        case "message.part.updated": {
          const part = event.properties?.part

          // DEBUG: Log RAW part data BEFORE any guards filter it out
          // Skip reasoning parts — they fire per-token and flood the log (1000s of lines)
          if (part && part.type !== "text" && part.type !== "reasoning") {
            debugLog(`part.updated RAW: type=${part.type}, name=${part.name}, toolName=${part.toolName}, tool=${part.tool}, sessionID=${part.sessionID}, messageID=${part.messageID}, keys=${Object.keys(part).join(",")}`)
          }

          if (!part || !part.sessionID || !part.messageID) break

          const sid = part.sessionID

          // v0.4.2: Cancel pending idle timer — still receiving parts, not truly idle
          if (idleTimers.has(sid)) {
            clearTimeout(idleTimers.get(sid))
            idleTimers.delete(sid)
          }

          // v0.3.7: score_memories detection removed — sidecar is sole scorer

          // Collect text content from assistant message parts
          // TextPart: { id, sessionID, messageID, type: "text", text: string }
          if (part.type !== "text") break

          // Race condition fix: message.part.updated can arrive BEFORE message.updated.
          // Instead of requiring pre-registration, auto-register the assistant message ID
          // from the part event itself. User message parts won't reach here because
          // chat.message clears assistantTextParts before each exchange, and only text
          // parts that accumulate in assistantTextParts get stored.
          if (!assistantMessageIds.has(sid)) {
            assistantMessageIds.set(sid, new Set())
          }
          assistantMessageIds.get(sid)!.add(part.messageID)

          if (!assistantTextParts.has(sid)) {
            assistantTextParts.set(sid, new Map())
          }
          // Overwrite with latest full text (part.text accumulates during streaming)
          assistantTextParts.get(sid)!.set(part.id, part.text || "")
          break
        }

        case "session.idle": {
          // v0.4.2: Debounce idle to avoid acting on subagent completions (OpenCode #13334).
          // If text parts are still arriving, the timer gets cancelled in message.part.updated.
          const sid = event.properties?.sessionID
          if (!sid) break

          // v0.4.9.4: Skip subagent sessions entirely (Issue #4).
          // Subagent sessions were flagged in chat.message where input.agent is available.
          if (subagentSessions.has(sid)) {
            debugLog(`session.idle: skipping subagent session ${sid}`)
            break
          }

          // Cancel any existing timer for this session
          if (idleTimers.has(sid)) {
            clearTimeout(idleTimers.get(sid))
          }

          debugLog(`session.idle: sid=${sid}, debouncing 1.5s`)

          idleTimers.set(sid, setTimeout(async () => {
            idleTimers.delete(sid)

            // v0.5.6: Refresh profile for THIS session before any HTTP call below.
            // Without this, _cachedProfile may belong to a different session's chat.message.
            await refreshProfile(sid)

            const ctx = sessionContextMap.get(sid)
            const textParts = assistantTextParts.get(sid)

            debugLog(`session.idle (debounced): sid=${sid}, userPrompt=${!!ctx?.userPrompt}, textParts=${textParts?.size || 0}, primary=${primarySessions.has(sid)}`)

            // v0.5.0: Skip sessions where chat.message never fired (subagents).
            // OpenCode doesn't fire chat.message for subagent child sessions,
            // so primarySessions won't contain them. This catches subagents even
            // when client.app.agents() is unavailable and name heuristic doesn't match.
            if (!ALLOW_SUBAGENTS && !primarySessions.has(sid)) {
              debugLog(`session.idle: skipping non-primary session ${sid} (no chat.message fired — likely subagent)`)
              assistantMessageIds.delete(sid)
              assistantTextParts.delete(sid)
              return
            }

            if (ctx?.userPrompt && textParts?.size) {
              const assistantText = Array.from(textParts.values()).join("\n")
              // v0.4.8: lifecycle_only=true — stores to JSONL for scoring prompts
              // and marks exchange as completed (triggers scoringRequired on next turn).
              // Does NOT store raw text to ChromaDB. Sidecar stores summary separately.
              try {
                await fetch(`${ROAMPAL_HOOK_URL}/stop`, {
                  method: "POST",
                  headers: roampalHeaders(),
                  body: JSON.stringify({
                    conversation_id: sid,
                    user_message: ctx.userPrompt,
                    assistant_response: assistantText,
                    lifecycle_only: true
                  })
                })
              } catch (err) {
                debugLog(`session.idle: lifecycle stop hook error: ${err}`)
              }
              debugLog(`session.idle: exchange registered (${ctx.userPrompt.length}+${assistantText.length} chars), sidecar will store summary`)

              // Reset for next exchange
              ctx.userPrompt = ""
            }

            // Retry deferred scoring from previous failures FIRST (before current scoring).
            // v0.5.6: Per-session map — multiple sessions can each have pending scores.
            // Fix F + G: Shared drainers also fire from a background interval — see
            // drainPendingScoringQueue/drainPendingSummaryQueue + setInterval at plugin init.
            await drainPendingScoringQueue("session.idle")
            await drainPendingSummaryQueue("session.idle")

            // v0.4.1: Sidecar is the sole scorer on OpenCode. No main LLM fallback.
            const scoringData = pendingScoringData.get(sid)
            if (scoringData) {
              pendingScoringData.delete(sid)

              if (SIDECAR_DISABLED) {
                debugLog(`session.idle: Sidecar DISABLED (testing) — skipping scoring entirely`)
              } else {
                // v0.5.0: No auto-reset. Counter only resets on success (inside scoreExchangeViaLLM).
                // Sidecar retries every exchange regardless of consecutiveFailures — never gives up.
                debugLog(`session.idle: Running sidecar for ${sid} (memories=${scoringData.memories?.length || 0}, consecutiveFailures=${consecutiveFailures})`)
                try {
                  const ok = await scoreExchangeViaLLM(sid, scoringData.userMessage, scoringData.exchange, scoringData.memories)
                  if (!ok) {
                    pendingScoringQueue.set(sid, { ...scoringData, retryAttempts: 0 })
                    debugLog(`session.idle: Sidecar failed — queued for deferred retry`)
                  }
                } catch (err) {
                  pendingScoringQueue.set(sid, { ...scoringData, retryAttempts: 0 })
                  debugLog(`session.idle: Sidecar error — queued for deferred retry: ${err}`)
                }
              }
            }

            // v0.4.8: autoSummarize removed (Ollama contention). Sidecar handles all summarization.

            // Clear state for next exchange
            assistantMessageIds.delete(sid)
            assistantTextParts.delete(sid)
            cachedContext.delete(sid)
            // v0.4.7: Clear compaction recovery flag ONLY if its gen is strictly
            // less than current compactionGen (flag predates this idle). Compaction
            // flags have gen === compactionGen (set after increment) and are NOT
            // cleared here — they are consumed by chat.message pre-fetch instead.
            const recEntry = includeRecentOnNextTurn.get(sid)
            if (recEntry && recEntry.gen < compactionGen) {
              includeRecentOnNextTurn.delete(sid)
              debugLog(`session.idle: Cleared stale recent exchanges flag (gen=${recEntry.gen})`)
            }
          }, 1500))
          break
        }

        // session.compacted fires AFTER compaction completes — set recovery flag
        // so next chat.message pre-fetches recent exchanges.
        case "session.compacted": {
          const sid = event.properties?.sessionID || event.properties?.info?.id
          if (!sid) break
          compactionGen++
          includeRecentOnNextTurn.set(sid, { flag: true, gen: compactionGen })
          debugLog(`session.compacted: Flagged ${sid} for recent exchange injection on next turn (gen=${compactionGen})`)
          break
        }

        case "session.deleted": {
          // Cleanup all session state
          const sid = event.properties?.info?.id
          if (!sid) break
          if (idleTimers.has(sid)) { clearTimeout(idleTimers.get(sid)); idleTimers.delete(sid) }
          sessionContextMap.delete(sid)
          lastUserMessage.delete(sid)
          assistantMessageIds.delete(sid)
          assistantTextParts.delete(sid)
          cachedContext.delete(sid)
          pendingScoringData.delete(sid)
          pendingScoringQueue.delete(sid)  // v0.5.6: cleanup deferred retry debris for destroyed session
          pendingSummaryQueue.delete(sid)  // v0.5.6 Fix G: same cleanup for summary queue
          includeRecentOnNextTurn.delete(sid)
          subagentSessions.delete(sid)  // v0.4.9.4: cleanup subagent tracking
          primarySessions.delete(sid)   // v0.5.0: cleanup primary session tracking
          sessionOnboarded.delete(sid)  // v0.4.0: prevent memory leak
          debugLog(`Session deleted: ${sid}`)
          break
        }
      }
    }
  }
}

// OpenCode requires default export to invoke the plugin function and register hooks
export default RoampalPlugin

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
 *         Compaction recovery via session.compacted + experimental.session.compacting events.
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
const ROAMPAL_HOOK_URL = `http://127.0.0.1:${ROAMPAL_PORT}/api/hooks`

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

// v0.3.7: mainLLMScored removed — sidecar is sole scorer in OpenCode

// v0.3.7: pendingScoringPrompt removed — scoring prompt injected via system.transform when sidecar broken

// Mutex for independent scoring — only one scoring call at a time to avoid 429 pile-ups
let scoringInFlight = false

// User-configurable sidecar — reads from opencode.json MCP environment at load time.
// These env vars are stored in opencode.json > mcp > roampal-core > environment by `roampal sidecar setup`.
// OpenCode passes MCP env vars to MCP server subprocesses but NOT to plugins, so we read
// the config file directly instead of process.env.
// Fallback: process.env still checked for manual/testing overrides.
function _loadSidecarConfig(): { url: string; key: string; model: string; disabled: boolean } {
  try {
    const configPath = join(
      process.env.USERPROFILE || process.env.HOME || ".",
      ".config", "opencode", "opencode.json"
    )
    const raw = readFileSync(configPath, "utf-8")
    const config = JSON.parse(raw)
    const env = config?.mcp?.["roampal-core"]?.environment || {}
    return {
      url: env.ROAMPAL_SIDECAR_URL || process.env.ROAMPAL_SIDECAR_URL || "",
      key: env.ROAMPAL_SIDECAR_KEY || process.env.ROAMPAL_SIDECAR_KEY || "",
      model: env.ROAMPAL_SIDECAR_MODEL || process.env.ROAMPAL_SIDECAR_MODEL || "",
      disabled: (env.ROAMPAL_SIDECAR_DISABLED === "true") || (process.env.ROAMPAL_SIDECAR_DISABLED === "true"),
    }
  } catch {
    // Config not found — fall back to process.env only
    return {
      url: process.env.ROAMPAL_SIDECAR_URL || "",
      key: process.env.ROAMPAL_SIDECAR_KEY || "",
      model: process.env.ROAMPAL_SIDECAR_MODEL || "",
      disabled: process.env.ROAMPAL_SIDECAR_DISABLED === "true",
    }
  }
}
const _sidecarCfg = _loadSidecarConfig()
const CUSTOM_SIDECAR_URL = _sidecarCfg.url
const CUSTOM_SIDECAR_KEY = _sidecarCfg.key
const CUSTOM_SIDECAR_MODEL = _sidecarCfg.model
const SIDECAR_DISABLED = _sidecarCfg.disabled
debugLog(`Sidecar config loaded: custom=${CUSTOM_SIDECAR_URL ? `${CUSTOM_SIDECAR_MODEL} via ${CUSTOM_SIDECAR_URL}` : "none (zen)"}, disabled=${SIDECAR_DISABLED}`)

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
const CIRCUIT_BREAKER_COOLDOWN_MS = 1800000  // 30 minutes — dead Zen models don't recover in 5min

// Deferred retry queue — if scoring fails, store the payload and retry next message.
// Ensures scoring data is NEVER lost even if all models are temporarily down.
let pendingScoring: {
  sessionId: string
  userMessage: string
  exchange: { user: string; assistant: string }
  memories: Array<{ id: string; content: string }> | null
} | null = null

// Pending scoring data — set in chat.message, consumed in session.idle.
// Sidecar scoring is deferred to session.idle so we can check if the main LLM
// already scored (prevents double-scoring memories — GitHub issue #1 follow-up).
const pendingScoringData = new Map<string, {
  userMessage: string
  exchange: { user: string; assistant: string }
  memories: Array<{ id: string; content: string }> | null
}>()

// v0.3.6: Post-compaction flag — inject recent exchanges on next system.transform
const includeRecentOnNextTurn = new Map<string, boolean>()

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

// Tracks whether sidecar scoring is currently broken (all models exhausted twice).
// Used to inject status into context so the model can inform the user.
let scoringBroken = SIDECAR_DISABLED  // Start broken if disabled for testing
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
      headers: { "Content-Type": "application/json" },
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
            headers: { "Content-Type": "application/json" },
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
          headers: { "Content-Type": "application/json" },
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

async function storeExchange(
  sessionId: string,
  userPrompt: string,
  assistantResponse: string
): Promise<void> {
  try {
    const response = await fetch(`${ROAMPAL_HOOK_URL}/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: sessionId,
        user_message: userPrompt,
        assistant_response: assistantResponse
      })
    })

    if (!response.ok) {
      // v0.3.2: Self-healing on 503
      if (response.status === 503) {
        console.error("[roampal] Server unhealthy (503) during exchange store, attempting restart...")
        if (await restartServer()) {
          const retry = await fetch(`${ROAMPAL_HOOK_URL}/stop`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conversation_id: sessionId, user_message: userPrompt, assistant_response: assistantResponse })
          })
          if (!retry.ok) console.error(`[roampal] Retry store exchange failed: ${retry.status}`)
        }
      } else {
        console.error(`[roampal] Failed to store exchange: ${response.status}`)
      }
    }
  } catch (error) {
    // v0.3.2: Self-healing on connection failure
    console.error("[roampal] Failed to store exchange, attempting restart...")
    if (await restartServer()) {
      try {
        const retry = await fetch(`${ROAMPAL_HOOK_URL}/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ conversation_id: sessionId, user_message: userPrompt, assistant_response: assistantResponse })
        })
        if (!retry.ok) console.error(`[roampal] Retry store exchange failed: ${retry.status}`)
      } catch {
        console.error("[roampal] Retry store exchange also failed")
      }
    }
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

function extractTextFromParts(parts: any[]): string {
  if (!Array.isArray(parts)) return ""
  const textPart = parts.find((p: any) => p.type === "text")
  return textPart?.text || ""
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
async function scoreExchangeViaLLM(
  sessionId: string,
  currentUserMessage: string,
  exchange: { user: string; assistant: string },
  memories: Array<{ id: string; content: string }> | null
): Promise<boolean> {
  // Mutex: only one scoring call at a time to avoid 429 pile-ups
  if (scoringInFlight) {
    debugLog(`scoreExchange SKIP — already in flight`)
    return false
  }
  scoringInFlight = true

  try {
    // v0.3.6: First-person prompt matching Claude Code sidecar (sidecar_service.py).
    // Asks for summary, outcome, and per-memory scores in one call.
    // Per-memory scoring uses relevance heuristic — sidecar can judge topic relevance
    // but not actual usage, so irrelevant memories get "unknown" instead of inheriting
    // unearned exchange outcomes. Conservative: defaults to "unknown" when unsure.
    const memorySection = memories?.length
      ? `\nThese memories were injected into your context for this exchange:\n${memories.map(m => `- ${m.id}: "${m.content.slice(0, 200)}"`).join("\n")}\n`
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

    const scoringPrompt = `You are part of a memory system for an AI assistant. You ARE the main AI in this system — write as if you are making a note for your future self.

The user said:
"${exchange.user}"

You responded:
"${exchange.assistant}"

The user then followed up with:
"${currentUserMessage}"
${memorySection}
Respond with ONLY a JSON object:
{ "summary": "<~300 chars>", "outcome": "<worked|failed|partial|unknown>"${memoryScoreSection} }

SUMMARY: Write a first-person note to your future self (~300 chars). Capture the important details — what the user needed, what you did, and any key specifics worth remembering.

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
          // max_tokens 2000: reasoning models (qwen3, glm) burn 500-1000 tokens thinking
          // before producing the JSON answer. 800 was too low — model ran out of tokens.
          const headers: Record<string, string> = { "Content-Type": "application/json" }
          if (target.key) headers["Authorization"] = `Bearer ${target.key}`

          const resp = await fetch(`${target.url}/chat/completions`, {
            method: "POST",
            headers,
            body: JSON.stringify({
              model: target.model,
              messages: [
                // /no_think disables qwen3's thinking mode (avoids burning all tokens on chain-of-thought).
                // Other models ignore it as a harmless prefix. Scoring is simple — doesn't need reasoning.
                { role: "system", content: "/no_think\nYou are part of a memory system. Return ONLY valid JSON with summary, outcome, and memory_scores fields. No other text. Be concise." },
                { role: "user", content: "/no_think\n" + scoringPrompt }
              ],
              max_tokens: 2000,
              temperature: 0,
              // Ollama-native flag to disable thinking/reasoning mode.
              // Works alongside /no_think prefix for belt-and-suspenders reliability.
              // Non-Ollama servers ignore unknown fields.
              think: false
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

          // Try content first (most models), then reasoning (thinking models)
          let jsonMatch = contentText.match(/\{[\s\S]*\}/)
          if (!jsonMatch && reasoningText) {
            // Reasoning models: JSON may be at the end of chain-of-thought.
            // Use last JSON object to skip thinking artifacts.
            const allMatches = [...reasoningText.matchAll(/\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}/g)]
            if (allMatches.length > 0) {
              jsonMatch = allMatches[allMatches.length - 1]  // last JSON object
              debugLog(`scoreExchange: extracted JSON from reasoning field (${allMatches.length} candidates)`)
            }
          }
          if (!jsonMatch) {
            debugLog(`scoreExchange no JSON from ${target.label}: content=${contentText.slice(0, 80)} reasoning=${reasoningText.slice(0, 80)}`)
            break  // bad output — try next model
          }

          const result = JSON.parse(jsonMatch[0])
          const outcome = result.outcome
          if (!["worked", "failed", "partial", "unknown"].includes(outcome)) {
            debugLog(`scoreExchange invalid outcome from ${target.label}: ${outcome}`)
            break  // bad output — try next model
          }

          const summary = typeof result.summary === "string" ? result.summary : ""

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
            const scoreResp = await fetch(`${ROAMPAL_HOOK_URL.replace("/hooks", "")}/record-outcome`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                conversation_id: sessionId,
                outcome,
                memory_scores: memoryScores
              })
            })

            if (scoreResp.ok) {
              debugLog(`scoreExchange SUCCESS: ${outcome} via ${target.label}, ${Object.keys(memoryScores).length} memories`)
              scoringBroken = false
              lastScorerLabel = target.label
            } else {
              debugLog(`scoreExchange post failed: ${scoreResp.status}`)
            }
          } else {
            debugLog(`scoreExchange SUCCESS: ${outcome} via ${target.label} (no memory scoring — no memories surfaced)`)
            scoringBroken = false
            lastScorerLabel = target.label
          }

          // v0.3.6: Store exchange summary as working memory (matches Claude Code sidecar)
          // This creates a separate compact entry alongside the full exchange from storeExchange()
          if (summary) {
            try {
              // Fingerprint: simple hash of first 200 chars of user+assistant (matches cli.py format)
              // cli.py uses MD5 — we use djb2 hash since crypto is heavy in plugin context.
              // Cross-implementation dedup isn't needed (user uses one client per exchange),
              // but format is kept compact (12 chars) for consistent metadata sizing.
              const fpInput = `${exchange.user.slice(0, 200)}:${exchange.assistant.slice(0, 200)}`
              let fpHash = 5381
              for (let i = 0; i < fpInput.length; i++) {
                fpHash = ((fpHash << 5) + fpHash + fpInput.charCodeAt(i)) | 0
              }
              const fingerprint = Math.abs(fpHash).toString(16).padStart(8, '0').slice(0, 12)

              // Dedup check: skip if this exchange was already summarized (matches cli.py)
              try {
                const dedupResp = await fetch(`${ROAMPAL_HOOK_URL.replace('/hooks', '/search')}`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
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
                    debugLog(`scoreExchange SUMMARY skipped — already exists (fingerprint=${fingerprint})`)
                    return true
                  }
                }
              } catch {
                // If dedup check fails, proceed with storing (better duplicate than lost)
              }

              const summaryResp = await fetch(`${ROAMPAL_HOOK_URL}/stop`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
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
                signal: AbortSignal.timeout(5000)
              })

              if (summaryResp.ok) {
                const storeData = await summaryResp.json() as { doc_id?: string }
                const docId = storeData.doc_id || ""
                debugLog(`scoreExchange SUMMARY stored: ${summary.length} chars, doc_id=${docId}`)

                // Score the summary itself with the exchange outcome (matches cli.py)
                if (docId && outcome !== "unknown") {
                  try {
                    await fetch(`${ROAMPAL_HOOK_URL.replace("/hooks", "")}/record-outcome`, {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
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
              } else {
                debugLog(`scoreExchange summary store failed: ${summaryResp.status}`)
              }
            } catch (err) {
              debugLog(`scoreExchange summary store error: ${err}`)
            }
          }

          // Clear circuit breaker on success — model is healthy again
          modelCircuitBreaker.delete(target.model)
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
    // Don't set scoringBroken here — only set it after deferred retry also fails (session.idle)
    return false
  } finally {
    scoringInFlight = false
  }
}

// ============================================================================
// Auto-Summarize Old Memories (v0.3.6 Change 10)
// ============================================================================

/**
 * Summarize one old long memory during idle. Sidecar-first, Zen fallback.
 *
 * 1. Call POST /api/memory/auto-summarize-one on the server
 * 2. If server returns {summarized: true} → done
 * 3. If server returns {summarized: false, reason: "backend_failed"} → fall back to Zen
 * 4. If server returns {summarized: false, reason: "no_candidates"} → silently return
 * 5. On any error: log warning, don't retry (next idle picks it up)
 */
async function autoSummarizeOldMemory(): Promise<void> {
  try {
    const resp = await fetch(`${ROAMPAL_HOOK_URL.replace('/hooks', '/memory/auto-summarize-one')}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(15000)  // 15s — sidecar can be slow
    })

    if (!resp.ok) {
      debugLog(`autoSummarize: server returned ${resp.status}`)
      return
    }

    const data = await resp.json() as {
      summarized: boolean
      reason?: string
      doc_id?: string
      collection?: string
      content?: string
      old_len?: number
      new_len?: number
    }

    if (data.summarized) {
      debugLog(`autoSummarize: SUCCESS ${data.doc_id} (${data.old_len} -> ${data.new_len} chars)`)
      return
    }

    if (data.reason === "no_candidates") {
      // Nothing to summarize — all caught up
      return
    }

    if (data.reason === "backend_failed" && data.content && data.doc_id && data.collection) {
      // Server's sidecar failed — fall back to Zen directly from plugin
      debugLog(`autoSummarize: sidecar failed for ${data.doc_id}, trying Zen fallback`)

      const summarizePrompt = `You are part of a memory system for an AI assistant. You ARE the main AI — write a first-person note to your future self (~300 chars). Capture the important details from this exchange.

Exchange:
"${data.content}"

Respond with ONLY a JSON object:
{"summary": "<~300 chars>"}`

      // Try Zen models (same rotation as scoring)
      const zenModels = [...ZEN_SCORING_MODELS]
      for (const model of zenModels) {
        try {
          const headers: Record<string, string> = { "Content-Type": "application/json" }
          if (zenApiKey) headers["Authorization"] = `Bearer ${zenApiKey}`

          const zenResp = await fetch(`${zenBaseURL}/chat/completions`, {
            method: "POST",
            headers,
            body: JSON.stringify({
              model,
              messages: [
                { role: "system", content: "You are part of a memory system. Return ONLY valid JSON with a summary field." },
                { role: "user", content: summarizePrompt }
              ],
              max_tokens: 800
            }),
            signal: AbortSignal.timeout(8000)
          })

          if (!zenResp.ok) {
            debugLog(`autoSummarize: Zen ${model} returned ${zenResp.status}`)
            continue
          }

          const zenData = await zenResp.json()
          const responseText = zenData.choices?.[0]?.message?.content
            || zenData.choices?.[0]?.message?.reasoning_content
            || ""

          const jsonMatch = responseText.match(/\{[\s\S]*\}/)
          if (!jsonMatch) {
            debugLog(`autoSummarize: Zen ${model} no JSON in response`)
            continue
          }

          const result = JSON.parse(jsonMatch[0])
          const summary = typeof result.summary === "string" ? result.summary : ""
          if (!summary) {
            debugLog(`autoSummarize: Zen ${model} empty summary`)
            continue
          }

          // Enforce length to prevent re-summarization loops
          const finalSummary = summary.length > 400 ? summary.slice(0, 380) + "... [truncated]" : summary

          // Save via update-content endpoint
          const updateResp = await fetch(`${ROAMPAL_HOOK_URL.replace('/hooks', '/memory/update-content')}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              doc_id: data.doc_id,
              collection: data.collection,
              new_content: finalSummary
            }),
            signal: AbortSignal.timeout(5000)
          })

          if (updateResp.ok) {
            debugLog(`autoSummarize: Zen ${model} SUCCESS ${data.doc_id} (${data.old_len} -> ${finalSummary.length} chars)`)
          } else {
            debugLog(`autoSummarize: update-content failed ${updateResp.status}`)
          }
          return  // Done regardless of update success

        } catch (err) {
          debugLog(`autoSummarize: Zen ${model} error: ${err}`)
          continue
        }
      }

      debugLog(`autoSummarize: all Zen models failed for ${data.doc_id}`)
      return
    }

    // Other reasons (memory_not_ready, etc.) — silently skip
    debugLog(`autoSummarize: skipped (${data.reason})`)

  } catch (err) {
    debugLog(`autoSummarize: error: ${err}`)
  }
}

// ============================================================================
// Plugin Export
// ============================================================================

export const RoampalPlugin: Plugin = async ({ client }) => {
  debugLog(`Plugin loaded (${ROAMPAL_DEV ? "DEV" : "PROD"} mode, port ${ROAMPAL_PORT})${CUSTOM_SIDECAR_URL ? `, custom sidecar: ${CUSTOM_SIDECAR_MODEL}@${CUSTOM_SIDECAR_URL}` : ""}`)

  // Discover available Zen free models at startup — replaces hardcoded list.
  // If discovery fails, the hardcoded fallback list is used.
  try {
    const resp = await fetch(`${ZEN_FALLBACK_URL}/models`, {
      headers: { "Authorization": `Bearer ${ZEN_FALLBACK_KEY}` },
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
      let recentExchanges: string | null = null
      if (includeRecentOnNextTurn.get(sessionId)) {
        includeRecentOnNextTurn.delete(sessionId)
        try {
          const recentResp = await fetch(`${ROAMPAL_HOOK_URL.replace('/hooks', '/search')}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              query: "recent exchange",
              collections: ["working"],
              limit: 4,
              sort_by: "recency",
              metadata_filters: { memory_type: "exchange_summary" }
            }),
            signal: AbortSignal.timeout(5000)
          })
          if (recentResp.ok) {
            const data = await recentResp.json() as { results: Array<{ text?: string; content?: string; metadata: { recency?: string; text?: string; content?: string } }> }
            if (data.results?.length > 0) {
              const lines = data.results
                .map((r, i) => {
                  // Search API returns `text` at root (from ChromaDB), or `content` if normalized.
                  // Also check metadata.text/metadata.content as fallbacks.
                  const body = r.text || r.content || r.metadata?.text || r.metadata?.content || ""
                  return `${i + 1}. ${r.metadata?.recency ? `[${r.metadata.recency}] ` : ""}${body.slice(0, 200)}`
                })
                .join("\n")
              recentExchanges = `RECENT EXCHANGES (last ${data.results.length}):\n${lines}`
              debugLog(`chat.message: Pre-fetched ${data.results.length} recent exchanges for cold start`)
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
      // after the main LLM responds. If scoringBroken, model may score first (check-scored).
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

      // Inject scoring status tag.
      const cached = cachedContext.get(sessionId)
      const scoringStatusTag = scoringBroken
        ? `[IMPORTANT — roampal scoring: BROKEN. The background scorer is down. Ask the user if they'd like you to run "roampal sidecar setup" to configure a local scorer. Do NOT run it without their permission. Their memories still work but scoring is paused until a scorer is configured.]`
        : lastScorerLabel
          ? `[roampal scoring: ok via ${lastScorerLabel}]`
          : `[roampal scoring: initializing]`
      output.system.push(scoringStatusTag)

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

        // v0.3.7: Sidecar handles scoring. When broken, inject scoring prompt
        // so the main LLM can score as fallback (+ IMPORTANT tag suggests CLI fix).
        if (scoringBroken && cached.scoringRequired && cached.scoringPromptSimple) {
          output.system.push(cached.scoringPromptSimple)
          debugLog(`system.transform: Sidecar broken — injected scoring prompt (${cached.scoringPromptSimple.length} chars):\n${cached.scoringPromptSimple}`)
        }
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
          // v0.3.6: Cold start — inject recent exchanges on first system.transform
          // Same mechanism as compaction recovery (reuses includeRecentOnNextTurn flag)
          includeRecentOnNextTurn.set(sid, true)
          debugLog(`Session created: ${sid}`)
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
          // Agent finished responding — store exchange + run deferred sidecar scoring
          const sid = event.properties?.sessionID
          if (!sid) break

          const ctx = sessionContextMap.get(sid)
          const textParts = assistantTextParts.get(sid)

          debugLog(`session.idle: sid=${sid}, userPrompt=${!!ctx?.userPrompt}, textParts=${textParts?.size || 0}`)

          if (ctx?.userPrompt && textParts?.size) {
            const assistantText = Array.from(textParts.values()).join("\n")
            await storeExchange(sid, ctx.userPrompt, assistantText)
            debugLog(`Stored exchange for session ${sid}`)

            // Reset for next exchange
            ctx.userPrompt = ""
          }

          // Retry deferred scoring from previous failures FIRST (before current scoring).
          // This ensures failed scoring from exchange N-1 gets retried when exchange N completes.
          if (pendingScoring && !SIDECAR_DISABLED) {
            debugLog(`session.idle: Retrying deferred scoring for ${pendingScoring.sessionId}`)
            try {
              const ok = await scoreExchangeViaLLM(pendingScoring.sessionId, pendingScoring.userMessage, pendingScoring.exchange, pendingScoring.memories)
              if (ok) {
                debugLog(`session.idle: Deferred scoring succeeded`)
                pendingScoring = null
              } else {
                // Two consecutive scoring failures — all models exhausted twice.
                // Set scoringBroken so next system.transform injects visible warning into context.
                scoringBroken = true
                console.warn("[roampal] Memory scoring failed twice in a row — check ROAMPAL_SIDECAR_URL/KEY/MODEL or Zen availability")
                debugLog(`session.idle: Deferred scoring also failed — scoringBroken=true, dropping payload`)
                pendingScoring = null
              }
            } catch (err) {
              scoringBroken = true
              console.warn("[roampal] Memory scoring failed twice in a row — check ROAMPAL_SIDECAR_URL/KEY/MODEL or Zen availability")
              debugLog(`session.idle: Deferred scoring error — scoringBroken=true, dropping payload: ${err}`)
              pendingScoring = null
            }
          }

          // Run sidecar scoring for this exchange.
          // When scoringBroken: check if main model already scored via score_memories tool.
          //   If yes: skip sidecar (avoid double-scoring), but still attempt sidecar next exchange.
          //   If no: run sidecar anyway (auto-recovery attempt).
          // When sidecar not broken: run sidecar normally (sole scorer).
          const scoringData = pendingScoringData.get(sid)
          if (scoringData) {
            pendingScoringData.delete(sid)

            // ROAMPAL_SIDECAR_DISABLED: skip sidecar entirely, let main LLM score via prompt
            if (SIDECAR_DISABLED) {
              debugLog(`session.idle: Sidecar DISABLED (testing) — skipping sidecar, main LLM scores via prompt`)
            } else {
              let modelAlreadyScored = false
              if (scoringBroken) {
                // Check if main model called score_memories this turn
                try {
                  const checkResp = await fetch(`${ROAMPAL_HOOK_URL}/check-scored?conversation_id=${encodeURIComponent(sid)}`)
                  if (checkResp.ok) {
                    const { scored } = await checkResp.json()
                    if (scored) {
                      modelAlreadyScored = true
                      debugLog(`session.idle: Main model scored this turn (scoringBroken mode) — skipping sidecar scoring for THIS exchange`)
                    }
                  }
                } catch {
                  // check-scored failed — proceed with sidecar attempt
                }
              }

              if (modelAlreadyScored) {
                // Model handled scoring. Still probe sidecar for auto-recovery (fire-and-forget, no data).
                // We use the first Zen model as a health check — if it responds, sidecar is back.
                try {
                  const zenProbe = ZEN_SCORING_MODELS[0]
                  const probeResp = await fetch(`${zenBaseURL}/chat/completions`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${zenApiKey}` },
                    body: JSON.stringify({ model: zenProbe, messages: [{ role: "user", content: "ping" }], max_tokens: 1 }),
                    signal: AbortSignal.timeout(5000)
                  })
                  if (probeResp.ok) {
                    scoringBroken = false
                    debugLog(`session.idle: Zen probe succeeded (${zenProbe}) — scoringBroken reset, sidecar will handle next turn`)
                  }
                } catch {
                  // Zen still down — model continues as scorer next turn
                }
              } else {
                debugLog(`session.idle: Running sidecar for ${sid} (memories=${scoringData.memories?.length || 0}, broken=${scoringBroken})`)
                try {
                  const ok = await scoreExchangeViaLLM(sid, scoringData.userMessage, scoringData.exchange, scoringData.memories)
                  if (!ok) {
                    pendingScoring = { sessionId: sid, ...scoringData }
                    debugLog(`session.idle: Sidecar failed — queued for deferred retry`)
                  }
                  // Note: if sidecar succeeds here while scoringBroken=true → scoringBroken resets
                  // in scoreExchangeViaLLM success path. Auto-recovery complete.
                } catch (err) {
                  pendingScoring = { sessionId: sid, ...scoringData }
                  debugLog(`session.idle: Sidecar error — queued for deferred retry: ${err}`)
                }
              }
            }
          }

          // v0.3.6: Auto-summarize one old memory during idle (Change 10)
          // Sidecar-first on server, Zen fallback in plugin. One per idle cycle.
          try {
            await autoSummarizeOldMemory()
          } catch (err) {
            debugLog(`session.idle: autoSummarize error: ${err}`)
          }

          // Clear state for next exchange
          assistantMessageIds.delete(sid)
          assistantTextParts.delete(sid)
          cachedContext.delete(sid)
          break
        }

        // v0.3.6: Compaction recovery — inject recent exchanges on next turn
        case "session.compacted": {
          const sid = event.properties?.sessionID || event.properties?.info?.id
          if (!sid) break
          includeRecentOnNextTurn.set(sid, true)
          debugLog(`session.compacted: Flagged ${sid} for recent exchange injection on next turn`)
          break
        }

        case "experimental.session.compacting": {
          // Fire BEFORE compaction — can inject context INTO the compaction prompt
          // so the model's own summary includes recent exchanges
          const sid = event.properties?.sessionID || event.properties?.info?.id
          if (!sid) break
          debugLog(`experimental.session.compacting: Injecting recent exchanges into compaction for ${sid}`)
          try {
            const recentResp = await fetch(`${ROAMPAL_HOOK_URL.replace('/hooks', '/search')}`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                query: "recent exchange",
                collections: ["working"],
                limit: 4,
                sort_by: "recency",
                metadata_filters: { memory_type: "exchange_summary" }
              }),
              signal: AbortSignal.timeout(5000)
            })
            if (recentResp.ok) {
              const data = await recentResp.json() as { results: Array<{ text?: string; content?: string; metadata: { recency?: string; text?: string; content?: string } }> }
              if (data.results?.length > 0) {
                const recentText = data.results
                  .map((r, i) => {
                    const body = r.text || r.content || r.metadata?.text || r.metadata?.content || ""
                    return `${i + 1}. ${r.metadata?.recency ? `[${r.metadata.recency}] ` : ""}${body.slice(0, 200)}`
                  })
                  .join("\n")
                // Inject into compaction context if the event supports output
                const output = (event as any).output
                if (output?.context && Array.isArray(output.context)) {
                  output.context.push(`<recent-exchanges>\n${recentText}\n</recent-exchanges>`)
                  debugLog(`experimental.session.compacting: Injected ${data.results.length} recent exchanges`)
                }
              }
            }
          } catch (err) {
            debugLog(`experimental.session.compacting: Failed to fetch recent exchanges: ${err}`)
          }
          break
        }

        case "session.deleted": {
          // Cleanup all session state
          const sid = event.properties?.info?.id
          if (!sid) break
          sessionContextMap.delete(sid)
          lastUserMessage.delete(sid)
          assistantMessageIds.delete(sid)
          assistantTextParts.delete(sid)
          cachedContext.delete(sid)
          pendingScoringData.delete(sid)
          includeRecentOnNextTurn.delete(sid)
          debugLog(`Session deleted: ${sid}`)
          break
        }
      }
    }
  }
}

// OpenCode requires default export to invoke the plugin function and register hooks
export default RoampalPlugin

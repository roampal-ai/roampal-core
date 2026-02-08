/**
 * Roampal Plugin for OpenCode
 *
 * Provides persistent memory capabilities through:
 * 1. Context fetch via chat.message hook (caches scoring prompt + context)
 * 2a. Memory context via experimental.chat.system.transform (system prompt, invisible in UI)
 * 2b. Scoring prompt via experimental.chat.messages.transform (deep-cloned user message, invisible in UI)
 * 3. Exchange capture via event hook (session lifecycle + message parts)
 * 4. MCP tools for memory operations (configured separately in opencode.json)
 *
 * Both Claude Code and OpenCode share the same server (27182/27183) to avoid ChromaDB locking.
 *
 * v0.3.4: Fix #1 — scoring prompt injection uses deep clone instead of in-place mutation.
 *         OpenCode holds refs to original message objects for UI rendering; mutating them
 *         caused garbled text in the input box. Clone goes to LLM, original stays in UI.
 *         Also: sidecar scoring deferred from chat.message to session.idle to prevent
 *         double-scoring memories (sidecar only runs if main LLM didn't call score_response).
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

// Debug logging to file (console.log is invisible from plugins)
import { appendFileSync } from "fs"
import { join } from "path"
const DEBUG_LOG = join(process.env.APPDATA || process.env.HOME || ".", "roampal_plugin_debug.log")
function debugLog(msg: string) {
  try {
    appendFileSync(DEBUG_LOG, `[${new Date().toISOString()}] ${msg}\n`)
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

// Track whether main LLM called score_response this turn (per session)
// If true, skip independent scoring — main LLM already provided precise per-memory scores
const mainLLMScored = new Map<string, boolean>()

// Session-independent scoring prompt — messages.transform may not get sessionID from OpenCode,
// so we store the most recent prompt here as a fallback.
let pendingScoringPrompt: string | null = null

// Mutex for independent scoring — only one scoring call at a time to avoid 429 pile-ups
let scoringInFlight = false

// Zen proxy — always used for scoring to save API credits (even for paid users).
// Defaults hardcoded; dynamically updated if "opencode" provider seen in chat.params.
const ZEN_FALLBACK_URL = "https://opencode.ai/zen/v1"
const ZEN_FALLBACK_KEY = "public"
let zenBaseURL = ZEN_FALLBACK_URL
let zenApiKey = ZEN_FALLBACK_KEY

// Free models on Zen, tried in order for scoring. If a model returns 404
// (removed by OpenCode update), the next is tried automatically.
// Verified models: glm-4.7-free (reasoning), kimi-k2.5-free, gpt-5-nano (reasoning)
const ZEN_SCORING_MODELS = ["glm-4.7-free", "kimi-k2.5-free", "gpt-5-nano"]

// Max time (ms) the entire scoring function can block before giving up.
// Prevents UI freezes if models are slow/down. Context injection still works.
const MAX_SCORING_TIME_MS = 8000

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

// Cached context from chat.message for system.transform to use (avoids double-fetch)
// Map<sessionID, { contextOnly, scoringPrompt, scoringPromptSimple, scoringExchange, scoringMemories, timestamp }>
const cachedContext = new Map<string, {
  contextOnly: string
  scoringPrompt: string
  scoringPromptSimple: string
  scoringRequired: boolean
  scoringExchange: { user: string; assistant: string } | null
  scoringMemories: Array<{ id: string; content: string }> | null
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
        const result = execSync("netstat -ano", { timeout: 5000 }).toString()
        for (const line of result.split("\n")) {
          if (line.includes(`127.0.0.1:${port}`) && line.includes("LISTENING")) {
            const pid = line.trim().split(/\s+/).pop()
            if (pid) {
              execSync(`taskkill /pid ${pid} /f`, { timeout: 5000 })
              console.log(`[roampal] Killed stale server process ${pid}`)
            }
            break
          }
        }
      } else {
        const result = execSync(`lsof -ti :${port}`, { timeout: 5000 }).toString().trim()
        if (result) {
          const pid = result.split("\n")[0]
          execSync(`kill -9 ${pid}`, { timeout: 5000 })
          console.log(`[roampal] Killed stale server process ${pid}`)
        }
      }
    } catch {
      // Best effort
    }

    await new Promise(resolve => setTimeout(resolve, 1000))

    // 2. Start fresh server
    const devMode = ROAMPAL_DEV
    const cmd = process.platform === "win32" ? "python" : "python3"
    const args = ["-m", "roampal.server.main", "--port", String(port)]
    if (devMode) args.push("--dev")

    spawn(cmd, args, {
      detached: true,
      stdio: "ignore",
      ...(process.platform === "win32" ? { windowsHide: true } : {})
    }).unref()

    console.log(`[roampal] Starting fresh server on port ${port}`)

    // 3. Poll for health
    const healthUrl = `http://127.0.0.1:${port}/api/health`
    const start = Date.now()
    const timeout = 15000

    while (Date.now() - start < timeout) {
      try {
        const resp = await fetch(healthUrl, { signal: AbortSignal.timeout(2000) })
        if (resp.ok) {
          console.log("[roampal] Server restarted successfully")
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
  // v0.3.2: Split fields — scoring in user message, context in system prompt
  scoring_prompt: string
  scoring_prompt_simple: string  // Simplified scoring prompt for non-Claude models (no XML, plain language)
  context_only: string
  // v0.3.2: Raw scoring data for independent LLM scoring call
  scoring_exchange: { user: string; assistant: string } | null
  scoring_memories: Array<{ id: string; content: string }> | null
}

async function getContextFromRoampal(
  sessionId: string,
  userPrompt: string
): Promise<{
  scoringPrompt: string
  scoringPromptSimple: string
  contextOnly: string
  injection: string
  scoringRequired: boolean
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
              scoringPrompt: data.scoring_prompt || "",
              scoringPromptSimple: data.scoring_prompt_simple || "",
              contextOnly: data.context_only || "",
              injection: data.formatted_injection || "",
              scoringRequired: data.scoring_required || false,
              scoringExchange: data.scoring_exchange || null,
              scoringMemories: data.scoring_memories || null
            }
          }
        }
      }
      console.error(`[roampal] Hook server returned ${response.status}`)
      return { scoringPrompt: "", scoringPromptSimple: "", contextOnly: "", injection: "", scoringRequired: false, scoringExchange: null, scoringMemories: null }
    }

    const data: ContextResponse = await response.json()
    return {
      scoringPrompt: data.scoring_prompt || "",
      scoringPromptSimple: data.scoring_prompt_simple || "",
      contextOnly: data.context_only || "",
      injection: data.formatted_injection || "",
      scoringRequired: data.scoring_required || false,
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
            scoringPrompt: data.scoring_prompt || "",
            scoringPromptSimple: data.scoring_prompt_simple || "",
            contextOnly: data.context_only || "",
            injection: data.formatted_injection || "",
            scoringRequired: data.scoring_required || false,
            scoringExchange: data.scoring_exchange || null,
            scoringMemories: data.scoring_memories || null
          }
        }
      } catch {
        // Retry also failed
      }
    }
    return { scoringPrompt: "", scoringPromptSimple: "", contextOnly: "", injection: "", scoringRequired: false, scoringExchange: null, scoringMemories: null }
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
 * Make an independent LLM call to score the previous exchange outcome.
 * Always uses Zen free models — saves paid users' API credits.
 * Falls through model list if one is unavailable (resilient to Zen updates).
 *
 * Only scores exchange outcome (worked/failed/partial/unknown).
 * Per-memory scoring is handled by the main LLM via score_response MCP tool.
 * If the main LLM doesn't score individual memories, all surfaced memories
 * inherit the exchange outcome as a fallback signal.
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
    const scoringPrompt = `Score this conversation exchange. Return ONLY valid JSON, nothing else.

Previous exchange:
- User asked: "${exchange.user}"
- Assistant answered: "${exchange.assistant}"

User's follow-up: "${currentUserMessage}"

Based on the user's follow-up reaction, what was the outcome?
- "worked" = user satisfied, says thanks, moves on
- "failed" = user corrects, says no/wrong, repeats question
- "partial" = lukewarm, "kind of", takes some but not all
- "unknown" = no clear signal

Return JSON: {"outcome": "worked"}`

    // Always use Zen free models — saves paid users' API credits, avoids rate
    // limit collision with main model. Filter out the main model if it's a Zen model.
    const mainModel = capturedProvider?.modelID || ""
    const modelQueue = ZEN_SCORING_MODELS.filter(m => m !== mainModel)
    if (modelQueue.length === 0) modelQueue.push(...ZEN_SCORING_MODELS)

    // Try each model in the fallback chain. Skip to next on 404/500 (model
    // removed or broken). Retry same model on 429 (rate limit, temporary).
    // Total timeout prevents UI freezes if all models are slow/down.
    const startTime = Date.now()

    for (const scoringModelID of modelQueue) {
      if (Date.now() - startTime > MAX_SCORING_TIME_MS) {
        debugLog(`scoreExchange TIMEOUT — ${Date.now() - startTime}ms elapsed, giving up`)
        break
      }
      debugLog(`scoreExchange trying ${scoringModelID} via ${zenBaseURL}`)

      for (let attempt = 0; attempt < 2; attempt++) {
        if (Date.now() - startTime > MAX_SCORING_TIME_MS) break

        try {
          // All Zen free models use OpenAI-compatible /chat/completions
          // Higher max_tokens (500) for reasoning models that burn tokens thinking
          const resp = await fetch(`${zenBaseURL}/chat/completions`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${zenApiKey}`
            },
            body: JSON.stringify({
              model: scoringModelID,
              messages: [
                { role: "system", content: "You are a scoring classifier. Return ONLY valid JSON." },
                { role: "user", content: scoringPrompt }
              ],
              max_tokens: 500
            }),
            signal: AbortSignal.timeout(6000)
          })

          if (resp.status === 429) {
            const retryAfter = resp.headers.get("retry-after")
            const delay = Math.min(retryAfter ? parseInt(retryAfter) * 1000 : 2000, 2000)
            debugLog(`scoreExchange 429 on ${scoringModelID} — retry ${attempt + 1} in ${delay}ms`)
            await new Promise(resolve => setTimeout(resolve, delay))
            continue  // retry same model
          }

          if (resp.status === 404 || resp.status >= 500) {
            // Model removed or broken — try next model in fallback chain
            debugLog(`scoreExchange ${scoringModelID} returned ${resp.status} — trying next model`)
            break
          }

          if (!resp.ok) {
            debugLog(`scoreExchange ${scoringModelID} returned ${resp.status}`)
            break  // unexpected error — try next model
          }

          const data = await resp.json()
          debugLog(`scoreExchange raw (${scoringModelID}): ${JSON.stringify(data).slice(0, 500)}`)

          // Handle standard content + reasoning models (glm puts answer in reasoning_content)
          const responseText = data.choices?.[0]?.message?.content
            || data.choices?.[0]?.message?.reasoning_content
            || ""

          // Parse JSON from response
          const jsonMatch = responseText.match(/\{[\s\S]*\}/)
          if (!jsonMatch) {
            debugLog(`scoreExchange no JSON from ${scoringModelID}: ${responseText.slice(0, 100)}`)
            break  // bad output — try next model
          }

          const result = JSON.parse(jsonMatch[0])
          const outcome = result.outcome
          if (!["worked", "failed", "partial", "unknown"].includes(outcome)) {
            debugLog(`scoreExchange invalid outcome from ${scoringModelID}: ${outcome}`)
            break  // bad output — try next model
          }

          // All surfaced memories inherit the exchange outcome as fallback
          const memoryScores: Record<string, string> = {}
          if (memories?.length) {
            for (const mem of memories) {
              memoryScores[mem.id] = outcome
            }
          }

          // Send scoring result to roampal server
          const scoreResp = await fetch(`${ROAMPAL_HOOK_URL.replace("/hooks", "")}/record-outcome`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              conversation_id: sessionId,
              outcome,
              memory_scores: Object.keys(memoryScores).length > 0 ? memoryScores : null
            })
          })

          if (scoreResp.ok) {
            debugLog(`scoreExchange SUCCESS: ${outcome} via ${scoringModelID}, ${Object.keys(memoryScores).length} memories`)
          } else {
            debugLog(`scoreExchange post failed: ${scoreResp.status}`)
          }
          return true  // Done — exit all loops

        } catch (error) {
          debugLog(`scoreExchange error on ${scoringModelID} (attempt ${attempt}): ${error}`)
          if (attempt === 1) break  // exhausted retries — try next model
        }
      }
    }

    debugLog(`scoreExchange FAILED — all models exhausted`)
    return false
  } finally {
    scoringInFlight = false
  }
}

// ============================================================================
// Plugin Export
// ============================================================================

export const RoampalPlugin: Plugin = async ({ client }) => {
  console.log(`[roampal] Plugin loaded (${ROAMPAL_DEV ? "DEV" : "PROD"} mode, port ${ROAMPAL_PORT})`)

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
    // NOTE: We do NOT modify output.parts — any changes are visible in the UI.
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
      // (mainLLMScored is cleared in session.idle after sidecar check)
      assistantMessageIds.delete(sessionId)
      assistantTextParts.delete(sessionId)

      // Fetch context from server — get split scoring_prompt + context_only + raw scoring data
      const { scoringPrompt, scoringPromptSimple, contextOnly, scoringRequired, scoringExchange, scoringMemories } = await getContextFromRoampal(sessionId, userText)

      // Cache for system.transform + messages.transform + independent scoring
      debugLog(`chat.message: scoringRequired=${scoringRequired}, scoringPromptLen=${scoringPrompt?.length || 0}, simpleLen=${scoringPromptSimple?.length || 0}, contextLen=${contextOnly?.length || 0}`)
      cachedContext.set(sessionId, { contextOnly, scoringPrompt, scoringPromptSimple, scoringRequired, scoringExchange, scoringMemories, timestamp: Date.now() })

      // Store scoring prompt in session-independent var for messages.transform
      // (OpenCode doesn't always pass sessionID to messages.transform)
      pendingScoringPrompt = scoringPromptSimple || scoringPrompt || null

      // Cache scoring data for session.idle — sidecar scoring is DEFERRED until
      // after the main LLM responds so we can check mainLLMScored first.
      // This prevents double-scoring: if the main LLM calls score_response with
      // per-memory scores, the sidecar's uniform scoring is skipped entirely.
      if (scoringRequired && scoringExchange) {
        pendingScoringData.set(sessionId, {
          userMessage: userText,
          exchange: scoringExchange,
          memories: scoringMemories
        })
        debugLog(`chat.message: Cached scoring data for session.idle (deferred to avoid double-scoring)`)
      }
    },

    // ========================================================================
    // Hook 2a: Inject memory context into system prompt (invisible in UI)
    //
    // Uses cached data from chat.message to avoid double-fetching.
    // Only context goes here — scoring prompt goes in messages.transform
    // via deep clone to avoid mutating UI-visible objects (GitHub issue #1).
    // ========================================================================
    "experimental.chat.system.transform": async (
      input: { sessionID?: string; model?: any },
      output: { system: string[] }
    ) => {
      const sessionId = input.sessionID
      if (!sessionId) return

      const cached = cachedContext.get(sessionId)

      if (cached) {
        if (cached.contextOnly) {
          output.system.push(cached.contextOnly)
          debugLog(`system.transform: Injected ${cached.contextOnly.length} chars context into system prompt (cached)`)
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
    },

    // ========================================================================
    // Hook 2b: Inject scoring prompt into message history (invisible in UI)
    //
    // experimental.chat.messages.transform fires right before the LLM API call.
    // We DEEP CLONE the user message before modifying it — OpenCode holds
    // direct references to the original objects for UI rendering, so mutating
    // in place causes garbled text in the input box (GitHub issue #1).
    // The clone goes to the LLM; the original stays untouched in the UI.
    // ========================================================================
    "experimental.chat.messages.transform": async (
      input: { sessionID?: string },
      output: { messages: Array<{ info: any; parts: Array<{ type: string; text?: string; [key: string]: any }> }> }
    ) => {
      const sessionId = input.sessionID
      debugLog(`messages.transform: sessionID=${sessionId}, messageCount=${output.messages?.length}, pendingPrompt=${pendingScoringPrompt?.length || 0}`)

      // Try session-keyed cache first, fall back to session-independent variable
      let prompt: string | null = null
      if (sessionId) {
        const cached = cachedContext.get(sessionId)
        prompt = cached?.scoringPromptSimple || cached?.scoringPrompt || null
      }
      if (!prompt) {
        prompt = pendingScoringPrompt
      }

      if (!prompt) return

      // Find the last user message and inject via DEEP CLONE (not mutation)
      for (let i = output.messages.length - 1; i >= 0; i--) {
        const msg = output.messages[i]
        if (msg.info?.role === "user") {
          const textPartIndex = msg.parts.findIndex((p: any) => p.type === "text" && p.text)
          if (textPartIndex >= 0) {
            // Deep clone the entire message to avoid mutating UI-visible objects
            const cloned = JSON.parse(JSON.stringify(msg))
            cloned.parts[textPartIndex].text = prompt + "\n\n" + cloned.parts[textPartIndex].text
            output.messages[i] = cloned
            debugLog(`messages.transform: INJECTED ${prompt.length} chars via clone into user message`)
            pendingScoringPrompt = null
          }
          break
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
          console.log(`[roampal] Session created: ${sid}`)
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
          if (!part || !part.sessionID || !part.messageID) break

          const sid = part.sessionID

          // Track tool invocations — detect if main LLM called score_response
          // If so, we skip independent scoring (main LLM provided precise per-memory scores)
          if (part.type === "tool-invocation" && part.name === "score_response") {
            mainLLMScored.set(sid, true)
            console.log(`[roampal] Main LLM called score_response for session ${sid}`)
            break
          }

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

          debugLog(`session.idle: sid=${sid}, userPrompt=${!!ctx?.userPrompt}, textParts=${textParts?.size || 0}, mainLLMScored=${mainLLMScored.get(sid) || false}`)

          if (ctx?.userPrompt && textParts?.size) {
            const assistantText = Array.from(textParts.values()).join("\n")
            await storeExchange(sid, ctx.userPrompt, assistantText)
            console.log(`[roampal] Stored exchange for session ${sid}`)

            // Reset for next exchange
            ctx.userPrompt = ""
          }

          // Retry deferred scoring from previous failures FIRST (before current scoring).
          // This ensures failed scoring from exchange N-1 gets retried when exchange N completes.
          if (pendingScoring) {
            debugLog(`session.idle: Retrying deferred scoring for ${pendingScoring.sessionId}`)
            try {
              const ok = await scoreExchangeViaLLM(pendingScoring.sessionId, pendingScoring.userMessage, pendingScoring.exchange, pendingScoring.memories)
              if (ok) {
                debugLog(`session.idle: Deferred scoring succeeded`)
                pendingScoring = null
              }
            } catch (err) {
              debugLog(`session.idle: Deferred scoring still failing: ${err}`)
            }
          }

          // Sidecar scoring for CURRENT exchange — ONLY if main LLM did NOT call score_response.
          // This prevents double-scoring: main LLM provides per-memory scores,
          // sidecar provides uniform fallback. Never both.
          const scoringData = pendingScoringData.get(sid)
          if (scoringData) {
            pendingScoringData.delete(sid)

            if (mainLLMScored.get(sid)) {
              debugLog(`session.idle: Main LLM scored — skipping sidecar for ${sid}`)
            } else {
              // Main LLM didn't score — run sidecar with uniform scoring
              debugLog(`session.idle: Main LLM did NOT score — running sidecar for ${sid}`)
              try {
                const ok = await scoreExchangeViaLLM(sid, scoringData.userMessage, scoringData.exchange, scoringData.memories)
                if (!ok) {
                  pendingScoring = { sessionId: sid, ...scoringData }
                  debugLog(`session.idle: Sidecar failed — queued for deferred retry`)
                }
              } catch (err) {
                pendingScoring = { sessionId: sid, ...scoringData }
                debugLog(`session.idle: Sidecar error — queued for deferred retry: ${err}`)
              }
            }
          }

          // Clear state for next exchange
          assistantMessageIds.delete(sid)
          assistantTextParts.delete(sid)
          mainLLMScored.delete(sid)
          cachedContext.delete(sid)
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
          mainLLMScored.delete(sid)
          cachedContext.delete(sid)
          pendingScoringData.delete(sid)
          console.log(`[roampal] Session deleted: ${sid}`)
          break
        }
      }
    }
  }
}

// OpenCode requires default export to invoke the plugin function and register hooks
export default RoampalPlugin

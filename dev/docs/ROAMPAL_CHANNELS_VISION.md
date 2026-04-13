# Roampal Channels

**One memory. Every conversation. It learns what works.**

## Vision

Roampal Channels extends the memory engine to work across chat platforms - Discord, WhatsApp, Telegram, and more. Instead of building a custom UI, we use the apps people already have as the interface.

The chat apps ARE the desktop app.

## Why This Matters

| Traditional Approach | Roampal Channels |
|---------------------|------------------|
| Build custom UI | Use Discord/WhatsApp as UI |
| Users learn new app | Users stay in familiar apps |
| Another window to manage | Memory lives where they already chat |
| Compete with established apps | Complement them |

## Installation (Target UX)

### For Discord

```bash
# 1. Install Roampal
pip install roampal

# 2. Initialize with Discord channel
roampal init --channel discord

# 3. Provide your API keys
Enter your Anthropic API key: sk-ant-...
Enter your Discord bot token: MTI...

# 4. Done - bot joins your server
Roampal Discord bot is running!
Invite link: https://discord.com/oauth2/authorize?client_id=...
```

### For WhatsApp

```bash
roampal init --channel whatsapp

# Scan QR code with WhatsApp
# Bot connects to your account
```

### For Multiple Channels (Same Memory)

```bash
# Already have Claude Code setup? Add Discord:
roampal channel add discord

# Your Discord bot shares memory with Claude Code
```

## How It Works

```
User sends message (Discord/WhatsApp/etc)
         |
         v
+------------------+
| Channel Adapter  |  <-- Formats for platform
+------------------+
         |
         v
+------------------+
| Roampal Core     |  <-- Search memory, inject context
| (existing)       |
+------------------+
         |
         v
+------------------+
| LLM (BYOK)       |  <-- User's own API key
+------------------+
         |
         v
+------------------+
| Channel Adapter  |  <-- Format response for platform
+------------------+
         |
         v
Response sent to user
```

## Outcome Scoring Per Channel

| Channel | How Users Score |
|---------|-----------------|
| Claude Code | `score_response()` tool call |
| Discord | React with thumbs up/down |
| WhatsApp | Reply "that worked" / "nope" |
| Telegram | Inline buttons |

All scoring feeds into the same Wilson-scored memory system.

## Cost Model: BYOK (Bring Your Own Key)

**User provides:**
- Anthropic/OpenAI API key (for LLM)
- Discord bot token (free to create)
- WhatsApp Business API (or personal via bridge)

**Roampal provides:**
- Memory storage (local ChromaDB)
- Scoring system
- Context injection
- Channel adapters

**User's cost:** Their existing LLM subscription + ~$0 for Roampal
**Our cost:** Zero per user (everything runs locally)

## Local Model Support

BYOK isn't just cloud APIs - it means ANY LLM backend, including fully local:

### Supported Providers

```bash
# Cloud providers
roampal init --channel discord --provider anthropic
roampal init --channel discord --provider openai

# Local models
roampal init --channel discord --provider ollama --model llama3
roampal init --channel discord --provider ollama --model mistral

# OpenAI-compatible (LM Studio, llama.cpp, vLLM, etc)
roampal init --channel discord --provider openai-compatible \
  --base-url http://localhost:1234/v1
```

### Cost Comparison

| Provider | Cost | Privacy | Speed |
|----------|------|---------|-------|
| Claude API | ~$15/MTok | Cloud | Fast |
| OpenAI | ~$10/MTok | Cloud | Fast |
| Ollama (local) | $0 | 100% local | Hardware dependent |
| LM Studio | $0 | 100% local | Hardware dependent |

### The Full Local Stack

For maximum privacy, everything can run locally:

- **Memory:** ChromaDB (local)
- **Embeddings:** sentence-transformers (local)
- **LLM:** Ollama/llama.cpp (local)
- **Chat UI:** Discord (browser) or Terminal

**Zero cloud. Zero API costs. Zero data leaving your machine.**

### Target Audience

> "Your AI assistant that never phones home."

- r/LocalLLaMA community
- Enterprise users with data sovereignty requirements
- Privacy-conscious individuals
- Developers who want to experiment without API costs
- Users in regions with limited cloud API access

### Implementation Notes

The LLM provider is abstracted behind a simple interface:

```python
class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, messages: List[Message]) -> str:
        """Send messages to LLM, get response"""

class AnthropicProvider(LLMProvider):
    async def complete(self, messages):
        return await self.client.messages.create(...)

class OllamaProvider(LLMProvider):
    async def complete(self, messages):
        return await ollama.chat(model=self.model, messages=messages)

class OpenAICompatibleProvider(LLMProvider):
    async def complete(self, messages):
        # Works with any OpenAI-compatible API
        return await self.client.chat.completions.create(...)
```

This abstraction means adding new providers is trivial.

## Architecture Changes Required

### Phase 1: Core Refactoring

1. **User isolation**
   - Add `user_id` parameter to memory system
   - Collection naming: `roampal_{user_id}_{collection}`
   - Session tracking: `{channel}:{user}:{session}`

2. **Channel abstraction**
   ```python
   class ChannelAdapter(ABC):
       @abstractmethod
       async def format_context(self, memories: List[Memory]) -> str:
           """Format memories for this platform"""

       @abstractmethod
       async def detect_outcome(self, message: str) -> Optional[str]:
           """Detect scoring signal from user message"""
   ```

3. **Webhook API**
   ```
   POST /api/channel/{channel_type}/{user_id}/message
   POST /api/channel/{channel_type}/{user_id}/reaction
   ```

### Phase 2: Discord Channel

- Discord.py bot
- DM and server channel support
- Reaction-based scoring
- Slash commands for memory management

### Phase 3: WhatsApp/Telegram

- Same pattern, different adapter
- Platform-specific auth flows

## CLI Commands (Target)

```bash
# Setup
roampal init                      # Claude Code (existing)
roampal init --channel discord    # Discord bot
roampal init --channel whatsapp   # WhatsApp bridge

# Manage channels
roampal channel list              # Show active channels
roampal channel add discord       # Add channel to existing setup
roampal channel remove whatsapp   # Remove channel

# All channels share the same memory
roampal search "auth patterns"    # Works across all channel memories
```

## The Pitch

> "Clawdbot lets you talk to Claude everywhere.
> Roampal makes Claude remember you everywhere - and learn from every conversation."

We're not building another AI assistant. We're building the memory layer that makes ANY AI assistant actually personal.

## Competitive Positioning: Better Clawdbot

**Decision:** Build Roampal Channels as a Clawdbot competitor, not a plugin.

### Why Compete, Not Integrate

| Clawdbot Plugin | Roampal Channels |
|-----------------|------------------|
| Quick to ship | More work |
| Their audience | Build your own |
| Dependent on their roadmap | Full control |
| Memory is a feature | Memory is THE product |
| No native scoring | Scoring is core |

### What Makes Us Better

| Clawdbot | Roampal Channels |
|----------|------------------|
| Memory stores things | Memory **learns** what works |
| No outcome tracking | 👍/👎 → Wilson scoring |
| Same advice forever | Bad advice demoted, good promoted |
| Manual context files (SOUL.md) | Automatic context injection |
| 8.9k users, established | Underdog with better core tech |

### The Differentiator

> "Clawdbot remembers. Roampal **learns**."

Every conversation makes Roampal smarter. Bad advice gets demoted. Good patterns get promoted. The memory system has natural selection built in.

## Discord Bot Capabilities

The Discord bot will have **full parity** with Roampal Desktop/Core features:

### Context Injection (Same as Desktop)

| Feature | How It Works in Discord |
|---------|------------------------|
| **Cold Start Detection** | New user (no memory_bank identity) → "I don't know you yet, tell me about yourself" |
| **Organic Recall** | User message triggers semantic search → relevant memories injected into system prompt |
| **Last 4 Exchanges** | Bot tracks conversation in `working` collection → injects recent context |
| **Scoring Prompt** | After bot response, prompt user to react 👍/👎 for learning |

### Message Flow

```
User: "How should I handle auth?"
              ↓
┌─────────────────────────────────────────────┐
│ 1. COLD START CHECK                         │
│    Is user in memory_bank? If no → new user │
│    prompt                                   │
├─────────────────────────────────────────────┤
│ 2. ORGANIC RECALL                           │
│    Search patterns/history for "auth"       │
│    → Found: "JWT 15min (92% worked)"        │
├─────────────────────────────────────────────┤
│ 3. LAST 4 EXCHANGES                         │
│    Get recent working memory for this       │
│    session                                  │
├─────────────────────────────────────────────┤
│ 4. BUILD SYSTEM PROMPT                      │
│    [Known Context] + [Recent Exchanges]     │
│    + [User Message]                         │
├─────────────────────────────────────────────┤
│ 5. CALL LLM                                 │
│    Send to Anthropic/OpenAI/Ollama          │
├─────────────────────────────────────────────┤
│ 6. STORE EXCHANGE                           │
│    Save to working collection               │
├─────────────────────────────────────────────┤
│ 7. PROMPT SCORING                           │
│    "React 👍 if helpful, 👎 if not"         │
└─────────────────────────────────────────────┘
              ↓
Bot: "For auth, JWT with 15min expiry has worked well
     for you before (92% success rate). Want me to
     apply that pattern?"
```

### Data Parity with Desktop

| Desktop Feature | Discord Equivalent |
|-----------------|-------------------|
| `get_context_insights()` | Auto-injected on every message |
| `search_memory()` | `/search [query]` or natural language |
| `add_to_memory_bank()` | `/remember [fact]` or "remember that..." |
| `score_response()` | 👍/👎 reactions |
| `record_response()` | Automatic on significant exchanges |
| Cold start profile | DM onboarding for new users |
| Working memory (session) | Per-channel or per-DM context |
| Books collection | `/upload` for documents |

### Session Management

```python
# Session ID format for Discord
session_id = f"discord:{user_id}:{channel_id}"

# DM sessions
session_id = f"discord:{user_id}:dm"

# Server channel sessions
session_id = f"discord:{user_id}:{guild_id}:{channel_id}"
```

**Same memory, different channels.** A pattern learned in Discord DMs applies when you use Claude Code.

## Phone Use Case (Remote Control)

When the user is on their phone messaging the bot:

```
┌──────────────────────────────────────────────────────────────┐
│  PHONE (Discord app)          COMPUTER (Roampal bot)         │
│  ────────────────────         ───────────────────────        │
│                                                              │
│  "What did we discuss      →   Memory search                 │
│   about the API?"              (local ChromaDB)              │
│                                                              │
│  "Read my config file"     →   Tool execution                │
│                                (MCP on computer)             │
│                                                              │
│  Phone = remote control        Computer = brain + hands      │
└──────────────────────────────────────────────────────────────┘
```

### Two Operating Modes

| Mode | When | What Works | What Doesn't |
|------|------|------------|--------------|
| **Memory-only** | Bot infers from request | Chat, recall, learning | File access, commands |
| **Full agent** | Explicit tool request | Everything | Requires bot on your machine |

**Default: Memory mode.** The bot infers from the request:
- "What did we decide about auth?" → Memory search (works from anywhere)
- "Read my package.json" → Tool execution (requires bot on that machine)

### The Experience

You're on the train, message Discord:
> "Remind me what we were working on yesterday"

Bot pulls from `working` + `history`, responds with context. Pure memory mode - works from anywhere because memory is just text.

Get home, message:
> "Open that file we were editing and show me line 50"

Now it executes tools because you're implicitly asking it to act on your computer.

### No Phone Setup Required

| Component | Where It Lives |
|-----------|----------------|
| Discord app | Phone (just a chat client) |
| Memory (ChromaDB) | Computer running bot |
| Tools (MCP) | Computer running bot |
| LLM calls | Computer → API (or local) |

**Phone just sends text. All intelligence lives on the computer.**

## Security & Data Transmission

### Threat Model

```
┌─────────────────────────────────────────────────────────────────────┐
│ User's Phone    →    Discord Servers    →    User's Computer        │
│ (untrusted)          (untrusted)             (trusted)              │
└─────────────────────────────────────────────────────────────────────┘
```

| Component | Trust Level | What It Sees |
|-----------|-------------|--------------|
| Phone | Untrusted | Message text only |
| Discord/WhatsApp servers | Untrusted | Message text (transit) |
| User's computer | Trusted | Everything (memory, tools, keys) |

### What Crosses the Wire

| Data | Encrypted in Transit? | Stored by Platform? | Mitigation |
|------|----------------------|---------------------|------------|
| User messages | Yes (TLS) | Yes (Discord stores) | Don't send secrets in chat |
| Bot responses | Yes (TLS) | Yes (Discord stores) | Truncate sensitive output |
| Memory content | **Never leaves machine** | No | Local ChromaDB |
| API keys | **Never leaves machine** | No | Stored in local config |
| Tool results | Yes (TLS) | Yes (in response) | Sanitize file paths |

### The Good News

**Memory is 100% local.** Your ChromaDB with all patterns, history, preferences - never transmitted. When you ask "what did we discuss about auth?":

1. Message goes to Discord (they see the question)
2. Bot on your machine searches local memory
3. LLM generates response using local context
4. Response goes back through Discord

Discord sees the question and answer, but **not your full memory corpus**.

### The Concerns

| Risk | Severity | Mitigation |
|------|----------|------------|
| Discord stores messages | Medium | They already store your DMs. Nothing new. |
| Man-in-the-middle | Low | TLS encryption standard |
| Bot token compromise | High | Store securely, rotate periodically |
| LLM sees sensitive data | Medium | BYOK means you control the provider |
| Tool output leakage | Medium | Sanitize before sending to chat |

### Tool Output Sanitization

When tools execute, results might contain sensitive paths:

```python
# Raw tool output (risky)
"Contents of C:/Users/john/Documents/secrets/api-keys.txt: ..."

# Sanitized output (safer)
"Contents of ~/Documents/secrets/api-keys.txt: ..."
```

**Best practices:**
- Truncate large outputs (max 2000 chars)
- Replace absolute paths with relative
- Never auto-output `.env` or credential files
- Require confirmation for sensitive reads

### Full Privacy Mode (Local LLM)

For maximum privacy, run fully local:

| Component | Cloud Option | Local Option |
|-----------|--------------|--------------|
| Chat interface | Discord | Terminal/local web UI |
| LLM | Anthropic API | Ollama/llama.cpp |
| Embeddings | OpenAI | sentence-transformers |
| Memory | ChromaDB | ChromaDB (already local) |

**Result: Zero data leaves your machine.**

### Comparison to Desktop

| Security Property | Desktop (localhost) | Channels (Discord) |
|-------------------|--------------------|--------------------|
| Memory storage | Local | Local |
| Message transit | None | Platform servers |
| LLM calls | Direct to API | Direct to API |
| Tool execution | Local | Local |
| Message persistence | Your machine | Platform + your machine |

**Channels adds one exposure point:** message content transits through chat platform. But memory, tools, and keys stay local.

### Recommendations for Users

1. **Don't paste secrets in chat** - Discord stores everything
2. **Use DMs for sensitive topics** - Still stored, but not visible to server admins
3. **Rotate bot tokens periodically** - Treat like any API key
4. **Consider local LLM for sensitive work** - Ollama + local memory = zero cloud
5. **Review tool outputs** - Bot should confirm before sending file contents

## Open Questions

1. **WhatsApp Business API vs bridges** - Official API requires business verification. Bridges (like whatsapp-web.js) are grayer but more accessible.

2. **Multi-user servers** - Discord servers have multiple users. Per-user memory? Per-server memory? Both?

3. **Privacy** - Users might not want their Discord convos in the same memory as their work (Claude Code). Channel isolation option?

4. **Rate limiting** - Platforms rate limit bots. How do we handle gracefully?

5. **End-to-end encryption** - Could we encrypt memory results before sending through Discord, decrypt on phone? Adds friction but max privacy.

## Clawdbot Security Failures (Lessons Learned)

> "Learn from the competition's mistakes."

Clawdbot (now rebranded as Moltbot) experienced major security breaches in late 2025/early 2026. These failures are directly relevant to Roampal Channels architecture.

### What Went Wrong

| Failure | Technical Cause | Impact |
|---------|-----------------|--------|
| **Authentication bypass** | Localhost auto-trusted + reverse proxy ignored X-Forwarded-For | Hundreds of control panels exposed |
| **Default insecure config** | `trusted_proxies` empty by default | All external connections treated as local |
| **No privilege separation** | Bot running as root | RCE with full system access |
| **Centralized secrets** | API keys, OAuth tokens, bot tokens all in one config | Single breach = total compromise |
| **WebSocket bypass** | Direct WS connection skipped auth checks | Config + conversations leaked |
| **Prompt injection** | No input sanitization on emails/messages | Private crypto keys extracted in 5 minutes |
| **Credential storage** | Signal pairing creds in readable temp files | Persistent access after initial breach |

### Specific Attack Vectors Exploited

**1. Shodan Discovery**
```
Attackers searched for "Clawdbot Control" HTML fingerprint
→ Found hundreds of exposed instances in seconds
```

**2. Reverse Proxy Misconfiguration (Root Cause)**
```
User deploys behind nginx/Caddy
→ All connections appear from 127.0.0.1
→ Clawdbot auto-approves localhost connections
→ External attacker gets full admin access
```

**3. Prompt Injection (Crypto Theft)**
```
Attacker sends email with malicious prompt
→ Clawdbot reads email, injects into context
→ LLM follows injected instructions
→ Private key exfiltrated via response
```

### How We Avoid These

| Clawdbot Failure | Roampal Channels Prevention |
|------------------|---------------------------|
| Localhost auto-trust | **No auto-trust.** Explicit auth required for all connections |
| Default insecure config | **Secure by default.** Require explicit opt-in for dangerous features |
| Running as root | **Sandboxed MCP tools.** Least privilege containers |
| Centralized secrets | **Separate stores.** Bot token ≠ API keys ≠ memory |
| No input sanitization | **Sanitize all inputs.** Never pass raw external content to LLM |
| WebSocket bypass | **Single auth path.** No alternate routes that skip validation |

### Architecture Differences

**Clawdbot Architecture (Vulnerable)**
```
┌─────────────────────────────────────────────────┐
│ SINGLE CONTROL LAYER                            │
│ ─────────────────────                           │
│ • Message reading                               │
│ • Secret storage                                │
│ • Command execution                             │
│ • User auth                                     │
│                                                 │
│ One breach = everything compromised             │
└─────────────────────────────────────────────────┘
```

**Roampal Channels Architecture (Proposed)**
```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Discord    │   │    Memory    │   │    Tools     │
│   Adapter    │   │   (local)    │   │   (MCP)      │
├──────────────┤   ├──────────────┤   ├──────────────┤
│ Bot token    │   │ ChromaDB     │   │ Sandboxed    │
│ only         │   │ No network   │   │ containers   │
│ No secrets   │   │ No secrets   │   │ Allowlisted  │
└──────────────┘   └──────────────┘   └──────────────┘
        │                 │                  │
        └─────────────────┼──────────────────┘
                          │
              ┌───────────────────────┐
              │   LLM Provider        │
              │   (User's API key)    │
              │   Direct connection   │
              └───────────────────────┘
```

**Key principle:** No single component has access to everything. Discord adapter doesn't touch filesystem. Memory doesn't touch network. Tools are sandboxed.

### Prompt Injection Prevention

Clawdbot's crypto theft happened because external content (emails) was passed to the LLM without sanitization.

**Prevention strategies:**
1. **Mark external content** - Wrap in clear delimiters the LLM recognizes
2. **Separate data from instructions** - System prompt ≠ user data
3. **Validate tool outputs** - Don't auto-execute commands from external sources
4. **Confirm high-risk operations** - "Are you sure you want to send private key X to Y?"

### MCP Spec Security Requirements (from official spec)

The MCP specification we're building on includes these CRITICAL requirements:

```
1. Tool descriptions are UNTRUSTED unless from trusted servers
2. Explicit user consent BEFORE invoking any tool
3. Validate Origin header on all HTTP connections (DNS rebinding)
4. Bind to localhost only when running locally
5. Human-in-the-loop MUST be maintained
```

These are non-negotiable for Roampal Channels.

### Sources

- [Bitdefender: Moltbot Security Alert](https://www.bitdefender.com/en-us/blog/hotforsecurity/moltbot-security-alert-exposed-clawdbot-control-panels-risk-credential-leaks-and-account-takeovers)
- [TrendingTopics: Clawbot Data Leak Warning](https://www.trendingtopics.eu/clawbot-hyped-ai-agent-risks-leaking-personal-data-security-experts-warn/)
- [MCP Official Specification](https://modelcontextprotocol.io/specification/2025-06-18)

## Success Metrics

- Discord bot installs
- Messages processed per day
- Cross-channel memory hits (searched from one channel, found in another)
- Outcome scoring rate (% of exchanges scored)

## UX Principles (from "Don't Make Me Think")

### Core Philosophy: Self-Evident > Self-Explanatory

> "I should be able to 'get it'—what it is and how to use it—without expending any effort thinking about it."

**Applied to Channels:**
- User adds bot to Discord → immediately understands "I talk to it, it remembers"
- No onboarding wizard, no config screens, no docs required
- The neighbor test: "Oh, it's a chatbot that remembers stuff"

### Principle 1: Get Rid of Question Marks

When users interact with Roampal in Discord/WhatsApp, they should never think:
- "How do I make it remember this?" ❌
- "Did that get saved?" ❌
- "How do I search my memories?" ❌

**Instead, make it mindless:**
- Talk → it remembers (automatic)
- React 👍/👎 → it learns (visible feedback)
- Say "remember when" → it finds it (natural language)

### Principle 2: Mindless Choices

> "It doesn't matter how many times I have to click, as long as each click is a mindless, unambiguous choice."

**Applied to scoring:**
- 👍 = worked
- 👎 = didn't work
- No "rate 1-5", no "was this helpful?", no forms

**Applied to memory:**
- One command: `@roampal remember [thing]`
- One search: `@roampal what did I say about [topic]`
- One forget: `@roampal forget [thing]`

### Principle 3: Goodwill Reservoir

> "Users have limited patience. A single mistake can empty it."

**Things that drain goodwill (avoid):**
- Requiring API key in chat (feels like setup)
- Long bot responses that flood the channel
- Memory failures with no feedback
- "I don't understand" without helpful suggestions

**Things that build goodwill (do):**
- Bot quietly remembers without being asked
- Short, scannable responses
- Proactive: "Last time this worked: [solution]"
- Graceful degradation: "Memory server unreachable, but I'll remember this when it's back"

### Principle 4: Users Scan, Don't Read

Chat messages are scanned even faster than web pages.

**Format memories for scanning:**
```
❌ Bad:
"Based on our previous conversations on January 15th, 2026,
you mentioned that you prefer using JWT tokens for authentication
with a 15-minute expiry, and this pattern has been successful
in 92% of your auth-related discussions."

✅ Good:
"Auth: JWT, 15min expiry (worked 92%)"
```

**Bold keywords, short sentences, scannable structure.**

### Principle 5: No Hover = No Discovery

Mobile chat apps have no hover states. Everything must be discoverable through:
- Explicit commands (`/roampal help`)
- Reactions (visible emoji options)
- Natural language (just ask)

Don't rely on:
- Tooltips
- Context menus
- "Click to reveal"

### The Accessibility Test

Before shipping any channel feature, ask:

1. **Can a first-time user figure this out in 10 seconds?**
2. **Does it work without reading any docs?**
3. **Is the value obvious immediately?**

If no to any: simplify until yes.

---

*This is a vision document. Implementation details will evolve.*
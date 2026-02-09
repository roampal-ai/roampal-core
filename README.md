<p align="center">
  <img src="assets/banner.svg" alt="Roampal - Outcome-Based Memory for AI Coding Tools" width="900">
</p>

<p align="center">
  <a href="https://pypi.org/project/roampal/"><img src="https://img.shields.io/pypi/v/roampal?color=blue&style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/roampal/"><img src="https://img.shields.io/pypi/dm/roampal?color=blue&style=flat-square" alt="Downloads"></a>
  <a href="https://github.com/roampal-ai/roampal-core/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square" alt="Python"></a>
</p>

<p align="center">
  <strong>Two commands. Your AI coding assistant gets outcome-based memory.</strong><br>
  Works with <strong>Claude Code</strong> and <strong>OpenCode</strong>.
</p>

---

## Quick Start

```bash
pip install roampal
roampal init
```

Auto-detects installed tools. Restart your editor and start chatting.

> Target a specific tool: `roampal init --claude-code` or `roampal init --opencode`

<p align="center">
  <img src="assets/init-demo.svg" alt="roampal init demo" width="720">
</p>

### Platform Differences

The core loop is identical — both platforms inject context, capture exchanges, and score outcomes. The delivery mechanism differs:

| | Claude Code | OpenCode |
|--|-------------|----------|
| Context injection | Hooks (stdout) | Plugin (system prompt) |
| Exchange capture | Stop hook | Plugin `session.idle` event |
| Scoring | Main LLM prompted via hooks | Main LLM prompted + independent sidecar fallback |
| Self-healing | Hooks auto-restart server on failure | Plugin auto-restarts server on failure |

Both platforms prompt the main LLM to score each exchange. OpenCode adds an independent sidecar call (using free models) as a fallback — sidecar only runs if the main LLM doesn't call `score_response`, so memories are never double-scored.

## How It Works

When you type a message, Roampal automatically injects relevant context before your AI sees it:

**You type:**
```
fix the auth bug
```

**Your AI sees:**
```
═══ KNOWN CONTEXT ═══
• [patterns] (3d, s:0.9, [YYY]) JWT refresh fixed auth loop
• [memory_bank] Alex, backend engineer — prefers: no git staging
═══ END CONTEXT ═══

fix the auth bug
```

No manual calls. No workflow changes. It just works.

### The Loop

1. **You type** a message
2. **Roampal injects** relevant context automatically (hooks in Claude Code, plugin in OpenCode)
3. **AI responds** with full awareness of your history, preferences, and what worked before
4. **Outcome scored** — good advice gets promoted, bad advice gets demoted
5. **Repeat** — the system gets smarter every exchange

### Five Memory Collections

| Collection | Purpose | Lifetime |
|------------|---------|----------|
| `working` | Current session context | 24h, then auto-promotes |
| `history` | Past conversations | 30 days, outcome-scored |
| `patterns` | Proven solutions | Permanent, promoted from history |
| `memory_bank` | Identity, preferences, goals | Permanent |
| `books` | Uploaded reference docs | Permanent |

## Commands

```bash
roampal init                # Auto-detect and configure installed tools
roampal init --claude-code  # Configure Claude Code explicitly
roampal init --opencode     # Configure OpenCode explicitly
roampal start               # Start the HTTP server manually
roampal stop                # Stop the HTTP server
roampal status              # Check if server is running
roampal stats               # View memory statistics
roampal doctor              # Diagnose installation issues
roampal ingest <file>       # Add documents to books collection
roampal books               # List all ingested books
roampal remove <title>      # Remove a book by title
```

## MCP Tools

Your AI gets 7 memory tools:

| Tool | Description |
|------|-------------|
| `get_context_insights` | Quick topic lookup — user profile + relevant memories |
| `search_memory` | Deep search across all collections |
| `add_to_memory_bank` | Store permanent facts (identity, preferences, goals) |
| `update_memory` | Correct or update existing memories |
| `delete_memory` | Remove outdated info |
| `score_response` | Score previous exchange — prompted automatically by hooks |
| `record_response` | Store key takeaways from significant exchanges |

## What's Different?

| Without Roampal | With Roampal |
|-----------------|--------------|
| Forgets everything between sessions | Remembers you, your preferences, what worked |
| You repeat context every time | Context injected automatically |
| No learning from mistakes | Outcomes tracked — bad advice gets demoted |
| No document memory | Ingest docs, searchable forever |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  pip install roampal && roampal init                    │
│    Claude Code: hooks + MCP → ~/.claude/                │
│    OpenCode:    plugin + MCP → ~/.config/opencode/      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  HTTP Hook Server (port 27182)                          │
│    Auto-started on first use, self-heals on failure     │
│    Manual control: roampal start / roampal stop         │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  User types message                                     │
│    → Hook/plugin calls HTTP server for context          │
│    → Server returns context + scoring prompt            │
│    → AI sees context, scores previous, responds         │
│    → Exchange stored for next turn                      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Single-Writer Backend                                  │
│    FastAPI → UnifiedMemorySystem → ChromaDB             │
│    All clients share one server, isolated by session    │
└─────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.10+
- One of: **Claude Code** or **OpenCode**
- **Platforms:** Windows, macOS, Linux (primarily developed and tested on Windows)

## Troubleshooting

<details>
<summary><strong>Hooks not working? (Claude Code)</strong></summary>

- Restart Claude Code (hooks load on startup)
- Check HTTP server: `curl http://127.0.0.1:27182/api/health`
</details>

<details>
<summary><strong>MCP not connecting? (Claude Code)</strong></summary>

- Verify `~/.claude.json` has the `roampal-core` MCP entry with correct Python path
- Check Claude Code output panel for MCP errors
</details>

<details>
<summary><strong>Context not appearing? (OpenCode)</strong></summary>

- Make sure you ran `roampal init --opencode`
- Check that the server auto-started: `curl http://127.0.0.1:27182/api/health`
- If not, start it manually: `roampal start`
</details>

<details>
<summary><strong>Server crashes and recovers?</strong></summary>

This is expected. Roampal has self-healing — if the HTTP server stops responding, hooks automatically restart it and retry.
</details>

**Still stuck?** Ask your AI for help — it can read logs and debug Roampal issues directly.

## Support

Roampal Core is completely free and open source.

- Support development: [roampal.gumroad.com](https://roampal.gumroad.com/l/aagzxv)
- Feature ideas & feedback: [Discord](https://discord.com/invite/F87za86R3v)
- Bug reports: [GitHub Issues](https://github.com/roampal-ai/roampal-core/issues)

## License

[Apache 2.0](LICENSE)

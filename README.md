# Roampal

**Persistent Memory for AI Coding Tools**

Two commands. Claude Code gets persistent memory.

## Installation

```bash
pip install roampal
roampal init
```

That's it. Restart Claude Code and your AI remembers everything.

> **Note:** No `roampal start` needed! The MCP server auto-starts the hook server when Claude Code launches.

## How It Works

### The Magic: Automatic Context Injection

When you type a message:

**You see:**
```
Help me fix this auth bug
```

**The AI sees:**
```
<roampal-score-required>
Score the previous exchange before responding.
Previous: User asked "How do I configure TypeScript?" You answered "..."
Call score_response(outcome="worked|failed|partial|unknown") FIRST.
</roampal-score-required>

═══ KNOWN CONTEXT ═══
• Alex, backend engineer at StartupCo (memory_bank)
• JWT refresh token pattern worked for auth issues (92% proven, patterns)
═══ END CONTEXT ═══

Help me fix this auth bug
```

You never have to ask "remember when..." - it's automatic.

### How?

1. **UserPromptSubmit Hook**: Before Claude sees your message, Roampal injects relevant context + scoring prompt
2. **Outcome Learning**: When things work (`score_response(worked)`), memories get promoted. When they fail, they get demoted.
3. **Five Memory Collections**:
   - `memory_bank`: Your identity, preferences, goals (never decays)
   - `patterns`: Proven solutions (auto-promoted from history)
   - `history`: Past conversations
   - `working`: Current session context
   - `books`: Uploaded reference docs

## Commands

```bash
roampal init      # Configure Claude Code (one-time)
roampal ingest    # Add documents (.txt, .md, .pdf) to books collection
roampal books     # List all ingested books
roampal remove    # Remove a book by title
roampal status    # Check if server is running
roampal stats     # View memory statistics
```

## MCP Tools

The AI has these tools for memory access:

| Tool | Description |
|------|-------------|
| `get_context_insights` | Get user profile + relevant memories |
| `search_memory` | Search across collections |
| `add_to_memory_bank` | Store permanent facts |
| `update_memory` | Update existing memories |
| `delete_memory` | Delete outdated info |
| `score_response` | Score previous exchange (prompted by hook) |
| `record_response` | Store key takeaways (optional) |

## What's Different?

| Vanilla Claude Code | With Roampal |
|---------------------|--------------|
| Forgets everything between sessions | Remembers you, your preferences, what worked |
| You repeat context every time | Context injected automatically |
| No learning from mistakes | Outcomes tracked - bad advice gets demoted |
| No document memory | Ingest docs, they're searchable forever |

## Requirements

- Python 3.10+
- Claude Code (VS Code extension or CLI)
- **Platforms:** Windows, macOS, Linux (macOS/Linux community-tested)

## Troubleshooting

**Hooks not working?**
- Restart Claude Code (hooks load on startup)
- Check HTTP server: `curl http://127.0.0.1:27182/api/health`

**MCP not connecting?**
- Verify `~/.claude/mcp.json` exists and has correct Python path
- Check Claude Code output panel for MCP errors

**Context not appearing?**
- Make sure you ran `roampal init`
- Restart Claude Code after init

**Still stuck?** Ask Claude Code for help - it can read logs and debug Roampal issues directly.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  pip install roampal && roampal init                    │
│    → Configures hooks + MCP in ~/.claude/               │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  User opens Claude Code                                 │
│    → Claude loads MCP server (roampal.mcp.server)       │
│    → MCP server spawns HTTP hook server (port 27182)    │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  User types message                                     │
│    → UserPromptSubmit hook calls HTTP server            │
│    → Server returns context + scoring prompt            │
│    → AI sees context, scores previous, responds         │
│    → Stop hook stores exchange for next turn            │
└─────────────────────────────────────────────────────────┘
```

## Support

Roampal Core is completely free.

Support development: [roampal.gumroad.com](https://roampal.gumroad.com/l/aagzxv)

Have feature ideas? Join the [Discord](https://discord.com/invite/F87za86R3v).

## License

Apache 2.0

## Credits

Built with love for the AI coding community. Based on learnings from building [Roampal Desktop](https://roampal.ai).

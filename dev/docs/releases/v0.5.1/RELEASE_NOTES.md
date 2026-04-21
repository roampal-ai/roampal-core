# Roampal Core v0.5.1 Release Notes

**Release Date:** TBD
**Type:** Minor — named memory profiles + sidecar consistency patch

## Summary

Two changes bundled:

1. **Named memory profiles** (feature) — support multiple isolated memory stores per user, switchable per command, per shell, per project (MCP config), or globally.
2. **Sidecar scoring cap** (patch) — cap exchange fields at 8K characters in the scoring prompt, matching the fact-extraction call.

---

## 1. Named Memory Profiles

### Motivation

Issue [#5](https://github.com/roampal-ai/roampal-core/issues/5) requested separate memory stores per project and per context. The existing `ROAMPAL_DATA_PATH` environment variable supported arbitrary paths but three CLI commands (`roampal doctor`, `roampal stats`, `roampal start`) resolved against the system default regardless of the variable. Named profiles close that gap and make multi-store use a first-class feature.

### Concept

A **profile** is a named isolated memory store. Each profile has its own ChromaDB instance and session files — fully segregated from other profiles.

The registry at `<config_dir>/profiles.json` maps a profile name to an optional path:

```json
{
  "work": "/existing/custom/path",
  "project-a": null,
  "home": null
}
```

- `null` path — auto-locate at `<system_default_base>/<slug>/`
- String path — use that path verbatim. Lets existing users register directories they already manage via `ROAMPAL_DATA_PATH` without any data migration.

### Path resolution precedence

When `UnifiedMemorySystem` resolves its data path (highest wins):

1. Explicit `data_path` constructor argument
2. `ROAMPAL_DATA_PATH` environment variable (backward-compat override)
3. Named profile from the registry (this feature)
4. System default path (dev/prod-mode aware, unchanged from v0.5.0)

When no `profile_name` is passed and the server is resolving its own default, `active_profile_name()` applies a secondary precedence to determine which profile is active:

1. `ROAMPAL_PROFILE` environment variable (per-shell or per-project via MCP config `env: {}`)
2. Persisted `<config_dir>/active_profile` file (written by `roampal profile use <name>`)
3. `"default"`

### New CLI commands

| Command | Purpose |
|---------|---------|
| `roampal profile list` | Enumerate registered profiles and their resolved paths |
| `roampal profile show` | Print the currently active profile and its source (env / file / default) |
| `roampal profile create <name>` | Create a new profile, auto-located under the system default base |
| `roampal profile register <name> --path <dir>` | Register an existing directory as a named profile (no data move) |
| `roampal profile delete <name> [--destroy-data]` | Remove a profile; optionally delete its data directory |
| `roampal profile use <name>` | Persist a profile as the user-global default |
| `roampal profile unuse` | Clear the persisted active profile |
| `roampal profile switch <name>` | Persist the new active profile and kill any running server so the next MCP tool call spawns on it |

Existing commands `roampal start` and `roampal doctor` gained a `--profile <name>` flag for per-invocation overrides.

### Per-project usage via MCP config

Each MCP client can bind its server to a specific profile via the `env` block of its config file:

```json
{
  "mcpServers": {
    "roampal": {
      "command": "roampal",
      "args": ["start"],
      "env": { "ROAMPAL_PROFILE": "project-a" }
    }
  }
}
```

Any MCP client that honors `env` (Claude Code, OpenCode, Cursor, Cline, etc.) can scope Roampal to a named profile per project.

### Architecture impact

The server resolves the active profile once at `lifespan` startup — it is bound to that profile for its lifetime. `profile switch` is the supported way to swap: it writes the new active profile and kills any running server so the next MCP tool call triggers `_ensure_server_running` and spawns a fresh server bound to the new profile.

### Files

- `roampal/profile_manager.py` (new) — `ProfileRegistry`, `profile_slug`, `resolve_data_path`, `active_profile_name`, `active_profile_source`, read/write/clear helpers for the persisted active_profile file, full error taxonomy (`ProfileError`, `ProfileNotFoundError`, `ProfileAlreadyExistsError`, `InvalidProfileNameError`).
- `roampal/test_profile_manager.py` (new) — 52 unit tests covering slug edge cases (Unicode, empty, special characters, max-length truncation), registry load/save/corruption recovery, create/register/delete mutations, path resolution for all branches, and the full precedence chain including persisted-file behavior.
- `roampal/backend/modules/memory/unified_memory_system.py` — `__init__` takes `profile_name: Optional[str]`. Resolves via `ProfileRegistry` when `data_path` is None and `ROAMPAL_DATA_PATH` is unset. Pre-v0.5.1 callers are unaffected.
- `roampal/cli.py` — `cmd_profile()` dispatcher, `profile` subparser with eight subcommands, `--profile` flag on `start` and `doctor`, extracted `_stop_server_on_port()` helper shared by `cmd_stop` and `cmd_profile.switch`.
- `roampal/server/main.py` — `lifespan` resolves the active profile via `active_profile_name()` and logs the source (`active_profile_source()`).

### Migration

- **Single-store users (default):** zero change. The default profile resolves to the same system path as v0.5.0. No data migration required.
- **Users of `ROAMPAL_DATA_PATH` overrides:** existing directories keep working. Register each as a named profile (`roampal profile register <name> --path <existing>`) to gain first-class CLI support. The `ROAMPAL_DATA_PATH` variable remains honored as a one-shot override for backward compatibility.

---

## 2. Sidecar Scoring Input Cap

### Problem

The scoring/summary LLM call in `roampal/plugins/opencode/roampal.ts` sent exchange fields (user message, assistant message, follow-up) without length limits, while the fact-extraction call already capped at 8,000 characters. Inconsistent limits left an edge case where a very long exchange could exceed the scoring model's context window.

### Observed exchange sizes

Across 18 sessions and 41 exchanges surveyed, the maximum user message was 138 characters and the maximum assistant response was 2,938 characters. No exchange exceeded 3,000 characters. The cap is a safety net, not a practical limit for current usage.

### Change

```typescript
// Before (uncapped):
"${exchange.user}"
"${exchange.assistant}"
"${currentUserMessage}"

// After (8K per field):
"${exchange.user.slice(0, 8000)}"
"${exchange.assistant.slice(0, 8000)}"
"${currentUserMessage.slice(0, 8000)}"
```

### Files

- `roampal/plugins/opencode/roampal.ts` — `.slice(0, 8000)` applied to the three exchange fields in the scoring prompt (lines 563, 566, 569).

---

## Testing

- `py -3.13 -m pytest roampal/test_profile_manager.py` — 52 unit tests, all passing.
- `py -3.13 -m pytest roampal/` — full backend suite 564 tests passing.
- Editable install verified end-to-end with manual CLI test covering `create` → `list` → `show` → `use` → `switch`, including MCP server profile isolation verified against Claude Code.

## Documentation updates

- `README.md` — removed the "Why?" prose intro, promoted benchmarks to their own section, added a commands block for profile subcommands, added a "Named Memory Profiles" section with usage examples and precedence, moved the Glama badge to the Support footer.
- `ARCHITECTURE.md` — added "Named Memory Profiles" section covering registry format, path resolution precedence, server lifecycle, and file layout. Updated "Data Storage" to reflect per-profile ChromaDB locations.

## Deferred to later releases

- Roampal Desktop (roampal-ai/roampal) does not yet expose profile switching in its UI.
- Convenience command to write `ROAMPAL_PROFILE` into a target MCP client's config file (Claude Code, OpenCode). The underlying mechanism already works today via hand-edited `env` blocks.
- Comment on issue #5 with migration instructions once the release ships.

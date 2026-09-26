# Roampal Core v0.6.0

**Status:** Released 2026-09-26.
**Tests:** 1122 passed, 4 skipped, 0 failed (Windows, Python 3.10, whole repo). Python 3.13 measured at 1094 passed, 2 skipped before the final round of fixes.
**Scope:** no embedder change — mpnet-INT8 and the cross-encoder stay exactly as in v0.5.9 (the e5-base upgrade is deferred; see Known issues).

Per-task evidence, test names and dates live in `IMPLEMENTATION_TASKS.md`. Item numbers below match the references there.

---

## Upgrade notes

No data migration: existing memories, profiles and `profiles.json` are read as-is, and configs written by older versions keep working.

- **Everyone:** run `roampal stop` after upgrading (or close all your apps). A server that is already running keeps the old code until it stops; the next prompt starts a 0.6.0 one.
- **Claude Code (optional):** rerun `roampal init` to get the new launch flags and the SessionStart `clear` hook. It only changes Roampal's own entries and keeps a backup (Item 15).
- **`ROAMPAL_PROFILE` in an MCP config now also applies to Claude Code hooks.** Before, only the MCP tools saw it, so hooks read and wrote your `profile use` profile (or `default`). They now use the same profile as the tools.
- **OpenCode users:** run `roampal init --force --opencode`, then restart OpenCode. Upgrading the package does not update the installed plugin, and the old plugin restarts the server with whatever Python is first on PATH. Since the server now shuts down after 30 idle minutes, that restart happens often, and it fails when that Python has no Roampal (pipx/venv installs), leaving OpenCode without memory until something else starts the server (Item 16). The old plugin also keeps the Zen default (Item 13). `init` now keeps your scoring model (Item 19).
- **OpenCode users who never chose a scoring model:** scoring, summaries and fact extraction are now off until you run `roampal sidecar setup`. Previously the plugin silently used Zen cloud models (Item 13). Retrieval is unaffected.
- **Folder bindings are new** (`roampal profile bind`). A `ROAMPAL_PROFILE` set in an app's MCP config still overrides them for that app — remove it if you want bindings to apply there; `roampal profile use` already sets your global default.
- **`roampal score` is removed.** Scoring is automatic in both Claude Code and OpenCode; use `roampal sidecar test` to check your scoring model.
- **Python 3.10–3.13** are supported; upgrading also upgrades ChromaDB to ≥ 1.5.9 (Item 17).
- **Model loading is cache-first.** When the model files are already in the local HF cache, startup uses them without a per-load network check; first downloads and `roampal reembed` behave as before.

---

## Profiles

### Item 3 — Bind a folder to a profile
**Change:** `roampal profile bind <name> [--path <dir>]` / `unbind` map a directory (and everything under it) to a profile; `profile show` / `list` report the binding and why it applies. Bindings live in a reserved `bindings` map in `profiles.json`; the innermost bound folder wins; Windows paths match case-insensitively. Binding an unregistered profile fails cleanly.
**Precedence:** `ROAMPAL_PROFILE` (env or the app's MCP config) → folder binding → `profile use` → `default`.
**Tests:** `test_profile_bindings.py`, `test_profile_bind_commands.py`.

### Item 6 — Every request names its profile
**Why:** the shared server used to fall back to its own cwd/env, i.e. those of whichever session happened to start it, so an unbound project's memories could land in another project's profile.
**Change:** clients always send their profile (explicit `default` included); OpenCode sends its project folder and the server applies that folder's binding; the server never guesses from its own environment (no header → `profile use` → `default`); spawners don't leak a project's `ROAMPAL_PROFILE` into the server; the MCP server re-resolves on every call, so bind/unbind takes effect mid-session. Found in the release smoke and fixed: CLI data commands (`stats`, `books`, `ingest`, `remove`, `summarize`, `retag`) sent no profile and acted on the wrong one inside a bound folder, and `ingest`'s server-down fallback ignored profiles entirely (since v0.5.1).
**Tests:** `test_multi_profile_routing.py` (one server, parallel clients on different profiles — failed before, passes now); `test_cli_profile_headers.py` (guard: every CLI server call except health checks carries the profile). Live: Claude Code and OpenCode in the same folder on different profiles, no cross-writes.

### Item 11 — CLI polish
`dir(roampal.cli)` no longer crashes; `bind default` allowed; `bindings` rejected as a profile name; `profile delete` reports the bindings it removes and where each folder now resolves; `use`/`switch` warn when a binding overrides them; `switch` is the same as `use` (it no longer stops the server); `profile create` suggests `bind`/`use`; dead code removed.

---

## Shared server

### Item 7 — No app owns the server
**Why:** the app that started the server killed it on exit; hooks killed a healthy server on a timeout or 503; `profile switch` killed it; and every launch ran `python -m` from the caller's folder, so a folder containing a `roampal/` package ran that code instead of the installed one.
**Change:** the server retires itself after 30 idle minutes (not when an app closes); restarts happen only when it is actually down, one restarter at a time across Claude Code hooks, the MCP server and the OpenCode plugin; the server always launches from a neutral folder with `-E` (plus `-P` on 3.11+); the commands `init` writes never import from the caller's folder.
**Live check:** closing OpenCode (which had started the server) left it running and Claude Code kept working.

### Item 16 — OpenCode restarts the server with Roampal's own Python
**Why:** the plugin guessed the Python from PATH, which is often a different install — on the dev machine a restart would have started a 0.5.9 server under 0.6.0 clients; for pipx/venv installs the PATH Python has no Roampal at all.
**Change:** the plugin reuses the interpreter and flags `roampal init` recorded in `opencode.json` (a sibling `pythonw.exe` on Windows, so no console window), and falls back to PATH only when that is missing — and logs why.
**Tests:** `test_plugin_server_launch.py` (the real resolver run under Node). Live: after `roampal stop`, OpenCode restarted the server as `Python310\pythonw.exe -E -m roampal.server.main`.

---

## Memory quality

### Item 12 — Memory text is never silently cut
**Why:** a 2026-09-18 audit found three silent cuts: RECENT EXCHANGES clipped summaries at 200 characters, OpenCode stored the user's message cut at 200 characters permanently, and the server cut add/update/takeaway text at 2,000 characters.
**Change:** one limits table (`roampal/memory_limits.py`) checked once on the server: target ~300 characters / 1–2 sentences, hard max 600 (facts 150), backstop 2,000. Over-limit writes are **rejected** with a short "rewrite shorter / split it" message the model actually sees (length checks moved out of the tool schemas); nothing is stored on rejection, and the OpenCode plugin re-asks its model once. Exchange memories store the summary alone on every platform — the OpenCode `User: … / Assistant: …` wrapper is gone. RECENT EXCHANGES and the cold-start profile block show up to 300 characters, cut at a word boundary with "…", and `search_memory` always returns full text. `roampal summarize` no longer truncates its own output. A guard test fails the build on any new slice of memory text outside the two display cuts.
**Not migrated:** existing wrapped OpenCode memories stay as they are (full text, searchable).

### Item 10 — Claude Code doesn't re-inject what's already in context
**Why:** Claude Code keeps every hook output in context until compaction, and the server re-sent memories already there.
**Change:** a memory already shown in the conversation (same ID and content) becomes a one-line pointer instead of full text; it is still flagged relevant and still scored. A memory counts as shown only after the hook confirms delivery, so a failed hook means full text again next turn. No size cap; retrieval stays at 8 memories per turn. The record resets on compaction and new sessions. OpenCode's prompt-cache impact was measured and its injection kept as is.

### Item 20 — One summary per OpenCode exchange
**Why:** since v0.4.8 the plugin sent each summary twice (with its scoring call and to store it), and the server stored both. The extra copy usually expired within 24 hours and, under 0.5.9, was worded differently, so it went unnoticed; with Item 12 the copies became identical.
**Change:** the plugin sends the summary once; the server also ignores an identical summary for the same conversation within 10 minutes, which covers plugin copies installed before this release. Claude Code was never affected.
**Tests:** `test_summary_single_store.py`. Live: one OpenCode exchange → one summary. Existing duplicates are not removed automatically (the dev machine's main profile had 4).

### Items 2 + 9 — Guard against a silent dedup collapse
**Why:** a v0.5.9 embedder trial silently merged distinct facts through the dedup threshold; nothing but the accuracy gate noticed.
**Change:** a test stores distinct facts about one entity through the real write path and asserts they stay distinct — on the real mpnet-INT8 model (cached in CI) and on recorded real vectors — and is shown to fail at a threshold that would merge them. The dedup feature itself is unchanged.
**Tests:** `test_dedup_regression.py`.

### Item 18 — A removed book can be ingested again
**Why:** `roampal remove` hides a book's chunks (v0.2.2), but `ingest`'s duplicate check still counted them, so re-ingesting a removed title printed "Stored" and stored nothing; removing it again also "succeeded".
**Change:** both lookups ignore removed chunks, and `roampal stats` counts only books that are still present, listing removed ones separately (`books: 1 items (1 removed, hidden)`; `stats --json` adds `"removed"`) — it used to count removed books too (Task 46).
**Tests:** `test_book_reingest.py`.

---

## OpenCode scoring and setup

### Item 13 — Nothing goes to Zen without an explicit opt-in
**Why:** v0.5.3's rule (no choice = scoring off) was never applied in the plugin, which kept its v0.3.7 default: from v0.3.7 to v0.5.9, OpenCode users without a custom model had exchange text sent to `opencode.ai/zen`, even after choosing "Skip" or running `sidecar disable`.
**Change:** the plugin scores only with a chosen custom model or a recorded Zen opt-in; otherwise it makes no scoring calls and tells the model scoring is off. A custom model wins over Zen.
**Tests:** `test_sidecar_privacy.py` (the plugin's decision code run under Node).

### Item 14 — Choose a scoring model without the menu
**Change:** `roampal sidecar setup --list [--json]` lists every choice with whether data leaves the machine, where it goes, and the command to pick it — never printing keys. One flag records a choice: `--model`, `--url … --model … [--key-env VAR]`, `--go`, `--zen`, or `--auto` (smallest detected local model, never cloud). Errors exit 1 and write nothing. The menu and the flags share one option builder.

### Item 19 — `init` keeps an existing scoring model
**Change:** when a model is configured, `init` says so and asks `Keep it? [Y/n]` (Enter keeps it); without a terminal it keeps it and says so. The menu shows the current model, and "Skip" no longer claims scoring was turned off.
**Tests:** `test_init_keeps_sidecar.py`. Verified live.

---

## Install and config safety

### Item 15 — `roampal init` never wipes or clutters config files
**Why:** Claude Code/Cursor setup could replace an unreadable config (e.g. a UTF-8 file read with the Windows locale codec) with a Roampal-only one, deleted the user's own hooks, kept no backups, and always dropped a `.mcp.json` into the current folder.
**Change:** every config write goes through `roampal/utils/safe_config.py`: UTF-8 read (BOM tolerated), timestamped backup (newest 3 kept), atomic write. An unreadable file is left byte-identical with instructions, that tool is skipped, and `init` exits 1. Hooks merge per event, touching only Roampal's own entries; running `init` twice is byte-identical. A project `.mcp.json` is written only with `--scope project` or when one already has Roampal. `init --no-input` never waits on a menu, and `--scope user` writes no project file.
**Tests:** `test_init_config_safety.py`.

---

## CLI

### Item 1 — `cli.py` split into a package
**Why:** a 4,700-line `cli.py` with 17 commands; v0.5.9 briefly shipped a bug that silently dropped subcommands from it.
**Change:** `roampal/cli/` package grouped by domain (setup, server, memory, sidecar, profile, diagnose) with a dispatch table; the entry point and every command's output are unchanged. A duplicate dead `cmd_context` was removed. `roampal score` — a Claude Code-only leftover from v0.3.6 that had stored nothing since v0.5.6 — was removed.
**Tests:** `test_cli_golden.py` (byte-identical output snapshots taken before the split), `test_cli_dispatch_table.py` (parser commands == dispatch table, no duplicate function names), a pyflakes undefined-name guard over `roampal/cli/`.

---

## Platforms and CI

### Item 17 — Python 3.13 and macOS
**Change:** the help-output snapshot comparison no longer depends on argparse's column width (3.13 lays it out one space wider); CI runs Python 3.10–3.13 on every job plus a new macOS job, with Node 22 so the plugin's decision tests run instead of skipping. `pyproject.toml` adds the 3.13 classifier and raises the floor to `chromadb>=1.5.9`, so an upgrade can't leave an older ChromaDB reading a newer store. README states 3.10+ (3.10 reaches end-of-life in October 2026).

### Item 8 — Tests don't depend on the developer's machine
**Change:** the unit suite runs in an isolated home by default and patching `roampal.cli.X` patches every module holding it; the golden suite runs on Ubuntu, Windows and macOS; `roampal status` reports `stopped` on a connect timeout too. Verified: the full suite gives the same result in an empty isolated home.

### Item 5 — WAL crash-recovery test (test-only)
The ChromaDB crash-recovery integration test failed in its own cleanup (unguarded `close()`); guarded the way production code already is. Integration tests now have explicit CI jobs on Ubuntu and Windows (30 of them had never run on Windows). No product behavior changed.

---

## Docs — Item 4
README and ARCHITECTURE document `profile bind`/`unbind`, `reembed` and `help`, the profile routing contract and the shared-server lifecycle; overclaims were corrected; version is 0.6.0 in `pyproject.toml` and `roampal/__init__.py`; `glama.json` gets its pinned commit after the release push.

---

## Known issues and deferred

- **Embedder upgrade (e5-base)** deferred until the dedup threshold is recalibrated for it.
- **Existing data is not migrated:** wrapped OpenCode memories (Item 12) and pre-existing duplicate summaries (Item 20) stay; `roampal summarize` can rewrite long memories on request.
- **Claude Code rewind fires no hook**, so an Item 10 pointer can refer to a rewound turn (each pointer keeps its ID for `search_memory`).
- **The OpenCode plugin fetches the Zen free-model list at startup** even without a Zen opt-in (no exchange text is sent).
- **Editable installs can report a stale version** (pip metadata wins over `__init__.py`); regular installs are unaffected.
- **Test/code cleanups for later:** `psutil` is not in the dev extras, so the memory-soak test always skips; a leftover `kg_service` argument shim (and its 3 tests); `count_successes_from_history` exists in two services; one redundant no-knowledge-graph test.

## Coordination

- **No RAM change** — the v0.5.9 footprint (~484 MB) carries over.
- **No accuracy-gate rerun** — no retrieval-ranking change; summary-only storage matches the benchmarked format.
- **Desktop** picks this up whenever its bundled core is bumped to 0.6.0.

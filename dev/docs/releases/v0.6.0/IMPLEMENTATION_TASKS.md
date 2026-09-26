# v0.6.0 — Implementation Log

Companion to `RELEASE_NOTES.md` (Item numbers match). Condensed 2026-09-26 from
the full per-task log; every task, decision, residual and lesson is kept, the
step-by-step narration and intermediate test counts are not.

## Status (2026-09-26)

- **Done:** Round 1 (Tasks 0a–16), Round 2 (17–37 + review fixes R1–R16),
  Round 3 (38–46). Nothing is committed yet — the working tree holds the whole
  release.
- **Tests:** whole tree **1122 passed, 4 skipped, 0 failed** (Windows, 3.10).
  Isolated-home run 1120/5 (one extra sanctioned skip, see Task 26). Python
  3.13: 1094 passed, 2 skipped (measured at Task 42).
- **Open:**
  - **Task 28 — CI proof.** `tests.yml` triggers only on `push: [main]` and
    `pull_request: [main]`, so a release-branch push runs nothing: open a PR
    to main (or add the branch / `workflow_dispatch`). Confirm in the run:
    every job green on Ubuntu/Windows/macOS × 3.10–3.13; the Node-executed
    plugin tests (`test_plugin_server_launch.py`, `test_sidecar_privacy.py`,
    type-strip parse checks) report **passed, not skipped** (Node 22 was added
    to the `test` and `unit-tests-macos` jobs because runners ship Node 20,
    below the 22.13 `stripTypeScriptTypes` gate); the real-model dedup test
    runs (HF cache step); `plugin-parse` (bun) green; watch the first macOS
    run for fork-safety crashes (`--forked` after onnxruntime import); observe
    the integration job go red once on a deliberately failing edit, then
    revert (carried from 0b).
  - **Task 34 — final gates:** every local gate passed (below); closes when
    28 is green.
  - **After push:** pin `glama.json`'s commit (bumped to 0.6.0, commit
    dropped until then).

## Task index

| # | Task | Item |
|---|------|------|
| 0a | Guard `close()` in the WAL hard-kill test teardown | 5 |
| 0b | Explicit `integration/` CI jobs (Ubuntu + Windows) | 5 |
| 1–2 | Golden snapshots + dispatch-table invariants, captured against the untouched monolith | 1 |
| 3–10 | Split `cli.py` into the `roampal/cli/` package | 1 |
| 11, 29 | Dedup regression guard (fake fixture, then real model) | 2, 9 |
| 12–14 | Folder bindings: resolver, `bind`/`unbind`, client seams | 3 |
| 15, 33 | Docs + version bump; doc corrections | 4 |
| 16, 34 | Final gates (round 1; round 2) | all |
| 17–21 | Profile routing contract (server stops guessing; clients name the profile; OpenCode cwd header; MCP per-call; multi-profile test) | 6 |
| 22–24, 36 | Shared-server lifecycle (no owner, idle retire, single-flight restart, `switch` no kill, neutral spawn) | 7 |
| 25–28 | Test hermeticity + cross-platform CI | 8 |
| 30–31 | Claude Code context dedup; OpenCode prompt-cache measurement | 10 |
| 32 | CLI polish | 11 |
| 35, 37 | Memory length rules — one table, never silently cut | 12 |
| 38 | No Zen without opt-in | 13 |
| 39 | Non-interactive `sidecar setup` | 14 |
| 40 | `init` never wipes/clutters config files | 15 |
| 41 | OpenCode restarts the server with Roampal's own Python | 16 |
| 42 | Python 3.13 + macOS | 17 |
| 43, 46 | Re-ingest a removed book; `stats` excludes removed books | 18 |
| 44 | `init` keeps an existing scoring model | 19 |
| 45 | One summary per OpenCode exchange | 20 |
| R1–R16 | Review / smoke fixes (below, under the area they touch) | — |

---

## Contracts that the code now holds (read before changing these areas)

**Profile resolution.**
- Client walk (hooks, MCP, CLI): `ROAMPAL_PROFILE` process env → project
  `.mcp.json` env → `~/.claude.json` per-project (local) scope → user-scope
  `mcpServers.*.env` → folder binding → persisted `use` → server pin file →
  `default`. Clients **always** send `X-Roampal-Profile`, including literal
  `default`. Single helper: `profile_manager.profile_header_value()`.
- OpenCode plugin: its own env / per-project / user-global resolution; when
  empty it sends `X-Roampal-Cwd: encodeURIComponent(<worktree>)` and the
  server resolves that folder's binding (no TS copy of the binding walk).
- Server (`_resolve_profile_name`): profile header → launch pin → cwd-header
  binding (unquoted) → persisted `use` → `default`. Never the server's own
  cwd/env (F1). An unregistered named profile → HTTP 404 with the
  `roampal profile create` hint (never silently the default store), except
  `default` or a `ROAMPAL_DATA_PATH` override.
- Pin (`roampal start --profile X`): `server_pin_<port>.txt` in the config
  dir, written after validation, cleared by bare `start`, `stop` and explicit
  shutdown (idle retirement keeps it so respawns re-pin). Respawners re-pass
  `--profile`. The pin is the **last** client tier — it only governs sessions
  configured nowhere (R5/R10).
- Bindings: reserved top-level `bindings` map (dir → profile) in
  `profiles.json`; innermost wins; Windows case-insensitive; `bindings` is a
  reserved profile name; `bind default` allowed; deleting a profile cascades
  its bindings (no resurrection on re-create).

**Shared server lifecycle.**
- No app owns it: the MCP `atexit` kill is gone. Spawned servers retire after
  `ROAMPAL_SERVER_IDLE_TIMEOUT_MINUTES` (default 30) with no requests
  (graceful drain); a foreground `roampal start` is exempt (R9).
- Health rule, same in all three clients: 200 = up; **503 = broken → restart**
  (the server's only 503s are broken states — R2); any other HTTP answer = up,
  never kill; connection failure = down. 404 = routing rejection: print the
  server's `detail`, no restart, no retry (R8). The UPS hook runs a ~2 ms
  preflight health GET and reacts only to an actual 503 (R10).
- Single-flight restart: `roampal/utils/proc_lock.py` — O_EXCL lock per port
  in the config dir (TS mirrors the path per platform), stale TTL **45 s**
  (must exceed the ~26 s worst-case restart), uuid token, ownership-checked
  release; TS creates the folder before the exclusive write (R4/R6).
- Spawn: neutral cwd (`profile_manager.neutral_spawn_dir()`, the data base
  dir); flags from `spawn_isolation_flags()` = `-E` (+ `-P` on 3.11+). Not
  `-I`: it implies `-s` and breaks Microsoft Store / `--user` installs (R1).
  Spawners strip `ROAMPAL_PROFILE`. `init`-written MCP/hook commands use the
  same flags. The plugin reuses the interpreter + flags from opencode.json
  (Task 41).
- `profile switch` == `use` + note; it never stops the server (Task 24).

**Memory length rules** (`roampal/memory_limits.py` is the single source;
checked once, on the server; TS constants asserted equal by a test).

| What | Target | Hard max | Over the max |
|---|---|---|---|
| Exchange summary (`score_memories`, sidecar) | ~300 chars, 1–2 sentences | 600 | rejected; main LLM adds a separate `record_response`; sidecar is re-asked once |
| Memory-bank entry (`add_to_memory_bank`, `update_memory`) | ~300, 1–2 sentences | 600 | rejected; extra goes in another `add_to_memory_bank` |
| Takeaway (`record_response`) | ~300, 1–2 sentences | 600 | rejected; extra goes in another `record_response` |
| Fact | one fact per item | 150 | rejected |
| Any other server write | — | 2,000 backstop | rejected; split |
| RECENT EXCHANGES / cold-start profile block | — | shows 300 | word-boundary cut, "…"; exchanges keep `[id:…]`; profile block ends with the "use search_memory" footer when clipped |
| KNOWN CONTEXT, `search_memory` | full text | — | never cut |
| Claude Code hook output | — | 10,000 platform cap | worst-case test stays under |
| Books | — | chunked 1,000 / 200 overlap | out of scope |

Rules line in every write-tool description and sidecar prompt: `Memory rules:
~300 chars, 1–2 sentences (max 600); facts ≤150 each. Over the max is rejected
— rewrite shorter.` Rejection messages: `Too long: {n}/600 chars. Rewrite in
~300 chars, 1–2 sentences.` + the tool's overflow line; facts `Too long:
{n}/150 chars. One fact per item, ≤150 chars.`; backstop `Too long: {n}/2000
chars. Split into separate memories, ~300 chars each.` MCP schemas carry **no
`maxLength`** — Claude Code validates the schema before the server, which
would hide our message (R11). Intentional non-memory caps: sidecar 8,000-char
input per side, 60-char `content_hint`, embedder 256-token window. Guard test
`TestNoSilentTruncation` fails the build on any new `[:N]` / `.slice(0, N)`
of memory text outside an allowlist with reasons.

**Context dedup (Claude Code only).** Server records surfaced memories per
conversation keyed `id:md5(content)[:12]` (both sides extract via
`normalize_memory`); a repeat renders as a one-line pointer with its id and
stays in the scoring list. Keys are held pending under an `injection_token`
until the UPS hook POSTs `/api/hooks/injection-ack` right before its zero exit
— no ack, full text next turn. Reset via `/api/hooks/session-reset` from the
SessionStart hook (startup/compact/**clear** matchers). Only requests with
`dedup_injections: true` (the UPS hook) are affected — OpenCode rebuilds its
prompt per request, so a pointer there would hide the memory.

**Config writes (`roampal/utils/safe_config.py`).** Read UTF-8 with BOM
tolerance; unreadable / invalid / non-object → `ConfigReadError`, file left
byte-identical, that tool skipped with the fix-it message, `init` exits 1.
Write: timestamped `.bak-YYYYmmddHHMMSS`, atomic replace, prune that file's
backups to the newest 3 (matches only `<name>.bak-<14 digits>`). Hooks merge
per event, removing only Roampal-owned commands (contain `roampal.hooks.` or
`roampal context`). Project `.mcp.json` only with `--scope project|both`, or
when an existing one already has `roampal-core`.

---

## Round 1 — 2026-09-16 (Tasks 0a–16)

- **Pre-flight.** Baseline: `cli.py` 4,697 lines, 17 subcommands, if/elif
  dispatch; `test_cli.py` 69 tests (not 84). **P1 correction:** CI *did*
  collect the WAL hard-kill test (root `pyproject` `testpaths =
  ["roampal"]`); the eval's "never collected" came from the nested
  `tests/pytest.ini` (`testpaths = unit`) — a standing footgun (same command,
  different collection by cwd), not fixed this release. Real gap: the 30
  non-WAL integration tests never ran on Windows CI. **P2:** editable-install
  version skew fixed with `pip install -e . --no-deps` (dev box only).
- **0a** `hasattr` guard on `close()` (mirrors `chromadb_adapter.py`).
  **0b** `integration-tests` + `integration-tests-windows` jobs with the
  explicit path. Lesson: an injected red-check named without `test_` is
  silently not collected.
- **1–2 Safety net.** 11 byte-goldens (`test_cli_golden.py`); `doctor` and
  `reembed --dry-run` are structural tests (they load models; machine-varying
  output; `doctor --json` never existed). Registry seeding format is
  top-level `name → path`. Dispatch invariants: parser set == dispatch set,
  no duplicate `cmd_*` (designed red on the dead `cmd_context` until Task 7),
  `NAMED_SUBCOMMANDS` table covers sub-subcommands. G4 was over-claimed:
  argparse auto-lists every command, so the epilog stays byte-identical
  (Task 4 no-op).
- **3–10 Split.** `cli.py` → `roampal/cli/` package (a package and same-named
  module cannot coexist). `__init__.py` is a PEP 562 proxy keeping `from
  roampal.cli import X`, `mock.patch("roampal.cli.X")`, `python -m
  roampal.cli` and the `roampal.cli:main` console script working; eager
  group-module import + mirror materialization (needed so `mock.patch`
  restores instead of deletes). Modules: `_common.py` (NO_INPUT accessors,
  colors, one shared logger, `_check_sidecar_configured`,
  `_get_opencode_config_path`, `get_data_dir`/ports/`_is_interactive`),
  `commands.py` (dispatch table, exit-code semantics copied from the if/elif),
  `update_check.py`, `setup.py`, `server.py`, `memory_cmds.py`,
  `sidecar/__init__.py` (largest, 1,143 lines), `profile_cmds.py`,
  `diagnose.py`, `_monolith_impl.py` (thin argparse entry; `parser.py` not
  split out to keep goldens byte-identical). Every moved function byte/AST
  verified. Unresolved-name audits caught latent NameErrors masked by broad
  `except`s and mocks (`platform`, `subprocess`, `os`, `_is_interactive`,
  `_stop_server_on_port`); the plugin source path moved one level
  (`Path(__file__).parent.parent`) and `test_plugin_install` had gone
  vacuous — both fixed.
- **11** Dedup guard v1 (md5 fake embedder + threshold/geometry negative
  controls) — later shown blind to real geometry (F4 → Task 29).
- **12–14** Bindings (see Contracts). `_quiet_bindings_map()` reads only the
  reserved key so loader warnings don't print twice (golden). `profile show`
  reports the matched folder as the reason; `list` gains a bindings section
  only when bindings exist.
- **15** Version 0.6.0 in `pyproject.toml` + `__init__.py`; README/
  ARCHITECTURE bindings, `reembed`, `help`. Golden harness pins child
  `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1` (cp1252 console round-trip).
- **16** Gates green (862 passed) — then reopened by the audit.

## Round 2 — 2026-09-18 audit (Tasks 17–37, R1–R16)

**Audit findings.** F1: the shared server resolved headerless requests from
*its own* cwd/env (inherited from whoever spawned it) → cross-profile writes.
F2: the spawning MCP killed the server on exit; hooks killed it on 503/timeout;
`profile switch` killed it (observed live: a forced kill mid-session, the
respawn had no session state). F3: CI would fail (proxy patched one holder
only; a test read real `~/.claude/projects`; Windows-only goldens). F4: dedup
guard couldn't see real-model geometry. F5: Claude Code hook output
accumulates and memories were re-injected. F6: CLI polish. F7: doc drift. F8:
the dev box's live apps ran the working tree (editable install).

**P3 (dev environment).** PyPI 0.5.9 installed into every interpreter; `-I`
added to the dev box's MCP/hook configs (later superseded by R1's `-E`
contract). Release eval installs the real wheel. Rollback:
`Python310\python.exe -m pip install -e C:\roampal-core --no-deps`.

**Routing (Item 6).**
- **21** `test_multi_profile_routing.py`: one real server with cwd in bound
  project A, four parallel clients (hook explicit default, OpenCode cwd header,
  env client, headerless) — red first on both F1 halves, green after 17–19.
- **17** `persisted_profile_fallback()`; server stops using its cwd/env;
  explicit `--profile` flag replaces pin-via-env; spawners strip
  `ROAMPAL_PROFILE`. Verified live with real spawned servers.
- **18** Always send the header; hooks read `ROAMPAL_PROFILE` from project
  `.mcp.json` and `~/.claude.json` (fixes a v0.5.4 gap).
- **19** Plugin `X-Roampal-Cwd`; server resolves it. Double-check: an
  unregistered named profile silently wrote to the default store (since
  v0.5.4) → now 404.
- **20** MCP profile cache removed; resolves per call.
- **R3** `~/.claude.json` project keys are `C:/proj` (forward slash, drive
  case varies) — lookup tries native, flipped and case-insensitive forms;
  user-scope env also read.
- **R4** Non-Latin paths crashed the plugin's header (Node ByteString) →
  percent-encode + server `unquote`.
- **R5** Pin made reachable (client-side tier; a server-side reorder was
  rejected by Task 21's contract). **R10** pin moved last in both server and
  client order; pin file records its pid and `profile show` surfaces it;
  explicit env `default` short-circuits before the pin in MCP and hooks.
- **R13** (smoke) CLI data commands (`stats`, `ingest`, `remove`, `books`,
  `summarize`, `retag`, `context` reset) sent no header; `ingest`'s offline
  fallback ignored profiles (since v0.5.1). AST guard
  `test_cli_profile_headers.py`: every `httpx` call under `roampal/cli/`
  carries headers except `/api/health` and the signup webhook.

**Lifecycle (Item 7).**
- **22** No atexit kill; idle self-retirement via `uvicorn.Server` handle +
  request-time middleware. **R9** foreground `start` exempt.
- **23** Health-first, down-only, single-flight restart in hooks, MCP and
  plugin. Double-check: plugin lock path differed from Python's on
  Windows/macOS (split single-flight) → mirrored. **R2** 503 = broken (see
  Contracts). **R6** TTL 45 s + ownership. **R8** 404 surfaced, no restart.
- **24** `switch` no longer kills.
- **36** Neutral cwd + isolation flags at every spawn site and in
  `init`-written commands; `validate_roampal_importable` tests under the same
  flags. **R1** `-I` → `-E [-P]` (user-site installs). Residual: 3.10 has no
  `-P`, so an `init`-written command run inside a folder containing a local
  `roampal/` still shadows (v0.5.9-equivalent; runtime spawns safe via the
  neutral cwd). The plugin passes only the flags recorded in opencode.json.
- **R10** health probe bypasses the embed cache (`skip_cache`) so a dead
  embedder no longer reads healthy forever; `PYTHONPATH` removed from
  written OpenCode env (the split made it point at the package dir, which
  would shadow the real `mcp` package without `-E`).

**Hermeticity + CI (Item 8).**
- **25** Proxy `__setattr__`/`__delattr__` rebind every holder (identity-
  filtered); eager `_materialize_all()` because `mock.patch.get_original()`
  reads `__dict__` directly.
- **26** Autouse `_sandbox_user_env` (HOME/USERPROFILE/APPDATA/LOCALAPPDATA/
  XDG_*/HF vars → tmp); `test_sandbox_probe.py` guard; sanctioned
  `unsandboxed_home` opt-in only for the HF-cache skip checks (so
  `test_reembed_dry_run_offline` skips in an empty isolated home). R12:
  `test_cli.py` wrote the repo's real `.mcp.json` → autouse temp cwd.
- **27** `status` reports `stopped` on `ConnectTimeout` (deliberate golden
  regen); normalizer masks `<REPOROOT>`, `<SITESP>`; golden job on Ubuntu +
  Windows.
- **29** Dedup guard on real mpnet-INT8: recorded real vectors
  (`dedup_real_vectors.json`, always runs) + live-model test (skip if
  uncached; CI caches `hf-mpnet-qint8-dedup-v1`); proven to collapse 3 facts
  → 2 at threshold 1.2 where the fake fixture passes. **Units:** ChromaDB l2
  on unit vectors returns the squared form `2−2cos` (audit F4 figures were
  plain L2).

**Memory quality (Items 10, 12).**
- **37 / 35** One limits module; server rejects instead of cutting (three
  silent `[:2000]` cuts removed; `/api/record-outcome` and
  `/api/memory/update-content` had no limits at all); `_api_call` surfaces the
  server's `detail`. Sidecar path: stop endpoint rejects >600 with
  `HTTPException(400)` (first cut built a wrong-field response → 500; caught in
  double-check); plugin re-asks the sidecar once, drops a second rejection,
  never queues it. Stop endpoint stores the summary alone (the v0.3.6 plan
  was always summary-only; the wrapper came from routing through the v0.2.9
  raw-exchange endpoint; old plugin copies are handled). `summarize` no longer
  cuts. RECENT EXCHANGES 200 → 300; profile block shows up to 300 chars.
  **Decision: no migration** of existing wrapped memories.
- **R11** schema `maxLength` removed; `score_memories` had swallowed record
  errors and claimed "Summary stored" → now returns the error; delivery ack.
- **30** Context dedup (see Contracts). Double-check: OpenCode would have lost
  content → `dedup_injections` flag; hash-source divergence fixed.
- **31** OpenCode prompt cache measured on 12 real sessions / 1,035 turns
  (opencode-go, auto prefix caching): 51.6% full prefix hits on large
  contexts, ~48% resets, cache writes billed 0. Likely driver: per-turn recency
  labels at the block's tail. **Decision: keep the block in the system
  prompt.** Re-measure if a provider bills cache writes or resets spike; only
  one provider family measured; `opencode run --pure` is broken upstream
  (1.15.7), so no A/B.

**CLI (Item 11).** **32** `dir(roampal.cli)` fixed; "(Task 13)" removed; `bind
default`; `bindings` reserved (legacy entry warned); `use`/`switch` warn when a
binding overrides them (default branches too); default binding not flagged
"not registered"; dead code removed. **R12** (from the real 0.5.9 → 0.6.0
`init --force`): missing `import urllib.request` had silently killed
Ollama/LM Studio detection → fixed + package-wide pyflakes undefined-name
guard (`pyflakes` in dev extras); `init --no-input` no longer shows the
picker; `--scope user` writes no project file. **R14 → R15** `roampal score`
had stored nothing since v0.5.6 (read old keys); fixed, then **removed**
(Claude Code-only v0.3.6 leftover, never wired by `init`, duplicated by
`score_memories`/the plugin; `sidecar test` covers model checks).
**R16** `profile delete` reports where each folder now resolves;
`switch --help` and `create` next-steps corrected.

**Docs.** **33** every F7 item closed (README overclaims softened,
ARCHITECTURE routing contract + Shared Server Lifecycle section,
`tests.yml` comment, glama 0.6.0).

## Round 3 — 2026-09-25/26 (Tasks 38–46)

- **38** Plugin scores only with a custom model or a recorded Zen opt-in
  (`ROAMPAL_SIDECAR_PRIORITY` contains `zen`); otherwise no calls, "scoring:
  off" notice; drainers gated. The v0.3.7 default (`if (!CUSTOM_SIDECAR_URL)`
  → Zen) had never been changed. `sidecar status` reports a Zen opt-in.
- **39** `sidecar setup --list [--json]` (stdout pure JSON, no keys; progress
  to stderr) and exactly-one-of `--model`, `--url --model [--key-env]`,
  `--go`, `--zen`, `--auto` (smallest local, never cloud; was a dead flag);
  shared option builder; choosing a model clears a Zen opt-in; errors exit 1
  and write nothing. Flags read strictly (MagicMock args are truthy).
- **40** `safe_config.py` (see Contracts). Pre-existing: unreadable
  `~/.claude.json`/`settings.json`/Cursor files were overwritten (Windows
  cp1252 read of UTF-8), user hooks replaced wholesale, no backups, 17
  opencode backups piled up, `.mcp.json` dropped in cwd. Double-check fixes:
  import-validation failure is not a config skip (continues, exit 0);
  `--scope user` doesn't read the project file; legacy
  `~/.claude/.mcp.json` migration stays best-effort and no longer deletes a
  file with other keys; non-dict `mcpServers` no longer crashes. Left out:
  sidecar read-only paths not switched to the safe reader.
- **41** `_loadMcpCommand()` + pure `resolveServerLaunch()` (user-global
  opencode.json only — project configs not consulted, sanctioned); sibling
  `pythonw.exe` on Windows, else configured `python.exe` via the VBS path
  with the full path quoted; no flags recorded → `-E`; PATH fallback only
  when missing, logged. Tested by running the sliced resolver under Node.
- **42** Help-golden layout normalizer (joins argparse-wrapped rows, collapses
  3+ spaces; goldens regenerated with `ROAMPAL_REGENERATE_GOLDEN=1` only);
  CI 3.10–3.13 everywhere + `unit-tests-macos`; 3.13 classifier;
  `chromadb>=1.5.9,<2.0.0` (older ChromaDB reading a newer store is the
  hazard — the dev `.venv` had 1.5.1, upgraded); README 3.10+ / `-P` note /
  3.10 EOL Oct 2026. Dropping 3.10 is a later, pre-announced release.
- **43** `_check_book_exists` and `remove_book` skip ghosted chunks — a
  removed title can be re-ingested; removing it twice says "not found".
- **44** `init` asks "Keep it? [Y/n]" when a scoring model is configured
  (EOF/Ctrl+C keep; non-interactive keeps and says so); menu shows the current
  model; Skip no longer claims scoring was disabled.
- **45** OpenCode stored each summary twice since v0.4.8 (record-outcome's
  no-exchange-doc branch + `tryStoreSummary`); hidden by the 24 h cleanup and
  the 0.5.9 wrapper. Plugin no longer sends the summary with the scoring call;
  server `_merge_into_recent_summary` merges an identical summary in the same
  conversation within 600 s (covers old plugin copies). Dev box `main`: 4
  duplicates deleted after a backup; existing duplicates are not cleaned for
  users.
- **46** `get_stats` books = live chunks, `removed` added only when non-zero;
  CLI `books: N items (M removed, hidden)`. User-verified live.

## Task 34 — live gates (2026-09-26, all PASS)

- **Manual CLI smoke** (throwaway `smoke060`, scratch folder): profiles,
  server start/stop/status, ingest/books/remove, sidecar status/list/test,
  context/summarize, reembed + cleanup. Produced R13, R14/R15, R16, Task 43,
  Task 46. Steps 3/5 first ran against a **0.5.9 server** — the repo's
  gitignored `.mcp.json` points at `.venv`, which held PyPI 0.5.9 — rerun
  after `.venv` got an editable 0.6.0.
- **Live 1:** Claude Code and OpenCode in the same folder on different
  profiles, no cross-writes; found Task 45. **1b:** OpenCode honors a binding
  once a hand-added `ROAMPAL_PROFILE=main` was removed from the user's
  opencode.json (a v0.5.x-era edit that outranks bindings, as designed).
  **2:** after `roampal stop`, OpenCode restarted the server as
  `Python310\pythonw.exe -E -m roampal.server.main --port 27182`. **3:**
  quitting OpenCode (the launcher) left the server running for Claude Code.
- **Isolated-home suite** 1120/5 (vs dev box 1121/4 at the time; the extra
  skip is the sanctioned HF-cache check). **`--version`/`--help` smoke** for
  all 15 commands from a neutral cwd. Docs secret scan clean.
- **Dev box state now:** Python310 and repo `.venv` on 0.6.0 (`.venv`
  editable); Python313 still PyPI 0.5.9 (nothing points at it); registry
  ghost/main/research, `use` = `main`; `live060`/`smoke060` deleted.

## Deferred / residuals (not fixed in 0.6.0)

- Profile **names** with non-Latin characters register fine but would crash
  the header in the plugin and the Python hooks (needs encode/unquote in all
  three clients + a naming decision).
- 3.10 has no `-P` (see Task 36).
- Health checks that time out (2 s) count as down; the 2026-09-18 unexplained
  kill's cause is still unknown.
- Claude Code rewind fires no hook (a pointer may refer to a rewound turn).
- OpenCode plugin during a long server outage: the retry queues keep one
  entry per session (a newer exchange replaces the older one) and drop it
  after 3 attempts, noting it only in `roampal_plugin_debug.log`; entries
  still queued when OpenCode closes are lost. The model's
  `[roampal scoring: failed (N consecutive failures)]` tag appears only after
  two exchanges' scoring has been dropped. It points at the sidecar config
  even when the real cause is the Roampal server being down, and dropped
  summary stores don't count toward it. Missing retrieval (get-context fails)
  produces no tag at all. Mostly a 0.5.9 exposure (PATH-Python restarts,
  fixed by Task 41); with working restarts, outages should be short.
- Plugin fetches the Zen free-model list at load (no exchange text) and logs
  4 loads per OpenCode start.
- Nested `tests/pytest.ini` vs root `testpaths` footgun.
- `psutil` not in dev extras → `test_rss_soak` always skips (decide: add +
  gate the marker, or accept).
- Code/test cleanup: `kg_service` kwarg shim + its 3 tests;
  `count_successes_from_history` duplicated in `outcome_service.py` and
  `scoring_service.py`; `test_init_no_kg` duplicates `test_no_knowledge_graph`.
- Cosmetic: `start` pre-banner prints the base data dir; HF_TOKEN warning on
  `reembed`; `init` prints "Created data directory" when it exists; destroy
  confirmation needs the full word `yes`; sidecar sample exchange printed
  abbreviated; empty follow-up passed verbatim to the sidecar ("Your empty
  reply gives no signal"); unused `force` param in onboarding.
- Embedder upgrade (e5-base) waits for a dedup-threshold recalibration.

## Lessons (tooling)

- `monkeypatch.delenv` is a no-op for absent vars; wrap CLI calls that write
  `os.environ` in `patch.dict(os.environ)`.
- Never rewrite UTF-8 files through PowerShell string pipelines (mojibake);
  use the edit tool.
- Commands that import `httpx` function-locally can't be intercepted by
  patching a module attribute — patch `httpx.post` globally.
- Reading stdin inside a hook command must be fully best-effort (pytest
  captures stdin).
- Long 3.13 pytest runs on this box were killed by external console
  KeyboardInterrupts; a `SetConsoleCtrlHandler(None, True)` wrapper fixed it.
- Red-check every new guard (inject the defect, see it fail, revert).

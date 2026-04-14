# Roampal v0.4.9 — Sidecar Robustness + Regex Tag Removal

**Release Date:** April 14, 2026  
**Status:** Complete  
**Target:** PyPI

---

## Overview

v0.4.9 fixes critical sidecar robustness issues discovered in production: **Zen free models are blocked/rate-limited, causing "sidecar not working" popups that interrupt users**. The sidecar architecture needs fundamental improvements for reliable out-of-the-box experience. Three major changes:

1. **Robust backend selection** — Intelligent health tracking, circuit breakers, priority-based fallbacks
2. **Better failure handling** — Non-annoying notifications, once-per-session hints
3. **Wire TagService to sidecar LLM** — Connect `extract_tags()` function to memory system (critical v0.4.8 fix)

### Key Changes

| Change | Impact | User Benefit |
|--------|--------|--------------|
| **Robust backend selection** | Health tracking, circuit breakers, intelligent fallbacks | Sidecar works reliably, skips failed backends |
| **Better failure notifications** | Once-per-session hints, not annoying popups | Users know when sidecar fails without interruption |
| **Model size guidance** | Empty state shows model options with RAM requirements | Users know what to install without guessing |
| **Wire TagService to sidecar LLM** | Fixes v0.4.8 bug where tags weren't extracted | OpenCode sidecar extracts LLM tags matching benchmark |
| **`roampal retag` cleanup tool** | Re-extract tags via scoring model, spinner + preview | Users clean up memories on their own schedule |

---

## Detailed Changes

### 1. Wire TagService to Sidecar LLM (Critical v0.4.8 Fix)

**Problem discovered in audit:** v0.4.8 had `extract_tags()` function in `sidecar_service.py` but it was **never called**. `TagService` was initialized without LLM function, so all tag extraction fell back to regex. OpenCode plugin removed `noun_tags` expecting server-side LLM extraction, but server was doing regex-only.

**Changes:**
- Wire `sidecar_service.extract_tags` as `llm_extract_fn` to `TagService` in `unified_memory_system.py`
- Update initialization: `TagService(llm_extract_fn=extract_tags)`
- **Platform gate:** Only import sidecar_service when `ROAMPAL_PLATFORM=opencode`. Claude Code has no sidecar — the main LLM provides tags via MCP tools. Importing sidecar_service on Claude Code caused 20s timeout spam on every startup.
- Same gate applied in `tag_migration.py`
- Verify OpenCode plugin flow: plugin correctly doesn't send `noun_tags`, server now extracts LLM tags

**Sidecar reliability:**
- Custom endpoint retries up to 3 times on empty/failed responses (local models occasionally return empty)
- Localhost endpoints get full caller timeout (no clamping to 10s)
- `extract_tags` timeout: 8s → 20s, `extract_facts`: 15s → 30s
- Unhealthy retry path gives custom localhost full timeout (was 5s)
- CLI commands read sidecar config from opencode.json (env vars weren't available in terminal)
- Tested: 10/10 summarize, 10/10 extract_tags on local gpt-oss:20b after fixes

**Sidecar-dependent commands check upfront:**
- `roampal retag`, `roampal summarize`, `roampal score` all check for a configured sidecar before running
- If no sidecar is available (no custom URL, no Ollama, no API key), they print instructions to run `roampal sidecar setup`
- Commands never silently attempt and fail — they tell you what's missing
- `retag` endpoint calls `sidecar_service.extract_tags()` directly (bypasses platform gate since it's user-initiated)

**Why this matters:**
- Fixes architecture mismatch between code and benchmark
- OpenCode sidecar now actually extracts LLM tags as benchmark requires
- Tag quality improves from regex pattern matching to LLM semantic understanding
- Claude Code-only users get clear guidance instead of silent failures

### 2. Remove Regex Tag Extraction (Architecture Alignment)

**Problem:** Roampal Core had regex fallback for tag extraction, but benchmark uses LLM-only. Regex tags are low quality and create noise in TagCascade retrieval.

**Changes:**
- **Remove `extract_tags_regex()` function** from `tag_service.py`
- **Remove regex fallback** in `TagService.extract_tags()` — LLM extraction only
- **Update `store_working()`** to skip tagging if LLM extraction fails (no regex fallback)
- **Add `roampal retag` command** — General-purpose memory cleanup tool
- **Preserve existing tags in data** — Don't delete existing memories

**Why LLM-only matches benchmark:**
- Benchmark `entity_routed.py:extract_tags()` is LLM-only (30s timeout)
- LLM extracts semantic understanding vs regex pattern matching
- Consistent tag quality across all memories
- No regex noise polluting TagCascade retrieval

**`roampal retag` — Memory Cleanup Tool:**

Long-term tool for cleaning up and improving tag quality. Not a migration tool — use whenever you want fresher tags.

- Sends each memory to your scoring model for fresh tag extraction
- Existing tags are **replaced** with new ones; memories without tags get them added
- `--dry-run` previews without changing anything
- `--limit N` tests on a few memories first
- Shows spinner during processing and sample before/after tag changes
- **Requires a scoring model** — checks upfront and gives setup instructions if not configured
- Calls sidecar directly (works regardless of platform — Claude Code or OpenCode)

```bash
roampal retag --dry-run --limit 10    # Preview
roampal retag --collection working    # Retag working memories
roampal retag                         # Retag everything
```

**Impact:**
- **OpenCode:** Sidecar now actually extracts LLM tags (fixed from v0.4.8)
- **Claude Code:** Main LLM provides tags via `score_memories` MCP tool
- **If LLM fails:** Returns empty list (matches benchmark, no regex fallback)

### 2. Robust Sidecar Backend Selection

**Problem:** Zen free models timeout/return HTTP 403, causing "sidecar not working" popups that interrupt user typing. Users see repetitive notifications with no easy fix.

**Solution:** Intelligent backend health tracking with graceful degradation.

**Backend Priority (configurable via `ROAMPAL_SIDECAR_PRIORITY`):**
1. **Custom endpoint** (user-configured via `roampal sidecar setup`) — exclusive if set
2. **Zen free models** — fast-fail with circuit breaker (5s timeout, skip if failed recently)
3. **Ollama local** — generous timeout for cold starts (30s)
4. **LM Studio local** — fallback for local OpenAI-compatible servers

**Health Tracking:**
- Each backend tracks: success rate, response time, consecutive failures
- Circuit breaker: skip backends with >3 consecutive failures for 5 minutes
- Automatic recovery: retry unhealthy backends after cooldown period
- Score-based selection: prefers reliable, fast backends

**OpenCode Plugin Improvements:**
- **Non-annoying notifications:** Once-per-session hints instead of every exchange
- **Helpful messages:** "Free models may be rate-limited. Run `roampal sidecar setup` for local model."
- **Minimal status:** After first notification, shows `[roampal scoring: unavailable]` instead of full message
- **No scary errors:** Removed "IMPORTANT — roampal scoring: BROKEN" panic messages

**Why this matters:**
- Users aren't interrupted by repetitive popups
- Sidecar automatically finds working backend
- Clear path to fix when free models fail
- Graceful degradation instead of complete failure

### 2. Respect User Configuration (No Silent Cascades)

**Problem:** Previous versions would silently fall back from custom endpoints to Zen/Ollama/LM Studio without user consent.

**Solution:** If user configures custom endpoint (ROAMPAL_SIDECAR_URL), we use ONLY that endpoint.

**Behavior:**
- **Custom endpoint configured:** Use ONLY that endpoint, no fallbacks
- **Custom endpoint fails:** Return error, user must fix configuration
- **No custom configured:** Smart cascade (zen → ollama → lmstudio) for users without setup
- **User intent respected:** No surprises, no using local models without consent

**Code change:**
```python
if CUSTOM_URL and CUSTOM_MODEL:
    # User explicitly configured custom - use ONLY this
    result = _call_custom(prompt)
    if not result:
        # FAIL: No silent fallback to Zen/Ollama/LM Studio
        logger.error("Custom endpoint failed. User must fix configuration.")
        return None
    return result
else:
    # No custom configured - use smart cascade
    # This is for users who haven't explicitly set up anything
```

### 3. Unified One-Step Onboarding

**Problem:** Previous onboarding had 3 separate code paths (init, sidecar setup, auto-setup) with inconsistent UX. Users had to navigate a preamble menu before even seeing their options. API models were auto-selected without asking.

**Solution:** One function (`_sidecar_model_picker`) used by both `roampal init --opencode` and `roampal sidecar setup`. Auto-detects everything, shows one numbered list, user picks. No preamble menu.

**Flow when models are found:**
```
$ roampal init --opencode

Memory scoring setup:
  Roampal learns what works by scoring exchanges in the background.
  This requires a small AI model — it doesn't need to be smart.

Scanning for available models...

Available scoring models:
  (Sidecar only needs a small model for JSON summaries)

  [1] qwen3:8b (Ollama, 5.2GB, free)  <-- recommended
  [2] gpt-oss:20b (Ollama, 13GB, free)
  [3] gemma4:31b (Ollama, 19GB, free)
  [4] DeepSeek V3 (DeepSeek) (API, costs money)

  [5] Configure custom API endpoint
  [6] Use free community models (may be rate-limited)
  [7] Skip for now

Choose [1-7]:
```

**Flow when nothing is detected:**
```
Scanning for available models...

  No local models or API keys detected.

  You can use local models from Ollama, LM Studio, or similar.
  Pick a model based on your available RAM:

    qwen3:8b     — 4.9GB download, needs ~8GB RAM (best quality)
    llama3.2:3b  — 1.8GB download, needs ~4GB RAM (lighter)
    tinyllama    — 0.6GB download, needs ~2GB RAM (minimal)

    Ollama:     https://ollama.com → ollama pull <model>
    LM Studio:  https://lmstudio.ai
    Then run:   roampal sidecar setup

  [1] Configure custom API (Groq, DeepSeek, etc.)
  [2] Use free community models (may be rate-limited)
  [3] Skip for now

Choose [1-3]:
```

**Key design decisions:**
1. **One code path** — `_sidecar_model_picker()` replaces `_prompt_smart_onboarding`, `_run_auto_setup`, and the old `_cmd_sidecar_setup` menu logic
2. **Local models first** — Ollama sorted smallest-first (sidecar doesn't need a big model)
3. **No auto-selection** — user always picks from the list
4. **API labeled "costs money"** — transparent about paid vs free options
5. **No system scanning** — model list with sizes/RAM requirements lets users decide
6. **Embedding models filtered** — `nomic-embed-text` and similar excluded from options

---

## Architecture Alignment: LLM-Only Tags

### Before (v0.4.8): Broken LLM Wiring + Regex Fallback
```
TagService.extract_tags(text):
  1. Try LLM extraction (BUT: llm_extract_fn=None, so never called!)
  2. Always falls back to regex extraction
  3. Regex extracts capitalized words, quoted strings
  
store_working(content, noun_tags=None):
  1. Use provided noun_tags if available
  2. If None → call TagService.extract_tags(content)
  3. Gets ONLY regex tags (LLM never wired!)
  
OpenCode plugin:
  1. Removes noun_tags expecting server-side LLM extraction
  2. Server does regex-only extraction
  3. Result: No LLM tags in v0.4.8
  
Migration (v0.4.8):
  1. Find facts with empty noun_tags
  2. Re-tag using TagService.extract_tags()
  3. Gets ONLY regex tags (LLM never wired!)
```

### After (v0.4.9): Wired LLM + No Regex Matching Benchmark
```
TagService.extract_tags(text):
  1. Try LLM extraction (sidecar.extract_tags wired via llm_extract_fn)
  2. If LLM fails → return [] (no tags)
  3. NO REGEX FALLBACK
  
store_working(content, noun_tags=None):
  1. Use provided noun_tags if available  
  2. If None → call TagService.extract_tags(content)
  3. Gets LLM tags or [] (no tags)
  
OpenCode plugin:
  1. Doesn't send noun_tags (correct)
  2. Server extracts LLM tags via wired TagService
  3. Result: LLM tags matching benchmark
  
Migration (v0.4.9):
  1. Find facts with empty noun_tags
  2. If sidecar available → use LLM extraction
  3. If sidecar unavailable → skip (don't use regex)
```

### Why This Matters
1. **Benchmark fidelity:** Matches `entity_routed.py:extract_tags()` exactly
2. **Tag quality:** LLM understands semantics vs regex pattern matching
3. **TagCascade performance:** No regex noise polluting retrieval
4. **Architecture purity:** One way to extract tags (LLM), not two

### Code Changes
```python
# BEFORE (v0.4.8): Broken wiring + regex fallback
# unified_memory_system.py:456
self._tag_service = TagService()  # NO llm_extract_fn!

# tag_service.py:242
def extract_tags(self, text: str) -> List[str]:
    if self._llm_extract_fn:  # ALWAYS FALSE in v0.4.8!
        try:
            tags = self._llm_extract_fn(text)
            if tags:
                return self._normalize_llm_tags(tags)
        except Exception as e:
            logger.debug(f"LLM tag extraction failed, using regex: {e}")
    
    return self.extract_tags_regex_and_register(text)  # ALWAYS REGEX

# AFTER (v0.4.9): Wired LLM + no regex
# unified_memory_system.py:456
from roampal.sidecar_service import extract_tags
self._tag_service = TagService(llm_extract_fn=extract_tags)  # WIRED!

# tag_service.py:242  
def extract_tags(self, text: str) -> List[str]:
    if not self._llm_extract_fn:
        return []  # NO LLM, NO TAGS
    
    try:
        tags = self._llm_extract_fn(text)
        if tags:
            return self._normalize_llm_tags(tags)
    except Exception as e:
        logger.debug(f"LLM tag extraction failed: {e}")
    
    return []  # NO REGEX FALLBACK
```

---

## Technical Implementation

### New Dependencies
- `requests` for HTTP downloads (already in dependencies)
- `tqdm` for progress bars (optional, graceful fallback)

### Platform-Specific Installers
```python
# Windows
OLLAMA_WINDOWS_URL = "https://ollama.com/download/OllamaSetup.exe"
# macOS  
OLLAMA_MAC_URL = "https://ollama.com/download/Ollama-darwin.zip"
# Linux
OLLAMA_LINUX_URL = "https://ollama.com/download/ollama-linux-amd64"
```

### Installation Verification
```python
def verify_ollama_installation():
    """Check Ollama is installed and running."""
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        # Test API connection
        requests.get("http://localhost:11434/api/version", timeout=5)
        return True
    except:
        return False
```

### Model Pull with Progress
```python
def pull_model_with_progress(model_name: str):
    """Pull Ollama model with progress tracking."""
    process = subprocess.Popen(
        ["ollama", "pull", model_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Parse Ollama's progress output
    for line in process.stdout:
        if "pulling" in line and "%" in line:
            # Extract percentage: "pulling manifest... 100%"
            display_progress(line)
    
    return process.wait() == 0
```

---

## User Experience Improvements

### 1. First-Time User Flow
```
$ pip install roampal
$ roampal init --opencode

🔍 Memory scoring setup:

[1] Smart recommendations (--auto)
    Analyze my system, suggest best options

[2] Use free cloud models  
    May be rate-limited, works for basic testing

[3] Manual setup
    I already have Ollama/LM Studio/API keys

[4] Skip for now
    No memory scoring, just basic functionality

Choose [1-4]: 1

🔍 Smart setup - analyzing your system...
💡 Recommendation: Install Ollama + qwen3:8b
  Powerful system (16+ GB RAM) - best quality model
  Model: qwen3:8b (4.9GB download)
  RAM: 64.0GB (good for this model)
  Disk: 128.0GB free

Manual install required:
  1. Visit https://ollama.com
  2. Download and install Ollama
  3. Run: ollama pull qwen3:8b
  4. Run: roampal sidecar setup

For now, using free community models.
```

### 2. Existing User Upgrade
```
$ roampal sidecar status
# "No sidecar configured - using Zen only"
$ roampal sidecar setup --auto

🔍 Smart setup - analyzing your system...
💡 Recommendation: Use existing API (Groq)
  You have Groq API key configured - fastest option

✅ Using existing Groq API for scoring.
```

### 3. Low Specs Scenario
```
$ roampal sidecar setup --auto

🔍 Smart setup - analyzing your system...
💡 Recommendation: Use free cloud models
  Low specs (RAM: 3.2GB, Disk: 1.8GB free) - use free cloud models

⚠️  Your system doesn't meet minimum requirements for local models.
  Option 1: Use free cloud models (may be rate-limited)
  Option 2: Free up disk space and try again
  Option 3: Manual setup with custom API

Choose [1-3]: 1

✅ Using free community models.
  Note: May see "sidecar unavailable" during peak times.
```

---

## Security Hardening (v0.4.9)

### SSL/TLS Certificate Verification
All external HTTPS calls now use `ssl.create_default_context()` for certificate verification:
- Anthropic API calls (`api.anthropic.com`)
- Zen free model calls (`opencode.ai`)
- Custom endpoint calls (when using HTTPS)
- Localhost calls (Ollama, LM Studio) correctly skip SSL since they use HTTP

This prevents MITM attacks on API key-bearing requests.

### Input Validation on /api/retag
- **Collection names**: Validated against whitelist (`working`, `history`, `patterns`, `memory_bank`, `all`). Invalid names return 422.
- **Limit parameter**: Capped at 5000 (minimum 1) to prevent DoS via unbounded memory processing.
- **Model name**: Max 100 characters to prevent abuse.

### URL Validation for ROAMPAL_SIDECAR_URL
Custom sidecar URLs are validated at module load:
- Only `http://` and `https://` schemes accepted
- `file://`, `ftp://`, and other protocols are rejected (SSRF prevention)
- Malformed URLs are logged and ignored

### Information Disclosure Fixes
- `get_sidecar_status()` returns booleans for model/config fields instead of raw strings
- Prevents leaking model names, API endpoints, and priority config through status checks

### Process Management
- PID validation added to `kill_server()` — only numeric PIDs passed to `taskkill`/`kill`
- Bare `except:` clauses replaced with specific exception types (`Exception`, `json.JSONDecodeError`, `TypeError`)

### Dependency
- No new dependencies added (psutil removed — hardware scan dropped)

---

## Testing Checklist

### Critical Fix Tests
- [x] `TagService` wired with `llm_extract_fn=extract_tags` 
- [x] `extract_tags_regex()` function removed
- [x] `TagService.extract_tags()` returns [] on LLM failure
- [x] `store_working()` stores memories with LLM tags when sidecar available
- [x] `store_working()` stores memories with [] tags when LLM fails
- [x] v0.4.8 migration uses sidecar LLM or skips (no regex)
- [x] OpenCode stores facts with LLM tags (not regex)
- [ ] Claude Code `score_memories` requires noun_tags from AI
- [x] TagCascade retrieval works with LLM-only tags
- [x] Regex tag tests removed from test suite (9 stale tests deleted, 18 remaining pass)

### Security Tests
- [x] SSL context applied to all external urllib calls
- [x] Retag endpoint validates collection names against whitelist
- [x] Retag endpoint caps limit at 5000
- [x] Custom URL validated for safe schemes (http/https only)
- [x] PID validated as numeric before subprocess calls
- [x] Bare except clauses replaced with specific exceptions
- [x] Status endpoint returns booleans not raw config strings
- [x] psutil removed (no hardware scan)

### Model Detection Tests
- [x] Existing Ollama models detected and listed
- [x] Existing LM Studio models detected and listed
- [x] Existing API models detected from opencode.json providers
- [x] Embedding models (nomic-embed-text etc.) filtered from list
- [x] Models sorted smallest-first for sidecar recommendation

### Integration Tests
- [ ] Sidecar scoring works with new model
- [ ] `roampal sidecar test` passes
- [ ] OpenCode plugin recognizes new config
- [ ] Memory scoring happens automatically
- [x] LLM tag extraction works (wired correctly, no regex fallback)
- [x] Graceful degradation when LLM unavailable

### Edge Cases
- [ ] No internet connection
- [ ] Insufficient permissions
- [ ] Sidecar LLM timeout during tag extraction
- [ ] Migration with unavailable sidecar
- [x] TagService with no llm_extract_fn returns [] (not regex)

### Unit Test Results
- **454 passed**, 2 pre-existing failures (health endpoint route — unrelated)
- 18 tag service tests pass (9 stale regex tests removed)
- Zero regressions from security fixes

---

## Backward Compatibility

### Breaking Changes
- None — all new features are opt-in

### Deprecations  
- None

### Migration Notes
- Existing sidecar configurations unchanged
- Zen fallback still default
- Model picker shows detected options, user always picks

---

## Performance Impact

### Tag Extraction Impact
- **Before (v0.4.8):** Regex-only (LLM never wired) → low-quality tags
- **After (v0.4.9):** LLM-only (wired correctly) → higher quality tags
- **OpenCode:** Now actually uses sidecar LLM for tags (FIXED)
- **Claude Code:** Main LLM must extract tags (already should)
- **Migration:** Requires sidecar LLM availability

### Model Size Awareness
- qwen3:8b: 4.9GB download, ~8GB RAM during inference
- llama3.2:3b: 1.8GB download, ~4GB RAM during inference  
- tinyllama:1.1b: 0.6GB download, ~2GB RAM during inference
- **Recommendations:** Match model size to available RAM/disk

### Runtime Impact
- No performance change for existing users
- Local scoring faster than Zen (2-5s vs 5-15s)
- Memory usage: +~6GB RAM during model load
- **Tag quality:** Improved (LLM semantic vs regex pattern)

---

## Documentation Updates

### CLI Help
```bash
$ roampal sidecar setup --help
usage: roampal sidecar setup [-h] [--auto] [--model MODEL]

options:
  -h, --help           show this help message and exit
  --auto               Smart hardware-aware recommendations
  --model MODEL        Specify which model to use
```

### README Updates
- Add "Smart Setup Recommendations" section
- Update quick start with hardware-aware advice
- Add troubleshooting for model selection

### Website Updates
- New "Smart Model Selection" tutorial
- Video demo of hardware detection
- System requirements guidance

---

## Future Considerations

### Potential Enhancements
1. **Model compression awareness** - Recommend quantized models for low-spec systems
2. **Performance benchmarking** - Test models on user's hardware before recommendation
3. **Community model registry** - User-submitted performance data for different hardware
4. **Auto-detection improvements** - Better detection of existing model installations

### Integration Opportunities
1. **OpenCode plugin store** - Easy Roampal installation
2. **Docker image** - Pre-configured for testing
3. **System package** - `apt install roampal` for Linux users

---

## Release Timeline

### Phase 1: Testing (1 day)
- Test hardware detection on all platforms
- Verify recommendation accuracy
- Test robust backend selection

### Phase 2: Documentation (1 day)  
- Update CLI help and documentation
- Create tutorial for smart setup
- Update website with new features

### Phase 3: Release (1 day)
- Update PyPI package
- Update documentation
- Announce on Discord/Twitter

**Total:** 3 days to release

---

## Metrics for Success

### Critical Fix Metrics
- **LLM wiring success:** % of memories getting LLM tags vs regex in v0.4.8
- **Tag quality:** LLM vs regex tag comparison (manual audit)
- **Tag coverage:** % of memories with LLM-extracted tags
- **Migration success:** % of facts successfully re-tagged with LLM
- **Retrieval impact:** TagCascade performance with LLM-only tags

### Smart Recommendation Metrics
- **Adoption rate:** % of users using `--auto` for recommendations
- **User satisfaction:** Post-recommendation survey
- **Support tickets:** Reduction in "what model should I use?" questions

### Secondary Metrics  
- **Scoring latency:** Improvement vs Zen
- **User satisfaction:** Post-install survey
- **Retention:** Users with local scoring vs Zen-only
- **Architecture alignment:** Benchmark vs Roampal tag extraction match

---

## Risk Mitigation

### Critical Fix Risks
1. **LLM wiring fails** → Test that `llm_extract_fn` is properly passed
2. **LLM tag extraction fails** → No tags (graceful degradation)
3. **Migration with unavailable sidecar** → Skip, don't use regex
4. **TagCascade performance drop** → Monitor retrieval metrics
5. **Claude Code AI doesn't provide tags** → Prompt improvements

### Smart Recommendation Risks
1. **Hardware detection fails** → Fall back to interactive setup
2. **Incorrect recommendations** → Conservative defaults (under-promise)
3. **Missing existing installations** → Manual option always available

### User Experience Risks
1. **Confusing options** → Simplified menu, clear recommendations
2. **Platform differences** → Clear OS-specific advice
3. **No tags on some memories** → Educate about LLM requirement
4. **Expectation mismatch** → Clarify recommendations ≠ auto-install

### Business Risks
1. **Increased support load** → Clear "manual install" instructions
2. **User frustration** → Set proper expectations upfront
3. **Benchmark deviation reversion** → Document architecture alignment

---

## Conclusion

v0.4.9 fixes a critical v0.4.8 bug and achieves three goals:

1. **Critical bug fix:** Wire TagService to sidecar LLM, fixing v0.4.8 architecture mismatch where OpenCode sidecar wasn't extracting LLM tags as benchmark requires.

2. **Architecture alignment:** Remove regex tag extraction, matching benchmark's LLM-only approach for consistent tag quality and TagCascade performance.

3. **Smart recommendations:** Hardware-aware model suggestions help users make informed decisions about what their system can run.

This release fixes a silent bug that undermined tag quality in v0.4.8 while making Roampal both architecturally pure (matching benchmark) and user-friendly (better guidance). The LLM wiring ensures tag quality matches research findings, while smart recommendations help users choose appropriate models.

This release represents our commitment to both scientific rigor (benchmark fidelity) and user control (informed decisions, not auto-install). Users get persistent, outcome-based memory with proper LLM tags while maintaining full control over their system setup.

---

**Prepared by:** Logan Teague  
**Date:** April 14, 2026  
**Next Steps:** Begin implementation, starting with Windows installer
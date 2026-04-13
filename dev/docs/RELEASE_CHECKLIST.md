# Roampal Release Checklist

Step-by-step process for buttoning up, verifying, and shipping a release. Follow in order.

---

## Phase 1: Code Freeze

### Version Bump
- [ ] `roampal/__init__.py` — update `__version__`
- [ ] `pyproject.toml` — update `version` field (must match `__init__.py`)
- [ ] Verify: `python -c "from roampal import __version__; print(__version__)"` matches pyproject.toml

### Unit Tests
```bash
python -m pytest roampal/backend/modules/memory/tests/unit/ -v
```
- [ ] All tests pass (currently 457+)
- [ ] No skipped tests without documented reason

### Integration Tests (mock embeddings)
```bash
cd dev/tests/integration
python test_comprehensive.py
python test_torture_suite.py
```
- [ ] Comprehensive: 31/31
- [ ] Torture suite: 10/10

### Benchmarks (real embeddings, optional for patch releases)
```bash
cd dev/benchmarks
set HF_HUB_OFFLINE=1 && python test_comprehensive_benchmark.py
set HF_HUB_OFFLINE=1 && python test_learning_curve.py
set HF_HUB_OFFLINE=1 && python test_roampal_vs_vector_db.py
set HF_HUB_OFFLINE=1 && python test_outcome_learning_ab.py
set HF_HUB_OFFLINE=1 && python test_token_efficiency.py
```
- [ ] Run sequentially (parallel hits HuggingFace rate limits)
- [ ] Use `HF_HUB_OFFLINE=1` if models already cached
- [ ] Update `dev/benchmarks/results/SUMMARY.txt` if numbers changed
- [ ] Update `dev/benchmarks/README.md` if numbers changed

---

## Phase 2: Documentation Audit

### PII Scan
```bash
# Run from project root - check tracked files only
git grep -i "logan\|logte\|C:\\\\Users\\\\logte" -- "*.py" "*.md" "*.toml" "*.json"
```
- [ ] No personal names in examples (use "Alex" for placeholder names)
- [ ] No personal file paths (C:\Users\username)
- [ ] No personal email addresses (use roampal@protonmail.com for project)
- [ ] No API keys, tokens, or credentials

### Files to Check
- [ ] `README.md` — accurate features, no PII, version current
- [ ] `ARCHITECTURE.md` — no internal debug logs, no PII
- [ ] `pyproject.toml` — version, description, URLs correct
- [ ] `dev/benchmarks/README.md` — numbers match latest runs, Limitations section present
- [ ] `dev/benchmarks/results/SUMMARY.txt` — consistent with README
- [ ] `dev/docs/releases/vX.Y.Z/RELEASE_NOTES.md` — exists, accurate

### Link Verification
- [ ] GitHub repo: `https://github.com/roampal-ai/roampal-core`
- [ ] Discord: `https://discord.com/invite/F87za86R3v`
- [ ] PyPI: `https://pypi.org/project/roampal/`
- [ ] No stale links to wrong repos or old URLs

### Platform Claims
- [ ] Only advertise platforms that actually work
- [ ] Currently supported: Claude Code, OpenCode
- [ ] Cursor: code ready but blocked by upstream bug — do NOT advertise as working

---

## Phase 3: Security Review

### Secrets Check
- [ ] No API keys or tokens in tracked files
- [ ] `.gitignore` includes: `.claude/`, `.mcp.json`, `.env`, `credentials.json`
- [ ] Webhook URL in cli.py is write-only (append to sheet, can't read)
- [ ] No hardcoded passwords or secrets

### Untracked Files
```bash
git status
```
- [ ] Review untracked files — nothing sensitive about to be committed
- [ ] `.mcp.json` (local dev config with user paths) is NOT staged
- [ ] `.claude/` directory is NOT staged
- [ ] No temp files with user paths (e.g., `Userslogte*.json`)

### Dependency Audit
- [ ] No known CVEs in pinned dependencies
- [ ] `chromadb>=1.0.0,<2.0.0` — pinned to major version
- [ ] All deps have version lower bounds

---

## Phase 4: Fresh Install Test

### Simulate New User
```python
# Back up existing configs
cp ~/.claude/settings.json ~/.claude/settings.json.bak
cp ~/.claude.json ~/.claude.json.bak
cp ~/.config/opencode/opencode.json ~/.config/opencode/opencode.json.bak
cp ~/.config/opencode/plugins/roampal.ts ~/.config/opencode/plugins/roampal.ts.bak

# Clean roampal entries (Python script or manual)
# Remove hooks, MCP entries, permissions, plugin

# Run init
python -m roampal.cli init

# Verify
# - Claude Code: hooks, MCP server, 7 permissions
# - OpenCode: MCP config with PYTHONPATH, plugin with default export
# - Data directory created

# Restore
cp ~/.claude/settings.json.bak ~/.claude/settings.json
cp ~/.claude.json.bak ~/.claude.json
cp ~/.config/opencode/opencode.json.bak ~/.config/opencode/opencode.json
cp ~/.config/opencode/plugins/roampal.ts.bak ~/.config/opencode/plugins/roampal.ts
```

### Verify Email Collection
- [ ] Email prompt appears after "initialized successfully"
- [ ] Enter to skip works cleanly (blank line, no crash)
- [ ] Test email reaches Google Sheet (check "Roampal Users" sheet)
- [ ] Invalid email shows warning and skips

### Verify Doctor
```bash
python -m roampal.cli doctor
```
- [ ] All checks pass
- [ ] Dependencies present
- [ ] MCP server module loads
- [ ] Memory system initializes

---

## Phase 5: Build & Publish

### Git
```bash
# Stage specific files (never git add -A)
git add roampal/ pyproject.toml README.md ARCHITECTURE.md dev/docs/ dev/benchmarks/ .gitignore

# Review what's staged
git diff --cached --stat

# Commit
git commit -m "v0.3.2: <description>"

# Tag
git tag v0.3.2

# Push
git push && git push --tags
```

### PyPI
```powershell
# Clean previous builds
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Build
python -m build

# Upload (use token from PyPI account)
twine upload dist/* -u __token__ -p <PYPI_TOKEN>
```

### Post-Publish Verification
- [ ] `pip install roampal==0.3.2` works in a clean venv
- [ ] `roampal --help` shows commands
- [ ] `roampal init` runs without errors
- [ ] PyPI page (https://pypi.org/project/roampal/) shows correct version and description
- [ ] README renders correctly on PyPI

---

## Phase 6: Post-Release

### Glama Release
- [ ] Update `Dockerfile.glama-test` — change `git checkout` to new release commit hash
- [ ] Go to https://glama.ai/mcp/servers/roampal-ai/roampal-core/admin/dockerfile
- [ ] Click **Deploy** — wait for build test to pass
- [ ] Click **Make Release** — enter version number and changelog
- [ ] Verify server score at https://glama.ai/mcp/servers/roampal-ai/roampal-core/score

### Announce
- [ ] Discord: Post changelog summary
- [ ] GitHub: Create release with notes (can link to RELEASE_NOTES.md)

### Monitor
- [ ] Check PyPI download stats after 24h
- [ ] Watch GitHub Issues for install problems
- [ ] Check "Roampal Users" Google Sheet for new signups

---

## Quick Reference

| Item | Location |
|------|----------|
| Version (code) | `roampal/__init__.py` |
| Version (package) | `pyproject.toml` |
| Unit tests | `roampal/backend/modules/memory/tests/unit/` |
| Integration tests | `dev/tests/integration/` |
| Benchmarks | `dev/benchmarks/` |
| Release notes | `dev/docs/releases/vX.Y.Z/RELEASE_NOTES.md` |
| PyPI token | Your PyPI account (never stored in repo) |
| Email sheet | Google Sheets "Roampal Users" |
| Webhook URL | `roampal/cli.py` SIGNUP_WEBHOOK_URL constant |

## Known Gotchas

1. **HuggingFace rate limits**: Run benchmarks sequentially with `HF_HUB_OFFLINE=1`
2. **numpy bool_ serialization**: Always wrap numpy bools with `bool()` before JSON encoding
3. **success_count consistency**: When setting benchmark metadata, always set `score`, `uses`, `success_count`, `outcome_history`, and `last_outcome` together
4. **Google Apps Script auth**: Must authorize script once (Run manually) before anonymous POST works
5. **urllib vs httpx**: Google Apps Script redirects break Python's urllib; use httpx with `follow_redirects=True`
6. **Cursor**: Code supports it but don't advertise — blocked by Cursor v2.4.7 bug

# v0.4.3 Release Notes

**Platforms:** Claude Code, OpenCode
**Theme:** Lightweight install — drop PyTorch, go pure ONNX

---

## Planned

### 1. Add negative example to `record_response` tool description
**File:** MCP tool definition for `record_response`

**Problem:** LLMs sometimes use `record_response` for permanent preferences/standing rules that belong in `add_to_memory_bank`. The tool description explains what `record_response` IS for but doesn't say what it ISN'T for.

**Fix:** Add a negative example to the `record_response` description: "NOT for permanent preferences or standing rules — use add_to_memory_bank for those."

---

## Shipped

### 2. ONNX Migration: Drop PyTorch/sentence-transformers
**Files:** `embedding_service.py`, `pyproject.toml`, `test_embedding_service.py`, `cli.py`, `server/main.py`, `__init__.py`

**What changed:** Replaced `sentence-transformers` + PyTorch with direct ONNX Runtime inference (`onnxruntime` + `tokenizers` + `numpy`). The embedding model is the same `paraphrase-multilingual-mpnet-base-v2`, loaded as an optimized ONNX graph (`model_O4.onnx`, 529MB) instead of a PyTorch model.

**Why:** Install size drops dramatically — torch alone was ~420MB+, and its transitive deps (sympy, filelock, jinja2, etc.) added ~100-200MB more. The replacement deps (`onnxruntime`, `tokenizers`, `huggingface-hub`, `numpy`) are all already installed as transitive dependencies of `chromadb`, so the net new install size is zero. Docker images shrink accordingly. No more CUDA/torch version conflicts.

**Impact:** Same model, same 768-dimension vectors, same ChromaDB collections — zero user-facing change. Existing embeddings remain compatible. Verified: cosine similarity between ONNX and PyTorch embeddings is 1.0000000 (max element diff: 0.0000001). Multilingual support confirmed across 8 languages.

**Details:**
- `embedding_service.py`: Full rewrite — uses `onnxruntime.InferenceSession` for inference, `tokenizers.Tokenizer` for tokenization, numpy for mean-pooling + L2-normalization. Same public interface (`embed_text`, `embed_texts`, `prewarm`, `get_embedding_dimension`), same LRU cache, same async `to_thread` wrapping.
- `pyproject.toml`: Removed `sentence-transformers>=3.2.0` from core deps. Added `onnxruntime>=1.14.0`, `tokenizers>=0.14.0`, `huggingface-hub>=0.20.0`, `numpy>=1.24.0`. Removed `[onnx]` optional group (now default). Added `[pytorch]` optional group for legacy fallback.
- `cli.py`: `roampal doctor` now checks `onnxruntime` + `tokenizers` instead of `sentence-transformers`. Added hint when torch is installed but no longer needed.
- `server/main.py`: Updated ImportError message to reference new deps.
- `__init__.py`: Updated comment, bumped version to 0.4.3.
- Model auto-downloads via `huggingface_hub.hf_hub_download()` on first use, cached in `~/.cache/huggingface/`.

---

## Files Modified

| File | Change |
|------|--------|
| `roampal/backend/modules/memory/embedding_service.py` | Full rewrite: ONNX Runtime + tokenizers |
| `pyproject.toml` | Swap sentence-transformers → onnxruntime/tokenizers, bump to 0.4.3 |
| `roampal/backend/modules/memory/tests/unit/test_embedding_service.py` | Updated mocks for ONNX internals |
| `roampal/cli.py` | Doctor checks new deps, torch removal hint |
| `roampal/server/main.py` | Updated ImportError guidance |
| `roampal/__init__.py` | Version bump, updated comment |

---

## Verification

```
pip install -e . && python -c "import roampal; print('import ok')"
pytest roampal/backend/modules/memory/tests/unit/test_embedding_service.py -v
roampal doctor
python -c "from roampal.backend.modules.memory.embedding_service import EmbeddingService; import asyncio; s = EmbeddingService(); print(asyncio.run(s.embed_text('hello'))[:5])"
```

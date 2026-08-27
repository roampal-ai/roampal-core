"""
Embedding Service

Handles text embedding using ONNX Runtime + HuggingFace tokenizers.
v0.5.9 Item 1: arena/mem-pattern off. Item 2 originally targeted
intfloat/multilingual-e5-base (INT8 ONNX) as the new default, but that's HELD
BACK pending a dedup-threshold recalibration (see the HF_REPO comment below) —
default is paraphrase-multilingual-mpnet-base-v2, now quantized to its own INT8
export instead of the old FP16 file. role="query"/"passage" prefix machinery
(_apply_prefix) stays in place, gated on repo name, ready for e5 once unblocked.

v0.4.3: Replaced sentence-transformers + PyTorch with direct ONNX inference.
Install size drops from ~2.5GB to ~200MB, faster startup, no CUDA deps.
Same 768d vectors, same ChromaDB collections — zero schema change either way.
"""

import asyncio
import logging
import os
import threading
from typing import List, Literal, Optional

import numpy as np

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    EMBEDDING_AVAILABLE = True
except ImportError:
    ort = None
    Tokenizer = None
    hf_hub_download = None
    EMBEDDING_AVAILABLE = False

logger = logging.getLogger(__name__)

# v0.5.9 Item 2, HELD BACK (2026-08-26): e5-base was the intended default, but
# the accuracy gate (dev/tests/integration/test_search_quality.py, 3x run) failed
# hard — 100% -> 57.5% — because FACT_DEDUP_DISTANCE_THRESHOLD (0.32,
# unified_memory_system.py:387) was calibrated for mpnet's embedding geometry and
# silently blocks most new fact writes as false duplicates under e5-base's tighter
# distance distribution. Shipping mpnet-INT8 instead for now: same memory/speed
# win as e5-base (both quantize to the same footprint), zero dedup risk since the
# geometry that threshold was tuned for doesn't change, and no migration risk since
# existing FP16 vectors stay valid against mpnet-INT8 queries (measured cosine
# agreement 0.990). Revisit e5-base once FACT_DEDUP_DISTANCE_THRESHOLD is
# recalibrated (or made model-aware) and the accuracy gate passes clean — tracked
# as v0.6.0 work. Overridable via env either way.
HF_REPO = os.environ.get("ROAMPAL_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
ONNX_FILE = os.environ.get("ROAMPAL_EMBED_ONNX_FILE", "onnx/model_qint8_avx512_vnni.onnx")
TOKENIZER_FILE = "tokenizer.json"
EMBEDDING_DIM = 768

# Default model name kept for backward compat (used in logs / repr)
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# v0.5.9 Item 2: e5-family models require "query: "/"passage: " prefixes. Other
# repos (e.g. mpnet) must not receive them, so gate prefixing on the active repo
# -- a ROAMPAL_EMBED_MODEL rollback to mpnet leaves text untouched.
def _apply_prefix(role: str, text: str) -> str:
    if "e5" in HF_REPO.lower():
        return f"{role}: {text}"
    return text


def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pooling — average token embeddings weighted by attention mask."""
    mask_expanded = np.expand_dims(attention_mask, axis=-1)  # (batch, seq, 1)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


class EmbeddingService:
    """
    Service for generating text embeddings.

    Uses ONNX Runtime with a multilingual model that works well
    for code and natural language.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._session: Optional["ort.InferenceSession"] = None
        self._tokenizer: Optional["Tokenizer"] = None
        # v0.4.2: Cache recent embeddings to avoid re-encoding the same query
        self._embed_cache: dict = {}
        self._embed_cache_max = 32
        # v0.5.9: serialize lazy loads. _load_model is check-then-act and the
        # session/tokenizer properties can be hit from several threads
        # (asyncio.to_thread encodes, prewarm, multiple profiles) — without
        # the lock, concurrent first-touches each build an InferenceSession
        # (N x 285 MB). Double-checked inside the lock.
        self._load_lock = threading.Lock()

    def _load_model(self):
        """Download (if needed) and load the ONNX model + tokenizer."""
        if not EMBEDDING_AVAILABLE:
            raise ImportError(
                "onnxruntime/tokenizers not installed. "
                "Run: pip install onnxruntime tokenizers huggingface-hub"
            )
        if self._session is not None and self._tokenizer is not None:
            return
        with self._load_lock:
            if self._session is not None and self._tokenizer is not None:
                return

            logger.info(f"Downloading/loading ONNX model: {self.model_name}")

            model_path = hf_hub_download(repo_id=HF_REPO, filename=ONNX_FILE)
            tokenizer_path = hf_hub_download(repo_id=HF_REPO, filename=TOKENIZER_FILE)

            # Use all available CPU cores but keep priority low.
            # v0.5.9 Item 1: disable the CPU memory arena + mem-pattern plan cache
            # so per-shape scratch buffers are returned to the OS instead of
            # ratcheting the working set up to a ~2.5-3 GB plateau.
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = int(os.environ.get("ROAMPAL_ORT_THREADS", "0"))  # 0 = auto
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False

            self._session = ort.InferenceSession(model_path, sess_options=opts,
                                                 providers=["CPUExecutionProvider"])
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            # v0.5.9: pad to longest in batch. Cap raised 128 -> 256 on measured
            # evidence (longest record in the corpus is 175 tokens, 3% truncated).
            # Deliberately above mpnet's declared max_seq_length 128; e5-base
            # (max 512) was held back to v0.6.0, see v0.5.9 RELEASE_NOTES Item 2.
            self._tokenizer.enable_padding()
            self._tokenizer.enable_truncation(max_length=256)

            logger.info(f"Embedding model loaded (ONNX): {self.model_name}")

    @property
    def session(self) -> "ort.InferenceSession":
        if self._session is None:
            self._load_model()
        return self._session

    @property
    def tokenizer(self) -> "Tokenizer":
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _encode(self, texts: List[str], role: str = "passage") -> np.ndarray:
        """Tokenize and run ONNX inference, return normalized embeddings."""
        encoded = self.tokenizer.encode_batch([_apply_prefix(role, t) for t in texts])

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        # Some models also want token_type_ids
        session_inputs = {inp.name for inp in self.session.get_inputs()}
        feeds = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in session_inputs:
            feeds["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, feeds)
        # outputs[0] is last_hidden_state: (batch, seq_len, hidden_dim)
        token_embeddings = outputs[0]

        pooled = _mean_pool(token_embeddings, attention_mask)
        return _normalize(pooled)

    async def embed_text(self, text: str, role: str = "passage") -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            role: "query" or "passage" — selects the e5 prefix (Item 2). Defaults
                to "passage" so an unclassified site fails safe toward stored-text
                semantics. Cache is keyed on (role, text).

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * EMBEDDING_DIM

        # v0.4.2: Return cached embedding if available (keyed on role+text)
        cache_key = (role, text)
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        # Run CPU-bound encode in thread to avoid blocking asyncio event loop
        embeddings = await asyncio.to_thread(self._encode, [text], role)
        result = embeddings[0].tolist()

        # Cache the result (evict oldest if full)
        if len(self._embed_cache) >= self._embed_cache_max:
            oldest_key = next(iter(self._embed_cache))
            del self._embed_cache[oldest_key]
        self._embed_cache[cache_key] = result

        return result

    async def embed_texts(self, texts: List[str], role: str = "passage") -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch).

        Args:
            texts: List of texts to embed
            role: "query" or "passage" — selects the e5 prefix (Item 2).

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[0.0] * EMBEDDING_DIM for _ in texts]

        embeddings = await asyncio.to_thread(self._encode, valid_texts, role)
        return [e.tolist() for e in embeddings]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return EMBEDDING_DIM

    async def prewarm(self):
        """Pre-warm the model by loading it."""
        # Trigger lazy load in a thread so it doesn't block
        await asyncio.to_thread(lambda: self.session)
        logger.info(f"Embedding model pre-warmed: {self.model_name}")

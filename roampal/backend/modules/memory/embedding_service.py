"""
Embedding Service

Handles text embedding using ONNX Runtime + HuggingFace tokenizers.
Uses the same model as before: paraphrase-multilingual-mpnet-base-v2 (exported to ONNX).

v0.4.3: Replaced sentence-transformers + PyTorch with direct ONNX inference.
Install size drops from ~2.5GB to ~200MB, faster startup, no CUDA deps.
Same model, same 768d vectors, same ChromaDB collections — zero user-facing change.
"""

import asyncio
import logging
from typing import List, Optional

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

# HuggingFace repo for the ONNX-exported model
HF_REPO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
ONNX_FILE = "onnx/model_O4.onnx"
TOKENIZER_FILE = "tokenizer.json"
EMBEDDING_DIM = 768

# Default model name kept for backward compat (used in logs / repr)
DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"


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
        self._embed_cache: dict[str, List[float]] = {}
        self._embed_cache_max = 32

    def _load_model(self):
        """Download (if needed) and load the ONNX model + tokenizer."""
        if not EMBEDDING_AVAILABLE:
            raise ImportError(
                "onnxruntime/tokenizers not installed. "
                "Run: pip install onnxruntime tokenizers huggingface-hub"
            )

        logger.info(f"Downloading/loading ONNX model: {self.model_name}")

        model_path = hf_hub_download(repo_id=HF_REPO, filename=ONNX_FILE)
        tokenizer_path = hf_hub_download(repo_id=HF_REPO, filename=TOKENIZER_FILE)

        # Use all available CPU cores but keep priority low
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 0  # auto-detect

        self._session = ort.InferenceSession(model_path, sess_options=opts,
                                             providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        # Match sentence-transformers default: pad to longest in batch, truncate at 128
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=128)

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

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Tokenize and run ONNX inference, return normalized embeddings."""
        encoded = self.tokenizer.encode_batch(texts)

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

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * EMBEDDING_DIM

        # v0.4.2: Return cached embedding if available
        if text in self._embed_cache:
            return self._embed_cache[text]

        # Run CPU-bound encode in thread to avoid blocking asyncio event loop
        embeddings = await asyncio.to_thread(self._encode, [text])
        result = embeddings[0].tolist()

        # Cache the result (evict oldest if full)
        if len(self._embed_cache) >= self._embed_cache_max:
            oldest_key = next(iter(self._embed_cache))
            del self._embed_cache[oldest_key]
        self._embed_cache[text] = result

        return result

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[0.0] * EMBEDDING_DIM for _ in texts]

        embeddings = await asyncio.to_thread(self._encode, valid_texts)
        return [e.tolist() for e in embeddings]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return EMBEDDING_DIM

    async def prewarm(self):
        """Pre-warm the model by loading it."""
        # Trigger lazy load in a thread so it doesn't block
        await asyncio.to_thread(lambda: self.session)
        logger.info(f"Embedding model pre-warmed: {self.model_name}")

"""
v0.5.9 Item 1: both ONNX loaders must construct sessions with the CPU memory
arena and memory-pattern plan cache DISABLED — the structural half of the
~484 MB claim (test_rss_soak.py is the behavioral half).

Why it matters: ORT's arena allocates per novel input shape and never returns
scratch buffers to the OS. With the arena on, the v0.5.8 server climbed to a
~2.5-3 GB plateau; arena-off is measured flat. A regression here (someone
re-enabling the arena "for speed") must fail a test, not a production server.
"""

import pytest
from unittest.mock import MagicMock

import onnxruntime as ort

import roampal.backend.modules.memory.embedding_service as es
import roampal.backend.modules.memory.search_service as ss
from roampal.backend.modules.memory.search_service import SearchService
from roampal.backend.modules.memory.scoring_service import ScoringService
from roampal.backend.modules.memory.routing_service import RoutingService
from roampal.backend.modules.memory.tag_service import TagService
from roampal.backend.modules.memory.config import MemoryConfig


class _FakeTokenizer:
    @classmethod
    def from_file(cls, path):
        inst = cls()
        inst.enable_padding = lambda: None
        inst.enable_truncation = lambda **kw: None
        return inst


class _FakeSession:
    pass


@pytest.fixture
def capture(monkeypatch):
    """Patch ORT + HF downloads; record the SessionOptions each loader builds."""
    recorded = []

    def fake_session(model_path, sess_options=None, providers=None):
        recorded.append({"model_path": model_path, "sess_options": sess_options,
                         "providers": providers})
        return _FakeSession()

    def fake_download(repo_id=None, filename=None, **kw):
        return f"cached::{repo_id}/{filename}"

    monkeypatch.setattr(ort, "InferenceSession", fake_session)
    monkeypatch.setattr("tokenizers.Tokenizer", _FakeTokenizer)
    # embedding_service binds Tokenizer at module level — patch that binding too.
    monkeypatch.setattr(es, "Tokenizer", _FakeTokenizer)
    monkeypatch.setattr(es, "hf_hub_download", fake_download)
    # search_service imports hf_hub_download inside _load_ce — patch the package.
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    # Reset the shared-CE holder so CE tests always exercise a fresh load.
    monkeypatch.setattr(ss, "_shared_ce_session", None)
    monkeypatch.setattr(ss, "_shared_ce_tokenizer", None)
    return recorded


def _make_search_service():
    return SearchService(
        collections={},
        scoring_service=MagicMock(spec=ScoringService),
        routing_service=MagicMock(spec=RoutingService),
        tag_service=MagicMock(spec=TagService),
        embed_fn=None,
        config=MemoryConfig(),
    )


class TestEmbedderSessionOptions:
    def test_arena_and_mem_pattern_off(self, capture, monkeypatch):
        monkeypatch.delenv("ROAMPAL_ORT_THREADS", raising=False)
        es.EmbeddingService()._load_model()
        assert len(capture) == 1
        opts = capture[0]["sess_options"]
        assert opts.enable_cpu_mem_arena is False
        assert opts.enable_mem_pattern is False
        assert opts.inter_op_num_threads == 1

    def test_threads_default_auto(self, capture, monkeypatch):
        monkeypatch.delenv("ROAMPAL_ORT_THREADS", raising=False)
        es.EmbeddingService()._load_model()
        assert capture[0]["sess_options"].intra_op_num_threads == 0

    def test_thread_override_respected(self, capture, monkeypatch):
        monkeypatch.setenv("ROAMPAL_ORT_THREADS", "4")
        es.EmbeddingService()._load_model()
        assert capture[0]["sess_options"].intra_op_num_threads == 4

    def test_loads_default_int8_artifact(self, capture, monkeypatch):
        """v0.5.9 ships mpnet INT8 (e5-base held back) — the loader must request
        the INT8 export from the mpnet repo by default."""
        monkeypatch.delenv("ROAMPAL_EMBED_MODEL", raising=False)
        monkeypatch.delenv("ROAMPAL_EMBED_ONNX_FILE", raising=False)
        es.EmbeddingService()._load_model()
        assert "paraphrase-multilingual-mpnet-base-v2" in capture[0]["model_path"]
        assert "model_qint8_avx512_vnni.onnx" in capture[0]["model_path"]


class TestCrossEncoderSessionOptions:
    def test_arena_and_mem_pattern_off(self, capture, monkeypatch):
        monkeypatch.delenv("ROAMPAL_ORT_THREADS", raising=False)
        svc = _make_search_service()
        assert svc._load_ce() is True
        assert len(capture) == 1
        opts = capture[0]["sess_options"]
        assert opts.enable_cpu_mem_arena is False
        assert opts.enable_mem_pattern is False
        assert opts.inter_op_num_threads == 1

    def test_thread_override_respected(self, capture, monkeypatch):
        monkeypatch.setenv("ROAMPAL_ORT_THREADS", "2")
        _make_search_service()._load_ce()
        assert capture[0]["sess_options"].intra_op_num_threads == 2

    def test_loads_default_int8_artifact(self, capture):
        _make_search_service()._load_ce()
        assert "model_qint8_avx512_vnni.onnx" in capture[0]["model_path"]


class TestEmbedderLazyLoadRace:
    def test_concurrent_first_touch_builds_one_session(self, capture, monkeypatch):
        """Post-review round 3: _load_model is serialized with double-checked
        locking — concurrent first-touches (multiple threads via prewarm /
        asyncio.to_thread) must construct exactly ONE InferenceSession, not
        one per racer (N x 285 MB)."""
        import threading

        monkeypatch.delenv("ROAMPAL_ORT_THREADS", raising=False)
        svc = es.EmbeddingService()
        barrier = threading.Barrier(4)

        def touch():
            barrier.wait()
            svc.session  # property triggers the lazy _load_model path

        threads = [threading.Thread(target=touch) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(capture) == 1, (
            f"concurrent first-touch built {len(capture)} sessions"
        )
        assert svc._session is not None

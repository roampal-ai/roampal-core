"""
v0.5.9 primary RAM regression gate (Item 1): with the ONNX memory arena and
mem-pattern cache disabled, the embedder's working set must PLATEAU across
varied input shapes — no monotonic growth. This is the behavioral half of the
~484 MB claim; test_onnx_memory_options.py is the structural half.

Uses the real INT8 embedder from the local HF cache (skipped when not cached —
no network downloads in tests). On the incident host the equivalent run
measured 1,016-1,020 MB across 500 varied-shape ops on the FP16 config, and
the live INT8 server holds ~322 MB flat.
"""

import pytest

psutil = pytest.importorskip("psutil")

import roampal.backend.modules.memory.embedding_service as es
from roampal.backend.modules.memory.embedding_service import EmbeddingService


def _model_cached() -> bool:
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=es.HF_REPO, filename=es.ONNX_FILE,
                        local_files_only=True)
        hf_hub_download(repo_id=es.HF_REPO, filename=es.TOKENIZER_FILE,
                        local_files_only=True)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _model_cached(),
                       reason="INT8 embedder not in local HF cache"),
]

# Word counts per embed call — spans the 256-token truncation cap.
_LENGTHS = [4, 16, 32, 64, 96, 128, 192, 256, 300]
_BATCHES = [1, 2, 4, 8]


async def test_embedder_rss_plateaus_under_varied_shapes():
    svc = EmbeddingService()
    await svc.prewarm()

    proc = psutil.Process()

    def rss_mb() -> float:
        return proc.memory_info().rss / (1024 * 1024)

    # Warm-up: let ORT allocate its one-time per-shape buffers for the low
    # lengths so the measured window starts past warmup noise.
    for n in _LENGTHS[:3]:
        await svc.embed_texts(["alpha beta gamma delta " * n], role="passage")

    samples = []
    for i in range(120):
        n = _LENGTHS[i % len(_LENGTHS)]
        batch = ["lorem ipsum dolor sit amet consectetur " * n
                 for _ in range(_BATCHES[i % len(_BATCHES)])]
        await svc.embed_texts(batch, role="passage")
        samples.append(rss_mb())

    # Ratchet detector: compare the first vs last fifth of the run. A true
    # ratchet climbs monotonically; GC noise averages out across 24 samples.
    fifth = len(samples) // 5
    early = sum(samples[:fifth]) / fifth
    late = sum(samples[-fifth:]) / len(samples[-fifth:])
    drift = (late - early) / early

    assert drift < 0.05, (
        f"RSS ratchet detected: {drift:.1%} drift "
        f"(early {early:.0f} MB -> late {late:.0f} MB). "
        f"Check enable_cpu_mem_arena/enable_mem_pattern on the loader."
    )

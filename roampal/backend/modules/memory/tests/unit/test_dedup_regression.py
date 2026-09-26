"""
Dedup regression guard (v0.6.0 Item 2 / Test Plan C).

Catches a silent dedup collapse at the first change to anything
cosine-distance-based, instead of discovering it weeks later in the
accuracy gate (the v0.5.9 e5-base failure mode).

Two assertions, two directions:

(test_dedup_guard_three_facts_survive)
  Real `store_memory_bank` path through a real short-lived ChromaDB
  instance: 3 distinct facts about one entity must land as 3 distinct
  ACTIVE memory_bank records, none carrying `duplicate_of` metadata.

(test_dedup_negative_control_can_fail)
  Same scenario re-run twice with the known-broken regime forced in,
  asserting the guard CAN trip — i.e. the store path IS sensitive to a
  threshold/geometry mismatch:
  - threshold forced while normal geometry is kept, and
  - normal threshold kept while the embed fixture is forced to the
    compressed "e5-like" geometry (near-identical vectors).
  Under mpnet-INT8 + threshold 0.32 neither collapse occurs; these
  reproduce the exact shape of the v0.5.9 incident.

Embeddings are a deterministic md5-seeded fixture (768-dim, unit-norm,
real cosine geometry) — NOT the production ONNX model. Budget: < 5s wall.
"""

import asyncio
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))
)

import pytest

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem

# The shared-entity fact trio used in the eval that caught the v0.5.9 bug
# (same style: distinct facts, shared subject).
FACTS = (
    "Max is a golden retriever",
    "Max is afraid of thunderstorms",
    "Max learned to open the gate latch in April",
)

E5_BROKEN_GEOMETRY = "compressed_e5_style"


class _FakeEmbeddingFixture:
    """Deterministic embedding fixture standing in for the real ONNX model.

    md5-seeded per text (stable across processes), 768-dim, unit-norm.
    mpnet_like mode: distinct texts embed ~2.0 apart under the adapter's
    distance metric — far above the 0.32 dedup threshold, so every
    distinct fact survives, matching real mpnet geometry's separation.
    e5-like mode: every vector is one warm-start base direction plus
    tiny per-fact noise — the compressed geometry the e5-base + old
    threshold pairing produced.
    """

    def __init__(self, dim=768, mode="mpnet_like"):
        self.dim = dim
        self.mode = mode

    def _token_vector(self, text):
        import hashlib
        import random

        seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "big")
        rng = random.Random(seed)
        vec = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    async def embed_text(self, text, role="passage"):
        if self.mode == E5_BROKEN_GEOMETRY:
            base = self._token_vector("shared base direction warm start")
            delta = self._token_vector(text)
            out = [3.0 * b + 0.05 * d for b, d in zip(base, delta)]
            norm = sum(v * v for v in out) ** 0.5
            return [v / norm for v in out]
        return self._token_vector(text)

    async def prewarm(self):
        return None


@pytest.fixture(autouse=True)
async def _cancel_warmup_tasks():
    """Cancel UMS v0.5.2 warmup tasks after each test.

    Same leak-prevention pattern as test_unified_memory_system.py:
    orphan asyncio.to_thread workers hang the next initialize() on
    Python 3.10. The reembed migration task is also cancelled if the
    fresh-install meta write was skipped.
    """
    yield
    try:
        for task in asyncio.all_tasks():
            if task.get_name() in ("warmup_ce", "warmup_embedding", "reembed") and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
    except RuntimeError:
        pass


async def _make_ums(tmp_path, embed_service):
    ums = UnifiedMemorySystem(
        data_path=str(tmp_path / "data"),
        embed_service=embed_service,
    )
    await ums.initialize()
    return ums


async def _store_three_facts(ums):
    ids = []
    for fact in FACTS:
        ids.append(
            await ums.store_memory_bank(
                fact,
                tags=["identity"],
                noun_tags=["max"],
                importance=0.7,
                confidence=0.7,
            )
        )
    return ids


async def _active_records(ums):
    """Return {id: metadata} for every non-archived memory_bank row."""
    col = ums.collections["memory_bank"]
    count = await col.get_collection_count()
    # No public "list all" in the adapter — query with an arbitrary unit
    # vector and top_k=count to drain (dedup scan uses <=count anyway).
    probe = [0.0] * 768
    probe[0] = 1.0
    hits = await col.query_vectors(query_vector=probe, top_k=max(count, 1))
    out = {}
    for h in hits:
        meta = h.get("metadata", {})
        if meta.get("status", "active") == "archived":
            continue
        out[h["id"]] = meta
    return out


@pytest.mark.asyncio
async def test_dedup_guard_three_facts_survive(tmp_path):
    """3 distinct facts about one entity -> 3 active records, no duplicate_of."""
    ums = await _make_ums(tmp_path, _FakeEmbeddingFixture())

    ids = await _store_three_facts(ums)
    assert len({i for i in ids}) == 3, f"collapsed to {len(set(ids))} distinct ids: {ids}"

    records = await _active_records(ums)
    assert len(records) == 3, f"expected 3 active memory_bank records, got {len(records)}"

    for doc_id, meta in records.items():
        assert "duplicate_of" not in meta, f"{doc_id} carried duplicate_of metadata"


@pytest.mark.asyncio
async def test_dedup_negative_control_can_fail(tmp_path, monkeypatch):
    """The guard must be able to trip: same 3-fact scenario under both
    broken regimes collapses to 1 record (v0.5.9 incident shape).

    Run A: real geometry kept, threshold forced to 10.0 (adapter metric
    puts orthogonal pairs near ~2.0) — a threshold-side regression trips
    the collapse.
    Run B: threshold kept at 0.32, embed fixture forced to compressed
    "e5-like" geometry — a geometry-side regression trips it too.
    """
    # Run A — threshold forced into the broken regime: high enough that
    # ANY pairwise distance trips the collapse regardless of fixture
    # geometry (adapter distances between orthogonal 768-dim unit vectors
    # concentrate near ~2.0).
    ums = await _make_ums(tmp_path / "data_a", _FakeEmbeddingFixture())
    monkeypatch.setattr(ums, "FACT_DEDUP_DISTANCE_THRESHOLD", 10.0)
    ids = await _store_three_facts(ums)
    assert len(set(ids)) == 1, (
        "threshold-forced regime did NOT collapse distinct facts — "
        "the positive guard above may be vacuous"
    )
    records = await _active_records(ums)
    assert len(records) == 1

    # Run B — geometry forced into the compressed e5-like regime,
    # threshold left at the real 0.32.
    ums_b = await _make_ums(tmp_path / "data_b", _FakeEmbeddingFixture(mode=E5_BROKEN_GEOMETRY))
    assert ums_b.FACT_DEDUP_DISTANCE_THRESHOLD == 0.32
    ids_b = await _store_three_facts(ums_b)
    assert len(set(ids_b)) == 1, (
        "e5-style compressed geometry did NOT collapse distinct facts — "
        "the geometry guard is not wired to the real distance check"
    )
    records_b = await _active_records(ums_b)
    assert len(records_b) == 1


# ============================================================================
# Task 29: the guard on the REAL model (mpnet-INT8).
#
# The fake fixture above sits distinct facts ~2.0 apart, so it passes for ANY
# threshold < ~2 and never sees an embedder swap (the v0.5.9 e5-base failure
# mode it was built to catch). Measured on the real model through the
# adapter's ACTUAL distance metric — ChromaDB l2 on unit vectors returns the
# squared form sum((x-y)^2) = 2-2cos (probe-verified against the store path;
# the plain-L2 form is exactly sqrt of it):
#   distinct FACTS: 1.12 / 1.40 / 1.50 (closest pair 1.12)
#   near-duplicates: 0.077 / 0.049 / 0.0
# against the 0.32 threshold (squared units). REAL_DISTINCT_MARGIN and
# REAL_NEAR_MARGIN below pin both margins — an embedder/threshold change
# flips one of them before the accuracy gate does.
#
# Three variants:
# - RECORDED-vectors tests ALWAYS run: vectors recorded from the real model
#   (5-decimal float precision, negligible vs every margin here) live
#   beside this file in dedup_real_vectors.json; they exercise the real
#   store path with production geometry on every machine, no model needed.
# - REAL-model test runs wherever the HF cache holds mpnet-INT8 (skip-if-
#   uncached locally; CI primes the cache in a dedicated step).
# - The fake-fixture tests above are kept unchanged as the code-path check.
#
# Known delta from the audit table (F4): the audit quoted plain-L2 numbers
# (1.13-1.40 / 0.03-0.09); the store path compares in the squared metric,
# so the recorded variant pins the units the real check actually uses.
# ============================================================================

import json
from pathlib import Path

_RECORDED_FILE = Path(__file__).parent / "dedup_real_vectors.json"

# Calibrated margins (units: adapter distance). Distinct facts must sit at
# least this far ABOVE the threshold; near-dups at least this far BELOW it.
REAL_DISTINCT_MARGIN = 0.1
REAL_NEAR_MARGIN = 0.03


def _load_recorded():
    return json.loads(_RECORDED_FILE.read_text(encoding="utf-8"))


class _RecordedEmbeddingFixture:
    """Serves the recorded real-model vectors (text → vector dict).

    Unknown texts raise — only recorded contract texts may be embedded, so a
    carelessly added fixture usage fails loudly instead of silently
    embedding with the wrong geometry."""

    def __init__(self):
        data = _load_recorded()
        self._vectors = dict(zip(data["texts"], data["vectors"]))

    async def embed_text(self, text, role="passage"):
        try:
            return self._vectors[text]
        except KeyError:
            raise KeyError(
                f"text not in the recorded dedup vector set (refresh dedup_real_vectors.json): {text!r}"
            )

    async def prewarm(self):
        return None


def _adapter_distance(a, b):
    """THE adapter's distance metric, measured against the real store path
    (Task 29 double-check probe): ChromaDB's l2 on unit vectors returns
    sum((x-y)^2) = 2 - 2*cos — for unit-norm rows that is the SQUARED plain
    euclidean, NOT sqrt. The 0.32 threshold lives in these units."""
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _recorded_geometry():
    """(min pairwise distinct distance, max near-dup distance) straight from
    the recorded real-model vectors — the numbers the store path will see."""
    data = _load_recorded()
    vecs = data["vectors"]
    fact_idx = [i for i, r in enumerate(data["roles"]) if r == "fact"]
    near_idx = [i for i, r in enumerate(data["roles"]) if r == "near_dup"]
    distinct_min = min(
        _adapter_distance(vecs[i], vecs[j])
        for i, j in itertools.combinations(fact_idx, 2)
    )
    # near-dups are paraphrases scored against every fact — a dup is a dup
    # of its closest fact; the contract needs the closest pairing under the
    # threshold, and only OVER-threshold pairs would survive.
    near_max_of_closest = 0.0
    for i in near_idx:
        best = min(_adapter_distance(vecs[i], vecs[j]) for j in fact_idx)
        near_max_of_closest = max(near_max_of_closest, best)
    return distinct_min, near_max_of_closest


@pytest.mark.asyncio
async def test_recorded_real_geometry_margins_hold(tmp_path):
    """The recorded geometry must keep BOTH margins (calibrated Task 29:
    distinct ≥ threshold + 0.1; near-dups < threshold − 0.03). Fails if a
    re-record put a text on the wrong side of either — the failure mode the
    fake-fixture guard is blind to (it sits distinct facts ~2.0 apart)."""
    threshold = UnifiedMemorySystem(data_path=str(tmp_path / "t")).FACT_DEDUP_DISTANCE_THRESHOLD

    distinct_min, near_max = _recorded_geometry()
    assert distinct_min >= threshold + REAL_DISTINCT_MARGIN, (
        f"distinct facts sit {distinct_min:.3f} apart — under threshold "
        f"{threshold} + {REAL_DISTINCT_MARGIN} margin: a threshold bump or "
        "model swap would merge DISTINCT facts and the fake-fixture guard "
        "would NOT see it (~2.0 apart)"
    )
    assert near_max < threshold - REAL_NEAR_MARGIN, (
        f"near-dup sits {near_max:.3f} away — not comfortably under "
        f"threshold {threshold}: the dedup-skip contract is not exercised"
    )


@pytest.mark.asyncio
async def test_recorded_real_geometry_dedup_holds(tmp_path):
    """Real (recorded) geometry at the real threshold: 3 distinct facts -> 3
    active records; 3 near-duplicates dedup into the first fact — nothing new."""
    ums = await _make_ums(tmp_path / "data_rec", _RecordedEmbeddingFixture())

    ids = await _store_three_facts(ums)
    assert len(set(ids)) == 3
    records = await _active_records(ums)
    assert len(records) == 3

    # Each near-dup dedups (returns an existing id); total stays 3.
    for near in _load_recorded()["texts"][3:]:
        dup_id = await ums.store_memory_bank(
            near, tags=["identity"], noun_tags=["max"]
        )
        assert dup_id in ids, (
            f"near-dup {near!r} did not dedup into an existing record "
            f"(got {dup_id}) — the real-geometry guard does NOT hold"
        )
    assert len(await _active_records(ums)) == 3


@pytest.mark.asyncio
async def test_recorded_real_geometry_collapses_at_threshold_1p2(tmp_path, monkeypatch):
    """The calibration proof the fake fixture CANNOT give: at threshold 1.2
    (the fake fixture's distinct facts sit ~2.0 apart, so the threshold-1.2
    regime looks healthy to it), the REAL geometry's closest distinct pair
    is 1.12 — UNDER 1.2 — and the 3-fact store collapses: fact 1 (1.40 in
    the squared adapter metric) survives, fact 2 merges into fact 0. 3
    stores -> 2 records. Proof the recorded variant is not vacuous."""
    ums = await _make_ums(tmp_path / "data_12", _RecordedEmbeddingFixture())
    monkeypatch.setattr(ums, "FACT_DEDUP_DISTANCE_THRESHOLD", 1.2)
    ids = await _store_three_facts(ums)
    assert len(set(ids)) == 2, (
        f"real geometry did NOT collapse at threshold 1.2 (got {len(set(ids))} "
        "records) — the recorded variant would pass a broken threshold; "
        "it is vacuous"
    )
    records = await _active_records(ums)
    assert len(records) == 2


# ---------------------------------------------------------------------------
# The REAL model, skip-if-uncached (CI primes the HF cache in a step added
# to tests.yml; locally this runs wherever mpnet-INT8 is already cached).
# ---------------------------------------------------------------------------

def _real_model_cached() -> bool:
    from roampal.backend.modules.memory import embedding_service as _es

    if not _es.EMBEDDING_AVAILABLE:
        return False
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=_es.HF_REPO, filename=_es.ONNX_FILE, local_files_only=True
        )
        hf_hub_download(
            repo_id=_es.HF_REPO, filename=_es.TOKENIZER_FILE, local_files_only=True
        )
        return True
    except Exception:
        return False


# Probe at collection time (real env; the per-test sandbox doesn't apply).
_REAL_MODEL_PRESENT = _real_model_cached()


@pytest.mark.skipif(
    not _REAL_MODEL_PRESENT,
    reason="mpnet-INT8 not in local HF cache (CI primes it in a cache step)",
)
@pytest.mark.asyncio
async def test_real_model_dedup_end_to_end(tmp_path, unsandboxed_home):
    """The dedub guard against the LIVE production embedder + a real
    ChromaDB store: distinct facts survive, near-dups collapse, and the
    measured geometry holds both margins. This is the variant an embedder
    swap (ROAMPAL_EMBED_MODEL override or refactor) cannot fool."""
    from roampal.backend.modules.memory.embedding_service import EmbeddingService

    svc = EmbeddingService()
    ums = await _make_ums(tmp_path / "data_real", svc)

    # Measured geometry under the LIVE model first: both margins must hold.
    threshold = ums.FACT_DEDUP_DISTANCE_THRESHOLD
    embeddings = []
    for fact in FACTS:
        embeddings.append(await svc.embed_text(fact, role="passage", skip_cache=True))
    distinct_min = min(
        _adapter_distance(embeddings[i], embeddings[j])
        for i, j in ((0, 1), (0, 2), (1, 2))
    )
    near_embeddings = []
    for near in _load_recorded()["texts"][3:]:
        near_embeddings.append(
            await svc.embed_text(near, role="passage", skip_cache=True)
        )
    near_max = max(
        _adapter_distance(ne, embeddings[0]) for ne in near_embeddings
    )
    assert distinct_min >= threshold + REAL_DISTINCT_MARGIN, (
        f"live model: distinct facts nearest {distinct_min:.3f} < "
        f"{threshold} + {REAL_DISTINCT_MARGIN} margin"
    )
    assert near_max < threshold, f"live model: near-dup {near_max:.3f} would NOT dedup"

    # End-to-end store path with the real embedder.
    ids = await _store_three_facts(ums)
    assert len(set(ids)) == 3
    for near in _load_recorded()["texts"][3:]:
        dup_id = await ums.store_memory_bank(near, tags=["identity"], noun_tags=["max"])
        assert dup_id in ids, f"live model: near-dup {near!r} stored as a NEW memory"
    assert len(await _active_records(ums)) == 3

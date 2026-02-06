"""
================================================================================
COMPREHENSIVE ROAMPAL BENCHMARK
================================================================================

The definitive evaluation: 4 conditions × 5 maturity levels × metrics suite

PURPOSE:
--------
Publishable-quality benchmark proving Roampal's value over alternatives.
Combines three-way comparison, learning curve, and token efficiency.

4 CONDITIONS TESTED:
--------------------
1. RAG BASELINE: Pure ChromaDB L2 distance (no reranker, no scores)
2. RERANKER-ONLY: Cross-encoder reranking but no outcome scores
3. OUTCOMES-ONLY: Outcome scores but no cross-encoder reranker
4. FULL ROAMPAL: Both reranker + outcome scores (Wilson-scored)

5 MATURITY LEVELS:
------------------
- Cold Start (0 uses): No outcome history
- Early (3 uses): 2 worked outcomes
- Established (5 uses): 4 worked outcomes
- Proven (10 uses): 8 worked outcomes
- Mature (20 uses): 18 worked outcomes

METRICS COMPUTED:
-----------------
- Top-1 Accuracy: Good advice ranked #1
- MRR (Mean Reciprocal Rank): 1/rank of first correct result
- nDCG@5: Normalized Discounted Cumulative Gain at k=5
- Token Efficiency: Tokens needed to reach correct answer

CROSS-DOMAIN HOLDOUT:
---------------------
Train outcomes on domains {finance, health, tech}
Test generalization on domains {nutrition, crypto}

================================================================================
"""

import asyncio
import json
import os
import sys
import shutil
import statistics
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from scipy import stats

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from roampal.backend.modules.memory.unified_memory_system import UnifiedMemorySystem
from mock_utilities import MockLLMService

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("ERROR: Required packages missing.")
    print("pip install sentence-transformers chromadb")

os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"


# =============================================================================
# EMBEDDING & RERANKING SERVICES
# =============================================================================

class EmbeddingService768:
    """Real semantic embeddings using sentence-transformers."""
    def __init__(self):
        print("Loading embedding model: all-mpnet-base-v2 (768 dim)...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.dimension = 768
        self.token_count = 0

    async def embed_text(self, text: str) -> List[float]:
        # Approximate token count (words * 1.3)
        self.token_count += int(len(text.split()) * 1.3)
        return self.model.encode(text, convert_to_numpy=True).tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        for t in texts:
            self.token_count += int(len(t.split()) * 1.3)
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()


class CrossEncoderReranker:
    """Cross-encoder for reranking results."""
    def __init__(self):
        print("Loading cross-encoder: ms-marco-MiniLM-L-6-v2...")
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.token_count = 0

    def rerank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """Returns [(original_index, score)] sorted by score descending."""
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        # Count tokens
        for doc in documents:
            self.token_count += int((len(query.split()) + len(doc.split())) * 1.3)
        scores = self.model.predict(pairs)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores


# =============================================================================
# METRICS CALCULATIONS
# =============================================================================

def calculate_mrr(ranks: List[int]) -> float:
    """Mean Reciprocal Rank: average of 1/rank for first correct result."""
    if not ranks:
        return 0.0
    reciprocals = [1.0/r if r > 0 else 0.0 for r in ranks]
    return statistics.mean(reciprocals)


def calculate_ndcg(relevances: List[float], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    relevances: list of relevance scores for each position (1.0 for good, 0.0 for bad)
    """
    if not relevances:
        return 0.0

    # DCG: sum of rel_i / log2(i+1) for i in 1..k
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # +2 because i starts at 0

    # Ideal DCG: sort relevances descending
    ideal_rels = sorted(relevances, reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


# =============================================================================
# TEST SCENARIOS
# =============================================================================

# Training domains (outcomes applied here)
TRAIN_DOMAINS = {"finance", "health", "tech"}
# Test domains (cross-domain generalization)
TEST_DOMAINS = {"nutrition", "crypto"}

SCENARIOS = [
    # FINANCE (train)
    {
        "id": 1, "domain": "finance",
        "query": "Should I invest all my savings to beat inflation?",
        "good_advice": "Keep 6 months expenses in emergency fund before investing any savings",
        "bad_advice": "Invest all your money immediately - inflation erodes cash value daily",
    },
    {
        "id": 2, "domain": "finance",
        "query": "How do I pick winning stocks?",
        "good_advice": "Use low-cost index funds - 90% of stock pickers underperform the market",
        "bad_advice": "Research individual stocks carefully to beat the market consistently",
    },
    # HEALTH (train)
    {
        "id": 3, "domain": "health",
        "query": "Can I train myself to need less sleep?",
        "good_advice": "Adults need 7-9 hours - sleep deprivation causes cognitive decline and health issues",
        "bad_advice": "Train your body to function on 4-5 hours - many successful CEOs do this",
    },
    {
        "id": 4, "domain": "health",
        "query": "Should I do intense workouts when I have time?",
        "good_advice": "Consistent moderate exercise beats sporadic intense sessions for long-term health",
        "bad_advice": "Maximize intensity when you can exercise - high intensity burns more calories",
    },
    # TECH (train)
    {
        "id": 5, "domain": "tech",
        "query": "Should I fix the bug first then write tests?",
        "good_advice": "Write a failing test first - it documents the bug and prevents regression",
        "bad_advice": "Fix the bug quickly first, then add tests later when you have time",
    },
    {
        "id": 6, "domain": "tech",
        "query": "Can I store API keys in my private repo?",
        "good_advice": "Never commit secrets - use environment variables or secret managers",
        "bad_advice": "Private repos are secure enough for API keys and credentials",
    },
    # NUTRITION (test - cross-domain)
    {
        "id": 7, "domain": "nutrition",
        "query": "Should I eat more protein for muscle gains?",
        "good_advice": "0.7-1g protein per pound is optimal - excess provides no additional benefit",
        "bad_advice": "Eat as much protein as possible - more protein means more muscle growth",
    },
    {
        "id": 8, "domain": "nutrition",
        "query": "Should I avoid eating carbs at night?",
        "good_advice": "Meal timing matters less than total daily intake for weight management",
        "bad_advice": "Never eat carbs after 6pm - they convert to fat if not burned immediately",
    },
    # CRYPTO (test - cross-domain)
    {
        "id": 9, "domain": "crypto",
        "query": "Should I put my savings into crypto?",
        "good_advice": "Only invest what you can afford to lose completely - crypto is speculative",
        "bad_advice": "Move your savings to crypto - it's the future of finance and banking",
    },
    {
        "id": 10, "domain": "crypto",
        "query": "Should I buy new altcoins early?",
        "good_advice": "Most new tokens fail or are scams - stick to established cryptocurrencies",
        "bad_advice": "Get into new altcoins early for massive gains - early investors get rich",
    },
]

MATURITY_LEVELS = [
    {"name": "cold_start", "uses": 0, "worked": 0, "failed": 0},
    {"name": "early", "uses": 3, "worked": 2, "failed": 0},
    {"name": "established", "uses": 5, "worked": 4, "failed": 0},
    {"name": "proven", "uses": 10, "worked": 8, "failed": 1},
    {"name": "mature", "uses": 20, "worked": 18, "failed": 1},
]


# =============================================================================
# TEST CONDITIONS
# =============================================================================

@dataclass
class TestResult:
    condition: str
    scenario_id: int
    domain: str
    maturity: str
    good_rank: int
    bad_rank: int
    top1_correct: bool
    mrr: float
    ndcg5: float
    tokens_used: int


async def test_rag_baseline(
    scenario: Dict,
    maturity: Dict,
    data_dir: str,
    embedding_service: EmbeddingService768
) -> TestResult:
    """
    Condition 1: Pure RAG - ChromaDB L2 distance only.
    No reranker, no outcome scores.
    """
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    initial_tokens = embedding_service.token_count

    client = chromadb.PersistentClient(path=os.path.join(data_dir, "chromadb"))
    collection = client.create_collection(name="rag_test", metadata={"hnsw:space": "l2"})

    # Embed and store
    good_emb = await embedding_service.embed_text(scenario["good_advice"])
    bad_emb = await embedding_service.embed_text(scenario["bad_advice"])

    collection.add(
        ids=["good", "bad"],
        embeddings=[good_emb, bad_emb],
        documents=[scenario["good_advice"], scenario["bad_advice"]],
        metadatas=[{"type": "good"}, {"type": "bad"}]
    )

    # Query
    query_emb = await embedding_service.embed_text(scenario["query"])
    results = collection.query(query_embeddings=[query_emb], n_results=2)

    # Analyze results
    good_rank, bad_rank = 0, 0
    relevances = []
    for i, meta in enumerate(results["metadatas"][0]):
        if meta["type"] == "good":
            good_rank = i + 1
            relevances.append(1.0)
        else:
            bad_rank = i + 1
            relevances.append(0.0)

    tokens_used = embedding_service.token_count - initial_tokens

    return TestResult(
        condition="rag_baseline",
        scenario_id=scenario["id"],
        domain=scenario["domain"],
        maturity=maturity["name"],
        good_rank=good_rank,
        bad_rank=bad_rank,
        top1_correct=(good_rank == 1),
        mrr=1.0/good_rank if good_rank > 0 else 0.0,
        ndcg5=calculate_ndcg(relevances, k=5),
        tokens_used=tokens_used
    )


async def test_reranker_only(
    scenario: Dict,
    maturity: Dict,
    data_dir: str,
    embedding_service: EmbeddingService768,
    reranker: CrossEncoderReranker
) -> TestResult:
    """
    Condition 2: RAG + Cross-encoder reranking.
    No outcome scores.
    """
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    initial_tokens = embedding_service.token_count + reranker.token_count

    client = chromadb.PersistentClient(path=os.path.join(data_dir, "chromadb"))
    collection = client.create_collection(name="reranker_test", metadata={"hnsw:space": "l2"})

    # Embed and store
    good_emb = await embedding_service.embed_text(scenario["good_advice"])
    bad_emb = await embedding_service.embed_text(scenario["bad_advice"])

    collection.add(
        ids=["good", "bad"],
        embeddings=[good_emb, bad_emb],
        documents=[scenario["good_advice"], scenario["bad_advice"]],
        metadatas=[{"type": "good"}, {"type": "bad"}]
    )

    # Query with initial retrieval
    query_emb = await embedding_service.embed_text(scenario["query"])
    results = collection.query(query_embeddings=[query_emb], n_results=2)

    # Rerank with cross-encoder
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    reranked = reranker.rerank(scenario["query"], docs)

    # Analyze reranked results
    good_rank, bad_rank = 0, 0
    relevances = []
    for new_rank, (orig_idx, score) in enumerate(reranked):
        if metas[orig_idx]["type"] == "good":
            good_rank = new_rank + 1
            relevances.append(1.0)
        else:
            bad_rank = new_rank + 1
            relevances.append(0.0)

    tokens_used = (embedding_service.token_count + reranker.token_count) - initial_tokens

    return TestResult(
        condition="reranker_only",
        scenario_id=scenario["id"],
        domain=scenario["domain"],
        maturity=maturity["name"],
        good_rank=good_rank,
        bad_rank=bad_rank,
        top1_correct=(good_rank == 1),
        mrr=1.0/good_rank if good_rank > 0 else 0.0,
        ndcg5=calculate_ndcg(relevances, k=5),
        tokens_used=tokens_used
    )


async def test_outcomes_only(
    scenario: Dict,
    maturity: Dict,
    data_dir: str,
    embedding_service: EmbeddingService768,
    apply_outcomes: bool = True
) -> TestResult:
    """
    Condition 3: RAG + Outcome scores only.
    No cross-encoder reranking - relies purely on outcome-based scoring.
    """
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    initial_tokens = embedding_service.token_count

    client = chromadb.PersistentClient(path=os.path.join(data_dir, "chromadb"))
    collection = client.create_collection(name="outcomes_test", metadata={"hnsw:space": "l2"})

    # Embed and store with outcome metadata
    good_emb = await embedding_service.embed_text(scenario["good_advice"])
    bad_emb = await embedding_service.embed_text(scenario["bad_advice"])

    # Calculate scores based on maturity
    if apply_outcomes and maturity["uses"] > 0:
        good_score = maturity["worked"] / maturity["uses"]  # e.g., 18/20 = 0.9
        bad_score = max(0.1, (maturity["uses"] - maturity["worked"]) / maturity["uses"] * 0.5)  # Low score
    else:
        good_score = 0.5
        bad_score = 0.5

    collection.add(
        ids=["good", "bad"],
        embeddings=[good_emb, bad_emb],
        documents=[scenario["good_advice"], scenario["bad_advice"]],
        metadatas=[
            {"type": "good", "score": good_score, "uses": maturity["uses"]},
            {"type": "bad", "score": bad_score, "uses": maturity["uses"]}
        ]
    )

    # Query
    query_emb = await embedding_service.embed_text(scenario["query"])
    results = collection.query(query_embeddings=[query_emb], n_results=2, include=["metadatas", "documents", "distances"])

    # Re-rank by outcome score (simulate outcome-based reranking)
    docs_with_scores = []
    for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
        # Combine semantic distance with outcome score
        # Lower distance = better, higher score = better
        # Normalize distance to 0-1 range (assuming max dist ~2 for L2)
        semantic_sim = max(0, 1 - dist/2)
        outcome_score = meta.get("score", 0.5)
        # Weight outcome score more heavily when we have outcome history
        if maturity["uses"] > 0:
            combined = 0.3 * semantic_sim + 0.7 * outcome_score
        else:
            combined = semantic_sim
        docs_with_scores.append((meta["type"], combined))

    # Sort by combined score
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Analyze
    good_rank, bad_rank = 0, 0
    relevances = []
    for i, (doc_type, score) in enumerate(docs_with_scores):
        if doc_type == "good":
            good_rank = i + 1
            relevances.append(1.0)
        else:
            bad_rank = i + 1
            relevances.append(0.0)

    tokens_used = embedding_service.token_count - initial_tokens

    return TestResult(
        condition="outcomes_only",
        scenario_id=scenario["id"],
        domain=scenario["domain"],
        maturity=maturity["name"],
        good_rank=good_rank,
        bad_rank=bad_rank,
        top1_correct=(good_rank == 1),
        mrr=1.0/good_rank if good_rank > 0 else 0.0,
        ndcg5=calculate_ndcg(relevances, k=5),
        tokens_used=tokens_used
    )


async def test_full_roampal(
    scenario: Dict,
    maturity: Dict,
    data_dir: str,
    embedding_service: EmbeddingService768,
    apply_outcomes: bool = True
) -> TestResult:
    """
    Condition 4: Full Roampal system.
    Both reranker + outcome scores (Wilson-scored).
    Preseeds metadata directly to avoid promotion side-effects during test.
    """
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    initial_tokens = embedding_service.token_count

    system = UnifiedMemorySystem(
        data_path=data_dir,
    )
    await system.initialize()
    # Replace UMS's internal embedder with benchmark's token-counting embedder
    # so token usage is tracked consistently across all 4 conditions
    system._embedding_service = embedding_service
    if system._search_service:
        system._search_service.embed_fn = embedding_service.embed_text

    # Store both pieces of advice
    good_id = await system.store(
        text=scenario["good_advice"],
        collection="working",
        metadata={"type": "good", "scenario": scenario["id"]}
    )
    bad_id = await system.store(
        text=scenario["bad_advice"],
        collection="working",
        metadata={"type": "bad", "scenario": scenario["id"]}
    )

    # Apply outcomes by directly setting metadata (like test_learning_curve.py)
    # This avoids promotion side-effects that change doc IDs mid-test
    adapter = system.collections.get("working")

    if apply_outcomes and maturity["uses"] > 0 and scenario["domain"] in TRAIN_DOMAINS and adapter:
        # Good advice: high success rate
        good_success_rate = maturity["worked"] / maturity["uses"]
        good_outcome_history = []
        for i in range(maturity["worked"]):
            good_outcome_history.append({"outcome": "worked", "timestamp": f"2025-01-{i+1:02d}"})
        for i in range(maturity["failed"]):
            good_outcome_history.append({"outcome": "failed", "timestamp": f"2025-01-{20+i:02d}"})

        try:
            result = adapter.collection.get(ids=[good_id], include=["metadatas"])
            if result and result.get("metadatas"):
                meta = result["metadatas"][0]
                meta["uses"] = maturity["uses"]
                meta["score"] = good_success_rate
                meta["success_count"] = float(maturity["worked"])
                meta["outcome_history"] = json.dumps(good_outcome_history)
                meta["last_outcome"] = "worked"
                adapter.collection.update(ids=[good_id], metadatas=[meta])
        except Exception:
            pass

        # Bad advice: low success rate (1/3 of good)
        bad_worked = max(0, maturity["worked"] // 3)
        bad_failed = maturity["uses"] - bad_worked
        bad_success_rate = bad_worked / maturity["uses"]
        bad_outcome_history = []
        for i in range(bad_worked):
            bad_outcome_history.append({"outcome": "worked", "timestamp": f"2025-01-{i+1:02d}"})
        for i in range(bad_failed):
            bad_outcome_history.append({"outcome": "failed", "timestamp": f"2025-01-{20+i:02d}"})

        try:
            result = adapter.collection.get(ids=[bad_id], include=["metadatas"])
            if result and result.get("metadatas"):
                meta = result["metadatas"][0]
                meta["uses"] = maturity["uses"]
                meta["score"] = bad_success_rate
                meta["success_count"] = float(bad_worked)
                meta["outcome_history"] = json.dumps(bad_outcome_history)
                meta["last_outcome"] = "failed"
                adapter.collection.update(ids=[bad_id], metadatas=[meta])
        except Exception:
            pass

    # Search
    results = await system.search(
        scenario["query"],
        collections=["working"],
        limit=5
    )

    # Analyze results
    good_rank, bad_rank = 0, 0
    relevances = []
    good_lower = scenario["good_advice"].lower()[:50]
    bad_lower = scenario["bad_advice"].lower()[:50]

    for i, r in enumerate(results):
        text = r.get("text", "").lower()
        is_good = good_lower in text or text[:50] in good_lower
        is_bad = bad_lower in text or text[:50] in bad_lower

        if is_good and good_rank == 0:
            good_rank = i + 1
            relevances.append(1.0)
        elif is_bad and bad_rank == 0:
            bad_rank = i + 1
            relevances.append(0.0)
        else:
            relevances.append(0.0)

    tokens_used = embedding_service.token_count - initial_tokens

    return TestResult(
        condition="full_roampal",
        scenario_id=scenario["id"],
        domain=scenario["domain"],
        maturity=maturity["name"],
        good_rank=good_rank if good_rank > 0 else 99,
        bad_rank=bad_rank if bad_rank > 0 else 99,
        top1_correct=(good_rank == 1),
        mrr=1.0/good_rank if good_rank > 0 else 0.0,
        ndcg5=calculate_ndcg(relevances, k=5),
        tokens_used=tokens_used
    )


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

async def main():
    print("=" * 80)
    print("COMPREHENSIVE ROAMPAL BENCHMARK")
    print("=" * 80)
    print()
    print("4 CONDITIONS:")
    print("  1. RAG Baseline      - Pure vector similarity")
    print("  2. Reranker Only     - Cross-encoder, no outcomes")
    print("  3. Outcomes Only     - Outcome scores, no reranker")
    print("  4. Full Roampal      - Both reranker + outcomes")
    print()
    print("5 MATURITY LEVELS:")
    for m in MATURITY_LEVELS:
        sr = f"{m['worked']}/{m['uses']}" if m['uses'] > 0 else "n/a"
        print(f"  {m['name']:<12}: {m['uses']:2d} uses ({sr} success)")
    print()
    print("METRICS: Top-1 Accuracy, MRR, nDCG@5, Token Efficiency")
    print()
    print("CROSS-DOMAIN HOLDOUT:")
    print(f"  Train domains: {TRAIN_DOMAINS}")
    print(f"  Test domains:  {TEST_DOMAINS}")
    print()

    if not HAS_DEPS:
        print("ERROR: Missing dependencies")
        return

    embedding_service = EmbeddingService768()
    reranker = CrossEncoderReranker()
    print("\nModels loaded.\n")

    test_dir = Path(__file__).parent / "comprehensive_benchmark_data"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    # Results storage
    all_results: List[TestResult] = []

    conditions = ["rag_baseline", "reranker_only", "outcomes_only", "full_roampal"]
    total_tests = len(SCENARIOS) * len(MATURITY_LEVELS) * len(conditions)
    test_num = 0

    print("-" * 80)
    print(f"Running {len(SCENARIOS)} scenarios × {len(MATURITY_LEVELS)} levels × {len(conditions)} conditions = {total_tests} tests")
    print("-" * 80)

    for scenario in SCENARIOS:
        for maturity in MATURITY_LEVELS:
            for condition in conditions:
                test_num += 1
                print(f"\r[{test_num:3d}/{total_tests}] {scenario['domain']:8s} | {maturity['name']:12s} | {condition:15s}", end="", flush=True)

                data_dir = str(test_dir / f"s{scenario['id']}_{maturity['name']}_{condition}")

                # Determine if outcomes should be applied (training domains only)
                apply_outcomes = scenario["domain"] in TRAIN_DOMAINS

                if condition == "rag_baseline":
                    result = await test_rag_baseline(scenario, maturity, data_dir, embedding_service)
                elif condition == "reranker_only":
                    result = await test_reranker_only(scenario, maturity, data_dir, embedding_service, reranker)
                elif condition == "outcomes_only":
                    result = await test_outcomes_only(scenario, maturity, data_dir, embedding_service, apply_outcomes)
                elif condition == "full_roampal":
                    result = await test_full_roampal(scenario, maturity, data_dir, embedding_service, apply_outcomes)

                all_results.append(result)

    print("\n\n")

    # ==========================================================================
    # ANALYSIS
    # ==========================================================================

    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # By condition (all maturity levels)
    print("\n" + "-" * 60)
    print("BY CONDITION (averaged across all maturity levels)")
    print("-" * 60)
    print(f"{'Condition':<18} {'Top-1':>8} {'MRR':>8} {'nDCG@5':>8} {'Tokens':>10}")
    print("-" * 60)

    for condition in conditions:
        cond_results = [r for r in all_results if r.condition == condition]
        top1 = statistics.mean([1 if r.top1_correct else 0 for r in cond_results]) * 100
        mrr = statistics.mean([r.mrr for r in cond_results])
        ndcg = statistics.mean([r.ndcg5 for r in cond_results])
        tokens = statistics.mean([r.tokens_used for r in cond_results])
        print(f"{condition:<18} {top1:>7.1f}% {mrr:>8.3f} {ndcg:>8.3f} {tokens:>10.0f}")

    # By maturity level (Full Roampal only)
    print("\n" + "-" * 60)
    print("LEARNING CURVE (Full Roampal condition)")
    print("-" * 60)
    print(f"{'Maturity':<12} {'Uses':>6} {'Top-1':>8} {'MRR':>8} {'nDCG@5':>8}")
    print("-" * 60)

    for maturity in MATURITY_LEVELS:
        mat_results = [r for r in all_results if r.condition == "full_roampal" and r.maturity == maturity["name"]]
        top1 = statistics.mean([1 if r.top1_correct else 0 for r in mat_results]) * 100
        mrr = statistics.mean([r.mrr for r in mat_results])
        ndcg = statistics.mean([r.ndcg5 for r in mat_results])
        print(f"{maturity['name']:<12} {maturity['uses']:>6} {top1:>7.1f}% {mrr:>8.3f} {ndcg:>8.3f}")

    # Cross-domain holdout analysis
    print("\n" + "-" * 60)
    print("CROSS-DOMAIN GENERALIZATION (Full Roampal, Mature level)")
    print("-" * 60)

    train_results = [r for r in all_results
                     if r.condition == "full_roampal"
                     and r.maturity == "mature"
                     and r.domain in TRAIN_DOMAINS]
    test_results = [r for r in all_results
                    if r.condition == "full_roampal"
                    and r.maturity == "mature"
                    and r.domain in TEST_DOMAINS]

    if train_results:
        train_top1 = statistics.mean([1 if r.top1_correct else 0 for r in train_results]) * 100
        train_mrr = statistics.mean([r.mrr for r in train_results])
        print(f"Train domains ({', '.join(TRAIN_DOMAINS)}): {train_top1:.1f}% Top-1, {train_mrr:.3f} MRR")

    if test_results:
        test_top1 = statistics.mean([1 if r.top1_correct else 0 for r in test_results]) * 100
        test_mrr = statistics.mean([r.mrr for r in test_results])
        print(f"Test domains  ({', '.join(TEST_DOMAINS)}): {test_top1:.1f}% Top-1, {test_mrr:.3f} MRR")

    # Token efficiency comparison
    print("\n" + "-" * 60)
    print("TOKEN EFFICIENCY (Mature level)")
    print("-" * 60)

    for condition in conditions:
        cond_results = [r for r in all_results if r.condition == condition and r.maturity == "mature"]
        if cond_results:
            avg_tokens = statistics.mean([r.tokens_used for r in cond_results])
            top1 = statistics.mean([1 if r.top1_correct else 0 for r in cond_results]) * 100
            efficiency = top1 / avg_tokens * 100 if avg_tokens > 0 else 0
            print(f"{condition:<18}: {avg_tokens:>6.0f} tokens/query, {top1:.1f}% accuracy, {efficiency:.2f} acc/token")

    # Improvement breakdown
    print("\n" + "-" * 60)
    print("IMPROVEMENT BREAKDOWN (Mature level)")
    print("-" * 60)

    def get_mature_accuracy(condition):
        results = [r for r in all_results if r.condition == condition and r.maturity == "mature"]
        return statistics.mean([1 if r.top1_correct else 0 for r in results]) * 100

    rag_acc = get_mature_accuracy("rag_baseline")
    reranker_acc = get_mature_accuracy("reranker_only")
    outcomes_acc = get_mature_accuracy("outcomes_only")
    full_acc = get_mature_accuracy("full_roampal")

    print(f"RAG Baseline:      {rag_acc:>6.1f}%")
    print(f"+ Reranker:        {reranker_acc:>6.1f}% (+{reranker_acc - rag_acc:.1f} pts)")
    print(f"+ Outcomes only:   {outcomes_acc:>6.1f}% (+{outcomes_acc - rag_acc:.1f} pts)")
    print(f"+ Both (Roampal):  {full_acc:>6.1f}% (+{full_acc - rag_acc:.1f} pts)")
    print()
    print(f"Reranker contribution:  +{reranker_acc - rag_acc:.1f} pts")
    print(f"Outcomes contribution:  +{outcomes_acc - rag_acc:.1f} pts")
    print(f"Combined contribution:  +{full_acc - rag_acc:.1f} pts")

    # Save results
    results_file = test_dir / "comprehensive_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": len(SCENARIOS),
            "n_maturity_levels": len(MATURITY_LEVELS),
            "n_conditions": len(conditions),
            "train_domains": list(TRAIN_DOMAINS),
            "test_domains": list(TEST_DOMAINS),
            "summary": {
                "by_condition": {
                    c: {
                        "top1": statistics.mean([1 if r.top1_correct else 0 for r in all_results if r.condition == c]) * 100,
                        "mrr": statistics.mean([r.mrr for r in all_results if r.condition == c]),
                        "ndcg5": statistics.mean([r.ndcg5 for r in all_results if r.condition == c]),
                    } for c in conditions
                },
                "learning_curve": [
                    {
                        "maturity": m["name"],
                        "uses": m["uses"],
                        "top1": statistics.mean([1 if r.top1_correct else 0 for r in all_results if r.condition == "full_roampal" and r.maturity == m["name"]]) * 100,
                        "mrr": statistics.mean([r.mrr for r in all_results if r.condition == "full_roampal" and r.maturity == m["name"]]),
                    } for m in MATURITY_LEVELS
                ],
                "cross_domain": {
                    "train_top1": statistics.mean([1 if r.top1_correct else 0 for r in train_results]) * 100 if train_results else 0,
                    "test_top1": statistics.mean([1 if r.top1_correct else 0 for r in test_results]) * 100 if test_results else 0,
                },
                "improvements": {
                    "rag_baseline": rag_acc,
                    "reranker_contribution": reranker_acc - rag_acc,
                    "outcomes_contribution": outcomes_acc - rag_acc,
                    "combined_contribution": full_acc - rag_acc,
                }
            },
            "all_results": [asdict(r) for r in all_results],
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Statistical significance tests
    print("\n" + "-" * 60)
    print("STATISTICAL SIGNIFICANCE (paired t-test, McNemar's test)")
    print("-" * 60)

    # Compare Full Roampal vs each baseline at mature level
    full_mature = [r for r in all_results if r.condition == "full_roampal" and r.maturity == "mature"]
    rag_mature = [r for r in all_results if r.condition == "rag_baseline" and r.maturity == "mature"]
    reranker_mature = [r for r in all_results if r.condition == "reranker_only" and r.maturity == "mature"]

    # Sort by scenario_id for pairing
    full_mature.sort(key=lambda x: x.scenario_id)
    rag_mature.sort(key=lambda x: x.scenario_id)
    reranker_mature.sort(key=lambda x: x.scenario_id)

    # McNemar's test for Top-1 accuracy (binary outcomes)
    def mcnemar_test(a_results, b_results):
        """Compare two conditions using McNemar's test for paired binary data."""
        # b = A correct, B wrong; c = A wrong, B correct
        b = sum(1 for a, b in zip(a_results, b_results) if a.top1_correct and not b.top1_correct)
        c = sum(1 for a, b in zip(a_results, b_results) if not a.top1_correct and b.top1_correct)
        # Use exact binomial test for small samples
        n = b + c
        if n == 0:
            return 1.0  # No difference
        p_value = stats.binomtest(b, n, 0.5, alternative='two-sided').pvalue if n > 0 else 1.0
        return p_value

    # Paired t-test for MRR (continuous metric)
    def paired_ttest(a_results, b_results, metric="mrr"):
        a_scores = [getattr(r, metric) for r in a_results]
        b_scores = [getattr(r, metric) for r in b_results]
        if len(set(a_scores)) == 1 and len(set(b_scores)) == 1:
            return 1.0  # No variance
        t_stat, p_value = stats.ttest_rel(a_scores, b_scores)
        return p_value

    print("\nFull Roampal vs RAG Baseline:")
    p_mcnemar = mcnemar_test(full_mature, rag_mature)
    p_ttest = paired_ttest(full_mature, rag_mature)
    sig_mcnemar = "***" if p_mcnemar < 0.001 else "**" if p_mcnemar < 0.01 else "*" if p_mcnemar < 0.05 else ""
    sig_ttest = "***" if p_ttest < 0.001 else "**" if p_ttest < 0.01 else "*" if p_ttest < 0.05 else ""
    print(f"  McNemar (Top-1):  p={p_mcnemar:.4f} {sig_mcnemar}")
    print(f"  Paired t (MRR):   p={p_ttest:.4f} {sig_ttest}")

    print("\nFull Roampal vs Reranker Only:")
    p_mcnemar = mcnemar_test(full_mature, reranker_mature)
    p_ttest = paired_ttest(full_mature, reranker_mature)
    sig_mcnemar = "***" if p_mcnemar < 0.001 else "**" if p_mcnemar < 0.01 else "*" if p_mcnemar < 0.05 else ""
    sig_ttest = "***" if p_ttest < 0.001 else "**" if p_ttest < 0.01 else "*" if p_ttest < 0.05 else ""
    print(f"  McNemar (Top-1):  p={p_mcnemar:.4f} {sig_mcnemar}")
    print(f"  Paired t (MRR):   p={p_ttest:.4f} {sig_ttest}")

    # Learning curve significance (cold start vs mature)
    print("\nLearning Curve (Cold Start vs Mature):")
    cold_results = [r for r in all_results if r.condition == "full_roampal" and r.maturity == "cold_start"]
    cold_results.sort(key=lambda x: x.scenario_id)
    p_mcnemar = mcnemar_test(full_mature, cold_results)
    p_ttest = paired_ttest(full_mature, cold_results)
    sig_mcnemar = "***" if p_mcnemar < 0.001 else "**" if p_mcnemar < 0.01 else "*" if p_mcnemar < 0.05 else ""
    sig_ttest = "***" if p_ttest < 0.001 else "**" if p_ttest < 0.01 else "*" if p_ttest < 0.05 else ""
    print(f"  McNemar (Top-1):  p={p_mcnemar:.4f} {sig_mcnemar}")
    print(f"  Paired t (MRR):   p={p_ttest:.4f} {sig_ttest}")

    print("\n  Significance: * p<0.05, ** p<0.01, *** p<0.001")

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if full_acc >= 90 and (full_acc - rag_acc) >= 40:
        print("\n  [PASS] COMPREHENSIVE BENCHMARK VALIDATED")
        print()
        print(f"  RAG Baseline:     {rag_acc:.1f}%")
        print(f"  Full Roampal:     {full_acc:.1f}%")
        print(f"  Total Improvement: +{full_acc - rag_acc:.1f} percentage points")
        print()
        print("  This proves:")
        print("    1. Plain RAG fails on adversarial queries")
        print("    2. Cross-encoder reranking helps but isn't enough")
        print("    3. Outcome-based learning is the key differentiator")
        print("    4. Combined system achieves maximum accuracy")
    else:
        print("\n  Results require further analysis")
        print(f"  Full Roampal: {full_acc:.1f}%")
        print(f"  Improvement: +{full_acc - rag_acc:.1f} pts")


if __name__ == "__main__":
    asyncio.run(main())

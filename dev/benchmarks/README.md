# Roampal Benchmarks (Real Embeddings)

**Purpose**: Measure retrieval quality using real embeddings (all-mpnet-base-v2, 768d) on adversarial scenarios designed to defeat pure vector similarity.

> **Integration tests** (infrastructure validation with mock embeddings) have been moved to `dev/tests/integration/`. Those tests validate system mechanics (dedup, scoring, capacity, concurrent access) — not retrieval quality.

## Benchmarks

### 1. Comprehensive 4-Way Benchmark (200 tests, ~5 min)
**Purpose**: Definitive comparison of RAG vs Reranker vs Outcomes vs Full Roampal

```bash
cd benchmarks
python test_comprehensive_benchmark.py
```

**Design**: 4 conditions x 5 maturity levels x 10 adversarial scenarios = 200 tests

**Conditions**:
- RAG Baseline: Pure ChromaDB L2 distance
- Reranker Only: Vector + ms-marco cross-encoder (no outcomes)
- Outcomes Only: Vector + Wilson scoring (no reranker)
- Full Roampal: Vector + reranker + Wilson scoring

**Metrics**: Top-1 Accuracy, MRR, nDCG@5, Token Efficiency

**Results** (verified 2026-02-06):
| Condition | Top-1 | MRR | nDCG@5 |
|-----------|-------|-----|--------|
| RAG Baseline | 10% | 0.550 | 0.668 |
| Reranker Only | 20% | 0.600 | 0.705 |
| Outcomes Only | 50% | 0.750 | 0.815 |
| Full Roampal | 42% | 0.710 | 0.786 |

> **Note:** Full Roampal < Outcomes Only at aggregate level because CE reranker promoted semantically-closer bad advice at low maturity. **v0.3.2 fix:** CE weight drops to 0 for scored memories (uses >= 3), preserving Wilson rankings. At mature level, Full Roampal matches Outcomes Only at 60%. CE still helps cold-start discovery.

**Learning Curve** (Full Roampal):
| Maturity | Uses | Top-1 | MRR |
|----------|------|-------|-----|
| Cold Start | 0 | 10% | 0.550 |
| Early | 3 | 40% | 0.700 |
| Established | 5 | 50% | 0.750 |
| Proven | 10 | 50% | 0.750 |
| Mature | 20 | 60% | 0.800 |

**Key Finding**: Outcome learning (+50 pts) dominates reranker (+10 pts) by 5x. Gradual improvement curve validates Wilson + dynamic weights.

### 2. Learning Curve Test (50 tests, ~2 min)
**Purpose**: Prove outcome history improves adversarial resistance

```bash
cd benchmarks
python test_learning_curve.py
```

**Design**: 10 scenarios x 5 maturity levels (cold_start -> mature)

**Results** (verified 2026-02-06):
| Maturity | Uses | Accuracy |
|----------|------|----------|
| Cold Start | 0 | 20% |
| Early | 3 | 20% |
| Established | 5 | 30% |
| Proven | 10 | 60% |
| Mature | 20 | 100% |

**Key Finding**: Gradual improvement from 20% to 100% as outcome history accumulates. 100% accuracy at mature level across all 5 domains.

### 3. Outcome Learning A/B Test (10 adversarial scenarios, ~60s)
**Purpose**: Prove outcome scoring overrides misleading semantic similarity

```bash
cd benchmarks
python test_outcome_learning_ab.py
```

**Design**: 10 adversarial scenarios (8 coding + 2 finance)
- Each scenario has good advice and bad advice
- Queries designed to semantically match BAD advice better
- Treatment: Apply outcome scores (good->0.9 via "worked", bad->0.2 via "failed")
- Control: No outcome scoring (pure vector similarity)

**Results** (verified 2026-02-06):
| Condition | Precision |
|-----------|-----------|
| WITH outcomes | **30%** (3/10) |
| WITHOUT outcomes | 0% (0/10) |

**Statistical Significance**:
- Cohen's d: 0.88 (LARGE effect)
- t-statistic: 1.96
- p-value: 0.081 (trending, n=10 is small sample)

> **Note:** Small sample size (n=10) limits statistical power. Effect size is large (d=0.88) but p > 0.05. The 4-way benchmark (n=200) and vs-vector-db test (n=30) provide stronger statistical evidence.

### 4. Token Efficiency Benchmark (100 scenarios, ~3 min)
**Purpose**: Measure per-query token efficiency vs traditional RAG retrieval

```bash
cd benchmarks
python test_token_efficiency.py
```

**Design**: 100 adversarial scenarios across 10 categories
- Queries designed to semantically match the WRONG answer
- Real embeddings: all-mpnet-base-v2 (768d)
- 3 conditions: RAG top-3, RAG top-5, Roampal (limit=1 outcome-weighted)

**Results**:
| Metric | RAG Top-3 | RAG Top-5 | Roampal |
|--------|-----------|-----------|---------|
| Avg tokens/query | 54.5 | 92.6 | 15.8 |
| Hit@1 (accuracy) | 1% | 1% | 0% |
| Efficiency ratio | 1.83 | 1.08 | 0.00 |

**Token Reduction** (structural — fewer results returned):
- vs RAG Top-3: **71% fewer tokens** (54 -> 16)
- vs RAG Top-5: **83% fewer tokens** (93 -> 16)
- 3.4-5.9x fewer tokens per query

> **Caveat:** All conditions score near-zero accuracy on these adversarial queries. Token reduction reflects `limit=1` vs `limit=3/5` — a structural property, not a quality improvement. This benchmark measures token cost per query, not retrieval quality. See the A/B test and 4-way benchmark for quality comparisons.

### 5. Roampal vs Vector DB (30 adversarial scenarios, ~2 min)
**Purpose**: Direct comparison against pure vector database retrieval

```bash
cd benchmarks
python test_roampal_vs_vector_db.py
```

**Results** (verified 2026-02-06):
- Roampal: 40% (12/30)
- Vector DB: 0% (0/30)
- p = 0.000135 (highly significant)
- Cohen's d: 1.14 (LARGE effect)
- 95% CI: [21.4%, 58.6%]

## Limitations

These benchmarks are designed for internal development validation, not as a published study. Known limitations:

1. **2-document scenarios only.** Each test pits one good memory against one bad memory. Real-world usage involves hundreds of memories with varying quality. Retrieval-at-scale is not tested.
2. **Adversarial-only design.** All queries are designed to trick vector search. Normal (non-adversarial) queries where semantic similarity aligns with quality are not benchmarked. The cross-encoder reranker likely helps on those.
3. **No cross-validation on scenarios.** The same 10 scenario templates are used for development and evaluation. There is no held-out test set for the scenarios themselves (though cross-domain holdout exists within the 4-way benchmark).
4. **No latency measurement** for real-embedding benchmarks. Latency benchmarks exist only for mock embeddings (p50 ~40ms). Real embedding + cross-encoder latency is not profiled.
5. **Small sample sizes** for some tests. The A/B test (n=10) does not reach statistical significance (p=0.081). The vs-vector-db (n=30) and 4-way (n=200) tests provide stronger evidence.

## Supporting Files

- `test_data_fixtures.py` - Shared adversarial test data
- `test_dynamic_weight_shift.py` - Dynamic weight shift test (requires sentence-transformers)
- `results/` - Saved benchmark output files
- `results/SUMMARY.txt` - Consolidated results summary

## Integration Tests (Mock Embeddings)

Infrastructure validation tests using mock embeddings (deterministic, no model required) are in [`dev/tests/integration/`](../tests/integration/):

- **Comprehensive Test** (30 tests) - All core features: 5 tiers, 3 KGs, scoring, promotion, dedup
- **Torture Suite** (10 tests) - Stress: 1000 stores, adversarial dedup, capacity, concurrent access
- **Semantic Confusion** - Quality ranking vs semantic noise (15:1 ratio)
- **Contradictions** - Quality-based resolution of conflicting facts
- **Edge Cases** (24 tests) - Unicode, injection, boundaries, malformed input
- **Context Poisoning** - Resistance to adversarial data attacks
- **Catastrophic Forgetting** - Old knowledge survives new additions
- **Learning Sabotage** (10 tests) - Resistance to manipulation attacks
- **Search Quality** (6 metrics) - Synonyms, typos, acronyms, diversity, relevance
- **Recovery Resilience** (6 tests) - Failure recovery, corruption handling
- **Learning Speed** (7 tests) - Learning curve, adaptation, retention
- **Latency Benchmark** - p50/p95/p99 at various collection sizes
- **Stale Data** - Fresh high-quality beats stale low-quality

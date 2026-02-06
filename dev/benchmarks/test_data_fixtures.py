"""
Deterministic test data for comprehensive memory system testing.
No LLM required - all data is pre-defined.
"""

# Storage test data (50 entries across all 5 collections)
STORAGE_FIXTURES = [
    # Books (10 items - permanent reference material)
    {"text": "Python uses dynamic typing and supports multiple programming paradigms", "collection": "books", "metadata": {"title": "Python Basics", "author": "Guide"}},
    {"text": "Docker containers provide isolation using Linux namespaces and cgroups", "collection": "books", "metadata": {"title": "Docker Guide", "author": "DevOps"}},
    {"text": "REST APIs use HTTP methods GET POST PUT DELETE for CRUD operations", "collection": "books", "metadata": {"title": "REST API Design", "author": "Web"}},
    {"text": "PostgreSQL supports ACID transactions with MVCC for concurrency control", "collection": "books", "metadata": {"title": "PostgreSQL Internals", "author": "Database"}},
    {"text": "React uses virtual DOM and component-based architecture for UI rendering", "collection": "books", "metadata": {"title": "React Fundamentals", "author": "Frontend"}},
    {"text": "Git uses SHA-1 hashing for content addressing and Merkle trees for history", "collection": "books", "metadata": {"title": "Git Internals", "author": "VCS"}},
    {"text": "Kubernetes orchestrates containers across clusters with declarative configuration", "collection": "books", "metadata": {"title": "K8s Guide", "author": "DevOps"}},
    {"text": "JWT tokens encode user claims and are signed to prevent tampering", "collection": "books", "metadata": {"title": "Authentication", "author": "Security"}},
    {"text": "Redis provides in-memory data structures with persistence and replication", "collection": "books", "metadata": {"title": "Redis Guide", "author": "Cache"}},
    {"text": "GraphQL allows clients to request exactly the data they need in a single query", "collection": "books", "metadata": {"title": "GraphQL Basics", "author": "API"}},

    # Working (20 items - current conversation context)
    {"text": "User asked how to debug Docker networking issues", "collection": "working", "metadata": {"conversation_id": "conv_001", "role": "exchange"}},
    {"text": "User wants to understand JWT token expiration handling", "collection": "working", "metadata": {"conversation_id": "conv_001", "role": "exchange"}},
    {"text": "User asked about PostgreSQL query optimization techniques", "collection": "working", "metadata": {"conversation_id": "conv_002", "role": "exchange"}},
    {"text": "User is learning React hooks and asked about useEffect dependencies", "collection": "working", "metadata": {"conversation_id": "conv_002", "role": "exchange"}},
    {"text": "User wants to set up Git pre-commit hooks for linting", "collection": "working", "metadata": {"conversation_id": "conv_003", "role": "exchange"}},
    {"text": "User asked how to configure Kubernetes ingress controllers", "collection": "working", "metadata": {"conversation_id": "conv_003", "role": "exchange"}},
    {"text": "User debugging Redis connection timeout errors", "collection": "working", "metadata": {"conversation_id": "conv_004", "role": "exchange"}},
    {"text": "User wants to implement GraphQL subscriptions for real-time updates", "collection": "working", "metadata": {"conversation_id": "conv_004", "role": "exchange"}},
    {"text": "User asked about Python asyncio best practices", "collection": "working", "metadata": {"conversation_id": "conv_005", "role": "exchange"}},
    {"text": "User implementing REST API pagination with cursor-based approach", "collection": "working", "metadata": {"conversation_id": "conv_005", "role": "exchange"}},
    {"text": "User debugging Docker build cache invalidation", "collection": "working", "metadata": {"conversation_id": "conv_006", "role": "exchange"}},
    {"text": "User asked about JWT refresh token rotation strategies", "collection": "working", "metadata": {"conversation_id": "conv_006", "role": "exchange"}},
    {"text": "User optimizing PostgreSQL indexes for query performance", "collection": "working", "metadata": {"conversation_id": "conv_007", "role": "exchange"}},
    {"text": "User learning React context API for state management", "collection": "working", "metadata": {"conversation_id": "conv_007", "role": "exchange"}},
    {"text": "User setting up Git rebase workflow for clean history", "collection": "working", "metadata": {"conversation_id": "conv_008", "role": "exchange"}},
    {"text": "User configuring Kubernetes resource limits and requests", "collection": "working", "metadata": {"conversation_id": "conv_008", "role": "exchange"}},
    {"text": "User implementing Redis pub/sub for event notifications", "collection": "working", "metadata": {"conversation_id": "conv_009", "role": "exchange"}},
    {"text": "User designing GraphQL schema with proper type definitions", "collection": "working", "metadata": {"conversation_id": "conv_009", "role": "exchange"}},
    {"text": "User asked about Python type hints and mypy static checking", "collection": "working", "metadata": {"conversation_id": "conv_010", "role": "exchange"}},
    {"text": "User implementing REST API rate limiting with Redis", "collection": "working", "metadata": {"conversation_id": "conv_010", "role": "exchange"}},

    # History (10 items - past conversations)
    {"text": "User previously asked about Docker multi-stage builds", "collection": "history", "metadata": {"conversation_id": "conv_old_001", "role": "exchange", "score": 0.6}},
    {"text": "User learned JWT signature verification process", "collection": "history", "metadata": {"conversation_id": "conv_old_002", "role": "exchange", "score": 0.7}},
    {"text": "User implemented PostgreSQL connection pooling", "collection": "history", "metadata": {"conversation_id": "conv_old_003", "role": "exchange", "score": 0.8}},
    {"text": "User mastered React component lifecycle methods", "collection": "history", "metadata": {"conversation_id": "conv_old_004", "role": "exchange", "score": 0.75}},
    {"text": "User configured Git SSH authentication successfully", "collection": "history", "metadata": {"conversation_id": "conv_old_005", "role": "exchange", "score": 0.65}},
    {"text": "User deployed Kubernetes application with Helm charts", "collection": "history", "metadata": {"conversation_id": "conv_old_006", "role": "exchange", "score": 0.85}},
    {"text": "User implemented Redis caching strategy for API", "collection": "history", "metadata": {"conversation_id": "conv_old_007", "role": "exchange", "score": 0.7}},
    {"text": "User built GraphQL resolver with data loaders", "collection": "history", "metadata": {"conversation_id": "conv_old_008", "role": "exchange", "score": 0.8}},
    {"text": "User wrote Python unit tests with pytest fixtures", "collection": "history", "metadata": {"conversation_id": "conv_old_009", "role": "exchange", "score": 0.75}},
    {"text": "User designed REST API versioning strategy", "collection": "history", "metadata": {"conversation_id": "conv_old_010", "role": "exchange", "score": 0.7}},

    # Patterns (5 items - proven solutions)
    {"text": "Docker networking: use bridge networks for inter-container communication", "collection": "patterns", "metadata": {"score": 0.95, "uses": 12}},
    {"text": "JWT tokens: always validate signature and check expiration before trusting claims", "collection": "patterns", "metadata": {"score": 0.92, "uses": 8}},
    {"text": "PostgreSQL: add indexes on foreign keys and frequently filtered columns", "collection": "patterns", "metadata": {"score": 0.98, "uses": 15}},
    {"text": "React: use useCallback to memoize event handlers passed to child components", "collection": "patterns", "metadata": {"score": 0.9, "uses": 10}},
    {"text": "Git: squash feature branch commits before merging to main for clean history", "collection": "patterns", "metadata": {"score": 0.93, "uses": 9}},

    # Memory_bank (15 items - permanent user/system facts)
    {"text": "User prefers Docker Compose for local development environments", "collection": "memory_bank", "metadata": {"tags": ["preference"], "importance": 0.8, "confidence": 0.9}},
    {"text": "User is experienced with Python and prefers asyncio for I/O-bound tasks", "collection": "memory_bank", "metadata": {"tags": ["identity", "skill"], "importance": 0.9, "confidence": 0.95}},
    {"text": "User works at TechCorp as a senior backend engineer", "collection": "memory_bank", "metadata": {"tags": ["identity", "context"], "importance": 0.85, "confidence": 0.9}},
    {"text": "User's current project uses PostgreSQL as primary database", "collection": "memory_bank", "metadata": {"tags": ["project", "context"], "importance": 0.8, "confidence": 0.85}},
    {"text": "User prefers functional programming style over object-oriented", "collection": "memory_bank", "metadata": {"tags": ["preference", "style"], "importance": 0.7, "confidence": 0.8}},
    {"text": "User is building a real-time chat application with WebSocket", "collection": "memory_bank", "metadata": {"tags": ["project", "goal"], "importance": 0.9, "confidence": 0.9}},
    {"text": "User has deadline for MVP release in 2 weeks", "collection": "memory_bank", "metadata": {"tags": ["project", "deadline"], "importance": 0.95, "confidence": 1.0}},
    {"text": "User dislikes overly verbose code and prefers concise solutions", "collection": "memory_bank", "metadata": {"tags": ["preference", "style"], "importance": 0.6, "confidence": 0.75}},
    {"text": "User is learning Kubernetes and wants production-ready examples", "collection": "memory_bank", "metadata": {"tags": ["goal", "learning"], "importance": 0.85, "confidence": 0.8}},
    {"text": "User's team uses GitHub Actions for CI/CD pipelines", "collection": "memory_bank", "metadata": {"tags": ["context", "workflow"], "importance": 0.7, "confidence": 0.85}},
    {"text": "User values code readability and maintainability over premature optimization", "collection": "memory_bank", "metadata": {"tags": ["preference", "philosophy"], "importance": 0.8, "confidence": 0.9}},
    {"text": "User has production experience with Redis for caching and session storage", "collection": "memory_bank", "metadata": {"tags": ["identity", "skill"], "importance": 0.75, "confidence": 0.85}},
    {"text": "User prefers pytest over unittest for Python testing", "collection": "memory_bank", "metadata": {"tags": ["preference", "tool"], "importance": 0.6, "confidence": 0.8}},
    {"text": "User's application handles 10000 requests per minute at peak load", "collection": "memory_bank", "metadata": {"tags": ["project", "scale"], "importance": 0.85, "confidence": 0.9}},
    {"text": "User is interested in learning GraphQL federation for microservices", "collection": "memory_bank", "metadata": {"tags": ["goal", "learning"], "importance": 0.7, "confidence": 0.75}},
]

# Search test queries (20 queries with expected best collections)
SEARCH_FIXTURES = [
    {"query": "Docker networking", "expected_collection": "patterns"},
    {"query": "JWT token validation", "expected_collection": "patterns"},
    {"query": "PostgreSQL indexing", "expected_collection": "patterns"},
    {"query": "React hooks", "expected_collection": "working"},
    {"query": "Git workflow", "expected_collection": "patterns"},
    {"query": "Kubernetes deployment", "expected_collection": "working"},
    {"query": "Redis caching", "expected_collection": "history"},
    {"query": "GraphQL schema", "expected_collection": "working"},
    {"query": "Python asyncio", "expected_collection": "working"},
    {"query": "REST API design", "expected_collection": "books"},
    {"query": "user preferences", "expected_collection": "memory_bank"},
    {"query": "current project", "expected_collection": "memory_bank"},
    {"query": "user skills", "expected_collection": "memory_bank"},
    {"query": "authentication best practices", "expected_collection": "books"},
    {"query": "container orchestration", "expected_collection": "books"},
    {"query": "database optimization", "expected_collection": "books"},
    {"query": "frontend framework", "expected_collection": "books"},
    {"query": "version control", "expected_collection": "books"},
    {"query": "API design patterns", "expected_collection": "books"},
    {"query": "caching strategies", "expected_collection": "books"},
]

# Outcome test scenarios (30 cases)
OUTCOME_FIXTURES = [
    {"outcome": "worked", "expected_delta": 0.2, "initial_score": 0.5, "expected_final": 0.7},
    {"outcome": "failed", "expected_delta": -0.3, "initial_score": 0.5, "expected_final": 0.2},
    {"outcome": "partial", "expected_delta": 0.05, "initial_score": 0.5, "expected_final": 0.55},
    {"outcome": "unknown", "expected_delta": 0.0, "initial_score": 0.5, "expected_final": 0.5},
    {"outcome": "worked", "expected_delta": 0.2, "initial_score": 0.9, "expected_final": 1.0},  # Cap at 1.0
    {"outcome": "failed", "expected_delta": -0.3, "initial_score": 0.2, "expected_final": 0.1},  # Floor at 0.1
    {"outcome": "worked", "expected_delta": 0.2, "initial_score": 0.95, "expected_final": 1.0},  # Already near max
    {"outcome": "failed", "expected_delta": -0.3, "initial_score": 0.15, "expected_final": 0.1},  # Already near min
]

# Promotion test scenarios
PROMOTION_FIXTURES = [
    {"collection": "working", "score": 0.8, "uses": 3, "should_promote": True, "target": "history"},
    {"collection": "working", "score": 0.6, "uses": 3, "should_promote": False, "target": None},
    {"collection": "working", "score": 0.8, "uses": 1, "should_promote": False, "target": None},
    {"collection": "history", "score": 0.95, "uses": 4, "should_promote": True, "target": "patterns"},
    {"collection": "history", "score": 0.85, "uses": 4, "should_promote": False, "target": None},
    {"collection": "history", "score": 0.95, "uses": 2, "should_promote": False, "target": None},
]

# Fast-track promotion scenarios
FAST_TRACK_FIXTURES = [
    {"score": 0.95, "uses": 3, "outcomes": ["worked", "worked", "worked"], "should_fast_track": True},
    {"score": 0.95, "uses": 3, "outcomes": ["worked", "worked", "partial"], "should_fast_track": False},
    {"score": 0.95, "uses": 3, "outcomes": ["worked", "failed", "worked"], "should_fast_track": False},
    {"score": 0.85, "uses": 3, "outcomes": ["worked", "worked", "worked"], "should_fast_track": False},
    {"score": 0.95, "uses": 2, "outcomes": ["worked", "worked"], "should_fast_track": False},
]

# Demotion test scenarios
DEMOTION_FIXTURES = [
    {"collection": "patterns", "score": 0.2, "should_demote": True, "target": "history"},
    {"collection": "patterns", "score": 0.4, "should_demote": False, "target": None},
    {"collection": "patterns", "score": 0.1, "should_demote": True, "target": "history"},
]

# Deletion test scenarios
DELETION_FIXTURES = [
    {"score": 0.15, "age_days": 10, "should_delete": True},
    {"score": 0.15, "age_days": 3, "should_delete": False},
    {"score": 0.05, "age_days": 3, "should_delete": True},
    {"score": 0.95, "age_days": 100, "should_delete": False},  # High-value never deletes
]

# Deduplication test scenarios
DEDUP_FIXTURES = [
    {
        "original": {"text": "User prefers Docker for development", "importance": 0.7, "confidence": 0.8},
        "duplicate": {"text": "User prefers Docker for development environments", "importance": 0.9, "confidence": 0.9},
        "similarity": 0.96,
        "expected_action": "replace_with_higher_quality"
    },
    {
        "original": {"text": "PostgreSQL indexing best practices", "importance": 0.9, "confidence": 0.95},
        "duplicate": {"text": "PostgreSQL indexing best practices explained", "importance": 0.7, "confidence": 0.8},
        "similarity": 0.97,
        "expected_action": "keep_original_increment_count"
    },
]

# Memory_bank capacity test
CAPACITY_FIXTURES = {
    "max_items": 500,
    "test_overflow": True
}

# Entity extraction test data
ENTITY_FIXTURES = [
    {"text": "User prefers Docker Compose for local development environments", "expected_entities": ["docker", "compose", "local", "development", "environments"]},
    {"text": "Project uses PostgreSQL version 14 with connection pooling enabled", "expected_entities": ["project", "postgresql", "version", "connection", "pooling"]},
    {"text": "Backend engineer role requires Python asyncio and FastAPI experience", "expected_entities": ["backend", "engineer", "python", "asyncio", "fastapi", "experience"]},
]

# Relationship test data
RELATIONSHIP_FIXTURES = [
    {"entities": ["docker", "development"], "expected_relationship": "docker__development"},
    {"entities": ["postgresql", "connection"], "expected_relationship": "postgresql__connection"},
]

# Action-effectiveness test data
ACTION_EFFECTIVENESS_FIXTURES = [
    {"context": "coding", "action": "search_memory", "collection": "patterns", "outcomes": ["worked"]*9 + ["failed"], "expected_rate": 0.9},
    {"context": "fitness", "action": "search_memory", "collection": "memory_bank", "outcomes": ["worked"]*8 + ["failed"]*2, "expected_rate": 0.8},
    {"context": "coding", "action": "create_memory", "collection": "working", "outcomes": ["failed"]*8 + ["worked"]*2, "expected_rate": 0.2},
]

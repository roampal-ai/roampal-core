import logging
from typing import List, Dict, Any, Optional
import chromadb
import sys
import os
from chromadb.config import Settings as ChromaSettings

# BM25 for hybrid search (v2.1 Enhanced Retrieval)
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("BM25 not available (pip install rank-bm25)")

# Add the backend directory to sys.path if not already there
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from .embedding_service import EmbeddingService
import shutil
from pathlib import Path
import time
import os
# Simple collection naming for single user
def get_loopsmith_collection():
    return "loopsmith_memories"

def get_user_chromadb_collection(user_id: str, shard_id: str = "default") -> str:
    """Generate user-specific collection name for multi-user support"""
    return f"user_{user_id}_{shard_id}_memories"

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "loopsmith_memories"

class ChromaDBAdapter:
    """
    Concrete implementation of the VectorDBInterface using ChromaDB.
    Handles persistent local vector storage and retrieval for Roampal memories.
    Supports collection-specific persistence directories.
    """

    def __init__(
        self,
        persistence_directory: str = None,
        use_server: bool = None,  # None = auto-detect
        user_id: Optional[str] = None,
        # Backwards compatibility with old API
        collection_name: str = None,
        persist_directory: str = None
    ):
        # Handle old API: ChromaDBAdapter(collection_name=..., persist_directory=...)
        if persist_directory is not None:
            persistence_directory = persist_directory
        if persistence_directory is None:
            persistence_directory = "./chromadb"  # Default fallback

        self.db_path = str(persistence_directory)  # Keep for compatibility but not used in server mode

        # Auto-detect server mode: use local embedded mode for benchmarks/tests
        if use_server is None:
            use_server = os.environ.get("ROAMPAL_USE_SERVER", "").lower() == "true"
        self.use_server = use_server
        self.client = None
        self.collection: Optional[chromadb.Collection] = None
        self.collection_name: Optional[str] = collection_name  # Store for auto-init
        self._auto_init_lock = False
        self._current_path = None
        self.user_id = user_id  # Add user context
        self._pending_collection_name = collection_name  # For backwards compat auto-init

        # BM25 index for hybrid search (v2.1)
        self.bm25_index = None
        self.bm25_docs = []
        self.bm25_ids = []
        self.bm25_metadatas = []
        self._bm25_needs_rebuild = True

        # Only create local dirs if not using server
        if not self.use_server:
            os.makedirs(self.db_path, exist_ok=True)
            # Disabled automatic cleanup - use cleanup_chromadb.py utility instead
            # self._clean_old_folders()  # Can cause lock issues on Windows

    def _clean_old_folders(self):
        """Delete old UUID folders in the vector store directory with retry on lock."""
        # Cleanup debug removed
        logger.info(f"ChromaDB cleanup called for: {self.db_path}")
        
        for entry in os.scandir(self.db_path):
            if entry.is_dir() and len(entry.name) == 36 and entry.name.count('-') == 4:  # UUID pattern
                folder_path = Path(self.db_path) / entry.name
                
                for attempt in range(3):  # Retry 3 times
                    try:
                        shutil.rmtree(folder_path)
                        logger.info(f"Deleted old folder: {entry.name}")
                        break
                    except PermissionError as e:
                        if attempt < 2:  # Wait and retry
                            time.sleep(1)
                            continue
                        logger.warning(f"Failed to delete old folder {entry.name} after retries: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old folder {entry.name}: {e}")
                        break

    async def initialize(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        fragment_id: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        if self.client is None:
            if self.use_server:
                # Connect to ChromaDB server
                self.client = chromadb.HttpClient(
                    host="localhost",
                    port=8003,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
                logger.info(f"ChromaDB client connected to server at localhost:8003")
            else:
                # Use local embedded mode (for testing only)
                self.client = chromadb.PersistentClient(path=self.db_path)
                logger.info(f"ChromaDB client initialized for local path: {self.db_path}")

        # Use user-isolated collection if user_id provided
        if user_id:
            self.user_id = user_id
            # Create user-specific collection name
            shard_id = fragment_id or "roampal"
            collection_name = get_user_chromadb_collection(user_id, shard_id)
            logger.info(f"Using user-isolated collection: {collection_name}")

        # Use constructor's collection_name if set and parameter is default
        if self.collection_name and collection_name == DEFAULT_COLLECTION_NAME:
            collection_name = self.collection_name
        else:
            self.collection_name = collection_name

        # Use get_or_create to reuse existing collection
        # Don't use ChromaDB's default embedding function - Roampal provides embeddings manually
        # This prevents dimension mismatch (ChromaDB default is 384d, Roampal uses 768d)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None  # Manual embeddings from EmbeddingService
        )
        logger.info(f"ChromaDB collection '{collection_name}' ready (lazy loaded)")
        
        # No need to force file creation in server mode
        if not self.use_server and self.collection.count() == 0:
            logger.info(f"Empty collection '{collection_name}' initialized in embedded mode")

    async def _ensure_initialized(self):
        if self.collection is None and not self._auto_init_lock:
            self._auto_init_lock = True
            logger.warning("ChromaDBAdapter auto-initializing collection on demand (explicit .initialize() was not called).")
            await self.initialize(collection_name=self.collection_name or DEFAULT_COLLECTION_NAME)
            self._auto_init_lock = False

    async def upsert_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ):
        await self._ensure_initialized()
        if not (len(ids) == len(vectors) == len(metadatas)):
            error_msg = (
                f"Length of ids ({len(ids)}), "
                f"vectors ({len(vectors)}), and "
                f"metadatas ({len(metadatas)}) must be the same."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Upserting {len(ids)} vectors into collection '{self.collection.name}'...")
        try:
            # Extract documents from metadata for ChromaDB persistence
            documents = []
            for metadata in metadatas:
                # Use the content or text field as the document
                doc = metadata.get('content', metadata.get('text', metadata.get('original_text', '')))
                documents.append(str(doc))
            
            self.collection.upsert(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=documents  # ChromaDB needs documents to persist properly
            )

            # Mark BM25 index as needing rebuild (v2.1 hybrid search)
            self._bm25_needs_rebuild = True

            # ChromaDB now handles persistence automatically in both modes
            # The reconnection workaround has been removed as of 2024-09-17
            # Data is persisted on write with proper transaction handling

            logger.info(f"Successfully upserted {len(ids)} vectors.")
        except Exception as e:
            logger.error(f"Failed to upsert vectors into ChromaDB: {e}", exc_info=True)
            raise

    async def query_vectors(self, query_vector: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Query vectors with comprehensive error handling."""
        try:
            # Check if collection is empty first
            await self._ensure_initialized()

            # v0.2.4: Refresh collection to see changes from other processes (e.g., UI uploads)
            # ChromaDB's PersistentClient caches collection state; re-fetching syncs with disk
            if self.client and self.collection_name:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=None,  # Must match initialize() - prevents 384d/768d mismatch
                    metadata={"hnsw:space": "l2"}
                )

            if self.collection and self.collection.count() == 0:
                logger.debug(f"[ChromaDB] Collection '{self.collection_name}' is empty, returning empty results")
                return []

            # Validate query vector
            if not query_vector or not isinstance(query_vector, list):
                logger.warning(f"[ChromaDB] Invalid query vector: {type(query_vector)}")
                return []

            # Check for None values in query vector
            if any(v is None for v in query_vector):
                logger.warning("[ChromaDB] Query vector contains None values")
                return []

            # Ensure query vector is numeric
            try:
                query_vector = [float(v) for v in query_vector]
            except (ValueError, TypeError) as e:
                logger.warning(f"[ChromaDB] Failed to convert query vector to floats: {e}")
                return []

            logger.info(f"Querying for top {top_k} vectors in collection '{self.collection_name}'...")
            
            # Perform query with error handling
            try:
                results = self.collection.query(
                    query_embeddings=[query_vector],
                    n_results=top_k,
                    where=filters
                )
            except Exception as e:
                logger.error(f"[ChromaDB] Query failed: {e}")
                return []
            
            # Process results with comprehensive error handling
            processed_results = []
            
            try:
                # Extract data from results
                ids = results.get('ids', [[]])[0] if results.get('ids') else []
                embeddings = results.get('embeddings', [[]])[0] if results.get('embeddings') else []
                documents = results.get('documents', [[]])[0] if results.get('documents') else []
                metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
                distances = results.get('distances', [[]])[0] if results.get('distances') else []
                
                # Process each result
                for i in range(len(ids)):
                    try:
                        # Safe data extraction
                        result_id = ids[i] if i < len(ids) else f"unknown_{i}"
                        result_embedding = embeddings[i] if i < len(embeddings) else []
                        result_document = documents[i] if i < len(documents) else ""
                        result_metadata = metadatas[i] if i < len(metadatas) else {}
                        result_distance = distances[i] if i < len(distances) else 2.0
                        
                        # Validate embedding
                        if result_embedding is None:
                            logger.warning(f"[ChromaDB] Unexpected embeddings type: {type(result_embedding)}")
                            continue
                        
                        # Create safe result object
                        result = {
                            'id': str(result_id),
                            'text': str(result_document) if result_document else "",
                            'metadata': result_metadata if isinstance(result_metadata, dict) else {},
                            'distance': float(result_distance) if result_distance is not None else 2.0,
                            'embedding': result_embedding if isinstance(result_embedding, list) else []
                        }
                        
                        processed_results.append(result)
                        
                    except Exception as e:
                        logger.warning(f"[ChromaDB] Error processing result {i}: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"[ChromaDB] Error processing query results: {e}")
                return []
            
            logger.info(f"Query returned {len(processed_results)} results.")
            return processed_results
            
        except Exception as e:
            logger.error(f"[ChromaDB] Critical error in query_vectors: {e}")
            return []

    async def _build_bm25_index(self):
        """Build BM25 index from all documents (v2.1 Hybrid Search)"""
        if not BM25_AVAILABLE:
            return

        await self._ensure_initialized()

        if self.collection.count() == 0:
            logger.debug("[BM25] Collection empty, skipping index build")
            return

        try:
            # Get all documents
            all_data = self.collection.get(include=["documents", "metadatas"])
            self.bm25_ids = all_data.get("ids", [])
            self.bm25_docs = all_data.get("documents", [])
            self.bm25_metadatas = all_data.get("metadatas", [])

            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in self.bm25_docs]

            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_docs)
            self._bm25_needs_rebuild = False

            logger.debug(f"[BM25] Index built with {len(self.bm25_docs)} documents")
        except Exception as e:
            logger.warning(f"[BM25] Index build failed: {e}")
            self.bm25_index = None

    async def hybrid_query(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector (semantic) + BM25 (lexical) with RRF fusion.
        Based on industry best practices (Elastic, Weaviate, Microsoft Azure, 2025).

        Args:
            query_vector: Dense embedding for semantic search
            query_text: Text query for BM25 lexical search
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Fused results ranked by Reciprocal Rank Fusion (RRF)
        """
        # 1. Vector search (semantic similarity)
        vector_results = await self.query_vectors(query_vector, top_k=top_k*2, filters=filters)

        # If BM25 not available, fall back to pure vector search
        if not BM25_AVAILABLE or not self.bm25_index:
            return vector_results[:top_k]

        try:
            # 2. Rebuild BM25 index if needed
            if self._bm25_needs_rebuild:
                await self._build_bm25_index()

            if not self.bm25_index:
                # BM25 build failed, use vector only
                return vector_results[:top_k]

            # 3. BM25 search (lexical matching)
            tokenized_query = query_text.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)

            # Get top BM25 results
            top_bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:top_k*2]

            bm25_results = []
            for idx in top_bm25_indices:
                if idx < len(self.bm25_ids):
                    bm25_results.append({
                        "id": self.bm25_ids[idx],
                        "text": self.bm25_docs[idx],
                        "metadata": self.bm25_metadatas[idx] if idx < len(self.bm25_metadatas) else {},
                        "bm25_score": float(bm25_scores[idx]),
                        "distance": max(0.0, 1.0 - (bm25_scores[idx] / 100.0))  # Normalize to distance
                    })

            # 4. Reciprocal Rank Fusion (RRF) with k=60 (research-backed constant)
            rrf_scores = {}

            # Add vector search rankings
            for rank, result in enumerate(vector_results):
                doc_id = result["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rank + 60)

            # Add BM25 rankings
            for rank, result in enumerate(bm25_results):
                doc_id = result["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rank + 60)

            # 5. Merge results and add RRF scores
            merged = {}
            for r in vector_results + bm25_results:
                doc_id = r["id"]
                if doc_id not in merged:
                    merged[doc_id] = r
                    merged[doc_id]["rrf_score"] = rrf_scores.get(doc_id, 0.0)
                    # Keep original distance, but add RRF for ranking

            # 6. Sort by RRF score and return top-k
            final_results = sorted(
                merged.values(),
                key=lambda x: x.get("rrf_score", 0.0),
                reverse=True
            )[:top_k]

            logger.debug(f"[HYBRID] Merged {len(vector_results)} vector + {len(bm25_results)} BM25 → {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.warning(f"[HYBRID] Hybrid search failed, falling back to vector only: {e}")
            return vector_results[:top_k]

    async def get_collection_count(self) -> int:
        """Get the total number of items in the collection"""
        await self._ensure_initialized()
        try:
            count = self.collection.count()
            logger.debug(f"Collection '{self.collection_name}' contains {count} items")
            return count
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0

    async def get_vectors_by_ids(
        self,
        ids: List[str]
    ) -> Dict[str, Any]:
        await self._ensure_initialized()
        try:
            result = self.collection.get(ids=ids, include=["documents", "embeddings", "metadatas"])
            return result
        except Exception as e:
            logger.error(f"Failed to get vectors by ids: {e}", exc_info=True)
            return {}

    def list_all_ids(self) -> List[str]:
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        try:
            result = self.collection.get(include=[])
            return result.get('ids', [])
        except Exception as e:
            # ChromaDB can throw "Error finding id" for ghost entries (IDs in index but no document)
            logger.warning(f"ChromaDB error listing all IDs: {e}")
            return []

    def delete_vectors(self, ids: List[str]):
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        self.collection.delete(ids=ids)
        # Explicitly persist after delete - ChromaDB 0.4.x may not auto-persist deletes
        if hasattr(self.client, 'persist'):
            self.client.persist()

    def get_all_vectors(self) -> List[Dict[str, Any]]:
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        results = self.collection.get(include=["embeddings", "metadatas"])
        ids = results.get("ids", [])
        vectors = results.get("embeddings", [])
        metadatas = results.get("metadatas", [])
        out = []
        for i in range(len(ids)):
            vector = None
            if isinstance(vectors, (list, tuple)) and len(vectors) > i:
                vector = vectors[i]
            elif hasattr(vectors, '__len__') and hasattr(vectors, '__getitem__') and len(vectors) > i:
                vector = vectors[i]
            metadata = metadatas[i] if isinstance(metadatas, (list, tuple)) and len(metadatas) > i else {}
            out.append({
                "id": ids[i],
                "vector": vector,
                "metadata": metadata,
            })
        return out

    def get_fragment(self, fragment_id: str) -> Optional[Dict[str, Any]]:
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized. Cannot get fragment.")
        try:
            result = self.collection.get(ids=[fragment_id], include=["embeddings", "metadatas", "documents"])
        except Exception as e:
            # ChromaDB can throw "Error finding id" for ghost entries (IDs in index but no document)
            # This happens when documents are deleted but index isn't fully cleaned
            logger.warning(f"ChromaDB error getting fragment {fragment_id}: {e}")
            return None
        if not result or not result.get("ids"):
            return None
        embeddings = result.get("embeddings", [])
        vector = None
        if isinstance(embeddings, (list, tuple)) and len(embeddings) > 0:
            vector = embeddings[0]
        elif hasattr(embeddings, '__len__') and hasattr(embeddings, '__getitem__') and len(embeddings) > 0:
            vector = embeddings[0]
        metadatas = result.get("metadatas", [])
        metadata = metadatas[0] if isinstance(metadatas, (list, tuple)) and len(metadatas) > 0 else {}
        documents = result.get("documents", [])
        content = documents[0] if isinstance(documents, (list, tuple)) and len(documents) > 0 else ""
        return {
            "id": result["ids"][0],
            "vector": vector,
            "metadata": metadata,
            "content": content,
        }

    def update_fragment_metadata(self, fragment_id: str, metadata_updates: Dict[str, Any]):
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        frag = self.get_fragment(fragment_id)
        if not frag:
            logger.warning(f"update_fragment_metadata: No fragment with id={fragment_id}")
            return
        if frag.get("vector") is None:
            logger.warning(
                f"Skipping metadata update for fragment {fragment_id} "
                "because it has no associated vector."
            )
            return
        metadata = frag.get("metadata", {}) or {}
        metadata.update(metadata_updates)
        self.collection.upsert(
            ids=[fragment_id],
            embeddings=[frag.get("vector")],
            metadatas=[metadata]
        )
        logger.info(f"Fragment {fragment_id} metadata updated with {metadata_updates}")

    def update_fragment_score(self, fragment_id: str, new_score: float):
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")
        frag = self.get_fragment(fragment_id)
        if not frag:
            logger.warning(f"update_fragment_score: No fragment with id={fragment_id}")
            return
        if frag.get("vector") is None:
            logger.warning(
                f"Skipping score update for fragment {fragment_id} "
                "because it has no associated vector."
            )
            return
        metadata = frag.get("metadata", {}) or {}
        metadata["composite_score"] = new_score
        self.collection.upsert(
            ids=[fragment_id],
            embeddings=[frag.get("vector")],
            metadatas=[metadata]
        )
        logger.info(f"Fragment {fragment_id} composite_score updated to {new_score}")

    async def update_metadata(self, doc_id: str, metadata: Dict[str, Any]):
        """Update metadata for existing document without re-embedding.

        Used by deduplication system to increment counters (e.g., mentioned_count)
        or update quality metrics without regenerating embeddings.
        """
        await self._ensure_initialized()
        if self.collection is None:
            raise RuntimeError("ChromaDB collection not initialized")

        try:
            # Get existing document to preserve embedding
            frag = self.get_fragment(doc_id)
            if not frag:
                logger.warning(f"update_metadata: No document with id={doc_id}")
                return

            if frag.get("vector") is None:
                logger.warning(
                    f"Skipping metadata update for {doc_id} "
                    "because it has no associated vector."
                )
                return

            # Update with new metadata while preserving embedding
            self.collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )
            logger.info(f"Metadata updated for document {doc_id}")

        except Exception as e:
            logger.error(f"Failed to update metadata for {doc_id}: {e}")
            raise

    async def cleanup(self):
        """Gracefully cleanup ChromaDB connections"""
        try:
            if self.collection:
                # Persist any pending writes
                if hasattr(self.collection, 'persist'):
                    self.collection.persist()
                self.collection = None

            if self.client:
                # Close the client connection
                if hasattr(self.client, 'close'):
                    self.client.close()
                self.client = None

            logger.info(f"ChromaDB adapter cleaned up for {self.collection_name}")
        except Exception as e:
            logger.warning(f"Error during ChromaDB cleanup: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self.client or self.collection:
                import asyncio
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()
        except Exception:
            pass  # Ignore errors in destructor

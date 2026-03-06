import logging
import os
from typing import List, Dict, Any, Optional
import chromadb
import sys
from chromadb.config import Settings as ChromaSettings


# Add the backend directory to sys.path if not already there
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from .embedding_service import EmbeddingService
import shutil
from pathlib import Path
import time
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
                    embedding_function=None  # Must match initialize() - prevents 384d/768d mismatch
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
                        
                        # v0.4.1.2: Skip phantom HNSW entries (deleted docs still in vector index)
                        if result_document is None and (result_metadata is None or not result_metadata):
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

    async def hybrid_query(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search by vector similarity.

        v0.4.1.2: BM25 hybrid search removed. It caused index-to-ID misalignment
        when ChromaDB had None documents (from TTL-deleted working memories),
        resulting in phantom results that broke memory injection.

        Args:
            query_vector: Dense embedding for semantic search
            query_text: Unused (kept for API compatibility)
            top_k: Number of results to return
            filters: Optional metadata filters
        """
        vector_results = await self.query_vectors(query_vector, top_k=top_k*2, filters=filters)
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
            logger.debug(f"ChromaDB ghost entry {fragment_id}: {e}")
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
        metadata = frag.get("metadata", {}) or {}
        metadata.update(metadata_updates)
        # v0.4.0: Use update() instead of upsert() — avoids re-sending the embedding
        self.collection.update(
            ids=[fragment_id],
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

    # v0.4.1: __del__ removed — it created a new asyncio event loop during GC,
    # which conflicts with the running loop and can leave SQLite WAL files locked.
    # Use explicit `await adapter.cleanup()` during lifespan shutdown instead.

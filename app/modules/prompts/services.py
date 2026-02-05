"""Business logic and database interactions for the prompts module."""

import re
import uuid
from typing import Any

import hdbscan
import numpy as np
from numpy.typing import NDArray
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer

from app.modules.prompts.schemas import DuplicateGroup, SimilarPrompt


class PromptService:
    """
    Singleton service for managing prompt embeddings, search, and deduplication.
    
    Uses sentence-transformers for embeddings and Milvus Lite for vector storage.
    """

    _instance: "PromptService | None" = None
    
    # Constants
    COLLECTION_NAME = "prompts"
    EMBEDDING_DIM = 384
    MODEL_NAME = "all-MiniLM-L6-v2"
    DB_PATH = "./data/prompts.db"
    VARIABLE_PATTERN = re.compile(r"\{\{[^}]+\}\}")
    TOKEN_VAR = "TOKEN_VAR"

    def __new__(cls) -> "PromptService":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes (actual init happens in initialize())."""
        if not hasattr(self, "_initialized"):
            self._initialized = False
        self._model: SentenceTransformer | None = None
        self._client: MilvusClient | None = None

    def initialize(self) -> None:
        """
        Initialize the service: load model and connect to Milvus.
        
        Should be called during application startup.
        """
        if self._initialized:
            return

        # Load the sentence transformer model
        self._model = SentenceTransformer(self.MODEL_NAME)

        # Connect to Milvus Lite
        self._client = MilvusClient(self.DB_PATH)

        # Create collection if it doesn't exist
        self._ensure_collection()
        
        self._initialized = True

    def _ensure_collection(self) -> None:
        """Create the prompts collection if it doesn't exist."""
        if self._client is None:
            raise RuntimeError("Milvus client not initialized")

        # Check if collection exists
        collections = self._client.list_collections()
        if self.COLLECTION_NAME in collections:
            return

        # Create collection with schema
        self._client.create_collection(
            collection_name=self.COLLECTION_NAME,
            dimension=self.EMBEDDING_DIM,
            metric_type="COSINE",
            id_type="string",
            max_length=36,  # UUID length
        )

    def close(self) -> None:
        """Close connections and cleanup resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._model = None
        self._initialized = False

    def mask_variables(self, text: str) -> str:
        """
        Replace template variables like {{name}} with a generic token.
        
        This ensures prompts with different variable names but identical
        structures are seen as similar/duplicates.
        
        Args:
            text: The prompt text to mask
            
        Returns:
            Text with all {{variable}} patterns replaced with TOKEN_VAR
        """
        return self.VARIABLE_PATTERN.sub(self.TOKEN_VAR, text)

    def _embed(self, texts: list[str]) -> NDArray[np.float32]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings with shape (len(texts), EMBEDDING_DIM)
        """
        if self._model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        embeddings: NDArray[np.float32] = self._model.encode(
            texts, 
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity
        )
        return embeddings

    def generate_embeddings(
        self,
        prompts: list[dict[str, Any]],
    ) -> list[str]:
        """
        Generate embeddings for prompts and store in Milvus.
        
        Args:
            prompts: List of prompt dicts with 'content' and optional 'metadata'
            
        Returns:
            List of generated prompt IDs
        """
        if self._client is None:
            raise RuntimeError("Milvus client not initialized")

        # Generate IDs for each prompt
        ids = [str(uuid.uuid4()) for _ in prompts]
        
        # Mask variables and generate embeddings
        contents = [p["content"] for p in prompts]
        masked_contents = [self.mask_variables(c) for c in contents]
        embeddings = self._embed(masked_contents)

        # Prepare data for upsert
        data = []
        for i, prompt in enumerate(prompts):
            data.append({
                "id": ids[i],
                "vector": embeddings[i].tolist(),
                "content": prompt["content"],
                "masked_content": masked_contents[i],
                "metadata": prompt.get("metadata") or {},
            })

        # Upsert to Milvus
        self._client.upsert(
            collection_name=self.COLLECTION_NAME,
            data=data,
        )

        return ids

    def find_similar(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[SimilarPrompt]:
        """
        Find prompts similar to the query.
        
        Args:
            query: The search query text
            top_k: Maximum number of results to return
            threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar prompts with scores
        """
        if self._client is None:
            raise RuntimeError("Milvus client not initialized")

        # Mask and embed the query
        masked_query = self.mask_variables(query)
        query_embedding = self._embed([masked_query])[0]

        # Search in Milvus
        results = self._client.search(
            collection_name=self.COLLECTION_NAME,
            data=[query_embedding.tolist()],
            limit=top_k,
            output_fields=["content", "metadata"],
        )

        # Convert results to SimilarPrompt objects
        similar_prompts: list[SimilarPrompt] = []
        for hits in results:
            for hit in hits:
                # Milvus returns distance; for cosine, similarity = 1 - distance
                # But with COSINE metric in Milvus Lite, it returns similarity directly
                score = float(hit.get("distance", 0))
                
                if score >= threshold:
                    similar_prompts.append(
                        SimilarPrompt(
                            id=hit["id"],
                            content=hit["entity"].get("content", ""),
                            score=score,
                            metadata=hit["entity"].get("metadata"),
                        )
                    )

        return similar_prompts

    def get_prompt_similar(
        self,
        prompt_id: str,
        top_k: int = 10,
    ) -> list[SimilarPrompt]:
        """
        Find prompts similar to an existing prompt by ID.
        
        Args:
            prompt_id: The ID of the prompt to find similar prompts for
            top_k: Maximum number of results to return (excluding the prompt itself)
            
        Returns:
            List of similar prompts with scores
        """
        if self._client is None:
            raise RuntimeError("Milvus client not initialized")

        # Fetch the prompt by ID
        results = self._client.get(
            collection_name=self.COLLECTION_NAME,
            ids=[prompt_id],
            output_fields=["content", "masked_content"],
        )

        if not results:
            return []

        # Get the masked content and search
        masked_content = results[0].get("masked_content", "")
        query_embedding = self._embed([masked_content])[0]

        # Search for similar, requesting one extra to account for self
        search_results = self._client.search(
            collection_name=self.COLLECTION_NAME,
            data=[query_embedding.tolist()],
            limit=top_k + 1,
            output_fields=["content", "metadata"],
        )

        # Filter out the original prompt
        similar_prompts: list[SimilarPrompt] = []
        for hits in search_results:
            for hit in hits:
                if hit["id"] != prompt_id:
                    similar_prompts.append(
                        SimilarPrompt(
                            id=hit["id"],
                            content=hit["entity"].get("content", ""),
                            score=float(hit.get("distance", 0)),
                            metadata=hit["entity"].get("metadata"),
                        )
                    )

        return similar_prompts[:top_k]

    def find_duplicates(
        self,
        threshold: float = 0.1,
    ) -> list[DuplicateGroup]:
        """
        Find duplicate prompts using sparse HDBSCAN clustering.
        
        Uses a sparse distance matrix approach:
        1. Fetch all prompt IDs
        2. For each prompt, query its top N neighbors
        3. Construct a sparse distance matrix
        4. Run HDBSCAN with metric='precomputed'
        5. Group results by cluster label
        
        Args:
            threshold: Sensitivity parameter mapped to cluster_selection_epsilon
            
        Returns:
            List of duplicate groups (clusters with 2+ prompts)
        """
        if self._client is None:
            raise RuntimeError("Milvus client not initialized")

        # Fetch all prompts
        # Note: For large datasets, this should be paginated
        all_prompts = self._client.query(
            collection_name=self.COLLECTION_NAME,
            filter="",
            output_fields=["id", "content", "metadata"],
            limit=10000,  # Reasonable limit for this use case
        )

        if len(all_prompts) < 2:
            return []

        # Build ID to index mapping
        prompt_ids = [p["id"] for p in all_prompts]
        id_to_idx = {pid: idx for idx, pid in enumerate(prompt_ids)}
        n = len(prompt_ids)

        # Build sparse distance matrix via neighbor queries
        neighbors_k = min(50, n)
        rows: list[int] = []
        cols: list[int] = []
        distances: list[float] = []

        for i, prompt in enumerate(all_prompts):
            # Get embedding for this prompt
            masked = self.mask_variables(prompt.get("content", ""))
            embedding = self._embed([masked])[0]

            # Query neighbors
            results = self._client.search(
                collection_name=self.COLLECTION_NAME,
                data=[embedding.tolist()],
                limit=neighbors_k,
            )

            for hits in results:
                for hit in hits:
                    j = id_to_idx.get(hit["id"])
                    if j is not None and i != j:
                        # Convert similarity to distance (1 - similarity)
                        similarity = float(hit.get("distance", 0))
                        distance = 1.0 - similarity
                        
                        rows.append(i)
                        cols.append(j)
                        distances.append(distance)

        # Create sparse distance matrix
        sparse_dist = csr_matrix(
            (distances, (rows, cols)),
            shape=(n, n),
        )

        # Make symmetric (take minimum of symmetric entries)
        sparse_dist = sparse_dist.minimum(sparse_dist.T)

        # Convert to dense for HDBSCAN (required for precomputed metric)
        # For very large datasets, consider using approximate methods
        dense_dist = sparse_dist.toarray()
        
        # Fill diagonal with 0 and missing entries with large distance
        np.fill_diagonal(dense_dist, 0)
        # Replace zeros (no edge) with max distance for non-diagonal
        max_dist = 2.0  # Max possible for cosine distance
        dense_dist = np.where(
            (dense_dist == 0) & ~np.eye(n, dtype=bool),
            max_dist,
            dense_dist,
        )

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=2,
            min_samples=1,
            cluster_selection_epsilon=threshold,
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(dense_dist)

        # Group by cluster label
        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label >= 0:  # -1 means noise/no cluster
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)

        # Convert to DuplicateGroup objects
        duplicate_groups: list[DuplicateGroup] = []
        for cluster_id, indices in clusters.items():
            if len(indices) >= 2:  # Only return groups with actual duplicates
                prompts_in_cluster = []
                for idx in indices:
                    prompt = all_prompts[idx]
                    prompts_in_cluster.append(
                        SimilarPrompt(
                            id=prompt["id"],
                            content=prompt.get("content", ""),
                            score=1.0,  # Within same cluster
                            metadata=prompt.get("metadata"),
                        )
                    )
                
                duplicate_groups.append(
                    DuplicateGroup(
                        cluster_id=int(cluster_id),
                        prompts=prompts_in_cluster,
                        size=len(prompts_in_cluster),
                    )
                )

        # Sort by cluster size descending
        duplicate_groups.sort(key=lambda g: g.size, reverse=True)
        
        return duplicate_groups


# Global singleton instance
prompt_service = PromptService()

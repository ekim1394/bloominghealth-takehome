"""FastAPI router for the prompts module."""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.modules.prompts.schemas import (
    DuplicatesResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    SearchRequest,
    SearchResponse,
    SimilarPrompt,
)
from app.modules.prompts.services import PromptService, prompt_service


router = APIRouter()


def get_prompt_service() -> PromptService:
    """Dependency injection for PromptService."""
    if not prompt_service._initialized:
        prompt_service.initialize()
    return prompt_service


@router.post(
    "/embeddings/generate",
    response_model=EmbeddingsResponse,
    summary="Generate embeddings for prompts",
    description="Accept a list of prompts, generate embeddings, and store them in the vector database.",
)
async def generate_embeddings(
    request: EmbeddingsRequest,
    service: PromptService = Depends(get_prompt_service),
) -> EmbeddingsResponse:
    """
    Generate embeddings for a batch of prompts.
    
    The prompts will be processed with variable masking ({{var}} -> TOKEN_VAR)
    to ensure structural similarity detection. Original content is preserved
    in storage.
    """
    try:
        # Convert Pydantic models to dicts for service
        prompts_data = [
            {"content": p.content, "metadata": p.metadata}
            for p in request.prompts
        ]
        
        ids = service.generate_embeddings(prompts_data)
        
        return EmbeddingsResponse(
            ids=ids,
            count=len(ids),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings: {str(e)}",
        )


@router.post(
    "/search/semantic",
    response_model=SearchResponse,
    summary="Semantic search across prompts",
    description="Find prompts semantically similar to the query text.",
)
async def semantic_search(
    request: SearchRequest,
    service: PromptService = Depends(get_prompt_service),
) -> SearchResponse:
    """
    Perform semantic search across all stored prompts.
    
    The query will be processed with variable masking to match
    prompts with similar structure regardless of variable names.
    """
    try:
        results = service.find_similar(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
        )
        
        return SearchResponse(
            results=results,
            query=request.query,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )


@router.get(
    "/{prompt_id}/similar",
    response_model=list[SimilarPrompt],
    summary="Find similar prompts",
    description="Find prompts similar to a specific prompt by ID.",
)
async def get_similar_prompts(
    prompt_id: str,
    top_k: int = Query(default=10, ge=1, le=100, description="Number of results"),
    service: PromptService = Depends(get_prompt_service),
) -> list[SimilarPrompt]:
    """
    Find prompts similar to an existing prompt.
    
    Returns up to `top_k` prompts most similar to the specified prompt,
    excluding the prompt itself.
    """
    try:
        results = service.get_prompt_similar(
            prompt_id=prompt_id,
            top_k=top_k,
        )
        
        if not results and not service._client:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find similar prompts: {str(e)}",
        )


@router.get(
    "/analysis/duplicates",
    response_model=DuplicatesResponse,
    summary="Detect duplicate prompts",
    description="Identify clusters of duplicate or near-duplicate prompts using HDBSCAN.",
)
async def find_duplicates(
    threshold: float = Query(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sensitivity threshold (maps to HDBSCAN cluster_selection_epsilon). "
                    "Lower values = stricter matching, higher = more permissive.",
    ),
    service: PromptService = Depends(get_prompt_service),
) -> DuplicatesResponse:
    """
    Detect duplicate prompts using sparse HDBSCAN clustering.
    
    The algorithm:
    1. Builds a sparse similarity graph using k-NN queries
    2. Converts to a distance matrix
    3. Runs HDBSCAN with precomputed distances
    4. Groups prompts by cluster labels
    
    The threshold parameter controls clustering sensitivity via
    HDBSCAN's cluster_selection_epsilon.
    """
    try:
        groups = service.find_duplicates(threshold=threshold)
        
        total_duplicates = sum(g.size for g in groups)
        
        return DuplicatesResponse(
            groups=groups,
            total_duplicates=total_duplicates,
            total_groups=len(groups),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Duplicate detection failed: {str(e)}",
        )

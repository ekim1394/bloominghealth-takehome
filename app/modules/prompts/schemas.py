"""Pydantic schemas for the prompts module."""

from typing import Any

from pydantic import BaseModel, Field


class PromptInput(BaseModel):
    """Single prompt input for embedding generation."""

    content: str = Field(..., description="The prompt text content")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata for the prompt"
    )


class EmbeddingsRequest(BaseModel):
    """Request body for generating embeddings."""

    prompts: list[PromptInput] = Field(
        ..., min_length=1, description="List of prompts to embed"
    )


class EmbeddingsResponse(BaseModel):
    """Response from embedding generation."""

    ids: list[str] = Field(..., description="List of generated prompt IDs")
    count: int = Field(..., description="Number of prompts embedded")


class SearchRequest(BaseModel):
    """Request body for semantic search."""

    query: str = Field(..., description="The search query text")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    threshold: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity score threshold"
    )


class SimilarPrompt(BaseModel):
    """A similar prompt result."""

    id: str = Field(..., description="Prompt ID")
    content: str = Field(..., description="Original prompt content")
    score: float = Field(..., description="Similarity score (0-1, higher is more similar)")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Prompt metadata if available"
    )


class SearchResponse(BaseModel):
    """Response from semantic search."""

    results: list[SimilarPrompt] = Field(..., description="List of similar prompts")
    query: str = Field(..., description="The original search query")


class DuplicateGroup(BaseModel):
    """A group of duplicate prompts."""

    cluster_id: int = Field(..., description="Cluster identifier")
    prompts: list[SimilarPrompt] = Field(
        ..., description="List of prompts in this duplicate cluster"
    )
    size: int = Field(..., description="Number of prompts in the cluster")


class DuplicatesResponse(BaseModel):
    """Response from duplicate detection."""

    groups: list[DuplicateGroup] = Field(..., description="List of duplicate clusters")
    total_duplicates: int = Field(
        ..., description="Total number of prompts identified as duplicates"
    )
    total_groups: int = Field(..., description="Number of duplicate clusters found")

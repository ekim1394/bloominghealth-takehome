"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.modules.prediction.router import router as prediction_router
from app.modules.prompts.router import router as prompts_router
from app.modules.prompts.services import prompt_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Handle application startup and shutdown."""
    # Startup: Initialize services
    prompt_service.initialize()
    yield
    # Shutdown: Cleanup resources
    prompt_service.close()


app = FastAPI(
    title="Prompt Similarity & Deduplication Service",
    description="A service for managing prompt embeddings, semantic search, and duplicate detection",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(prompts_router, prefix="/prompts", tags=["prompts"])
app.include_router(prediction_router, prefix="/prediction", tags=["prediction"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

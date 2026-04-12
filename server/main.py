"""
FastAPI server for the Latent Space Explorer.

Exposes endpoints for embedding extraction, layer introspection,
attention visualization, and dimensionality reduction.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import config
from model import LatentModel
from schemas import (
    AttentionRequest,
    AttentionResponse,
    BatchEmbedRequest,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    LayersRequest,
    LayersResponse,
    ReduceRequest,
    ReduceResponse,
    ReductionMethod,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global model instance, initialised during the lifespan startup event.
latent_model = LatentModel(config.model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, release on shutdown."""
    latent_model.load()
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Latent Space Explorer",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the frontend dev server to talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _ensure_loaded() -> None:
    if not latent_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")


def _normalise_texts(text: str | list[str]) -> list[str]:
    """Accept a single string or a list and always return a list."""
    if isinstance(text, str):
        return [text]
    return text


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    """Embed one or more words/phrases using the last hidden state."""
    _ensure_loaded()
    texts = _normalise_texts(req.text)
    vectors = latent_model.embed(texts)
    return EmbedResponse(
        embeddings=vectors.tolist(),
        dim=vectors.shape[1],
    )


@app.post("/embed/batch", response_model=EmbedResponse)
async def embed_batch(req: BatchEmbedRequest) -> EmbedResponse:
    """Batch-embed a list of words/phrases."""
    _ensure_loaded()
    vectors = latent_model.embed(req.texts)
    return EmbedResponse(
        embeddings=vectors.tolist(),
        dim=vectors.shape[1],
    )


@app.post("/layers", response_model=LayersResponse)
async def layers(req: LayersRequest) -> LayersResponse:
    """Return hidden states from every layer for a single input."""
    _ensure_loaded()
    states = latent_model.hidden_states(req.text)  # (n_layers+1, hidden_dim)
    return LayersResponse(
        hidden_states=states.tolist(),
        n_layers=states.shape[0],
        dim=states.shape[1],
    )


@app.post("/attention", response_model=AttentionResponse)
async def attention(req: AttentionRequest) -> AttentionResponse:
    """Return attention weights from all layers for a single input."""
    _ensure_loaded()
    weights, tokens = latent_model.attention_weights(req.text)
    n_layers, n_heads, seq_len, _ = weights.shape
    return AttentionResponse(
        attention=weights.tolist(),
        tokens=tokens,
        n_layers=n_layers,
        n_heads=n_heads,
        seq_len=seq_len,
    )


@app.post("/reduce", response_model=ReduceResponse)
async def reduce(req: ReduceRequest) -> ReduceResponse:
    """Reduce high-dimensional vectors to 2D or 3D via UMAP or PCA."""
    vectors = np.array(req.vectors, dtype=np.float32)

    if vectors.shape[0] < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 vectors for dimensionality reduction.",
        )

    if req.method == ReductionMethod.umap:
        import umap

        reducer = umap.UMAP(
            n_components=req.n_components,
            n_neighbors=min(req.n_neighbors, vectors.shape[0] - 1),
            min_dist=req.min_dist,
            metric="cosine",
        )
        coords = reducer.fit_transform(vectors)
    else:
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=req.n_components)
        coords = reducer.fit_transform(vectors)

    return ReduceResponse(
        coordinates=coords.tolist(),
        method=req.method,
        n_components=req.n_components,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=config.model.model_name,
        model_loaded=latent_model.is_loaded,
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )

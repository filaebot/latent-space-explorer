"""
Pydantic request/response models for the latent space explorer API.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class ReductionMethod(str, Enum):
    umap = "umap"
    pca = "pca"


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    text: str | list[str] = Field(
        ...,
        description="A single word/phrase or a list of them to embed.",
    )


class EmbedResponse(BaseModel):
    embeddings: list[list[float]] = Field(
        ...,
        description="One embedding vector per input text. Shape: (n_texts, hidden_dim).",
    )
    dim: int = Field(..., description="Dimensionality of each embedding vector.")


# ---------------------------------------------------------------------------
# Batch embed (explicit batch endpoint)
# ---------------------------------------------------------------------------

class BatchEmbedRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        description="List of words/phrases to embed.",
        min_length=1,
    )


# Reuses EmbedResponse


# ---------------------------------------------------------------------------
# Layers (hidden states from every layer)
# ---------------------------------------------------------------------------

class LayersRequest(BaseModel):
    text: str = Field(..., description="Word or phrase to analyze.")


class LayersResponse(BaseModel):
    # Outer list = layers (0 = embedding layer, 1..N = transformer layers)
    # Inner list = the hidden-state vector for the last token at that layer
    hidden_states: list[list[float]] = Field(
        ...,
        description="Hidden state of the last token at each layer. Shape: (n_layers, hidden_dim).",
    )
    n_layers: int
    dim: int


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class AttentionRequest(BaseModel):
    text: str = Field(..., description="Word or phrase to analyze.")


class AttentionResponse(BaseModel):
    # Shape per layer: (n_heads, seq_len, seq_len)
    # We send them as nested lists so the frontend can pick heads/layers.
    attention: list[list[list[list[float]]]] = Field(
        ...,
        description=(
            "Attention weights for every layer. "
            "Shape: (n_layers, n_heads, seq_len, seq_len)."
        ),
    )
    tokens: list[str] = Field(
        ...,
        description="The tokenized input (human-readable token strings).",
    )
    n_layers: int
    n_heads: int
    seq_len: int


# ---------------------------------------------------------------------------
# Reduce (dimensionality reduction)
# ---------------------------------------------------------------------------

class ReduceRequest(BaseModel):
    vectors: list[list[float]] = Field(
        ...,
        description="High-dimensional vectors to reduce.",
        min_length=2,
    )
    method: ReductionMethod = Field(
        ReductionMethod.umap,
        description="Reduction algorithm to use.",
    )
    n_components: int = Field(
        2,
        ge=2,
        le=3,
        description="Target dimensionality (2 or 3).",
    )
    # UMAP-specific
    n_neighbors: int = Field(15, ge=2, description="UMAP n_neighbors.")
    min_dist: float = Field(0.1, ge=0.0, description="UMAP min_dist.")


class ReduceResponse(BaseModel):
    coordinates: list[list[float]] = Field(
        ...,
        description="Reduced coordinates. Shape: (n_vectors, n_components).",
    )
    method: ReductionMethod
    n_components: int


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    model_name: str
    model_loaded: bool

"""
Configuration for the latent space explorer server.

All settings can be overridden via environment variables prefixed with LSE_.
"""

import os
from dataclasses import dataclass, field

import torch

# Detect hardware capabilities once at import time
_has_cuda = torch.cuda.is_available()
_has_mps = torch.backends.mps.is_available()


def _default_quantize() -> str:
    """Default to 4-bit quantization only when CUDA is available."""
    return "true" if _has_cuda else "false"


def _default_device_map() -> str:
    """Pick a sensible device map based on available hardware."""
    if _has_cuda:
        return "auto"
    if _has_mps:
        return "mps"
    return "cpu"


@dataclass
class ModelConfig:
    # Which model to load. Supported:
    #   - "Qwen/Qwen3.5-9B" (causal LM, last-hidden-state embeddings)
    #   - "Qwen/Qwen3-Embedding-8B" (dedicated embedding model)
    model_name: str = os.getenv("LSE_MODEL_NAME", "Qwen/Qwen3.5-9B")

    # Where to cache downloaded model weights
    cache_dir: str = os.getenv("LSE_CACHE_DIR", os.path.expanduser("~/.cache/huggingface"))

    # Device map passed to from_pretrained. "auto" uses accelerate on CUDA,
    # "mps" for Apple Silicon, "cpu" as fallback.
    device_map: str = os.getenv("LSE_DEVICE_MAP", _default_device_map())

    # Enable 4-bit quantization via bitsandbytes (CUDA only).
    quantize_4bit: bool = os.getenv("LSE_QUANTIZE_4BIT", _default_quantize()).lower() == "true"

    # torch dtype string: "float16", "bfloat16", or "float32"
    dtype: str = os.getenv("LSE_DTYPE", "bfloat16")

    # Maximum sequence length for tokenization
    max_length: int = int(os.getenv("LSE_MAX_LENGTH", "512"))


@dataclass
class ServerConfig:
    host: str = os.getenv("LSE_HOST", "0.0.0.0")
    port: int = int(os.getenv("LSE_PORT", "8042"))
    reload: bool = os.getenv("LSE_RELOAD", "false").lower() == "true"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


config = Config()

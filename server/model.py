"""
Model loading and inference for the latent space explorer.

Handles:
  - Loading Qwen models via transformers (with optional 4-bit quantization)
  - Sequence-level embedding extraction (last token hidden state)
  - Layer-by-layer hidden state extraction
  - Attention weight extraction
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig

_has_cuda = torch.cuda.is_available()

# bitsandbytes is CUDA-only; import conditionally
BitsAndBytesConfig = None
if _has_cuda:
    try:
        from transformers import BitsAndBytesConfig as _BnBConfig
        BitsAndBytesConfig = _BnBConfig
    except ImportError:
        pass

logger = logging.getLogger(__name__)


class LatentModel:
    """Wraps a Qwen causal-LM for embedding and introspection."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model and tokenizer into memory."""
        logger.info("Loading model %s ...", self.cfg.model_name)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)

        # Optional 4-bit quantization via bitsandbytes (CUDA only)
        quantization_config = None
        if self.cfg.quantize_4bit and BitsAndBytesConfig is not None:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            logger.info("4-bit quantization enabled (nf4, double quant)")
        elif self.cfg.quantize_4bit:
            logger.warning(
                "4-bit quantization requested but bitsandbytes is not available "
                "(requires CUDA). Loading without quantization."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            cache_dir=self.cfg.cache_dir,
            trust_remote_code=True,
        )
        # Ensure pad token is set (Qwen models sometimes lack one)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_mps = self.cfg.device_map == "mps"

        # On MPS (Apple Silicon), load to CPU first then move to MPS.
        # device_map="auto" requires accelerate and is designed for CUDA.
        if use_mps:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_name,
                cache_dir=self.cfg.cache_dir,
                dtype=torch_dtype,
                trust_remote_code=True,
            )
            self.model = self.model.to("mps")
            logger.info("Model moved to MPS (Apple Silicon)")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_name,
                cache_dir=self.cfg.cache_dir,
                device_map=self.cfg.device_map,
                dtype=torch_dtype,
                quantization_config=quantization_config,
                trust_remote_code=True,
            )

        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully on %s", self.cfg.device_map)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self.model.config.num_attention_heads

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def _tokenize(self, texts: list[str]) -> dict:
        """Tokenize a batch of texts, returning tensors on the model device."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in encoded.items()}

    def _last_token_indices(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the index of the last real (non-padding) token per sequence."""
        # attention_mask shape: (batch, seq_len)
        # Sum along seq dimension and subtract 1 to get last valid index
        return attention_mask.sum(dim=1) - 1

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Compute sequence embeddings from the last token's final hidden state.

        Returns: np.ndarray of shape (len(texts), hidden_dim)
        """
        inputs = self._tokenize(texts)
        outputs = self.model(**inputs, output_hidden_states=True)

        # Last layer hidden states: (batch, seq_len, hidden_dim)
        last_hidden = outputs.hidden_states[-1]

        # Extract the last real token's hidden state per sequence
        last_idx = self._last_token_indices(inputs["attention_mask"])
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        embeddings = last_hidden[batch_idx, last_idx]  # (batch, hidden_dim)

        return embeddings.float().cpu().numpy()

    # ------------------------------------------------------------------
    # Layer-by-layer hidden states
    # ------------------------------------------------------------------

    @torch.no_grad()
    def hidden_states(self, text: str) -> np.ndarray:
        """
        Extract the last token's hidden state from every layer.

        Returns: np.ndarray of shape (n_layers + 1, hidden_dim)
            Layer 0 = token embedding layer output
            Layers 1..N = transformer block outputs
        """
        inputs = self._tokenize([text])
        outputs = self.model(**inputs, output_hidden_states=True)

        last_idx = self._last_token_indices(inputs["attention_mask"])[0]

        # outputs.hidden_states is a tuple of (n_layers + 1) tensors,
        # each of shape (1, seq_len, hidden_dim)
        states = []
        for layer_hidden in outputs.hidden_states:
            vec = layer_hidden[0, last_idx]  # (hidden_dim,)
            states.append(vec.float().cpu().numpy())

        return np.stack(states)  # (n_layers + 1, hidden_dim)

    # ------------------------------------------------------------------
    # Attention weights
    # ------------------------------------------------------------------

    @torch.no_grad()
    def attention_weights(self, text: str) -> tuple[np.ndarray, list[str]]:
        """
        Extract attention weights from all layers.

        Returns:
            weights: np.ndarray of shape (n_layers, n_heads, seq_len, seq_len)
            tokens: list of human-readable token strings
        """
        inputs = self._tokenize([text])
        outputs = self.model(**inputs, output_attentions=True)

        # outputs.attentions is a tuple of n_layers tensors,
        # each of shape (1, n_heads, seq_len, seq_len)
        attn_layers = []
        for layer_attn in outputs.attentions:
            attn_layers.append(layer_attn[0].float().cpu().numpy())

        weights = np.stack(attn_layers)  # (n_layers, n_heads, seq_len, seq_len)

        # Decode individual tokens for display
        token_ids = inputs["input_ids"][0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        return weights, tokens

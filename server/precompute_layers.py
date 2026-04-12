#!/usr/bin/env python3
"""
Pre-compute layer-by-layer hidden states and dimensionality reductions.

Fetches hidden states from every transformer layer for each word, then runs
UMAP and PCA locally to produce 3D coordinates per layer. Saves the result
as a static JSON file the frontend can load directly.

Usage:
    python server/precompute_layers.py [--server http://localhost:8042] [--output frontend/public/layers.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

DEFAULT_SERVER = "http://localhost:8042"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "frontend" / "public" / "layers.json"


def api_get(server: str, path: str) -> dict:
    url = f"{server}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def api_post(server: str, path: str, body: dict) -> dict:
    url = f"{server}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def round_coords(coords: np.ndarray, decimals: int = 4) -> list[list[float]]:
    """Round an (N, 3) array to the given precision and return as nested lists."""
    return np.round(coords, decimals).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute layer-by-layer analysis for the latent space explorer.")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="API server URL")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON file path")
    args = parser.parse_args()

    server: str = args.server.rstrip("/")
    output_path = Path(args.output)

    # Check server health
    print(f"Checking server at {server}...")
    health = api_get(server, "/api/health")
    if not health.get("model_loaded"):
        print("ERROR: Model is not loaded yet. Wait for it to finish loading.", file=sys.stderr)
        sys.exit(1)
    model_name = health["model_name"]
    print(f"Model: {model_name} (loaded)")

    # Fetch word list
    print("Fetching word list...")
    words_data = api_get(server, "/api/words")
    words: list[str] = []
    categories: dict[str, str] = {}
    for cat in words_data["categories"]:
        for item in cat["items"]:
            words.append(item)
            categories[item] = cat["name"]
    print(f"  {len(words)} words in {len(set(categories.values()))} categories")

    # Fetch hidden states for every word, one at a time
    print("Fetching hidden states from all layers...")
    num_layers = None
    hidden_dim = None
    # all_hidden[layer_idx] will hold a list of vectors, one per word
    all_hidden: dict[int, list[list[float]]] = {}

    t0 = time.time()
    for i, word in enumerate(words):
        data = api_post(server, "/api/layers", {"text": word})

        if num_layers is None:
            # n_layers from API = total layers including embedding layer
            num_layers = data["n_layers"]
            hidden_dim = data["dim"]
            # Pre-initialize the dict with empty lists
            for layer_idx in range(num_layers):
                all_hidden[layer_idx] = []
            print(f"  {num_layers} layers total (embedding + {num_layers - 1} transformer), hidden_dim={hidden_dim}")

        hidden_states = data["hidden_states"]
        for layer_idx in range(num_layers):
            all_hidden[layer_idx].append(hidden_states[layer_idx])

        if (i + 1) % 50 == 0 or (i + 1) == len(words):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(words) - (i + 1)) / rate if rate > 0 else 0
            print(f"  {i + 1}/{len(words)} words ({rate:.1f} words/s, ~{remaining:.0f}s remaining)")

    print(f"Hidden states collected in {time.time() - t0:.1f}s")

    # Run UMAP and PCA for each layer
    print(f"\nReducing {num_layers} layers to 3D (UMAP + PCA)...")
    layers_result: dict[str, dict] = {}

    for layer_idx in range(num_layers):
        t1 = time.time()
        vectors = np.array(all_hidden[layer_idx], dtype=np.float32)

        # UMAP
        reducer = UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        umap_coords = reducer.fit_transform(vectors)

        # PCA
        pca = PCA(n_components=3)
        pca_coords = pca.fit_transform(vectors)

        layers_result[str(layer_idx)] = {
            "umap": round_coords(umap_coords),
            "pca": round_coords(pca_coords),
        }

        elapsed = time.time() - t1
        print(f"  Layer {layer_idx}/{num_layers - 1} done ({elapsed:.1f}s)")

    # Build output
    result = {
        "words": words,
        "categories": categories,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "layers": layers_result,
        "model": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

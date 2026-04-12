#!/usr/bin/env python3
"""
Pre-compute embeddings and dimensionality reductions for the word list.

Hits the running API server to get embeddings and reductions, then saves
everything as a static JSON file that the frontend can load directly.

Usage:
    python precompute_embeddings.py [--server http://localhost:8042] [--output ../frontend/public/embeddings.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SERVER = "http://localhost:8042"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "frontend" / "public" / "embeddings.json"
BATCH_SIZE = 50


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute embeddings for the latent space explorer.")
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

    # Batch embed all words
    print("Embedding words...")
    all_embeddings: list[list[float]] = []
    for i in range(0, len(words), BATCH_SIZE):
        batch = words[i : i + BATCH_SIZE]
        data = api_post(server, "/api/batch_embed", {"texts": batch})
        all_embeddings.extend(data["embeddings"])
        done = min(i + BATCH_SIZE, len(words))
        print(f"  {done}/{len(words)} embedded")

    dim = len(all_embeddings[0])
    print(f"  Embedding dim: {dim}")

    # Run UMAP reduction
    print("Running UMAP reduction (3D)...")
    t0 = time.time()
    umap_data = api_post(server, "/api/reduce", {
        "vectors": all_embeddings,
        "method": "umap",
        "n_components": 3,
        "n_neighbors": 15,
        "min_dist": 0.1,
    })
    print(f"  UMAP done in {time.time() - t0:.1f}s")

    # Run PCA reduction
    print("Running PCA reduction (3D)...")
    t0 = time.time()
    pca_data = api_post(server, "/api/reduce", {
        "vectors": all_embeddings,
        "method": "pca",
        "n_components": 3,
    })
    print(f"  PCA done in {time.time() - t0:.1f}s")

    # Build output
    result = {
        "words": words,
        "categories": categories,
        "raw_embeddings": all_embeddings,
        "reductions": {
            "umap": {
                "points": umap_data["coordinates"],
                "params": {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
            },
            "pca": {
                "points": pca_data["coordinates"],
                "params": {},
            },
        },
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

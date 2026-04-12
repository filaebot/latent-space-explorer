# Latent Space Explorer

A web-based tool for exploring embedding spaces and language model internals.

## Architecture

- **Backend**: Python FastAPI server wrapping HuggingFace transformers
- **Model**: Qwen 3.5 9B (with optional Qwen3-Embedding-8B comparison)
- **Frontend**: Interactive explorer UI with seed+radiate navigation, user-defined semantic axes, and 3D visualization

## Phases

### Phase 1: Embedding Explorer
- Embed words and phrases using Qwen 3.5 9B last hidden state
- Navigate the embedding space with seed+radiate and custom axes
- Compare with dedicated embedding model (Qwen3-Embedding-8B)

### Phase 2: Layer Activation Explorer
- Visualize how representations transform across model layers
- Attention pattern visualization
- Track concept evolution through the network

## Setup

```bash
# Backend
cd server
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
npm install
npm run dev
```

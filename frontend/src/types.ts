// ---------------------------------------------------------------------------
// API types (mirror server/schemas.py)
// ---------------------------------------------------------------------------

export type ReductionMethod = "umap" | "pca";

export interface EmbedRequest {
  text: string | string[];
}

export interface EmbedResponse {
  embeddings: number[][];
  dim: number;
}

export interface BatchEmbedRequest {
  texts: string[];
}

export interface ReduceRequest {
  vectors: number[][];
  method: ReductionMethod;
  n_components: 2 | 3;
  n_neighbors?: number;
  min_dist?: number;
}

export interface ReduceResponse {
  coordinates: number[][];
  method: ReductionMethod;
  n_components: number;
}

export interface LayersRequest {
  text: string;
}

export interface LayersResponse {
  hidden_states: number[][];
  n_layers: number;
  dim: number;
}

export interface AttentionRequest {
  text: string;
}

export interface AttentionResponse {
  attention: number[][][][];
  tokens: string[];
  n_layers: number;
  n_heads: number;
  seq_len: number;
}

export interface HealthResponse {
  status: string;
  model_name: string;
  model_loaded: boolean;
}

// ---------------------------------------------------------------------------
// App-level types
// ---------------------------------------------------------------------------

export interface WordPoint {
  word: string;
  category: string;
  /** High-dimensional embedding (optional, may not be loaded) */
  embedding?: number[];
  /** 3D coordinates for display */
  position: [number, number, number];
}

export interface DataSet {
  points: WordPoint[];
  categories: string[];
}

export interface SemanticAxis {
  poleA: string;
  poleB: string;
  /** Direction vector in embedding space (poleB - poleA, normalized) */
  direction?: number[];
}

export interface AppState {
  dataset: DataSet;
  seed: string | null;
  neighbors: WordPoint[];
  selectedPoint: WordPoint | null;
  searchQuery: string;
  reductionMethod: ReductionMethod;
  neighborCount: number;
  axes: (SemanticAxis | null)[];
  hiddenCategories: Set<string>;
  usingMockData: boolean;
  mode: "embeddings" | "layers";
  currentLayer: number;
  layerData: LayerData | null;
  isPlaying: boolean;
  playSpeed: number;
}

// ---------------------------------------------------------------------------
// Layer analysis data (loaded from layers.json)
// ---------------------------------------------------------------------------

export interface LayerReduction {
  umap: [number, number, number][];
  pca: [number, number, number][];
}

export interface LayerData {
  words: string[];
  categories: Record<string, string>;
  num_layers: number;
  hidden_dim: number;
  layers: Record<string, LayerReduction>;
  model: string;
  generated_at: string;
}

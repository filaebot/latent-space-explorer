import type {
  EmbedResponse,
  BatchEmbedRequest,
  ReduceRequest,
  ReduceResponse,
  LayersResponse,
  AttentionResponse,
  HealthResponse,
  ReductionMethod,
  WordPoint,
  DataSet,
  LayerData,
} from "./types";

const API_BASE = "/api";

let _backendAvailable: boolean | null = null;

// ---------------------------------------------------------------------------
// Generic fetch helper
// ---------------------------------------------------------------------------

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${path} ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function checkHealth(): Promise<HealthResponse | null> {
  try {
    const h = await get<HealthResponse>("/health");
    _backendAvailable = true;
    return h;
  } catch {
    _backendAvailable = false;
    return null;
  }
}

export function isBackendAvailable(): boolean {
  return _backendAvailable === true;
}

export async function embed(text: string | string[]): Promise<EmbedResponse> {
  return post<EmbedResponse>("/embed", { text });
}

export async function batchEmbed(texts: string[]): Promise<EmbedResponse> {
  return post<EmbedResponse>("/batch_embed", { texts } as BatchEmbedRequest);
}

export async function reduce(req: ReduceRequest): Promise<ReduceResponse> {
  return post<ReduceResponse>("/reduce", req);
}

export async function layers(text: string): Promise<LayersResponse> {
  return post<LayersResponse>("/layers", { text });
}

export async function attention(text: string): Promise<AttentionResponse> {
  return post<AttentionResponse>("/attention", { text });
}

// ---------------------------------------------------------------------------
// Word list
// ---------------------------------------------------------------------------

interface WordListCategory {
  name: string;
  description: string;
  items: string[];
}

interface WordListResponse {
  categories: WordListCategory[];
}

/** Fetch the curated word list from the server. */
export async function fetchWordList(): Promise<{ word: string; category: string }[] | null> {
  try {
    const data = await get<WordListResponse>("/words");
    const words: { word: string; category: string }[] = [];
    for (const cat of data.categories) {
      for (const item of cat.items) {
        words.push({ word: item, category: cat.name });
      }
    }
    return words;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Mock data generation
// ---------------------------------------------------------------------------

const MOCK_CATEGORIES: Record<string, string[]> = {
  animals: [
    "cat", "dog", "wolf", "lion", "tiger", "bear", "eagle", "hawk",
    "salmon", "whale", "dolphin", "horse", "deer", "fox", "owl",
    "shark", "rabbit", "snake", "frog", "elephant",
  ],
  colors: [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "gray", "crimson", "teal", "amber", "violet",
    "indigo", "scarlet", "cyan", "magenta", "gold", "silver",
  ],
  emotions: [
    "joy", "anger", "fear", "love", "hate", "hope", "grief", "pride",
    "shame", "guilt", "awe", "envy", "calm", "rage", "bliss",
    "dread", "trust", "doubt", "peace", "thrill",
  ],
  tools: [
    "hammer", "wrench", "drill", "saw", "pliers", "screwdriver",
    "chisel", "level", "clamp", "file", "anvil", "lathe", "vise",
    "ruler", "compass", "knife", "axe", "shovel", "rake", "hoe",
  ],
  food: [
    "bread", "cheese", "apple", "rice", "pasta", "salmon", "steak",
    "soup", "salad", "cake", "pie", "butter", "honey", "sugar",
    "pepper", "garlic", "onion", "tomato", "potato", "carrot",
  ],
};

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

/** Generate a gaussian-clustered mock embedding for each category. */
function generateMockPositions(
  words: { word: string; category: string }[],
): WordPoint[] {
  const rand = seededRandom(42);
  const categories = [...new Set(words.map((w) => w.category))];
  const centroids = new Map<string, [number, number, number]>();

  // Place category centroids in a spread-out arrangement
  categories.forEach((cat, i) => {
    const angle = (i / categories.length) * Math.PI * 2;
    const r = 4;
    centroids.set(cat, [
      Math.cos(angle) * r,
      Math.sin(angle) * r,
      (rand() - 0.5) * 3,
    ]);
  });

  return words.map(({ word, category }) => {
    const c = centroids.get(category)!;
    const spread = 1.2;
    return {
      word,
      category,
      position: [
        c[0] + (rand() - 0.5) * spread * 2,
        c[1] + (rand() - 0.5) * spread * 2,
        c[2] + (rand() - 0.5) * spread * 2,
      ] as [number, number, number],
    };
  });
}

export function generateMockDataset(): DataSet {
  const words: { word: string; category: string }[] = [];
  for (const [cat, ws] of Object.entries(MOCK_CATEGORIES)) {
    for (const w of ws) {
      words.push({ word: w, category: cat });
    }
  }
  const points = generateMockPositions(words);
  const categories = Object.keys(MOCK_CATEGORIES);
  return { points, categories };
}

/** Euclidean distance in 3D. */
function dist3(a: [number, number, number], b: [number, number, number]): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/** Find k nearest neighbors to a given point (by 3D position). */
export function findNeighbors(
  target: WordPoint,
  all: WordPoint[],
  k: number,
): { point: WordPoint; distance: number }[] {
  return all
    .filter((p) => p.word !== target.word)
    .map((p) => ({ point: p, distance: dist3(target.position, p.position) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k);
}

/** Cosine similarity between two vectors. */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

/** Find k nearest neighbors using cosine similarity on raw embeddings. */
export function findSemanticNeighbors(
  target: WordPoint,
  all: WordPoint[],
  k: number,
): { point: WordPoint; distance: number }[] {
  if (!target.embedding) {
    return findNeighbors(target, all, k);
  }

  return all
    .filter((p) => p.word !== target.word && p.embedding)
    .map((p) => ({
      point: p,
      distance: cosineSimilarity(target.embedding!, p.embedding!),
    }))
    .sort((a, b) => b.distance - a.distance) // higher similarity = closer
    .slice(0, k);
}

// ---------------------------------------------------------------------------
// Pre-computed layer data (static file)
// ---------------------------------------------------------------------------

let _layerData: LayerData | null = null;

export async function fetchLayerData(): Promise<LayerData | null> {
  if (_layerData) return _layerData;

  try {
    const res = await fetch("/layers.json");
    if (!res.ok) return null;

    const data = (await res.json()) as LayerData;

    if (!data.words || !data.layers || !data.num_layers) {
      console.warn("Layer data file is malformed");
      return null;
    }

    _layerData = data;
    return data;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Pre-computed embeddings (static file)
// ---------------------------------------------------------------------------

interface PrecomputedData {
  words: string[];
  categories: Record<string, string>;
  raw_embeddings: number[][];
  reductions: {
    umap: { points: number[][]; params: Record<string, unknown> };
    pca: { points: number[][]; params: Record<string, unknown> };
  };
  model: string;
  generated_at: string;
}

// Cache the static file so switching reductions is instant
let _precomputed: PrecomputedData | null = null;

async function fetchPrecomputed(
  onProgress?: ProgressCallback,
): Promise<PrecomputedData | null> {
  if (_precomputed) return _precomputed;

  try {
    onProgress?.(10, "Loading pre-computed embeddings...");
    const res = await fetch("/embeddings.json");
    if (!res.ok) return null;

    onProgress?.(50, "Parsing embeddings...");
    const data = (await res.json()) as PrecomputedData;

    // Basic validation
    if (!data.words || !data.reductions?.umap || !data.reductions?.pca) {
      console.warn("Pre-computed embeddings file is malformed");
      return null;
    }

    onProgress?.(90, "Ready");
    _precomputed = data;
    return data;
  } catch {
    return null;
  }
}

function precomputedToDataset(
  data: PrecomputedData,
  method: ReductionMethod,
): DataSet {
  const reduction = data.reductions[method];
  const points: WordPoint[] = data.words.map((word, i) => ({
    word,
    category: data.categories[word] ?? "unknown",
    embedding: data.raw_embeddings[i],
    position: reduction.points[i] as [number, number, number],
  }));
  const categories = [...new Set(Object.values(data.categories))];
  return { points, categories };
}

// ---------------------------------------------------------------------------
// Load dataset: try static file, then backend API, fall back to mock
// ---------------------------------------------------------------------------

export type ProgressCallback = (percent: number, message: string) => void;

const BATCH_SIZE = 50;

export async function loadDataset(
  method: ReductionMethod,
  onProgress?: ProgressCallback,
): Promise<{ dataset: DataSet; mock: boolean }> {
  // Try pre-computed static file first
  const precomputed = await fetchPrecomputed(onProgress);
  if (precomputed) {
    onProgress?.(100, "Done");
    return { dataset: precomputedToDataset(precomputed, method), mock: false };
  }

  // Fall back to live API
  const health = await checkHealth();

  if (health && health.model_loaded) {
    try {
      onProgress?.(0, "Fetching word list...");
      const fetched = await fetchWordList();
      let allWords: { word: string; category: string }[];

      if (fetched && fetched.length > 0) {
        allWords = fetched;
      } else {
        allWords = [];
        for (const [cat, ws] of Object.entries(MOCK_CATEGORIES)) {
          for (const w of ws) {
            allWords.push({ word: w, category: cat });
          }
        }
      }

      const totalWords = allWords.length;

      // Batch embed with progress
      const allEmbeddings: number[][] = [];
      for (let i = 0; i < totalWords; i += BATCH_SIZE) {
        const batch = allWords.slice(i, i + BATCH_SIZE);
        const batchTexts = batch.map((w) => w.word);
        const embRes = await batchEmbed(batchTexts);
        allEmbeddings.push(...embRes.embeddings);

        const done = Math.min(i + BATCH_SIZE, totalWords);
        const percent = (done / totalWords) * 90; // reserve 10% for reduction
        onProgress?.(percent, `Embedding words... ${done}/${totalWords}`);
      }

      // Reduce
      onProgress?.(90, "Reducing dimensions...");
      const redRes = await reduce({
        vectors: allEmbeddings,
        method,
        n_components: 3,
      });

      onProgress?.(100, "Done");

      const points: WordPoint[] = allWords.map((w, i) => ({
        word: w.word,
        category: w.category,
        embedding: allEmbeddings[i],
        position: redRes.coordinates[i] as [number, number, number],
      }));

      const categories = [...new Set(allWords.map((w) => w.category))];
      return { dataset: { points, categories }, mock: false };
    } catch (err) {
      console.warn("Backend load failed, falling back to mock data:", err);
    }
  }

  return { dataset: generateMockDataset(), mock: true };
}

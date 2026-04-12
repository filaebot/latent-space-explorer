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

// ---------------------------------------------------------------------------
// Load dataset: try backend, fall back to mock
// ---------------------------------------------------------------------------

export async function loadDataset(
  method: ReductionMethod,
): Promise<{ dataset: DataSet; mock: boolean }> {
  const health = await checkHealth();

  if (health && health.model_loaded) {
    try {
      // Collect all words from mock categories as our word list
      const allWords: { word: string; category: string }[] = [];
      for (const [cat, ws] of Object.entries(MOCK_CATEGORIES)) {
        for (const w of ws) {
          allWords.push({ word: w, category: cat });
        }
      }
      const texts = allWords.map((w) => w.word);

      const embRes = await batchEmbed(texts);
      const redRes = await reduce({
        vectors: embRes.embeddings,
        method,
        n_components: 3,
      });

      const points: WordPoint[] = allWords.map((w, i) => ({
        word: w.word,
        category: w.category,
        embedding: embRes.embeddings[i],
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

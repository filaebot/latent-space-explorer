import type { WordPoint } from "./types";
import type { Cluster } from "./clusters";
import { findBridgeWords } from "./clusters";

// ---------------------------------------------------------------------------
// Guided Tours
// ---------------------------------------------------------------------------

export interface TourStep {
  target: [number, number, number];
  highlightWords: string[];
  annotation: string;
  neighborSeed?: string; // if set, select this word as seed
}

export interface Tour {
  name: string;
  steps: TourStep[];
}

export interface TourState {
  tour: Tour | null;
  stepIndex: number;
  isPlaying: boolean;
  autoAdvanceDelay: number; // ms
}

/**
 * Generate category overview tour: fly to each category centroid.
 */
function categoryOverviewTour(
  categoryCentroids: Map<string, [number, number, number]>,
  points: WordPoint[],
): Tour {
  const steps: TourStep[] = [];

  for (const [cat, centroid] of categoryCentroids) {
    const catWords = points
      .filter((p) => p.category === cat)
      .slice(0, 5)
      .map((p) => p.word);

    steps.push({
      target: centroid,
      highlightWords: catWords,
      annotation: `Category: ${cat} (${points.filter((p) => p.category === cat).length} words)`,
    });
  }

  return { name: "Category Overview", steps };
}

/**
 * Generate polysemy bridge tour: visit words that sit between clusters.
 */
function polysemyBridgeTour(
  points: WordPoint[],
  clusters: Cluster[],
): Tour {
  const bridges = findBridgeWords(points, clusters, 6);
  const steps: TourStep[] = [];

  for (const bridge of bridges) {
    // Find which clusters this word is near
    const nearClusters = clusters
      .map((c) => ({
        label: c.label,
        dist: Math.sqrt(
          (bridge.position[0] - c.centroid[0]) ** 2 +
          (bridge.position[1] - c.centroid[1]) ** 2 +
          (bridge.position[2] - c.centroid[2]) ** 2,
        ),
      }))
      .sort((a, b) => a.dist - b.dist)
      .slice(0, 2);

    steps.push({
      target: bridge.position,
      highlightWords: [bridge.word],
      annotation: `"${bridge.word}" bridges ${nearClusters.map((c) => c.label).join(" and ")}`,
      neighborSeed: bridge.word,
    });
  }

  return { name: "Polysemy Bridges", steps };
}

/**
 * Generate semantic neighborhoods tour: pick interesting seed words.
 */
function neighborhoodsTour(
  points: WordPoint[],
  categories: string[],
): Tour {
  const steps: TourStep[] = [];

  // Pick one representative word per category
  for (const cat of categories.slice(0, 5)) {
    const catPoints = points.filter((p) => p.category === cat);
    if (catPoints.length === 0) continue;

    // Pick the point closest to the category centroid
    const cx = catPoints.reduce((s, p) => s + p.position[0], 0) / catPoints.length;
    const cy = catPoints.reduce((s, p) => s + p.position[1], 0) / catPoints.length;
    const cz = catPoints.reduce((s, p) => s + p.position[2], 0) / catPoints.length;

    let bestPoint = catPoints[0];
    let bestDist = Infinity;
    for (const p of catPoints) {
      const d = Math.sqrt(
        (p.position[0] - cx) ** 2 +
        (p.position[1] - cy) ** 2 +
        (p.position[2] - cz) ** 2,
      );
      if (d < bestDist) {
        bestDist = d;
        bestPoint = p;
      }
    }

    steps.push({
      target: bestPoint.position,
      highlightWords: [bestPoint.word],
      annotation: `Exploring "${bestPoint.word}" neighborhood (${cat})`,
      neighborSeed: bestPoint.word,
    });
  }

  return { name: "Semantic Neighborhoods", steps };
}

/**
 * Generate all available tours from the current dataset.
 */
export function generateTours(
  points: WordPoint[],
  categories: string[],
  clusters: Cluster[],
  categoryCentroids: Map<string, [number, number, number]>,
): Tour[] {
  const tours: Tour[] = [];

  tours.push(categoryOverviewTour(categoryCentroids, points));

  if (clusters.length >= 2) {
    tours.push(polysemyBridgeTour(points, clusters));
  }

  tours.push(neighborhoodsTour(points, categories));

  return tours;
}

export function createTourState(): TourState {
  return {
    tour: null,
    stepIndex: 0,
    isPlaying: false,
    autoAdvanceDelay: 3000,
  };
}

import type { WordPoint } from "./types";

// ---------------------------------------------------------------------------
// DBSCAN clustering on 3D positions
// ---------------------------------------------------------------------------

export interface Cluster {
  id: number;
  label: string;
  points: WordPoint[];
  centroid: [number, number, number];
  dominantCategory: string;
  categoryCounts: Map<string, number>;
}

function dist3(a: [number, number, number], b: [number, number, number]): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Simple DBSCAN implementation.
 * Returns an array of cluster labels (-1 = noise).
 */
function dbscan(
  positions: [number, number, number][],
  eps: number,
  minPts: number,
): number[] {
  const n = positions.length;
  const labels = new Array<number>(n).fill(-1);
  let clusterId = 0;

  function regionQuery(idx: number): number[] {
    const neighbors: number[] = [];
    const p = positions[idx];
    for (let i = 0; i < n; i++) {
      if (dist3(p, positions[i]) <= eps) {
        neighbors.push(i);
      }
    }
    return neighbors;
  }

  for (let i = 0; i < n; i++) {
    if (labels[i] !== -1) continue;

    const neighbors = regionQuery(i);
    if (neighbors.length < minPts) {
      // noise, stays -1
      continue;
    }

    labels[i] = clusterId;
    const queue = [...neighbors.filter((j) => j !== i)];
    const visited = new Set<number>([i]);

    while (queue.length > 0) {
      const j = queue.shift()!;
      if (visited.has(j)) continue;
      visited.add(j);

      if (labels[j] === -1) {
        labels[j] = clusterId; // was noise, now border point
      }
      if (labels[j] !== -1 && labels[j] !== clusterId) continue;

      labels[j] = clusterId;

      const jNeighbors = regionQuery(j);
      if (jNeighbors.length >= minPts) {
        for (const k of jNeighbors) {
          if (!visited.has(k)) queue.push(k);
        }
      }
    }

    clusterId++;
  }

  return labels;
}

function computeCentroid(points: WordPoint[]): [number, number, number] {
  const sum: [number, number, number] = [0, 0, 0];
  for (const p of points) {
    sum[0] += p.position[0];
    sum[1] += p.position[1];
    sum[2] += p.position[2];
  }
  const n = points.length;
  return [sum[0] / n, sum[1] / n, sum[2] / n];
}

/**
 * Run DBSCAN on the dataset and return cluster info.
 * Automatically tunes eps based on average nearest-neighbor distance.
 */
export function computeClusters(points: WordPoint[]): Cluster[] {
  if (points.length === 0) return [];

  const positions = points.map((p) => p.position);

  // Estimate eps: use ~2x the average distance to 5th nearest neighbor
  const sampleSize = Math.min(points.length, 50);
  const step = Math.max(1, Math.floor(points.length / sampleSize));
  let totalDist = 0;
  let count = 0;
  for (let i = 0; i < points.length; i += step) {
    const dists: number[] = [];
    for (let j = 0; j < points.length; j++) {
      if (i !== j) dists.push(dist3(positions[i], positions[j]));
    }
    dists.sort((a, b) => a - b);
    if (dists.length >= 5) {
      totalDist += dists[4];
      count++;
    }
  }
  const avgDist5 = count > 0 ? totalDist / count : 1.5;
  const eps = avgDist5 * 1.5;
  const minPts = 3;

  const labels = dbscan(positions, eps, minPts);

  // Group points by cluster
  const clusterMap = new Map<number, WordPoint[]>();
  for (let i = 0; i < points.length; i++) {
    const label = labels[i];
    if (label < 0) continue;
    if (!clusterMap.has(label)) clusterMap.set(label, []);
    clusterMap.get(label)!.push(points[i]);
  }

  const clusters: Cluster[] = [];
  for (const [id, clusterPoints] of clusterMap) {
    const categoryCounts = new Map<string, number>();
    for (const p of clusterPoints) {
      categoryCounts.set(p.category, (categoryCounts.get(p.category) ?? 0) + 1);
    }

    let dominantCategory = "";
    let maxCount = 0;
    for (const [cat, cnt] of categoryCounts) {
      if (cnt > maxCount) {
        maxCount = cnt;
        dominantCategory = cat;
      }
    }

    const centroid = computeCentroid(clusterPoints);
    const label = dominantCategory + (clusterMap.size > categoryCounts.size ? ` #${id + 1}` : "");

    clusters.push({
      id,
      label,
      points: clusterPoints,
      centroid,
      dominantCategory,
      categoryCounts,
    });
  }

  // Sort by size descending
  clusters.sort((a, b) => b.points.length - a.points.length);
  return clusters;
}

/**
 * Compute centroid for each category.
 */
export function computeCategoryCentroids(
  points: WordPoint[],
  categories: string[],
): Map<string, [number, number, number]> {
  const sums = new Map<string, { s: [number, number, number]; n: number }>();
  for (const cat of categories) {
    sums.set(cat, { s: [0, 0, 0], n: 0 });
  }
  for (const p of points) {
    const entry = sums.get(p.category);
    if (!entry) continue;
    entry.s[0] += p.position[0];
    entry.s[1] += p.position[1];
    entry.s[2] += p.position[2];
    entry.n++;
  }
  const result = new Map<string, [number, number, number]>();
  for (const [cat, { s, n }] of sums) {
    if (n > 0) {
      result.set(cat, [s[0] / n, s[1] / n, s[2] / n]);
    }
  }
  return result;
}

/**
 * Find "bridge" words that sit between clusters (close to multiple cluster centroids).
 */
export function findBridgeWords(
  points: WordPoint[],
  clusters: Cluster[],
  count: number,
): WordPoint[] {
  if (clusters.length < 2) return [];

  // Score each point by how close it is to its second-nearest cluster centroid
  const scored: { point: WordPoint; score: number }[] = [];
  for (const p of points) {
    const dists = clusters
      .map((c) => dist3(p.position, c.centroid))
      .sort((a, b) => a - b);
    if (dists.length >= 2) {
      // Lower ratio = more "between" clusters
      const ratio = dists[0] / (dists[1] + 0.001);
      scored.push({ point: p, score: ratio });
    }
  }

  // Words with ratio closest to 1.0 are most "between" clusters
  scored.sort((a, b) => Math.abs(1 - a.score) - Math.abs(1 - b.score));
  return scored.slice(0, count).map((s) => s.point);
}

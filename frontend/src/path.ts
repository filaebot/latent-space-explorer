import * as THREE from "three";
import type { SceneContext } from "./scene";
import type { WordPoint } from "./types";

// ---------------------------------------------------------------------------
// Path tracing: find and draw semantic paths between two words
// ---------------------------------------------------------------------------

export interface PathResult {
  words: WordPoint[];
  totalDistance: number;
}

export interface PathRenderer {
  drawPath(path: PathResult, colorFn: (cat: string) => number): void;
  clear(): void;
  dispose(): void;
}

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

/**
 * Find a greedy semantic path from start to end via nearest embedding neighbors.
 * At each step, pick the unvisited point most similar to the current point
 * that is also closer to the target than the current point.
 */
export function findSemanticPath(
  start: WordPoint,
  end: WordPoint,
  allPoints: WordPoint[],
  maxSteps: number = 20,
): PathResult | null {
  if (!start.embedding || !end.embedding) return null;
  if (start.word === end.word) return { words: [start], totalDistance: 0 };

  const path: WordPoint[] = [start];
  const visited = new Set<string>([start.word]);
  let current = start;
  let totalDist = 0;

  for (let step = 0; step < maxSteps; step++) {
    if (current.word === end.word) break;

    // Direct similarity to target
    const currentToEnd = cosineSimilarity(current.embedding!, end.embedding);

    let bestPoint: WordPoint | null = null;
    let bestSim = -Infinity;

    for (const p of allPoints) {
      if (visited.has(p.word) || !p.embedding) continue;

      const simToCurrent = cosineSimilarity(current.embedding!, p.embedding);
      const simToEnd = cosineSimilarity(p.embedding, end.embedding);

      // Prefer points that are both similar to current AND closer to end
      // Use a weighted score favoring progress toward the target
      const score = simToCurrent * 0.4 + simToEnd * 0.6;

      if (score > bestSim) {
        bestSim = score;
        bestPoint = p;
      }
    }

    if (!bestPoint) break;

    const sim = cosineSimilarity(current.embedding!, bestPoint.embedding!);
    totalDist += 1 - sim; // distance = 1 - similarity
    path.push(bestPoint);
    visited.add(bestPoint.word);
    current = bestPoint;

    // If we reached the end
    if (bestPoint.word === end.word) break;

    // If the best candidate is the end point's neighbor, add end
    const simBestToEnd = cosineSimilarity(bestPoint.embedding!, end.embedding);
    if (simBestToEnd > currentToEnd && !visited.has(end.word)) {
      // Check if end is now the best next step
      const directToEnd = cosineSimilarity(bestPoint.embedding!, end.embedding);
      let anyBetter = false;
      for (const p of allPoints) {
        if (visited.has(p.word) || !p.embedding || p.word === end.word) continue;
        const s = cosineSimilarity(bestPoint.embedding!, p.embedding) * 0.4 +
          cosineSimilarity(p.embedding, end.embedding) * 0.6;
        if (s > directToEnd * 0.4 + 1.0 * 0.6) {
          anyBetter = true;
          break;
        }
      }
      if (!anyBetter) {
        totalDist += 1 - directToEnd;
        path.push(end);
        break;
      }
    }
  }

  // If we didn't reach the end, force-add it
  if (path[path.length - 1].word !== end.word) {
    const lastSim = cosineSimilarity(
      path[path.length - 1].embedding!,
      end.embedding,
    );
    totalDist += 1 - lastSim;
    path.push(end);
  }

  return { words: path, totalDistance: totalDist };
}

/**
 * Renderer for drawing 3D path lines and labels.
 */
export function createPathRenderer(sceneCtx: SceneContext): PathRenderer {
  let group: THREE.Group | null = null;

  function clear(): void {
    if (group) {
      sceneCtx.scene.remove(group);
      group.traverse((child) => {
        if (child instanceof THREE.Line) {
          child.geometry.dispose();
          (child.material as THREE.Material).dispose();
        }
        if (child instanceof THREE.Sprite) {
          (child.material as THREE.SpriteMaterial).map?.dispose();
          (child.material as THREE.Material).dispose();
        }
      });
      group = null;
    }
  }

  function makeTextSprite(text: string, color: string): THREE.Sprite {
    const canvas = document.createElement("canvas");
    const size = 256;
    canvas.width = size;
    canvas.height = 64;
    const ctx2 = canvas.getContext("2d")!;
    ctx2.font = "bold 24px -apple-system, sans-serif";
    ctx2.fillStyle = color;
    ctx2.textAlign = "center";
    ctx2.textBaseline = "middle";
    ctx2.fillText(text, size / 2, 32);

    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    const mat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      opacity: 0.9,
      depthTest: false,
    });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(1.5, 0.4, 1);
    return sprite;
  }

  function drawPath(path: PathResult, colorFn: (cat: string) => number): void {
    clear();
    if (path.words.length < 2) return;

    group = new THREE.Group();

    // Draw line segments
    const linePoints = path.words.map(
      (w) => new THREE.Vector3(...w.position),
    );
    const lineGeom = new THREE.BufferGeometry().setFromPoints(linePoints);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.8,
      linewidth: 2,
    });
    group.add(new THREE.Line(lineGeom, lineMat));

    // Draw node markers and labels along the path
    for (let i = 0; i < path.words.length; i++) {
      const w = path.words[i];
      const hex = colorFn(w.category);
      const color = `#${hex.toString(16).padStart(6, "0")}`;

      // Small sphere at each node
      const sphereGeom = new THREE.SphereGeometry(0.08, 8, 8);
      const sphereMat = new THREE.MeshBasicMaterial({ color: hex });
      const sphere = new THREE.Mesh(sphereGeom, sphereMat);
      sphere.position.set(...w.position);
      group.add(sphere);

      // Text label offset above
      const sprite = makeTextSprite(w.word, color);
      sprite.position.set(
        w.position[0],
        w.position[1] + 0.35,
        w.position[2],
      );
      group.add(sprite);
    }

    sceneCtx.scene.add(group);
  }

  function dispose(): void {
    clear();
  }

  return { drawPath, clear, dispose };
}

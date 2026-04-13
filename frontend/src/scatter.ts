import * as THREE from "three";
import type { SceneContext } from "./scene";
import type { WordPoint } from "./types";

// ---------------------------------------------------------------------------
// Category color palette: amber/teal-inspired with high contrast on dark bg
// ---------------------------------------------------------------------------

const PALETTE = [
  0xf59e0b, // amber
  0x14b8a6, // teal
  0xf472b6, // pink
  0x60a5fa, // blue
  0xa78bfa, // violet
  0x34d399, // emerald
  0xfb923c, // orange
  0xe879f9, // fuchsia
  0x38bdf8, // sky
  0xfbbf24, // yellow
];

const DIM_OPACITY = 0.12;
const NORMAL_OPACITY = 0.85;

export interface ScatterPlot {
  /** Rebuild all points from a dataset. */
  setPoints(points: WordPoint[], categories: string[]): void;

  /** Update visibility per hidden categories and search query. */
  updateVisibility(
    hiddenCategories: Set<string>,
    searchQuery: string,
  ): void;

  /** Highlight a seed and its neighbors. Draw connecting lines. */
  highlightSeed(
    seed: WordPoint | null,
    neighbors: WordPoint[],
  ): void;

  /** Highlight a single selected point (for info panel). */
  selectPoint(point: WordPoint | null): void;

  /** Returns the intersected WordPoint on hover, or null. */
  raycast(event: MouseEvent, container: HTMLElement): WordPoint | null;

  /** Map category -> hex color. */
  categoryColor(category: string): number;

  /** Smoothly animate point positions to new targets over duration ms. */
  animateToPositions(
    targets: [number, number, number][],
    duration: number,
  ): Promise<void>;

  /** Set per-point opacity overrides (for focus mode). Values 0-1. Pass null to reset. */
  setOpacities(opacities: Float32Array | null): void;

  /** Update base point size (for adaptive sizing). */
  setPointSize(size: number): void;

  /** Get current points array. */
  getPoints(): WordPoint[];

  /** Dispose Three.js objects. */
  dispose(): void;
}

export function createScatterPlot(ctx: SceneContext): ScatterPlot {
  let pointsMesh: THREE.Points | null = null;
  let linesGroup: THREE.Group | null = null;
  let selectionRing: THREE.Mesh | null = null;
  let currentPoints: WordPoint[] = [];
  let categoryMap = new Map<string, number>();

  const geometry = new THREE.BufferGeometry();
  const material = new THREE.PointsMaterial({
    size: 0.18,
    vertexColors: true,
    transparent: true,
    opacity: NORMAL_OPACITY,
    sizeAttenuation: true,
    depthWrite: false,
  });

  function categoryColor(category: string): number {
    const idx = categoryMap.get(category) ?? 0;
    return PALETTE[idx % PALETTE.length];
  }

  function setPoints(points: WordPoint[], categories: string[]): void {
    currentPoints = points;
    categoryMap.clear();
    categories.forEach((c, i) => categoryMap.set(c, i));

    // Remove old mesh
    if (pointsMesh) {
      ctx.scene.remove(pointsMesh);
      pointsMesh.geometry.dispose();
    }

    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    const sizes = new Float32Array(points.length);

    const tmpColor = new THREE.Color();

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      positions[i * 3] = p.position[0];
      positions[i * 3 + 1] = p.position[1];
      positions[i * 3 + 2] = p.position[2];

      tmpColor.setHex(categoryColor(p.category));
      colors[i * 3] = tmpColor.r;
      colors[i * 3 + 1] = tmpColor.g;
      colors[i * 3 + 2] = tmpColor.b;

      sizes[i] = 0.18;
    }

    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

    pointsMesh = new THREE.Points(geometry, material);
    ctx.scene.add(pointsMesh);
  }

  function updateVisibility(
    hiddenCategories: Set<string>,
    searchQuery: string,
  ): void {
    if (!pointsMesh) return;

    const colors = geometry.getAttribute("color") as THREE.BufferAttribute;
    const tmpColor = new THREE.Color();
    const query = searchQuery.toLowerCase().trim();

    for (let i = 0; i < currentPoints.length; i++) {
      const p = currentPoints[i];
      const hidden = hiddenCategories.has(p.category);
      const matchesSearch = !query || p.word.toLowerCase().includes(query);
      const visible = !hidden && matchesSearch;

      tmpColor.setHex(categoryColor(p.category));

      if (!visible) {
        // Dim hidden points
        colors.setXYZ(i, tmpColor.r * DIM_OPACITY, tmpColor.g * DIM_OPACITY, tmpColor.b * DIM_OPACITY);
      } else if (query && matchesSearch) {
        // Brighten search matches
        colors.setXYZ(
          i,
          Math.min(tmpColor.r * 1.3, 1),
          Math.min(tmpColor.g * 1.3, 1),
          Math.min(tmpColor.b * 1.3, 1),
        );
      } else {
        colors.setXYZ(i, tmpColor.r, tmpColor.g, tmpColor.b);
      }
    }
    colors.needsUpdate = true;
  }

  function highlightSeed(
    seed: WordPoint | null,
    neighbors: WordPoint[],
  ): void {
    // Remove old lines
    if (linesGroup) {
      ctx.scene.remove(linesGroup);
      linesGroup.traverse((child) => {
        if (child instanceof THREE.Line) {
          child.geometry.dispose();
          (child.material as THREE.Material).dispose();
        }
      });
      linesGroup = null;
    }

    if (!seed || neighbors.length === 0) return;

    linesGroup = new THREE.Group();
    const seedPos = new THREE.Vector3(...seed.position);

    for (const nb of neighbors) {
      const nbPos = new THREE.Vector3(...nb.position);
      const lineGeom = new THREE.BufferGeometry().setFromPoints([seedPos, nbPos]);
      const lineMat = new THREE.LineBasicMaterial({
        color: 0xf59e0b,
        transparent: true,
        opacity: 0.35,
      });
      linesGroup.add(new THREE.Line(lineGeom, lineMat));
    }

    ctx.scene.add(linesGroup);
  }

  function selectPoint(point: WordPoint | null): void {
    if (selectionRing) {
      ctx.scene.remove(selectionRing);
      (selectionRing.material as THREE.Material).dispose();
      selectionRing.geometry.dispose();
      selectionRing = null;
    }

    if (!point) return;

    const ringGeom = new THREE.RingGeometry(0.2, 0.28, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9,
    });
    selectionRing = new THREE.Mesh(ringGeom, ringMat);
    selectionRing.position.set(...point.position);
    // Billboard: always face camera
    selectionRing.lookAt(ctx.camera.position);
    ctx.scene.add(selectionRing);
  }

  function raycast(event: MouseEvent, container: HTMLElement): WordPoint | null {
    if (!pointsMesh) return null;

    const rect = container.getBoundingClientRect();
    ctx.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    ctx.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    ctx.raycaster.setFromCamera(ctx.mouse, ctx.camera);
    const intersects = ctx.raycaster.intersectObject(pointsMesh);

    if (intersects.length > 0 && intersects[0].index !== undefined) {
      return currentPoints[intersects[0].index] ?? null;
    }
    return null;
  }

  let animationId: number | null = null;

  function animateToPositions(
    targets: [number, number, number][],
    duration: number,
  ): Promise<void> {
    // Cancel any in-progress animation
    if (animationId !== null) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }

    if (!pointsMesh || currentPoints.length === 0) return Promise.resolve();

    const posAttr = geometry.getAttribute("position") as THREE.BufferAttribute;
    const count = currentPoints.length;

    // Snapshot current positions as starting state
    const startPositions = new Float32Array(count * 3);
    for (let i = 0; i < count * 3; i++) {
      startPositions[i] = posAttr.array[i];
    }

    const startTime = performance.now();

    return new Promise<void>((resolve) => {
      function step() {
        const elapsed = performance.now() - startTime;
        const t = Math.min(elapsed / duration, 1);
        // Smooth ease-out
        const ease = 1 - (1 - t) * (1 - t);

        for (let i = 0; i < count; i++) {
          const target = targets[i];
          if (!target) continue;

          const x = startPositions[i * 3] + (target[0] - startPositions[i * 3]) * ease;
          const y = startPositions[i * 3 + 1] + (target[1] - startPositions[i * 3 + 1]) * ease;
          const z = startPositions[i * 3 + 2] + (target[2] - startPositions[i * 3 + 2]) * ease;

          posAttr.setXYZ(i, x, y, z);

          // Keep the logical positions in sync so raycasting works
          currentPoints[i].position = [x, y, z];
        }
        posAttr.needsUpdate = true;

        if (t < 1) {
          animationId = requestAnimationFrame(step);
        } else {
          animationId = null;
          resolve();
        }
      }
      animationId = requestAnimationFrame(step);
    });
  }

  function setOpacities(opacities: Float32Array | null): void {
    if (!pointsMesh) return;

    const colors = geometry.getAttribute("color") as THREE.BufferAttribute;
    const tmpColor = new THREE.Color();

    if (!opacities) {
      // Reset to normal colors
      for (let i = 0; i < currentPoints.length; i++) {
        tmpColor.setHex(categoryColor(currentPoints[i].category));
        colors.setXYZ(i, tmpColor.r, tmpColor.g, tmpColor.b);
      }
    } else {
      for (let i = 0; i < currentPoints.length; i++) {
        tmpColor.setHex(categoryColor(currentPoints[i].category));
        const a = opacities[i];
        colors.setXYZ(i, tmpColor.r * a, tmpColor.g * a, tmpColor.b * a);
      }
    }
    colors.needsUpdate = true;
  }

  function setPointSize(size: number): void {
    material.size = size;
  }

  function getPoints(): WordPoint[] {
    return currentPoints;
  }

  function dispose(): void {
    if (pointsMesh) {
      ctx.scene.remove(pointsMesh);
      geometry.dispose();
      material.dispose();
    }
    if (linesGroup) {
      ctx.scene.remove(linesGroup);
    }
    if (selectionRing) {
      ctx.scene.remove(selectionRing);
    }
  }

  return {
    setPoints,
    updateVisibility,
    highlightSeed,
    selectPoint,
    raycast,
    categoryColor,
    animateToPositions,
    setOpacities,
    setPointSize,
    getPoints,
    dispose,
  };
}

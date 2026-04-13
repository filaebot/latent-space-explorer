import type * as THREE from "three";
import type { WordPoint } from "./types";

// ---------------------------------------------------------------------------
// Minimap: 2D canvas overlay showing top-down view of all points
// ---------------------------------------------------------------------------

export interface Minimap {
  update(camera: THREE.PerspectiveCamera): void;
  setPoints(points: WordPoint[], colorFn: (cat: string) => number): void;
  dispose(): void;
}

const SIZE = 180;
const PADDING = 10;

export function createMinimap(
  onNavigate: (worldX: number, worldZ: number) => void,
): Minimap {
  const canvas = document.createElement("canvas");
  canvas.id = "minimap-canvas";
  canvas.width = SIZE;
  canvas.height = SIZE;
  canvas.style.cssText = `
    position: fixed;
    bottom: 44px;
    left: 12px;
    z-index: 10;
    width: ${SIZE}px;
    height: ${SIZE}px;
    border: 1px solid #2a2a3a;
    border-radius: 6px;
    background: rgba(17, 17, 24, 0.85);
    cursor: crosshair;
  `;
  document.body.appendChild(canvas);

  const ctx = canvas.getContext("2d")!;

  let currentPoints: WordPoint[] = [];
  let colorFn: (cat: string) => number = () => 0xffffff;
  let minX = -10, maxX = 10, minZ = -10, maxZ = 10;

  function worldToMinimap(wx: number, wz: number): [number, number] {
    const rangeX = maxX - minX || 1;
    const rangeZ = maxZ - minZ || 1;
    const mx = PADDING + ((wx - minX) / rangeX) * (SIZE - 2 * PADDING);
    const my = PADDING + ((wz - minZ) / rangeZ) * (SIZE - 2 * PADDING);
    return [mx, my];
  }

  function minimapToWorld(mx: number, my: number): [number, number] {
    const rangeX = maxX - minX || 1;
    const rangeZ = maxZ - minZ || 1;
    const wx = minX + ((mx - PADDING) / (SIZE - 2 * PADDING)) * rangeX;
    const wz = minZ + ((my - PADDING) / (SIZE - 2 * PADDING)) * rangeZ;
    return [wx, wz];
  }

  function setPoints(points: WordPoint[], cf: (cat: string) => number): void {
    currentPoints = points;
    colorFn = cf;

    if (points.length === 0) return;
    minX = Infinity; maxX = -Infinity;
    minZ = Infinity; maxZ = -Infinity;
    for (const p of points) {
      if (p.position[0] < minX) minX = p.position[0];
      if (p.position[0] > maxX) maxX = p.position[0];
      if (p.position[2] < minZ) minZ = p.position[2];
      if (p.position[2] > maxZ) maxZ = p.position[2];
    }
    const marginX = (maxX - minX) * 0.1 || 1;
    const marginZ = (maxZ - minZ) * 0.1 || 1;
    minX -= marginX; maxX += marginX;
    minZ -= marginZ; maxZ += marginZ;
  }

  function hexToRgb(hex: number): string {
    const r = (hex >> 16) & 0xff;
    const g = (hex >> 8) & 0xff;
    const b = hex & 0xff;
    return `rgb(${r},${g},${b})`;
  }

  function update(camera: THREE.PerspectiveCamera): void {
    ctx.clearRect(0, 0, SIZE, SIZE);

    // Draw points
    for (const p of currentPoints) {
      const [mx, my] = worldToMinimap(p.position[0], p.position[2]);
      const color = colorFn(p.category);
      ctx.fillStyle = hexToRgb(color);
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      ctx.arc(mx, my, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw camera position indicator
    const [cx, cy] = worldToMinimap(camera.position.x, camera.position.z);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.stroke();

    // Camera look direction from the model-view matrix
    // The camera looks along -Z in local space; extract forward from world matrix
    const e = camera.matrixWorld.elements;
    // Forward direction is the negated third column of the world matrix
    const fwdX = -e[8];
    const fwdZ = -e[10];
    const fwdLen = Math.sqrt(fwdX * fwdX + fwdZ * fwdZ) || 1;
    const dirX = (fwdX / fwdLen) * 14;
    const dirZ = (fwdZ / fwdLen) * 14;

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + dirX, cy + dirZ);
    ctx.stroke();

    ctx.globalAlpha = 1;
  }

  // Click to navigate
  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (SIZE / rect.width);
    const my = (e.clientY - rect.top) * (SIZE / rect.height);
    const [wx, wz] = minimapToWorld(mx, my);
    onNavigate(wx, wz);
  });

  function dispose(): void {
    canvas.remove();
  }

  return { update, setPoints, dispose };
}

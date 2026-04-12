import type { WordPoint } from "./types";

// ---------------------------------------------------------------------------
// DOM references (cached on init)
// ---------------------------------------------------------------------------

let seedInput: HTMLInputElement;
let seedGo: HTMLButtonElement;
let searchInput: HTMLInputElement;
let reductionSelect: HTMLSelectElement;
let neighborCount: HTMLInputElement;
let categoryList: HTMLElement;
let infoContent: HTMLElement;
let statusText: HTMLElement;
let pointCountEl: HTMLElement;
let hoverLabel: HTMLElement;

// ---------------------------------------------------------------------------
// Callbacks set by main.ts
// ---------------------------------------------------------------------------

type SeedCallback = (word: string) => void;
type SearchCallback = (query: string) => void;
type ReductionCallback = (method: "umap" | "pca") => void;
type NeighborCountCallback = (n: number) => void;
type CategoryToggleCallback = (category: string, visible: boolean) => void;
type AxisCallback = (axisIndex: number, poleA: string, poleB: string) => void;
type AxesResetCallback = () => void;

let onSeed: SeedCallback = () => {};
let onSearch: SearchCallback = () => {};
let onReduction: ReductionCallback = () => {};
let onNeighborCount: NeighborCountCallback = () => {};
let onCategoryToggle: CategoryToggleCallback = () => {};
let onAxisSet: AxisCallback = () => {};
let onAxesReset: AxesResetCallback = () => {};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

export function initUI(callbacks: {
  onSeed: SeedCallback;
  onSearch: SearchCallback;
  onReduction: ReductionCallback;
  onNeighborCount: NeighborCountCallback;
  onCategoryToggle: CategoryToggleCallback;
  onAxisSet: AxisCallback;
  onAxesReset: AxesResetCallback;
}): void {
  onSeed = callbacks.onSeed;
  onSearch = callbacks.onSearch;
  onReduction = callbacks.onReduction;
  onNeighborCount = callbacks.onNeighborCount;
  onCategoryToggle = callbacks.onCategoryToggle;
  onAxisSet = callbacks.onAxisSet;
  onAxesReset = callbacks.onAxesReset;

  seedInput = document.getElementById("seed-input") as HTMLInputElement;
  seedGo = document.getElementById("seed-go") as HTMLButtonElement;
  searchInput = document.getElementById("search-input") as HTMLInputElement;
  reductionSelect = document.getElementById("reduction-method") as HTMLSelectElement;
  neighborCount = document.getElementById("neighbor-count") as HTMLInputElement;
  categoryList = document.getElementById("category-list")!;
  infoContent = document.getElementById("info-content")!;
  statusText = document.getElementById("status-text")!;
  pointCountEl = document.getElementById("point-count")!;
  hoverLabel = document.getElementById("hover-label")!;

  // Seed input
  seedGo.addEventListener("click", () => {
    const v = seedInput.value.trim();
    if (v) onSeed(v);
  });
  seedInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const v = seedInput.value.trim();
      if (v) onSeed(v);
    }
  });

  // Search with debounce
  let searchTimer: ReturnType<typeof setTimeout>;
  searchInput.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => onSearch(searchInput.value), 200);
  });

  // Reduction method
  reductionSelect.addEventListener("change", () => {
    onReduction(reductionSelect.value as "umap" | "pca");
  });

  // Neighbor count
  neighborCount.addEventListener("change", () => {
    const n = parseInt(neighborCount.value, 10);
    if (n > 0) onNeighborCount(n);
  });

  // Axis buttons
  document.querySelectorAll(".axis-apply").forEach((btn) => {
    btn.addEventListener("click", () => {
      const row = btn.closest(".axis-row") as HTMLElement;
      const idx = parseInt(row.dataset.axis!, 10);
      const a = (row.querySelector(".axis-pole-a") as HTMLInputElement).value.trim();
      const b = (row.querySelector(".axis-pole-b") as HTMLInputElement).value.trim();
      if (a && b) onAxisSet(idx, a, b);
    });
  });

  document.getElementById("axes-reset")!.addEventListener("click", () => {
    onAxesReset();
  });
}

// ---------------------------------------------------------------------------
// Updates
// ---------------------------------------------------------------------------

export function setStatus(msg: string): void {
  statusText.textContent = msg;
}

export function setPointCount(n: number): void {
  pointCountEl.textContent = `${n} points`;
}

export function buildCategoryList(
  categories: string[],
  colorFn: (cat: string) => number,
  hiddenCategories: Set<string>,
): void {
  categoryList.innerHTML = "";

  for (const cat of categories) {
    const row = document.createElement("label");
    row.className = "category-row";

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !hiddenCategories.has(cat);
    cb.addEventListener("change", () => {
      onCategoryToggle(cat, cb.checked);
    });

    const swatch = document.createElement("span");
    swatch.className = "category-swatch";
    swatch.style.backgroundColor = `#${colorFn(cat).toString(16).padStart(6, "0")}`;

    const name = document.createElement("span");
    name.className = "category-name";
    name.textContent = cat;

    row.appendChild(cb);
    row.appendChild(swatch);
    row.appendChild(name);
    categoryList.appendChild(row);
  }
}

export function updateInfoPanel(
  selected: WordPoint | null,
  neighbors: { point: WordPoint; distance: number }[],
  colorFn: (cat: string) => number,
): void {
  if (!selected) {
    infoContent.innerHTML = `<p class="info-placeholder">Click a point to inspect</p>`;
    return;
  }

  const hex = `#${colorFn(selected.category).toString(16).padStart(6, "0")}`;

  let html = `
    <div class="info-word">${escapeHtml(selected.word)}</div>
    <div class="info-category" style="color:${hex}">${escapeHtml(selected.category)}</div>
  `;

  if (neighbors.length > 0) {
    html += `<div class="info-neighbors-title">Nearest neighbors</div><ul class="info-neighbors">`;
    for (const n of neighbors) {
      const nHex = `#${colorFn(n.point.category).toString(16).padStart(6, "0")}`;
      html += `<li>
        <span class="neighbor-word" data-word="${escapeAttr(n.point.word)}">${escapeHtml(n.point.word)}</span>
        <span class="neighbor-cat" style="color:${nHex}">${escapeHtml(n.point.category)}</span>
        <span class="neighbor-dist">${n.distance.toFixed(3)}</span>
      </li>`;
    }
    html += `</ul>`;
  }

  infoContent.innerHTML = html;

  // Clicking a neighbor word sets it as seed
  infoContent.querySelectorAll(".neighbor-word").forEach((el) => {
    el.addEventListener("click", () => {
      const w = (el as HTMLElement).dataset.word;
      if (w) {
        seedInput.value = w;
        onSeed(w);
      }
    });
  });
}

export function showHoverLabel(
  word: string,
  category: string,
  x: number,
  y: number,
  colorHex: string,
): void {
  hoverLabel.style.display = "block";
  hoverLabel.style.left = `${x + 12}px`;
  hoverLabel.style.top = `${y - 8}px`;
  hoverLabel.innerHTML = `<span style="color:${colorHex}">${escapeHtml(word)}</span> <span class="hover-cat">${escapeHtml(category)}</span>`;
}

export function hideHoverLabel(): void {
  hoverLabel.style.display = "none";
}

export function setSeedValue(word: string): void {
  seedInput.value = word;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function escapeAttr(s: string): string {
  return escapeHtml(s).replace(/"/g, "&quot;");
}

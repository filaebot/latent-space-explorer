import type { WordPoint } from "./types";
import type { Cluster } from "./clusters";
import type { Tour, TourState } from "./tour";

// ---------------------------------------------------------------------------
// DOM references (cached on init)
// ---------------------------------------------------------------------------

let seedInput: HTMLInputElement;
let seedGo: HTMLButtonElement;
let searchInput: HTMLInputElement;
let reductionSelect: HTMLSelectElement;
let neighborCount: HTMLInputElement;
let neighborModeSelect: HTMLSelectElement;
let categoryList: HTMLElement;
let clusterList: HTMLElement;
let infoContent: HTMLElement;
let statusText: HTMLElement;
let pointCountEl: HTMLElement;
let hoverLabel: HTMLElement;
let progressOverlay: HTMLElement;
let progressMessage: HTMLElement;
let progressBarFill: HTMLElement;
let progressPercent: HTMLElement;
let modeEmbeddingsBtn: HTMLButtonElement;
let modeLayersBtn: HTMLButtonElement;
let layerControls: HTMLElement;
let layerSlider: HTMLInputElement;
let layerLabel: HTMLElement;
let layerPlayBtn: HTMLButtonElement;
let layerSpeedSlider: HTMLInputElement;
let layerSpeedLabel: HTMLElement;
let tourBtn: HTMLButtonElement;
let tourMenu: HTMLElement;
let tourMenuList: HTMLElement;
let tourMenuClose: HTMLButtonElement;
let tourOverlay: HTMLElement;
let tourPrev: HTMLButtonElement;
let tourPlayPause: HTMLButtonElement;
let tourNext: HTMLButtonElement;
let tourStop: HTMLButtonElement;
let tourAnnotation: HTMLElement;
let tourStepCounter: HTMLElement;
let focusBackBtn: HTMLElement;
let focusExitBtn: HTMLButtonElement;

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
type ModeCallback = (mode: "embeddings" | "layers") => void;
type LayerChangeCallback = (layer: number) => void;
type PlayToggleCallback = () => void;
type SpeedChangeCallback = (speed: number) => void;
type NeighborModeCallback = (mode: "spatial" | "semantic") => void;
type ClusterClickCallback = (clusterId: number) => void;
type TourSelectCallback = (tourIndex: number) => void;
type TourControlCallback = (action: "prev" | "next" | "play-pause" | "stop") => void;
type FocusExitCallback = () => void;

let onSeed: SeedCallback = () => {};
let onSearch: SearchCallback = () => {};
let onReduction: ReductionCallback = () => {};
let onNeighborCount: NeighborCountCallback = () => {};
let onCategoryToggle: CategoryToggleCallback = () => {};
let onAxisSet: AxisCallback = () => {};
let onAxesReset: AxesResetCallback = () => {};
let onModeChange: ModeCallback = () => {};
let onLayerChange: LayerChangeCallback = () => {};
let onPlayToggle: PlayToggleCallback = () => {};
let onSpeedChange: SpeedChangeCallback = () => {};
let onNeighborMode: NeighborModeCallback = () => {};
let onClusterClick: ClusterClickCallback = () => {};
let onTourSelect: TourSelectCallback = () => {};
let onTourControl: TourControlCallback = () => {};
let onFocusExit: FocusExitCallback = () => {};

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
  onModeChange: ModeCallback;
  onLayerChange: LayerChangeCallback;
  onPlayToggle: PlayToggleCallback;
  onSpeedChange: SpeedChangeCallback;
  onNeighborMode: NeighborModeCallback;
  onClusterClick: ClusterClickCallback;
  onTourSelect: TourSelectCallback;
  onTourControl: TourControlCallback;
  onFocusExit: FocusExitCallback;
}): void {
  onSeed = callbacks.onSeed;
  onSearch = callbacks.onSearch;
  onReduction = callbacks.onReduction;
  onNeighborCount = callbacks.onNeighborCount;
  onCategoryToggle = callbacks.onCategoryToggle;
  onAxisSet = callbacks.onAxisSet;
  onAxesReset = callbacks.onAxesReset;
  onModeChange = callbacks.onModeChange;
  onLayerChange = callbacks.onLayerChange;
  onPlayToggle = callbacks.onPlayToggle;
  onSpeedChange = callbacks.onSpeedChange;
  onNeighborMode = callbacks.onNeighborMode;
  onClusterClick = callbacks.onClusterClick;
  onTourSelect = callbacks.onTourSelect;
  onTourControl = callbacks.onTourControl;
  onFocusExit = callbacks.onFocusExit;

  seedInput = document.getElementById("seed-input") as HTMLInputElement;
  seedGo = document.getElementById("seed-go") as HTMLButtonElement;
  searchInput = document.getElementById("search-input") as HTMLInputElement;
  reductionSelect = document.getElementById("reduction-method") as HTMLSelectElement;
  neighborCount = document.getElementById("neighbor-count") as HTMLInputElement;
  neighborModeSelect = document.getElementById("neighbor-mode") as HTMLSelectElement;
  categoryList = document.getElementById("category-list")!;
  clusterList = document.getElementById("cluster-list")!;
  infoContent = document.getElementById("info-content")!;
  statusText = document.getElementById("status-text")!;
  pointCountEl = document.getElementById("point-count")!;
  hoverLabel = document.getElementById("hover-label")!;
  progressOverlay = document.getElementById("progress-overlay")!;
  progressMessage = document.getElementById("progress-message")!;
  progressBarFill = document.getElementById("progress-bar-fill")!;
  progressPercent = document.getElementById("progress-percent")!;
  modeEmbeddingsBtn = document.getElementById("mode-embeddings") as HTMLButtonElement;
  modeLayersBtn = document.getElementById("mode-layers") as HTMLButtonElement;
  layerControls = document.getElementById("layer-controls")!;
  layerSlider = document.getElementById("layer-slider") as HTMLInputElement;
  layerLabel = document.getElementById("layer-label")!;
  layerPlayBtn = document.getElementById("layer-play") as HTMLButtonElement;
  layerSpeedSlider = document.getElementById("layer-speed") as HTMLInputElement;
  layerSpeedLabel = document.getElementById("layer-speed-label")!;
  tourBtn = document.getElementById("tour-btn") as HTMLButtonElement;
  tourMenu = document.getElementById("tour-menu")!;
  tourMenuList = document.getElementById("tour-menu-list")!;
  tourMenuClose = document.getElementById("tour-menu-close") as HTMLButtonElement;
  tourOverlay = document.getElementById("tour-overlay")!;
  tourPrev = document.getElementById("tour-prev") as HTMLButtonElement;
  tourPlayPause = document.getElementById("tour-play-pause") as HTMLButtonElement;
  tourNext = document.getElementById("tour-next") as HTMLButtonElement;
  tourStop = document.getElementById("tour-stop") as HTMLButtonElement;
  tourAnnotation = document.getElementById("tour-annotation")!;
  tourStepCounter = document.getElementById("tour-step-counter")!;
  focusBackBtn = document.getElementById("focus-back-btn")!;
  focusExitBtn = document.getElementById("focus-exit") as HTMLButtonElement;

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

  // Neighbor mode
  neighborModeSelect.addEventListener("change", () => {
    onNeighborMode(neighborModeSelect.value as "spatial" | "semantic");
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

  // Mode toggle
  modeEmbeddingsBtn.addEventListener("click", () => onModeChange("embeddings"));
  modeLayersBtn.addEventListener("click", () => onModeChange("layers"));

  // Layer slider
  layerSlider.addEventListener("input", () => {
    onLayerChange(parseInt(layerSlider.value, 10));
  });

  // Play/pause
  layerPlayBtn.addEventListener("click", () => onPlayToggle());

  // Speed slider
  layerSpeedSlider.addEventListener("input", () => {
    const speed = parseFloat(layerSpeedSlider.value);
    layerSpeedLabel.textContent = `${speed.toFixed(1)}x`;
    onSpeedChange(speed);
  });

  // Tour button
  tourBtn.addEventListener("click", () => {
    tourMenu.style.display = tourMenu.style.display === "none" ? "block" : "none";
  });
  tourMenuClose.addEventListener("click", () => {
    tourMenu.style.display = "none";
  });

  // Tour controls
  tourPrev.addEventListener("click", () => onTourControl("prev"));
  tourPlayPause.addEventListener("click", () => onTourControl("play-pause"));
  tourNext.addEventListener("click", () => onTourControl("next"));
  tourStop.addEventListener("click", () => onTourControl("stop"));

  // Focus exit
  focusExitBtn.addEventListener("click", () => onFocusExit());
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

export function buildClusterList(
  clusters: Cluster[],
  colorFn: (cat: string) => number,
): void {
  clusterList.innerHTML = "";

  for (const cluster of clusters) {
    const item = document.createElement("div");
    item.className = "cluster-item";
    item.addEventListener("click", () => onClusterClick(cluster.id));

    const swatch = document.createElement("span");
    swatch.className = "cluster-swatch";
    swatch.style.backgroundColor = `#${colorFn(cluster.dominantCategory).toString(16).padStart(6, "0")}`;

    const label = document.createElement("span");
    label.className = "cluster-label";
    label.textContent = cluster.label;

    const count = document.createElement("span");
    count.className = "cluster-count";
    count.textContent = `${cluster.points.length}`;

    item.appendChild(swatch);
    item.appendChild(label);
    item.appendChild(count);
    clusterList.appendChild(item);

    // Stats line
    const cats = [...cluster.categoryCounts.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([c, n]) => `${c}(${n})`)
      .join(", ");
    const stats = document.createElement("div");
    stats.className = "cluster-stats";
    stats.textContent = cats;
    clusterList.appendChild(stats);
  }
}

export function updateInfoPanel(
  selected: WordPoint | null,
  neighbors: { point: WordPoint; distance: number }[],
  colorFn: (cat: string) => number,
  neighborMode?: "spatial" | "semantic",
): void {
  if (!selected) {
    infoContent.innerHTML = `<p class="info-placeholder">Click a point to inspect</p>`;
    return;
  }

  const hex = `#${colorFn(selected.category).toString(16).padStart(6, "0")}`;
  const isSemantic = neighborMode === "semantic";
  const distLabel = isSemantic ? "cosine sim" : "distance";

  let html = `
    <div class="info-word">${escapeHtml(selected.word)}</div>
    <div class="info-category" style="color:${hex}">${escapeHtml(selected.category)}</div>
  `;

  if (neighbors.length > 0) {
    html += `<div class="info-neighbor-mode">${isSemantic ? "Semantic" : "Spatial"} neighbors (${distLabel})</div>`;
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

export function updateInfoPanelWithPath(
  words: WordPoint[],
  colorFn: (cat: string) => number,
): void {
  let html = `<div class="info-neighbors-title">Semantic Path</div><ul class="info-neighbors">`;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const nHex = `#${colorFn(w.category).toString(16).padStart(6, "0")}`;
    const prefix = i === 0 ? "START " : i === words.length - 1 ? "END " : `${i}. `;
    html += `<li>
      <span class="neighbor-word" data-word="${escapeAttr(w.word)}">${escapeHtml(prefix + w.word)}</span>
      <span class="neighbor-cat" style="color:${nHex}">${escapeHtml(w.category)}</span>
    </li>`;
  }
  html += `</ul>`;
  infoContent.innerHTML = html;

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

export function showProgress(percent: number, message: string): void {
  progressOverlay.classList.add("visible");
  progressMessage.textContent = message;
  progressBarFill.style.width = `${Math.round(percent)}%`;
  progressPercent.textContent = `${Math.round(percent)}%`;
}

export function hideProgress(): void {
  progressOverlay.classList.remove("visible");
}

// ---------------------------------------------------------------------------
// Layer mode UI
// ---------------------------------------------------------------------------

export function setMode(mode: "embeddings" | "layers"): void {
  if (mode === "embeddings") {
    modeEmbeddingsBtn.classList.add("active");
    modeLayersBtn.classList.remove("active");
    layerControls.classList.remove("visible");
    document.body.classList.remove("layers-mode");
  } else {
    modeEmbeddingsBtn.classList.remove("active");
    modeLayersBtn.classList.add("active");
    layerControls.classList.add("visible");
    document.body.classList.add("layers-mode");
  }
}

export function setLayerSliderMax(max: number): void {
  layerSlider.max = String(max);
}

export function setLayerSliderValue(value: number): void {
  layerSlider.value = String(value);
}

export function setLayerLabel(layer: number, numLayers: number): void {
  let desc: string;
  if (layer === 0) {
    desc = `Layer 0 (Embedding)`;
  } else if (layer === numLayers - 1) {
    desc = `Layer ${layer} (Output)`;
  } else {
    desc = `Layer ${layer}`;
  }
  layerLabel.textContent = desc;
}

export function setPlayButton(isPlaying: boolean): void {
  layerPlayBtn.textContent = isPlaying ? "Pause" : "Play";
}

// ---------------------------------------------------------------------------
// Focus mode UI
// ---------------------------------------------------------------------------

export function showFocusBack(): void {
  focusBackBtn.style.display = "block";
}

export function hideFocusBack(): void {
  focusBackBtn.style.display = "none";
}

// ---------------------------------------------------------------------------
// Tour UI
// ---------------------------------------------------------------------------

export function buildTourMenu(tours: Tour[]): void {
  tourMenuList.innerHTML = "";
  tours.forEach((tour, i) => {
    const btn = document.createElement("button");
    btn.className = "tour-menu-item";
    btn.textContent = `${tour.name} (${tour.steps.length} stops)`;
    btn.addEventListener("click", () => {
      tourMenu.style.display = "none";
      onTourSelect(i);
    });
    tourMenuList.appendChild(btn);
  });
}

export function showTourOverlay(): void {
  tourOverlay.style.display = "block";
}

export function hideTourOverlay(): void {
  tourOverlay.style.display = "none";
}

export function updateTourStep(tourState: TourState): void {
  if (!tourState.tour) return;
  const step = tourState.tour.steps[tourState.stepIndex];
  tourAnnotation.textContent = step.annotation;
  tourStepCounter.textContent = `${tourState.stepIndex + 1}/${tourState.tour.steps.length}`;
  tourPlayPause.textContent = tourState.isPlaying ? "Pause" : "Play";
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

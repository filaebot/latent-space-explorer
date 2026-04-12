import { createScene, animate } from "./scene";
import { createScatterPlot } from "./scatter";
import {
  initUI,
  setStatus,
  setPointCount,
  buildCategoryList,
  updateInfoPanel,
  showHoverLabel,
  hideHoverLabel,
  setSeedValue,
} from "./ui";
import { loadDataset, findNeighbors } from "./api";
import type { AppState } from "./types";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state: AppState = {
  dataset: { points: [], categories: [] },
  seed: null,
  neighbors: [],
  selectedPoint: null,
  searchQuery: "",
  reductionMethod: "umap",
  neighborCount: 10,
  axes: [null, null, null],
  hiddenCategories: new Set(),
  usingMockData: true,
};

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const container = document.getElementById("canvas-container")!;
  const sceneCtx = createScene(container);
  const scatter = createScatterPlot(sceneCtx);

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  function refreshVisibility(): void {
    scatter.updateVisibility(state.hiddenCategories, state.searchQuery);
  }

  function applySeed(word: string): void {
    const point = state.dataset.points.find(
      (p) => p.word.toLowerCase() === word.toLowerCase(),
    );

    if (!point) {
      setStatus(`"${word}" not found in dataset`);
      return;
    }

    state.seed = word;
    state.selectedPoint = point;

    const nbs = findNeighbors(point, state.dataset.points, state.neighborCount);
    state.neighbors = nbs.map((n) => n.point);

    scatter.highlightSeed(point, state.neighbors);
    scatter.selectPoint(point);
    updateInfoPanel(point, nbs, scatter.categoryColor);
    setSeedValue(word);
    setStatus(`Seed: ${word} (${nbs.length} neighbors)`);
  }

  // -----------------------------------------------------------------------
  // UI callbacks
  // -----------------------------------------------------------------------

  initUI({
    onSeed: (word) => applySeed(word),

    onSearch: (query) => {
      state.searchQuery = query;
      refreshVisibility();
      setStatus(query ? `Search: "${query}"` : "Ready");
    },

    onReduction: async (method) => {
      state.reductionMethod = method;
      setStatus(`Switching to ${method.toUpperCase()}...`);
      const { dataset, mock } = await loadDataset(method);
      state.dataset = dataset;
      state.usingMockData = mock;
      scatter.setPoints(dataset.points, dataset.categories);
      buildCategoryList(dataset.categories, scatter.categoryColor, state.hiddenCategories);
      setPointCount(dataset.points.length);
      refreshVisibility();
      setStatus(mock ? `Mock data (${method.toUpperCase()})` : `${method.toUpperCase()} reduction`);
    },

    onNeighborCount: (n) => {
      state.neighborCount = n;
      if (state.selectedPoint) {
        applySeed(state.selectedPoint.word);
      }
    },

    onCategoryToggle: (category, visible) => {
      if (visible) {
        state.hiddenCategories.delete(category);
      } else {
        state.hiddenCategories.add(category);
      }
      refreshVisibility();
    },

    onAxisSet: (_idx, _poleA, _poleB) => {
      // Semantic axes require embedding-space projection via the backend.
      // For now, store the axis definition and show a status message.
      setStatus("Semantic axes require a running backend");
    },

    onAxesReset: () => {
      state.axes = [null, null, null];
      setStatus("Axes reset to reduction coordinates");
    },
  });

  // -----------------------------------------------------------------------
  // Mouse interaction
  // -----------------------------------------------------------------------

  container.addEventListener("mousemove", (e) => {
    const hit = scatter.raycast(e, container);

    if (hit) {
      const hex = `#${scatter.categoryColor(hit.category).toString(16).padStart(6, "0")}`;
      showHoverLabel(hit.word, hit.category, e.clientX, e.clientY, hex);
      container.style.cursor = "pointer";
    } else {
      hideHoverLabel();
      container.style.cursor = "default";
    }
  });

  container.addEventListener("click", (e) => {
    const hit = scatter.raycast(e, container);
    if (hit) {
      applySeed(hit.word);
    }
  });

  // -----------------------------------------------------------------------
  // Render loop
  // -----------------------------------------------------------------------

  animate(sceneCtx, () => {
    // Keep selection ring facing camera
    // (handled implicitly since we rebuild on select)
  });

  // -----------------------------------------------------------------------
  // Initial load
  // -----------------------------------------------------------------------

  setStatus("Loading dataset...");
  const { dataset, mock } = await loadDataset(state.reductionMethod);
  state.dataset = dataset;
  state.usingMockData = mock;

  scatter.setPoints(dataset.points, dataset.categories);
  buildCategoryList(dataset.categories, scatter.categoryColor, state.hiddenCategories);
  setPointCount(dataset.points.length);
  refreshVisibility();

  if (mock) {
    setStatus("Mock data mode (backend unavailable)");
  } else {
    setStatus("Ready");
  }
}

main().catch((err) => {
  console.error("Fatal:", err);
  const statusText = document.getElementById("status-text");
  if (statusText) statusText.textContent = `Error: ${err.message}`;
});

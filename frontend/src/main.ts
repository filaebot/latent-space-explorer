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
  showProgress,
  hideProgress,
  setMode,
  setLayerSliderMax,
  setLayerSliderValue,
  setLayerLabel,
  setPlayButton,
} from "./ui";
import { loadDataset, findNeighbors, fetchLayerData } from "./api";
import type { AppState, LayerData, WordPoint } from "./types";

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
  mode: "embeddings",
  currentLayer: 0,
  layerData: null,
  isPlaying: false,
  playSpeed: 1,
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

  let playTimer: ReturnType<typeof setTimeout> | null = null;

  function refreshVisibility(): void {
    scatter.updateVisibility(state.hiddenCategories, state.searchQuery);
  }

  /** Build a DataSet from layer data at a specific layer. */
  function layerDataToDataset(data: LayerData, layer: number, method: "umap" | "pca"): { points: WordPoint[]; categories: string[] } {
    const layerEntry = data.layers[String(layer)];
    if (!layerEntry) return { points: [], categories: [] };

    const coords = layerEntry[method];
    const points: WordPoint[] = data.words.map((word, i) => ({
      word,
      category: data.categories[word] ?? "unknown",
      position: coords[i] as [number, number, number],
    }));
    const categories = [...new Set(Object.values(data.categories))];
    return { points, categories };
  }

  /** Switch to a specific layer with animated transition. */
  async function goToLayer(layer: number): Promise<void> {
    if (!state.layerData) return;

    const layerEntry = state.layerData.layers[String(layer)];
    if (!layerEntry) return;

    state.currentLayer = layer;
    setLayerSliderValue(layer);
    setLayerLabel(layer, state.layerData.num_layers);

    const coords = layerEntry[state.reductionMethod];

    // If points haven't been set yet (first load), use setPoints instead of animating
    if (state.dataset.points.length === 0) {
      const ds = layerDataToDataset(state.layerData, layer, state.reductionMethod);
      state.dataset = ds;
      scatter.setPoints(ds.points, ds.categories);
      buildCategoryList(ds.categories, scatter.categoryColor, state.hiddenCategories);
      setPointCount(ds.points.length);
      refreshVisibility();
      return;
    }

    const targets = coords as [number, number, number][];
    await scatter.animateToPositions(targets, 300);

    // Sync logical dataset positions after animation
    for (let i = 0; i < state.dataset.points.length; i++) {
      if (targets[i]) {
        state.dataset.points[i].position = [...targets[i]] as [number, number, number];
      }
    }

    // Re-apply seed highlights with new positions
    if (state.selectedPoint) {
      const nbs = findNeighbors(state.selectedPoint, state.dataset.points, state.neighborCount);
      state.neighbors = nbs.map((n) => n.point);
      scatter.highlightSeed(state.selectedPoint, state.neighbors);
      scatter.selectPoint(state.selectedPoint);
      updateInfoPanel(state.selectedPoint, nbs, scatter.categoryColor);
    }
  }

  function stopAutoPlay(): void {
    state.isPlaying = false;
    setPlayButton(false);
    if (playTimer !== null) {
      clearTimeout(playTimer);
      playTimer = null;
    }
  }

  function startAutoPlay(): void {
    state.isPlaying = true;
    setPlayButton(true);
    advanceLayer();
  }

  function advanceLayer(): void {
    if (!state.isPlaying || !state.layerData) return;

    const next = state.currentLayer + 1;
    if (next >= state.layerData.num_layers) {
      stopAutoPlay();
      return;
    }

    goToLayer(next).then(() => {
      if (!state.isPlaying) return;
      // Wait proportional to speed before advancing again
      const delay = Math.max(50, 400 / state.playSpeed);
      playTimer = setTimeout(advanceLayer, delay);
    });
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

      if (state.mode === "layers" && state.layerData) {
        // In layers mode, switch reduction for the current layer with animation
        setStatus(`Switching to ${method.toUpperCase()}...`);
        const layerEntry = state.layerData.layers[String(state.currentLayer)];
        if (layerEntry) {
          const targets = layerEntry[method] as [number, number, number][];
          await scatter.animateToPositions(targets, 300);
          for (let i = 0; i < state.dataset.points.length; i++) {
            if (targets[i]) {
              state.dataset.points[i].position = [...targets[i]] as [number, number, number];
            }
          }
        }
        setStatus(`Layers mode (${method.toUpperCase()})`);
        return;
      }

      setStatus(`Switching to ${method.toUpperCase()}...`);
      const { dataset, mock } = await loadDataset(method, (pct, msg) => {
        showProgress(pct, msg);
      });
      hideProgress();
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

    onModeChange: async (mode) => {
      if (mode === state.mode) return;
      stopAutoPlay();
      state.mode = mode;
      setMode(mode);

      if (mode === "layers") {
        setStatus("Loading layer data...");
        const data = await fetchLayerData();
        if (!data) {
          setStatus("Layer data not available -- run precompute_layers.py");
          state.mode = "embeddings";
          setMode("embeddings");
          return;
        }
        state.layerData = data;
        setLayerSliderMax(data.num_layers - 1);

        // Build initial dataset from layer 0
        const ds = layerDataToDataset(data, 0, state.reductionMethod);
        state.dataset = ds;
        state.currentLayer = 0;
        state.seed = null;
        state.selectedPoint = null;
        state.neighbors = [];

        scatter.setPoints(ds.points, ds.categories);
        buildCategoryList(ds.categories, scatter.categoryColor, state.hiddenCategories);
        setPointCount(ds.points.length);
        setLayerSliderValue(0);
        setLayerLabel(0, data.num_layers);
        refreshVisibility();
        setStatus(`Layers mode: ${data.model} (${data.num_layers} layers)`);
      } else {
        // Switch back to embeddings mode
        state.layerData = null;
        setStatus("Loading dataset...");
        const { dataset, mock } = await loadDataset(state.reductionMethod, (pct, msg) => {
          showProgress(pct, msg);
        });
        hideProgress();
        state.dataset = dataset;
        state.usingMockData = mock;
        state.seed = null;
        state.selectedPoint = null;
        state.neighbors = [];

        scatter.setPoints(dataset.points, dataset.categories);
        buildCategoryList(dataset.categories, scatter.categoryColor, state.hiddenCategories);
        setPointCount(dataset.points.length);
        refreshVisibility();
        setStatus(mock ? "Mock data mode (backend unavailable)" : "Ready");
      }
    },

    onLayerChange: (layer) => {
      stopAutoPlay();
      goToLayer(layer);
    },

    onPlayToggle: () => {
      if (state.isPlaying) {
        stopAutoPlay();
      } else {
        startAutoPlay();
      }
    },

    onSpeedChange: (speed) => {
      state.playSpeed = speed;
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
  const { dataset, mock } = await loadDataset(state.reductionMethod, (pct, msg) => {
    showProgress(pct, msg);
  });
  hideProgress();
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

import * as THREE from "three";
import { createScene, animate } from "./scene";
import { createScatterPlot } from "./scatter";
import { initUI, setStatus, setPointCount, buildCategoryList, buildClusterList, updateInfoPanel, updateInfoPanelWithPath, showHoverLabel, hideHoverLabel, setSeedValue, showProgress, hideProgress, setMode, setLayerSliderMax, setLayerSliderValue, setLayerLabel, setPlayButton, showFocusBack, hideFocusBack, buildTourMenu, showTourOverlay, hideTourOverlay, updateTourStep, } from "./ui";
import { loadDataset, findNeighbors, findSemanticNeighbors, fetchLayerData } from "./api";
import { computeClusters, computeCategoryCentroids } from "./clusters";
import { createMinimap } from "./minimap";
import { findSemanticPath, createPathRenderer } from "./path";
import { generateTours, createTourState } from "./tour";
// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
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
// Extended state for new features
let neighborMode = "semantic";
let clusters = [];
let categoryCentroids = new Map();
let focusMode = false;
let focusSavedCamera = null;
let pathStartPoint = null;
let tours = [];
const tourState = createTourState();
let tourAutoTimer = null;
let categorySprites = null;
// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function main() {
    const container = document.getElementById("canvas-container");
    const sceneCtx = createScene(container);
    const scatter = createScatterPlot(sceneCtx);
    const pathRenderer = createPathRenderer(sceneCtx);
    // Minimap setup
    const minimap = createMinimap((worldX, worldZ) => {
        // Fly camera to clicked minimap position
        flyCamera(new THREE.Vector3(worldX, 2, worldZ), new THREE.Vector3(worldX, 0, worldZ), 600);
    });
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    let playTimer = null;
    function refreshVisibility() {
        scatter.updateVisibility(state.hiddenCategories, state.searchQuery);
    }
    /** Fly camera smoothly to a target position/lookAt. */
    function flyCamera(newPos, newTarget, duration) {
        const startPos = sceneCtx.camera.position.clone();
        const startTarget = sceneCtx.controls.target.clone();
        const startTime = performance.now();
        return new Promise((resolve) => {
            function step() {
                const elapsed = performance.now() - startTime;
                const t = Math.min(elapsed / duration, 1);
                const ease = 1 - (1 - t) * (1 - t); // ease-out
                sceneCtx.camera.position.lerpVectors(startPos, newPos, ease);
                sceneCtx.controls.target.lerpVectors(startTarget, newTarget, ease);
                sceneCtx.controls.update();
                if (t < 1) {
                    requestAnimationFrame(step);
                }
                else {
                    resolve();
                }
            }
            requestAnimationFrame(step);
        });
    }
    /** Create text sprites for category centroids. */
    function updateCategorySprites() {
        if (categorySprites) {
            sceneCtx.scene.remove(categorySprites);
            categorySprites.traverse((child) => {
                if (child instanceof THREE.Sprite) {
                    child.material.map?.dispose();
                    child.material.dispose();
                }
            });
            categorySprites = null;
        }
        if (categoryCentroids.size === 0)
            return;
        categorySprites = new THREE.Group();
        for (const [cat, centroid] of categoryCentroids) {
            const hex = scatter.categoryColor(cat);
            const color = `#${hex.toString(16).padStart(6, "0")}`;
            const canvas = document.createElement("canvas");
            const size = 256;
            canvas.width = size;
            canvas.height = 64;
            const ctx2 = canvas.getContext("2d");
            ctx2.font = "bold 20px -apple-system, sans-serif";
            ctx2.fillStyle = color;
            ctx2.globalAlpha = 0.6;
            ctx2.textAlign = "center";
            ctx2.textBaseline = "middle";
            ctx2.fillText(cat.toUpperCase(), size / 2, 32);
            const texture = new THREE.CanvasTexture(canvas);
            texture.needsUpdate = true;
            const mat = new THREE.SpriteMaterial({
                map: texture,
                transparent: true,
                opacity: 0.5,
                depthTest: false,
            });
            const sprite = new THREE.Sprite(mat);
            sprite.position.set(centroid[0], centroid[1] + 0.6, centroid[2]);
            sprite.scale.set(2.5, 0.65, 1);
            categorySprites.add(sprite);
        }
        sceneCtx.scene.add(categorySprites);
    }
    /** Recompute clusters and update UI. */
    function rebuildClusters() {
        clusters = computeClusters(state.dataset.points);
        categoryCentroids = computeCategoryCentroids(state.dataset.points, state.dataset.categories);
        buildClusterList(clusters, scatter.categoryColor);
        updateCategorySprites();
        // Regenerate tours
        tours = generateTours(state.dataset.points, state.dataset.categories, clusters, categoryCentroids);
        buildTourMenu(tours);
    }
    /** Get neighbors using current mode. */
    function getNeighbors(point, k) {
        if (neighborMode === "semantic" && point.embedding) {
            return findSemanticNeighbors(point, state.dataset.points, k);
        }
        return findNeighbors(point, state.dataset.points, k);
    }
    /** Build a DataSet from layer data at a specific layer. */
    function layerDataToDataset(data, layer, method) {
        const layerEntry = data.layers[String(layer)];
        if (!layerEntry)
            return { points: [], categories: [] };
        const coords = layerEntry[method];
        const points = data.words.map((word, i) => ({
            word,
            category: data.categories[word] ?? "unknown",
            position: coords[i],
        }));
        const categories = [...new Set(Object.values(data.categories))];
        return { points, categories };
    }
    /** Switch to a specific layer with animated transition. */
    async function goToLayer(layer) {
        if (!state.layerData)
            return;
        const layerEntry = state.layerData.layers[String(layer)];
        if (!layerEntry)
            return;
        state.currentLayer = layer;
        setLayerSliderValue(layer);
        setLayerLabel(layer, state.layerData.num_layers);
        const coords = layerEntry[state.reductionMethod];
        if (state.dataset.points.length === 0) {
            const ds = layerDataToDataset(state.layerData, layer, state.reductionMethod);
            state.dataset = ds;
            scatter.setPoints(ds.points, ds.categories);
            buildCategoryList(ds.categories, scatter.categoryColor, state.hiddenCategories);
            setPointCount(ds.points.length);
            refreshVisibility();
            return;
        }
        const targets = coords;
        await scatter.animateToPositions(targets, 300);
        for (let i = 0; i < state.dataset.points.length; i++) {
            if (targets[i]) {
                state.dataset.points[i].position = [...targets[i]];
            }
        }
        // Refresh minimap and clusters with the new positions
        minimap.setPoints(state.dataset.points, scatter.categoryColor);
        rebuildClusters();
        if (state.selectedPoint) {
            const nbs = getNeighbors(state.selectedPoint, state.neighborCount);
            state.neighbors = nbs.map((n) => n.point);
            scatter.highlightSeed(state.selectedPoint, state.neighbors);
            scatter.selectPoint(state.selectedPoint);
            updateInfoPanel(state.selectedPoint, nbs, scatter.categoryColor, neighborMode);
        }
    }
    function stopAutoPlay() {
        state.isPlaying = false;
        setPlayButton(false);
        if (playTimer !== null) {
            clearTimeout(playTimer);
            playTimer = null;
        }
    }
    function startAutoPlay() {
        state.isPlaying = true;
        setPlayButton(true);
        advanceLayer();
    }
    function advanceLayer() {
        if (!state.isPlaying || !state.layerData)
            return;
        const next = state.currentLayer + 1;
        if (next >= state.layerData.num_layers) {
            stopAutoPlay();
            return;
        }
        goToLayer(next).then(() => {
            if (!state.isPlaying)
                return;
            const delay = Math.max(50, 400 / state.playSpeed);
            playTimer = setTimeout(advanceLayer, delay);
        });
    }
    function applySeed(word) {
        const point = state.dataset.points.find((p) => p.word.toLowerCase() === word.toLowerCase());
        if (!point) {
            setStatus(`"${word}" not found in dataset`);
            return;
        }
        state.seed = word;
        state.selectedPoint = point;
        // Clear path state on new seed
        pathStartPoint = null;
        pathRenderer.clear();
        const nbs = getNeighbors(point, state.neighborCount);
        state.neighbors = nbs.map((n) => n.point);
        scatter.highlightSeed(point, state.neighbors);
        scatter.selectPoint(point);
        updateInfoPanel(point, nbs, scatter.categoryColor, neighborMode);
        setSeedValue(word);
        setStatus(`Seed: ${word} (${nbs.length} neighbors)`);
    }
    // -----------------------------------------------------------------------
    // Focus mode
    // -----------------------------------------------------------------------
    function enterFocusMode(point) {
        if (focusMode)
            return;
        focusMode = true;
        focusSavedCamera = {
            pos: sceneCtx.camera.position.clone(),
            target: sceneCtx.controls.target.clone(),
        };
        // Apply seed
        applySeed(point.word);
        // Fade non-relevant points
        const relevantWords = new Set([point.word, ...state.neighbors.map((n) => n.word)]);
        const opacities = new Float32Array(state.dataset.points.length);
        for (let i = 0; i < state.dataset.points.length; i++) {
            opacities[i] = relevantWords.has(state.dataset.points[i].word) ? 1.0 : 0.05;
        }
        scatter.setOpacities(opacities);
        // Fly close
        const target = new THREE.Vector3(...point.position);
        const offset = new THREE.Vector3(0.8, 0.5, 0.8);
        flyCamera(target.clone().add(offset), target, 800);
        showFocusBack();
        setStatus(`Focus: ${point.word}`);
    }
    function exitFocusMode() {
        if (!focusMode)
            return;
        focusMode = false;
        scatter.setOpacities(null);
        if (focusSavedCamera) {
            flyCamera(focusSavedCamera.pos, focusSavedCamera.target, 600);
            focusSavedCamera = null;
        }
        hideFocusBack();
        refreshVisibility();
        setStatus("Ready");
    }
    // -----------------------------------------------------------------------
    // Tour control
    // -----------------------------------------------------------------------
    function executeTourStep() {
        if (!tourState.tour)
            return;
        const step = tourState.tour.steps[tourState.stepIndex];
        // Fly camera
        const target = new THREE.Vector3(...step.target);
        const offset = new THREE.Vector3(2, 1.5, 2);
        flyCamera(target.clone().add(offset), target, 1000);
        // Highlight words
        if (step.neighborSeed) {
            applySeed(step.neighborSeed);
        }
        else if (step.highlightWords.length > 0) {
            // Highlight these specific words
            const highlighted = state.dataset.points.filter((p) => step.highlightWords.includes(p.word));
            if (highlighted.length > 0) {
                scatter.highlightSeed(highlighted[0], highlighted.slice(1));
            }
        }
        updateTourStep(tourState);
        setStatus(`Tour: ${step.annotation}`);
    }
    function startTour(tourIndex) {
        stopTour();
        tourState.tour = tours[tourIndex];
        tourState.stepIndex = 0;
        tourState.isPlaying = true;
        showTourOverlay();
        executeTourStep();
        scheduleTourAdvance();
    }
    function stopTour() {
        tourState.tour = null;
        tourState.isPlaying = false;
        if (tourAutoTimer) {
            clearTimeout(tourAutoTimer);
            tourAutoTimer = null;
        }
        hideTourOverlay();
    }
    function scheduleTourAdvance() {
        if (!tourState.isPlaying || !tourState.tour)
            return;
        if (tourAutoTimer)
            clearTimeout(tourAutoTimer);
        tourAutoTimer = setTimeout(() => {
            if (!tourState.tour || !tourState.isPlaying)
                return;
            if (tourState.stepIndex < tourState.tour.steps.length - 1) {
                tourState.stepIndex++;
                executeTourStep();
                scheduleTourAdvance();
            }
            else {
                tourState.isPlaying = false;
                updateTourStep(tourState);
            }
        }, tourState.autoAdvanceDelay);
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
                setStatus(`Switching to ${method.toUpperCase()}...`);
                const layerEntry = state.layerData.layers[String(state.currentLayer)];
                if (layerEntry) {
                    const targets = layerEntry[method];
                    await scatter.animateToPositions(targets, 300);
                    for (let i = 0; i < state.dataset.points.length; i++) {
                        if (targets[i]) {
                            state.dataset.points[i].position = [...targets[i]];
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
            rebuildClusters();
            minimap.setPoints(dataset.points, scatter.categoryColor);
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
            }
            else {
                state.hiddenCategories.add(category);
            }
            refreshVisibility();
        },
        onAxisSet: (_idx, _poleA, _poleB) => {
            setStatus("Semantic axes require a running backend");
        },
        onAxesReset: () => {
            state.axes = [null, null, null];
            setStatus("Axes reset to reduction coordinates");
        },
        onModeChange: async (mode) => {
            if (mode === state.mode)
                return;
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
                rebuildClusters();
                minimap.setPoints(ds.points, scatter.categoryColor);
                setStatus(`Layers mode: ${data.model} (${data.num_layers} layers)`);
            }
            else {
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
                rebuildClusters();
                minimap.setPoints(dataset.points, scatter.categoryColor);
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
            }
            else {
                startAutoPlay();
            }
        },
        onSpeedChange: (speed) => {
            state.playSpeed = speed;
        },
        onNeighborMode: (mode) => {
            neighborMode = mode;
            if (state.selectedPoint) {
                applySeed(state.selectedPoint.word);
            }
        },
        onClusterClick: (clusterId) => {
            const cluster = clusters.find((c) => c.id === clusterId);
            if (!cluster)
                return;
            const target = new THREE.Vector3(...cluster.centroid);
            const offset = new THREE.Vector3(3, 2, 3);
            flyCamera(target.clone().add(offset), target, 800);
            setStatus(`Cluster: ${cluster.label} (${cluster.points.length} words)`);
        },
        onTourSelect: (tourIndex) => {
            startTour(tourIndex);
        },
        onTourControl: (action) => {
            if (!tourState.tour)
                return;
            switch (action) {
                case "prev":
                    if (tourState.stepIndex > 0) {
                        tourState.stepIndex--;
                        executeTourStep();
                        if (tourState.isPlaying)
                            scheduleTourAdvance();
                    }
                    break;
                case "next":
                    if (tourState.stepIndex < tourState.tour.steps.length - 1) {
                        tourState.stepIndex++;
                        executeTourStep();
                        if (tourState.isPlaying)
                            scheduleTourAdvance();
                    }
                    break;
                case "play-pause":
                    tourState.isPlaying = !tourState.isPlaying;
                    updateTourStep(tourState);
                    if (tourState.isPlaying) {
                        scheduleTourAdvance();
                    }
                    else if (tourAutoTimer) {
                        clearTimeout(tourAutoTimer);
                        tourAutoTimer = null;
                    }
                    break;
                case "stop":
                    stopTour();
                    setStatus("Ready");
                    break;
            }
        },
        onFocusExit: () => {
            exitFocusMode();
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
        }
        else {
            hideHoverLabel();
            container.style.cursor = "default";
        }
    });
    container.addEventListener("click", (e) => {
        const hit = scatter.raycast(e, container);
        if (!hit)
            return;
        // Shift+click: second point for path tracing
        if (e.shiftKey && state.selectedPoint) {
            pathStartPoint = state.selectedPoint;
            const pathResult = findSemanticPath(pathStartPoint, hit, state.dataset.points);
            if (pathResult) {
                pathRenderer.drawPath(pathResult, scatter.categoryColor);
                updateInfoPanelWithPath(pathResult.words, scatter.categoryColor);
                setStatus(`Path: ${pathStartPoint.word} -> ${hit.word} (${pathResult.words.length} steps)`);
            }
            else {
                setStatus("Could not find semantic path (embeddings may be missing)");
            }
            return;
        }
        applySeed(hit.word);
    });
    // Double-click for focus mode
    container.addEventListener("dblclick", (e) => {
        const hit = scatter.raycast(e, container);
        if (hit) {
            enterFocusMode(hit);
        }
    });
    // Escape to exit focus mode
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            if (focusMode) {
                exitFocusMode();
            }
            if (tourState.tour) {
                stopTour();
                setStatus("Ready");
            }
        }
    });
    // -----------------------------------------------------------------------
    // Render loop
    // -----------------------------------------------------------------------
    animate(sceneCtx, () => {
        // Adaptive point sizing based on camera distance
        const camDist = sceneCtx.camera.position.length();
        const baseSize = 0.18;
        const scale = Math.max(0.04, Math.min(baseSize, baseSize * (camDist / 12)));
        scatter.setPointSize(scale);
        // Update minimap every frame
        minimap.update(sceneCtx.camera);
        // Keep selection ring facing camera (handled implicitly)
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
    // Default to semantic if embeddings are available
    const hasEmbeddings = dataset.points.some((p) => p.embedding && p.embedding.length > 0);
    if (hasEmbeddings) {
        neighborMode = "semantic";
    }
    else {
        neighborMode = "spatial";
    }
    scatter.setPoints(dataset.points, dataset.categories);
    buildCategoryList(dataset.categories, scatter.categoryColor, state.hiddenCategories);
    setPointCount(dataset.points.length);
    refreshVisibility();
    // Compute clusters and category centroids
    rebuildClusters();
    // Setup minimap
    minimap.setPoints(dataset.points, scatter.categoryColor);
    if (mock) {
        setStatus("Mock data mode (backend unavailable)");
    }
    else {
        setStatus("Ready");
    }
}
main().catch((err) => {
    console.error("Fatal:", err);
    const statusText = document.getElementById("status-text");
    if (statusText)
        statusText.textContent = `Error: ${err.message}`;
});

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export interface SceneContext {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  raycaster: THREE.Raycaster;
  mouse: THREE.Vector2;
}

export function createScene(container: HTMLElement): SceneContext {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0f);

  // Subtle fog to fade distant points
  scene.fog = new THREE.FogExp2(0x0a0a0f, 0.04);

  const camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.1,
    200,
  );
  camera.position.set(8, 6, 8);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.3;
  controls.maxDistance = 50;

  // Lighting
  const ambient = new THREE.AmbientLight(0x404060, 0.6);
  scene.add(ambient);

  const directional = new THREE.DirectionalLight(0xffffff, 0.8);
  directional.position.set(10, 15, 10);
  scene.add(directional);

  const fill = new THREE.DirectionalLight(0x445566, 0.3);
  fill.position.set(-5, -3, -8);
  scene.add(fill);

  // Subtle grid for orientation
  const grid = new THREE.GridHelper(20, 20, 0x1a1a2e, 0x111122);
  grid.position.y = -0.01;
  scene.add(grid);

  // Axis helper (small, unobtrusive)
  const axes = new THREE.AxesHelper(1.5);
  axes.position.set(-9, 0, -9);
  scene.add(axes);

  const raycaster = new THREE.Raycaster();
  raycaster.params.Points = { threshold: 0.2 };
  const mouse = new THREE.Vector2();

  // Handle resize
  const onResize = () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  };
  window.addEventListener("resize", onResize);

  return { scene, camera, renderer, controls, raycaster, mouse };
}

export function animate(ctx: SceneContext, onFrame?: () => void): void {
  const loop = () => {
    requestAnimationFrame(loop);
    ctx.controls.update();
    if (onFrame) onFrame();
    ctx.renderer.render(ctx.scene, ctx.camera);
  };
  loop();
}

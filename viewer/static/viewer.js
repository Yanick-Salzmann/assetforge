import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { loadTerrain } from "./terrain.js";

const EYE_HEIGHT_M = 1.7;
const HUMAN_HEIGHT_M = 1.8;
const HUMAN_RADIUS_M = 0.22;
const LOD_PATTERN = /_LOD(\d+)$/;
const MODES = ["textured", "splat", "backfaces", "normals", "wireframe"];
const BACKGROUND = 0xb9bcc2;
const BUILDING_KIND = /building/;
const FIGURE_STANDOFF_M = 1.5;
const EYE_CLEARING_M = 10;
const MAX_PIXEL_RATIO = 1.5;
const TERRAIN_NEAR_FRACTION = 0.25;
const TERRAIN_NEAR_MIN_M = 0.1;
const TERRAIN_NEAR_MAX_M = 100;

const canvas = document.getElementById("view");
const assetSelect = document.getElementById("asset");
const terrainSelect = document.getElementById("terrain");
const seaBox = document.getElementById("sea");
const waterTintBox = document.getElementById("water-tint");
const scatterBox = document.getElementById("scatter");
const buildingsBox = document.getElementById("buildings");
const lodSelect = document.getElementById("lod");
const spinBox = document.getElementById("spin");
const humanBox = document.getElementById("human");
const infoList = document.getElementById("info");
const statusLine = document.getElementById("status");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(BACKGROUND);

const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 2000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.autoRotateSpeed = 1.5;

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.8;
pmrem.dispose();
scene.add(new THREE.HemisphereLight(0xf2f4ff, 0x6b665c, 0.6));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0005;
scene.add(sun);
scene.add(sun.target);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(1, 1),
  new THREE.MeshStandardMaterial({ color: 0x8f9188, roughness: 1, envMapIntensity: 0.35 }),
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

let grid = null;

const human = new THREE.Mesh(
  new THREE.CapsuleGeometry(HUMAN_RADIUS_M, HUMAN_HEIGHT_M - 2 * HUMAN_RADIUS_M, 6, 16),
  new THREE.MeshStandardMaterial({ color: 0x4f86b0, roughness: 0.6 }),
);
human.castShadow = true;
scene.add(human);

const loader = new GLTFLoader();
const debugMaterials = {
  normals: new THREE.MeshNormalMaterial(),
  wireframe: new THREE.MeshBasicMaterial({ color: 0x1d1f24, wireframe: true }),
  backface: new THREE.MeshBasicMaterial({ color: 0xff2020, side: THREE.BackSide }),
};

const state = {
  assets: [],
  asset: null,
  root: null,
  meshes: [],
  lod: 0,
  mode: "textured",
  camera: "frame",
  terrains: [],
  terrain: null,
  terrainLoading: null,
};
window.viewerState = { ready: false, asset: null, terrain: null, lod: null, error: null };
const view = { dirty: true, matrix: new THREE.Matrix4(), projection: new THREE.Matrix4() };

function setStatus(text) {
  statusLine.textContent = text;
  invalidate();
}

function readHash() {
  const params = new URLSearchParams(window.location.hash.slice(1));
  return {
    asset: params.get("asset"),
    terrain: params.get("terrain"),
    lod: params.get("lod"),
    mode: params.get("mode"),
    camera: params.get("cam"),
  };
}

function writeHash() {
  if (state.terrain) {
    const params = new URLSearchParams({ terrain: state.terrain.name, mode: state.mode, cam: state.camera });
    history.replaceState(null, "", `#${params}`);
    return;
  }
  if (!state.asset) {
    return;
  }
  const params = new URLSearchParams({
    asset: state.asset.name,
    lod: `LOD${state.lod}`,
    mode: state.mode,
    cam: state.camera,
  });
  history.replaceState(null, "", `#${params}`);
}

function lodIndexOf(object) {
  for (let node = object; node; node = node.parent) {
    const match = LOD_PATTERN.exec(node.name);
    if (match) {
      return Number(match[1]);
    }
  }
  return 0;
}

function disposeRoot() {
  if (!state.root) {
    return;
  }
  scene.remove(state.root);
  state.root.traverse((node) => {
    if (node.isMesh) {
      node.geometry.dispose();
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (const material of materials) {
        if (!Object.values(debugMaterials).includes(material)) {
          material.dispose();
        }
      }
    }
  });
  state.root = null;
  state.meshes = [];
}

function visibleBox() {
  const box = new THREE.Box3();
  for (const entry of state.meshes) {
    if (entry.lod === state.lod) {
      box.expandByObject(entry.mesh);
    }
  }
  return box;
}

function applyLod() {
  for (const entry of state.meshes) {
    entry.mesh.visible = entry.lod === state.lod;
  }
  updateInfo();
}

function applyTerrainMode() {
  const terrain = state.terrain;
  if (state.mode === "normals") {
    terrain.setSurfaceMaterial(debugMaterials.normals);
  } else if (state.mode === "wireframe") {
    terrain.setSurfaceMaterial(debugMaterials.wireframe);
  } else {
    terrain.setSurfaceMaterial(terrain.splatMaterial);
  }
  terrain.uniforms.uFalseColour.value = state.mode === "splat" ? 1 : 0;
  terrain.showBackfaces(state.mode === "backfaces", debugMaterials.backface);
  terrain.sea.visible = seaBox.checked;
  terrain.uniforms.uWaterTint.value = waterTintBox.checked ? 1 : 0;
  terrain.scatter.visible = scatterBox.checked;
  terrain.buildings.group.visible = buildingsBox.checked;
  updateInfo();
}

function applyMode() {
  if (state.terrain) {
    applyTerrainMode();
    return;
  }
  for (const entry of state.meshes) {
    if (state.mode === "normals") {
      entry.mesh.material = debugMaterials.normals;
    } else if (state.mode === "wireframe") {
      entry.mesh.material = debugMaterials.wireframe;
    } else {
      entry.mesh.material = entry.original;
    }
    entry.overlay.visible = state.mode === "backfaces";
  }
}

function fitStage(box) {
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const extent = Math.max(size.x, size.z, 4) * 6;
  ground.scale.set(extent, extent, 1);
  ground.position.set(centre.x, 0, centre.z);
  if (grid) {
    scene.remove(grid);
    grid.dispose();
  }
  const divisions = Math.ceil(extent);
  grid = new THREE.GridHelper(divisions, divisions, 0x55585e, 0x7d8087);
  grid.position.set(Math.round(centre.x), 0.002, Math.round(centre.z));
  scene.add(grid);
  human.position.set(box.max.x + 1 + HUMAN_RADIUS_M, HUMAN_HEIGHT_M / 2, centre.z);
  const reach = Math.max(size.x, size.y, size.z) * 1.5 + 2;
  sun.position.set(centre.x + reach * 0.6, reach * 1.2, centre.z + reach * 0.8);
  sun.target.position.copy(centre);
  const shadow = sun.shadow.camera;
  shadow.left = -reach;
  shadow.right = reach;
  shadow.top = reach;
  shadow.bottom = -reach;
  shadow.near = 0.1;
  shadow.far = reach * 4;
  shadow.updateProjectionMatrix();
}

function terrainStage(terrain) {
  const size = terrain.manifest.world_size_m;
  camera.near = 0.1;
  camera.far = size * 4;
  camera.updateProjectionMatrix();
  scene.fog = new THREE.Fog(BACKGROUND, size * 0.9, size * 3);
  ground.visible = false;
  if (grid) {
    grid.visible = false;
  }
  sun.castShadow = false;
  sun.position.set(size * 0.4, size * 0.6, size * 0.3);
  sun.target.position.set(0, 0, 0);
}

function assetStage() {
  camera.near = 0.05;
  camera.far = 2000;
  camera.updateProjectionMatrix();
  scene.fog = null;
  ground.visible = true;
  if (grid) {
    grid.visible = true;
  }
  sun.castShadow = true;
}

function unlockControls() {
  controls.minPolarAngle = 0;
  controls.maxPolarAngle = Math.PI;
  controls.minDistance = 0;
  controls.maxDistance = Infinity;
  controls.enablePan = true;
  controls.enableZoom = true;
}

function frameTerrain() {
  state.camera = "frame";
  const terrain = state.terrain;
  const size = terrain.manifest.world_size_m;
  const box = terrain.bounds;
  const centre = new THREE.Vector3(0, (box.min.y + box.max.y) / 2, 0);
  const direction = new THREE.Vector3(1, 0.75, 1.2).normalize();
  unlockControls();
  camera.position.copy(centre).addScaledVector(direction, size * 0.95);
  controls.target.copy(centre);
  controls.update();
  human.position.set(terrain.ground.x, terrain.ground.y + HUMAN_HEIGHT_M / 2, terrain.ground.z);
  writeHash();
  updateInfo();
}

function buildingCentre(sites) {
  if (sites.length === 0) {
    return null;
  }
  const centre = new THREE.Vector3();
  for (const site of sites) {
    centre.add(new THREE.Vector3(site.x, site.base, site.z));
  }
  return centre.divideScalar(sites.length);
}

function figureBeside(site) {
  const terrain = state.terrain;
  const outward = new THREE.Vector3(terrain.ground.x - site.x, 0, terrain.ground.z - site.z).normalize();
  const figure = new THREE.Vector3(site.x, 0, site.z).addScaledVector(outward, site.front + FIGURE_STANDOFF_M);
  figure.y = terrain.heightAt(figure.x, figure.z) + HUMAN_HEIGHT_M / 2;
  return figure;
}

function nearestSite(sites, point) {
  let best = null;
  for (const site of sites) {
    if (!best || Math.hypot(site.x - point.x, site.z - point.z) < Math.hypot(best.x - point.x, best.z - point.z)) {
      best = site;
    }
  }
  return best;
}

function frameBuildings() {
  const terrain = state.terrain;
  const sites = terrain.buildings.sites;
  const centre = buildingCentre(sites);
  if (!centre) {
    frameTerrain();
    return;
  }
  state.camera = "buildings";
  let spread = 0;
  for (const site of sites) {
    spread = Math.max(spread, Math.hypot(site.x - centre.x, site.z - centre.z) + site.radius);
  }
  const target = centre.clone();
  target.y += HUMAN_HEIGHT_M * 3;
  const direction = new THREE.Vector3(terrain.ground.x - centre.x, 0, terrain.ground.z - centre.z);
  if (direction.lengthSq() < 1) {
    direction.set(1, 0, 1.2);
  }
  direction.normalize().setY(0.55).normalize();
  unlockControls();
  camera.position.copy(target).addScaledVector(direction, Math.max(spread, 20) * 2.2);
  controls.target.copy(target);
  controls.update();
  human.position.copy(figureBeside(nearestSite(sites, terrain.ground)));
  writeHash();
  updateInfo();
}

function groundTerrain() {
  state.camera = "ground";
  const terrain = state.terrain;
  const sites = terrain.buildings.sites;
  const eye = terrain.ground.clone();
  eye.y += EYE_HEIGHT_M;
  const focus = buildingCentre(sites) ?? terrain.peak;
  const look = focus.clone().sub(terrain.ground).setY(0);
  if (look.lengthSq() < 1) {
    look.set(0, 0, -1);
  }
  look.normalize();
  if (sites.length > 0) {
    human.position.copy(figureBeside(nearestSite(sites, terrain.ground)));
  } else {
    const figure = terrain.ground.clone().addScaledVector(look, 15);
    figure.y = terrain.heightAt(figure.x, figure.z) + HUMAN_HEIGHT_M / 2;
    human.position.copy(figure);
  }
  unlockControls();
  camera.position.copy(eye);
  controls.target.copy(eye).addScaledVector(look, 0.5);
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.minPolarAngle = Math.PI / 2 - 0.9;
  controls.maxPolarAngle = Math.PI / 2 + 0.5;
  controls.update();
  writeHash();
  updateInfo();
}

function frameCamera() {
  if (state.terrain) {
    frameTerrain();
    return;
  }
  state.camera = "frame";
  const box = visibleBox();
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius, HUMAN_HEIGHT_M);
  const direction = new THREE.Vector3(1, 0.55, 1.2).normalize();
  camera.position.copy(sphere.center).addScaledVector(direction, radius * 2.6);
  controls.target.copy(sphere.center);
  unlockControls();
  controls.update();
  writeHash();
}

function groundCamera() {
  if (state.terrain) {
    groundTerrain();
    return;
  }
  state.camera = "ground";
  const box = visibleBox();
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
  const fitHeight = ((box.max.y - EYE_HEIGHT_M) / Math.tan(halfFov)) * 0.95;
  const distance = Math.max(Math.hypot(size.x, size.z) * 1.1, fitHeight, 2.5);
  controls.target.set(centre.x, Math.min(EYE_HEIGHT_M, centre.y), centre.z);
  camera.position.set(centre.x + distance * 0.7, EYE_HEIGHT_M, centre.z + distance * 0.7);
  unlockControls();
  controls.update();
  const polar = controls.getPolarAngle();
  controls.minPolarAngle = polar;
  controls.maxPolarAngle = polar;
  controls.enablePan = false;
  controls.update();
  writeHash();
}

function infoRows(rows) {
  infoList.replaceChildren(
    ...rows.flatMap(([term, value, colour]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      if (colour !== undefined) {
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = `#${colour.toString(16).padStart(6, "0")}`;
        dd.append(swatch);
      }
      dd.append(value);
      return [dt, dd];
    }),
  );
}

function terrainInfo() {
  const terrain = state.terrain;
  const manifest = terrain.manifest;
  const eyeGround = terrain.heightAt(camera.position.x, camera.position.z);
  const rows = [
    ["biome", manifest.splat.biome],
    ["size", `${manifest.world_size_m} m (${manifest.resolution}², ${manifest.metres_per_pixel} m/px)`],
    ["relief", `${manifest.height_range_m} m`],
    ["sea level", `${manifest.water.sea_level_m} m`],
    ["camera", `${(camera.position.y - eyeGround).toFixed(1)} m above ground`],
  ];
  for (const [kind, stats] of Object.entries(terrain.scatterStats)) {
    rows.push([kind, `${stats.count.toLocaleString()} @ ${stats.spacing_m.toFixed(1)} m`]);
  }
  for (const site of terrain.buildings.sites) {
    rows.push([site.name, `footprint relief ${site.relief.toFixed(1)} m`]);
  }
  for (const name of terrain.buildings.failed) {
    rows.push([name, "not placed"]);
  }
  for (const [role, layers] of Object.entries(terrain.missingMaps)) {
    if (layers.length > 0) {
      rows.push([`no ${role}`, layers.join(", ")]);
    }
  }
  if (state.mode === "splat") {
    for (const [layer, colour] of terrain.layerColours) {
      rows.push(["layer", layer, colour]);
    }
  }
  infoRows(rows);
}

function updateInfo() {
  if (state.terrain) {
    terrainInfo();
    return;
  }
  if (!state.asset) {
    infoList.replaceChildren();
    return;
  }
  const box = visibleBox();
  const size = box.getSize(new THREE.Vector3());
  let triangles = 0;
  for (const entry of state.meshes) {
    if (entry.lod === state.lod) {
      const geometry = entry.mesh.geometry;
      triangles += (geometry.index ? geometry.index.count : geometry.attributes.position.count) / 3;
    }
  }
  const rows = [
    ["kind", state.asset.kind ?? "?"],
    ["LOD", `LOD${state.lod}`],
    ["tris", triangles.toLocaleString()],
    ["width", `${size.x.toFixed(2)} m`],
    ["depth", `${size.z.toFixed(2)} m`],
    ["height", `${size.y.toFixed(2)} m`],
  ];
  infoRows(rows);
  window.viewerState.triangles = triangles;
}

function populateLods(lods) {
  lodSelect.replaceChildren(
    ...lods.map((lod) => {
      const option = document.createElement("option");
      option.value = lod;
      option.textContent = lod;
      return option;
    }),
  );
}

async function loadAsset(name, wanted = {}) {
  const asset = state.assets.find((entry) => entry.name === name);
  if (!asset) {
    setStatus(`unknown asset ${name}`);
    return;
  }
  window.viewerState.ready = false;
  setStatus(`loading ${name}…`);
  state.terrainLoading = null;
  unloadTerrain();
  disposeRoot();
  state.asset = asset;
  assetSelect.value = name;
  let gltf;
  try {
    gltf = await loader.loadAsync(asset.glb);
  } catch (error) {
    window.viewerState.error = String(error);
    setStatus(`failed to load ${name}: ${error}`);
    return;
  }
  if (state.asset !== asset) {
    return;
  }
  state.root = gltf.scene;
  const loaded = [];
  state.root.traverse((node) => {
    if (node.isMesh) {
      loaded.push(node);
    }
  });
  for (const node of loaded) {
    node.castShadow = true;
    node.receiveShadow = true;
    const overlay = new THREE.Mesh(node.geometry, debugMaterials.backface);
    overlay.visible = false;
    overlay.raycast = () => {};
    node.add(overlay);
    state.meshes.push({ mesh: node, original: node.material, overlay, lod: lodIndexOf(node) });
  }
  scene.add(state.root);
  const lods = [...new Set(state.meshes.map((entry) => entry.lod))].sort((a, b) => a - b);
  populateLods(lods.map((lod) => `LOD${lod}`));
  const wantedLod = wanted.lod ? Number(wanted.lod.replace("LOD", "")) : 0;
  state.lod = lods.includes(wantedLod) ? wantedLod : lods[0];
  lodSelect.value = `LOD${state.lod}`;
  applyLod();
  applyMode();
  fitStage(visibleBox());
  if ((wanted.camera ?? state.camera) === "ground") {
    groundCamera();
  } else {
    frameCamera();
  }
  setStatus("");
  window.viewerState = { ready: true, asset: name, terrain: null, lod: state.lod, error: null, triangles: window.viewerState.triangles };
}

function footprintOf(asset) {
  const [minX, minY] = asset.bbox_min;
  const [maxX, maxY] = asset.bbox_max;
  const halfWidth = Math.max(-minX, maxX);
  const halfDepth = Math.max(-minY, maxY);
  return { name: asset.name, radius: Math.hypot(halfWidth, halfDepth), front: halfDepth };
}

function disposeBuildings(buildings) {
  buildings.group.traverse((node) => {
    if (node.isMesh) {
      node.geometry.dispose();
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (const material of materials) {
        material.dispose();
      }
    }
  });
}

async function loadBuildings(terrain) {
  const assets = state.assets.filter((asset) => BUILDING_KIND.test(asset.kind ?? "") && asset.bbox_min && asset.bbox_max);
  const footprints = assets.map(footprintOf);
  const sites = terrain.siteBuildings(footprints);
  const group = new THREE.Group();
  group.name = "buildings";
  const placed = [];
  const failed = [];
  for (const [index, asset] of assets.entries()) {
    const site = sites[index];
    if (!site) {
      failed.push(asset.name);
      continue;
    }
    let gltf;
    try {
      gltf = await loader.loadAsync(asset.glb);
    } catch (error) {
      failed.push(asset.name);
      continue;
    }
    const root = gltf.scene;
    root.name = `building_${asset.name}`;
    root.traverse((node) => {
      if (node.isMesh) {
        node.visible = lodIndexOf(node) === 0;
      }
    });
    root.position.set(site.x, site.base, site.z);
    root.rotation.y = site.yaw;
    group.add(root);
    placed.push({ ...site, front: footprints[index].front });
  }
  terrain.clearScatter([...placed, { x: terrain.ground.x, z: terrain.ground.z, radius: EYE_CLEARING_M }]);
  return { group, sites: placed, failed };
}

function unloadTerrain() {
  if (!state.terrain) {
    return;
  }
  scene.remove(state.terrain.group);
  disposeBuildings(state.terrain.buildings);
  state.terrain.dispose();
  state.terrain = null;
  terrainSelect.value = "";
  document.body.classList.remove("terrain-view");
  assetStage();
}

async function showTerrain(name, wanted = {}) {
  const entry = state.terrains.find((terrain) => terrain.name === name);
  if (!entry) {
    setStatus(`unknown terrain ${name}`);
    return;
  }
  window.viewerState = { ready: false, asset: null, terrain: null, lod: null, error: null };
  state.terrainLoading = name;
  terrainSelect.value = name;
  let terrain;
  try {
    terrain = await loadTerrain(entry, renderer, (stage) => setStatus(`loading ${name}: ${stage}…`));
    setStatus(`loading ${name}: buildings…`);
    terrain.buildings = await loadBuildings(terrain);
  } catch (error) {
    window.viewerState.error = String(error);
    setStatus(`failed to load ${name}: ${error}`);
    return;
  }
  if (state.terrainLoading !== name) {
    disposeBuildings(terrain.buildings);
    terrain.dispose();
    return;
  }
  terrain.group.add(terrain.buildings.group);
  unloadTerrain();
  disposeRoot();
  state.asset = null;
  state.terrain = terrain;
  terrainSelect.value = name;
  scene.add(terrain.group);
  document.body.classList.add("terrain-view");
  terrainStage(terrain);
  applyTerrainMode();
  const wantedCamera = wanted.camera ?? state.camera;
  if (wantedCamera === "ground") {
    groundTerrain();
  } else if (wantedCamera === "buildings") {
    frameBuildings();
  } else {
    frameTerrain();
  }
  setStatus("");
  window.viewerState = {
    ready: true,
    asset: null,
    terrain: name,
    lod: null,
    error: null,
    scatter: terrain.scatterStats,
    missingMaps: terrain.missingMaps,
    buildings: terrain.buildings.sites.map(({ name: building, x, z, base, relief, yaw }) => ({ name: building, x, z, base, relief, yaw })),
    unplaced: terrain.buildings.failed,
  };
}

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.floor(width * renderer.getPixelRatio()) || canvas.height !== Math.floor(height * renderer.getPixelRatio())) {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    invalidate();
  }
}

function fitTerrainClip() {
  const clearance = camera.position.y - state.terrain.heightAt(camera.position.x, camera.position.z);
  const near = THREE.MathUtils.clamp(clearance * TERRAIN_NEAR_FRACTION, TERRAIN_NEAR_MIN_M, TERRAIN_NEAR_MAX_M);
  if (near !== camera.near) {
    camera.near = near;
    camera.updateProjectionMatrix();
  }
}

function invalidate() {
  view.dirty = true;
}

function cameraMoved() {
  camera.updateMatrixWorld();
  return !camera.matrixWorld.equals(view.matrix) || !camera.projectionMatrix.equals(view.projection);
}

function tick() {
  requestAnimationFrame(tick);
  resize();
  controls.update();
  if (state.terrain) {
    fitTerrainClip();
  }
  if (!view.dirty && !cameraMoved()) {
    return;
  }
  view.dirty = false;
  view.matrix.copy(camera.matrixWorld);
  view.projection.copy(camera.projectionMatrix);
  renderer.render(scene, camera);
}

assetSelect.addEventListener("change", () => {
  loadAsset(assetSelect.value, { camera: state.camera });
});

terrainSelect.addEventListener("change", () => {
  if (terrainSelect.value) {
    showTerrain(terrainSelect.value, { camera: state.camera });
  } else if (state.assets.length > 0) {
    loadAsset(state.assets[0].name, { camera: state.camera });
  }
});

for (const box of [seaBox, waterTintBox, scatterBox, buildingsBox]) {
  box.addEventListener("change", () => {
    if (state.terrain) {
      applyTerrainMode();
    }
  });
}

lodSelect.addEventListener("change", () => {
  state.lod = Number(lodSelect.value.replace("LOD", ""));
  applyLod();
  writeHash();
});

for (const radio of document.querySelectorAll('input[name="mode"]')) {
  radio.addEventListener("change", () => {
    state.mode = radio.value;
    applyMode();
    writeHash();
  });
}

document.getElementById("frame").addEventListener("click", frameCamera);
document.getElementById("ground").addEventListener("click", groundCamera);
document.getElementById("village").addEventListener("click", () => {
  if (state.terrain) {
    frameBuildings();
  }
});
spinBox.addEventListener("change", () => {
  controls.autoRotate = spinBox.checked;
});
humanBox.addEventListener("change", () => {
  human.visible = humanBox.checked;
});

async function start() {
  const [assetResponse, terrainResponse] = await Promise.all([fetch("/api/assets"), fetch("/api/terrains")]);
  state.assets = await assetResponse.json();
  state.terrains = await terrainResponse.json();
  terrainSelect.append(
    ...state.terrains.map((terrain) => {
      const option = document.createElement("option");
      option.value = terrain.name;
      option.textContent = `${terrain.name} (${terrain.manifest.splat.biome})`;
      return option;
    }),
  );
  if (state.assets.length === 0 && state.terrains.length === 0) {
    setStatus("no exported assets under out/assets/ and no exported terrains under out/terrain/");
    return;
  }
  assetSelect.replaceChildren(
    ...state.assets.map((asset) => {
      const option = document.createElement("option");
      option.value = asset.name;
      option.textContent = asset.kind ? `${asset.name} (${asset.kind})` : asset.name;
      return option;
    }),
  );
  await loadFromHash();
}

async function loadFromHash() {
  const wanted = readHash();
  if (wanted.mode && MODES.includes(wanted.mode)) {
    state.mode = wanted.mode;
    document.querySelector(`input[name="mode"][value="${wanted.mode}"]`).checked = true;
  }
  if (wanted.terrain && state.terrains.some((terrain) => terrain.name === wanted.terrain)) {
    await showTerrain(wanted.terrain, wanted);
    return;
  }
  if (state.assets.length === 0) {
    await showTerrain(state.terrains[0].name, wanted);
    return;
  }
  const first = state.assets.some((asset) => asset.name === wanted.asset) ? wanted.asset : state.assets[0].name;
  await loadAsset(first, wanted);
}

window.addEventListener("hashchange", () => {
  const wanted = readHash();
  const sameTerrain = state.terrain && wanted.terrain === state.terrain.name && wanted.mode === state.mode && wanted.camera === state.camera;
  if (sameTerrain) {
    return;
  }
  const unchanged =
    state.asset &&
    wanted.asset === state.asset.name &&
    wanted.lod === `LOD${state.lod}` &&
    wanted.mode === state.mode &&
    wanted.camera === state.camera;
  if (!unchanged) {
    loadFromHash();
  }
});

function orbitTo(degrees) {
  const offset = camera.position.clone().sub(controls.target);
  const radius = Math.hypot(offset.x, offset.z);
  const radians = THREE.MathUtils.degToRad(degrees);
  camera.position.set(
    controls.target.x + radius * Math.sin(radians),
    camera.position.y,
    controls.target.z + radius * Math.cos(radians),
  );
  controls.update();
}

window.viewerApi = { orbitTo, frameCamera, groundCamera, frameBuildings, camera, controls, scene, renderer, invalidate, terrain: () => state.terrain };

for (const type of ["change", "input", "click"]) {
  document.addEventListener(type, invalidate, true);
}

tick();
start();

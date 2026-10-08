import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const EYE_HEIGHT_M = 1.7;
const HUMAN_HEIGHT_M = 1.8;
const HUMAN_RADIUS_M = 0.22;
const LOD_PATTERN = /_LOD(\d+)$/;

const canvas = document.getElementById("view");
const assetSelect = document.getElementById("asset");
const lodSelect = document.getElementById("lod");
const spinBox = document.getElementById("spin");
const humanBox = document.getElementById("human");
const infoList = document.getElementById("info");
const statusLine = document.getElementById("status");

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xb9bcc2);

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
};
window.viewerState = { ready: false, asset: null, lod: null, error: null };

function setStatus(text) {
  statusLine.textContent = text;
}

function readHash() {
  const params = new URLSearchParams(window.location.hash.slice(1));
  return {
    asset: params.get("asset"),
    lod: params.get("lod"),
    mode: params.get("mode"),
    camera: params.get("cam"),
  };
}

function writeHash() {
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

function applyMode() {
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

function frameCamera() {
  state.camera = "frame";
  const box = visibleBox();
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const radius = Math.max(sphere.radius, HUMAN_HEIGHT_M);
  const direction = new THREE.Vector3(1, 0.55, 1.2).normalize();
  camera.position.copy(sphere.center).addScaledVector(direction, radius * 2.6);
  controls.target.copy(sphere.center);
  controls.minPolarAngle = 0;
  controls.maxPolarAngle = Math.PI;
  controls.enablePan = true;
  controls.update();
  writeHash();
}

function groundCamera() {
  state.camera = "ground";
  const box = visibleBox();
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
  const fitHeight = ((box.max.y - EYE_HEIGHT_M) / Math.tan(halfFov)) * 0.95;
  const distance = Math.max(Math.hypot(size.x, size.z) * 1.1, fitHeight, 2.5);
  controls.target.set(centre.x, Math.min(EYE_HEIGHT_M, centre.y), centre.z);
  camera.position.set(centre.x + distance * 0.7, EYE_HEIGHT_M, centre.z + distance * 0.7);
  controls.minPolarAngle = 0;
  controls.maxPolarAngle = Math.PI;
  controls.update();
  const polar = controls.getPolarAngle();
  controls.minPolarAngle = polar;
  controls.maxPolarAngle = polar;
  controls.enablePan = false;
  controls.update();
  writeHash();
}

function updateInfo() {
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
  infoList.replaceChildren(
    ...rows.flatMap(([term, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      return [dt, dd];
    }),
  );
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
  window.viewerState = { ready: true, asset: name, lod: state.lod, error: null, triangles: window.viewerState.triangles };
}

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.floor(width * renderer.getPixelRatio()) || canvas.height !== Math.floor(height * renderer.getPixelRatio())) {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
}

function tick() {
  resize();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

assetSelect.addEventListener("change", () => {
  loadAsset(assetSelect.value, { camera: state.camera });
});

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
spinBox.addEventListener("change", () => {
  controls.autoRotate = spinBox.checked;
});
humanBox.addEventListener("change", () => {
  human.visible = humanBox.checked;
});

async function start() {
  const response = await fetch("/api/assets");
  state.assets = await response.json();
  if (state.assets.length === 0) {
    setStatus("no exported assets under out/assets/");
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
  if (wanted.mode && ["textured", "backfaces", "normals", "wireframe"].includes(wanted.mode)) {
    state.mode = wanted.mode;
    document.querySelector(`input[name="mode"][value="${wanted.mode}"]`).checked = true;
  }
  const first = state.assets.some((asset) => asset.name === wanted.asset) ? wanted.asset : state.assets[0].name;
  await loadAsset(first, wanted);
}

window.addEventListener("hashchange", () => {
  const wanted = readHash();
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

window.viewerApi = { orbitTo, frameCamera, groundCamera, camera, controls, scene };

tick();
start();

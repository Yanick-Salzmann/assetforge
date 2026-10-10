import * as THREE from "three";

const MAX_GRID_VERTICES = 1025;
const CHUNK_CELLS = 64;
const CHUNK_LOD_STRIDES = [1, 2, 4, 8, 16];
const CHUNK_LOD_REACH = 1.5;
const CHUNK_LOD_HYSTERESIS = 0.1;
const SKIRT_MARGIN_M = 1;
const MAX_SCATTER_CELLS = 4_000_000;
const GROUND_FLAT_REACH_M = 20;
const GROUND_MAX_SLOPE = 0.08;
const LAYER_MAP_SIZE = 1024;
const SITE_GAP_M = 8;
const SITE_EYE_CLEARANCE_M = 14;
const SITE_MAX_RELIEF_M = 2.5;
const SITE_MAX_WET = 0.2;
const SITE_SEA_MARGIN_M = 1;
const SITE_SAMPLES = 9;
const SITE_SCATTER_MARGIN_M = 2;
const LAYER_ROLES = ["albedo", "normal", "roughness"];
const NEUTRAL_TEXEL = {
  albedo: [128, 128, 128, 255],
  normal: [128, 128, 255, 255],
  roughness: [235, 235, 235, 255],
};
const WATER_COLOUR = new THREE.Color(0x2b5d74);
const WATER_SHALLOW = 0x3d8c86;
const WATER_DEEP = 0x0b2c42;
const WATER_SKY_ZENITH = 0x5b7fae;
const WATER_CLARITY_M = 3.5;
const WATER_MIN_WET = 0.01;
const WATER_RENDER_ORDER = 1;
const DEFAULT_MACRO_SCALE = 7.3;
const DEFAULT_BLEND_CONTRAST = 0.3;
const DEFAULT_BLEND_DEPTH = 0.08;
const NEUTRAL_HEIGHT = 128;
const LAYER_COLOURS = [0xd9a441, 0x4f9fd8, 0xe8dca0, 0x8c8478, 0x7fb24a, 0x6a4e42, 0x2e7d4f, 0xf2f2f2];

const SCATTER_KINDS = {
  conifer: { cap: 50000, sink_m: 0.3, tile_m: 768, reach_m: Infinity, parts: coniferParts },
  broadleaf: { cap: 40000, sink_m: 0.3, tile_m: 768, reach_m: Infinity, parts: broadleafParts },
  cactus: { cap: 30000, sink_m: 0.1, tile_m: 384, reach_m: 1500, parts: cactusParts },
  shrub: { cap: 50000, sink_m: 0.1, tile_m: 256, reach_m: 600, parts: shrubParts },
  rock: { cap: 40000, sink_m: 0.35, tile_m: 384, reach_m: 1500, parts: rockParts },
  grass: { cap: 80000, sink_m: 0.02, tile_m: 128, reach_m: 250, parts: grassParts },
  flower: { cap: 60000, sink_m: 0.02, tile_m: 128, reach_m: 200, parts: flowerParts },
  debris: { cap: 40000, sink_m: 0.08, tile_m: 192, reach_m: 400, parts: debrisParts },
};

function coniferParts() {
  const trunk = new THREE.CylinderGeometry(0.22, 0.3, 2.2, 6);
  trunk.translate(0, 1.1, 0);
  const crown = new THREE.ConeGeometry(1.8, 6, 7);
  crown.translate(0, 5, 0);
  return [
    [trunk, new THREE.MeshStandardMaterial({ color: 0x5a4330, roughness: 0.9 })],
    [crown, new THREE.MeshStandardMaterial({ color: 0x3f6b35, roughness: 0.85 })],
  ];
}

function broadleafParts() {
  const trunk = new THREE.CylinderGeometry(0.25, 0.35, 3, 6);
  trunk.translate(0, 1.5, 0);
  const crown = new THREE.IcosahedronGeometry(2.4, 1);
  crown.scale(1, 0.8, 1);
  crown.translate(0, 4.6, 0);
  return [
    [trunk, new THREE.MeshStandardMaterial({ color: 0x5e4a36, roughness: 0.9 })],
    [crown, new THREE.MeshStandardMaterial({ color: 0x55803a, roughness: 0.85, flatShading: true })],
  ];
}

function cactusParts() {
  const stem = new THREE.CylinderGeometry(0.3, 0.35, 4, 8);
  stem.translate(0, 2, 0);
  return [[stem, new THREE.MeshStandardMaterial({ color: 0x5f7d45, roughness: 0.8 })]];
}

function shrubParts() {
  const bush = new THREE.IcosahedronGeometry(0.7, 0);
  bush.scale(1, 0.7, 1);
  bush.translate(0, 0.45, 0);
  return [[bush, new THREE.MeshStandardMaterial({ color: 0x4d6b38, roughness: 0.9, flatShading: true })]];
}

function flowerParts() {
  const stalk = new THREE.ConeGeometry(0.06, 0.35, 4);
  stalk.translate(0, 0.175, 0);
  const bloom = new THREE.IcosahedronGeometry(0.08, 0);
  bloom.translate(0, 0.38, 0);
  return [
    [stalk, new THREE.MeshStandardMaterial({ color: 0x6f8f45, roughness: 0.9 })],
    [bloom, new THREE.MeshStandardMaterial({ color: 0xd8c84a, roughness: 0.8 })],
  ];
}

function rockParts() {
  const rock = new THREE.IcosahedronGeometry(0.8, 0);
  rock.scale(1, 0.6, 1);
  return [[rock, new THREE.MeshStandardMaterial({ color: 0x7d7a74, roughness: 0.95, flatShading: true })]];
}

function grassParts() {
  const tuft = new THREE.ConeGeometry(0.12, 0.45, 5);
  tuft.translate(0, 0.225, 0);
  return [[tuft, new THREE.MeshStandardMaterial({ color: 0x9aa452, roughness: 0.9 })]];
}

function debrisParts() {
  const piece = new THREE.DodecahedronGeometry(0.25, 0);
  return [[piece, new THREE.MeshStandardMaterial({ color: 0x6e655a, roughness: 0.95, flatShading: true })]];
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export async function fetchRaw(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  const width = Number(response.headers.get("X-Width"));
  const height = Number(response.headers.get("X-Height"));
  const channels = Number(response.headers.get("X-Channels"));
  const bits = Number(response.headers.get("X-Bits"));
  const data = bits === 16 ? new Uint16Array(buffer) : new Uint8Array(buffer);
  if (data.length !== width * height * channels) {
    throw new Error(`${url}: ${data.length} samples, expected ${width * height * channels}`);
  }
  return { width, height, channels, bits, data };
}

function dataTexture(raw) {
  const format = raw.channels === 4 ? THREE.RGBAFormat : THREE.RedFormat;
  const texture = new THREE.DataTexture(raw.data, raw.width, raw.height, format, THREE.UnsignedByteType);
  texture.colorSpace = THREE.NoColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}

async function layerPixels(url, size) {
  if (!url) {
    return null;
  }
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }
    const bitmap = await createImageBitmap(await response.blob(), {
      resizeWidth: size,
      resizeHeight: size,
      resizeQuality: "high",
      imageOrientation: "flipY",
      colorSpaceConversion: "none",
      premultiplyAlpha: "none",
    });
    const canvas = new OffscreenCanvas(size, size);
    const context = canvas.getContext("2d");
    context.drawImage(bitmap, 0, 0);
    bitmap.close();
    return context.getImageData(0, 0, size, size).data;
  } catch (error) {
    return null;
  }
}

function packHeight(data, offset, texels, pixels) {
  if (!pixels) {
    for (let index = offset + 1; index < offset + texels; index += 4) {
      data[index] = NEUTRAL_HEIGHT;
    }
    return;
  }
  let low = 255;
  let high = 0;
  for (let index = 0; index < texels; index += 4) {
    low = Math.min(low, pixels[index]);
    high = Math.max(high, pixels[index]);
  }
  const span = Math.max(high - low, 1);
  for (let index = 0; index < texels; index += 4) {
    data[offset + index + 1] = Math.round(((pixels[index] - low) * 255) / span);
  }
}

async function layerArray(urls, role, anisotropy, heightUrls = null) {
  const size = LAYER_MAP_SIZE;
  const texels = size * size * 4;
  const data = new Uint8Array(texels * Math.max(urls.length, 1));
  const missing = [];
  const loaded = await Promise.all(urls.map((url) => layerPixels(url, size)));
  const heights = heightUrls ? await Promise.all(heightUrls.map((url) => layerPixels(url, size))) : null;
  loaded.forEach((pixels, layer) => {
    if (pixels) {
      data.set(pixels, layer * texels);
      return;
    }
    missing.push(layer);
    const fill = NEUTRAL_TEXEL[role];
    for (let offset = layer * texels; offset < (layer + 1) * texels; offset += 4) {
      data.set(fill, offset);
    }
  });
  if (heights) {
    heights.forEach((pixels, layer) => packHeight(data, layer * texels, texels, pixels));
  }
  const texture = new THREE.DataArrayTexture(data, size, size, Math.max(urls.length, 1));
  texture.colorSpace = role === "albedo" ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;
  texture.anisotropy = anisotropy;
  texture.needsUpdate = true;
  return { texture, missing };
}

function bilinear(samples, resolution, px, pz) {
  const last = resolution - 1;
  const x = Math.min(Math.max(px, 0), last);
  const z = Math.min(Math.max(pz, 0), last);
  const x0 = Math.floor(x);
  const z0 = Math.floor(z);
  const x1 = Math.min(x0 + 1, last);
  const z1 = Math.min(z0 + 1, last);
  const fx = x - x0;
  const fz = z - z0;
  const row0 = z0 * resolution;
  const row1 = z1 * resolution;
  const top = samples[row0 + x0] * (1 - fx) + samples[row0 + x1] * fx;
  const bottom = samples[row1 + x0] * (1 - fx) + samples[row1 + x1] * fx;
  return top * (1 - fz) + bottom * fz;
}

class HeightField {
  constructor(raw, manifest) {
    this.resolution = raw.width;
    this.samples = raw.data;
    this.worldSize = manifest.world_size_m;
    this.metresPerPixel = manifest.metres_per_pixel;
    this.range = manifest.height_range_m;
  }

  atPixel(px, pz) {
    return (bilinear(this.samples, this.resolution, px, pz) / 65535) * this.range;
  }

  toPixel(metres) {
    return (metres + this.worldSize / 2) / this.metresPerPixel - 0.5;
  }

  toWorld(pixel) {
    return (pixel + 0.5) * this.metresPerPixel - this.worldSize / 2;
  }

  at(x, z) {
    return this.atPixel(this.toPixel(x), this.toPixel(z));
  }
}

class SurfaceGrid {
  constructor(field) {
    const wanted = Math.min(field.resolution, MAX_GRID_VERTICES) - 1;
    this.cells = Math.max(CHUNK_CELLS, Math.round(wanted / CHUNK_CELLS) * CHUNK_CELLS);
    this.count = this.cells + 1;
    this.step = (field.resolution - 1) / this.cells;
    this.spacing = this.step * field.metresPerPixel;
    this.field = field;
    this.heights = new Float32Array(this.count * this.count);
    for (let j = 0; j < this.count; j += 1) {
      for (let i = 0; i < this.count; i += 1) {
        this.heights[j * this.count + i] = field.atPixel(i * this.step, j * this.step);
      }
    }
  }

  height(i, j) {
    const last = this.count - 1;
    return this.heights[Math.min(Math.max(j, 0), last) * this.count + Math.min(Math.max(i, 0), last)];
  }

  x(i) {
    return this.field.toWorld(i * this.step);
  }

  normal(i, j, target) {
    const dx = (this.height(i + 1, j) - this.height(i - 1, j)) / (2 * this.spacing);
    const dz = (this.height(i, j + 1) - this.height(i, j - 1)) / (2 * this.spacing);
    return target.set(-dx, 1, -dz).normalize();
  }

  edgeError(i0, j0, di, dj, stride) {
    let worst = 0;
    for (let k = 0; k < CHUNK_CELLS; k += stride) {
      const start = this.height(i0 + di * k, j0 + dj * k);
      const end = this.height(i0 + di * (k + stride), j0 + dj * (k + stride));
      for (let m = 1; m < stride; m += 1) {
        const expected = start + ((end - start) * m) / stride;
        worst = Math.max(worst, Math.abs(this.height(i0 + di * (k + m), j0 + dj * (k + m)) - expected));
      }
    }
    return worst;
  }

  skirtDepth() {
    const coarsest = CHUNK_LOD_STRIDES[CHUNK_LOD_STRIDES.length - 1];
    let worst = 0;
    for (let j = 0; j <= this.cells; j += CHUNK_CELLS) {
      for (let i = 0; i < this.cells; i += CHUNK_CELLS) {
        worst = Math.max(worst, this.edgeError(i, j, 1, 0, coarsest), this.edgeError(j, i, 0, 1, coarsest));
      }
    }
    return worst + SKIRT_MARGIN_M;
  }
}

function chunkPerimeter() {
  const n = CHUNK_CELLS;
  const loop = [];
  for (let i = 0; i < n; i += 1) {
    loop.push([i, 0]);
  }
  for (let j = 0; j < n; j += 1) {
    loop.push([n, j]);
  }
  for (let i = n; i > 0; i -= 1) {
    loop.push([i, n]);
  }
  for (let j = n; j > 0; j -= 1) {
    loop.push([0, j]);
  }
  return loop;
}

function chunkIndices(stride, perimeter) {
  const n = CHUNK_CELLS;
  const side = n + 1;
  const cells = n / stride;
  const ring = perimeter.length;
  const indices = new Uint16Array(cells * cells * 6 + (ring / stride) * 6);
  let cursor = 0;
  const push = (...values) => {
    indices.set(values, cursor);
    cursor += values.length;
  };
  for (let j = 0; j < n; j += stride) {
    for (let i = 0; i < n; i += stride) {
      const a = j * side + i;
      const b = a + stride;
      const c = a + stride * side;
      const d = c + stride;
      push(a, c, b, b, c, d);
    }
  }
  const skirtBase = side * side;
  for (let k = 0; k < ring; k += stride) {
    const next = (k + stride) % ring;
    const e0 = perimeter[k][1] * side + perimeter[k][0];
    const e1 = perimeter[next][1] * side + perimeter[next][0];
    push(e0, e1, skirtBase + k, e1, skirtBase + next, skirtBase + k);
  }
  return new THREE.BufferAttribute(indices, 1);
}

function chunkAttributes(grid, ci, cj, perimeter, skirt) {
  const n = CHUNK_CELLS;
  const side = n + 1;
  const i0 = ci * n;
  const j0 = cj * n;
  let low = Infinity;
  let high = -Infinity;
  for (let j = 0; j <= n; j += 1) {
    for (let i = 0; i <= n; i += 1) {
      const elevation = grid.height(i0 + i, j0 + j);
      low = Math.min(low, elevation);
      high = Math.max(high, elevation);
    }
  }
  const centre = new THREE.Vector3((grid.x(i0) + grid.x(i0 + n)) / 2, (low + high) / 2, (grid.x(j0) + grid.x(j0 + n)) / 2);
  const total = side * side + perimeter.length;
  const positions = new Float32Array(total * 3);
  const normals = new Float32Array(total * 3);
  const normal = new THREE.Vector3();
  const write = (index, i, j, drop) => {
    positions[index * 3] = grid.x(i0 + i) - centre.x;
    positions[index * 3 + 1] = grid.height(i0 + i, j0 + j) - drop - centre.y;
    positions[index * 3 + 2] = grid.x(j0 + j) - centre.z;
    grid.normal(i0 + i, j0 + j, normal).toArray(normals, index * 3);
  };
  for (let j = 0; j <= n; j += 1) {
    for (let i = 0; i <= n; i += 1) {
      write(j * side + i, i, j, 0);
    }
  }
  perimeter.forEach(([i, j], k) => write(side * side + k, i, j, skirt));
  return {
    centre,
    position: new THREE.BufferAttribute(positions, 3),
    normal: new THREE.BufferAttribute(normals, 3),
  };
}

function buildSurface(field, material) {
  const grid = new SurfaceGrid(field);
  const perimeter = chunkPerimeter();
  const indices = CHUNK_LOD_STRIDES.map((stride) => chunkIndices(stride, perimeter));
  const skirt = grid.skirtDepth();
  const chunkSize = CHUNK_CELLS * grid.spacing;
  const chunks = grid.cells / CHUNK_CELLS;
  const group = new THREE.Group();
  group.name = "terrain_surface";
  const meshes = [];
  const geometries = [];
  for (let cj = 0; cj < chunks; cj += 1) {
    for (let ci = 0; ci < chunks; ci += 1) {
      const { centre, position, normal } = chunkAttributes(grid, ci, cj, perimeter, skirt);
      const lod = new THREE.LOD();
      lod.name = `terrain_chunk_${ci}_${cj}`;
      lod.position.copy(centre);
      let sphere = null;
      CHUNK_LOD_STRIDES.forEach((stride, level) => {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", position);
        geometry.setAttribute("normal", normal);
        geometry.setIndex(indices[level]);
        if (!sphere) {
          geometry.computeBoundingSphere();
          sphere = geometry.boundingSphere;
        }
        geometry.boundingSphere = sphere.clone();
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = `${lod.name}_lod${level}`;
        const reach = level === 0 ? 0 : chunkSize * CHUNK_LOD_REACH * 2 ** (level - 1);
        lod.addLevel(mesh, reach, CHUNK_LOD_HYSTERESIS);
        meshes.push(mesh);
        geometries.push(geometry);
      });
      group.add(lod);
    }
  }
  let low = Infinity;
  let high = -Infinity;
  for (const elevation of grid.heights) {
    low = Math.min(low, elevation);
    high = Math.max(high, elevation);
  }
  const half = field.worldSize / 2;
  const bounds = new THREE.Box3(new THREE.Vector3(-half, low, -half), new THREE.Vector3(half, high, half));
  return { group, meshes, geometries, bounds, grid };
}

const WATER_VERTEX = `
attribute float aDepth;
attribute float aWet;
varying vec3 vWaterWorld;
varying vec3 vWaterNormal;
varying float vWaterDepth;
varying float vWaterWet;
#include <fog_pars_vertex>
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWaterWorld = world.xyz;
  vWaterNormal = normalize(mat3(modelMatrix) * normal);
  vWaterDepth = aDepth;
  vWaterWet = aWet;
  vec4 mvPosition = viewMatrix * world;
  gl_Position = projectionMatrix * mvPosition;
  #include <fog_vertex>
}
`;

const WATER_FRAGMENT = `
#define WATER_OCTAVES 5
#define WATER_STEEPNESS 0.055
#define WATER_SWELL 0.35
#define WATER_SHORE_FADE_M 0.2
#define WATER_FOAM_DEPTH_M 0.5
#define WATER_RAPIDS_START 0.06
#define WATER_RAPIDS_FULL 0.2
uniform float uTime;
uniform vec3 uSunDirection;
uniform vec3 uSunColour;
uniform vec3 uSkyHorizon;
uniform vec3 uSkyZenith;
uniform vec3 uShallowColour;
uniform vec3 uDeepColour;
uniform float uClarity;
varying vec3 vWaterWorld;
varying vec3 vWaterNormal;
varying float vWaterDepth;
varying float vWaterWet;
#include <fog_pars_fragment>
float waterHash(vec2 p) {
  vec3 q = fract(p.xyx * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}
vec3 waterNoise(vec2 p) {
  vec2 cell = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  vec2 du = 6.0 * f * (1.0 - f);
  float a = waterHash(cell);
  float b = waterHash(cell + vec2(1.0, 0.0));
  float c = waterHash(cell + vec2(0.0, 1.0));
  float d = waterHash(cell + vec2(1.0, 1.0));
  float k = a - b - c + d;
  float value = a + (b - a) * u.x + (c - a) * u.y + k * u.x * u.y;
  vec2 slope = du * (vec2(b - a, c - a) + k * u.yx);
  return vec3(value, slope);
}
vec2 waterRipples(vec2 world, float footprint) {
  vec2 slope = vec2(0.0);
  float frequency = 0.06;
  float angle = 0.4;
  for (int octave = 0; octave < WATER_OCTAVES; octave++) {
    float fade = 1.0 - smoothstep(0.25, 0.75, footprint * frequency);
    vec2 direction = vec2(cos(angle), sin(angle));
    mat2 turn = mat2(direction.x, -direction.y, direction.y, direction.x);
    float speed = WATER_SWELL * sqrt(9.81 / (6.2832 * frequency));
    vec3 wave = waterNoise((turn * world + vec2(speed * uTime, 0.0)) * frequency + float(octave) * 17.0);
    slope += fade * WATER_STEEPNESS * (wave.yz * turn);
    frequency *= 2.3;
    angle += 2.1;
  }
  return slope;
}
void main() {
  float footprint = max(length(fwidth(vWaterWorld.xz)), 1e-4);
  vec3 base = normalize(vWaterNormal);
  vec2 ripple = waterRipples(vWaterWorld.xz, footprint);
  vec3 normal = normalize(base - vec3(ripple.x, 0.0, ripple.y));
  vec3 view = normalize(cameraPosition - vWaterWorld);
  float facing = clamp(dot(normal, view), 0.0, 1.0);
  float fresnel = 0.02 + 0.98 * pow(1.0 - facing, 5.0);
  vec3 reflected = reflect(-view, normal);
  vec3 sky = mix(uSkyHorizon, uSkyZenith, sqrt(clamp(reflected.y, 0.0, 1.0)));
  float sunAlign = max(dot(reflected, uSunDirection), 0.0);
  float sharpness = mix(1200.0, 150.0, smoothstep(0.05, 2.0, footprint));
  float glint = pow(sunAlign, sharpness) * 6.0 + pow(sunAlign, 60.0) * 0.15;
  float depth = max(vWaterDepth, 0.0);
  float murk = 1.0 - exp(-depth / uClarity);
  float lit = 0.4 + 0.6 * max(dot(base, uSunDirection), 0.0);
  vec3 body = mix(uShallowColour, uDeepColour, murk) * uSunColour * lit;
  float grade = length(base.xz) / max(base.y, 1e-3);
  float churn = max(1.0 - smoothstep(0.0, WATER_FOAM_DEPTH_M, depth), smoothstep(WATER_RAPIDS_START, WATER_RAPIDS_FULL, grade));
  float froth = waterNoise(vWaterWorld.xz * 0.9 + vec2(uTime * 0.35, uTime * 0.12)).x;
  froth = 0.5 * froth + 0.5 * waterNoise(vWaterWorld.xz * 2.7 - vec2(uTime * 0.2, uTime * 0.45)).x;
  float foam = smoothstep(0.45, 0.85, churn * (0.5 + 0.7 * froth));
  vec3 colour = mix(body, sky, fresnel) + glint * uSunColour;
  colour = mix(colour, vec3(0.9) * lit, foam);
  float alpha = max(max(mix(0.3, 1.0, murk), fresnel), max(foam, min(glint, 1.0)));
  alpha *= smoothstep(0.0, WATER_SHORE_FADE_M, depth) * clamp(vWaterWet, 0.0, 1.0);
  gl_FragColor = vec4(colour, alpha);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
  #include <fog_fragment>
}
`;

function waterMaterial() {
  const uniforms = {
    ...THREE.UniformsUtils.clone(THREE.UniformsLib.fog),
    uTime: { value: 0 },
    uSunDirection: { value: new THREE.Vector3(0.4, 0.6, 0.3).normalize() },
    uSunColour: { value: new THREE.Color(0xffffff) },
    uSkyHorizon: { value: new THREE.Color(0xb9bcc2) },
    uSkyZenith: { value: new THREE.Color(WATER_SKY_ZENITH) },
    uShallowColour: { value: new THREE.Color(WATER_SHALLOW) },
    uDeepColour: { value: new THREE.Color(WATER_DEEP) },
    uClarity: { value: WATER_CLARITY_M },
  };
  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: WATER_VERTEX,
    fragmentShader: WATER_FRAGMENT,
    transparent: true,
    depthWrite: false,
    fog: true,
  });
  return { material, uniforms };
}

function buildWater(grid, surfaceField, waterRaw, seaLevel, material) {
  const count = grid.count;
  const resolution = grid.field.resolution;
  const total = count * count;
  const positions = new Float32Array(total * 3);
  const depths = new Float32Array(total);
  const wets = new Float32Array(total);
  for (let j = 0; j < count; j += 1) {
    for (let i = 0; i < count; i += 1) {
      const index = j * count + i;
      const terrain = grid.heights[index];
      const px = i * grid.step;
      const pz = j * grid.step;
      const level = surfaceField ? Math.max(surfaceField.atPixel(px, pz), terrain) : Math.max(seaLevel, terrain);
      positions[index * 3] = grid.x(i);
      positions[index * 3 + 1] = level;
      positions[index * 3 + 2] = grid.x(j);
      depths[index] = level - terrain;
      wets[index] = surfaceField ? bilinear(waterRaw.data, resolution, px, pz) / 255 : Number(terrain < seaLevel);
    }
  }
  const cells = [];
  for (let j = 0; j < count - 1; j += 1) {
    for (let i = 0; i < count - 1; i += 1) {
      const a = j * count + i;
      const b = a + 1;
      const c = a + count;
      const d = c + 1;
      if (Math.max(wets[a], wets[b], wets[c], wets[d]) > WATER_MIN_WET) {
        cells.push(a, c, b, b, c, d);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aDepth", new THREE.BufferAttribute(depths, 1));
  geometry.setAttribute("aWet", new THREE.BufferAttribute(wets, 1));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(cells), 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "water_surface";
  mesh.renderOrder = WATER_RENDER_ORDER;
  mesh.raycast = () => {};
  return mesh;
}

const SPLAT_GLSL = `
#define SPLAT_HEX_DENSITY 1.25
#define SPLAT_HEX_SHARPNESS 4.0
#define SPLAT_MACRO_NEAR_TILES 6.0
#define SPLAT_MACRO_FAR_TILES 60.0
#define SPLAT_MACRO_MIX 0.5
#define SPLAT_TRIPLANAR_START 0.15
#define SPLAT_TRIPLANAR_FULL 0.35
#define SPLAT_TRIPLANAR_SHARPNESS 4.0
#define SPLAT_JITTER_PERIOD_TEXELS 2.5
#define SPLAT_JITTER_TEXELS 0.9
#define SPLAT_HEX_CUTOFF 0.05
#define SPLAT_HEX_FADE_START_TILES 30.0
#define SPLAT_HEX_FADE_END_TILES 45.0
#define SPLAT_SIDE_CUTOFF 0.05
#define SPLAT_LAYER_CUTOFF 0.02
#define SPLAT_HEIGHT_FADE_START_M 30.0
#define SPLAT_HEIGHT_FADE_END_M 90.0
struct SplatTap {
  vec3 albedo;
  vec3 normal;
  float roughness;
  float height;
};
SplatTap splatTapPlain(float layer, vec2 uv, vec2 dx, vec2 dy) {
  SplatTap tap;
  tap.albedo = textureGrad(uLayerAlbedo, vec3(uv, layer), dx, dy).rgb;
  tap.normal = textureGrad(uLayerNormal, vec3(uv, layer), dx, dy).xyz * 2.0 - 1.0;
  vec2 surface = textureGrad(uLayerRoughness, vec3(uv, layer), dx, dy).rg;
  tap.roughness = surface.x;
  tap.height = surface.y;
  return tap;
}
vec2 splatHash(vec2 p) {
  return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453);
}
float splatValueNoise(vec2 p) {
  vec2 cell = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float a = splatHash(cell).x;
  float b = splatHash(cell + vec2(1.0, 0.0)).x;
  float c = splatHash(cell + vec2(0.0, 1.0)).x;
  float d = splatHash(cell + vec2(1.0, 1.0)).x;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
vec2 splatWeightUv(vec2 uv, vec2 size) {
  vec2 p = uv * size / SPLAT_JITTER_PERIOD_TEXELS;
  vec2 jitter = vec2(splatValueNoise(p), splatValueNoise(p + vec2(31.7, 17.3))) - 0.5;
  return uv + jitter * (2.0 * SPLAT_JITTER_TEXELS) / size;
}
vec4 splatCubic(sampler2D map, vec2 uv, vec2 size) {
  vec2 st = uv * size - 0.5;
  vec2 base = floor(st);
  vec2 f = st - base;
  vec2 f2 = f * f;
  vec2 f3 = f2 * f;
  vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
  vec2 w3 = f3 / 6.0;
  vec2 g0 = w0 + w1;
  vec2 g1 = w2 + w3;
  vec2 h0 = (base - 0.5 + w1 / g0) / size;
  vec2 h1 = (base + 1.5 + w3 / g1) / size;
  vec4 low = g0.x * texture(map, vec2(h0.x, h0.y)) + g1.x * texture(map, vec2(h1.x, h0.y));
  vec4 high = g0.x * texture(map, vec2(h0.x, h1.y)) + g1.x * texture(map, vec2(h1.x, h1.y));
  return g0.y * low + g1.y * high;
}
SplatTap splatTapScrambled(float layer, vec2 uv, vec2 dx, vec2 dy, vec2 cell) {
  vec2 h = splatHash(cell);
  float angle = h.x * 6.2831853;
  mat2 turn = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
  SplatTap tap = splatTapPlain(layer, turn * uv + h.yx, turn * dx, turn * dy);
  tap.normal.xy = transpose(turn) * tap.normal.xy;
  return tap;
}
SplatTap splatAccumulate(SplatTap sum, SplatTap tap, float weight) {
  sum.albedo += tap.albedo * weight;
  sum.normal += tap.normal * weight;
  sum.roughness += tap.roughness * weight;
  sum.height += tap.height * weight;
  return sum;
}
SplatTap splatBlend3(SplatTap a, SplatTap b, SplatTap c, vec3 w) {
  SplatTap tap;
  tap.albedo = a.albedo * w.x + b.albedo * w.y + c.albedo * w.z;
  tap.normal = a.normal * w.x + b.normal * w.y + c.normal * w.z;
  tap.roughness = a.roughness * w.x + b.roughness * w.y + c.roughness * w.z;
  tap.height = a.height * w.x + b.height * w.y + c.height * w.z;
  return tap;
}
SplatTap splatTapHex(float layer, vec2 uv, vec2 dx, vec2 dy) {
  vec2 skewed = mat2(1.0, 0.0, -0.57735027, 1.15470054) * (uv * SPLAT_HEX_DENSITY);
  vec2 base = floor(skewed);
  vec3 f = vec3(fract(skewed), 0.0);
  f.z = 1.0 - f.x - f.y;
  float s = step(0.0, -f.z);
  float s2 = 2.0 * s - 1.0;
  vec3 w = max(vec3(-f.z * s2, s - f.y * s2, s - f.x * s2), 0.0);
  w = pow(w, vec3(SPLAT_HEX_SHARPNESS));
  w /= max(w.x + w.y + w.z, 1e-5);
  w = max(w - SPLAT_HEX_CUTOFF, 0.0);
  w /= max(w.x + w.y + w.z, 1e-5);
  SplatTap tap = SplatTap(vec3(0.0), vec3(0.0), 0.0, 0.0);
  if (w.x > 0.0) {
    tap = splatAccumulate(tap, splatTapScrambled(layer, uv, dx, dy, base + vec2(s, s)), w.x);
  }
  if (w.y > 0.0) {
    tap = splatAccumulate(tap, splatTapScrambled(layer, uv, dx, dy, base + vec2(s, 1.0 - s)), w.y);
  }
  if (w.z > 0.0) {
    tap = splatAccumulate(tap, splatTapScrambled(layer, uv, dx, dy, base + vec2(1.0 - s, s)), w.z);
  }
  return tap;
}
SplatTap splatTapNear(float layer, vec2 uv, vec2 dx, vec2 dy, float distanceTiles) {
  float single = smoothstep(SPLAT_HEX_FADE_START_TILES, SPLAT_HEX_FADE_END_TILES, distanceTiles);
  SplatTap tap = SplatTap(vec3(0.0), vec3(0.0), 0.0, 0.0);
  if (single < 1.0) {
    tap = splatAccumulate(tap, splatTapHex(layer, uv, dx, dy), 1.0 - single);
  }
  if (single > 0.0) {
    tap = splatAccumulate(tap, splatTapScrambled(layer, uv, dx, dy, vec2(layer, 7.0)), single);
  }
  return tap;
}
SplatTap splatTapAntiTiled(float layer, vec2 uv, vec2 dx, vec2 dy, float macroScale, float distanceTiles) {
  SplatTap near = splatTapNear(layer, uv, dx, dy, distanceTiles);
  float macroMix = SPLAT_MACRO_MIX * smoothstep(SPLAT_MACRO_NEAR_TILES, SPLAT_MACRO_FAR_TILES, distanceTiles);
  if (macroMix <= 0.0 || macroScale <= 1.0) {
    return near;
  }
  SplatTap far = splatTapScrambled(layer, uv / macroScale, dx / macroScale, dy / macroScale, vec2(layer, -7.0));
  SplatTap tap;
  tap.albedo = mix(near.albedo, far.albedo, macroMix);
  tap.normal = mix(near.normal, far.normal, macroMix);
  tap.roughness = mix(near.roughness, far.roughness, macroMix);
  tap.height = mix(near.height, far.height, macroMix);
  return tap;
}
SplatTap splatTapLayer(float layer, vec2 uv, vec2 dx, vec2 dy, bool antiTile, float macroScale, float distanceTiles) {
  if (antiTile) {
    return splatTapAntiTiled(layer, uv, dx, dy, macroScale, distanceTiles);
  }
  return splatTapPlain(layer, uv, dx, dy);
}
vec3 splatSideNormal(vec3 tangentNormal, vec3 axisU, vec3 axisV, vec3 frameT, vec3 frameB) {
  vec3 tilt = tangentNormal.x * axisU + tangentNormal.y * axisV;
  return vec3(dot(tilt, frameT), dot(tilt, frameB), tangentNormal.z);
}
SplatTap splatTriplanar(SplatTap top, SplatTap sideX, SplatTap sideZ, vec2 sideWeight, vec3 frameT, vec3 frameB) {
  sideX.normal = splatSideNormal(sideX.normal, vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), frameT, frameB);
  sideZ.normal = splatSideNormal(sideZ.normal, vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), frameT, frameB);
  return splatBlend3(top, sideX, sideZ, vec3(1.0 - sideWeight.x - sideWeight.y, sideWeight));
}
`;

function splatShader(layers, splatSamplers, hasMacroNormal, hasColourMacro, blendDepth) {
  const macro = hasMacroNormal
    ? "vec3 splatMacroPacked = texture2D(uMacroNormal, vSplatUv).xyz * 2.0 - 1.0;\nvec3 splatMacroWorld = normalize(vec3(splatMacroPacked.x, splatMacroPacked.z, splatMacroPacked.y));"
    : "vec3 splatMacroWorld = normalize(vSplatNormal);";
  const lines = [
    "vec3 splatAlbedo = vec3(0.0);",
    "vec3 splatTangentNormal = vec3(0.0);",
    "float splatRoughness = 0.0;",
    "float splatTotal = 0.0;",
    "float splatWeight;",
    "float splatBlend;",
    "float splatTop = -1e3;",
    "float splatWeightTotal = 0.0;",
    "float splatHeightTotal = 0.0;",
    "SplatTap splatTap;",
    "float splatDistance = length(vViewPosition);",
    "vec2 splatWorldDx = dFdx(vSplatWorld);",
    "vec2 splatWorldDy = dFdy(vSplatWorld);",
    "vec2 splatSideXDx = dFdx(vSplatPosition.zy);",
    "vec2 splatSideXDy = dFdy(vSplatPosition.zy);",
    "vec2 splatSideZDx = dFdx(vSplatPosition.xy);",
    "vec2 splatSideZDy = dFdy(vSplatPosition.xy);",
    macro,
    "vec3 splatFrameRef = normalize(mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), smoothstep(0.7, 0.95, abs(splatMacroWorld.x))));",
    "vec3 splatFrameT = normalize(splatFrameRef - splatMacroWorld * dot(splatFrameRef, splatMacroWorld));",
    "vec3 splatFrameB = cross(splatFrameT, splatMacroWorld);",
    "vec3 splatAxes = pow(abs(splatMacroWorld), vec3(SPLAT_TRIPLANAR_SHARPNESS));",
    "splatAxes /= max(splatAxes.x + splatAxes.y + splatAxes.z, 1e-5);",
    "float splatSteep = smoothstep(SPLAT_TRIPLANAR_START, SPLAT_TRIPLANAR_FULL, 1.0 - abs(splatMacroWorld.y));",
    "vec2 splatSideWeight = max(splatAxes.xz * splatSteep - SPLAT_SIDE_CUTOFF, 0.0) / (1.0 - SPLAT_SIDE_CUTOFF);",
  ];
  lines.push(`vec2 splatMapSize = vec2(textureSize(${splatSamplers[0]}, 0));`);
  lines.push("vec2 splatMapUv = splatWeightUv(vSplatUv, splatMapSize);");
  for (const name of splatSamplers) {
    lines.push(`vec4 ${name}Sample = splatCubic(${name}, splatMapUv, splatMapSize);`);
  }
  layers.forEach((layer, index) => {
    const sampler = `uSplat${layer.texture.replace(/\D/g, "")}`;
    const tiling = Math.max(layer.tiling_m, 0.01).toFixed(4);
    const antiTile = layer.anti_tile ?? true;
    const macroScale = Math.max(layer.macro_scale ?? DEFAULT_MACRO_SCALE, 1).toFixed(4);
    const contrast = Math.max(layer.blend_contrast ?? DEFAULT_BLEND_CONTRAST, 0).toFixed(4);
    lines.push(`float splatWeight${index} = max(${sampler}Sample.${layer.channel} - SPLAT_LAYER_CUTOFF, 0.0);`);
    lines.push(`SplatTap splatTap${index} = SplatTap(vec3(0.0), vec3(0.0), 0.0, 0.0);`);
    lines.push(`float splatScore${index} = -1e3;`);
    lines.push(`if (splatWeight${index} > 0.0) {`);
    lines.push(`  splatWeight = splatWeight${index};`);
    const tail = `${antiTile}, ${macroScale}, splatDistance / ${tiling}`;
    lines.push(`  splatTap = splatTapLayer(${index}.0, vSplatWorld / ${tiling}, splatWorldDx / ${tiling}, splatWorldDy / ${tiling}, ${tail});`);
    lines.push("  if (splatSideWeight.x + splatSideWeight.y > 0.0) {");
    lines.push("    SplatTap splatSideX = splatTap;");
    lines.push("    SplatTap splatSideZ = splatTap;");
    lines.push("    if (splatSideWeight.x > 0.0) {");
    lines.push(
      `      splatSideX = splatTapLayer(${index}.0, vSplatPosition.zy / ${tiling}, splatSideXDx / ${tiling}, splatSideXDy / ${tiling}, ${tail});`,
    );
    lines.push("    }");
    lines.push("    if (splatSideWeight.y > 0.0) {");
    lines.push(
      `      splatSideZ = splatTapLayer(${index}.0, vSplatPosition.xy / ${tiling}, splatSideZDx / ${tiling}, splatSideZDy / ${tiling}, ${tail});`,
    );
    lines.push("    }");
    lines.push("    splatTap = splatTriplanar(splatTap, splatSideX, splatSideZ, splatSideWeight, splatFrameT, splatFrameB);");
    lines.push("  }");
    lines.push(`  splatTap${index} = splatTap;`);
    lines.push(`  splatScore${index} = splatWeight + ${contrast} * splatTap.height;`);
    lines.push(`  splatTop = max(splatTop, splatScore${index});`);
    lines.push("}");
  });
  lines.push(`float splatCut = splatTop - ${Math.max(blendDepth, 0.01).toFixed(4)};`);
  lines.push("float splatHeightFade = uHeightBlend * (1.0 - smoothstep(SPLAT_HEIGHT_FADE_START_M, SPLAT_HEIGHT_FADE_END_M, splatDistance));");
  layers.forEach((layer, index) => {
    lines.push(`float splatHeight${index} = splatWeight${index} > 0.0 ? max(splatScore${index} - splatCut, 0.0) : 0.0;`);
    lines.push(`splatWeightTotal += splatWeight${index};`);
    lines.push(`splatHeightTotal += splatHeight${index};`);
  });
  lines.push("splatWeightTotal = max(splatWeightTotal, 1e-4);");
  lines.push("splatHeightTotal = max(splatHeightTotal, 1e-4);");
  layers.forEach((layer, index) => {
    const colour = new THREE.Color(LAYER_COLOURS[index % LAYER_COLOURS.length]).convertSRGBToLinear();
    lines.push(`if (splatWeight${index} > 0.0) {`);
    lines.push(`  splatBlend = mix(splatWeight${index} / splatWeightTotal, splatHeight${index} / splatHeightTotal, splatHeightFade);`);
    lines.push(
      `  splatAlbedo += splatBlend * mix(splatTap${index}.albedo, vec3(${colour.r.toFixed(4)}, ${colour.g.toFixed(4)}, ${colour.b.toFixed(4)}), uFalseColour);`,
    );
    lines.push(`  splatTangentNormal += splatBlend * splatTap${index}.normal;`);
    lines.push(`  splatRoughness += splatBlend * splatTap${index}.roughness;`);
    lines.push("  splatTotal += splatBlend;");
    lines.push("}");
  });
  lines.push("splatAlbedo /= max(splatTotal, 1e-4);");
  lines.push("splatRoughness = splatTotal > 1e-4 ? splatRoughness / splatTotal : 0.92;");
  lines.push("splatTangentNormal = splatTotal > 1e-4 ? splatTangentNormal / splatTotal : vec3(0.0, 0.0, 1.0);");
  if (hasColourMacro) {
    lines.push("vec3 splatTint = texture2D(uColourMacro, vSplatUv).rgb * (255.0 / 128.0);");
    lines.push("float splatTintStrength = mix(uColourMacroRange.x, 1.0, smoothstep(uColourMacroRange.y, uColourMacroRange.z, splatDistance));");
    lines.push("splatAlbedo *= mix(vec3(1.0), splatTint, splatTintStrength * uColourMacroOn * (1.0 - uFalseColour));");
  }
  lines.push("float waterAmount = smoothstep(0.3, 0.7, texture2D(uWaterMask, vSplatUv).r) * uWaterTint;");
  lines.push("splatAlbedo = mix(splatAlbedo, uWaterColour, waterAmount);");
  lines.push("splatRoughness = mix(splatRoughness, 0.35, waterAmount);");
  lines.push("diffuseColor.rgb *= splatAlbedo;");
  return lines.join("\n");
}

const ROUGHNESS_SHADER = "#include <roughnessmap_fragment>\nroughnessFactor = mix(splatRoughness, 0.92, uFalseColour);";

const NORMAL_SHADER = [
  "#include <normal_fragment_maps>",
  "vec3 detailNormal = splatTangentNormal;",
  "detailNormal.xy *= uDetailNormal * (1.0 - uFalseColour);",
  "detailNormal = normalize(vec3(detailNormal.xy, max(detailNormal.z, 1e-3)));",
  "vec3 shadedWorld = normalize(splatFrameT * detailNormal.x + splatFrameB * detailNormal.y + splatMacroWorld * detailNormal.z);",
  "normal = normalize((viewMatrix * vec4(shadedWorld, 0.0)).xyz);",
].join("\n");

function colourMacroRange(spec) {
  if (!spec) {
    return new THREE.Vector3(0, 0, 1);
  }
  return new THREE.Vector3(spec.near_strength, spec.near_distance_m, spec.far_distance_m);
}

function splatMaterial(manifest, splatTextures, layerArrays, waterMask, macroNormal, colourMacro) {
  const layers = manifest.splat.layers;
  const uniforms = {
    uWorldSize: { value: manifest.world_size_m },
    uFalseColour: { value: 0 },
    uWaterTint: { value: 1 },
    uDetailNormal: { value: 1 },
    uWaterColour: { value: WATER_COLOUR.clone().convertSRGBToLinear() },
    uWaterMask: { value: waterMask },
    uMacroNormal: { value: macroNormal },
    uLayerAlbedo: { value: layerArrays.albedo.texture },
    uLayerNormal: { value: layerArrays.normal.texture },
    uLayerRoughness: { value: layerArrays.roughness.texture },
    uColourMacro: { value: colourMacro },
    uColourMacroOn: { value: 1 },
    uHeightBlend: { value: 1 },
    uColourMacroRange: { value: colourMacroRange(manifest.colour_macro) },
  };
  const splatSamplers = [];
  for (const [name, texture] of Object.entries(splatTextures)) {
    const uniform = `uSplat${name.replace(/\D/g, "")}`;
    uniforms[uniform] = { value: texture };
    splatSamplers.push(uniform);
  }
  const declarations = [
    "uniform float uFalseColour;",
    "uniform float uWaterTint;",
    "uniform float uDetailNormal;",
    "uniform vec3 uWaterColour;",
    "uniform sampler2D uWaterMask;",
    "uniform sampler2D uMacroNormal;",
    "uniform sampler2D uColourMacro;",
    "uniform float uColourMacroOn;",
    "uniform float uHeightBlend;",
    "uniform vec3 uColourMacroRange;",
    "uniform highp sampler2DArray uLayerAlbedo;",
    "uniform highp sampler2DArray uLayerNormal;",
    "uniform highp sampler2DArray uLayerRoughness;",
    ...splatSamplers.map((name) => `uniform sampler2D ${name};`),
    "varying vec2 vSplatUv;",
    "varying vec2 vSplatWorld;",
    "varying vec3 vSplatPosition;",
    "varying vec3 vSplatNormal;",
    SPLAT_GLSL,
  ].join("\n");
  const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.92, metalness: 0, envMapIntensity: 0.4 });
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nuniform float uWorldSize;\nvarying vec2 vSplatUv;\nvarying vec2 vSplatWorld;\nvarying vec3 vSplatPosition;\nvarying vec3 vSplatNormal;",
      )
      .replace(
        "#include <project_vertex>",
        "#include <project_vertex>\nvec4 splatWorldPosition = modelMatrix * vec4(transformed, 1.0);\nvSplatWorld = splatWorldPosition.xz;\nvSplatPosition = splatWorldPosition.xyz;\nvSplatNormal = normalize(mat3(modelMatrix) * objectNormal);\nvSplatUv = (splatWorldPosition.xz + 0.5 * uWorldSize) / uWorldSize;",
      );
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>\n${declarations}`)
      .replace("#include <map_fragment>", splatShader(layers, splatSamplers, macroNormal !== null, colourMacro !== null, manifest.splat.blend_depth ?? DEFAULT_BLEND_DEPTH))
      .replace("#include <roughnessmap_fragment>", ROUGHNESS_SHADER)
      .replace("#include <normal_fragment_maps>", NORMAL_SHADER);
  };
  const layerKey = layers
    .map((layer) => `${layer.tiling_m}/${layer.anti_tile ?? true}/${layer.macro_scale ?? DEFAULT_MACRO_SCALE}/${layer.blend_contrast ?? DEFAULT_BLEND_CONTRAST}`)
    .join(",");
  material.customProgramCacheKey = () => `splat:${manifest.name}:${layerKey}:${macroNormal !== null}:${colourMacro !== null}`;
  return { material, uniforms };
}

function alignedUp(field, x, z, align) {
  if (align <= 0) {
    return [0, 1, 0];
  }
  const step = field.metresPerPixel;
  const dx = (field.at(x + step, z) - field.at(x - step, z)) / (2 * step);
  const dz = (field.at(x, z + step) - field.at(x, z - step)) / (2 * step);
  const nx = -dx * align;
  const nz = -dz * align;
  const length = Math.hypot(nx, 1, nz);
  return [nx / length, 1 / length, nz / length];
}

function scatterPlacements(entry, mask, field, manifest) {
  const spec = SCATTER_KINDS[entry.kind];
  const resolution = mask.width;
  let total = 0;
  for (let index = 0; index < mask.data.length; index += 1) {
    total += mask.data[index];
  }
  const meanDensity = total / (255 * mask.data.length);
  const worldSize = manifest.world_size_m;
  let spacing = entry.spacing_m;
  const expected = meanDensity * (worldSize / spacing) ** 2;
  if (expected > spec.cap) {
    spacing *= Math.sqrt(expected / spec.cap);
  }
  spacing = Math.max(spacing, worldSize / Math.sqrt(MAX_SCATTER_CELLS));
  const cells = Math.floor(worldSize / spacing);
  const random = mulberry32(manifest.seed ^ hashString(entry.species));
  const placements = [];
  for (let j = 0; j < cells; j += 1) {
    for (let i = 0; i < cells; i += 1) {
      const x = (i + random()) * spacing - worldSize / 2;
      const z = (j + random()) * spacing - worldSize / 2;
      const px = Math.min(resolution - 1, Math.max(0, Math.round(field.toPixel(x))));
      const pz = Math.min(resolution - 1, Math.max(0, Math.round(field.toPixel(z))));
      const density = mask.data[pz * resolution + px] / 255;
      const roll = random();
      const yaw = random() * Math.PI * 2;
      const scaleRoll = random();
      if (roll >= density) {
        continue;
      }
      if (placements.length >= spec.cap) {
        continue;
      }
      const scale = entry.scale[0] + (entry.scale[1] - entry.scale[0]) * scaleRoll;
      const [nx, ny, nz] = alignedUp(field, x, z, entry.slope_align);
      placements.push([x, field.at(x, z) - spec.sink_m * scale, z, yaw, scale, nx, ny, nz]);
    }
  }
  return { placements, spacing, meanDensity };
}

function writePlacements(mesh, placements) {
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const position = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  const normal = new THREE.Vector3();
  const tilt = new THREE.Quaternion();
  mesh.count = placements.length;
  placements.forEach(([x, y, z, yaw, size, nx, ny, nz], index) => {
    position.set(x, y, z);
    tilt.setFromUnitVectors(up, normal.set(nx, ny, nz));
    quaternion.setFromAxisAngle(up, yaw).premultiply(tilt);
    scale.setScalar(size);
    matrix.compose(position, quaternion, scale);
    mesh.setMatrixAt(index, matrix);
  });
  mesh.instanceMatrix.needsUpdate = true;
  mesh.computeBoundingSphere();
}

function tileScatter(group) {
  for (const tile of [...group.children]) {
    group.remove(tile);
    for (const mesh of tile.children) {
      mesh.dispose();
    }
  }
  const { kind, placements, parts, worldSize } = group.userData;
  const tiles = Math.max(1, Math.round(worldSize / SCATTER_KINDS[kind].tile_m));
  const tileSize = worldSize / tiles;
  const buckets = new Map();
  for (const placement of placements) {
    const ti = Math.min(tiles - 1, Math.max(0, Math.floor((placement[0] + worldSize / 2) / tileSize)));
    const tj = Math.min(tiles - 1, Math.max(0, Math.floor((placement[2] + worldSize / 2) / tileSize)));
    const key = tj * tiles + ti;
    if (!buckets.has(key)) {
      buckets.set(key, []);
    }
    buckets.get(key).push(placement);
  }
  for (const [key, bucket] of buckets) {
    const tile = new THREE.Group();
    const ti = key % tiles;
    const tj = Math.floor(key / tiles);
    tile.name = `${group.name}_${ti}_${tj}`;
    tile.userData.centre = new THREE.Vector2((ti + 0.5) * tileSize - worldSize / 2, (tj + 0.5) * tileSize - worldSize / 2);
    tile.userData.radius = tileSize * Math.SQRT1_2;
    for (const [geometry, material] of parts) {
      const mesh = new THREE.InstancedMesh(geometry, material, bucket.length);
      writePlacements(mesh, bucket);
      tile.add(mesh);
    }
    group.add(tile);
  }
}

function scatterGroup(species, kind, placements, worldSize) {
  const group = new THREE.Group();
  group.name = `scatter_${species}`;
  group.userData = { species, kind, placements, worldSize, parts: SCATTER_KINDS[kind].parts() };
  tileScatter(group);
  return group;
}

function cullScatter(scatter, eye) {
  for (const group of scatter.children) {
    const reach = SCATTER_KINDS[group.userData.kind].reach_m;
    for (const tile of group.children) {
      const { centre, radius } = tile.userData;
      tile.visible = Math.hypot(eye.x - centre.x, eye.z - centre.y) - radius < reach;
    }
  }
}

function disposeScatter(scatter) {
  for (const group of scatter.children) {
    for (const tile of group.children) {
      for (const mesh of tile.children) {
        mesh.dispose();
      }
    }
    for (const [geometry, material] of group.userData.parts) {
      geometry.dispose();
      material.dispose();
    }
  }
}

function maskAt(mask, field, x, z) {
  const resolution = mask.width;
  const px = Math.min(resolution - 1, Math.max(0, Math.round(field.toPixel(x))));
  const pz = Math.min(resolution - 1, Math.max(0, Math.round(field.toPixel(z))));
  return mask.data[pz * resolution + px] / 255;
}

function surveyFootprint(field, waterRaw, x, z, radius) {
  let low = Infinity;
  let high = -Infinity;
  let wet = 0;
  for (let j = 0; j < SITE_SAMPLES; j += 1) {
    for (let i = 0; i < SITE_SAMPLES; i += 1) {
      const dx = ((2 * i) / (SITE_SAMPLES - 1) - 1) * radius;
      const dz = ((2 * j) / (SITE_SAMPLES - 1) - 1) * radius;
      if (Math.hypot(dx, dz) > radius) {
        continue;
      }
      const elevation = field.at(x + dx, z + dz);
      low = Math.min(low, elevation);
      high = Math.max(high, elevation);
      wet = Math.max(wet, maskAt(waterRaw, field, x + dx, z + dz));
    }
  }
  return { low, high, wet };
}

function siteCandidates(field, ground) {
  const resolution = field.resolution;
  const step = Math.max(1, Math.floor(resolution / 256));
  const candidates = [];
  for (let pz = 0; pz < resolution; pz += step) {
    for (let px = 0; px < resolution; px += step) {
      const x = field.toWorld(px);
      const z = field.toWorld(pz);
      candidates.push([Math.hypot(x - ground.x, z - ground.z), x, z]);
    }
  }
  candidates.sort((a, b) => a[0] - b[0]);
  return candidates;
}

function chooseSite(footprint, candidates, field, waterRaw, seaLevel, ground, taken) {
  const half = field.worldSize / 2;
  let fallback = null;
  for (const [distance, x, z] of candidates) {
    if (distance < footprint.radius + SITE_EYE_CLEARANCE_M) {
      continue;
    }
    if (Math.abs(x) + footprint.radius > half || Math.abs(z) + footprint.radius > half) {
      continue;
    }
    if (taken.some((site) => Math.hypot(site.x - x, site.z - z) < site.radius + footprint.radius + SITE_GAP_M)) {
      continue;
    }
    const survey = surveyFootprint(field, waterRaw, x, z, footprint.radius);
    if (survey.wet > SITE_MAX_WET || survey.low < seaLevel + SITE_SEA_MARGIN_M) {
      continue;
    }
    const relief = survey.high - survey.low;
    const site = {
      name: footprint.name,
      x,
      z,
      base: survey.low,
      relief,
      radius: footprint.radius,
      yaw: Math.atan2(-(ground.x - x), -(ground.z - z)),
    };
    if (relief <= SITE_MAX_RELIEF_M) {
      return site;
    }
    if (!fallback || relief < fallback.relief) {
      fallback = site;
    }
  }
  return fallback;
}

function buildingSites(footprints, field, waterRaw, seaLevel, ground) {
  const candidates = siteCandidates(field, ground);
  const order = [...footprints].sort((a, b) => b.radius - a.radius);
  const taken = [];
  for (const footprint of order) {
    const site = chooseSite(footprint, candidates, field, waterRaw, seaLevel, ground, taken);
    if (site) {
      taken.push(site);
    }
  }
  return footprints.map((footprint) => taken.find((site) => site.name === footprint.name) ?? null);
}

function clearScatter(scatter, scatterStats, sites) {
  let cleared = 0;
  for (const group of scatter.children) {
    const placements = group.userData.placements;
    const kept = placements.filter(
      ([x, , z]) => !sites.some((site) => Math.hypot(site.x - x, site.z - z) < site.radius + SITE_SCATTER_MARGIN_M),
    );
    cleared += placements.length - kept.length;
    group.userData.placements = kept;
    scatterStats[group.userData.species].count = kept.length;
    tileScatter(group);
  }
  return cleared;
}

function groundPoint(field, waterMask, seaLevel) {
  const resolution = field.resolution;
  const centre = (resolution - 1) / 2;
  const step = Math.max(1, Math.floor(resolution / 256));
  let best = null;
  let bestDistance = Infinity;
  for (let pz = 0; pz < resolution; pz += step) {
    for (let px = 0; px < resolution; px += step) {
      const distance = Math.hypot(px - centre, pz - centre);
      if (distance >= bestDistance) {
        continue;
      }
      const elevation = field.atPixel(px, pz);
      const wet = waterMask.data[pz * resolution + px] / 255;
      if (elevation < seaLevel + 2 || wet > 0.2) {
        continue;
      }
      const reach = Math.max(2, Math.ceil(GROUND_FLAT_REACH_M / field.metresPerPixel));
      const slopeX = Math.abs(field.atPixel(px + reach, pz) - field.atPixel(px - reach, pz));
      const slopeZ = Math.abs(field.atPixel(px, pz + reach) - field.atPixel(px, pz - reach));
      if (Math.max(slopeX, slopeZ) / (2 * reach * field.metresPerPixel) > GROUND_MAX_SLOPE) {
        continue;
      }
      best = [px, pz];
      bestDistance = distance;
    }
  }
  const [px, pz] = best ?? [centre, centre];
  return new THREE.Vector3(field.toWorld(px), field.atPixel(px, pz), field.toWorld(pz));
}

function highestPoint(field) {
  let best = 0;
  for (let index = 1; index < field.samples.length; index += 1) {
    if (field.samples[index] > field.samples[best]) {
      best = index;
    }
  }
  const px = best % field.resolution;
  const pz = Math.floor(best / field.resolution);
  return new THREE.Vector3(field.toWorld(px), field.atPixel(px, pz), field.toWorld(pz));
}

export async function loadTerrain(entry, renderer, onProgress = () => {}) {
  const manifest = entry.manifest;
  onProgress("height");
  const heightRaw = await fetchRaw(`${entry.raw}${manifest.heightmap}`);
  if (heightRaw.bits !== 16 || heightRaw.width !== manifest.resolution) {
    throw new Error(`${manifest.heightmap} must be ${manifest.resolution}² 16-bit, got ${heightRaw.width}² ${heightRaw.bits}-bit`);
  }
  const field = new HeightField(heightRaw, manifest);
  onProgress("masks");
  const waterRaw = await fetchRaw(`${entry.raw}${manifest.water_mask}`);
  const splatRaws = {};
  for (const texture of manifest.splat.textures) {
    splatRaws[texture] = await fetchRaw(`${entry.raw}${texture}`);
  }
  const surfaceRaw = manifest.water_surface ? await fetchRaw(`${entry.raw}${manifest.water_surface}`) : null;
  const normalRaw = manifest.normal_map ? await fetchRaw(`${entry.raw}${manifest.normal_map}`) : null;
  const colourMacroRaw = manifest.colour_macro ? await fetchRaw(`${entry.raw}${manifest.colour_macro.path}`) : null;
  onProgress("materials");
  const anisotropy = renderer.capabilities.getMaxAnisotropy();
  const layerArrays = {};
  for (const role of LAYER_ROLES) {
    const urls = manifest.splat.layers.map((layer) => entry.materials[layer.material]?.[role]);
    const heightUrls = role === "roughness" ? manifest.splat.layers.map((layer) => entry.materials[layer.material]?.height) : null;
    layerArrays[role] = await layerArray(urls, role, anisotropy, heightUrls);
  }
  onProgress("mesh");
  const splatTextures = Object.fromEntries(Object.entries(splatRaws).map(([name, raw]) => [name, dataTexture(raw)]));
  const waterMask = dataTexture(waterRaw);
  const macroNormal = normalRaw ? dataTexture(normalRaw) : null;
  const colourMacro = colourMacroRaw ? dataTexture(colourMacroRaw) : null;
  const { material, uniforms } = splatMaterial(manifest, splatTextures, layerArrays, waterMask, macroNormal, colourMacro);
  const surface = buildSurface(field, material);
  const group = new THREE.Group();
  group.name = `terrain_${manifest.name}`;
  group.add(surface.group);

  const seaLevel = manifest.water.sea_level_m;
  const surfaceField = surfaceRaw ? new HeightField(surfaceRaw, manifest) : null;
  const { material: waterShader, uniforms: waterUniforms } = waterMaterial();
  const water = buildWater(surface.grid, surfaceField, waterRaw, seaLevel, waterShader);
  group.add(water);

  onProgress("scatter");
  const scatter = new THREE.Group();
  scatter.name = "scatter";
  const scatterStats = {};
  for (const mask of manifest.scatter) {
    if (!(mask.kind in SCATTER_KINDS)) {
      continue;
    }
    const raw = await fetchRaw(`${entry.raw}${mask.path}`);
    const { placements, spacing, meanDensity } = scatterPlacements(mask, raw, field, manifest);
    scatter.add(scatterGroup(mask.species, mask.kind, placements, manifest.world_size_m));
    scatterStats[mask.species] = { kind: mask.kind, count: placements.length, spacing_m: spacing, mean_density: meanDensity };
  }
  group.add(scatter);

  const ground = groundPoint(field, waterRaw, seaLevel);
  const peak = highestPoint(field);

  const disposables = [
    ...surface.geometries,
    material,
    water.geometry,
    waterShader,
    waterMask,
    ...(macroNormal ? [macroNormal] : []),
    ...Object.values(layerArrays).map((array) => array.texture),
    ...Object.values(splatTextures),
  ];

  return {
    name: manifest.name,
    manifest,
    group,
    surface: surface.group,
    bounds: surface.bounds,
    setSurfaceMaterial(surfaceMaterial) {
      for (const mesh of surface.meshes) {
        mesh.material = surfaceMaterial;
      }
    },
    showBackfaces(visible, backfaceMaterial) {
      for (const mesh of surface.meshes) {
        let overlay = mesh.userData.overlay;
        if (!overlay && visible) {
          overlay = new THREE.Mesh(mesh.geometry, backfaceMaterial);
          overlay.raycast = () => {};
          mesh.userData.overlay = overlay;
          mesh.add(overlay);
        }
        if (overlay) {
          overlay.visible = visible;
        }
      }
    },
    water,
    waterUniforms,
    scatter,
    field,
    ground,
    peak,
    scatterStats,
    missingMaps: Object.fromEntries(
      LAYER_ROLES.map((role) => [role, layerArrays[role].missing.map((index) => manifest.splat.layers[index].layer)]),
    ),
    uniforms,
    splatMaterial: material,
    layerColours: manifest.splat.layers.map((layer, index) => [layer.layer, LAYER_COLOURS[index % LAYER_COLOURS.length]]),
    heightAt: (x, z) => field.at(x, z),
    siteBuildings(footprints) {
      return buildingSites(footprints, field, waterRaw, seaLevel, ground);
    },
    clearScatter(sites) {
      return clearScatter(scatter, scatterStats, sites);
    },
    cullScatter(eye) {
      cullScatter(scatter, eye);
    },
    dispose() {
      disposeScatter(scatter);
      for (const item of disposables) {
        item.dispose();
      }
    },
  };
}

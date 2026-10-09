import * as THREE from "three";

const MAX_GRID_VERTICES = 1025;
const MAX_SCATTER_CELLS = 4_000_000;
const GROUND_FLAT_REACH_M = 20;
const GROUND_MAX_SLOPE = 0.08;
const WATER_COLOUR = new THREE.Color(0x2b5d74);
const LAYER_COLOURS = [0xd9a441, 0x4f9fd8, 0xe8dca0, 0x8c8478, 0x7fb24a, 0x6a4e42, 0x2e7d4f, 0xf2f2f2];

const SCATTER_KINDS = {
  tree: { spacing_m: 6, cap: 60000, sink_m: 0.3, scale: [0.7, 1.4], parts: treeParts },
  rock: { spacing_m: 5, cap: 40000, sink_m: 0.35, scale: [0.6, 2.2], parts: rockParts },
  grass: { spacing_m: 1.5, cap: 80000, sink_m: 0.02, scale: [0.7, 1.3], parts: grassParts },
  debris: { spacing_m: 3, cap: 40000, sink_m: 0.08, scale: [0.6, 1.6], parts: debrisParts },
};

function treeParts() {
  const trunk = new THREE.CylinderGeometry(0.22, 0.3, 2.2, 6);
  trunk.translate(0, 1.1, 0);
  const crown = new THREE.ConeGeometry(1.8, 6, 7);
  crown.translate(0, 5, 0);
  return [
    [trunk, new THREE.MeshStandardMaterial({ color: 0x5a4330, roughness: 0.9 })],
    [crown, new THREE.MeshStandardMaterial({ color: 0x3f6b35, roughness: 0.85 })],
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

function neutralTexture() {
  const texture = new THREE.DataTexture(new Uint8Array([128, 128, 128, 255]), 1, 1);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

async function layerTexture(loader, url, anisotropy) {
  if (!url) {
    return neutralTexture();
  }
  try {
    const texture = await loader.loadAsync(url);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.anisotropy = anisotropy;
    return texture;
  } catch (error) {
    return neutralTexture();
  }
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
    const last = this.resolution - 1;
    const x = Math.min(Math.max(px, 0), last);
    const z = Math.min(Math.max(pz, 0), last);
    const x0 = Math.floor(x);
    const z0 = Math.floor(z);
    const x1 = Math.min(x0 + 1, last);
    const z1 = Math.min(z0 + 1, last);
    const fx = x - x0;
    const fz = z - z0;
    const row0 = z0 * this.resolution;
    const row1 = z1 * this.resolution;
    const top = this.samples[row0 + x0] * (1 - fx) + this.samples[row0 + x1] * fx;
    const bottom = this.samples[row1 + x0] * (1 - fx) + this.samples[row1 + x1] * fx;
    return ((top * (1 - fz) + bottom * fz) / 65535) * this.range;
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

function buildGeometry(field) {
  const count = Math.min(field.resolution, MAX_GRID_VERTICES);
  const step = (field.resolution - 1) / (count - 1);
  const positions = new Float32Array(count * count * 3);
  for (let j = 0; j < count; j += 1) {
    const pz = j * step;
    for (let i = 0; i < count; i += 1) {
      const px = i * step;
      const offset = (j * count + i) * 3;
      positions[offset] = field.toWorld(px);
      positions[offset + 1] = field.atPixel(px, pz);
      positions[offset + 2] = field.toWorld(pz);
    }
  }
  const indices = new Uint32Array((count - 1) * (count - 1) * 6);
  let cursor = 0;
  for (let j = 0; j < count - 1; j += 1) {
    for (let i = 0; i < count - 1; i += 1) {
      const a = j * count + i;
      const b = a + 1;
      const c = a + count;
      const d = c + 1;
      indices[cursor] = a;
      indices[cursor + 1] = c;
      indices[cursor + 2] = b;
      indices[cursor + 3] = b;
      indices[cursor + 4] = c;
      indices[cursor + 5] = d;
      cursor += 6;
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

function splatShader(layers, textureIndex) {
  const lines = ["vec3 splatAlbedo = vec3(0.0);", "float splatTotal = 0.0;", "float splatWeight;"];
  for (const name of Object.keys(textureIndex)) {
    lines.push(`vec4 ${name}Sample = texture2D(${name}, vSplatUv);`);
  }
  layers.forEach((layer, index) => {
    const sampler = `uSplat${layer.texture.replace(/\D/g, "")}`;
    const colour = new THREE.Color(LAYER_COLOURS[index % LAYER_COLOURS.length]).convertSRGBToLinear();
    const tiling = Math.max(layer.tiling_m, 0.01).toFixed(4);
    lines.push(`splatWeight = ${sampler}Sample.${layer.channel};`);
    lines.push(
      `splatAlbedo += splatWeight * mix(texture2D(uLayer${index}, vSplatWorld / ${tiling}).rgb, vec3(${colour.r.toFixed(4)}, ${colour.g.toFixed(4)}, ${colour.b.toFixed(4)}), uFalseColour);`,
    );
    lines.push("splatTotal += splatWeight;");
  });
  lines.push("splatAlbedo /= max(splatTotal, 1e-4);");
  lines.push("float waterAmount = smoothstep(0.3, 0.7, texture2D(uWaterMask, vSplatUv).r) * uWaterTint;");
  lines.push("splatAlbedo = mix(splatAlbedo, uWaterColour, waterAmount);");
  lines.push("diffuseColor.rgb *= splatAlbedo;");
  return lines.join("\n");
}

function splatMaterial(manifest, splatTextures, layerTextures, waterMask) {
  const layers = manifest.splat.layers;
  const uniforms = {
    uWorldSize: { value: manifest.world_size_m },
    uFalseColour: { value: 0 },
    uWaterTint: { value: 1 },
    uWaterColour: { value: WATER_COLOUR.clone().convertSRGBToLinear() },
    uWaterMask: { value: waterMask },
  };
  const textureIndex = {};
  for (const [name, texture] of Object.entries(splatTextures)) {
    const uniform = `uSplat${name.replace(/\D/g, "")}`;
    uniforms[uniform] = { value: texture };
    textureIndex[uniform] = true;
  }
  layerTextures.forEach((texture, index) => {
    uniforms[`uLayer${index}`] = { value: texture };
  });
  const material = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.92, metalness: 0, envMapIntensity: 0.4 });
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    const samplerDeclarations = [
      ...Object.keys(textureIndex).map((name) => `uniform sampler2D ${name};`),
      ...layerTextures.map((_, index) => `uniform sampler2D uLayer${index};`),
    ].join("\n");
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nuniform float uWorldSize;\nvarying vec2 vSplatUv;\nvarying vec2 vSplatWorld;",
      )
      .replace(
        "#include <project_vertex>",
        "#include <project_vertex>\nvec4 splatWorldPosition = modelMatrix * vec4(transformed, 1.0);\nvSplatWorld = splatWorldPosition.xz;\nvSplatUv = (splatWorldPosition.xz + 0.5 * uWorldSize) / uWorldSize;",
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>\nuniform float uFalseColour;\nuniform float uWaterTint;\nuniform vec3 uWaterColour;\nuniform sampler2D uWaterMask;\n${samplerDeclarations}\nvarying vec2 vSplatUv;\nvarying vec2 vSplatWorld;`,
      )
      .replace("#include <map_fragment>", splatShader(layers, textureIndex));
  };
  material.customProgramCacheKey = () => `splat:${manifest.name}:${layers.length}`;
  return { material, uniforms };
}

function scatterPlacements(kind, mask, field, manifest) {
  const spec = SCATTER_KINDS[kind];
  const resolution = mask.width;
  let total = 0;
  for (let index = 0; index < mask.data.length; index += 1) {
    total += mask.data[index];
  }
  const meanDensity = total / (255 * mask.data.length);
  const worldSize = manifest.world_size_m;
  let spacing = spec.spacing_m;
  const expected = meanDensity * (worldSize / spacing) ** 2;
  if (expected > spec.cap) {
    spacing *= Math.sqrt(expected / spec.cap);
  }
  spacing = Math.max(spacing, worldSize / Math.sqrt(MAX_SCATTER_CELLS));
  const cells = Math.floor(worldSize / spacing);
  const random = mulberry32(manifest.seed ^ hashString(kind));
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
      const scale = spec.scale[0] + (spec.scale[1] - spec.scale[0]) * scaleRoll;
      placements.push([x, field.at(x, z) - spec.sink_m * scale, z, yaw, scale]);
    }
  }
  return { placements, spacing, meanDensity };
}

function scatterGroup(kind, placements) {
  const group = new THREE.Group();
  group.name = `scatter_${kind}`;
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const position = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  for (const [geometry, material] of SCATTER_KINDS[kind].parts()) {
    const mesh = new THREE.InstancedMesh(geometry, material, Math.max(placements.length, 1));
    mesh.count = placements.length;
    placements.forEach(([x, y, z, yaw, size], index) => {
      position.set(x, y, z);
      quaternion.setFromAxisAngle(up, yaw);
      scale.setScalar(size);
      matrix.compose(position, quaternion, scale);
      mesh.setMatrixAt(index, matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
    mesh.computeBoundingSphere();
    group.add(mesh);
  }
  return group;
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
  onProgress("materials");
  const loader = new THREE.TextureLoader();
  const anisotropy = renderer.capabilities.getMaxAnisotropy();
  const layerTextures = await Promise.all(
    manifest.splat.layers.map((layer) => layerTexture(loader, entry.materials[layer.material]?.albedo, anisotropy)),
  );
  onProgress("mesh");
  const splatTextures = Object.fromEntries(Object.entries(splatRaws).map(([name, raw]) => [name, dataTexture(raw)]));
  const waterMask = dataTexture(waterRaw);
  const { material, uniforms } = splatMaterial(manifest, splatTextures, layerTextures, waterMask);
  const geometry = buildGeometry(field);
  const group = new THREE.Group();
  group.name = `terrain_${manifest.name}`;
  const surface = new THREE.Mesh(geometry, material);
  surface.name = "terrain_surface";
  group.add(surface);

  const seaLevel = manifest.water.sea_level_m;
  const sea = new THREE.Mesh(
    new THREE.PlaneGeometry(manifest.world_size_m, manifest.world_size_m),
    new THREE.MeshStandardMaterial({ color: WATER_COLOUR, roughness: 0.08, metalness: 0, transparent: true, opacity: 0.82 }),
  );
  sea.name = "sea_level";
  sea.rotation.x = -Math.PI / 2;
  sea.position.y = seaLevel;
  group.add(sea);

  onProgress("scatter");
  const scatter = new THREE.Group();
  scatter.name = "scatter";
  const scatterStats = {};
  for (const mask of manifest.scatter) {
    if (!(mask.kind in SCATTER_KINDS)) {
      continue;
    }
    const raw = await fetchRaw(`${entry.raw}${mask.path}`);
    const { placements, spacing, meanDensity } = scatterPlacements(mask.kind, raw, field, manifest);
    scatter.add(scatterGroup(mask.kind, placements));
    scatterStats[mask.kind] = { count: placements.length, spacing_m: spacing, mean_density: meanDensity };
  }
  group.add(scatter);

  const ground = groundPoint(field, waterRaw, seaLevel);
  const peak = highestPoint(field);

  const disposables = [geometry, material, sea.geometry, sea.material, waterMask, ...layerTextures, ...Object.values(splatTextures)];
  scatter.traverse((node) => {
    if (node.isMesh) {
      disposables.push(node.geometry, node.material, node);
    }
  });

  return {
    name: manifest.name,
    manifest,
    group,
    surface,
    sea,
    scatter,
    field,
    ground,
    peak,
    scatterStats,
    uniforms,
    splatMaterial: material,
    layerColours: manifest.splat.layers.map((layer, index) => [layer.layer, LAYER_COLOURS[index % LAYER_COLOURS.length]]),
    heightAt: (x, z) => field.at(x, z),
    dispose() {
      for (const item of disposables) {
        item.dispose();
      }
    },
  };
}

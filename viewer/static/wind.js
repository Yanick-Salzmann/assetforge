import * as THREE from "three";

export const DEFAULT_WIND = { from_deg: 225, strength: 1, gust_period_s: 6 };
const BONE_INDEX = /_bone(\d+)$/;
const POLE_CROSS_MIN = 0.9;
const POLE_OFFSET_M = 0.032;
const MIN_CLIP_SPEED = 0.25;
const GUST_WAVELENGTH_M = 60;
const SWAY_KINDS = {
  conifer: { lean: 0.025, height_m: 12, flutter_hz: 0.35 },
  broadleaf: { lean: 0.035, height_m: 9, flutter_hz: 0.45 },
  shrub: { lean: 0.06, height_m: 1.5, flutter_hz: 0.9 },
  grass: { lean: 0.18, height_m: 0.6, flutter_hz: 1.6 },
  flower: { lean: 0.14, height_m: 0.5, flutter_hz: 1.4 },
};

const SWAY_VERTEX = `
float swayHeight = max(transformed.y, 0.0);
vec3 swayWorldDir = vec3(uWindDir.x, 0.0, uWindDir.y);
#ifdef USE_INSTANCING
mat3 swayBasis = mat3(instanceMatrix);
vec2 swayOrigin = instanceMatrix[3].xz;
vec3 swayDir = normalize(transpose(swayBasis) * swayWorldDir);
#else
vec2 swayOrigin = vec2(0.0);
vec3 swayDir = swayWorldDir;
#endif
float swayTravel = dot(swayOrigin, uWindDir) / ${GUST_WAVELENGTH_M.toFixed(1)};
float swayGust = 0.5 + 0.5 * sin(6.2831853 * (uWindTime / uWindGust - swayTravel));
float swayPhase = fract(sin(dot(swayOrigin, vec2(12.9898, 78.233))) * 43758.5453) * 6.2831853;
float swayFlutter = 0.25 * sin(6.2831853 * uSwayFlutter * uWindTime + swayPhase);
float swayAmount = uSwayLean * uWindStrength * swayHeight * (swayHeight / uSwayHeight) * (0.55 + 0.45 * swayGust + swayFlutter);
transformed += swayDir * swayAmount;
transformed.y -= 0.5 * swayAmount * swayAmount / max(swayHeight, 0.01);
`;

export class Wind {
  constructor(entry) {
    const settings = { ...DEFAULT_WIND, ...(entry ?? {}) };
    this.fromDeg = settings.from_deg;
    this.strength = settings.strength;
    this.gustPeriod = settings.gust_period_s;
    const bearing = THREE.MathUtils.degToRad(this.fromDeg);
    this.downwind = new THREE.Vector3(-Math.sin(bearing), 0, Math.cos(bearing));
    this.uniforms = {
      uWindTime: { value: 0 },
      uWindDir: { value: new THREE.Vector2(this.downwind.x, this.downwind.z) },
      uWindStrength: { value: this.strength },
      uWindGust: { value: this.gustPeriod },
    };
  }

  update(seconds) {
    this.uniforms.uWindTime.value = seconds;
  }

  get clipSpeed() {
    return Math.max(MIN_CLIP_SPEED, this.strength);
  }

  snapshot() {
    return {
      from_deg: this.fromDeg,
      strength: this.strength,
      gust_period_s: this.gustPeriod,
      downwind: [this.downwind.x, this.downwind.z],
    };
  }
}

export function swayMaterial(material, kind, wind) {
  const sway = SWAY_KINDS[kind];
  if (!sway || !wind) {
    return material;
  }
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, wind.uniforms, {
      uSwayLean: { value: sway.lean },
      uSwayHeight: { value: sway.height_m },
      uSwayFlutter: { value: sway.flutter_hz },
    });
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nuniform float uWindTime;\nuniform vec2 uWindDir;\nuniform float uWindStrength;\nuniform float uWindGust;\nuniform float uSwayLean;\nuniform float uSwayHeight;\nuniform float uSwayFlutter;",
      )
      .replace("#include <begin_vertex>", `#include <begin_vertex>\n${SWAY_VERTEX}`);
  };
  material.customProgramCacheKey = () => `sway_${kind}`;
  return material;
}

export function swayDepthMaterial(material, kind, wind) {
  if (!SWAY_KINDS[kind] || !wind) {
    return null;
  }
  const cutout = material.alphaTest > 0;
  const depth = new THREE.MeshDepthMaterial({
    depthPacking: THREE.RGBADepthPacking,
    map: cutout ? material.map : null,
    alphaMap: cutout ? material.alphaMap : null,
    alphaTest: material.alphaTest,
    side: material.side,
  });
  return swayMaterial(depth, kind, wind);
}

export function windVelocity(wind, speedPerStrength) {
  return wind.downwind.clone().multiplyScalar(wind.strength * speedPerStrength);
}

function boneChains(root) {
  const chains = new Map();
  root.traverse((node) => {
    if (!node.isSkinnedMesh) {
      return;
    }
    const bones = node.skeleton.bones
      .filter((bone) => BONE_INDEX.test(bone.name))
      .sort((a, b) => Number(a.name.match(BONE_INDEX)[1]) - Number(b.name.match(BONE_INDEX)[1]));
    if (bones.length >= 2 && !chains.has(bones[0])) {
      chains.set(bones[0], bones);
    }
  });
  return [...chains.values()];
}

export function orientCloth(root, wind) {
  root.updateMatrixWorld(true);
  const turned = [];
  for (const bones of boneChains(root)) {
    const [first] = bones;
    const rig = first.parent;
    const cross = new THREE.Vector3(0, 0, 1).transformDirection(first.matrixWorld);
    if (!rig || Math.abs(cross.y) < POLE_CROSS_MIN) {
      continue;
    }
    const head = new THREE.Vector3().setFromMatrixPosition(first.matrixWorld);
    const tip = new THREE.Vector3().setFromMatrixPosition(bones[bones.length - 1].matrixWorld);
    const rest = tip.sub(head).setY(0);
    if (rest.lengthSq() < 1e-8) {
      continue;
    }
    rest.normalize();
    const pivot = head.clone().addScaledVector(rest, -POLE_OFFSET_M);
    const turn = new THREE.Quaternion().setFromUnitVectors(rest, wind.downwind);
    const about = new THREE.Matrix4()
      .makeTranslation(pivot.x, pivot.y, pivot.z)
      .multiply(new THREE.Matrix4().makeRotationFromQuaternion(turn))
      .multiply(new THREE.Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z));
    const world = about.multiply(rig.matrixWorld);
    const local = rig.parent ? rig.parent.matrixWorld.clone().invert().multiply(world) : world;
    local.decompose(rig.position, rig.quaternion, rig.scale);
    rig.updateMatrixWorld(true);
    turned.push(rig.name);
  }
  return turned;
}

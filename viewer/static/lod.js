import * as THREE from "three";

export const LOD_SCREEN_FRACTIONS = [0.25, 0.1, 0.04];
export const REFERENCE_FOV_DEG = 50;
const FADE_FRACTION = 0.1;
const DENSITY_FADE_START = 0.55;
const DENSITY_FEATHER = 0.2;
const FAR_M = 1e9;
const IMPOSTOR_ALPHA_TEST = 0.5;

const LOD_DECLARATIONS = `
uniform vec3 uLodEye;
uniform float uLodScale;
uniform vec4 uLodBand;
uniform vec2 uLodReach;
varying float vLodIn;
varying float vLodOut;
varying float vLodDensity;
`;

const LOD_VERTEX = `
#ifdef USE_INSTANCING
vec3 lodCentre = (modelMatrix * vec4(instanceMatrix[3].xyz, 1.0)).xyz;
float lodSize = max(length(instanceMatrix[0].xyz), 1e-4);
#else
vec3 lodCentre = (modelMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
float lodSize = 1.0;
#endif
float lodRange = distance(lodCentre, uLodEye);
float lodDistance = lodRange * uLodScale / lodSize;
vLodIn = uLodBand.y > 0.0 ? smoothstep(uLodBand.x, uLodBand.y, lodDistance) : 1.0;
vLodOut = 1.0 - smoothstep(uLodBand.z, uLodBand.w, lodDistance);
float lodHash = fract(sin(dot(lodCentre.xz, vec2(41.37, 289.11))) * 43758.5453);
float lodThinning = 1.0 - smoothstep(uLodReach.x, uLodReach.y, lodRange);
vLodDensity = clamp((lodThinning - lodHash * ${(1 - DENSITY_FEATHER).toFixed(2)}) / ${DENSITY_FEATHER.toFixed(2)}, 0.0, 1.0);
if (vLodIn <= 0.0 || vLodOut <= 0.0 || vLodDensity <= 0.0) {
  gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
}
`;

const LOD_FRAGMENT = `
float lodNoise = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
float lodThinNoise = fract(52.9829189 * fract(dot(gl_FragCoord.yx + 13.0, vec2(0.06711056, 0.00583715))));
if (lodNoise >= vLodOut || 1.0 - lodNoise > vLodIn || lodThinNoise >= vLodDensity) {
  discard;
}
`;

const IMPOSTOR_DECLARATIONS = `
uniform vec3 uImpostorCentre;
uniform float uImpostorRadius;
uniform float uImpostorFrames;
varying mat3 vImpostorToView;
`;

const IMPOSTOR_VERTEX = `
#ifdef USE_INSTANCING
mat4 impostorInstance = modelMatrix * instanceMatrix;
#else
mat4 impostorInstance = modelMatrix;
#endif
vec3 impostorDir = normalize((inverse(impostorInstance) * vec4(uLodEye, 1.0)).xyz - uImpostorCentre);
impostorDir.y = max(impostorDir.y, 0.0);
vec2 impostorAb = impostorDir.xz / max(abs(impostorDir.x) + impostorDir.y + abs(impostorDir.z), 1e-5);
vec2 impostorOct = vec2(impostorAb.x + impostorAb.y, impostorAb.x - impostorAb.y);
vec2 impostorCell = clamp(floor((impostorOct * 0.5 + 0.5) * uImpostorFrames), 0.0, uImpostorFrames - 1.0);
vec2 impostorFrame = (impostorCell + 0.5) / uImpostorFrames * 2.0 - 1.0;
float impostorA = 0.5 * (impostorFrame.x + impostorFrame.y);
float impostorB = 0.5 * (impostorFrame.x - impostorFrame.y);
vec3 impostorView = normalize(vec3(impostorA, 1.0 - abs(impostorA) - abs(impostorB), impostorB));
vec3 impostorUp = abs(impostorView.y) > 0.999 ? vec3(0.0, 0.0, -1.0) : normalize(vec3(0.0, 1.0, 0.0) - impostorView * impostorView.y);
vec3 impostorRight = cross(impostorUp, impostorView);
transformed = uImpostorCentre + (position.x * impostorRight + position.y * impostorUp) * uImpostorRadius;
#ifdef USE_MAP
vMapUv = (impostorCell + position.xy * 0.5 + 0.5) / uImpostorFrames;
#endif
vImpostorToView = mat3(viewMatrix * impostorInstance);
`;

const IMPOSTOR_NORMAL = `
vec3 impostorNormal = texture2D(uImpostorNormal, vMapUv).xyz * 2.0 - 1.0;
normal = normalize(vImpostorToView * impostorNormal);
`;

export class LodView {
  constructor() {
    this.uniforms = {
      uLodEye: { value: new THREE.Vector3() },
      uLodScale: { value: 1 },
    };
  }

  update(camera) {
    this.uniforms.uLodEye.value.copy(camera.position);
    const reference = Math.tan(THREE.MathUtils.degToRad(REFERENCE_FOV_DEG) / 2);
    this.uniforms.uLodScale.value = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) / reference;
  }

  get eye() {
    return this.uniforms.uLodEye.value;
  }

  get scale() {
    return this.uniforms.uLodScale.value;
  }
}

export function switchDistances(diameter, transitions) {
  const focal = 2 * Math.tan(THREE.MathUtils.degToRad(REFERENCE_FOV_DEG) / 2);
  return LOD_SCREEN_FRACTIONS.slice(0, transitions).map((fraction) => diameter / (focal * fraction));
}

export function levelBands(distances, levels) {
  return Array.from({ length: levels }, (_, level) => {
    const fadeIn = level === 0 ? [0, 0] : [distances[level - 1] * (1 - FADE_FRACTION), distances[level - 1] * (1 + FADE_FRACTION)];
    const fadeOut = level === levels - 1 ? [FAR_M, 2 * FAR_M] : [distances[level] * (1 - FADE_FRACTION), distances[level] * (1 + FADE_FRACTION)];
    return [...fadeIn, ...fadeOut];
  });
}

export function reachRange(reach) {
  return Number.isFinite(reach) ? [reach * DENSITY_FADE_START, reach] : [FAR_M, 2 * FAR_M];
}

export function lodMaterial(material, band, reach, view) {
  const previous = material.onBeforeCompile;
  const previousKey = material.customProgramCacheKey;
  const uniforms = {
    ...view.uniforms,
    uLodBand: { value: new THREE.Vector4(...band) },
    uLodReach: { value: new THREE.Vector2(...reach) },
  };
  material.onBeforeCompile = (shader, renderer) => {
    previous.call(material, shader, renderer);
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>\n${LOD_DECLARATIONS}`)
      .replace("#include <project_vertex>", `#include <project_vertex>\n${LOD_VERTEX}`);
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>\n${LOD_DECLARATIONS}`)
      .replace("#include <clipping_planes_fragment>", `#include <clipping_planes_fragment>\n${LOD_FRAGMENT}`);
  };
  material.customProgramCacheKey = () => `${previousKey.call(material)}:lod`;
  material.userData.lodBand = band;
  return material;
}

export function depthMaterial(material) {
  const cutout = material.alphaTest > 0;
  return new THREE.MeshDepthMaterial({
    depthPacking: THREE.RGBADepthPacking,
    map: cutout ? material.map : null,
    alphaMap: cutout ? material.alphaMap : null,
    alphaTest: material.alphaTest,
    side: material.side,
  });
}

function impostorShader(material, impostor, unitScale, view, normalMap) {
  const uniforms = {
    ...view.uniforms,
    uImpostorCentre: { value: new THREE.Vector3(...impostor.centre).multiplyScalar(unitScale) },
    uImpostorRadius: { value: impostor.radius * unitScale },
    uImpostorFrames: { value: impostor.frames },
    ...(normalMap ? { uImpostorNormal: { value: normalMap } } : {}),
  };
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>\n${IMPOSTOR_DECLARATIONS}`)
      .replace("#include <begin_vertex>", `#include <begin_vertex>\n${IMPOSTOR_VERTEX}`);
    if (normalMap) {
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", `#include <common>\n${IMPOSTOR_DECLARATIONS}\nuniform sampler2D uImpostorNormal;`)
        .replace("#include <normal_fragment_maps>", IMPOSTOR_NORMAL);
    }
  };
  material.customProgramCacheKey = () => `impostor:${normalMap !== null}`;
  return material;
}

async function loadTexture(url, colorSpace) {
  const texture = await new THREE.TextureLoader().loadAsync(url);
  texture.colorSpace = colorSpace;
  texture.anisotropy = 4;
  return texture;
}

export async function impostorPart(impostor, unitScale, view) {
  const [albedo, normal] = await Promise.all([
    loadTexture(impostor.albedo, THREE.SRGBColorSpace),
    loadTexture(impostor.normal, THREE.NoColorSpace),
  ]);
  const geometry = new THREE.PlaneGeometry(2, 2);
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(...impostor.centre).multiplyScalar(unitScale), impostor.radius * unitScale);
  const material = impostorShader(
    new THREE.MeshStandardMaterial({ map: albedo, alphaTest: IMPOSTOR_ALPHA_TEST, side: THREE.DoubleSide, roughness: 0.9, metalness: 0 }),
    impostor,
    unitScale,
    view,
    normal,
  );
  const depth = impostorShader(
    new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking, map: albedo, alphaTest: IMPOSTOR_ALPHA_TEST, side: THREE.DoubleSide }),
    impostor,
    unitScale,
    view,
    null,
  );
  material.userData.depth = depth;
  material.userData.textures = [albedo, normal];
  return [geometry, material];
}

export function tileRange(tile, eye, scale) {
  const { centre, radius, minY, maxY, minSize, maxSize } = tile.userData;
  const horizontal = Math.hypot(eye.x - centre.x, eye.z - centre.y);
  const below = Math.max(0, minY - eye.y, eye.y - maxY);
  const near = Math.hypot(Math.max(0, horizontal - radius), below);
  const far = Math.hypot(horizontal + radius, Math.max(Math.abs(eye.y - minY), Math.abs(eye.y - maxY)));
  return { near, nearLod: (near * scale) / maxSize, farLod: (far * scale) / minSize };
}

export function bandVisible(band, range) {
  return range.nearLod < band[3] && range.farLod > band[0];
}

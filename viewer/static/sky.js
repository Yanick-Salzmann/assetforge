import * as THREE from "three";

export const DEFAULT_ATMOSPHERE = {
  latitude_deg: 45,
  day_of_year: 172,
  time_of_day_h: 10,
  turbidity: 3,
  rayleigh: 1.5,
  visibility_km: 30,
  haze_colour: "#c8d2dc",
  ground_albedo: "#6b6a55",
};

const AXIAL_TILT_DEG = 23.44;
const SUN_ILLUMINANCE = 3.6;
const MOON_ILLUMINANCE = 0.08;
const MOON_COLOUR = new THREE.Color(0.66, 0.76, 1.0);
const RAYLEIGH_BETA = new THREE.Vector3(5.8e-6, 13.5e-6, 33.1e-6);
const RAYLEIGH_REFERENCE = 1.5;
const RAYLEIGH_SCALE_HEIGHT_M = 8000;
const MIE_SCALE_HEIGHT_M = 1200;
const MIE_EXTINCTION = 1.1;
const MIE_CLEAN = 4e-6;
const MIE_PER_TURBIDITY = 7e-6;
const MIE_G = 0.78;
const SKY_GAIN = 2.5;
const SUN_ANGULAR_RADIUS = THREE.MathUtils.degToRad(0.4);
const SUN_DISC_RADIANCE = 40;
const AIRGLOW = new THREE.Color(0.00012, 0.0002, 0.0004);
const TWILIGHT_GLOW = new THREE.Color(0.006, 0.009, 0.02);
const WHITE_BALANCE_STRENGTH = 0.7;
const EXPOSURE_KEY = 0.9;
const EXPOSURE_ADAPTATION = 0.4;
const EXPOSURE_MIN = 0.35;
const EXPOSURE_MAX = 12;
const KOSCHMIEDER_SQUARED = Math.sqrt(3.912);
const HAZE_SCALE_HEIGHT_M = 1500;
const CAPTURE_SIZE = 32;
const CAPTURE_SCALE = 100;
const GROUND_DROP = 5;
const STAR_COUNT = 2600;
const STAR_RADIUS = 0.42;
const SKY_SCALE_OF_FAR = 0.9;
const HAZE_WEIGHT = 0.35;
const FOG_SUNWARD_WEIGHT = 0.3;

const SKY_VERTEX = `
varying vec3 vDirection;
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vDirection = world.xyz - cameraPosition;
  gl_Position = projectionMatrix * viewMatrix * world;
  gl_Position.z = gl_Position.w;
}
`;

const SKY_FRAGMENT = `
#define PRIMARY_STEPS 16
#define LIGHT_STEPS 6
const float PLANET_RADIUS = 6371e3;
const float ATMOSPHERE_RADIUS = 6471e3;
const float VIEW_ALTITUDE = 600.0;
uniform vec3 uSunDirection;
uniform float uSunIntensity;
uniform vec3 uMoonDirection;
uniform float uMoonIntensity;
uniform vec3 uBetaR;
uniform float uBetaM;
uniform float uMieG;
uniform vec3 uSunDisc;
uniform float uSunCos;
uniform vec3 uAirglow;
uniform float uRayleighHeight;
uniform float uMieHeight;
uniform float uMieExtinction;
uniform float uSkyGain;
uniform vec3 uWhiteBalance;
varying vec3 vDirection;

vec2 sphere(vec3 origin, vec3 direction, float radius) {
  float b = dot(origin, direction);
  float c = dot(origin, origin) - radius * radius;
  float h = b * b - c;
  if (h < 0.0) {
    return vec2(1e9, -1e9);
  }
  h = sqrt(h);
  return vec2(-b - h, -b + h);
}

vec3 scatter(vec3 direction, vec3 light, float intensity) {
  vec3 origin = vec3(0.0, PLANET_RADIUS + VIEW_ALTITUDE, 0.0);
  float far = sphere(origin, direction, ATMOSPHERE_RADIUS).y;
  vec2 ground = sphere(origin, direction, PLANET_RADIUS);
  if (ground.x > 0.0 && ground.y > ground.x) {
    far = min(far, ground.x);
  }
  vec3 sumR = vec3(0.0);
  vec3 sumM = vec3(0.0);
  float depthR = 0.0;
  float depthM = 0.0;
  for (int i = 0; i < PRIMARY_STEPS; i++) {
    float near = far * pow(float(i) / float(PRIMARY_STEPS), 2.0);
    float next = far * pow(float(i + 1) / float(PRIMARY_STEPS), 2.0);
    float stepLength = next - near;
    vec3 point = origin + direction * (0.5 * (near + next));
    float height = max(length(point) - PLANET_RADIUS, 0.0);
    float densityR = exp(-height / uRayleighHeight) * stepLength;
    float densityM = exp(-height / uMieHeight) * stepLength;
    depthR += 0.5 * densityR;
    depthM += 0.5 * densityM;
    vec2 blocker = sphere(point, light, PLANET_RADIUS);
    bool shadowed = blocker.x > 0.0 && blocker.y > blocker.x;
    if (shadowed) {
      depthR += 0.5 * densityR;
      depthM += 0.5 * densityM;
      continue;
    }
    float lightLength = sphere(point, light, ATMOSPHERE_RADIUS).y / float(LIGHT_STEPS);
    float lightR = 0.0;
    float lightM = 0.0;
    for (int j = 0; j < LIGHT_STEPS; j++) {
      vec3 probe = point + light * (lightLength * (float(j) + 0.5));
      float sampleHeight = max(length(probe) - PLANET_RADIUS, 0.0);
      lightR += exp(-sampleHeight / uRayleighHeight) * lightLength;
      lightM += exp(-sampleHeight / uMieHeight) * lightLength;
    }
    vec3 tau = uBetaR * (depthR + lightR) + uBetaM * uMieExtinction * (depthM + lightM);
    vec3 attenuation = exp(-tau);
    sumR += attenuation * densityR;
    sumM += attenuation * densityM;
    depthR += 0.5 * densityR;
    depthM += 0.5 * densityM;
  }
  float mu = dot(direction, light);
  float g = uMieG;
  float phaseR = 3.0 / (16.0 * 3.14159265) * (1.0 + mu * mu);
  float phaseM = 3.0 / (8.0 * 3.14159265) * ((1.0 - g * g) * (1.0 + mu * mu)) / ((2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * mu, 1.5));
  return uSkyGain * intensity * (sumR * uBetaR * phaseR + sumM * uBetaM * phaseM);
}

void main() {
  vec3 direction = normalize(vDirection);
  vec3 above = normalize(vec3(direction.x, max(direction.y, 0.0), direction.z));
  vec3 colour = scatter(above, uSunDirection, uSunIntensity);
  if (uMoonIntensity > 0.0) {
    colour += scatter(above, uMoonDirection, uMoonIntensity);
  }
  float up = clamp(above.y, 0.0, 1.0);
  colour += uAirglow * mix(2.0, 0.6, sqrt(up));
  float disc = smoothstep(uSunCos, mix(uSunCos, 1.0, 0.3), dot(direction, uSunDirection));
  colour += uSunDisc * disc * step(0.0, direction.y + 0.01);
  gl_FragColor = vec4(colour * uWhiteBalance, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

function smoothstep(edge0, edge1, x) {
  const t = THREE.MathUtils.clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function luminance(colour) {
  return 0.2126 * colour.r + 0.7152 * colour.g + 0.0722 * colour.b;
}

export function declination(dayOfYear) {
  return THREE.MathUtils.degToRad(-AXIAL_TILT_DEG) * Math.cos(((2 * Math.PI) / 365) * (dayOfYear + 10));
}

export function celestialDirection(latitudeDeg, decl, hours) {
  const latitude = THREE.MathUtils.degToRad(latitudeDeg);
  const hourAngle = THREE.MathUtils.degToRad(15 * (hours - 12));
  const sinElevation = Math.sin(latitude) * Math.sin(decl) + Math.cos(latitude) * Math.cos(decl) * Math.cos(hourAngle);
  const elevation = Math.asin(THREE.MathUtils.clamp(sinElevation, -1, 1));
  const azimuth = Math.atan2(-Math.sin(hourAngle), Math.tan(decl) * Math.cos(latitude) - Math.sin(latitude) * Math.cos(hourAngle));
  const flat = Math.cos(elevation);
  return {
    elevation,
    azimuth: (azimuth + 2 * Math.PI) % (2 * Math.PI),
    direction: new THREE.Vector3(Math.sin(azimuth) * flat, Math.sin(elevation), -Math.cos(azimuth) * flat),
  };
}

export function airMass(elevation) {
  const degrees = Math.max(THREE.MathUtils.radToDeg(elevation), -1);
  return 1 / (Math.sin(THREE.MathUtils.degToRad(degrees)) + 0.50572 * Math.pow(degrees + 6.07995, -1.6364));
}

function rayleighBeta(settings) {
  return RAYLEIGH_BETA.clone().multiplyScalar(settings.rayleigh / RAYLEIGH_REFERENCE);
}

function mieBeta(settings) {
  return MIE_CLEAN + MIE_PER_TURBIDITY * (settings.turbidity - 1);
}

export function sunTransmittance(elevation, settings) {
  const mass = airMass(elevation);
  const rayleigh = rayleighBeta(settings).multiplyScalar(RAYLEIGH_SCALE_HEIGHT_M);
  const mie = mieBeta(settings) * MIE_EXTINCTION * MIE_SCALE_HEIGHT_M;
  return new THREE.Color(Math.exp(-mass * (rayleigh.x + mie)), Math.exp(-mass * (rayleigh.y + mie)), Math.exp(-mass * (rayleigh.z + mie)));
}

export const TONE_MAPPING = THREE.NeutralToneMapping;
const NEUTRAL_START = 0.76;
const NEUTRAL_DESATURATION = 0.15;

export function displayColour(linear, exposure) {
  let r = linear.r * exposure;
  let g = linear.g * exposure;
  let b = linear.b * exposure;
  const low = Math.min(r, g, b);
  const offset = low < 0.08 ? low - 6.25 * low * low : 0.04;
  r -= offset;
  g -= offset;
  b -= offset;
  const peak = Math.max(r, g, b);
  if (peak >= NEUTRAL_START) {
    const span = 1 - NEUTRAL_START;
    const newPeak = 1 - (span * span) / (peak + span - NEUTRAL_START);
    const scale = newPeak / peak;
    const fade = 1 - 1 / (NEUTRAL_DESATURATION * (peak - newPeak) + 1);
    r = THREE.MathUtils.lerp(r * scale, newPeak, fade);
    g = THREE.MathUtils.lerp(g * scale, newPeak, fade);
    b = THREE.MathUtils.lerp(b * scale, newPeak, fade);
  }
  const out = new THREE.Color();
  out.setRGB(THREE.MathUtils.clamp(r, 0, 1), THREE.MathUtils.clamp(g, 0, 1), THREE.MathUtils.clamp(b, 0, 1), THREE.LinearSRGBColorSpace);
  return out.convertLinearToSRGB();
}

function skyMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: {
      uSunDirection: { value: new THREE.Vector3(0, 1, 0) },
      uSunIntensity: { value: SUN_ILLUMINANCE },
      uMoonDirection: { value: new THREE.Vector3(0, -1, 0) },
      uMoonIntensity: { value: 0 },
      uBetaR: { value: RAYLEIGH_BETA.clone() },
      uBetaM: { value: 2e-5 },
      uMieG: { value: MIE_G },
      uSunDisc: { value: new THREE.Color(0, 0, 0) },
      uSunCos: { value: Math.cos(SUN_ANGULAR_RADIUS) },
      uAirglow: { value: AIRGLOW.clone() },
      uRayleighHeight: { value: RAYLEIGH_SCALE_HEIGHT_M },
      uMieHeight: { value: MIE_SCALE_HEIGHT_M },
      uMieExtinction: { value: MIE_EXTINCTION },
      uSkyGain: { value: SKY_GAIN },
      uWhiteBalance: { value: new THREE.Color(1, 1, 1) },
    },
    vertexShader: SKY_VERTEX,
    fragmentShader: SKY_FRAGMENT,
    side: THREE.BackSide,
    depthWrite: false,
  });
}

function starField(seed) {
  let state = seed >>> 0 || 1;
  const random = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const positions = new Float32Array(STAR_COUNT * 3);
  const colours = new Float32Array(STAR_COUNT * 3);
  for (let index = 0; index < STAR_COUNT; index += 1) {
    const y = random() * 1.1 - 0.1;
    const theta = random() * 2 * Math.PI;
    const ring = Math.sqrt(Math.max(0, 1 - y * y));
    positions.set([Math.cos(theta) * ring * STAR_RADIUS, y * STAR_RADIUS, Math.sin(theta) * ring * STAR_RADIUS], index * 3);
    const brightness = Math.pow(random(), 6) * 0.9 + 0.08;
    const warmth = random();
    colours.set([brightness * (0.85 + 0.15 * warmth), brightness * 0.92, brightness * (1.05 - 0.15 * warmth)], index * 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colours, 3));
  const material = new THREE.PointsMaterial({
    size: 2,
    sizeAttenuation: false,
    vertexColors: true,
    transparent: true,
    opacity: 0,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    fog: false,
    toneMapped: false,
  });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  return points;
}

export function hazeDensity(base, clearanceM) {
  const height = Math.max(clearanceM, 0) / HAZE_SCALE_HEIGHT_M;
  if (height < 1e-3) {
    return base;
  }
  return base * ((1 - Math.exp(-height)) / height);
}

export function normaliseSettings(recorded) {
  return { ...DEFAULT_ATMOSPHERE, ...(recorded ?? {}) };
}

export class Atmosphere {
  constructor(renderer) {
    this.renderer = renderer;
    this.pmrem = new THREE.PMREMGenerator(renderer);
    this.environment = null;
    this.settings = normaliseSettings(null);
    this.time = this.settings.time_of_day_h;
    this.state = null;
    this.skyMaterial = skyMaterial();

    const box = new THREE.BoxGeometry(1, 1, 1);
    const sky = new THREE.Mesh(box, this.skyMaterial);
    sky.renderOrder = -2;
    sky.frustumCulled = false;
    this.stars = starField(7);
    this.stars.renderOrder = -1;
    this.group = new THREE.Group();
    this.group.name = "atmosphere";
    this.group.add(sky, this.stars);

    this.captureScene = new THREE.Scene();
    const captureSky = new THREE.Mesh(box, this.skyMaterial);
    captureSky.scale.setScalar(CAPTURE_SCALE);
    captureSky.renderOrder = -1;
    this.groundMaterial = new THREE.MeshBasicMaterial({ color: 0x000000, toneMapped: false });
    this.captureGround = new THREE.Mesh(new THREE.PlaneGeometry(CAPTURE_SCALE * 20, CAPTURE_SCALE * 20), this.groundMaterial);
    this.captureGround.rotation.x = -Math.PI / 2;
    this.captureGround.position.y = -GROUND_DROP;
    this.captureScene.add(captureSky, this.captureGround);
    this.captureTarget = new THREE.WebGLCubeRenderTarget(CAPTURE_SIZE, { type: THREE.FloatType });
    this.captureCamera = new THREE.CubeCamera(0.5, CAPTURE_SCALE * 30, this.captureTarget);
  }

  configure(recorded) {
    this.settings = normaliseSettings(recorded);
    this.time = this.settings.time_of_day_h;
  }

  follow(camera) {
    this.group.position.copy(camera.position);
    this.group.scale.setScalar(camera.far * SKY_SCALE_OF_FAR);
  }

  readRows(face, rowStart, rows) {
    const buffer = new Float32Array(CAPTURE_SIZE * rows * 4);
    this.renderer.readRenderTargetPixels(this.captureTarget, 0, rowStart, CAPTURE_SIZE, rows, buffer, face);
    const sum = new THREE.Color(0, 0, 0);
    const count = CAPTURE_SIZE * rows;
    for (let index = 0; index < count; index += 1) {
      sum.r += buffer[index * 4];
      sum.g += buffer[index * 4 + 1];
      sum.b += buffer[index * 4 + 2];
    }
    return sum.multiplyScalar(1 / count);
  }

  capture() {
    const middle = CAPTURE_SIZE / 2 - 1;
    this.captureGround.visible = false;
    this.captureCamera.update(this.renderer, this.captureScene);
    const horizon = new THREE.Color(0, 0, 0);
    let away = null;
    for (const face of [0, 1, 4, 5]) {
      const side = this.readRows(face, middle, 2);
      horizon.add(side);
      if (!away || luminance(side) < luminance(away)) {
        away = side;
      }
    }
    horizon.multiplyScalar(0.25);
    const zenith = this.readRows(2, middle, 2);
    this.captureGround.visible = true;
    return { horizon, away, zenith };
  }

  update(time = this.time) {
    this.time = ((time % 24) + 24) % 24;
    const settings = this.settings;
    const decl = declination(settings.day_of_year);
    const sun = celestialDirection(settings.latitude_deg, decl, this.time);
    const moon = celestialDirection(settings.latitude_deg, -decl, this.time + 12);
    const sinSun = Math.sin(sun.elevation);

    const transmittance = sunTransmittance(sun.elevation, settings);
    const sunFade = smoothstep(-0.02, 0.02, sinSun);
    const sunIntensity = SUN_ILLUMINANCE * luminance(transmittance) * sunFade;
    const sunColour = transmittance.clone().multiplyScalar(1 / Math.max(transmittance.r, transmittance.g, transmittance.b, 1e-6));

    const night = 1 - smoothstep(-0.15, -0.05, sinSun);
    const moonUp = smoothstep(-0.02, 0.12, Math.sin(moon.elevation));
    const moonIntensity = MOON_ILLUMINANCE * night * moonUp;

    const uniforms = this.skyMaterial.uniforms;
    uniforms.uSunDirection.value.copy(sun.direction);
    uniforms.uSunIntensity.value = SUN_ILLUMINANCE;
    uniforms.uMoonDirection.value.copy(moon.direction);
    uniforms.uMoonIntensity.value = moonIntensity;
    uniforms.uBetaR.value.copy(rayleighBeta(settings));
    uniforms.uBetaM.value = mieBeta(settings);
    uniforms.uSunDisc.value.copy(transmittance).multiplyScalar(SUN_DISC_RADIANCE * sunFade);
    const twilight = smoothstep(-0.25, -0.02, sinSun) * (1 - smoothstep(0.02, 0.2, sinSun));
    uniforms.uAirglow.value.copy(AIRGLOW).multiplyScalar(night).add(TWILIGHT_GLOW.clone().multiplyScalar(twilight));
    this.stars.material.opacity = night * (1 - 0.6 * moonUp);

    const useSun = sinSun > -0.05;
    const keyDirection = useSun ? sun.direction : moon.direction;
    const keyIntensity = useSun ? sunIntensity : moonIntensity;
    uniforms.uWhiteBalance.value.setRGB(1, 1, 1);
    const neutral = this.capture();
    const illuminant = neutral.zenith.clone().add(neutral.horizon).multiplyScalar(0.5 * Math.PI);
    if (useSun) {
      illuminant.add(sunColour.clone().multiplyScalar(sunIntensity * Math.max(sun.direction.y, 0)));
    }
    const strength = WHITE_BALANCE_STRENGTH * smoothstep(0.05, 0.5, sinSun);
    const balance = new THREE.Color(1, 1, 1);
    if (luminance(illuminant) > 1e-6) {
      const tint = illuminant.clone().multiplyScalar(1 / luminance(illuminant));
      balance.setRGB(Math.pow(tint.r, -strength), Math.pow(tint.g, -strength), Math.pow(tint.b, -strength));
      balance.multiplyScalar(1 / luminance(balance));
    }
    uniforms.uWhiteBalance.value.copy(balance);
    const { horizon, away, zenith } = this.capture();
    const keyColour = (useSun ? sunColour.clone() : MOON_COLOUR.clone()).multiply(balance);

    const skyIrradiance = (luminance(zenith) + luminance(horizon)) * 0.5 * Math.PI;
    const direct = keyIntensity * Math.max(keyDirection.y, 0);
    const ground = keyColour.clone().multiplyScalar(direct);
    ground.add(new THREE.Color(skyIrradiance, skyIrradiance, skyIrradiance));
    ground.multiply(new THREE.Color(settings.ground_albedo)).multiplyScalar(1 / Math.PI);
    this.groundMaterial.color.copy(ground);
    this.captureCamera.update(this.renderer, this.captureScene);
    if (this.environment) {
      this.environment.dispose();
    }
    this.environment = this.pmrem.fromCubemap(this.captureTarget.texture);

    const adaptation = (direct + skyIrradiance) / Math.PI;
    const exposure = THREE.MathUtils.clamp(EXPOSURE_KEY / Math.pow(Math.max(adaptation, 1e-4), EXPOSURE_ADAPTATION), EXPOSURE_MIN, EXPOSURE_MAX);
    const haze = new THREE.Color(settings.haze_colour);
    const aerial = away.clone().lerp(horizon, FOG_SUNWARD_WEIGHT);
    const hazeLinear = haze.clone().multiplyScalar(luminance(aerial) / Math.max(luminance(haze), 1e-6));
    const fogLinear = aerial.clone().lerp(hazeLinear, HAZE_WEIGHT);
    const fogDensity = KOSCHMIEDER_SQUARED / (settings.visibility_km * 1000);

    this.state = {
      time: this.time,
      sun,
      moon,
      sunColour,
      sunIntensity,
      moonIntensity,
      keyDirection,
      keyColour,
      keyIntensity,
      horizon,
      zenith,
      exposure,
      fogColour: displayColour(fogLinear, exposure),
      fogDensity,
      environment: this.environment.texture,
    };
    return this.state;
  }

  dispose() {
    if (this.environment) {
      this.environment.dispose();
    }
    this.captureTarget.dispose();
    this.pmrem.dispose();
    this.skyMaterial.dispose();
    this.stars.geometry.dispose();
    this.stars.material.dispose();
  }
}

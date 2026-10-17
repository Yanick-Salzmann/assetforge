import * as THREE from "three";
import { windVelocity } from "./wind.js";

export const FX_TYPES = ["fire", "smoke", "sparks", "steam", "flicker_light"];
const MAX_PARTICLES = 400;
const GRAVITY_M_S2 = 9.81;
const GLOW = 1.6;
const DAYLIGHT_GLOW = 0.25;
const LIGHT_CANDELA_PER_M = 60;
const LIGHT_REACH_PER_M = 12;
const WIND_M_S_PER_STRENGTH = 1.5;

const PROFILES = {
  fire: { life: [0.5, 0.9], speed: [1.4, 2.2], spread: 0.35, grow: [1.2, 0.2], fade: [0.4, 0], cool: 0.35, emissive: true, gravity: 0 },
  smoke: { life: [4, 6], speed: [0.35, 0.7], spread: 0.25, grow: [0.6, 2.8], fade: [0.8, 0], cool: 1, emissive: false, gravity: 0, drift: 0.25, carry: 0.6 },
  sparks: { life: [0.4, 0.9], speed: [1.5, 3], spread: 0.15, cone: 0.6, grow: [1, 0.6], fade: [1, 0], cool: 0.5, emissive: true, gravity: 1 },
  steam: { life: [1.5, 2.5], speed: [0.5, 0.9], spread: 0.2, grow: [0.5, 2.2], fade: [0.6, 0], cool: 1, emissive: false, gravity: 0, drift: 0.15, carry: 0.8 },
};

const VERTEX = `
attribute float size;
attribute float alpha;
attribute vec3 tint;
uniform float uScale;
varying float vAlpha;
varying vec3 vTint;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = size * uScale / max(-mv.z, 0.01);
  vAlpha = alpha;
  vTint = tint;
  gl_Position = projectionMatrix * mv;
}
`;

const FRAGMENT = `
uniform float uBrightness;
uniform float uSoftness;
varying float vAlpha;
varying vec3 vTint;
void main() {
  float r = length(gl_PointCoord - 0.5) * 2.0;
  if (r > 1.0 || vAlpha <= 0.0) {
    discard;
  }
  float falloff = pow(1.0 - r, uSoftness);
  gl_FragColor = vec4(vTint * uBrightness, vAlpha * falloff);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`;

function lerp(range, t) {
  return range[0] + (range[1] - range[0]) * t;
}

function random(range) {
  return lerp(range, Math.random());
}

class Emitter {
  constructor(socket, entry, wind) {
    this.entry = entry;
    this.socket = socket;
    this.profile = PROFILES[entry.type];
    this.wind = wind && this.profile.carry ? windVelocity(wind, WIND_M_S_PER_STRENGTH) : null;
    this.localWind = new THREE.Vector3();
    this.toLocal = new THREE.Matrix3();
    this.colour = new THREE.Color().fromArray(entry.colour);
    this.capacity = Math.min(MAX_PARTICLES, Math.ceil(entry.rate * this.profile.life[1]) + 1);
    this.age = new Float32Array(this.capacity).fill(Infinity);
    this.life = new Float32Array(this.capacity).fill(1);
    this.velocity = new Float32Array(this.capacity * 3);
    this.debt = 0;
    this.next = 0;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(this.capacity * 3), 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(new Float32Array(this.capacity), 1));
    geometry.setAttribute("alpha", new THREE.BufferAttribute(new Float32Array(this.capacity), 1));
    geometry.setAttribute("tint", new THREE.BufferAttribute(new Float32Array(this.capacity * 3), 3));
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      uniforms: { uScale: { value: 1 }, uBrightness: { value: 1 }, uSoftness: { value: this.profile.emissive ? 1.5 : 0.8 } },
      transparent: true,
      depthWrite: false,
      blending: this.profile.emissive ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    this.points = new THREE.Points(geometry, this.material);
    this.points.name = `${entry.node}_particles`;
    this.points.frustumCulled = false;
    this.points.raycast = () => {};
    socket.add(this.points);
  }

  spawn(index) {
    const profile = this.profile;
    const spread = profile.spread * this.entry.size;
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.sqrt(Math.random()) * spread;
    const positions = this.points.geometry.attributes.position.array;
    positions[index * 3] = Math.cos(angle) * radius;
    positions[index * 3 + 1] = 0;
    positions[index * 3 + 2] = Math.sin(angle) * radius;
    const speed = random(profile.speed) * Math.max(this.entry.size, 0.2);
    const cone = profile.cone ?? 0.15;
    const tilt = Math.random() * cone;
    const heading = Math.random() * Math.PI * 2;
    this.velocity[index * 3] = Math.sin(tilt) * Math.cos(heading) * speed;
    this.velocity[index * 3 + 1] = Math.cos(tilt) * speed;
    this.velocity[index * 3 + 2] = Math.sin(tilt) * Math.sin(heading) * speed;
    this.age[index] = 0;
    this.life[index] = random(profile.life);
  }

  update(delta, active) {
    const profile = this.profile;
    if (active) {
      this.debt += delta * this.entry.rate;
      while (this.debt >= 1) {
        this.debt -= 1;
        this.spawn(this.next);
        this.next = (this.next + 1) % this.capacity;
      }
    } else {
      this.debt = 0;
    }
    const attributes = this.points.geometry.attributes;
    const positions = attributes.position.array;
    const sizes = attributes.size.array;
    const alphas = attributes.alpha.array;
    const tints = attributes.tint.array;
    if (this.wind) {
      this.toLocal.setFromMatrix4(this.socket.matrixWorld).invert();
      this.localWind.copy(this.wind).applyMatrix3(this.toLocal);
    }
    const carry = this.wind ? Math.min(1, profile.carry * delta) : 0;
    let alive = 0;
    for (let index = 0; index < this.capacity; index += 1) {
      this.age[index] += delta;
      const t = this.age[index] / this.life[index];
      if (!(t < 1)) {
        alphas[index] = 0;
        sizes[index] = 0;
        continue;
      }
      alive += 1;
      const v = index * 3;
      this.velocity[v + 1] -= profile.gravity * GRAVITY_M_S2 * delta;
      if (this.wind) {
        this.velocity[v] += (this.localWind.x - this.velocity[v]) * carry;
        this.velocity[v + 2] += (this.localWind.z - this.velocity[v + 2]) * carry;
      } else if (profile.drift) {
        this.velocity[v] += profile.drift * delta;
      }
      positions[v] += this.velocity[v] * delta;
      positions[v + 1] += this.velocity[v + 1] * delta;
      positions[v + 2] += this.velocity[v + 2] * delta;
      sizes[index] = lerp(profile.grow, t) * this.entry.size;
      alphas[index] = lerp(profile.fade, t);
      const warmth = 1 - t * (1 - profile.cool);
      tints[v] = this.colour.r * warmth;
      tints[v + 1] = this.colour.g * warmth * warmth;
      tints[v + 2] = this.colour.b * warmth * warmth * warmth;
    }
    attributes.position.needsUpdate = true;
    attributes.size.needsUpdate = true;
    attributes.alpha.needsUpdate = true;
    attributes.tint.needsUpdate = true;
    return alive;
  }

  drift() {
    const positions = this.points.geometry.attributes.position.array;
    const world = new THREE.Vector3();
    const origin = new THREE.Vector3().setFromMatrixPosition(this.socket.matrixWorld);
    const sum = new THREE.Vector3();
    let count = 0;
    for (let index = 0; index < this.capacity; index += 1) {
      if (this.age[index] < this.life[index]) {
        world.fromArray(positions, index * 3).applyMatrix4(this.socket.matrixWorld).sub(origin);
        sum.add(world);
        count += 1;
      }
    }
    return count > 0 ? [sum.x / count, sum.y / count, sum.z / count] : null;
  }

  dispose() {
    this.points.removeFromParent();
    this.points.geometry.dispose();
    this.material.dispose();
  }
}

class Flicker {
  constructor(socket, entry) {
    this.entry = entry;
    this.base = LIGHT_CANDELA_PER_M * entry.size;
    this.light = new THREE.PointLight(new THREE.Color().fromArray(entry.colour), 0, entry.size * LIGHT_REACH_PER_M, 2);
    this.light.name = `${entry.node}_light`;
    this.light.castShadow = false;
    this.phase = Math.random() * 100;
    socket.add(this.light);
  }

  update(delta, active) {
    this.phase += delta * Math.max(this.entry.rate, 0.1);
    const wobble = 0.7 + 0.2 * Math.sin(this.phase * 2.3) + 0.1 * Math.sin(this.phase * 7.1 + 1.3);
    this.light.intensity = active ? this.base * wobble : 0;
    return active ? 1 : 0;
  }

  dispose() {
    this.light.removeFromParent();
    this.light.dispose();
  }
}

export class FxSystem {
  constructor(root, entries = [], wind = null) {
    this.state = "idle";
    this.wind = wind;
    this.sockets = [];
    for (const entry of entries) {
      if (!FX_TYPES.includes(entry.type)) {
        continue;
      }
      const socket = root.getObjectByName(entry.node);
      if (!socket) {
        this.sockets.push({ entry, missing: true, parts: [] });
        continue;
      }
      const parts = [];
      if (entry.type !== "flicker_light") {
        parts.push(new Emitter(socket, entry, wind));
      }
      if (entry.type === "fire" || entry.type === "flicker_light") {
        parts.push(new Flicker(socket, entry));
      }
      this.sockets.push({ entry, missing: false, parts, alive: 0 });
    }
  }

  get empty() {
    return this.sockets.length === 0;
  }

  setState(state) {
    this.state = state;
  }

  isActive(entry) {
    return entry.state === "idle" || entry.state === this.state;
  }

  update(delta, camera, viewportHeight, exposure = 1) {
    const scale = camera.isPerspectiveCamera ? viewportHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) : viewportHeight;
    const lit = 1 / Math.max(exposure, 1e-6);
    const glow = GLOW * Math.max(1, DAYLIGHT_GLOW * lit);
    let alive = 0;
    for (const socket of this.sockets) {
      const active = this.isActive(socket.entry);
      socket.alive = 0;
      for (const part of socket.parts) {
        if (part.material) {
          part.material.uniforms.uScale.value = scale;
          part.material.uniforms.uBrightness.value = part.profile.emissive ? glow : lit;
        }
        socket.alive += part.update(Math.min(delta, 0.1), active);
      }
      alive += socket.alive;
    }
    return alive;
  }

  snapshot() {
    return this.sockets.map(({ entry, missing, alive, parts }) => ({
      node: entry.node,
      type: entry.type,
      state: entry.state,
      missing,
      active: !missing && this.isActive(entry),
      particles: alive ?? 0,
      drift: this.wind && !missing ? (parts.find((part) => part.wind)?.drift() ?? null) : null,
    }));
  }

  dispose() {
    for (const socket of this.sockets) {
      for (const part of socket.parts) {
        part.dispose();
      }
    }
    this.sockets = [];
  }
}

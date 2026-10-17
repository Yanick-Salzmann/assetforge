import * as THREE from "three";

export const STATES = ["idle", "working"];

function stateOf(name, entry) {
  if (entry && STATES.includes(entry.state)) {
    return entry.state;
  }
  const prefix = name.split("_")[0];
  return STATES.includes(prefix) ? prefix : "idle";
}

function targetNodes(clip) {
  return [...new Set(clip.tracks.map((track) => THREE.PropertyBinding.parseTrackName(track.name).nodeName))];
}

export function describeClips(clips, entries = []) {
  const byName = new Map(entries.map((entry) => [entry.name, entry]));
  return clips.map((clip) => {
    const entry = byName.get(clip.name);
    return {
      clip,
      name: clip.name,
      state: stateOf(clip.name, entry),
      kind: entry?.kind ?? null,
      duration_s: entry?.duration_s ?? clip.duration,
      loop: entry?.loop ?? true,
      nodes: entry?.nodes ?? targetNodes(clip),
    };
  });
}

export function selectTracks(tracks, state, clip = "all") {
  if (clip !== "all") {
    return tracks.filter((track) => track.name === clip);
  }
  const chosen = tracks.filter((track) => track.state === state);
  if (state === "idle") {
    return chosen;
  }
  const busy = new Set(chosen.flatMap((track) => track.nodes));
  const ambient = tracks.filter((track) => track.state === "idle" && !track.nodes.some((node) => busy.has(node)));
  return [...chosen, ...ambient];
}

export function phaseOf(key) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < key.length; index += 1) {
    hash ^= key.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash / 0x100000000;
}

export class ClipPlayer {
  constructor(root, tracks) {
    this.root = root;
    this.tracks = tracks;
    this.mixer = new THREE.AnimationMixer(root);
    this.actions = new Map(tracks.map((track) => [track.name, this.mixer.clipAction(track.clip)]));
    this.state = "idle";
    this.clip = "all";
    this.paused = false;
    this.active = [];
  }

  select(state, clip = "all") {
    this.state = state;
    this.clip = clip;
    this.active = selectTracks(this.tracks, state, clip);
    for (const action of this.actions.values()) {
      action.stop();
    }
    for (const track of this.active) {
      const action = this.actions.get(track.name);
      action.setLoop(track.loop ? THREE.LoopRepeat : THREE.LoopOnce, Infinity);
      action.clampWhenFinished = !track.loop;
      action.play();
    }
    this.mixer.setTime(0);
  }

  windSpeed(speed) {
    for (const track of this.tracks) {
      if (track.kind === "skinned" && track.state === "idle") {
        this.actions.get(track.name).timeScale = speed;
      }
    }
  }

  get duration() {
    return Math.max(0, ...this.active.map((track) => track.duration_s));
  }

  get time() {
    const duration = this.duration;
    return duration > 0 ? this.mixer.time % duration : 0;
  }

  update(delta) {
    if (!this.paused) {
      this.mixer.update(delta);
    }
  }

  seek(seconds) {
    this.mixer.setTime(seconds);
  }

  phase(fraction) {
    for (const track of this.active) {
      this.actions.get(track.name).time = fraction * track.duration_s;
    }
    this.mixer.update(0);
  }

  dispose() {
    this.mixer.stopAllAction();
    this.mixer.uncacheRoot(this.root);
  }
}

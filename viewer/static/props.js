import * as THREE from "three";

const REDUCED_LOD = /_LOD[1-9]\d*$/;

function isReduced(node) {
  for (let walk = node; walk; walk = walk.parent) {
    if (REDUCED_LOD.test(walk.name)) {
      return true;
    }
  }
  return false;
}

function placementMatrix(entry) {
  const [x, y, z] = entry.translation;
  const [qx, qy, qz, qw] = entry.rotation;
  return new THREE.Matrix4().compose(
    new THREE.Vector3(x, y, z),
    new THREE.Quaternion(qx, qy, qz, qw),
    new THREE.Vector3().setScalar(entry.scale),
  );
}

export async function attachProps(loader, parent, props) {
  const group = new THREE.Group();
  group.name = "props";
  const byGlb = new Map();
  for (const entry of props ?? []) {
    if (!byGlb.has(entry.glb)) {
      byGlb.set(entry.glb, []);
    }
    byGlb.get(entry.glb).push(entry);
  }
  const failed = [];
  let count = 0;
  for (const [url, entries] of byGlb) {
    let gltf;
    try {
      gltf = await loader.loadAsync(url);
    } catch (error) {
      failed.push(...entries.map((entry) => entry.mesh));
      continue;
    }
    gltf.scene.updateMatrixWorld(true);
    const meshes = [];
    gltf.scene.traverse((node) => {
      if (node.isMesh && !isReduced(node)) {
        meshes.push(node);
      }
    });
    for (const mesh of meshes) {
      const instanced = new THREE.InstancedMesh(mesh.geometry, mesh.material, entries.length);
      instanced.name = `prop_${entries[0].mesh}_${mesh.name}`;
      entries.forEach((entry, index) => {
        instanced.setMatrixAt(index, placementMatrix(entry).multiply(mesh.matrixWorld));
      });
      instanced.instanceMatrix.needsUpdate = true;
      instanced.computeBoundingSphere();
      instanced.castShadow = true;
      instanced.receiveShadow = true;
      group.add(instanced);
    }
    count += entries.length;
  }
  parent.add(group);
  return { group, count, failed };
}

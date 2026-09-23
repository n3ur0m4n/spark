import * as THREE from "three";
import { ExtSplats, SparkRenderer, SplatMesh } from "/src/index.ts";

const checks = [];
const errors = [];
window.addEventListener("error", (event) => errors.push(event.message));
window.addEventListener("unhandledrejection", (event) =>
  errors.push(String(event.reason)),
);
let renderer;
let spark;
let source;
try {
  renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setSize(160, 120);
  document.body.append(renderer.domElement);
  const gl = renderer.getContext();
  const extension = gl.getExtension("WEBGL_debug_renderer_info");
  const adapter = extension
    ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL)
    : null;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, 4 / 3, 0.1, 100);
  spark = new SparkRenderer({
    renderer,
    enableLod: false,
    autoUpdate: false,
    minSortIntervalMs: 0,
  });
  scene.add(spark);
  const position = new THREE.Vector3(0, 0, -3);
  const scales = new THREE.Vector3(0.2, 0.2, 0.2);
  const rotation = new THREE.Quaternion();
  const color = new THREE.Color(1, 0, 0);
  source = new ExtSplats({
    construct: (data) => data.pushSplat(position, scales, rotation, 0.5, color),
  });
  await source.initialized;
  const mesh = new SplatMesh({ extSplats: source });
  await mesh.initialized;
  scene.add(mesh);

  async function measure(name, expectedDepth) {
    mesh.needsUpdate = true;
    await spark.update({ scene, camera });
    renderer.render(scene, camera);
    const current = spark.current;
    const readback = new Uint32Array(current.maxSplats);
    await spark.readbackDepth({
      current,
      renderer,
      numSplats: current.numSplats,
      readback,
    });
    const depth = new Float32Array(readback.buffer)[0];
    const exactSource = source.textures.every((texture, index) => {
      if (texture === ExtSplats.emptyTexture) return false;
      renderer.initTexture(texture);
      const saved = gl.getParameter(gl.FRAMEBUFFER_BINDING);
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTextureLayer(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        renderer.properties.get(texture).__webglTexture,
        0,
        0,
      );
      const pixels = new Uint32Array(4);
      gl.readPixels(0, 0, 1, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, pixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, saved);
      gl.deleteFramebuffer(framebuffer);
      return pixels.every(
        (value, word) => value === source.extArrays[index][word],
      );
    });
    const glError = gl.getError();
    checks.push({
      name,
      depth,
      expectedDepth,
      exactSource,
      count: source.numSplats,
      capacity: source.maxSplats,
      arrayLengths: source.extArrays.map((array) => array.length),
      textureSizes: source.textures.map((texture) => [
        texture.image.width,
        texture.image.height,
        texture.image.depth,
      ]),
      cpuCenter: source.numSplats ? source.getSplat(0).center.toArray() : null,
      glError,
      pass:
        exactSource &&
        glError === gl.NO_ERROR &&
        Math.abs(depth - expectedDepth) < 0.0001,
    });
  }

  await measure("constructed source", 3);
  const allocation = source.extArrays[0].buffer;
  source.setSplat(0, position.setZ(-17), scales, rotation, 0.5, color);
  await measure("in-place mutation", 17);
  checks.push({
    name: "mutation retains allocation",
    pass: allocation === source.extArrays[0].buffer,
  });
  const versions = source.textures.map((texture) => texture.version);
  await measure("unchanged source", 17);
  checks.push({
    name: "unchanged source skips upload",
    pass: source.textures.every(
      (texture, i) => texture.version === versions[i],
    ),
  });

  const arrays = source.extArrays.map((array) => {
    const padded = new Uint32Array(array.length + 8);
    const view = padded.subarray(4, array.length + 4);
    view.set(array);
    return view;
  });
  source.initialize({ extArrays: arrays, numSplats: 1 });
  await measure("replaced views with byte offsets", 17);
  checks.push({
    name: "Uint32 view ownership",
    pass: source.textures.every(
      (texture, i) => texture.image.data === arrays[i],
    ),
  });

  let disposals = 0;
  for (const texture of source.textures)
    texture.addEventListener("dispose", () => disposals++);
  source.ensureSplats(source.maxSplats + 1);
  await measure("allocation growth", 17);
  checks.push({
    name: "growth disposes both old textures",
    pass: disposals === 2,
  });
  const view = source.extArrays[0];
  new Float32Array(view.buffer, view.byteOffset, view.length)[2] = -23;
  source.needsUpdate = true;
  await measure("explicit raw-array update", 23);

  await source.reinitialize({
    construct: (data) =>
      data.pushSplat(position.setZ(-11), scales, rotation, 0.5, color),
  });
  await source.initialized;
  await measure("reinitialize constructed source", 11);
  checks.push({
    name: "WebGL error state",
    pass: gl.getError() === gl.NO_ERROR,
  });
  window.__extTextureValidation = {
    ready: true,
    adapter,
    checks,
    errors,
    pass: checks.every((check) => check.pass) && errors.length === 0,
  };
} catch (error) {
  window.__extTextureValidation = {
    ready: false,
    error: String(error),
    checks,
    errors,
    pass: false,
  };
} finally {
  spark?.dispose();
  source?.dispose();
  renderer?.dispose();
}

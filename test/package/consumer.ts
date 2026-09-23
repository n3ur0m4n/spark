import {
  type SparkRenderStats,
  SparkRenderer,
  type SparkRendererOptions,
  type SplatMesh,
  getSparkRendererCapabilities,
} from "@sparkjsdev/spark";
import type * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";

// Compile against the packed public declarations, not the source checkout.
type IsAny<T> = 0 extends 1 & T ? true : false;
const rendererTypeMustResolve: false = null as unknown as IsAny<
  typeof SparkRenderer
>;
export function legacyMeshUpdate(
  mesh: SplatMesh,
  renderer: THREE.WebGLRenderer,
  camera: THREE.Camera,
  viewToWorld: THREE.Matrix4,
  renderSize: THREE.Vector2,
) {
  // Existing callers can pass the complete frame context as an object literal.
  mesh.update({
    renderer,
    object: mesh,
    time: 0,
    deltaTime: 0,
    viewToWorld,
    camera,
    renderSize,
    globalEdits: [],
  });
}
export function classicWebGL(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
) {
  const options: SparkRendererOptions = { renderer };
  const spark: SparkRenderer = new SparkRenderer(options);
  const material: THREE.ShaderMaterial = spark.material;
  const stats: SparkRenderStats = spark.getRenderStats();
  const measured: Promise<SparkRenderStats> = spark.readRenderStatsAsync();
  const context: WebGLRenderingContext | WebGL2RenderingContext =
    spark.renderer.getContext();
  // biome-ignore lint/suspicious/noConfusingVoidType: Assert the existing synchronous render signature.
  const sync: void = spark.render(scene, camera);
  const submitted: Promise<void> = spark.renderAsync(scene, camera);
  return { context, sync, submitted };
}
export function nativeWebGPU(
  renderer: WebGPURenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
) {
  const options: SparkRendererOptions<WebGPURenderer> = { renderer };
  const spark = new SparkRenderer(options);
  const stats: SparkRenderStats = spark.getRenderStats();
  const measured: Promise<SparkRenderStats> = spark.readRenderStatsAsync();
  const native: WebGPURenderer = spark.renderer;
  // biome-ignore lint/suspicious/noConfusingVoidType: Assert the existing synchronous render signature.
  const sync: void = spark.render(scene, camera);
  const submitted: Promise<void> = spark.renderAsync(scene, camera);
  // @ts-expect-error A native renderer must not be typed as classic WebGL.
  const incompatible: THREE.WebGLRenderer = spark.renderer;
  return {
    native,
    sync,
    submitted,
    capabilities: getSparkRendererCapabilities(renderer),
  };
}

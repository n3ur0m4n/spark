import type * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";

export type SparkHostRenderer = THREE.WebGLRenderer | WebGPURenderer;
export type RendererAdapter =
  | { kind: "webgl"; renderer: THREE.WebGLRenderer }
  | { kind: "webgpu"; renderer: WebGPURenderer; device: GPUDevice };

export function isWebGLRenderer(
  renderer: SparkHostRenderer,
): renderer is THREE.WebGLRenderer {
  return "isWebGLRenderer" in renderer && renderer.isWebGLRenderer === true;
}

export function createRendererAdapter(
  renderer: SparkHostRenderer,
): RendererAdapter {
  if (isWebGLRenderer(renderer)) {
    return { kind: "webgl", renderer };
  }
  if (
    "backend" in renderer &&
    "device" in renderer.backend &&
    typeof GPUDevice !== "undefined" &&
    renderer.backend.device instanceof GPUDevice
  ) {
    return { kind: "webgpu", renderer, device: renderer.backend.device };
  }
  throw new Error(
    "Spark: initialize a native WebGPURenderer before construction. forceWebGL requires THREE.WebGLRenderer.",
  );
}

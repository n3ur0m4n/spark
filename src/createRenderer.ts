import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import { getSparkRendererCapabilities } from "./RendererCapabilities";

/**
 * Features the native WebGPU backend does not support yet. createRenderer()
 * falls back to WebGL when `needs` lists one of them; remove entries as native
 * support lands. They mirror the features SparkNativeRenderer and
 * getSparkRendererCapabilities() reject.
 */
export const WEBGPU_UNSUPPORTED_FEATURES = [
  "xr",
  "modifiers",
  "skinning",
  "specializedTargets",
  "covarianceSplats",
  "transparentMeshes",
  "transparentSprites",
  "logarithmicDepth",
  "reversedDepth",
  "msaa",
  "customViewport",
  "scissorTest",
  "manualClear",
  "nonuniformSplatTransforms",
  "arrayCameras",
] as const;

export type CreateRendererOptions = THREE.WebGLRendererParameters &
  NonNullable<ConstructorParameters<typeof WebGPURenderer>[0]> & {
    /** Features the app relies on; any unsupported by native WebGPU selects WebGL. */
    needs?: readonly string[];
    /** Always create a THREE.WebGLRenderer. */
    forceWebGL?: boolean;
  };

export type CreateRendererResult =
  | { renderer: WebGPURenderer; backend: "webgpu" }
  | { renderer: THREE.WebGLRenderer; backend: "webgl" };

/**
 * Create the host renderer for SparkRenderer: an initialized native
 * WebGPURenderer when the browser has a WebGPU adapter and nothing in `needs`
 * requires WebGL, otherwise a THREE.WebGLRenderer.
 */
export async function createRenderer(
  options: CreateRendererOptions = {},
): Promise<CreateRendererResult> {
  const { needs = [], forceWebGL = false, ...params } = options;
  const unsupported: readonly string[] = WEBGPU_UNSUPPORTED_FEATURES;
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!forceWebGL && gpu && !needs.some((need) => unsupported.includes(need))) {
    const adapter = await gpu.requestAdapter().catch(() => null);
    if (adapter) {
      // Dynamic so WebGL-only bundles never load three/webgpu.
      const { WebGPURenderer } = await import("three/webgpu");
      const renderer = new WebGPURenderer(params);
      try {
        await renderer.init();
        // WebGPURenderer silently falls back to its WebGL2 backend.
        if (getSparkRendererCapabilities(renderer).backend === "webgpu") {
          return { renderer, backend: "webgpu" };
        }
      } catch {}
      renderer.dispose();
    }
  }
  return { renderer: new THREE.WebGLRenderer(params), backend: "webgl" };
}

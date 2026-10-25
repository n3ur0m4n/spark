import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
/**
 * Features the native WebGPU backend does not support yet. createRenderer()
 * falls back to WebGL when `needs` lists one of them; remove entries as native
 * support lands. The first group mirrors the features SparkNativeRenderer and
 * getSparkRendererCapabilities() reject; the second is fork-only WebGL work
 * (pickAsync, stochastic transparency, sort cone culling) the native path
 * does not implement.
 */
export declare const WEBGPU_UNSUPPORTED_FEATURES: readonly ["paging", "xr", "modifiers", "editing", "skinning", "specializedTargets", "covarianceSplats", "transparentMeshes", "transparentSprites", "logarithmicDepth", "reversedDepth", "msaa", "customViewport", "scissorTest", "manualClear", "nonuniformSplatTransforms", "arrayCameras", "pickAsync", "stochastic", "sortCull"];
export type CreateRendererOptions = THREE.WebGLRendererParameters & NonNullable<ConstructorParameters<typeof WebGPURenderer>[0]> & {
    /** Features the app relies on; any unsupported by native WebGPU selects WebGL. */
    needs?: readonly string[];
    /** Always create a THREE.WebGLRenderer. */
    forceWebGL?: boolean;
};
export type CreateRendererResult = {
    renderer: WebGPURenderer;
    backend: "webgpu";
} | {
    renderer: THREE.WebGLRenderer;
    backend: "webgl";
};
/**
 * Create the host renderer for SparkRenderer: an initialized native
 * WebGPURenderer when the browser has a WebGPU adapter and nothing in `needs`
 * requires WebGL, otherwise a THREE.WebGLRenderer.
 */
export declare function createRenderer(options?: CreateRendererOptions): Promise<CreateRendererResult>;

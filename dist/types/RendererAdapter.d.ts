import type * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
export type SparkHostRenderer = THREE.WebGLRenderer | WebGPURenderer;
export type RendererAdapter = {
    kind: "webgl";
    renderer: THREE.WebGLRenderer;
} | {
    kind: "webgpu";
    renderer: WebGPURenderer;
    device: GPUDevice;
};
export declare function isWebGLRenderer(renderer: SparkHostRenderer): renderer is THREE.WebGLRenderer;
export declare function createRendererAdapter(renderer: SparkHostRenderer): RendererAdapter;

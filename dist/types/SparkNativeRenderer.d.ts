import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { SparkHostRenderer } from "./RendererAdapter";
import type { SparkRenderStats } from "./SparkRenderStats";
import type { SparkRenderer } from "./SparkRenderer";
import { SplatMesh } from "./SplatMesh";
import { WebGPUSplatBackend } from "./WebGPUSplatBackend";
/** The native submission path; the WebGL renderer and LoD traversal stay unchanged. */
export declare class SparkNativeRenderer {
    private readonly owner;
    private readonly renderer;
    readonly backend: WebGPUSplatBackend;
    private composite;
    private queue;
    private disposed;
    private renderVisible;
    private readonly hostSize;
    private readonly hostViewport;
    private sequence;
    private currentSignature;
    private updateQueue;
    private worker;
    private ordering;
    private readonly identities;
    private nextIdentity;
    private stats;
    getRenderStats(): SparkRenderStats;
    private isVisible;
    readRenderStatsAsync(): Promise<SparkRenderStats>;
    private identity;
    constructor(owner: SparkRenderer<SparkHostRenderer>, renderer: WebGPURenderer, device: GPUDevice);
    assertSupported(scene: THREE.Scene, camera: THREE.Camera): void;
    /** Each call keeps its camera snapshot and has its own submission promise. */
    renderAsync(scene: THREE.Scene, camera: THREE.Camera): Promise<void>;
    private updateUniforms;
    /** Runs ordinary mesh callbacks, then snapshots and commits one complete generation. */
    update(scene: THREE.Scene, camera: THREE.Camera): Promise<SplatMesh[]>;
    private updateFrame;
    dispose(): void;
}

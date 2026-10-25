import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
import type { SparkHostRenderer } from "./RendererAdapter";
import type { SparkRenderStats } from "./SparkRenderStats";
import type { SparkRenderer } from "./SparkRenderer";
import { SplatMesh } from "./SplatMesh";
import { type WebGPUPickPixels, WebGPUSplatBackend } from "./WebGPUSplatBackend";
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
    /**
     * pickAsync() render: the w x h CSS-pixel crop at (ox, oy) of the
     * fullW x fullH view, drawn from the last submitted frame's ordering. Queued
     * after pending renders so it sees the ordering they committed.
     */
    pickPixels(camera: THREE.PerspectiveCamera | THREE.OrthographicCamera, crop: {
        fullW: number;
        fullH: number;
        ox: number;
        oy: number;
        w: number;
        h: number;
        alphaThreshold: number;
        farBig: number;
        solid: boolean;
    }): Promise<WebGPUPickPixels>;
    private updateUniforms;
    /** Runs ordinary mesh callbacks, then snapshots and commits one complete generation. */
    update(scene: THREE.Scene, camera: THREE.Camera): Promise<SplatMesh[]>;
    private updateFrame;
    /** Paged meshes read SplatPager's page pool through their LoD selection. */
    private pagedGeneration;
    /** Snapshot of the compiled SplatEdits that SplatMesh.update() just encoded. */
    private meshEdits;
    dispose(): void;
}

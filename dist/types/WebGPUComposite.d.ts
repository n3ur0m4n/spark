import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
export type CompositeAttachments = {
    color: GPUTexture;
    depth: GPUTexture;
};
/** Owns the linear color/depth target shared by Three.js and the splat pass. */
export declare class WebGPUComposite {
    private readonly target;
    private readonly quad;
    private readonly configureOutput;
    private outputKey;
    private constructor();
    static create(): Promise<WebGPUComposite>;
    render(renderer: WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera, composite: (attachments: CompositeAttachments) => void): void;
    dispose(): void;
}

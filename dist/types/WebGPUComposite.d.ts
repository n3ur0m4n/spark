import * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
/**
 * srgb: how the splat pass must blend into `color` (null = linear, as is).
 * "bytes": blend on the raw bytes of an sRGB-format color target.
 * "layer": `color` is a cleared 8-bit layer of sRGB-encoded premultiplied
 * splats, composited over the tone-mapped scene by the output quad.
 */
export type CompositeAttachments = {
    color: GPUTexture;
    depth: GPUTexture;
    srgb: "bytes" | "layer" | null;
};
/** Owns the linear color/depth target shared by Three.js and the splat pass. */
export declare class WebGPUComposite {
    private readonly target;
    private readonly splatTarget;
    private readonly lateTarget;
    private readonly quad;
    private readonly copyQuad;
    private readonly configureOutput;
    private outputKey;
    private hdr;
    private constructor();
    static create(): Promise<WebGPUComposite>;
    render(renderer: WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera, composite: (attachments: CompositeAttachments) => void, splitOrder?: number, output?: THREE.RenderTarget | null, srgbBlend?: boolean): void;
    private renderLate;
    dispose(): void;
}

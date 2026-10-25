import type * as THREE from "three";
/** Worker scratch may be transferred only after its texture owner releases it. */
export declare class OrderingBufferPool {
    private spare;
    private readonly observed;
    take(length: number): Uint32Array<ArrayBuffer>;
    private release;
    /** Keep current CPU data attached for initial upload and context restoration. */
    attach(texture: THREE.DataTexture, data: Uint32Array): void;
    clear(): void;
}

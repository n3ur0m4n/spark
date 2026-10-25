import type * as THREE from "three";
import type { WebGPURenderer } from "three/webgpu";
export type SparkRendererCapabilities = {
    backend: "webgl" | "webgpu" | "webgpu-force-webgl" | "uninitialized";
    nativeStatus: "reference-preview" | null;
    features: Readonly<{
        paging: boolean;
        xr: boolean;
        modifiers: boolean;
        editing: boolean;
        skinning: boolean;
        specializedTargets: boolean;
        covarianceSplats: boolean;
        transparentMeshes: boolean;
        transparentSprites: boolean;
        logarithmicDepth: boolean;
        reversedDepth: boolean;
        msaa: boolean;
        customViewport: boolean;
        scissorTest: boolean;
        manualClear: boolean;
        nonuniformSplatTransforms: boolean;
        multipleSplatMeshes: boolean;
        arrayCameras: boolean;
    }>;
    limits: Readonly<{
        maxBufferSize: number | null;
        maxStorageBufferBindingSize: number | null;
        maxTextureDimension2D: number | null;
    }>;
};
/** Inspect the host renderer before allocating Spark resources. */
export declare function getSparkRendererCapabilities(renderer: THREE.WebGLRenderer | WebGPURenderer): SparkRendererCapabilities;

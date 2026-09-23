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
export function getSparkRendererCapabilities(
  renderer: THREE.WebGLRenderer | WebGPURenderer,
): SparkRendererCapabilities {
  const native = "backend" in renderer ? renderer.backend : null;
  const device: GPUDevice | null =
    native &&
    "device" in native &&
    typeof GPUDevice !== "undefined" &&
    native.device instanceof GPUDevice
      ? native.device
      : null;
  const webgl =
    "isWebGLRenderer" in renderer && renderer.isWebGLRenderer === true;
  const forcedWebGL =
    native && "isWebGLBackend" in native && native.isWebGLBackend === true;
  return {
    backend: webgl
      ? "webgl"
      : device
        ? "webgpu"
        : forcedWebGL
          ? "webgpu-force-webgl"
          : "uninitialized",
    nativeStatus: device ? "reference-preview" : null,
    features: {
      paging: webgl,
      xr: webgl,
      modifiers: webgl,
      editing: webgl,
      skinning: webgl,
      specializedTargets: webgl,
      covarianceSplats: webgl,
      transparentMeshes: webgl,
      transparentSprites: webgl,
      logarithmicDepth: webgl,
      reversedDepth: webgl,
      msaa: webgl,
      customViewport: webgl,
      scissorTest: webgl,
      manualClear: webgl,
      nonuniformSplatTransforms: webgl,
      multipleSplatMeshes: webgl || !!device,
      arrayCameras: webgl,
    },
    limits: {
      maxBufferSize: device?.limits.maxBufferSize ?? null,
      maxStorageBufferBindingSize:
        device?.limits.maxStorageBufferBindingSize ?? null,
      maxTextureDimension2D: device?.limits.maxTextureDimension2D ?? null,
    },
  };
}

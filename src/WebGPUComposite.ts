import * as THREE from "three";
import type { QuadMesh, WebGPURenderer } from "three/webgpu";

export type CompositeAttachments = { color: GPUTexture; depth: GPUTexture };

/** Owns the linear color/depth target shared by Three.js and the splat pass. */
export class WebGPUComposite {
  private outputKey = "";

  private constructor(
    private readonly target: THREE.RenderTarget,
    private readonly quad: QuadMesh,
    private readonly configureOutput: (
      toneMapping: THREE.ToneMapping,
      colorSpace: string,
    ) => void,
  ) {}

  static async create(): Promise<WebGPUComposite> {
    const [
      { MeshBasicNodeMaterial, QuadMesh },
      { texture, vec4, renderOutput },
    ] = await Promise.all([import("three/webgpu"), import("three/tsl")]);
    const target = new THREE.RenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      depthBuffer: true,
      stencilBuffer: false,
      samples: 0,
    });
    target.depthTexture = new THREE.DepthTexture(1, 1, THREE.UnsignedIntType);
    const sample = texture(target.texture);
    const material = new MeshBasicNodeMaterial({
      transparent: true,
      blending: THREE.NoBlending,
      depthTest: false,
      depthWrite: false,
    });
    return new WebGPUComposite(
      target,
      new QuadMesh(material),
      (toneMapping, colorSpace) => {
        // The shared target stores premultiplied linear color. Nonlinear color
        // conversion and tone mapping must operate on the straight color, then
        // premultiply for the host canvas (WebGPU alphaMode: "premultiplied").
        // Give renderOutput opaque straight color: newer Three.js versions
        // unpremultiply/premultiply internally, whereas r180 does neither.
        // Alpha 1 makes that difference irrelevant to this conversion.
        const straight = vec4(sample.rgb.div(sample.a.max(1e-6)), 1);
        const converted = renderOutput(straight, toneMapping, colorSpace);
        material.fragmentNode = vec4(converted.rgb.mul(sample.a), sample.a);
        material.needsUpdate = true;
      },
    );
  }

  render(
    renderer: WebGPURenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
    composite: (attachments: CompositeAttachments) => void,
  ): void {
    const previous = renderer.getRenderTarget();
    if (previous || renderer.xr.isPresenting) {
      throw new Error(
        "Spark: native composition currently requires the canvas target outside XR; use WebGL for this scene.",
      );
    }
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    this.target.setSize(size.x, size.y);
    try {
      renderer.setRenderTarget(this.target);
      renderer.render(scene, camera);
      const backend = renderer.backend;
      if (!("get" in backend) || typeof backend.get !== "function") {
        throw new Error(
          "Spark: unsupported Three.js texture interop; use WebGL.",
        );
      }
      const get = backend.get.bind(backend);
      const getTexture = (source: THREE.Texture): GPUTexture => {
        const data: unknown = get(source);
        if (
          !data ||
          typeof data !== "object" ||
          !("texture" in data) ||
          !(data.texture instanceof GPUTexture)
        ) {
          throw new Error(
            "Spark: Three.js did not expose a composite attachment.",
          );
        }
        return data.texture;
      };
      const color = getTexture(this.target.texture);
      if (!this.target.depthTexture)
        throw new Error("Spark: composite depth texture is unavailable.");
      const depth = getTexture(this.target.depthTexture);
      composite({ color, depth });
    } finally {
      renderer.setRenderTarget(previous);
    }
    const toneMapping = renderer.toneMapping;
    const colorSpace = renderer.outputColorSpace;
    const outputKey = `${toneMapping}:${colorSpace}`;
    if (outputKey !== this.outputKey) {
      this.configureOutput(toneMapping, colorSpace);
      this.outputKey = outputKey;
    }
    try {
      // The quad owns the final conversion. Suppress Three's automatic output
      // pass for this draw so it cannot convert or tone-map the result twice.
      renderer.toneMapping = THREE.NoToneMapping;
      renderer.outputColorSpace = THREE.ColorManagement.workingColorSpace;
      this.quad.render(renderer);
    } finally {
      renderer.toneMapping = toneMapping;
      renderer.outputColorSpace = colorSpace;
    }
  }

  dispose(): void {
    this.target.dispose();
    const materials = Array.isArray(this.quad.material)
      ? this.quad.material
      : [this.quad.material];
    for (const material of materials) material.dispose();
  }
}

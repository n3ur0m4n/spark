import * as THREE from "three";
import type { QuadMesh, WebGPURenderer } from "three/webgpu";

export type CompositeAttachments = { color: GPUTexture; depth: GPUTexture };

type RenderObjectArgs = Parameters<
  NonNullable<ReturnType<WebGPURenderer["getRenderObjectFunction"]>>
>;

/** Owns the linear color/depth target shared by Three.js and the splat pass. */
export class WebGPUComposite {
  private outputKey = "";
  private hdr = true;

  private constructor(
    private readonly target: THREE.RenderTarget,
    private readonly quad: QuadMesh,
    private readonly copyQuad: QuadMesh,
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
    // External targets get the canvas image (same tone mapping) as linear
    // premultiplied color; an sRGB target texture encodes it on write.
    const copyMaterial = new MeshBasicNodeMaterial({
      transparent: true,
      blending: THREE.NoBlending,
      depthTest: false,
      depthWrite: false,
    });
    return new WebGPUComposite(
      target,
      new QuadMesh(material),
      new QuadMesh(copyMaterial),
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
        const linear = renderOutput(
          straight,
          toneMapping,
          THREE.LinearSRGBColorSpace,
        );
        copyMaterial.fragmentNode = vec4(linear.rgb.mul(sample.a), sample.a);
        copyMaterial.needsUpdate = true;
      },
    );
  }

  render(
    renderer: WebGPURenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
    composite: (attachments: CompositeAttachments) => void,
    splitOrder = Number.POSITIVE_INFINITY,
    output: THREE.RenderTarget | null = null,
  ): void {
    const previous = renderer.getRenderTarget();
    if (previous || renderer.xr.isPresenting) {
      throw new Error(
        "Spark: native composition currently requires the canvas target outside XR; use WebGL for this scene.",
      );
    }
    // Without tone mapping the canvas clamps to [0, 1] anyway, so an sRGB
    // 8-bit target (linear blending, half the bandwidth of half float) is
    // enough. The splat pass is fill-rate bound: this is its main cost.
    const hdr = renderer.toneMapping !== THREE.NoToneMapping;
    if (hdr !== this.hdr) {
      this.hdr = hdr;
      const texture = this.target.texture;
      texture.type = hdr ? THREE.HalfFloatType : THREE.UnsignedByteType;
      texture.colorSpace = hdr ? THREE.NoColorSpace : THREE.SRGBColorSpace;
      this.target.dispose();
    }
    const size = output
      ? new THREE.Vector2(output.width, output.height)
      : renderer.getDrawingBufferSize(new THREE.Vector2());
    this.target.setSize(size.x, size.y);
    // Like the WebGL transparent list, transparent objects with renderOrder at
    // or above the SparkRenderer's draw after the splats: pass A skips them,
    // pass B (no clear, same color/depth) draws only them.
    const hostFunction = renderer.getRenderObjectFunction();
    const draw = (...args: RenderObjectArgs) =>
      hostFunction ? hostFunction(...args) : renderer.renderObject(...args);
    const late = (object: THREE.Object3D, material: THREE.Material) =>
      material.transparent && object.renderOrder >= splitOrder;
    let deferred = false;
    try {
      renderer.setRenderTarget(this.target);
      renderer.setRenderObjectFunction(((...args: RenderObjectArgs) => {
        if (late(args[0], args[4])) deferred = true;
        else draw(...args);
      }) as Parameters<WebGPURenderer["setRenderObjectFunction"]>[0]);
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
      if (deferred) this.renderLate(renderer, scene, camera, late, draw);
    } finally {
      renderer.setRenderObjectFunction(
        hostFunction as Parameters<
          WebGPURenderer["setRenderObjectFunction"]
        >[0],
      );
      renderer.setRenderTarget(previous);
    }
    const toneMapping = renderer.toneMapping;
    const colorSpace = renderer.outputColorSpace;
    const outputKey = `${toneMapping}:${colorSpace}`;
    if (outputKey !== this.outputKey) {
      this.configureOutput(toneMapping, colorSpace);
      this.outputKey = outputKey;
    }
    if (output) {
      // A target that is not the output target gets no automatic tone
      // mapping or color conversion from Three.js.
      try {
        renderer.setRenderTarget(output);
        this.copyQuad.render(renderer);
      } finally {
        renderer.setRenderTarget(previous);
      }
      return;
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

  private renderLate(
    renderer: WebGPURenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
    late: (object: THREE.Object3D, material: THREE.Material) => boolean,
    draw: (...args: RenderObjectArgs) => void,
  ): void {
    const state = {
      opaque: renderer.opaque,
      autoClear: renderer.autoClear,
      autoClearColor: renderer.autoClearColor,
      autoClearDepth: renderer.autoClearDepth,
      autoClearStencil: renderer.autoClearStencil,
      background: scene.background,
    };
    try {
      renderer.opaque = false;
      renderer.autoClear = false;
      renderer.autoClearColor = false;
      renderer.autoClearDepth = false;
      renderer.autoClearStencil = false;
      // A color background forces a clear even without autoClear.
      scene.background = null;
      renderer.setRenderObjectFunction(((...args: RenderObjectArgs) => {
        if (late(args[0], args[4])) draw(...args);
      }) as Parameters<WebGPURenderer["setRenderObjectFunction"]>[0]);
      renderer.render(scene, camera);
    } finally {
      renderer.opaque = state.opaque;
      renderer.autoClear = state.autoClear;
      renderer.autoClearColor = state.autoClearColor;
      renderer.autoClearDepth = state.autoClearDepth;
      renderer.autoClearStencil = state.autoClearStencil;
      scene.background = state.background;
    }
  }

  dispose(): void {
    this.target.dispose();
    for (const quad of [this.quad, this.copyQuad]) {
      const materials = Array.isArray(quad.material)
        ? quad.material
        : [quad.material];
      for (const material of materials) material.dispose();
    }
  }
}

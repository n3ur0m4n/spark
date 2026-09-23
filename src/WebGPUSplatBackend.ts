import { subscribeDeviceLost } from "./WebGPUDeviceLost";
import {
  type NativeMeshGeneration,
  WebGPUGeneration,
} from "./WebGPUGeneration";
import { calcDispatchSize } from "./WebGPUUtils";
import {
  splatDefinesWgsl,
  splatFragmentWgsl,
  splatVertexWgsl,
} from "./shaders-wgsl";
import { getTextureSize } from "./utils";

type TextureExtent = { width: number; height: number; depth: number };

type WebGPURenderSlot = {
  generation: number;
  orderingGeneration: number;
  generatedSpan: number | null;
  nativeDepths: GPUBuffer | null;
  splatTexture: GPUTexture | null;
  splatTextureView: GPUTextureView | null;
  splatTexSize: TextureExtent | null;
  splatTexture2: GPUTexture | null;
  splatTextureView2: GPUTextureView | null;
  splatTex2Size: TextureExtent | null;
  orderingTexture: GPUTexture | null;
  orderingRows: number;
  bindGroup: GPUBindGroup | null;
  bindGroupDirty: boolean;
  bindGroupVersion: number;
};

export type WebGPUOrderingTarget = "active" | "working" | "write";

function makeRenderSlot(): WebGPURenderSlot {
  return {
    generation: 0,
    orderingGeneration: 0,
    generatedSpan: null,
    nativeDepths: null,
    splatTexture: null,
    splatTextureView: null,
    splatTexSize: null,
    splatTexture2: null,
    splatTextureView2: null,
    splatTex2Size: null,
    orderingTexture: null,
    orderingRows: 0,
    bindGroup: null,
    bindGroupDirty: true,
    bindGroupVersion: -1,
  };
}

/**
 * WebGPU rendering backend for SparkRenderer.
 *
 * Owns all GPU-specific state: pipeline, textures, buffers, and draw calls.
 * Generation evaluates source SH and LoD before this draw stage.
 */
export class WebGPUSplatBackend {
  private static readonly MAX_TIMESTAMP_PASSES = 8;
  device: GPUDevice;

  pipeline: GPURenderPipeline | null = null;
  bindGroupLayout: GPUBindGroupLayout | null = null;
  bindGroup: GPUBindGroup | null = null;
  private bindGroupLayoutVersion = 0;
  pipelineFormat: GPUTextureFormat | null = null;
  private pipelineHalvedAlphaState = false;
  private pipelinePremultipliedAlphaState = true;

  private renderSlots: [WebGPURenderSlot, WebGPURenderSlot] = [
    makeRenderSlot(),
    makeRenderSlot(),
  ];
  private activeSlotIndex = 0;
  private workingSlotIndex = 0;
  private pendingRenderSlot = false;
  private nextGeneration = 1;
  webgpuDebug = false;

  // Self-managed GPU textures (RGBA32UI)
  splatTexture: GPUTexture | null = null;
  splatTextureView: GPUTextureView | null = null;
  splatTexSize: { width: number; height: number; depth: number } | null = null;

  // Second texture for 32-byte extended splats (extSplats2, binding 4)
  splatTexture2: GPUTexture | null = null;
  splatTextureView2: GPUTextureView | null = null;
  splatTex2Size: { width: number; height: number; depth: number } | null = null;

  orderingTexture: GPUTexture | null = null;
  orderingRows = 0;
  private syncDepthReadbackBuffer: GPUBuffer | null = null;

  halvedAlpha = false;
  /**
   * Mirrors SparkRenderer's public `premultipliedAlpha` option. Drives both the
   * `PREMULTIPLIED_ALPHA` shader constant and the blend descriptor, so WebGPU
   * composites the same way WebGL does when an app opts out.
   */
  premultipliedAlpha = true;

  // Splat encoding: rgbMin, rgbMax, lnScaleMin, lnScaleMax
  // Set from PackedSplats.splatEncoding per frame; defaults match DEFAULT_SPLAT_ENCODING.
  splatEncoding: [number, number, number, number] = [0, 1, -12, 9];

  // Uniform buffers
  sparkUniformBuffer: GPUBuffer;
  projMatrixBuffer: GPUBuffer;
  fragUniformBuffer: GPUBuffer;

  // Quad geometry
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;

  // Own depth texture for the separate render pass
  depthTexture: GPUTexture | null = null;
  depthTextureView: GPUTextureView | null = null;
  /** Externally-owned scene depth (three.js), adopted via setExternalDepthTexture(). */
  private externalDepthTexture: GPUTexture | null = null;
  private externalDepthView: GPUTextureView | null = null;
  /** Count of frames where the splat draw was skipped, keyed by reason. */
  readonly drawSkipped: Record<string, number> = {};
  /** Reason recorded for the most recent skipped draw. */
  lastDrawSkippedReason: string | null = null;
  /** True once the WebGPU device has been reported lost. */
  deviceLost = false;
  /** Message from the device-lost event, if any. */
  deviceLostReason: string | null = null;
  /** Count of uncaptured WebGPU validation/out-of-memory errors observed. */
  uncapturedErrorCount = 0;
  /** Message of the first uncaptured error, retained for the debug snapshot. */
  firstUncapturedError: string | null = null;
  /** True after dispose(); guards in-flight async completions. */
  private disposed = false;
  private readonly releaseDeviceLost: () => void;
  /** Why the last offered external depth texture was rejected, if it was. */
  externalDepthRejectReason: string | null = null;
  depthWidth = 0;
  depthHeight = 0; // current write target (cycles 0 -> 1 -> 2 -> 0)

  private generatedOutputBuffer: GPUBuffer | null = null;
  private generatedOutputBuffer2: GPUBuffer | null = null;

  private nativeGeneration: WebGPUGeneration | null = null;
  private nativeFillPipeline: GPUComputePipeline | null = null;

  // Reusable typed arrays for uniform writes
  private sparkUniformData = new ArrayBuffer(208);
  private fragUniformData = new ArrayBuffer(48);

  private readonly onUncapturedError = (event: Event): void => {
    if (this.disposed) return;
    this.uncapturedErrorCount++;
    const message =
      (event as GPUUncapturedErrorEvent).error?.message ?? "unknown";
    if (!this.firstUncapturedError) {
      this.firstUncapturedError = message;
      console.error("Spark: uncaptured WebGPU error", message);
    }
  };

  constructor(device: GPUDevice) {
    this.device = device;

    // SparkUniforms: 208 bytes (see struct layout in splatVertex.wgsl)
    this.sparkUniformBuffer = device.createBuffer({
      label: "spark-uniforms",
      size: 208,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // projectionMatrix: mat4x4f = 64 bytes
    this.projMatrixBuffer = device.createBuffer({
      label: "spark-proj-matrix",
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // FragmentUniforms: 36 bytes, padded to 48
    this.fragUniformBuffer = device.createBuffer({
      label: "spark-frag-uniforms",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Quad geometry: 4 vertices, 6 indices
    const quadVertices = new Float32Array([-1, -1, 1, -1, 1, 1, -1, 1]);
    const quadIndices = new Uint16Array([0, 1, 2, 0, 2, 3]);

    this.vertexBuffer = device.createBuffer({
      label: "spark-quad-verts",
      size: quadVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.vertexBuffer, 0, quadVertices);

    this.indexBuffer = device.createBuffer({
      label: "spark-quad-idx",
      size: quadIndices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.indexBuffer, 0, quadIndices);

    this.releaseDeviceLost = subscribeDeviceLost(device, (info) => {
      if (this.disposed) return;
      this.deviceLost = true;
      this.deviceLostReason = `${info.reason}: ${info.message}`;
      console.error(`Spark: WebGPU device lost (${this.deviceLostReason})`);
    });
    device.addEventListener("uncapturederror", this.onUncapturedError);
  }

  private activeRenderSlot(): WebGPURenderSlot {
    return this.renderSlots[this.activeSlotIndex];
  }

  private writeRenderSlot(): WebGPURenderSlot {
    return this.pendingRenderSlot
      ? this.renderSlots[this.workingSlotIndex]
      : this.activeRenderSlot();
  }

  private depthRenderSlot(): WebGPURenderSlot {
    return this.writeRenderSlot();
  }

  private orderingRenderSlot(target: WebGPUOrderingTarget): WebGPURenderSlot {
    if (target === "active") {
      return this.activeRenderSlot();
    }
    if (target === "working") {
      return this.renderSlots[this.workingSlotIndex];
    }
    return this.writeRenderSlot();
  }

  private syncLegacyFields(
    slot: WebGPURenderSlot = this.activeRenderSlot(),
  ): void {
    this.splatTexture = slot.splatTexture;
    this.splatTextureView = slot.splatTextureView;
    this.splatTexSize = slot.splatTexSize;
    this.splatTexture2 = slot.splatTexture2;
    this.splatTextureView2 = slot.splatTextureView2;
    this.splatTex2Size = slot.splatTex2Size;
    this.orderingTexture = slot.orderingTexture;
    this.orderingRows = slot.orderingRows;
    this.bindGroup = slot.bindGroup;
  }

  private markRenderSlotsDirty(): void {
    for (const slot of this.renderSlots) {
      slot.bindGroupDirty = true;
    }
  }

  private debugLog(label: string, extra: Record<string, unknown> = {}): void {
    if (this.webgpuDebug) console.log("[Spark WebGPU]", label, extra);
  }

  beginRenderSlotUpdate(reason = "upload"): void {
    // Even the first update must be isolated: allocation or compilation can
    // fail before commit, and the active generation must stay unchanged.
    this.workingSlotIndex = 1 - this.activeSlotIndex;
    this.pendingRenderSlot = true;
    const slot = this.renderSlots[this.workingSlotIndex];
    slot.generation = this.nextGeneration++;
    slot.orderingGeneration = 0;
    slot.generatedSpan = null;
    slot.bindGroupDirty = true;
    this.syncLegacyFields(slot);
    this.debugLog("begin", { reason });
  }

  commitRenderSlot(): void {
    const slot = this.writeRenderSlot();
    if (
      slot.generatedSpan === null ||
      (slot.generatedSpan > 0 && !slot.splatTextureView) ||
      slot.orderingGeneration !== slot.generation
    )
      throw new Error("Spark: cannot commit incomplete native generation");

    const committedRenderSlot = this.pendingRenderSlot;
    if (this.pendingRenderSlot) {
      this.activeSlotIndex = this.workingSlotIndex;
    }
    this.pendingRenderSlot = false;
    this.syncLegacyFields(this.activeRenderSlot());
    this.debugLog("commit", { committedRenderSlot });
  }

  cancelRenderSlotUpdate(expectedGeneration?: number): boolean {
    if (!this.pendingRenderSlot) return false;
    const slot = this.renderSlots[this.workingSlotIndex];
    if (
      expectedGeneration !== undefined &&
      slot.generation !== expectedGeneration
    ) {
      return false;
    }
    this.pendingRenderSlot = false;
    this.workingSlotIndex = this.activeSlotIndex;
    this.syncLegacyFields(this.activeRenderSlot());
    this.debugLog("cancel", { expectedGeneration });
    return true;
  }

  hasPendingRenderSlot(): boolean {
    return this.pendingRenderSlot;
  }

  hasSplatData(): boolean {
    const active = this.activeRenderSlot();
    const write = this.writeRenderSlot();
    return !!(active.splatTextureView || write.splatTextureView);
  }

  hasDepthSplatData(): boolean {
    const slot = this.depthRenderSlot();
    return !!slot.splatTextureView;
  }

  getDepthSlotGeneration(): number {
    return this.depthRenderSlot().generation;
  }

  getActiveSlotGeneration(): number {
    return this.activeRenderSlot().generation;
  }

  getWorkingSlotGeneration(): number {
    return this.renderSlots[this.workingSlotIndex].generation;
  }

  /**
   * Back-to-front alpha blending for the splat draw.
   *
   * Mirrors what three.js programs for `NormalBlending`: premultiplied source
   * colors blend with `ONE`, straight-alpha ones with `SRC_ALPHA`. The alpha
   * channel accumulates identically either way.
   */
  private splatBlendState(premultiplied: boolean): GPUBlendState {
    return {
      color: {
        srcFactor: premultiplied ? "one" : "src-alpha",
        dstFactor: "one-minus-src-alpha",
        operation: "add",
      },
      alpha: {
        srcFactor: "one",
        dstFactor: "one-minus-src-alpha",
        operation: "add",
      },
    };
  }

  /**
   * Create the render pipeline on first use (needs canvas format).
   */
  ensurePipeline(colorFormat: GPUTextureFormat): void {
    if (
      this.pipeline &&
      this.pipelineFormat === colorFormat &&
      this.pipelineHalvedAlphaState === this.halvedAlpha &&
      this.pipelinePremultipliedAlphaState === this.premultipliedAlpha
    )
      return;

    const { device } = this;
    const premultiplied = this.premultipliedAlpha;

    const vertexShaderModule = device.createShaderModule({
      label: "spark-vertex",
      code: `${splatDefinesWgsl}\n${splatVertexWgsl}`,
    });
    const fragmentShaderModule = device.createShaderModule({
      label: "spark-fragment",
      code: `${splatDefinesWgsl}\n${splatFragmentWgsl}`,
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        // @binding(0) var<uniform> uniforms: SparkUniforms
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
        // @binding(1) var<uniform> projectionMatrix: mat4x4f
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
        // @binding(2) var ordering: texture_2d<u32>
        {
          binding: 2,
          visibility: GPUShaderStage.VERTEX,
          texture: { sampleType: "uint" },
        },
        // @binding(3) var extSplats: texture_2d_array<u32>
        {
          binding: 3,
          visibility: GPUShaderStage.VERTEX,
          texture: { sampleType: "uint", viewDimension: "2d-array" },
        },
        // @binding(4) var extSplats2: texture_2d_array<u32>
        {
          binding: 4,
          visibility: GPUShaderStage.VERTEX,
          texture: { sampleType: "uint", viewDimension: "2d-array" },
        },
        // @binding(5) var<uniform> fragUniforms: FragmentUniforms
        {
          binding: 5,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = device.createRenderPipeline({
      label: "spark-splat-pipeline",
      layout: pipelineLayout,
      vertex: {
        module: vertexShaderModule,
        entryPoint: "vs_main",
        constants: {
          PREMULTIPLIED_ALPHA: premultiplied ? 1 : 0,
          HALVED_ALPHA: this.halvedAlpha ? 1 : 0,
        },
        buffers: [
          {
            arrayStride: 8,
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: "float32x2" as GPUVertexFormat,
              },
            ],
          },
        ],
      },
      fragment: {
        module: fragmentShaderModule,
        entryPoint: "fs_main",
        constants: { PREMULTIPLIED_ALPHA: premultiplied ? 1 : 0 },
        targets: [
          {
            format: colorFormat,
            blend: this.splatBlendState(premultiplied),
          },
        ],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: false,
        depthCompare: "less-equal",
      },
    });

    this.pipelineFormat = colorFormat;
    this.pipelineHalvedAlphaState = this.halvedAlpha;
    this.pipelinePremultipliedAlphaState = premultiplied;
    this.bindGroupLayoutVersion++;

    // Rebuild bind group if textures already exist
    this.markRenderSlotsDirty();
    this.rebuildBindGroup();
  }

  /**
   * Write SparkUniforms + projection matrix + fragment uniforms to GPU.
   */
  updateUniforms(
    uniforms: {
      renderSize: { value: { x: number; y: number } };
      renderToViewQuat: {
        value: { x: number; y: number; z: number; w: number };
      };
      renderToViewPos: { value: { x: number; y: number; z: number } };
      maxStdDev: { value: number };
      renderToViewBasis: { value: { elements: number[] } };
      minPixelRadius: { value: number };
      maxPixelRadius: { value: number };
      enableExtSplats: { value: boolean };
      enableCovSplats: { value: boolean };
      time: { value: number };
      deltaTime: { value: number };
      debugFlag: { value: boolean };
      minAlpha: { value: number };
      enable2DGS: { value: boolean };
      lodInflate: { value: boolean };
      blurAmount: { value: number };
      preBlurAmount: { value: number };
      focalDistance: { value: number };
      apertureAngle: { value: number };
      clipXY: { value: number };
      focalAdjustment: { value: number };
      encodeLinear: { value: boolean };
      near: { value: number };
      far: { value: number };
      falloff: { value: number };
    },
    projectionMatrix: { elements: number[] },
    isOrthographic: boolean,
  ): void {
    const { device } = this;

    // --- SparkUniforms (160 bytes) ---
    const buf = this.sparkUniformData;
    const f32 = new Float32Array(buf);
    const u32 = new Uint32Array(buf);

    const rs = uniforms.renderSize.value;
    f32[0] = rs.x;
    f32[1] = rs.y;
    // padding at offset 8 (2 floats)

    const q = uniforms.renderToViewQuat.value;
    f32[4] = q.x;
    f32[5] = q.y;
    f32[6] = q.z;
    f32[7] = q.w;

    const p = uniforms.renderToViewPos.value;
    f32[8] = p.x;
    f32[9] = p.y;
    f32[10] = p.z;

    f32[11] = uniforms.maxStdDev.value;

    // renderToViewBasis: mat3x3f (3 columns, each padded to vec4f)
    const e = uniforms.renderToViewBasis.value.elements;
    // Column 0 at float index 12 (offset 48)
    f32[12] = e[0];
    f32[13] = e[1];
    f32[14] = e[2]; // f32[15] = padding
    // Column 1 at float index 16 (offset 64)
    f32[16] = e[3];
    f32[17] = e[4];
    f32[18] = e[5]; // f32[19] = padding
    // Column 2 at float index 20 (offset 80)
    f32[20] = e[6];
    f32[21] = e[7];
    f32[22] = e[8]; // f32[23] = padding

    f32[24] = uniforms.minPixelRadius.value;
    f32[25] = uniforms.maxPixelRadius.value;
    u32[26] = uniforms.enableExtSplats.value ? 1 : 0;
    u32[27] = uniforms.enableCovSplats.value ? 1 : 0;
    f32[28] = uniforms.time.value;
    f32[29] = uniforms.deltaTime.value;
    u32[30] = uniforms.debugFlag.value ? 1 : 0;
    f32[31] = uniforms.minAlpha.value;
    u32[32] = uniforms.enable2DGS.value ? 1 : 0;
    f32[33] = uniforms.blurAmount.value;
    f32[34] = uniforms.preBlurAmount.value;
    f32[35] = uniforms.focalDistance.value;
    f32[36] = uniforms.apertureAngle.value;
    f32[37] = uniforms.clipXY.value;
    f32[38] = uniforms.focalAdjustment.value;
    u32[39] = isOrthographic ? 1 : 0;
    u32[43] = uniforms.lodInflate.value ? 1 : 0;
    // splatEncoding: vec4f at offset 192 (float index 48)
    f32[48] = this.splatEncoding[0]; // rgbMin
    f32[49] = this.splatEncoding[1]; // rgbMax
    f32[50] = this.splatEncoding[2]; // lnScaleMin
    f32[51] = this.splatEncoding[3]; // lnScaleMax

    device.queue.writeBuffer(this.sparkUniformBuffer, 0, buf);

    // --- Projection matrix (64 bytes) ---
    // The camera's projectionMatrix is already in WebGPU coordinate system
    // (Z mapped to [0,1]) because SparkRenderer forces camera.coordinateSystem
    // = WebGPUCoordinateSystem before reading it.
    const projData = new Float32Array(projectionMatrix.elements);
    device.queue.writeBuffer(this.projMatrixBuffer, 0, projData);

    // --- FragmentUniforms (48 bytes) ---
    const fbuf = this.fragUniformData;
    const ff32 = new Float32Array(fbuf);
    const fu32 = new Uint32Array(fbuf);

    ff32[0] = uniforms.near.value;
    ff32[1] = uniforms.far.value;
    fu32[2] = uniforms.encodeLinear.value ? 1 : 0;
    ff32[3] = uniforms.time.value;
    fu32[4] = uniforms.debugFlag.value ? 1 : 0;
    ff32[5] = uniforms.maxStdDev.value;
    ff32[6] = uniforms.minAlpha.value;
    fu32[7] = 0; // disableFalloff
    ff32[8] = uniforms.falloff.value;

    device.queue.writeBuffer(this.fragUniformBuffer, 0, fbuf);
  }

  /**
   * Upload the ordering texture from sort worker results.
   */
  uploadOrdering(
    data: Uint32Array,
    width: number,
    rows: number,
    target: WebGPUOrderingTarget = "write",
  ): void {
    const { device } = this;
    const slot = this.orderingRenderSlot(target);

    let orderingTexture = slot.orderingTexture;
    if (!orderingTexture || slot.orderingRows < rows) {
      orderingTexture?.destroy();

      orderingTexture = device.createTexture({
        label: "spark-ordering",
        size: [width, rows],
        format: "rgba32uint",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      slot.orderingTexture = orderingTexture;
      slot.orderingRows = rows;
      slot.bindGroupDirty = true; // force rebuild
    }

    device.queue.writeTexture(
      { texture: orderingTexture },
      data.buffer,
      { offset: data.byteOffset, bytesPerRow: width * 16 },
      [width, rows],
    );

    slot.orderingGeneration = slot.generation;
    this.syncLegacyFields(slot);
    this.debugLog("ordering upload", { rows, width, target });
  }

  /**
   * Ensure we have a depth texture of the right size.
   */
  private ensureDepthTexture(width: number, height: number): GPUTexture {
    if (
      this.depthTexture &&
      this.depthWidth === width &&
      this.depthHeight === height
    ) {
      return this.depthTexture;
    }
    if (this.depthTexture) this.depthTexture.destroy();

    this.depthTexture = this.device.createTexture({
      label: "spark-depth",
      size: [width, height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    });
    this.depthTextureView = this.depthTexture.createView();
    this.depthWidth = width;
    this.depthHeight = height;
    return this.depthTexture;
  }

  /**
   * Device-loss-safe buffer mapping. `mapAsync()` rejects when the device is
   * lost or the buffer is destroyed mid-map; unguarded `await`s therefore turn
   * a lost device into unhandled rejections and leaked staging buffers.
   *
   * Returns true if the buffer is mapped and safe to read.
   */
  async mapBufferAsync(
    buffer: GPUBuffer,
    mode: GPUMapModeFlags = GPUMapMode.READ,
  ): Promise<boolean> {
    if (this.deviceLost) return false;
    try {
      await buffer.mapAsync(mode);
      return true;
    } catch (error) {
      // AbortError is the expected outcome of losing the device mid-map.
      const name = (error as { name?: string })?.name;
      if (name !== "AbortError" && !this.deviceLost) {
        console.error("Spark: buffer mapAsync failed", error);
      }
      return false;
    }
  }

  /**
   * Split a workgroup count across X and Y so it never exceeds
   * `maxComputeWorkgroupsPerDimension` (65535 by default).
   *
   * A single-dimension dispatch ceilings at 65535 * 256 = 16,776,960 splats,
   * which is reachable: LoD source sets of ~20M splats already exist. Shaders
   * reconstruct the linear index as `(wid.y * nwg.x + wid.x) * 256 + lid.x`, so
   * this must stay in sync with splatDepth.wgsl / splatGather.wgsl.
   *
   * Uses a near-square split so the padding never exceeds one row of groups.
   */
  computeDispatchSize(workgroups: number): [number, number] {
    return calcDispatchSize(
      workgroups,
      this.device.limits?.maxComputeWorkgroupsPerDimension ?? 65535,
    );
  }

  // ========================================================================
  // GPU timing (timestamp-query)
  // ========================================================================
  //
  // Opt-in via `gpuTimingEnabled`, and only effective when the app requested
  // the `timestamp-query` feature at device creation. These samples are
  // separate from CPU wall-clock timings and render submission promises.
  //
  // Frame GPU time is reported as the SPAN from the earliest begin to the
  // latest end across timed passes, never the sum: on tile-based GPUs (Apple
  // Silicon in particular) passes overlap, so summing double-counts and grows
  // with the number of passes even when the GPU is idle.
  //
  // Browser privacy protections can quantize timestamps; a zero duration
  // does not establish that a pass has no GPU cost.

  /** Enable GPU timestamp timing. No effect without the device feature. */
  gpuTimingEnabled = false;
  private timestampQuerySet: GPUQuerySet | null = null;
  private timestampResolveBuffer: GPUBuffer | null = null;
  private timestampReadbackBuffer: GPUBuffer | null = null;
  private timestampSlotLabels: string[] = [];
  private timestampSlotsUsed = 0;
  private timestampReadInFlight = false;
  /** Last completed per-pass GPU durations in ms, keyed by pass label. */
  lastGpuPassMs: Record<string, number> = {};
  /** Span of timed Spark passes; excludes untimed host work and final conversion. */
  lastGpuFrameMs: number | null = null;
  gpuTimingSequence = 0;
  gpuTimingGeneration: number | null = null;

  /** True when timestamp timing can actually be recorded on this device. */
  get gpuTimingAvailable(): boolean {
    return (
      this.gpuTimingEnabled &&
      this.device.features?.has("timestamp-query") === true
    );
  }

  private ensureTimestampResources(): boolean {
    if (this.disposed || this.deviceLost) return false;
    if (this.timestampReadInFlight) return false;
    if (!this.gpuTimingAvailable) return false;
    if (this.timestampQuerySet) return true;
    const count = WebGPUSplatBackend.MAX_TIMESTAMP_PASSES * 2;
    try {
      this.timestampQuerySet = this.device.createQuerySet({
        label: "spark-timestamps",
        type: "timestamp",
        count,
      });
      this.timestampResolveBuffer = this.device.createBuffer({
        label: "spark-timestamp-resolve",
        size: count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.timestampReadbackBuffer = this.device.createBuffer({
        label: "spark-timestamp-readback",
        size: count * 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      return true;
    } catch (error) {
      console.warn("Spark: GPU timing unavailable", error);
      this.gpuTimingEnabled = false;
      return false;
    }
  }

  /**
   * Reserve a timestamp pair for a pass. Returns the descriptor fragment to
   * spread into a pass descriptor, or undefined when timing is off/exhausted.
   */
  private timestampWritesFor(
    label: string,
  ): { timestampWrites: GPUComputePassTimestampWrites } | undefined {
    if (!this.ensureTimestampResources()) return undefined;
    const querySet = this.timestampQuerySet;
    if (!querySet) return undefined;
    if (this.timestampSlotsUsed >= WebGPUSplatBackend.MAX_TIMESTAMP_PASSES) {
      return undefined;
    }
    const slot = this.timestampSlotsUsed++;
    this.timestampSlotLabels[slot] = label;
    return {
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: slot * 2,
        endOfPassWriteIndex: slot * 2 + 1,
      },
    };
  }

  /**
   * Resolve the frame's timestamps. Call once per frame after the last timed
   * pass has been submitted. Reads back asynchronously; frames that arrive
   * while a read is in flight are skipped rather than queued.
   */
  resolveGpuTiming(): void {
    if (this.disposed || this.deviceLost) return;
    if (!this.gpuTimingAvailable || this.timestampSlotsUsed === 0) return;
    const slots = this.timestampSlotsUsed;
    this.timestampSlotsUsed = 0;
    if (this.timestampReadInFlight) return;
    const querySet = this.timestampQuerySet;
    const resolveBuffer = this.timestampResolveBuffer;
    const readbackBuffer = this.timestampReadbackBuffer;
    if (!querySet || !resolveBuffer || !readbackBuffer) return;
    this.timestampReadInFlight = true;

    const encoder = this.device.createCommandEncoder({
      label: "spark-timestamp-resolve",
    });
    encoder.resolveQuerySet(querySet, 0, slots * 2, resolveBuffer, 0);
    encoder.copyBufferToBuffer(
      resolveBuffer,
      0,
      readbackBuffer,
      0,
      slots * 2 * 8,
    );
    this.device.queue.submit([encoder.finish()]);

    const labels = this.timestampSlotLabels.slice(0, slots);
    const generation = this.activeRenderSlot().generation;
    void this.mapBufferAsync(readbackBuffer).then((mapped) => {
      try {
        if (!mapped) return;
        // dispose() destroys the readback buffer; touching it after that is a
        // use-after-destroy, and mapAsync may already have resolved by then.
        if (this.disposed || this.deviceLost) return;
        const raw = new BigUint64Array(
          readbackBuffer.getMappedRange().slice(0, slots * 2 * 8),
        );
        readbackBuffer.unmap();
        const passes: Record<string, number> = {};
        let earliest: bigint | null = null;
        let latest: bigint | null = null;
        for (let i = 0; i < slots; i++) {
          const begin = raw[i * 2];
          const end = raw[i * 2 + 1];
          if (begin === 0n || end === 0n || end < begin) continue;
          // Accumulate, since a label may be timed more than once per frame.
          passes[labels[i]] =
            (passes[labels[i]] ?? 0) + Number(end - begin) / 1e6;
          if (earliest === null || begin < earliest) earliest = begin;
          if (latest === null || end > latest) latest = end;
        }
        this.lastGpuPassMs = passes;
        this.lastGpuFrameMs =
          earliest !== null && latest !== null
            ? Number(latest - earliest) / 1e6
            : null;
        if (Object.keys(passes).length) {
          this.gpuTimingGeneration = generation;
          this.gpuTimingSequence++;
        }
      } finally {
        this.timestampReadInFlight = false;
      }
    });
  }

  /** True when three.js scene depth is currently bound for occlusion. */
  hasExternalDepth(): boolean {
    return !!this.externalDepthView;
  }

  /**
   * Device limits and features relevant to splat rendering, for debug captures.
   *
   * Actual limits and feature negotiation are part of capture provenance;
   * source and generated storage are checked against these limits before use.
   */
  describeLimits(): Record<string, unknown> {
    const limits = this.device.limits;
    const features: string[] = [];
    try {
      for (const feature of this.device.features) {
        features.push(feature as string);
      }
    } catch {
      // features is not iterable on some polyfills; absence is informative too.
    }
    return {
      maxStorageBufferBindingSize: limits?.maxStorageBufferBindingSize,
      maxBufferSize: limits?.maxBufferSize,
      maxComputeWorkgroupsPerDimension:
        limits?.maxComputeWorkgroupsPerDimension,
      maxComputeInvocationsPerWorkgroup:
        limits?.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupStorageSize: limits?.maxComputeWorkgroupStorageSize,
      maxTextureDimension2D: limits?.maxTextureDimension2D,
      maxSampledTexturesPerShaderStage:
        limits?.maxSampledTexturesPerShaderStage,
      hasTimestampQuery: features.includes("timestamp-query"),
      features,
    };
  }

  /** Record a skipped splat draw so captures can distinguish it from flicker. */
  private noteDrawSkipped(reason: string): void {
    this.drawSkipped[reason] = (this.drawSkipped[reason] ?? 0) + 1;
    this.lastDrawSkippedReason = reason;
  }

  /**
   * Adopt an externally-owned depth buffer (three.js's scene depth) as this
   * pass's depth attachment, so splats are occluded by scene geometry.
   *
   * This replaces copying: `depth24plus` is not a copyable format in WebGPU, so
   * `copyTextureToTexture` on it is a validation error. Binding three's depth
   * texture directly is both correct and cheaper. Our pipeline declares
   * `depthWriteEnabled: false`, so three's depth contents are never modified.
   *
   * Pass `null` to fall back to the backend's own cleared depth texture.
   * Returns true if the texture was accepted.
   */
  setExternalDepthTexture(
    texture: GPUTexture | null,
    targetWidth?: number,
    targetHeight?: number,
  ): boolean {
    if (!texture) {
      this.externalDepthTexture = null;
      this.externalDepthView = null;
      this.externalDepthRejectReason = null;
      return false;
    }
    // Every one of these mismatches is a WebGPU validation error that would
    // drop the entire splat pass, so reject rather than bind.
    let reason: string | null = null;
    if (texture.format !== "depth24plus") {
      // Must equal the pipeline's declared depthStencil format.
      reason = `format ${texture.format} != depth24plus`;
    } else if (texture.sampleCount !== 1) {
      // Our color attachment is the 1-sample swapchain texture and the
      // pipeline declares no multisample state; three uses sampleCount 4 when
      // the renderer was constructed with antialias: true.
      reason = `sampleCount ${texture.sampleCount} != 1 (renderer antialias?)`;
    } else if (
      targetWidth !== undefined &&
      targetHeight !== undefined &&
      (texture.width !== targetWidth || texture.height !== targetHeight)
    ) {
      reason = `size ${texture.width}x${texture.height} != target ${targetWidth}x${targetHeight}`;
    }
    if (reason) {
      this.externalDepthTexture = null;
      this.externalDepthView = null;
      this.externalDepthRejectReason = reason;
      return false;
    }
    // Cache the view; createView() every frame would allocate needlessly.
    if (this.externalDepthTexture !== texture) {
      this.externalDepthTexture = texture;
      this.externalDepthView = texture.createView();
    }
    this.externalDepthRejectReason = null;
    return true;
  }

  /**
   * Rebuild the bind group when textures change.
   */
  private rebuildBindGroup(
    slot: WebGPURenderSlot = this.activeRenderSlot(),
  ): void {
    const splatView = slot.splatTextureView;
    const splatView2 = slot.splatTextureView2 ?? splatView;
    const layout = this.bindGroupLayout;
    const orderingTexture = slot.orderingTexture;
    if (!layout || !splatView || !splatView2 || !orderingTexture) {
      return; // Can't rebuild yet — leave old bind group in place
    }

    slot.bindGroupDirty = false;
    slot.bindGroupVersion = this.bindGroupLayoutVersion;

    slot.bindGroup = this.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.sparkUniformBuffer } },
        { binding: 1, resource: { buffer: this.projMatrixBuffer } },
        { binding: 2, resource: orderingTexture.createView() },
        { binding: 3, resource: splatView },
        { binding: 4, resource: splatView2 },
        { binding: 5, resource: { buffer: this.fragUniformBuffer } },
      ],
    });
    this.syncLegacyFields(slot);
  }

  /** Generate ordinary meshes using the same output layout/encoding as WebGL. */
  generateNative(args: {
    meshes: NativeMeshGeneration[];
    numSplats: number;
    outputExt: boolean;
    viewOrigin: number[];
    viewDirection: number[];
    sortRadial: boolean;
  }): void {
    const { device } = this;
    const slot = this.writeRenderSlot();
    if (!args.numSplats) {
      this.nativeGeneration?.clearSources();
      slot.generatedSpan = 0;
      return;
    }
    const size = getTextureSize(args.numSplats);
    const bytes = size.width * size.height * size.depth * 16;
    this.assertStorageBufferFits(bytes, "native generated splats");
    if (
      size.width > device.limits.maxTextureDimension2D ||
      size.height > device.limits.maxTextureDimension2D ||
      size.depth > device.limits.maxTextureArrayLayers
    )
      throw new Error(
        "Spark: native output exceeds device texture limits; use THREE.WebGLRenderer.",
      );
    this.ensureSplatTexture(args.numSplats);
    if (
      !this.generatedOutputBuffer ||
      this.generatedOutputBuffer.size < bytes
    ) {
      this.generatedOutputBuffer?.destroy();
      this.generatedOutputBuffer = device.createBuffer({
        label: "spark-native-output",
        size: bytes,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
    }
    if (args.outputExt) this.ensureSplatTexture2(args.numSplats);
    const secondaryBytes = args.outputExt ? bytes : 16;
    if (
      !this.generatedOutputBuffer2 ||
      this.generatedOutputBuffer2.size < secondaryBytes
    ) {
      this.generatedOutputBuffer2?.destroy();
      this.generatedOutputBuffer2 = device.createBuffer({
        label: "spark-native-output2",
        size: secondaryBytes,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
    }
    const depthBytes = Math.ceil(args.numSplats / 256) * 256 * 4;
    this.assertStorageBufferFits(depthBytes, "native generated depths");
    if (!slot.nativeDepths || slot.nativeDepths.size < depthBytes) {
      slot.nativeDepths?.destroy();
      slot.nativeDepths = device.createBuffer({
        label: "spark-native-depths",
        size: depthBytes,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
    }
    if (!this.nativeFillPipeline) {
      this.nativeFillPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
          module: device.createShaderModule({
            code: `
          @group(0) @binding(0) var<storage, read_write> depths: array<u32>;
          @compute @workgroup_size(256) fn fill(@builtin(workgroup_id) g:vec3u,
          @builtin(local_invocation_id) l:vec3u, @builtin(num_workgroups) n:vec3u) {
            let i = (g.y*n.x+g.x)*256u+l.x;
            if(i<arrayLength(&depths)){ depths[i]=0x7f800000u; }
          }`,
          }),
          entryPoint: "fill",
        },
      });
    }
    const encoder = device.createCommandEncoder({
      label: "spark-native-generation",
    });
    encoder.clearBuffer(this.generatedOutputBuffer);
    encoder.clearBuffer(this.generatedOutputBuffer2);
    const fill = encoder.beginComputePass();
    fill.setPipeline(this.nativeFillPipeline);
    fill.setBindGroup(
      0,
      device.createBindGroup({
        layout: this.nativeFillPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: slot.nativeDepths } }],
      }),
    );
    fill.dispatchWorkgroups(
      ...this.computeDispatchSize(slot.nativeDepths.size / 1024),
    );
    fill.end();
    try {
      this.nativeGeneration ??= new WebGPUGeneration(device);
      this.nativeGeneration.encode(encoder, {
        ...args,
        output: this.generatedOutputBuffer,
        output2: this.generatedOutputBuffer2,
        depths: slot.nativeDepths,
      });
    } catch (error) {
      encoder.finish();
      throw error;
    }
    const texture = slot.splatTexture;
    const texture2 = slot.splatTexture2;
    if (!texture || (args.outputExt && !texture2)) {
      encoder.finish();
      throw new Error("Spark: native generation textures are unavailable");
    }
    for (let layer = 0; layer < size.depth; layer++) {
      const source = {
        buffer: this.generatedOutputBuffer,
        offset: layer * size.width * size.height * 16,
        bytesPerRow: size.width * 16,
        rowsPerImage: size.height,
      };
      encoder.copyBufferToTexture(source, { texture, origin: [0, 0, layer] }, [
        size.width,
        size.height,
        1,
      ]);
      if (args.outputExt && texture2)
        encoder.copyBufferToTexture(
          { ...source, buffer: this.generatedOutputBuffer2 },
          { texture: texture2, origin: [0, 0, layer] },
          [size.width, size.height, 1],
        );
    }
    device.queue.submit([encoder.finish()]);
    slot.generatedSpan = args.numSplats;
    this.halvedAlpha = !args.outputExt;
    this.splatEncoding = [0, 1, -12, 9];
    this.syncLegacyFields(slot);
  }

  async readNativeDepths(count: number): Promise<Uint32Array<ArrayBuffer>> {
    if (this.disposed || this.deviceLost)
      throw new Error(
        "Spark: native backend is disposed or its device is lost",
      );
    if (!count) return new Uint32Array();
    const source = this.depthRenderSlot().nativeDepths;
    if (!source) throw new Error("Spark: no generated native depths");
    const size = count * 4;
    let buffer = this.syncDepthReadbackBuffer;
    if (!buffer || buffer.size < size) {
      buffer?.destroy();
      buffer = this.device.createBuffer({
        label: "spark-native-depth-readback",
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      this.syncDepthReadbackBuffer = buffer;
    }
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, buffer, 0, size);
    this.device.queue.submit([encoder.finish()]);
    if (!(await this.mapBufferAsync(buffer)))
      throw new Error("Spark: native depth readback failed");
    try {
      if (this.disposed || this.deviceLost)
        throw new Error(
          "Spark: native backend is disposed or its device is lost",
        );
      return new Uint32Array(buffer.getMappedRange(0, size).slice(0));
    } finally {
      buffer.unmap();
    }
  }

  /**
   * Execute a separate render pass that composites splats onto the
   * provided color texture, preserving existing content (loadOp: "load").
   */
  renderSeparatePass(
    colorTexture: GPUTexture,
    numInstances: number,
    preloadedDepth = false,
  ): void {
    const slot = this.activeRenderSlot();
    if (numInstances <= 0) {
      this.noteDrawSkipped("zero-instances");
      return;
    }
    let drawPipeline: GPURenderPipeline;
    let drawBindGroup: GPUBindGroup;
    {
      // Lazy bind group rebuild — ensures we never skip rendering due to stale null
      // or pipeline/bind-group layout version mismatch after ensurePipeline().
      if (
        slot.bindGroupDirty ||
        slot.bindGroupVersion !== this.bindGroupLayoutVersion
      ) {
        this.rebuildBindGroup(slot);
      }
      const pipeline = this.pipeline;
      if (!pipeline) {
        throw new Error("Spark: native draw pipeline is unavailable");
      }
      const bindGroup = slot.bindGroup;
      if (!bindGroup) {
        throw new Error("Spark: native draw resources are incomplete");
      }
      if (slot.bindGroupVersion !== this.bindGroupLayoutVersion) {
        throw new Error("Spark: native draw resource layout is stale");
      }
      drawPipeline = pipeline;
      drawBindGroup = bindGroup;
    }

    const { device } = this;
    const width = colorTexture.width;
    const height = colorTexture.height;

    // Prefer three.js's scene depth so splats are occluded by scene geometry;
    // fall back to our own cleared depth texture when it is unavailable.
    const useExternalDepth = !!this.externalDepthView;
    let depthView = this.externalDepthView;
    if (!useExternalDepth) {
      this.ensureDepthTexture(width, height);
      depthView = this.depthTextureView;
    }
    if (!depthView) {
      this.noteDrawSkipped("no-depth-view");
      return;
    }
    const loadExistingDepth = useExternalDepth || preloadedDepth;

    const encoder = device.createCommandEncoder({ label: "spark-render" });
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: colorTexture.createView(),
          loadOp: "load" as GPULoadOp,
          storeOp: "store" as GPUStoreOp,
        },
      ],
      depthStencilAttachment: {
        view: depthView,
        ...(loadExistingDepth
          ? { depthLoadOp: "load" as GPULoadOp }
          : { depthClearValue: 1.0, depthLoadOp: "clear" as GPULoadOp }),
        depthStoreOp: "store" as GPUStoreOp,
      },
      ...(this.timestampWritesFor("render") as
        | { timestampWrites: GPURenderPassTimestampWrites }
        | undefined),
    });

    pass.setPipeline(drawPipeline);
    pass.setBindGroup(0, drawBindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, "uint16");
    pass.drawIndexed(6, numInstances);

    pass.end();
    device.queue.submit([encoder.finish()]);

    // The splat render pass is the last GPU work Spark submits in a frame, so
    // resolve the frame's timestamps here. No-op unless timing is enabled.
    this.resolveGpuTiming();
  }

  /**
   * Fail early, and actionably, when a buffer would exceed device limits.
   *
   * Each generated output and each source binding must fit the host device.
   * Reducing the selected LoD budget does not reduce source storage.
   */
  private assertStorageBufferFits(byteSize: number, label: string): void {
    const limits = this.device.limits;
    const maxBinding = limits?.maxStorageBufferBindingSize ?? 134217728;
    const maxBuffer = limits?.maxBufferSize ?? 268435456;
    const cap = Math.min(maxBinding, maxBuffer);
    if (byteSize <= cap) return;
    const mib = (bytes: number) => `${(bytes / 1048576).toFixed(1)} MiB`;
    throw new Error(
      `Spark: ${label} needs ${mib(byteSize)} in a single storage buffer, but this device allows ${mib(cap)} (maxStorageBufferBindingSize=${mib(maxBinding)}, maxBufferSize=${mib(maxBuffer)}). Request higher limits when creating the WebGPU device, or reduce the source buffer size. Lowering only the selected LoD budget does not reduce source storage.`,
    );
  }

  /**
   * Ensure the splat texture exists at the right size.
   * Allocates generation output before copyBufferToTexture.
   */
  ensureSplatTexture(numSplats: number): {
    width: number;
    height: number;
    depth: number;
  } {
    const slot = this.writeRenderSlot();
    const texSize = getTextureSize(numSplats);
    const { width, height, depth } = texSize;

    if (
      !slot.splatTexture ||
      !slot.splatTexSize ||
      slot.splatTexSize.width !== width ||
      slot.splatTexSize.height !== height ||
      slot.splatTexSize.depth !== depth
    ) {
      if (slot.splatTexture) slot.splatTexture.destroy();

      slot.splatTexture = this.device.createTexture({
        label: "spark-splatData",
        size: [width, height, depth],
        format: "rgba32uint",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        dimension: "2d",
      });
      slot.splatTextureView = slot.splatTexture.createView({
        dimension: "2d-array",
      });
      slot.splatTexSize = { width, height, depth };
      slot.bindGroupDirty = true;
    }

    this.syncLegacyFields(slot);
    return { width, height, depth };
  }

  /**
   * Ensure the ext splat texture exists at the right size.
   */
  ensureSplatTexture2(numSplats: number): {
    width: number;
    height: number;
    depth: number;
  } {
    const slot = this.writeRenderSlot();
    const texSize = getTextureSize(numSplats);
    const { width, height, depth } = texSize;

    if (
      !slot.splatTexture2 ||
      !slot.splatTex2Size ||
      slot.splatTex2Size.width !== width ||
      slot.splatTex2Size.height !== height ||
      slot.splatTex2Size.depth !== depth
    ) {
      if (slot.splatTexture2) slot.splatTexture2.destroy();

      slot.splatTexture2 = this.device.createTexture({
        label: "spark-splatData2",
        size: [width, height, depth],
        format: "rgba32uint",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        dimension: "2d",
      });
      slot.splatTextureView2 = slot.splatTexture2.createView({
        dimension: "2d-array",
      });
      slot.splatTex2Size = { width, height, depth };
      slot.bindGroupDirty = true;
    }

    this.syncLegacyFields(slot);
    return { width, height, depth };
  }

  dispose(): void {
    // Set first: in-flight mapAsync chains (timestamp readback, depth readback)
    // must not touch buffers this method is about to destroy.
    if (this.disposed) return;
    this.disposed = true;
    this.releaseDeviceLost();
    this.device.removeEventListener("uncapturederror", this.onUncapturedError);
    this.externalDepthTexture = null;
    this.externalDepthView = null;
    for (const slot of this.renderSlots) {
      slot.nativeDepths?.destroy();
      slot.nativeDepths = null;
      slot.splatTexture?.destroy();
      slot.splatTexture = null;
      slot.splatTextureView = null;
      slot.splatTexture2?.destroy();
      slot.splatTexture2 = null;
      slot.splatTextureView2 = null;
      slot.orderingTexture?.destroy();
      slot.orderingTexture = null;
      slot.bindGroup = null;
    }
    this.splatTexture = null;
    this.splatTextureView = null;
    this.splatTexture2 = null;
    this.splatTextureView2 = null;
    this.orderingTexture = null;
    this.depthTexture?.destroy();
    this.depthTexture = null;
    this.depthTextureView = null;
    this.syncDepthReadbackBuffer?.destroy();
    this.syncDepthReadbackBuffer = null;
    this.timestampQuerySet?.destroy();
    this.timestampQuerySet = null;
    this.timestampResolveBuffer?.destroy();
    this.timestampResolveBuffer = null;
    this.timestampReadbackBuffer?.destroy();
    this.timestampReadbackBuffer = null;
    this.timestampSlotLabels = [];
    this.timestampSlotsUsed = 0;
    this.timestampReadInFlight = false;

    this.nativeGeneration?.dispose();
    this.nativeGeneration = null;
    this.generatedOutputBuffer?.destroy();
    this.generatedOutputBuffer = null;
    this.generatedOutputBuffer2?.destroy();
    this.generatedOutputBuffer2 = null;

    this.sparkUniformBuffer.destroy();
    this.projMatrixBuffer.destroy();
    this.fragUniformBuffer.destroy();
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();

    this.pipeline = null;
    this.bindGroupLayout = null;
    this.bindGroup = null;
  }
}

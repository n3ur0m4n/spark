import { calcDispatchSize } from "./WebGPUUtils";
import { splatDefinesWgsl } from "./shaders-wgsl";
import generateSplats from "./shaders/generateSplats.wgsl?raw";

/** Immutable CPU description of one ordinary mesh in an accumulator generation. */
export type NativeMeshGeneration = {
  source: object;
  revision: number;
  primary: Uint32Array;
  secondary?: Uint32Array;
  sh: (Uint32Array | undefined)[];
  sourceCount: number;
  sourceExt: boolean;
  base: number;
  count: number;
  indices?: Uint32Array;
  rotation: number[];
  translationScale: number[];
  localViewOrigin: number[];
  encoding: number[];
  shMaxOpacity: number[];
  numSh: number;
  recolor: number[];
};

type SourceBuffers = {
  revision: number;
  arrays: (Uint32Array | undefined)[];
  sourceCount: number;
  primary: GPUBuffer;
  secondary: GPUBuffer;
  sh: GPUBuffer;
  shOffsets: number[];
};

/** Source uploads are shared across instances; output ownership stays with the backend slot. */
export class WebGPUGeneration {
  private readonly sources = new Map<object, SourceBuffers>();
  private readonly pipeline: GPUComputePipeline;
  private readonly dummy: GPUBuffer;
  private readonly meshes: { uniform: GPUBuffer; indices: GPUBuffer }[] = [];
  private disposed = false;

  constructor(private readonly device: GPUDevice) {
    this.dummy = this.buffer(16, GPUBufferUsage.STORAGE, "native-empty-source");
    this.pipeline = device.createComputePipeline({
      label: "spark-native-generate",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          code: `${splatDefinesWgsl}\n${generateSplats}`,
        }),
        entryPoint: "generate",
      },
    });
  }

  private buffer(size: number, usage: GPUBufferUsageFlags, label: string) {
    const bytes = Math.max(16, Math.ceil(size / 4) * 4);
    if (
      bytes > this.device.limits.maxBufferSize ||
      ((usage & GPUBufferUsage.STORAGE) !== 0 &&
        bytes > this.device.limits.maxStorageBufferBindingSize)
    ) {
      throw new Error(
        `Spark: ${label} needs ${bytes} bytes, exceeding native device limits; use THREE.WebGLRenderer.`,
      );
    }
    return this.device.createBuffer({
      size: bytes,
      usage,
      label: `spark-${label}`,
    });
  }

  private destroySource(buffers: SourceBuffers) {
    buffers.primary.destroy();
    if (buffers.secondary !== this.dummy) buffers.secondary.destroy();
    if (buffers.sh !== this.dummy) buffers.sh.destroy();
  }

  private uploadSource(mesh: NativeMeshGeneration): SourceBuffers {
    const arrays = [mesh.primary, mesh.secondary, ...mesh.sh];
    const old = this.sources.get(mesh.source);
    if (
      old &&
      old.revision === mesh.revision &&
      old.sourceCount === mesh.sourceCount &&
      arrays.every((array, i) => old.arrays[i] === array)
    )
      return old;
    const created: GPUBuffer[] = [];
    const upload = (
      array: Uint32Array | undefined,
      size = array?.byteLength ?? 0,
    ) => {
      if (!array || size === 0) return this.dummy;
      const buffer = this.buffer(
        size,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        "native-source",
      );
      created.push(buffer);
      this.device.queue.writeBuffer(
        buffer,
        0,
        array.buffer,
        array.byteOffset,
        size,
      );
      return buffer;
    };
    try {
      const primary = upload(mesh.primary, mesh.sourceCount * 16);
      const secondary = upload(mesh.secondary, mesh.sourceCount * 16);
      const shOffsets: number[] = [];
      let shWords = 0;
      for (const band of mesh.sh) {
        shOffsets.push(shWords);
        shWords += band?.length ?? 0;
      }
      let sh = this.dummy;
      if (shWords) {
        sh = this.buffer(
          shWords * 4,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          "native-source-sh",
        );
        created.push(sh);
        mesh.sh.forEach((band, i) => {
          if (band?.length)
            this.device.queue.writeBuffer(
              sh,
              shOffsets[i] * 4,
              band.buffer,
              band.byteOffset,
              band.byteLength,
            );
        });
      }
      const result = {
        revision: mesh.revision,
        arrays,
        sourceCount: mesh.sourceCount,
        primary,
        secondary,
        sh,
        shOffsets,
      };
      this.sources.set(mesh.source, result);
      if (old) this.destroySource(old);
      return result;
    } catch (error) {
      for (const buffer of created) buffer.destroy();
      throw error;
    }
  }

  encode(
    encoder: GPUCommandEncoder,
    args: {
      meshes: NativeMeshGeneration[];
      output: GPUBuffer;
      output2: GPUBuffer;
      depths: GPUBuffer;
      outputExt: boolean;
      viewOrigin: number[];
      viewDirection: number[];
      sortRadial: boolean;
    },
  ) {
    if (this.disposed) throw new Error("Spark: native generation is disposed");
    const live = new Set(args.meshes.map((mesh) => mesh.source));
    for (const [source, buffers] of this.sources) {
      if (!live.has(source)) {
        this.destroySource(buffers);
        this.sources.delete(source);
      }
    }
    while (this.meshes.length > args.meshes.length) {
      const state = this.meshes.pop();
      if (state) {
        state.uniform.destroy();
        if (state.indices !== this.dummy) state.indices.destroy();
      }
    }
    for (let i = 0; i < args.meshes.length; i++) {
      const mesh = args.meshes[i];
      if (!mesh.count) continue;
      const source = this.uploadSource(mesh);
      let state = this.meshes[i];
      if (!state) {
        state = {
          uniform: this.buffer(
            176,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            "native-mesh-uniform",
          ),
          indices: this.dummy,
        };
        this.meshes[i] = state;
      }
      if (mesh.indices) {
        const bytes = Math.max(16, mesh.count * 4);
        if (state.indices === this.dummy || state.indices.size < bytes) {
          if (state.indices !== this.dummy) state.indices.destroy();
          state.indices = this.buffer(
            bytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "native-selection",
          );
        }
        this.device.queue.writeBuffer(
          state.indices,
          0,
          mesh.indices.buffer,
          mesh.indices.byteOffset,
          mesh.count * 4,
        );
      }
      const data = new ArrayBuffer(176);
      const u32 = new Uint32Array(data);
      const f32 = new Float32Array(data);
      u32.set([
        mesh.base,
        mesh.count,
        mesh.sourceCount,
        mesh.indices ? 1 : 0,
        mesh.sourceExt ? 1 : 0,
        args.outputExt ? 1 : 0,
        args.sortRadial ? 1 : 0,
        mesh.numSh,
      ]);
      f32.set(mesh.rotation, 8);
      f32.set(mesh.translationScale, 12);
      f32.set(args.viewOrigin, 16);
      f32.set(args.viewDirection, 20);
      f32.set(mesh.localViewOrigin, 24);
      f32.set(mesh.encoding, 28);
      f32.set(mesh.shMaxOpacity, 32);
      u32.set(source.shOffsets, 36);
      f32.set(mesh.recolor, 40);
      this.device.queue.writeBuffer(state.uniform, 0, data);
      const buffers = [
        state.uniform,
        source.primary,
        source.secondary,
        source.sh,
        mesh.indices ? state.indices : this.dummy,
        args.output,
        args.output2,
        args.depths,
      ];
      const group = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({
          binding,
          resource: { buffer },
        })),
      });
      const pass = encoder.beginComputePass({ label: "spark-native-generate" });
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, group);
      const [x, y] = calcDispatchSize(
        Math.ceil(mesh.count / 256),
        this.device.limits.maxComputeWorkgroupsPerDimension,
      );
      pass.dispatchWorkgroups(x, y);
      pass.end();
    }
  }

  /** Release source ownership when the committed scene becomes empty. */
  clearSources() {
    for (const source of this.sources.values()) this.destroySource(source);
    this.sources.clear();
    for (const mesh of this.meshes) {
      mesh.uniform.destroy();
      if (mesh.indices !== this.dummy) mesh.indices.destroy();
    }
    this.meshes.length = 0;
  }

  dispose() {
    this.disposed = true;
    this.clearSources();
    this.dummy.destroy();
  }
}

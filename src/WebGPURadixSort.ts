// Copyright © 2011-2026 PlayCanvas Ltd.
// Copyright © kishimisu (WebGPU-Radix-Sort)
// Copyright © 2025 WORLD LABS TECHNOLOGIES, INC.
// SPDX-License-Identifier: MIT
//
// Spark GPUDevice dispatcher for the portable 4-bit multipass radix sort.
// WGSL kernels are adapted from playcanvas/engine at
// 329a94115f5af7ff8ccf8d4348aded800b0e3fbd. This file is not a port of
// PlayCanvas GraphicsDevice / Compute / Shader classes. See NOTICE.

import prefixSumWgsl from "./shaders/prefixSum.wgsl?raw";
import radixSort4bitWgsl from "./shaders/radixSort4bit.wgsl?raw";
import radixSortReorderWgsl from "./shaders/radixSortReorder.wgsl?raw";

const WORKGROUP_SIZE_X = 16;
const WORKGROUP_SIZE_Y = 16;
const THREADS_PER_WORKGROUP = WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y;
const ELEMENTS_PER_THREAD = 8;
const ELEMENTS_PER_WORKGROUP = THREADS_PER_WORKGROUP * ELEMENTS_PER_THREAD;
const BITS_PER_PASS = 4;
const BUCKET_COUNT = 16;
const PREFIX_ITEMS_PER_WORKGROUP = 2 * THREADS_PER_WORKGROUP;

function dispatch2d(
  workgroupCount: number,
  maxDim: number,
): { x: number; y: number } {
  if (workgroupCount <= maxDim) {
    return { x: Math.max(1, workgroupCount), y: 1 };
  }
  const x = Math.max(1, Math.ceil(Math.sqrt(workgroupCount)));
  const y = Math.ceil(workgroupCount / x);
  return { x, y };
}

function align256(bytes: number): number {
  return Math.ceil(bytes / 256) * 256;
}

type PrefixLevel = {
  items: GPUBuffer;
  blockSums: GPUBuffer;
  count: number;
  dispatch: { x: number; y: number };
};

/**
 * Portable 4-bit GPU radix sort. Direct dispatch only (CPU knows N).
 * Returns a storage buffer of sorted u32 indices.
 */
export class WebGPURadixSort {
  readonly radixBits = BITS_PER_PASS;

  private device: GPUDevice;
  private maxWorkgroups: number;
  private available = false;

  private histogramModule: GPUShaderModule | null = null;
  private reorderModule: GPUShaderModule | null = null;
  private prefixModule: GPUShaderModule | null = null;

  private histogramLayout: GPUBindGroupLayout | null = null;
  private reorderLayout: GPUBindGroupLayout | null = null;
  private prefixLayout: GPUBindGroupLayout | null = null;

  private histogramPipes: GPUComputePipeline[] = [];
  private reorderPipes: GPUComputePipeline[] = [];
  private prefixScanPipe: GPUComputePipeline | null = null;
  private prefixAddPipe: GPUComputePipeline | null = null;

  private sortUniform: GPUBuffer | null = null;
  private prefixUniforms: GPUBuffer[] = [];

  private keys0: GPUBuffer | null = null;
  private keys1: GPUBuffer | null = null;
  private values0: GPUBuffer | null = null;
  private values1: GPUBuffer | null = null;
  private blockSums: GPUBuffer | null = null;
  private prefixLevels: PrefixLevel[] = [];

  private capacity = 0;
  private allocatedWorkgroups = 0;
  private workgroupCount = 0;
  private prefixWorkgroupCount = 0;
  private activePrefixLevels = 0;
  private lastNumBits = 0;
  private lastSkipLastKeyWrite = false;
  private sortedIndices: GPUBuffer | null = null;

  /**
   * Incremented every time the internal ping/pong buffers are destroyed and
   * reallocated. Callers that cache a bind group referencing `encodeSort`'s
   * returned buffer must record the epoch alongside it and rebuild the bind
   * group when the epoch changes — the old GPUBuffer is destroyed at that
   * point and using it is a validation error.
   */
  bufferEpoch = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    this.maxWorkgroups =
      device.limits.maxComputeWorkgroupsPerDimension || 65535;
    this.available = this.compile();
  }

  isAvailable(): boolean {
    return this.available;
  }

  /**
   * Compile histogram/reorder pipelines for `numBits` before the first
   * encodeSort so a compile failure can latch fallback without abandoning
   * an in-flight command encoder.
   */
  warmup(numBits = 32, skipLastKeyWrite = true): boolean {
    if (!this.available) return false;
    return this.ensurePassPipelines(numBits, skipLastKeyWrite);
  }

  destroy(): void {
    this.destroyBuffers();
    this.sortUniform?.destroy();
    for (const buf of this.prefixUniforms) buf.destroy();
    this.sortUniform = null;
    this.prefixUniforms = [];
    this.histogramPipes = [];
    this.reorderPipes = [];
    this.available = false;
  }

  private compile(): boolean {
    try {
      const { device } = this;
      this.histogramModule = device.createShaderModule({
        label: "spark-radix-histogram",
        code: radixSort4bitWgsl,
      });
      this.reorderModule = device.createShaderModule({
        label: "spark-radix-reorder",
        code: radixSortReorderWgsl,
      });
      this.prefixModule = device.createShaderModule({
        label: "spark-prefix-sum",
        code: prefixSumWgsl,
      });

      this.histogramLayout = device.createBindGroupLayout({
        label: "spark-radix-histogram-layout",
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" },
          },
        ],
      });
      this.reorderLayout = device.createBindGroupLayout({
        label: "spark-radix-reorder-layout",
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
          {
            binding: 5,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" },
          },
        ],
      });
      this.prefixLayout = device.createBindGroupLayout({
        label: "spark-prefix-layout",
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" },
          },
        ],
      });

      this.sortUniform = device.createBuffer({
        label: "spark-radix-uniform",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.prefixUniforms = [];

      this.prefixScanPipe = device.createComputePipeline({
        label: "spark-prefix-scan",
        layout: device.createPipelineLayout({
          bindGroupLayouts: [this.prefixLayout],
        }),
        compute: { module: this.prefixModule, entryPoint: "reduce_downsweep" },
      });
      this.prefixAddPipe = device.createComputePipeline({
        label: "spark-prefix-add",
        layout: device.createPipelineLayout({
          bindGroupLayouts: [this.prefixLayout],
        }),
        compute: { module: this.prefixModule, entryPoint: "add_block_sums" },
      });

      return true;
    } catch (error) {
      console.warn(
        "Spark: GPU radix sort failed to compile; falling back.",
        error,
      );
      return false;
    }
  }

  private ensurePassPipelines(
    numBits: number,
    skipLastKeyWrite: boolean,
  ): boolean {
    if (
      this.histogramPipes.length === numBits / BITS_PER_PASS &&
      this.lastNumBits === numBits &&
      this.lastSkipLastKeyWrite === skipLastKeyWrite
    ) {
      return true;
    }
    if (
      !this.histogramModule ||
      !this.reorderModule ||
      !this.histogramLayout ||
      !this.reorderLayout
    ) {
      return false;
    }
    try {
      const numPasses = numBits / BITS_PER_PASS;
      const histogramPipes: GPUComputePipeline[] = [];
      const reorderPipes: GPUComputePipeline[] = [];
      for (let pass = 0; pass < numPasses; pass++) {
        const bit = pass * BITS_PER_PASS;
        histogramPipes.push(
          this.device.createComputePipeline({
            label: `spark-radix-hist-${bit}`,
            layout: this.device.createPipelineLayout({
              bindGroupLayouts: [this.histogramLayout],
            }),
            compute: {
              module: this.histogramModule,
              entryPoint: "main",
              constants: { CURRENT_BIT: bit },
            },
          }),
        );
        reorderPipes.push(
          this.device.createComputePipeline({
            label: `spark-radix-reorder-${bit}`,
            layout: this.device.createPipelineLayout({
              bindGroupLayouts: [this.reorderLayout],
            }),
            compute: {
              module: this.reorderModule,
              entryPoint: "main",
              constants: {
                CURRENT_BIT: bit,
                IS_FIRST_PASS: pass === 0 ? 1 : 0,
                IS_LAST_PASS:
                  skipLastKeyWrite && pass === numPasses - 1 ? 1 : 0,
              },
            },
          }),
        );
      }
      this.histogramPipes = histogramPipes;
      this.reorderPipes = reorderPipes;
      this.lastNumBits = numBits;
      this.lastSkipLastKeyWrite = skipLastKeyWrite;
      return true;
    } catch (error) {
      console.warn(
        "Spark: GPU radix pass pipelines failed; falling back.",
        error,
      );
      this.available = false;
      return false;
    }
  }

  private destroyBuffers(): void {
    this.keys0?.destroy();
    this.keys1?.destroy();
    this.values0?.destroy();
    this.values1?.destroy();
    this.blockSums?.destroy();
    for (const level of this.prefixLevels) {
      if (level.blockSums !== this.blockSums) {
        level.blockSums.destroy();
      }
    }
    this.keys0 = null;
    this.keys1 = null;
    this.values0 = null;
    this.values1 = null;
    this.blockSums = null;
    // The last returned sorted-values buffer aliases one of the ping/pong
    // buffers just destroyed above; drop the reference so it cannot be handed
    // out again.
    this.sortedIndices = null;
    for (const buf of this.prefixUniforms) buf.destroy();
    this.prefixUniforms = [];
    this.prefixLevels = [];
    this.capacity = 0;
    this.allocatedWorkgroups = 0;
    this.prefixWorkgroupCount = 0;
    this.activePrefixLevels = 0;
    // Invalidate any bind group a caller cached against the old buffers.
    this.bufferEpoch++;
  }

  private storageBuffer(label: string, size: number): GPUBuffer {
    return this.device.createBuffer({
      label,
      size: align256(size),
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
  }

  private allocate(elementCount: number): void {
    const allocCount = Math.max(elementCount, this.capacity);
    const allocWorkgroups = Math.max(
      1,
      Math.ceil(allocCount / ELEMENTS_PER_WORKGROUP),
    );
    this.workgroupCount = Math.max(
      1,
      Math.ceil(elementCount / ELEMENTS_PER_WORKGROUP),
    );

    if (
      elementCount <= this.capacity &&
      allocWorkgroups === this.allocatedWorkgroups &&
      this.keys0 &&
      this.blockSums
    ) {
      if (this.prefixWorkgroupCount !== this.workgroupCount) {
        this.configurePrefixLevels();
      }
      return;
    }

    this.destroyBuffers();
    this.capacity = allocCount;
    this.allocatedWorkgroups = allocWorkgroups;

    const elementBytes = allocCount * 4;
    this.keys0 = this.storageBuffer("spark-radix-keys0", elementBytes);
    this.keys1 = this.storageBuffer("spark-radix-keys1", elementBytes);
    this.values0 = this.storageBuffer("spark-radix-values0", elementBytes);
    this.values1 = this.storageBuffer("spark-radix-values1", elementBytes);
    this.blockSums = this.storageBuffer(
      "spark-radix-block-sums",
      BUCKET_COUNT * allocWorkgroups * 4,
    );
    this.rebuildPrefixLevels();
    this.configurePrefixLevels();
  }

  private rebuildPrefixLevels(): void {
    if (!this.blockSums) return;
    for (const level of this.prefixLevels) {
      if (level.blockSums !== this.blockSums) {
        level.blockSums.destroy();
      }
    }
    this.prefixLevels = [];
    for (const buf of this.prefixUniforms) buf.destroy();
    this.prefixUniforms = [];
    let items = this.blockSums;
    let count = BUCKET_COUNT * this.allocatedWorkgroups;
    while (count > 0) {
      const workgroups = Math.max(
        1,
        Math.ceil(count / PREFIX_ITEMS_PER_WORKGROUP),
      );
      const blockSums = this.storageBuffer(
        `spark-prefix-blocks-${this.prefixLevels.length}`,
        workgroups * 4,
      );
      const uniform = this.device.createBuffer({
        label: `spark-prefix-uniform-${this.prefixLevels.length}`,
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(
        uniform,
        0,
        new Uint32Array([count, 0, 0, 0]),
      );
      this.prefixUniforms.push(uniform);
      this.prefixLevels.push({
        items,
        blockSums,
        count,
        dispatch: dispatch2d(workgroups, this.maxWorkgroups),
      });
      if (workgroups <= 1) break;
      items = blockSums;
      count = workgroups;
    }
  }

  // Scratch follows allocation capacity; dispatches follow the current sort.
  // Do not scan inactive retained levels or their stale tail elements.
  private configurePrefixLevels(): void {
    let count = BUCKET_COUNT * this.workgroupCount;
    this.activePrefixLevels = 0;
    while (count > 0) {
      const index = this.activePrefixLevels++;
      const level = this.prefixLevels[index];
      const groups = Math.ceil(count / PREFIX_ITEMS_PER_WORKGROUP);
      level.count = count;
      level.dispatch = dispatch2d(groups, this.maxWorkgroups);
      this.device.queue.writeBuffer(
        this.prefixUniforms[index],
        0,
        new Uint32Array([count, 0, 0, 0]),
      );
      if (groups <= 1) break;
      count = groups;
    }
    this.prefixWorkgroupCount = this.workgroupCount;
  }

  /**
   * Encode a radix sort into `encoder`. Keys are u32. Returns the buffer of
   * sorted indices (valid until the next sort or destroy).
   */
  encodeSort(
    encoder: GPUCommandEncoder,
    keysBuffer: GPUBuffer,
    elementCount: number,
    numBits = 32,
    skipLastPassKeyWrite = true,
    destructiveKeys = true,
    timestamps?: GPUComputePassTimestampWrites,
  ): GPUBuffer | null {
    if (!this.available || elementCount <= 0) return null;
    if (numBits % BITS_PER_PASS !== 0) {
      throw new Error(
        `WebGPURadixSort: numBits must be a multiple of ${BITS_PER_PASS}`,
      );
    }
    if (!this.ensurePassPipelines(numBits, skipLastPassKeyWrite)) return null;
    if (
      !this.sortUniform ||
      !this.histogramLayout ||
      !this.reorderLayout ||
      !this.prefixLayout ||
      !this.prefixScanPipe ||
      !this.prefixAddPipe
    ) {
      return null;
    }

    this.allocate(elementCount);
    if (
      !this.keys0 ||
      !this.keys1 ||
      !this.values0 ||
      !this.values1 ||
      !this.blockSums
    ) {
      return null;
    }

    const pingKeys = this.keys0;
    const pongKeys = destructiveKeys ? keysBuffer : this.keys1;
    const dispatch = dispatch2d(this.workgroupCount, this.maxWorkgroups);

    this.device.queue.writeBuffer(
      this.sortUniform,
      0,
      new Uint32Array([this.workgroupCount, elementCount, 0, 0]),
    );

    let currentKeys = keysBuffer;
    let nextKeys = pingKeys;
    let currentValues = this.values0;
    let nextValues = this.values1;
    const numPasses = numBits / BITS_PER_PASS;

    for (let pass = 0; pass < numPasses; pass++) {
      const histBg = this.device.createBindGroup({
        layout: this.histogramLayout,
        entries: [
          { binding: 0, resource: { buffer: currentKeys } },
          { binding: 1, resource: { buffer: this.blockSums } },
          { binding: 2, resource: { buffer: this.sortUniform } },
        ],
      });
      {
        const passEnc = encoder.beginComputePass({
          label: `spark-radix-hist-${pass}`,
          ...(pass === 0 && timestamps
            ? {
                timestampWrites: {
                  querySet: timestamps.querySet,
                  beginningOfPassWriteIndex:
                    timestamps.beginningOfPassWriteIndex,
                },
              }
            : {}),
        });
        passEnc.setPipeline(this.histogramPipes[pass]);
        passEnc.setBindGroup(0, histBg);
        passEnc.dispatchWorkgroups(dispatch.x, dispatch.y);
        passEnc.end();
      }

      this.encodePrefixSum(encoder);

      const reorderBg = this.device.createBindGroup({
        layout: this.reorderLayout,
        entries: [
          { binding: 0, resource: { buffer: currentKeys } },
          { binding: 1, resource: { buffer: nextKeys } },
          { binding: 2, resource: { buffer: this.blockSums } },
          { binding: 3, resource: { buffer: currentValues } },
          { binding: 4, resource: { buffer: nextValues } },
          { binding: 5, resource: { buffer: this.sortUniform } },
        ],
      });
      {
        const passEnc = encoder.beginComputePass({
          label: `spark-radix-reorder-${pass}`,
          ...(pass === numPasses - 1 && timestamps
            ? {
                timestampWrites: {
                  querySet: timestamps.querySet,
                  endOfPassWriteIndex: timestamps.endOfPassWriteIndex,
                },
              }
            : {}),
        });
        passEnc.setPipeline(this.reorderPipes[pass]);
        passEnc.setBindGroup(0, reorderBg);
        passEnc.dispatchWorkgroups(dispatch.x, dispatch.y);
        passEnc.end();
      }

      if (pass < numPasses - 1) {
        currentKeys = nextKeys;
        nextKeys = currentKeys === pingKeys ? pongKeys : pingKeys;
        const tmp = currentValues;
        currentValues = nextValues;
        nextValues = tmp;
      } else {
        this.sortedIndices = nextValues;
      }
    }

    return this.sortedIndices;
  }

  private encodePrefixSum(encoder: GPUCommandEncoder): void {
    if (!this.prefixLayout || !this.prefixScanPipe || !this.prefixAddPipe) {
      return;
    }
    for (let i = 0; i < this.activePrefixLevels; i++) {
      const level = this.prefixLevels[i];
      const uniform = this.prefixUniforms[i];
      const bg = this.device.createBindGroup({
        layout: this.prefixLayout,
        entries: [
          { binding: 0, resource: { buffer: level.items } },
          { binding: 1, resource: { buffer: level.blockSums } },
          { binding: 2, resource: { buffer: uniform } },
        ],
      });
      const passEnc = encoder.beginComputePass({ label: "spark-prefix-scan" });
      passEnc.setPipeline(this.prefixScanPipe);
      passEnc.setBindGroup(0, bg);
      passEnc.dispatchWorkgroups(level.dispatch.x, level.dispatch.y);
      passEnc.end();
    }
    for (let i = this.activePrefixLevels - 2; i >= 0; i--) {
      const level = this.prefixLevels[i];
      const uniform = this.prefixUniforms[i];
      const bg = this.device.createBindGroup({
        layout: this.prefixLayout,
        entries: [
          { binding: 0, resource: { buffer: level.items } },
          { binding: 1, resource: { buffer: level.blockSums } },
          { binding: 2, resource: { buffer: uniform } },
        ],
      });
      const passEnc = encoder.beginComputePass({ label: "spark-prefix-add" });
      passEnc.setPipeline(this.prefixAddPipe);
      passEnc.setBindGroup(0, bg);
      passEnc.dispatchWorkgroups(level.dispatch.x, level.dispatch.y);
      passEnc.end();
    }
  }
}

/**
 * CPU replica of WASM sort32_internal polarity: radix on ~key so the largest
 * original metric (far) comes first. Matches premultiplied back-to-front.
 */
export function cpuRadixSort4bitDescending(
  keys: Uint32Array,
  numBits = 32,
): Uint32Array {
  const inverted = new Uint32Array(keys.length);
  for (let i = 0; i < keys.length; i++) {
    inverted[i] = ~keys[i];
  }
  return cpuRadixSort4bit(inverted, numBits);
}

/** CPU 4-bit LSD radix used to test the algorithm independently of WebGPU. */
export function cpuRadixSort4bit(keys: Uint32Array, numBits = 32): Uint32Array {
  const n = keys.length;
  const values = new Uint32Array(n);
  for (let i = 0; i < n; i++) values[i] = i;
  let inKeys = keys.slice();
  let outKeys = new Uint32Array(n);
  let inVals = values;
  let outVals = new Uint32Array(n);
  const passes = Math.ceil(numBits / 4);
  const counts = new Uint32Array(16);
  for (let pass = 0; pass < passes; pass++) {
    const shift = pass * 4;
    counts.fill(0);
    for (let i = 0; i < n; i++) {
      counts[(inKeys[i] >>> shift) & 0xf]++;
    }
    let sum = 0;
    for (let d = 0; d < 16; d++) {
      const c = counts[d];
      counts[d] = sum;
      sum += c;
    }
    for (let i = 0; i < n; i++) {
      const digit = (inKeys[i] >>> shift) & 0xf;
      const dest = counts[digit]++;
      outKeys[dest] = inKeys[i];
      outVals[dest] = inVals[i];
    }
    const tk = inKeys;
    inKeys = outKeys;
    outKeys = tk;
    const tv = inVals;
    inVals = outVals;
    outVals = tv;
  }
  return inVals;
}

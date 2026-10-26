// Copyright © 2011-2026 PlayCanvas Ltd.
// Copyright © Thomas Smith (b0nes164/GPUSorting)
// Copyright © 2025 WORLD LABS TECHNOLOGIES, INC.
// SPDX-License-Identifier: MIT
//
// Spark GPUDevice dispatcher for the OneSweep 8-bit radix sort (Adinets &
// Merrill 2022): one global histogram, one scan and one fused binning
// dispatch per 8-bit pass. Kernels adapted from playcanvas/engine
// 209ce5752527ab5050ec9632ca6dc1272542b70d. See NOTICE.

import onesweepBinningWgsl from "./shaders/onesweepBinning.wgsl?raw";
import onesweepGlobalHistWgsl from "./shaders/onesweepGlobalHist.wgsl?raw";
import onesweepScanWgsl from "./shaders/onesweepScan.wgsl?raw";

const RADIX = 256;
const G_HIST_PART_SIZE = 32768;
// Byte offsets of the two dispatchWorkgroupsIndirect triples in `args`.
export const ONESWEEP_BINNING_ARGS = 0;
export const ONESWEEP_HIST_ARGS = 12;

type Adapterish = { vendor?: string; subgroupMinSize?: number };

/**
 * OneSweep's decoupled lookback spins on earlier partitions, which needs
 * forward-progress guarantees. Like PlayCanvas, enable it on NVIDIA only with
 * a <=32 lane subgroup (the binning ballots use 32-bit masks).
 */
export function oneSweepSupported(device: GPUDevice): boolean {
  if (!device.features.has("subgroups")) return false;
  const info = (device as GPUDevice & { adapterInfo?: Adapterish }).adapterInfo;
  if (!info?.vendor?.toLowerCase().includes("nvidia")) return false;
  return (info.subgroupMinSize || 32) <= 32;
}

/**
 * Sorts `count` u32 keys (count read on the GPU from `countBuffer[0]`) and
 * returns the buffer of sorted values. Workgroup counts come from `args`
 * (binning triple at ONESWEEP_BINNING_ARGS, histogram triple at
 * ONESWEEP_HIST_ARGS); see `dispatchArgs`. The input keys buffer is
 * clobbered. Stable, ascending.
 */
export class WebGPUOneSweepSort {
  readonly partSize: number;
  private device: GPUDevice;
  private histPipes = new Map<number, GPUComputePipeline>();
  private scanPipe: GPUComputePipeline;
  private binningPipes = new Map<string, GPUComputePipeline>();
  private histModule: GPUShaderModule;
  private binningModule: GPUShaderModule;

  private capacity = 0;
  private allocatedBlocks = 0;
  private keys0: GPUBuffer | null = null;
  private values0: GPUBuffer | null = null;
  private values1: GPUBuffer | null = null;
  private globalHist: GPUBuffer;
  private passHist: GPUBuffer | null = null;
  private index: GPUBuffer;
  private bindGroups = new Map<string, GPUBindGroup>();
  private ids = new WeakMap<GPUBuffer, number>();
  private nextId = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    // 15 keys/thread need 16420 B of workgroup memory.
    const keysPerThread =
      device.limits.maxComputeWorkgroupStorageSize >= 16420 ? 15 : 14;
    this.partSize = 256 * keysPerThread;
    const info = (device as GPUDevice & { adapterInfo?: Adapterish })
      .adapterInfo;
    const maxSubgroups = Math.ceil(256 / (info?.subgroupMinSize || 32));
    const patch = (code: string) =>
      code
        .replace(
          /const KEYS_PER_THREAD: u32 = \d+u;/,
          `const KEYS_PER_THREAD: u32 = ${keysPerThread}u;`,
        )
        .replace(
          /const PART_SIZE: u32 = \d+u;/,
          `const PART_SIZE: u32 = ${this.partSize}u;`,
        )
        .replace(
          /const MAX_SUBGROUPS: u32 = \d+u;/,
          `const MAX_SUBGROUPS: u32 = ${maxSubgroups}u;`,
        );
    this.histModule = device.createShaderModule({
      label: "spark-onesweep-hist",
      code: onesweepGlobalHistWgsl,
    });
    this.binningModule = device.createShaderModule({
      label: "spark-onesweep-binning",
      code: patch(onesweepBinningWgsl),
    });
    this.scanPipe = device.createComputePipeline({
      label: "spark-onesweep-scan",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "spark-onesweep-scan",
          code: patch(onesweepScanWgsl),
        }),
        entryPoint: "main",
      },
    });
    this.globalHist = this.buffer("spark-onesweep-global-hist", 4 * RADIX * 4);
    this.index = this.buffer("spark-onesweep-index", 16);
  }

  /** Compile the pipelines for `numBits` ahead of the first sort. */
  warmup(numBits: number, initialValues: boolean): void {
    this.histPipe(numBits / 8);
    for (let pass = 0; pass < numBits / 8; pass++)
      this.binningPipe(pass, this.flags(pass, numBits / 8, initialValues));
  }

  /** Indirect args for a host-known count (binning, then histogram). */
  dispatchArgs(count: number): Uint32Array {
    return new Uint32Array([
      Math.max(1, Math.ceil(count / this.partSize)),
      1,
      1,
      Math.max(1, Math.ceil(count / G_HIST_PART_SIZE)),
      1,
      1,
    ]);
  }

  encodeSort(
    encoder: GPUCommandEncoder,
    keys: GPUBuffer,
    capacity: number,
    countBuffer: GPUBuffer,
    args: GPUBuffer,
    numBits: number,
    initialValues?: GPUBuffer,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): GPUBuffer {
    const numPasses = numBits / 8;
    this.allocate(capacity);
    const passHist = this.passHist as GPUBuffer;
    const keys0 = this.keys0 as GPUBuffer;
    const values0 = this.values0 as GPUBuffer;
    const values1 = this.values1 as GPUBuffer;
    encoder.clearBuffer(this.globalHist);
    encoder.clearBuffer(
      passHist,
      0,
      numPasses * this.allocatedBlocks * RADIX * 4,
    );
    encoder.clearBuffer(this.index);

    const pass = encoder.beginComputePass({
      label: "spark-onesweep",
      timestampWrites,
    });
    const hist = this.histPipe(numPasses);
    pass.setPipeline(hist);
    pass.setBindGroup(
      0,
      this.bindGroup(hist, "hist", [keys, this.globalHist, countBuffer]),
    );
    pass.dispatchWorkgroupsIndirect(args, ONESWEEP_HIST_ARGS);
    pass.setPipeline(this.scanPipe);
    pass.setBindGroup(
      0,
      this.bindGroup(this.scanPipe, "scan", [
        this.globalHist,
        passHist,
        countBuffer,
      ]),
    );
    pass.dispatchWorkgroups(numPasses);

    // Destructive ping-pong: the caller's key buffer is the second key slot.
    let inKeys = keys;
    let outKeys = keys0;
    let inValues = initialValues ?? values0;
    let outValues = values1;
    for (let p = 0; p < numPasses; p++) {
      const pipe = this.binningPipe(
        p,
        this.flags(p, numPasses, !!initialValues),
      );
      pass.setPipeline(pipe);
      pass.setBindGroup(
        0,
        this.bindGroup(pipe, `bin${p}`, [
          inKeys,
          outKeys,
          inValues,
          outValues,
          passHist,
          this.index,
          countBuffer,
        ]),
      );
      pass.dispatchWorkgroupsIndirect(args, ONESWEEP_BINNING_ARGS);
      if (p < numPasses - 1) {
        inKeys = outKeys;
        outKeys = inKeys === keys0 ? keys : keys0;
        inValues = outValues;
        outValues = inValues === values0 ? values1 : values0;
      }
    }
    pass.end();
    return outValues;
  }

  destroy(): void {
    this.keys0?.destroy();
    this.values0?.destroy();
    this.values1?.destroy();
    this.passHist?.destroy();
    this.globalHist.destroy();
    this.index.destroy();
    this.bindGroups.clear();
  }

  private flags(pass: number, numPasses: number, initialValues: boolean) {
    return (
      (pass === 0 && !initialValues ? 1 : 0) | (pass === numPasses - 1 ? 2 : 0)
    );
  }

  private histPipe(numPasses: number): GPUComputePipeline {
    let pipe = this.histPipes.get(numPasses);
    if (!pipe) {
      pipe = this.device.createComputePipeline({
        label: `spark-onesweep-hist-${numPasses}`,
        layout: "auto",
        compute: {
          module: this.histModule,
          entryPoint: "main",
          constants: { NUM_PASSES: numPasses },
        },
      });
      this.histPipes.set(numPasses, pipe);
    }
    return pipe;
  }

  private binningPipe(pass: number, flags: number): GPUComputePipeline {
    const key = `${pass}:${flags}`;
    let pipe = this.binningPipes.get(key);
    if (!pipe) {
      pipe = this.device.createComputePipeline({
        label: `spark-onesweep-binning-${key}`,
        layout: "auto",
        compute: {
          module: this.binningModule,
          entryPoint: "main",
          constants: { PASS: pass, FLAGS: flags },
        },
      });
      this.binningPipes.set(key, pipe);
    }
    return pipe;
  }

  // Bind groups are cached per pipeline slot and buffer identity; buffers
  // are only replaced on growth, so steady-state sorts create none.
  private bindGroup(
    pipe: GPUComputePipeline,
    slot: string,
    buffers: GPUBuffer[],
  ) {
    const key = `${slot}:${pipe.label}:${buffers.map((b) => this.id(b)).join(",")}`;
    let bg = this.bindGroups.get(key);
    if (!bg) {
      if (this.bindGroups.size > 64) this.bindGroups.clear();
      bg = this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({
          binding,
          resource: { buffer },
        })),
      });
      this.bindGroups.set(key, bg);
    }
    return bg;
  }

  private id(buffer: GPUBuffer): number {
    let id = this.ids.get(buffer);
    if (id === undefined) {
      id = this.nextId++;
      this.ids.set(buffer, id);
    }
    return id;
  }

  private buffer(label: string, size: number): GPUBuffer {
    return this.device.createBuffer({
      label,
      size: Math.ceil(size / 256) * 256,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });
  }

  private allocate(capacity: number): void {
    if (capacity <= this.capacity && this.keys0) return;
    this.keys0?.destroy();
    this.values0?.destroy();
    this.values1?.destroy();
    this.passHist?.destroy();
    this.capacity = capacity;
    this.allocatedBlocks = Math.max(1, Math.ceil(capacity / this.partSize));
    this.keys0 = this.buffer("spark-onesweep-keys0", capacity * 4);
    this.values0 = this.buffer("spark-onesweep-values0", capacity * 4);
    this.values1 = this.buffer("spark-onesweep-values1", capacity * 4);
    this.passHist = this.buffer(
      "spark-onesweep-pass-hist",
      4 * this.allocatedBlocks * RADIX * 4,
    );
  }
}

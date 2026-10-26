export declare const ONESWEEP_BINNING_ARGS = 0;
export declare const ONESWEEP_HIST_ARGS = 12;
/**
 * OneSweep's decoupled lookback spins on earlier partitions, which needs
 * forward-progress guarantees. Like PlayCanvas, enable it on NVIDIA only with
 * a <=32 lane subgroup (the binning ballots use 32-bit masks).
 */
export declare function oneSweepSupported(device: GPUDevice): boolean;
/**
 * Sorts `count` u32 keys (count read on the GPU from `countBuffer[0]`) and
 * returns the buffer of sorted values. Workgroup counts come from `args`
 * (binning triple at ONESWEEP_BINNING_ARGS, histogram triple at
 * ONESWEEP_HIST_ARGS); see `dispatchArgs`. The input keys buffer is
 * clobbered. Stable, ascending.
 */
export declare class WebGPUOneSweepSort {
    readonly partSize: number;
    private device;
    private histPipes;
    private scanPipe;
    private binningPipes;
    private histModule;
    private binningModule;
    private capacity;
    private allocatedBlocks;
    private keys0;
    private values0;
    private values1;
    private globalHist;
    private passHist;
    private index;
    private bindGroups;
    private ids;
    private nextId;
    constructor(device: GPUDevice);
    /** Compile the pipelines for `numBits` ahead of the first sort. */
    warmup(numBits: number, initialValues: boolean): void;
    /** Indirect args for a host-known count (binning, then histogram). */
    dispatchArgs(count: number): Uint32Array;
    encodeSort(encoder: GPUCommandEncoder, keys: GPUBuffer, capacity: number, countBuffer: GPUBuffer, args: GPUBuffer, numBits: number, initialValues?: GPUBuffer, timestampWrites?: GPUComputePassTimestampWrites): GPUBuffer;
    destroy(): void;
    private flags;
    private histPipe;
    private binningPipe;
    private bindGroup;
    private id;
    private buffer;
    private allocate;
}

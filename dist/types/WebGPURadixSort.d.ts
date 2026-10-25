/**
 * Portable 4-bit GPU radix sort. Direct dispatch only (CPU knows N).
 * Returns a storage buffer of sorted u32 indices.
 */
export declare class WebGPURadixSort {
    readonly radixBits = 4;
    private device;
    private maxWorkgroups;
    private available;
    private histogramModule;
    private reorderModule;
    private prefixModule;
    private histogramLayout;
    private reorderLayout;
    private prefixLayout;
    private histogramPipes;
    private reorderPipes;
    private prefixScanPipe;
    private prefixAddPipe;
    private sortUniform;
    private prefixUniforms;
    private keys0;
    private keys1;
    private values0;
    private values1;
    private blockSums;
    private prefixLevels;
    private capacity;
    private allocatedWorkgroups;
    private workgroupCount;
    private prefixWorkgroupCount;
    private activePrefixLevels;
    private lastNumBits;
    private lastSkipLastKeyWrite;
    private sortedIndices;
    /**
     * Incremented every time the internal ping/pong buffers are destroyed and
     * reallocated. Callers that cache a bind group referencing `encodeSort`'s
     * returned buffer must record the epoch alongside it and rebuild the bind
     * group when the epoch changes — the old GPUBuffer is destroyed at that
     * point and using it is a validation error.
     */
    bufferEpoch: number;
    constructor(device: GPUDevice);
    isAvailable(): boolean;
    /**
     * Compile histogram/reorder pipelines for `numBits` before the first
     * encodeSort so a compile failure can latch fallback without abandoning
     * an in-flight command encoder.
     */
    warmup(numBits?: number, skipLastKeyWrite?: boolean): boolean;
    destroy(): void;
    private compile;
    private ensurePassPipelines;
    private destroyBuffers;
    private storageBuffer;
    private allocate;
    private rebuildPrefixLevels;
    private configurePrefixLevels;
    /**
     * Encode a radix sort into `encoder`. Keys are u32. Returns the buffer of
     * sorted indices (valid until the next sort or destroy).
     */
    encodeSort(encoder: GPUCommandEncoder, keysBuffer: GPUBuffer, elementCount: number, numBits?: number, skipLastPassKeyWrite?: boolean, destructiveKeys?: boolean, timestamps?: GPUComputePassTimestampWrites): GPUBuffer | null;
    private encodePrefixSum;
}
/**
 * CPU replica of WASM sort32_internal polarity: radix on ~key so the largest
 * original metric (far) comes first. Matches premultiplied back-to-front.
 */
export declare function cpuRadixSort4bitDescending(keys: Uint32Array, numBits?: number): Uint32Array;
/** CPU 4-bit LSD radix used to test the algorithm independently of WebGPU. */
export declare function cpuRadixSort4bit(keys: Uint32Array, numBits?: number): Uint32Array;

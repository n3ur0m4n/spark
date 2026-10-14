/** Below this many splats the single-worker sort is used. */
export declare const SORT_PARALLEL_MIN_SPLATS = 1000000;
export type ParallelSortBuffers = {
    capacity: number;
    numWorkers: number;
    keys: Uint32Array;
    pairs?: Uint32Array;
    keys16?: Uint16Array;
    ordering: Uint32Array;
    hists: Uint32Array;
};
export declare function ensureParallelSortBuffers(buffers: ParallelSortBuffers | undefined, capacity: number, numWorkers: number): ParallelSortBuffers;
/**
 * Turn per-worker histograms (in place) into per-worker global scatter
 * offsets and split the buckets into numRanges contiguous ranges with ~equal
 * splat counts. Returns flat [start, end] per range.
 */
export declare function planSort32(hists: Uint32Array, numWorkers: number, numRanges: number): {
    active: number;
    ranges: number[];
};
export type SortCall = (worker: number, name: "sortParHist" | "sortParScatter" | "sortParRegion" | "sortPar16Hist" | "sortPar16Scatter", args: Record<string, unknown>) => Promise<unknown>;
/** Sort buffers.keys[0..numSplats) into buffers.ordering, returns activeSplats. */
export declare function parallelSort32({ numSplats, buffers, call, }: {
    numSplats: number;
    buffers: ParallelSortBuffers;
    call: SortCall;
}): Promise<number>;
/**
 * 16-bit sort: keys rounded to f16 in WASM (sort16_hist), one counting pass.
 * Order matches sort32_as16_splats: descending f16 key, ties by ascending
 * index. Same contract as parallelSort32.
 */
export declare function parallelSort16({ numSplats, buffers, call, }: {
    numSplats: number;
    buffers: ParallelSortBuffers;
    call: SortCall;
}): Promise<number>;
export declare function sortPar16ScatterJs({ keys16, start, end, offsets, ordering, }: {
    keys16: Uint16Array;
    start: number;
    end: number;
    offsets: Uint32Array;
    ordering: Uint32Array;
}): void;
export declare function sortParScatterJs({ keys, start, end, offsets, pairs, }: {
    keys: Uint32Array;
    start: number;
    end: number;
    offsets: Uint32Array;
    pairs: Uint32Array;
}): void;

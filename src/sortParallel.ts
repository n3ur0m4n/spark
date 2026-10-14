// Multi-worker depth sort, coordinated from JS over SharedArrayBuffer.
// Each SplatWorker runs the single-threaded WASM; the sort is a hybrid
// MSD (high 16 bits, global) then local LSD (per contiguous bucket range).
// Output order matches sort32_internal exactly: active keys (< 0x7f800000)
// sorted by descending key, ties by ascending index.
// No imports on purpose: bench/sort-bench loads this file directly in Node.

/** Below this many splats the single-worker sort is used. */
export const SORT_PARALLEL_MIN_SPLATS = 1_000_000;

const RADIX = 65536;
const DEPTH_INFINITY_F32 = 0x7f800000;
// 16-bit sort: f16 keys, 0x7c00 (+inf) and above excluded
const DEPTH_INFINITY_F16 = 0x7c00;
const HIST16 = DEPTH_INFINITY_F16 + 1;

export type ParallelSortBuffers = {
  capacity: number;
  numWorkers: number;
  keys: Uint32Array;
  // interleaved (key, index), grouped by high 16 bits (32-bit sort only)
  pairs?: Uint32Array;
  // f16 keys (16-bit sort only)
  keys16?: Uint16Array;
  ordering: Uint32Array;
  // numWorkers x 65536 histograms, turned into per-worker scatter offsets
  hists: Uint32Array;
};

function sharedU32(length: number) {
  return new Uint32Array(new SharedArrayBuffer(length * 4));
}

export function ensureParallelSortBuffers(
  buffers: ParallelSortBuffers | undefined,
  capacity: number,
  numWorkers: number,
): ParallelSortBuffers {
  if (
    buffers &&
    buffers.capacity >= capacity &&
    buffers.numWorkers === numWorkers
  ) {
    return buffers;
  }
  return {
    capacity,
    numWorkers,
    keys: sharedU32(capacity),
    ordering: sharedU32(capacity),
    hists: sharedU32(numWorkers * RADIX),
  };
}

/**
 * Turn per-worker histograms (in place) into per-worker global scatter
 * offsets and split the buckets into numRanges contiguous ranges with ~equal
 * splat counts. Returns flat [start, end] per range.
 */
export function planSort32(
  hists: Uint32Array,
  numWorkers: number,
  numRanges: number,
) {
  const bucketStart = new Uint32Array(RADIX + 1);
  let sum = 0;
  for (let b = 0; b < RADIX; ++b) {
    bucketStart[b] = sum;
    for (let w = 0; w < numWorkers; ++w) {
      const i = w * RADIX + b;
      const count = hists[i];
      hists[i] = sum;
      sum += count;
    }
  }
  const active = sum;
  bucketStart[RADIX] = active;

  const ranges: number[] = [];
  let b0 = 0;
  let k = 1;
  for (let b = 0; b < RADIX && k < numRanges; ++b) {
    if (bucketStart[b + 1] * numRanges >= active * k) {
      ranges.push(bucketStart[b0], bucketStart[b + 1]);
      b0 = b + 1;
      k += 1;
    }
  }
  ranges.push(bucketStart[b0], active);
  return { active, ranges };
}

export type SortCall = (
  worker: number,
  name:
    | "sortParHist"
    | "sortParScatter"
    | "sortParRegion"
    | "sortPar16Hist"
    | "sortPar16Scatter",
  args: Record<string, unknown>,
) => Promise<unknown>;

/** Sort buffers.keys[0..numSplats) into buffers.ordering, returns activeSplats. */
export async function parallelSort32({
  numSplats,
  buffers,
  call,
}: {
  numSplats: number;
  buffers: ParallelSortBuffers;
  call: SortCall;
}): Promise<number> {
  const { numWorkers, keys, ordering, hists } = buffers;
  buffers.pairs ??= sharedU32(2 * buffers.capacity);
  const pairs = buffers.pairs;
  const chunk = Math.ceil(numSplats / numWorkers);
  const slice = (w: number) => ({
    start: Math.min(w * chunk, numSplats),
    end: Math.min((w + 1) * chunk, numSplats),
  });
  const workers = Array.from({ length: numWorkers }, (_, w) => w);

  await Promise.all(
    workers.map((w) =>
      call(w, "sortParHist", {
        keys,
        ...slice(w),
        hist: hists.subarray(w * RADIX, (w + 1) * RADIX),
      }),
    ),
  );

  const { active, ranges } = planSort32(hists, numWorkers, numWorkers);

  await Promise.all(
    workers.map((w) =>
      call(w, "sortParScatter", {
        keys,
        ...slice(w),
        offsets: hists.subarray(w * RADIX, (w + 1) * RADIX),
        pairs,
      }),
    ),
  );

  const regions = [];
  for (let r = 0; r * 2 < ranges.length; ++r) {
    const start = ranges[r * 2];
    const end = ranges[r * 2 + 1];
    if (start < end) {
      regions.push(
        call(r, "sortParRegion", {
          pairs: pairs.subarray(2 * start, 2 * end),
          ordering: ordering.subarray(start, end),
        }),
      );
    }
  }
  await Promise.all(regions);
  return active;
}

/**
 * 16-bit sort: keys rounded to f16 in WASM (sort16_hist), one counting pass.
 * Order matches sort32_as16_splats: descending f16 key, ties by ascending
 * index. Same contract as parallelSort32.
 */
export async function parallelSort16({
  numSplats,
  buffers,
  call,
}: {
  numSplats: number;
  buffers: ParallelSortBuffers;
  call: SortCall;
}): Promise<number> {
  const { numWorkers, keys, ordering, hists } = buffers;
  buffers.keys16 ??= new Uint16Array(
    new SharedArrayBuffer(buffers.capacity * 2),
  );
  const keys16 = buffers.keys16;
  const chunk = Math.ceil(numSplats / numWorkers);
  const workers = Array.from({ length: numWorkers }, (_, w) => w);
  const start = (w: number) => Math.min(w * chunk, numSplats);
  const end = (w: number) => Math.min((w + 1) * chunk, numSplats);
  const hist = (w: number) => hists.subarray(w * HIST16, (w + 1) * HIST16);

  await Promise.all(
    workers.map((w) =>
      call(w, "sortPar16Hist", {
        keys: keys.subarray(start(w), end(w)),
        keys16: keys16.subarray(start(w), end(w)),
        hist: hist(w),
      }),
    ),
  );

  // counts -> per-worker offsets, largest key first, excluded keys last
  // (after the active ones: the scatter needs no branch)
  let sum = 0;
  let active = 0;
  for (let b = DEPTH_INFINITY_F16 - 1; b >= -1; --b) {
    if (b < 0) {
      active = sum;
    }
    const bucket = b < 0 ? DEPTH_INFINITY_F16 : b;
    for (let w = 0; w < numWorkers; ++w) {
      const i = w * HIST16 + bucket;
      const count = hists[i];
      hists[i] = sum;
      sum += count;
    }
  }

  await Promise.all(
    workers.map((w) =>
      call(w, "sortPar16Scatter", {
        keys16,
        start: start(w),
        end: end(w),
        offsets: hist(w),
        ordering,
      }),
    ),
  );
  return active;
}

export function sortPar16ScatterJs({
  keys16,
  start,
  end,
  offsets,
  ordering,
}: {
  keys16: Uint16Array;
  start: number;
  end: number;
  offsets: Uint32Array;
  ordering: Uint32Array;
}) {
  const offs = offsets.slice();
  for (let i = start; i < end; ++i) {
    ordering[offs[keys16[i]]++] = i;
  }
}

// Scatter kernel (runs inside workers). JS directly on the shared buffers:
// the writes are global, so WASM would need its (unshared) memory copied out
// per bucket, which measured slower (bench/sort-bench).

export function sortParScatterJs({
  keys,
  start,
  end,
  offsets,
  pairs,
}: {
  keys: Uint32Array;
  start: number;
  end: number;
  offsets: Uint32Array;
  pairs: Uint32Array;
}) {
  const offs = offsets.slice();
  for (let i = start; i < end; ++i) {
    const key = keys[i];
    if (key < DEPTH_INFINITY_F32) {
      const pos = offs[~key >>> 16]++ * 2;
      pairs[pos] = key;
      pairs[pos + 1] = i;
    }
  }
}

// Incremental sort candidate (bench only, see matrix.mjs).
// Input: previous ordering + f16 keys of the new frame. A small camera move
// keeps the previous order almost sorted, so: gather the keys in previous
// order keeping a non-increasing run (one-step dip repair), push violators
// to "outliers", add splats that became active, counting-sort the outliers
// and merge them with the run(s). O(n) when few elements move; worst case
// falls back to sorting the outliers plus one merge pass.
const INF16 = 0x7c00;

// prev[0..prevLen) -> run (non-increasing by keys16) + outliers; marks seen.
// Returns [runLen, outLen].
export function incrGather(prev, keys16, run, out, seen) {
  let m = 0;
  let o = 0;
  let last = 0xffff;
  for (let i = 0; i < prev.length; ++i) {
    const idx = prev[i];
    seen[idx] = 1;
    const k = keys16[idx];
    if (k >= INF16) continue;
    if (k <= last) {
      run[m++] = idx;
      last = k;
    } else if (m >= 2 && k <= keys16[run[m - 2]]) {
      // previous element dipped: eject it instead
      out[o++] = run[m - 1];
      run[m - 1] = idx;
      last = k;
    } else {
      out[o++] = idx;
    }
  }
  return [m, o];
}

// Active splats in [start, end) not seen in the previous ordering; clears seen.
export function incrNew(keys16, seen, start, end, out, o) {
  for (let i = start; i < end; ++i) {
    if (seen[i]) seen[i] = 0;
    else if (keys16[i] < INF16) out[o++] = i;
  }
  return o;
}

// Stable counting sort of indices by keys16, descending.
export function countingSort16(idx, n, keys16, dst) {
  const hist = new Uint32Array(INF16 + 1);
  for (let i = 0; i < n; ++i) hist[keys16[idx[i]]]++;
  let sum = 0;
  for (let b = INF16 - 1; b >= 0; --b) {
    const c = hist[b];
    hist[b] = sum;
    sum += c;
  }
  for (let i = 0; i < n; ++i) dst[hist[keys16[idx[i]]]++] = idx[i];
}

// k-way merge of descending runs [{arr, start, end}] into dst.
export function mergeRuns(runs, keys16, dst) {
  const heads = runs.map((r) => r.start);
  let d = 0;
  for (;;) {
    let best = -1;
    let bestKey = -1;
    for (let r = 0; r < runs.length; ++r) {
      if (heads[r] < runs[r].end) {
        const k = keys16[runs[r].arr[heads[r]]];
        if (k > bestKey) {
          bestKey = k;
          best = r;
        }
      }
    }
    if (best < 0) return d;
    dst[d++] = runs[best].arr[heads[best]++];
  }
}

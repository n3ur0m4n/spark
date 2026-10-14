// Sort matrix: {32-bit, 16-bit, incremental} x {1, 4 threads} x cull variants,
// on radial depths of a realistic point set after a small camera move.
//   node bench/sort-bench/matrix.mjs [n] [reps]
import { readFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import init, {
  sort32_as16_splats,
  sort32_splats,
  sort16_par_hist,
} from "../../rust/spark-rs/pkg/spark_rs.js";
import {
  ensureParallelSortBuffers,
  parallelSort16,
  parallelSort32,
} from "../../src/sortParallel.ts";
import {
  countingSort16,
  incrGather,
  incrNew,
  mergeRuns,
} from "./incremental.mjs";

await init({
  module_or_path: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

const N = Number(process.argv[2] ?? 16 * 1024 * 1024);
const REPS = Number(process.argv[3] ?? 5);
const THREADS = 4;
const INF = 0x7f800000;
const INF16 = 0x7c00;

let seed = 12345;
const rand = () => {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
};

// Points in a 100x10x100 m box, x-sorted runs ("sorted-ish" like splat files)
const pts = new Float32Array(3 * N);
for (let i = 0; i < N; ++i) {
  pts[3 * i] = (i / N) * 100 + (rand() - 0.5) * 2;
  pts[3 * i + 1] = rand() * 10;
  pts[3 * i + 2] = rand() * 100;
}

const deg = Math.PI / 180;
// cone = diagonal FOV of a 60 deg vfov 16:9 camera + 30 deg (as the app)
const diagHalf = Math.atan(Math.tan(30 * deg) * Math.sqrt(1 + (16 / 9) ** 2));
const coneFov = Math.min(179, (2 * diagHalf) / deg + 30);
const coneCos = Math.cos((coneFov / 2) * deg);

const camA = { pos: [10, 1.6, 50], yaw: Math.atan2(-0.6, 0.8) };
// small move: 5 cm translation, 3 deg yaw
const camB = { pos: [10.03, 1.6, 50.04], yaw: camA.yaw + 3 * deg };

const f32 = new Float32Array(N);
const keyBits = new Uint32Array(f32.buffer);
// radial metric as OutputSplatDepth (sortRadial), optional cone cull
function keysFor(cam, cosHalf) {
  const [cx, cy, cz] = cam.pos;
  const dx = Math.cos(cam.yaw);
  const dz = Math.sin(cam.yaw);
  for (let i = 0; i < N; ++i) {
    const x = pts[3 * i] - cx;
    const y = pts[3 * i + 1] - cy;
    const z = pts[3 * i + 2] - cz;
    const len = Math.sqrt(x * x + y * y + z * z);
    f32[i] = len > 0 && x * dx + z * dz < cosHalf * len ? Number.POSITIVE_INFINITY : len;
  }
  return keyBits.slice();
}

// Independent f16 reference (Float16Array rounding), excluded -> 0x7c00
function f16Keys(keys) {
  const h = new Uint16Array(
    new Float16Array(new Float32Array(keys.buffer)).buffer,
  );
  for (let i = 0; i < keys.length; ++i) {
    if (keys[i] >= INF) h[i] = INF16;
    else if (h[i] >= INF16) h[i] = 0x7bff;
  }
  return h;
}
function reference16(k16) {
  const idx = new Uint32Array(k16.length);
  let n = 0;
  for (let i = 0; i < k16.length; ++i) if (k16[i] < INF16) idx[n++] = i;
  const dst = new Uint32Array(n);
  countingSort16(idx, n, k16, dst);
  return dst;
}

function verify(keys, active, ordering, cmpKeys) {
  let expected = 0;
  for (let i = 0; i < keys.length; ++i) if (keys[i] < INF) expected++;
  if (expected !== active) throw new Error(`active ${active} != ${expected}`);
  const seen = new Uint8Array(keys.length);
  for (let i = 0; i < active; ++i) {
    const idx = ordering[i];
    if (idx >= keys.length || seen[idx] || keys[idx] >= INF)
      throw new Error(`not a permutation of active at ${i}`);
    seen[idx] = 1;
    if (i > 0 && cmpKeys[ordering[i - 1]] < cmpKeys[idx])
      throw new Error(`not non-increasing at ${i}`);
  }
}

function makePool(numWorkers) {
  let nextId = 0;
  const pending = new Map();
  const workers = [];
  const ready = [];
  for (let w = 0; w < numWorkers; ++w) {
    const worker = new Worker(new URL("./worker.mjs", import.meta.url));
    ready.push(
      new Promise((resolve) => {
        worker.on("message", (msg) => {
          if (msg.ready) return resolve();
          pending.get(msg.id)(msg.ms);
          pending.delete(msg.id);
        });
      }),
    );
    workers.push(worker);
  }
  return {
    ready: Promise.all(ready),
    call(w, name, args, mode = "wasm") {
      const id = nextId++;
      return new Promise((resolve) => {
        pending.set(id, resolve);
        workers[w].postMessage({ id, name, args, mode });
      });
    },
    terminate: () => Promise.all(workers.map((w) => w.terminate())),
  };
}
const pool = makePool(THREADS);
await pool.ready;

const shared = (C, n) => new C(new SharedArrayBuffer(n * C.BYTES_PER_ELEMENT));
const buffers = ensureParallelSortBuffers(undefined, N, THREADS);
const out = new Uint32Array(N);
const JS_KERNELS = new Set(["sortParScatter", "sortPar16Scatter"]);
const callPool = (w, name, args) =>
  pool.call(w, name, args, JS_KERNELS.has(name) ? "js" : "wasm");

// Incremental state (shared so workers can use it)
const keys16S = shared(Uint16Array, N);
const seen = shared(Uint8Array, N);
const runS = shared(Uint32Array, N);
const outlS = shared(Uint32Array, 2 * N);
const lensS = shared(Uint32Array, 2 * THREADS);
const prevS = shared(Uint32Array, N);
const histS = shared(Uint32Array, THREADS * (INF16 + 1));
const outlAll = new Uint32Array(N);
const outlSorted = new Uint32Array(N);
const keys16 = new Uint16Array(N);
const hist1 = new Uint32Array(INF16 + 1);
const seen1 = new Uint8Array(N);

const algos = {
  "32 1T": async (keys) => {
    const readback = keys.slice();
    const t0 = performance.now();
    const active = sort32_splats(N, readback, out);
    return { ms: performance.now() - t0, active };
  },
  "32 4T": async (keys) => {
    const t0 = performance.now();
    buffers.keys.set(keys);
    const active = await parallelSort32({
      numSplats: N,
      buffers,
      call: callPool,
    });
    out.set(buffers.ordering.subarray(0, active));
    return { ms: performance.now() - t0, active };
  },
  "16 1T": async (keys) => {
    const readback = keys.slice();
    const t0 = performance.now();
    const active = sort32_as16_splats(N, readback, out);
    return { ms: performance.now() - t0, active };
  },
  "16 4T": async (keys) => {
    const t0 = performance.now();
    buffers.keys.set(keys);
    const active = await parallelSort16({
      numSplats: N,
      buffers,
      call: callPool,
    });
    out.set(buffers.ordering.subarray(0, active));
    return { ms: performance.now() - t0, active };
  },
  "incr 1T": async (keys, prev) => {
    const t0 = performance.now();
    sort16_par_hist(keys, keys16, hist1);
    const [m, o0] = incrGather(prev, keys16, runS, outlS, seen1);
    const o = incrNew(keys16, seen1, 0, N, outlS, o0);
    countingSort16(outlS, o, keys16, outlSorted);
    const active = mergeRuns(
      [
        { arr: runS, start: 0, end: m },
        { arr: outlSorted, start: 0, end: o },
      ],
      keys16,
      out,
    );
    return { ms: performance.now() - t0, active, outliers: o };
  },
  "incr 4T": async (keys, prev) => {
    const t0 = performance.now();
    buffers.keys.set(keys);
    prevS.set(prev);
    const chunk = Math.ceil(N / THREADS);
    const pchunk = Math.ceil(prev.length / THREADS);
    const ws = [...Array(THREADS).keys()];
    const lo = (w, c, n) => Math.min(w * c, n);
    const hi = (w, c, n) => Math.min((w + 1) * c, n);
    const lens = (w) => lensS.subarray(2 * w, 2 * w + 2);
    await Promise.all(
      ws.map((w) =>
        pool.call(w, "sortPar16Hist", {
          keys: buffers.keys.subarray(lo(w, chunk, N), hi(w, chunk, N)),
          keys16: keys16S.subarray(lo(w, chunk, N), hi(w, chunk, N)),
          hist: histS.subarray(w * (INF16 + 1), (w + 1) * (INF16 + 1)),
        }),
      ),
    );
    await Promise.all(
      ws.map((w) =>
        pool.call(
          w,
          "incrGather",
          {
            prev: prevS.subarray(
              lo(w, pchunk, prev.length),
              hi(w, pchunk, prev.length),
            ),
            keys16: keys16S,
            run: runS.subarray(w * pchunk),
            out: outlS.subarray(2 * w * chunk),
            seen,
            lens: lens(w),
          },
          "js",
        ),
      ),
    );
    await Promise.all(
      ws.map((w) =>
        pool.call(
          w,
          "incrNew",
          {
            keys16: keys16S,
            seen,
            start: lo(w, chunk, N),
            end: hi(w, chunk, N),
            out: outlS.subarray(2 * w * chunk),
            lens: lens(w),
          },
          "js",
        ),
      ),
    );
    // outliers of all workers, then main-thread counting sort + k-way merge
    let o = 0;
    for (const w of ws) {
      const n = lensS[2 * w + 1];
      outlAll.set(outlS.subarray(2 * w * chunk, 2 * w * chunk + n), o);
      o += n;
    }
    countingSort16(outlAll, o, keys16S, outlSorted);
    const runs = ws.map((w) => ({
      arr: runS,
      start: w * pchunk,
      end: w * pchunk + lensS[2 * w],
    }));
    runs.push({ arr: outlSorted, start: 0, end: o });
    const active = mergeRuns(runs, keys16S, out);
    return { ms: performance.now() - t0, active, outliers: o };
  },
};

const median = (xs) => xs.slice().sort((a, b) => a - b)[xs.length >> 1];
const rows = [];

// variants: cull fraction (random mask, same in both frames) or cone
const variants = [
  ["cull 0%", 0, -2],
  ["cull 30%", 0.3, -2],
  ["cull 50%", 0.5, -2],
  [`cone ${coneFov.toFixed(1)}deg`, 0, coneCos],
];
for (const [vname, frac, cosHalf] of variants) {
  const keysA = keysFor(camA, cosHalf);
  const keysB = keysFor(camB, cosHalf);
  if (frac > 0) {
    seed = 999;
    for (let i = 0; i < N; ++i) {
      if (rand() < frac) {
        keysA[i] = INF;
        keysB[i] = INF;
      }
    }
  }
  let culled = 0;
  for (let i = 0; i < N; ++i) if (keysB[i] >= INF) culled++;
  const k16 = f16Keys(keysB);
  const ref16 = reference16(k16);
  // previous frame ordering (16-bit, as the renderer would keep it)
  const prevAll = new Uint32Array(N);
  const prevActive = sort32_as16_splats(N, keysA.slice(), prevAll);
  const prev = prevAll.slice(0, prevActive);

  for (const [aname, fn] of Object.entries(algos)) {
    const times = [];
    let res;
    for (let r = 0; r < REPS; ++r) {
      out.fill(0xffffffff);
      res = await fn(keysB, prev);
      times.push(res.ms);
    }
    let correct = "PASS";
    try {
      verify(keysB, res.active, out, aname.startsWith("32") ? keysB : k16);
      if (aname.startsWith("16")) {
        for (let i = 0; i < res.active; ++i)
          if (out[i] !== ref16[i]) throw new Error(`!= f16 reference at ${i}`);
      }
    } catch (e) {
      correct = `FAIL ${e.message}`;
    }
    rows.push({
      variant: vname,
      culled: `${((100 * culled) / N).toFixed(1)}%`,
      algo: aname,
      ms: median(times).toFixed(1),
      outliers:
        res.outliers === undefined
          ? ""
          : `${((100 * res.outliers) / Math.max(1, res.active)).toFixed(1)}%`,
      correct,
    });
    console.error(JSON.stringify(rows.at(-1)));
  }
}
console.log(`n=${N} reps=${REPS} (median ms), threads 1/${THREADS}`);
console.table(rows);
await pool.terminate();

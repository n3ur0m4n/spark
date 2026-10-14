// Parallel depth sort bench + correctness. Run from spark-fork:
//   node bench/sort-bench/bench.mjs [n] [reps]
import { readFileSync } from "node:fs";
import { Worker } from "node:worker_threads";
import init, { sort32_splats } from "../../rust/spark-rs/pkg/spark_rs.js";
import {
  ensureParallelSortBuffers,
  parallelSort32,
} from "../../src/sortParallel.ts";

await init({
  module_or_path: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

const N = Number(process.argv[2] ?? 16 * 1024 * 1024);
const REPS = Number(process.argv[3] ?? 5);
const INF = 0x7f800000;

let seed = 12345;
const rand = () => {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
};

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
const bits = (x) => {
  f32[0] = x;
  return u32[0];
};

function randomKeys(n) {
  const keys = new Uint32Array(n);
  for (let i = 0; i < n; ++i) {
    keys[i] = rand() < 0.05 ? INF : (rand() * 4294967296) >>> 0;
  }
  return keys;
}

// Radial depths of points in a 100m box, generated in x-sorted runs
// ("sorted-ish", like splat files); points behind the camera are inactive.
function realisticKeys(n) {
  const keys = new Uint32Array(n);
  const cam = [10, 1.6, 50];
  const dir = [0.8, 0, -0.6];
  for (let i = 0; i < n; ++i) {
    const x = (i / n) * 100 + (rand() - 0.5) * 2;
    const y = rand() * 10;
    const z = rand() * 100;
    const d = [x - cam[0], y - cam[1], z - cam[2]];
    const front = d[0] * dir[0] + d[1] * dir[1] + d[2] * dir[2];
    keys[i] = front < 0 ? INF : bits(Math.hypot(d[0], d[1], d[2]));
  }
  return keys;
}

function sortSingle(keys) {
  const readback = keys.slice();
  const ordering = new Uint32Array(keys.length);
  const t0 = performance.now();
  const active = sort32_splats(keys.length, readback, ordering);
  return { ms: performance.now() - t0, active, ordering };
}

function verify(keys, active, ordering, label) {
  let expected = 0;
  for (let i = 0; i < keys.length; ++i) if (keys[i] < INF) expected++;
  if (expected !== active) throw new Error(`${label}: active ${active} != ${expected}`);
  const seen = new Uint8Array(keys.length);
  for (let i = 0; i < active; ++i) {
    const idx = ordering[i];
    if (idx >= keys.length || seen[idx] || keys[idx] >= INF)
      throw new Error(`${label}: not a permutation of active at ${i}`);
    seen[idx] = 1;
    if (i > 0 && keys[ordering[i - 1]] < keys[idx])
      throw new Error(`${label}: not non-increasing at ${i}`);
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
    call(w, name, args, mode) {
      const id = nextId++;
      return new Promise((resolve) => {
        pending.set(id, resolve);
        workers[w].postMessage({ id, name, args, mode });
      });
    },
    terminate: () => Promise.all(workers.map((w) => w.terminate())),
  };
}

async function sortParallel(pool, keys, numWorkers, modes, out) {
  const t0 = performance.now();
  const buffers = ensureParallelSortBuffers(
    sortParallel.buffers,
    keys.length,
    numWorkers,
  );
  sortParallel.buffers = buffers;
  buffers.keys.set(keys); // readback -> SAB copy, as in SparkRenderer
  const phase = {};
  const active = await parallelSort32({
    numSplats: keys.length,
    buffers,
    call: async (w, name, args) => {
      const ms = await pool.call(w, name, args, modes[name]);
      phase[name] = Math.max(phase[name] ?? 0, ms);
      if (process.env.DIAG) console.log(name, w, args.end - args.start, ms.toFixed(1));
    },
  });
  out.set(buffers.ordering.subarray(0, active)); // SAB -> upload buffer copy
  return { ms: performance.now() - t0, active, phase };
}

const median = (xs) => xs.slice().sort((a, b) => a - b)[xs.length >> 1];
const MODES = {
  wasm: { sortParHist: "wasm", sortParScatter: "js", sortParRegion: "wasm" },
  jsHist: { sortParHist: "js", sortParScatter: "js", sortParRegion: "wasm" },
};

const pools = {};
for (const w of [2, 3, 4]) {
  pools[w] = makePool(w);
  await pools[w].ready;
}

const rows = [];
for (const [name, gen] of [
  ["random", randomKeys],
  ["realistic", realisticKeys],
]) {
  const keys = gen(N);
  const single = [];
  let ref;
  for (let r = 0; r < REPS; ++r) {
    ref = sortSingle(keys);
    single.push(ref.ms);
  }
  verify(keys, ref.active, ref.ordering, `${name} single`);
  const stMs = median(single);
  rows.push({ data: name, n: N, threads: 1, mode: "-", ms: stMs.toFixed(1), speedup: "1.00", correct: "PASS" });

  const out = new Uint32Array(N);
  for (const threads of [2, 3, 4]) {
    for (const [modeName, modes] of Object.entries(MODES)) {
      const times = [];
      let res;
      const phases = [];
      for (let r = 0; r < REPS; ++r) {
        out.fill(0xffffffff);
        res = await sortParallel(pools[threads], keys, threads, modes, out);
        times.push(res.ms);
        phases.push(res.phase);
      }
      let correct = "PASS";
      try {
        if (res.active !== ref.active) throw new Error("active mismatch");
        for (let i = 0; i < res.active; ++i) {
          if (keys[out[i]] !== keys[ref.ordering[i]]) throw new Error(`key seq mismatch at ${i}`);
        }
        verify(keys, res.active, out, `${name} ${threads} ${modeName}`);
      } catch (e) {
        correct = `FAIL ${e.message}`;
      }
      const ms = median(times);
      const ph = phases[phases.length >> 1];
      rows.push({
        data: name,
        n: N,
        threads,
        mode: modeName,
        ms: ms.toFixed(1),
        speedup: (stMs / ms).toFixed(2),
        correct,
        hist: ph.sortParHist.toFixed(1),
        scatter: ph.sortParScatter.toFixed(1),
        region: ph.sortParRegion.toFixed(1),
      });
    }
  }
}
console.table(rows);
await Promise.all(Object.values(pools).map((p) => p.terminate()));

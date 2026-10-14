// Sort worker for bench.mjs / matrix.mjs: same kernels as src/worker.ts
// (hist: wasm or JS) plus the incremental candidate.
import { readFileSync } from "node:fs";
import { parentPort } from "node:worker_threads";
import init, {
  sort32_par_hist,
  sort32_par_region,
  sort16_par_hist,
} from "../../rust/spark-rs/pkg/spark_rs.js";
import {
  sortPar16ScatterJs,
  sortParScatterJs,
} from "../../src/sortParallel.ts";
import { incrGather, incrNew } from "./incremental.mjs";

await init({
  module_or_path: readFileSync(
    new URL("../../rust/spark-rs/pkg/spark_rs_bg.wasm", import.meta.url),
  ),
});

const INF = 0x7f800000;

function histJs({ keys, start, end, hist }) {
  const h = new Uint32Array(65536);
  for (let i = start; i < end; ++i) {
    const key = keys[i];
    if (key < INF) h[~key >>> 16]++;
  }
  hist.set(h);
}

const wasm16Hist = ({ keys, keys16, hist }) =>
  sort16_par_hist(keys, keys16, hist);

const kernels = {
  sortParHist: {
    js: histJs,
    wasm: ({ keys, start, end, hist }) =>
      sort32_par_hist(keys.subarray(start, end), hist),
  },
  sortParScatter: { js: sortParScatterJs, wasm: sortParScatterJs },
  sortParRegion: {
    wasm: ({ pairs, ordering }) => sort32_par_region(pairs, ordering),
  },
  sortPar16Hist: { wasm: wasm16Hist, js: wasm16Hist },
  sortPar16Scatter: { js: sortPar16ScatterJs, wasm: sortPar16ScatterJs },
  // incremental: run/out are this worker's region of shared arrays
  incrGather: {
    js: ({ prev, keys16, run, out, seen, lens }) =>
      lens.set(incrGather(prev, keys16, run, out, seen)),
  },
  incrNew: {
    js: ({ keys16, seen, start, end, out, lens }) => {
      lens[1] = incrNew(keys16, seen, start, end, out, lens[1]);
    },
  },
};

parentPort.on("message", ({ id, name, args, mode }) => {
  const t0 = performance.now();
  kernels[name][mode ?? "js"](args);
  parentPort.postMessage({ id, ms: performance.now() - t0 });
});
parentPort.postMessage({ ready: true });

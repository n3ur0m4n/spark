/** Differential test and WASM-worker benchmark. No scene assets are required. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { arch, availableParallelism, cpus, platform } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  Worker,
  isMainThread,
  parentPort,
  workerData,
} from "node:worker_threads";

function half(x) {
  const f = new Float32Array([x]);
  const u = new Uint32Array(f.buffer)[0];
  const sign = (u >>> 16) & 0x8000;
  const exponent = ((u >>> 23) & 255) - 127 + 15;
  const mantissa = u & 0x7fffff;
  if (exponent <= 0) return sign;
  if (exponent >= 31) return sign | 0x7c00;
  // Fixtures use representable multiples; round once when encoding their tree.
  return sign | (exponent << 10) | ((mantissa + 0x1000) >>> 13);
}

function fixture(seed, count) {
  let rng = seed >>> 0;
  const next = () => {
    rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
    return rng;
  };
  const data = new Uint32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const xyz = Array.from({ length: 3 }, () => ((next() % 257) - 128) / 8);
    const size = (next() % 65) / 8;
    data[i * 4] = half(xyz[0]) | (half(xyz[1]) << 16);
    data[i * 4 + 1] = half(xyz[2]) | (half(size) << 16);
  }
  let first = 1;
  for (let i = 0; i < count && first < count; i++) {
    const children = Math.min(1 + (next() % 6), count - first);
    data[i * 4 + 2] = children;
    data[i * 4 + 3] = first;
    first += children;
  }
  return data;
}

function summarize(result, full) {
  const hash = createHash("sha256");
  const selections = result.instanceIndices.map(({ indices, numSplats }) => {
    assert.equal(indices.length % 16384, 0, "WASM row padding");
    const sorted = indices.slice(0, numSplats).sort();
    for (let i = 1; i < sorted.length; i++)
      assert.notEqual(sorted[i], sorted[i - 1], "duplicate selection");
    hash.update(
      Buffer.from(
        indices.subarray(0, numSplats).buffer,
        indices.byteOffset,
        numSplats * 4,
      ),
    );
    return full ? [...indices.subarray(0, numSplats)] : { count: numSplats };
  });
  return {
    selections,
    chunks: result.chunks,
    stats: {
      pixelLimit: result.pixelLimit,
      outputSize: result.outputSize,
      frontierSize: result.frontierSize,
      leafCount: result.leafCount,
    },
    selectionHash: hash.digest("hex"),
    selected: result.instanceIndices.reduce((n, x) => n + x.numSplats, 0),
  };
}

if (!isMainThread) {
  const pkg = resolve(workerData.pkg);
  const wasmPath = resolve(pkg, "spark_rs_bg.wasm");
  const bytes = await readFile(wasmPath);
  const wasm = await import(pathToFileURL(resolve(pkg, "spark_rs.js")).href);
  await wasm.default({ module_or_path: bytes });
  let lodId;
  parentPort.on("message", ({ id, request }) => {
    try {
      if (request.type === "load") {
        if (lodId !== undefined) wasm.dispose_lod_tree(lodId);
        const data = fixture(request.seed, request.nodes);
        lodId = wasm.init_lod_tree(request.nodes, data).lodId;
        parentPort.postMessage({
          id,
          result: {
            assetHash: createHash("sha256")
              .update(Buffer.from(data.buffer))
              .digest("hex"),
            wasmHash: createHash("sha256").update(bytes).digest("hex"),
          },
        });
        return;
      }
      const n = request.instances;
      const ids = new Uint32Array(n).fill(lodId);
      const roots = new Uint32Array(n).fill(0xffffffff);
      const views = new Float32Array(n * 16);
      for (let i = 0; i < n; i++) {
        const angle = (request.view ?? 0) * 0.17;
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        views.set(
          [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, i * 2, 0, 0, 1],
          i * 16,
        );
      }
      const ones = new Float32Array(n).fill(1);
      const start = performance.now();
      const result = wasm.traverse_lod_trees(
        request.budget,
        request.limit,
        undefined,
        ids,
        roots,
        views,
        ones,
        new Float32Array(n).fill(request.foveate ?? 1),
        new Float32Array(n).fill(request.foveate ?? 1),
        new Float32Array(n).fill(90),
        new Float32Array(n).fill(120),
      );
      const elapsedMs = performance.now() - start;
      parentPort.postMessage({
        id,
        result: { ...summarize(result, request.full), elapsedMs },
      });
    } catch (error) {
      parentPort.postMessage({ id, error: error.stack ?? String(error) });
    }
  });
} else {
  const args = process.argv.slice(2);
  const option = (key, fallback) => {
    const index = args.indexOf(key);
    return index === -1 ? fallback : args[index + 1];
  };
  const reference = option("--reference");
  if (!reference)
    throw new Error(
      "Usage: node scripts/compare-lod-traversal.mjs --reference /absolute/upstream/rust/spark-rs/pkg [--benchmark] [--output result.json]",
    );
  const candidate = option(
    "--candidate",
    fileURLToPath(new URL("../rust/spark-rs/pkg", import.meta.url)),
  );
  const workers = [reference, candidate].map(
    (pkg) => new Worker(new URL(import.meta.url), { workerData: { pkg } }),
  );
  let serial = 0;
  function rpc(worker, request) {
    return new Promise((resolvePromise, reject) => {
      const id = ++serial;
      const timeout = setTimeout(
        () => finish(new Error("WASM worker timed out")),
        120000,
      );
      const onError = (error) => finish(error);
      const onMessage = (message) => {
        if (message.id === id)
          finish(
            message.error ? new Error(message.error) : undefined,
            message.result,
          );
      };
      function finish(error, value) {
        clearTimeout(timeout);
        worker.off("error", onError);
        worker.off("message", onMessage);
        if (error) reject(error);
        else resolvePromise(value);
      }
      worker.on("error", onError);
      worker.on("message", onMessage);
      worker.postMessage({ id, request });
    });
  }
  const report = {
    kind: "SparkLodTraversalComparison",
    referenceCommit: option("--reference-commit") ?? null,
    runtime: {
      node: process.version,
      platform: platform(),
      arch: arch(),
      cpu: cpus()[0]?.model,
      parallelism: availableParallelism(),
    },
    testCases: 0,
    samples: [],
  };
  try {
    for (let seed = 0; seed < 16; seed++) {
      const loads = await Promise.all(
        workers.map((w) =>
          rpc(w, { type: "load", seed, nodes: seed % 3 === 0 ? 19 : 2049 }),
        ),
      );
      assert.equal(loads[0].assetHash, loads[1].assetHash);
      for (const budget of [0, 1, 3, 32, 256, 2048, 10000]) {
        for (const limit of [0, 0.03, 0.5, Number.POSITIVE_INFINITY]) {
          const request = {
            type: "traverse",
            instances: 1 + (seed % 3),
            budget,
            limit,
            view: seed,
            foveate: seed % 2 ? 0.2 : 1,
            full: true,
          };
          const [a, b] = await Promise.all(workers.map((w) => rpc(w, request)));
          assert.deepEqual(
            b.selections,
            a.selections,
            `selection: seed=${seed}, budget=${budget}, limit=${limit}`,
          );
          assert.deepEqual(b.chunks, a.chunks, "paging request order");
          assert.deepEqual(
            b.stats,
            a.stats,
            "all traversal result counters and pixel limit",
          );
          report.testCases++;
        }
      }
    }
    if (args.includes("--benchmark")) {
      report.fixture = {
        generator: "synthetic-lcg-integer-grid-v1",
        seed: 20260906,
        nodes: 4000001,
      };
      report.builds = [];
      for (const worker of workers)
        report.builds.push(
          await rpc(worker, { type: "load", ...report.fixture }),
        );
      assert.equal(report.builds[0].assetHash, report.builds[1].assetHash);
      for (const budget of [1500000, 2500000]) {
        const request = {
          type: "traverse",
          instances: 1,
          budget,
          limit: 0,
          full: false,
        };
        // Both workers retain the same upstream-style scratch state between runs.
        for (const worker of workers) await rpc(worker, request);
        for (let repetition = 1; repetition <= 5; repetition++) {
          const values = [];
          for (const index of repetition % 2 ? [0, 1] : [1, 0])
            values[index] = await rpc(workers[index], request);
          assert.equal(values[0].selectionHash, values[1].selectionHash);
          assert.deepEqual(values[0].chunks, values[1].chunks);
          const sample = {
            budget,
            repetition,
            selected: values[0].selected,
            upstreamMs: values[0].elapsedMs,
            optimizedMs: values[1].elapsedMs,
          };
          report.samples.push(sample);
          console.log(JSON.stringify(sample));
        }
      }
    }
    report.passed = true;
    if (option("--output"))
      await writeFile(
        option("--output"),
        `${JSON.stringify(report, null, 2)}\n`,
      );
    console.log(`Passed ${report.testCases} WASM-worker differential cases.`);
  } finally {
    await Promise.all(workers.map((w) => w.terminate()));
  }
}

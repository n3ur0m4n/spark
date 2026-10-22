import * as THREE from "three";
import { describe, expect, test } from "vitest";
import { parseLodMeta, selectLodMeta } from "../../src/LodMeta";

// 2 foglie, 2 livelli: file 0 = L0 (fine), file 1 = L1 (grossolano)
const json = {
  lodLevels: 2,
  filenames: ["0_0/meta.json", "1_0/meta.json"],
  tree: {
    bound: { min: [-10, -1, -1], max: [10, 1, 1] },
    children: [
      {
        bound: { min: [-10, -1, -1], max: [-8, 1, 1] },
        lods: {
          "0": { file: 0, offset: 0, count: 100 },
          "1": { file: 1, offset: 0, count: 10 },
        },
      },
      {
        // livello 1 mancante: deve riusare il livello 0
        bound: { min: [8, -1, -1], max: [10, 1, 1] },
        lods: { "0": { file: 0, offset: 100, count: 50 } },
      },
    ],
  },
};

const data = parseLodMeta(json, "https://x.invalid/scene/lod-meta.json");
const camera = new THREE.Matrix4(); // camera nell'origine, guarda -Z

function select(maxSplats: number, resident: number[]) {
  return selectLodMeta({
    instances: [{ splats: "s", data, viewToObject: camera, lodScale: 1 }],
    maxSplats,
    maxFiles: 100,
    residentPage: (_s, f) => (resident.includes(f) ? 10 + f : -1),
  });
}

describe("LodMeta", () => {
  test("parse: url risolti e livelli mancanti riempiti", () => {
    expect(data.numLeaves).toBe(2);
    expect(data.fileUrls[1]).toBe("https://x.invalid/scene/1_0/meta.json");
    // foglia 1 (ordine stack: ultima figlia prima) — cerca quella con count 50
    const leaf = data.count[0] === 50 ? 0 : 1;
    expect(data.file[leaf * 2 + 1]).toBe(0);
    expect(data.count[leaf * 2 + 1]).toBe(50);
  });

  test("budget ampio: tutto livello 0, indici pagina*65536+offset", () => {
    const r = select(1e6, [0, 1]);
    expect(r.numSplats[0]).toBe(150);
    const idx = Array.from(r.indices[0].subarray(0, 150)).sort((a, b) => a - b);
    expect(idx[0]).toBe(10 * 65536);
    expect(idx[149]).toBe(10 * 65536 + 149);
    expect(r.priority.map((p) => p.chunk)).toEqual([0]);
  });

  test("budget stretto: livello grossolano dove esiste", () => {
    const r = select(60, [0, 1]);
    expect(r.numSplats[0]).toBe(60); // 10 (L1) + 50 (unico livello)
  });

  test("non residente: fallback al livello grossolano residente, priorità al voluto", () => {
    const r = select(1e6, [1]);
    expect(r.numSplats[0]).toBe(10); // foglia 0 in fallback su L1, foglia 1 vuota
    expect(r.priority.map((p) => p.chunk)).toEqual([1, 0]);
  });

  test("range oltre una pagina: errore chiaro", () => {
    const bad = structuredClone(json);
    bad.tree.children[1].lods["0"].offset = 65500;
    expect(() => parseLodMeta(bad, "https://x.invalid/")).toThrow(/65536/);
  });
});

// Asset reale: LODMETA_JSON=<path lod-meta.json> npx vitest run test/unit/LodMeta.test.ts
test.skipIf(!process.env.LODMETA_JSON)(
  "asset reale: parse + selezione",
  async () => {
    const fs = await import("node:fs");
    const real = parseLodMeta(
      JSON.parse(fs.readFileSync(process.env.LODMETA_JSON as string, "utf8")),
      "https://x.invalid/lod-meta.json",
    );
    const run = () =>
      selectLodMeta({
        instances: [
          { splats: "s", data: real, viewToObject: camera, lodScale: 1 },
        ],
        maxSplats: 2500000,
        maxFiles: 192,
        residentPage: (_s, f) => f % 256, // tutto residente
      });
    run(); // riscaldamento JIT
    const t0 = performance.now();
    for (let i = 0; i < 10; i++) run();
    const r = run();
    const ms = (performance.now() - t0) / 11;
    console.log(
      `foglie ${real.numLeaves}, livelli ${real.numLevels}, file ${real.fileUrls.length}: ${r.numSplats[0]} splat, ${r.priority.length} file, ${ms.toFixed(1)} ms`,
    );
    expect(r.numSplats[0]).toBeLessThanOrEqual(2500000);
    expect(r.priority.length).toBeLessThanOrEqual(192);
  },
);

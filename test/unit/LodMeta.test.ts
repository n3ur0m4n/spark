import * as THREE from "three";
import { describe, expect, test } from "vitest";
import {
  type LodMetaData,
  lodMetaDistances,
  lodMetaFovScale,
  parseLodMeta,
  selectLodMeta,
} from "../../src/LodMeta";

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
        // livello 1 mancante: oltre il più grossolano = livello vuoto (come PC)
        bound: { min: [8, -1, -1], max: [10, 1, 1] },
        lods: { "0": { file: 0, offset: 100, count: 50 } },
      },
    ],
  },
};

const camera = new THREE.Matrix4(); // camera nell'origine, guarda -Z

// dati nuovi per test (selezione con stato: livello mostrato e bootstrap)
function load(j: Parameters<typeof parseLodMeta>[0] = json, bootstrap = false) {
  const d = parseLodMeta(j, "https://x.invalid/scene/lod-meta.json");
  if (!bootstrap) d.rangeMin = 0;
  return d;
}

function select(
  data: LodMetaData,
  maxSplats: number,
  resident: number[],
  viewToObject = camera,
) {
  return selectLodMeta({
    instances: [{ splats: "s", data, viewToObject, lodScale: 1 }],
    maxSplats,
    maxFiles: 100,
    residentPage: (_s, f) => (resident.includes(f) ? 10 + f : -1),
  });
}

// foglia con 50 splat a L0 = quella senza livello 1
const leafB = (d: LodMetaData) => (d.count[0] === 50 ? 0 : 1);

describe("LodMeta", () => {
  test("parse: url risolti, livello mancante = nessun dato", () => {
    const data = load();
    expect(data.numLeaves).toBe(2);
    expect(data.fileUrls[1]).toBe("https://x.invalid/scene/1_0/meta.json");
    const b = leafB(data);
    expect(data.file[b * 2 + 1]).toBe(-1);
    expect(data.count[b * 2 + 1]).toBe(0);
    expect(Array.from(data.fileLevel)).toEqual([0, 1]);
  });

  test("budget ampio: tutto livello 0, indici pagina*65536+offset", () => {
    const r = select(load(), 1e6, [0, 1]);
    expect(r.numSplats[0]).toBe(150);
    const idx = Array.from(r.indices[0].subarray(0, 150)).sort((a, b) => a - b);
    expect(idx[0]).toBe(10 * 65536);
    expect(idx[149]).toBe(10 * 65536 + 149);
    expect(r.priority.map((p) => p.chunk)).toEqual([0]);
  });

  test("a: budget stretto → foglia senza livello grossolano sparisce", () => {
    const r = select(load(), 60, [0, 1]);
    expect(r.numSplats[0]).toBe(10); // solo L1 della foglia A, B = livello vuoto
  });

  test("a: buco interno → livello più fine con dati", () => {
    const j = {
      lodLevels: 3,
      filenames: ["0", "1", "2"],
      tree: {
        bound: { min: [-1, -1, -11], max: [1, 1, -9] },
        lods: {
          "0": { file: 0, offset: 0, count: 40 },
          "2": { file: 2, offset: 0, count: 10 },
        },
      },
    };
    // budget 40: banda 1 (buco) → L0 (40) sta, banda 0 = idem → L0
    const d = load(j);
    expect(select(d, 40, [0, 2]).numSplats[0]).toBe(40);
    // budget 39: solo banda 2 → L2
    expect(select(load(j), 39, [0, 2]).numSplats[0]).toBe(10);
  });

  test("b: distanza × scala mondo del mesh, FOV, penalità dietro", () => {
    const data = load();
    const a = data.count[0] === 50 ? 1 : 0; // foglia A (x -10..-8)
    expect(lodMetaDistances(data, camera)[a]).toBeCloseTo(8);
    // mesh scalata ×2: viewToObject = M⁻¹·C
    const mesh = new THREE.Matrix4().makeScale(2, 2, 2);
    const v2o = mesh.clone().invert().multiply(camera);
    expect(lodMetaDistances(data, v2o)[a]).toBeCloseTo(8 * 2 * 0.5 * 2);
    // FOV 90° → tan 45 / tan 22.5
    expect(lodMetaFovScale(1, 1)).toBeCloseTo(1 / Math.tan(Math.PI / 8));
    // foglia dietro: camera guarda +X verso B, A è dietro → ×3
    const back = new THREE.Matrix4().makeRotationY(-Math.PI / 2);
    expect(lodMetaDistances(data, back)[a]).toBeCloseTo(8 * 3);
  });

  test("f: underfill mostra il residente fino a 5 livelli più grossolano", () => {
    const j = {
      lodLevels: 3,
      filenames: ["0", "1", "2"],
      tree: {
        bound: { min: [-1, -1, -11], max: [1, 1, -9] },
        lods: {
          "0": { file: 0, offset: 0, count: 40 },
          "1": { file: 1, offset: 0, count: 20 },
          "2": { file: 2, offset: 0, count: 10 },
        },
      },
    };
    const d = load(j);
    let r = select(d, 1e6, [2]);
    expect(r.numSplats[0]).toBe(10); // ottimo L0, mostrato L2 residente
    expect(r.prefetch.map((p) => p.chunk)).toEqual([1]); // passo successivo
    // niente residente: richiesto il più grossolano della finestra, niente a schermo
    const e = load(j);
    r = select(e, 1e6, []);
    expect(r.numSplats[0]).toBe(0);
    expect(r.priority.map((p) => p.chunk)).toEqual([2]);
    expect(r.prefetch.map((p) => p.chunk)).toEqual([1]);
    // il livello mostrato resta finché il nuovo non è residente
    r = select(e, 1e6, [2]);
    expect(r.numSplats[0]).toBe(10);
    r = select(e, 1e6, [0]); // L2 sparito: L0 residente → L0
    expect(r.numSplats[0]).toBe(40);
  });

  test("g: bootstrap al livello più grossolano, poi range completo", () => {
    const d = load(json, true);
    expect(d.rangeMin).toBe(1);
    let r = select(d, 1e6, []);
    expect(r.numSplats[0]).toBe(0);
    expect(r.priority.map((p) => p.chunk)).toEqual([1]); // solo L1
    expect(r.rangeChanged).toBe(false);
    r = select(d, 1e6, [1]);
    expect(r.numSplats[0]).toBe(10);
    expect(r.rangeChanged).toBe(true);
    expect(d.rangeMin).toBe(0);
    r = select(d, 1e6, [0, 1]);
    expect(r.numSplats[0]).toBe(150);
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
    real.rangeMin = 0;
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

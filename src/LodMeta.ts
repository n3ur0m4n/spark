import type * as THREE from "three";

// LoD "streamed SOG" di PlayCanvas (lod-meta.json, splat-transform):
// albero k-d di foglie, ogni foglia ha un range contiguo [offset, offset+count)
// in un file SOG per ciascun livello (0 = più dettagliato). 1 file = 1 chunk/pagina
// del SplatPager. Per ogni foglia si disegna UN livello (sostituzione, niente blending).

const PAGE_SPLATS = 65536;
const LOD_BASE = 5; // b nella formula PlayCanvas
const LN_LOD_MULT = Math.log(3); // m = 3
const BEHIND_PENALTY = 1.5;

export interface LodMetaData {
  /** URL assoluti di filenames[i] (meta.json della cartella SOG) */
  fileUrls: string[];
  numLeaves: number;
  numLevels: number;
  /** min xyz, max xyz per foglia (spazio PLY del file) */
  bounds: Float32Array;
  /** per foglia×livello: file (indice in fileUrls), offset, count.
   * I livelli mancanti sono già sostituiti col più vicino (prima grossolano). */
  file: Int32Array;
  offset: Uint32Array;
  count: Uint32Array;
}

type LodMetaNode = {
  bound: { min: number[]; max: number[] };
  children?: LodMetaNode[];
  lods?: Record<string, { file: number; offset: number; count: number }>;
};

export function isLodMetaUrl(url: string): boolean {
  return /lod-meta\.json$/i.test(url.split(/[?#]/)[0]);
}

export function parseLodMeta(
  json: {
    lodLevels: number;
    filenames: string[];
    tree: LodMetaNode;
  },
  baseUrl: string,
): LodMetaData {
  const numLevels = json.lodLevels;
  const fileUrls = json.filenames.map((f) => new URL(f, baseUrl).toString());

  const leaves: LodMetaNode[] = [];
  const stack = [json.tree];
  while (stack.length > 0) {
    const node = stack.pop() as LodMetaNode;
    if (node.children) {
      stack.push(...node.children);
    } else if (node.lods && Object.keys(node.lods).length > 0) {
      leaves.push(node);
    }
  }

  const numLeaves = leaves.length;
  const bounds = new Float32Array(numLeaves * 6);
  const file = new Int32Array(numLeaves * numLevels).fill(-1);
  const offset = new Uint32Array(numLeaves * numLevels);
  const count = new Uint32Array(numLeaves * numLevels);

  for (let i = 0; i < numLeaves; i++) {
    const { bound, lods } = leaves[i];
    bounds.set(bound.min, i * 6);
    bounds.set(bound.max, i * 6 + 3);
    for (const [level, lod] of Object.entries(lods ?? {})) {
      const l = Number(level);
      if (!(l >= 0 && l < numLevels)) continue;
      if (lod.offset + lod.count > PAGE_SPLATS) {
        throw new Error(
          `lod-meta: range ${lod.offset}+${lod.count} oltre ${PAGE_SPLATS} splat in "${json.filenames[lod.file]}" (1 file deve stare in una pagina)`,
        );
      }
      file[i * numLevels + l] = lod.file;
      offset[i * numLevels + l] = lod.offset;
      count[i * numLevels + l] = lod.count;
    }
    // Livello mancante: copia il più vicino, prima i più grossolani poi i più fini
    for (let l = 0; l < numLevels; l++) {
      const k = i * numLevels + l;
      if (file[k] >= 0) continue;
      let src = -1;
      for (let d = 1; d < numLevels && src < 0; d++) {
        if (l + d < numLevels && file[i * numLevels + l + d] >= 0) {
          src = l + d;
        } else if (l - d >= 0 && file[i * numLevels + l - d] >= 0) {
          src = l - d;
        }
      }
      const s = i * numLevels + src;
      file[k] = file[s];
      offset[k] = offset[s];
      count[k] = count[s];
    }
  }

  return { fileUrls, numLeaves, numLevels, bounds, file, offset, count };
}

export interface LodMetaInstance<S> {
  splats: S;
  data: LodMetaData;
  /** camera → spazio locale della mesh */
  viewToObject: THREE.Matrix4;
  lodScale: number;
}

export interface LodMetaResult<S> {
  /** indici splat da disegnare per istanza (page*65536 + offset + k) */
  numSplats: number[];
  indices: Uint32Array[];
  /** file da tenere/caricare, già ordinati per priorità */
  priority: { splats: S; chunk: number }[];
  /** prefetch del livello più fine (priorità più bassa) */
  prefetch: { splats: S; chunk: number }[];
}

/**
 * Selezione stile PlayCanvas: livello = floor(1 + log_3(d / (s·b))), d = distanza
 * camera → punto più vicino dell'AABB foglia (×1.5 se dietro la camera).
 * s globale = la più grande che sta nel budget splat (e pagine). Se il file del
 * livello voluto non è residente si mostra il residente più vicino (prima più
 * grossolano, poi più fine), altrimenti niente.
 * `residentPage(splats, file)` = pagina del file già caricata, o -1.
 */
export function selectLodMeta<S>({
  instances,
  maxSplats,
  maxFiles,
  residentPage,
  indicesCache,
}: {
  instances: LodMetaInstance<S>[];
  maxSplats: number;
  maxFiles: number;
  residentPage: (splats: S, file: number) => number;
  /** buffer indici riusabili per istanza (stesso ordine), può essere più corto */
  indicesCache?: (Uint32Array | undefined)[];
}): LodMetaResult<S> {
  // a[i] = 1 + log_3(d_i / (b·lodScale)) → livello = floor(a - c), c = log_3(s)
  const as = instances.map(({ data, viewToObject, lodScale }) => {
    const e = viewToObject.elements;
    const cx = e[12];
    const cy = e[13];
    const cz = e[14];
    // asse -Z della camera nello spazio locale
    const fx = -e[8];
    const fy = -e[9];
    const fz = -e[10];
    const bias = Math.log(LOD_BASE * lodScale);
    const a = new Float64Array(data.numLeaves);
    const b = data.bounds;
    for (let i = 0; i < data.numLeaves; i++) {
      const j = i * 6;
      const px = Math.min(Math.max(cx, b[j]), b[j + 3]);
      const py = Math.min(Math.max(cy, b[j + 1]), b[j + 4]);
      const pz = Math.min(Math.max(cz, b[j + 2]), b[j + 5]);
      const dx = px - cx;
      const dy = py - cy;
      const dz = pz - cz;
      let d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dx * fx + dy * fy + dz * fz < 0) d *= BEHIND_PENALTY;
      a[i] = 1 + (Math.log(Math.max(d, 1e-6)) - bias) / LN_LOD_MULT;
    }
    return a;
  });

  const stamps = instances.map(
    ({ data }) => new Int32Array(data.fileUrls.length),
  );
  let stamp = 0;
  const cost = (c: number) => {
    stamp++;
    let splats = 0;
    let files = 0;
    for (let m = 0; m < instances.length; m++) {
      const { data } = instances[m];
      const a = as[m];
      const st = stamps[m];
      const K = data.numLevels;
      for (let i = 0; i < data.numLeaves; i++) {
        const l = Math.min(Math.max(Math.floor(a[i] - c), 0), K - 1);
        const k = i * K + l;
        splats += data.count[k];
        const f = data.file[k];
        if (st[f] !== stamp) {
          st[f] = stamp;
          files++;
        }
      }
    }
    return { splats, files };
  };
  const fits = (c: number) => {
    const { splats, files } = cost(c);
    return splats <= maxSplats && files <= maxFiles;
  };

  // Ricerca binaria sulla c più grande (s più grande = più dettaglio) che sta nel budget
  let minA = Number.POSITIVE_INFINITY;
  let maxA = Number.NEGATIVE_INFINITY;
  let maxK = 1;
  for (let m = 0; m < instances.length; m++) {
    maxK = Math.max(maxK, instances[m].data.numLevels);
    for (const v of as[m]) {
      minA = Math.min(minA, v);
      maxA = Math.max(maxA, v);
    }
  }
  let c = maxA + 1; // tutto a livello 0
  if (instances.length > 0 && !fits(c)) {
    let lo = minA - maxK; // tutto al livello più grossolano
    let hi = c;
    if (fits(lo)) {
      for (let iter = 0; iter < 32; iter++) {
        const mid = 0.5 * (lo + hi);
        if (fits(mid)) lo = mid;
        else hi = mid;
      }
    }
    // ponytail: se nemmeno il più grossolano sta nel budget si mostra quello comunque
    c = lo;
  }

  const numSplats: number[] = [];
  const indices: Uint32Array[] = [];
  const entries: { splats: S; chunk: number; key: number }[] = [];
  const prefetchEntries: { splats: S; chunk: number; key: number }[] = [];

  for (let m = 0; m < instances.length; m++) {
    const { splats, data } = instances[m];
    const a = as[m];
    const K = data.numLevels;
    const numFiles = data.fileUrls.length;
    const pageOf = new Int32Array(numFiles).fill(-2);
    const getPage = (f: number) => {
      if (pageOf[f] === -2) pageOf[f] = residentPage(splats, f);
      return pageOf[f];
    };
    const keys = new Float64Array(numFiles).fill(Number.POSITIVE_INFINITY);
    const prefetchKeys = new Float64Array(numFiles).fill(
      Number.POSITIVE_INFINITY,
    );
    // categorie: 0 = mostrato, 1 = voluto per foglia vuota, 2 = voluto per foglia
    // in attesa cambio; a parità prima i grossolani, poi i vicini
    const setKey = (
      arr: Float64Array,
      f: number,
      cat: number,
      l: number,
      i: number,
    ) => {
      const key = (cat * 64 + (63 - l)) * 1e9 + Math.min(a[i] + 64, 1e9 - 1);
      if (key < arr[f]) arr[f] = key;
    };

    const drawnK = new Int32Array(data.numLeaves).fill(-1);
    let total = 0;
    for (let i = 0; i < data.numLeaves; i++) {
      const want = Math.min(Math.max(Math.floor(a[i] - c), 0), K - 1);
      let drawn = -1;
      if (getPage(data.file[i * K + want]) >= 0) {
        drawn = want;
      } else {
        for (let l = want + 1; l < K && drawn < 0; l++) {
          if (getPage(data.file[i * K + l]) >= 0) drawn = l;
        }
        for (let l = want - 1; l >= 0 && drawn < 0; l--) {
          if (getPage(data.file[i * K + l]) >= 0) drawn = l;
        }
      }
      if (drawn >= 0) {
        const k = i * K + drawn;
        drawnK[i] = k;
        total += data.count[k];
        setKey(keys, data.file[k], 0, drawn, i);
      }
      if (drawn !== want) {
        setKey(keys, data.file[i * K + want], drawn < 0 ? 1 : 2, want, i);
      } else if (want > 0) {
        setKey(prefetchKeys, data.file[i * K + want - 1], 3, want - 1, i);
      }
    }

    const capacity = Math.max(1, Math.ceil(total / 16384)) * 16384;
    let out = indicesCache?.[m];
    if (!out || out.length < capacity) {
      out = new Uint32Array(Math.ceil((capacity * 1.25) / 16384) * 16384);
    }
    let n = 0;
    for (let i = 0; i < data.numLeaves; i++) {
      const k = drawnK[i];
      if (k < 0) continue;
      const base = getPage(data.file[k]) * PAGE_SPLATS + data.offset[k];
      const cnt = data.count[k];
      for (let j = 0; j < cnt; j++) out[n + j] = base + j;
      n += cnt;
    }
    numSplats.push(n);
    indices.push(out);

    for (let f = 0; f < numFiles; f++) {
      if (keys[f] !== Number.POSITIVE_INFINITY) {
        entries.push({ splats, chunk: f, key: keys[f] });
      } else if (prefetchKeys[f] !== Number.POSITIVE_INFINITY) {
        prefetchEntries.push({ splats, chunk: f, key: prefetchKeys[f] });
      }
    }
  }

  entries.sort((x, y) => x.key - y.key);
  prefetchEntries.sort((x, y) => x.key - y.key);
  const strip = ({ splats, chunk }: { splats: S; chunk: number }) => ({
    splats,
    chunk,
  });
  return {
    numSplats,
    indices,
    priority: entries.map(strip),
    prefetch: prefetchEntries.map(strip),
  };
}

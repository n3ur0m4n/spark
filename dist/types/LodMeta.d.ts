import type * as THREE from "three";
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
    bound: {
        min: number[];
        max: number[];
    };
    children?: LodMetaNode[];
    lods?: Record<string, {
        file: number;
        offset: number;
        count: number;
    }>;
};
export declare function isLodMetaUrl(url: string): boolean;
export declare function parseLodMeta(json: {
    lodLevels: number;
    filenames: string[];
    tree: LodMetaNode;
}, baseUrl: string): LodMetaData;
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
    priority: {
        splats: S;
        chunk: number;
    }[];
    /** prefetch del livello più fine (priorità più bassa) */
    prefetch: {
        splats: S;
        chunk: number;
    }[];
}
/**
 * Selezione stile PlayCanvas: livello = floor(1 + log_3(d / (s·b))), d = distanza
 * camera → punto più vicino dell'AABB foglia (×1.5 se dietro la camera).
 * s globale = la più grande che sta nel budget splat (e pagine). Se il file del
 * livello voluto non è residente si mostra il residente più vicino (prima più
 * grossolano, poi più fine), altrimenti niente.
 * `residentPage(splats, file)` = pagina del file già caricata, o -1.
 */
export declare function selectLodMeta<S>({ instances, maxSplats, maxFiles, residentPage, indicesCache, }: {
    instances: LodMetaInstance<S>[];
    maxSplats: number;
    maxFiles: number;
    residentPage: (splats: S, file: number) => number;
    /** buffer indici riusabili per istanza (stesso ordine), può essere più corto */
    indicesCache?: (Uint32Array | undefined)[];
}): LodMetaResult<S>;
export {};

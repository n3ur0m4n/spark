/** Immutable CPU description of one ordinary mesh in an accumulator generation. */
export type NativeMeshGeneration = {
    source: object;
    revision: number;
    primary: Uint32Array;
    secondary?: Uint32Array;
    sh: (Uint32Array | undefined)[];
    sourceCount: number;
    sourceExt: boolean;
    base: number;
    count: number;
    indices?: Uint32Array;
    rotation: number[];
    translationScale: number[];
    localViewOrigin: number[];
    encoding: number[];
    shMaxOpacity: number[];
    numSh: number;
    recolor: number[];
};
/** Source uploads are shared across instances; output ownership stays with the backend slot. */
export declare class WebGPUGeneration {
    private readonly device;
    private readonly sources;
    private readonly pipeline;
    private readonly dummy;
    private readonly meshes;
    private disposed;
    constructor(device: GPUDevice);
    private buffer;
    private destroySource;
    private uploadSource;
    encode(encoder: GPUCommandEncoder, args: {
        meshes: NativeMeshGeneration[];
        output: GPUBuffer;
        output2: GPUBuffer;
        depths: GPUBuffer;
        outputExt: boolean;
        viewOrigin: number[];
        viewDirection: number[];
        sortRadial: boolean;
        timestampWrites?: () => GPUComputePassTimestampWrites | undefined;
    }): void;
    /** Release source ownership when the committed scene becomes empty. */
    clearSources(): void;
    dispose(): void;
}

import type { Camera } from "three";
import type { SplatAccumulator } from "./SplatAccumulator";
type Request = Readonly<{
    requestId: number;
    lifecycleEpoch: number;
    mappingVersion: number;
    accumulatorVersion: number;
    requestedAtMs: number;
    generatedAtMs: number | null;
    viewOrigin: number[];
    viewDirection: number[];
    sortRadial: boolean;
}>;
export declare class OrderingTrace {
    private clock;
    enabled: boolean;
    dropped: number;
    private epoch;
    private serial;
    private generated;
    private order;
    private events;
    private context;
    constructor(clock?: () => number);
    enable(context: HTMLCanvasElement): void;
    private detach;
    private reset;
    dispose(): void;
    private record;
    drain(): {
        events: Record<string, unknown>[];
        dropped: number;
    };
    generatedAt(accumulator: SplatAccumulator): void;
    request(accumulator: SplatAccumulator, sortRadial: boolean): Request | null;
    complete(request: Request | null): number | null;
    commit(request: Request | null, completedAtMs: number | null, activeSplats: number): void;
    draw(mappingVersion: number, camera: Camera, drawnSplats: number): void;
}
export {};

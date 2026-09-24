// Internal, opt-in diagnostics. Not exported from the package entry point.
import type { Camera } from "three";
import { Vector3 } from "three";
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
type Commit = {
  request: Request;
  completedAtMs: number;
  committedAtMs: number;
  activeSplats: number;
  firstDraw: boolean;
};
export class OrderingTrace {
  enabled = false;
  dropped = 0;
  private epoch = 0;
  private serial = 0;
  private generated = new WeakMap<object, number>();
  private order: Commit | null = null;
  private events: Record<string, unknown>[] = [];
  private context: HTMLCanvasElement | null = null;
  constructor(private clock = () => performance.now()) {}
  enable(context: HTMLCanvasElement) {
    this.detach();
    this.enabled = true;
    this.events = [];
    this.dropped = 0;
    this.reset();
    this.context = context;
    context.addEventListener("webglcontextlost", this.reset);
    context.addEventListener("webglcontextrestored", this.reset);
  }
  private detach() {
    this.context?.removeEventListener("webglcontextlost", this.reset);
    this.context?.removeEventListener("webglcontextrestored", this.reset);
    this.context = null;
  }
  private reset = () => {
    this.epoch++;
    this.order = null;
    this.generated = new WeakMap();
    this.record("epoch", { lifecycleEpoch: this.epoch });
  };
  dispose() {
    this.reset();
    this.detach();
    this.enabled = false;
  }
  private record(stage: string, data: Record<string, unknown>) {
    if (!this.enabled) return;
    if (this.events.length >= 8192) {
      this.dropped++;
      return;
    }
    this.events.push({ stage, atMs: this.clock(), ...data });
  }
  drain() {
    const events = this.events;
    this.events = [];
    return { events, dropped: this.dropped };
  }
  generatedAt(accumulator: SplatAccumulator) {
    if (this.enabled) this.generated.set(accumulator, this.clock());
  }
  request(accumulator: SplatAccumulator, sortRadial: boolean): Request | null {
    if (!this.enabled) return null;
    const request = {
      requestId: ++this.serial,
      lifecycleEpoch: this.epoch,
      mappingVersion: accumulator.mappingVersion,
      accumulatorVersion: accumulator.version,
      requestedAtMs: this.clock(),
      generatedAtMs: this.generated.get(accumulator) ?? null,
      viewOrigin: accumulator.viewOrigin.toArray(),
      viewDirection: accumulator.viewDirection.toArray(),
      sortRadial,
    };
    this.record("requested", request);
    return request;
  }
  complete(request: Request | null) {
    if (!request) return null;
    const at = this.clock();
    this.record("completed", { requestId: request.requestId });
    return at;
  }
  commit(
    request: Request | null,
    completedAtMs: number | null,
    activeSplats: number,
  ) {
    if (!request || completedAtMs === null) return;
    if (request.lifecycleEpoch !== this.epoch || !this.enabled) {
      this.record("obsolete-result", { requestId: request.requestId });
      return;
    }
    this.order = {
      request,
      completedAtMs,
      committedAtMs: this.clock(),
      activeSplats,
      firstDraw: true,
    };
    this.record("committed", { ...this.order, firstDraw: undefined });
  }
  draw(mappingVersion: number, camera: Camera, drawnSplats: number) {
    if (!this.enabled) return;
    const order = this.order;
    if (!order) {
      this.record("drawn", { attribution: "unavailable", drawnSplats });
      return;
    }
    const origin = camera.getWorldPosition(new Vector3());
    const direction = camera.getWorldDirection(new Vector3());
    const now = this.clock();
    const request = order.request;
    const dot = direction.dot(new Vector3().fromArray(request.viewDirection));
    this.record("drawn", {
      requestId: request.requestId,
      lifecycleEpoch: this.epoch,
      firstDraw: order.firstDraw,
      mappingCompatible: mappingVersion === request.mappingVersion,
      requestAgeMs: now - request.requestedAtMs,
      generationAgeMs:
        request.generatedAtMs === null ? null : now - request.generatedAtMs,
      viewDisplacement: origin.distanceTo(
        new Vector3().fromArray(request.viewOrigin),
      ),
      directionRadians: request.sortRadial
        ? null
        : Math.acos(Math.min(1, Math.max(-1, dot))),
      drawnSplats,
    });
    order.firstDraw = false;
  }
}

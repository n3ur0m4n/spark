import { PerspectiveCamera, Vector3 } from "three";
import { expect, test } from "vitest";
import { OrderingTrace } from "../../src/OrderingTrace";
import type { SplatAccumulator } from "../../src/SplatAccumulator";

test("draw attribution preserves original order age and rejects a previous lifecycle", () => {
  let now = 10;
  const trace = new OrderingTrace(() => now);
  const context = new EventTarget() as HTMLCanvasElement;
  const accumulator = {
    mappingVersion: 1,
    version: 7,
    viewOrigin: new Vector3(),
    viewDirection: new Vector3(0, 0, -1),
  } as SplatAccumulator;
  expect(trace.request(accumulator, true)).toBeNull();
  trace.enable(context);
  trace.generatedAt(accumulator);
  now = 20;
  const request = trace.request(accumulator, true);
  accumulator.version = 8;
  accumulator.viewOrigin.x = 9;
  now = 30;
  const completed = trace.complete(request);
  trace.commit(request, completed, 3);
  const camera = new PerspectiveCamera();
  camera.updateMatrixWorld();
  now = 40;
  trace.draw(1, camera, 3);
  now = 50;
  trace.draw(1, camera, 3);
  const draws = trace.drain().events.filter((e) => e.stage === "drawn");
  expect(draws.map((d) => d.requestAgeMs)).toEqual([20, 30]);
  expect(draws.map((d) => d.firstDraw)).toEqual([true, false]);
  expect(draws[0].viewDisplacement).toBe(0);
  context.dispatchEvent(new Event("webglcontextlost"));
  trace.commit(request, completed, 3);
  trace.draw(1, camera, 3);
  const events = trace.drain().events;
  expect(events.some((e) => e.stage === "obsolete-result")).toBe(true);
  expect(events.at(-1)?.attribution).toBe("unavailable");
  trace.dispose();
  context.dispatchEvent(new Event("webglcontextrestored"));
  expect(trace.enabled).toBe(false);
});

test("bounded diagnostics explicitly report truncation", () => {
  const trace = new OrderingTrace(() => 1);
  trace.enable(new EventTarget() as HTMLCanvasElement);
  const camera = new PerspectiveCamera();
  for (let i = 0; i < 9000; i++) trace.draw(0, camera, 0);
  const result = trace.drain();
  expect(result.events).toHaveLength(8192);
  expect(result.dropped).toBeGreaterThan(0);
  expect(trace.drain().events).toEqual([]);
});

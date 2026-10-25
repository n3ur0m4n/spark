import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

class FakeGPUDevice {
  limits = {};
}

const webgpu = vi.hoisted(() => ({ native: true, initError: false }));

vi.mock("three", async (importOriginal) => ({
  ...(await importOriginal<typeof import("three")>()),
  WebGLRenderer: class {
    readonly isWebGLRenderer = true;
    constructor(readonly params: unknown) {}
  },
}));

vi.mock("three/webgpu", () => ({
  WebGPURenderer: class {
    backend: { device?: unknown } = {};
    disposed = false;
    constructor(readonly params: unknown) {}
    async init() {
      if (webgpu.initError) throw new Error("init failed");
      if (webgpu.native) this.backend.device = new FakeGPUDevice();
    }
    dispose() {
      this.disposed = true;
    }
  },
}));

const { createRenderer, WEBGPU_UNSUPPORTED_FEATURES } = await import(
  "../../src/createRenderer"
);

function stubGpu(adapter: object | null) {
  const requestAdapter = vi.fn(async () => adapter);
  vi.stubGlobal("navigator", { gpu: { requestAdapter } });
  return requestAdapter;
}

describe("createRenderer", () => {
  beforeEach(() => {
    vi.stubGlobal("GPUDevice", FakeGPUDevice);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    webgpu.native = true;
    webgpu.initError = false;
  });

  test("chooses WebGPU when an adapter is available", async () => {
    stubGpu({});
    const { renderer, backend } = await createRenderer({ antialias: false });
    expect(backend).toBe("webgpu");
    expect(renderer).toMatchObject({ params: { antialias: false } });
  });

  test("falls back to WebGL without navigator.gpu", async () => {
    vi.stubGlobal("navigator", {});
    const { renderer, backend } = await createRenderer({ antialias: false });
    expect(backend).toBe("webgl");
    expect(renderer).toMatchObject({
      isWebGLRenderer: true,
      params: { antialias: false },
    });
  });

  test("falls back to WebGL when the adapter is null", async () => {
    stubGpu(null);
    expect((await createRenderer()).backend).toBe("webgl");
  });

  test("falls back to WebGL when a need is unsupported", async () => {
    const requestAdapter = stubGpu({});
    expect(WEBGPU_UNSUPPORTED_FEATURES).toContain("xr");
    const { backend } = await createRenderer({ needs: ["xr"] });
    expect(backend).toBe("webgl");
    expect(requestAdapter).not.toHaveBeenCalled();
  });

  test("keeps WebGPU for supported needs", async () => {
    stubGpu({});
    expect(WEBGPU_UNSUPPORTED_FEATURES).not.toContain("paging");
    expect(WEBGPU_UNSUPPORTED_FEATURES).not.toContain("editing");
    expect(WEBGPU_UNSUPPORTED_FEATURES).not.toContain("sortCull");
    expect(WEBGPU_UNSUPPORTED_FEATURES).not.toContain("stochastic");
    const { backend } = await createRenderer({
      needs: [
        "multipleSplatMeshes",
        "paging",
        "editing",
        "sortCull",
        "stochastic",
      ],
    });
    expect(backend).toBe("webgpu");
  });

  test("forceWebGL skips WebGPU", async () => {
    const requestAdapter = stubGpu({});
    expect((await createRenderer({ forceWebGL: true })).backend).toBe("webgl");
    expect(requestAdapter).not.toHaveBeenCalled();
  });

  test("falls back when WebGPURenderer lands on its WebGL2 backend", async () => {
    stubGpu({});
    webgpu.native = false;
    expect((await createRenderer()).backend).toBe("webgl");
  });

  test("falls back when init() rejects", async () => {
    stubGpu({});
    webgpu.initError = true;
    expect((await createRenderer()).backend).toBe("webgl");
  });
});

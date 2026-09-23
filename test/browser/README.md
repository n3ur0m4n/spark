# ExtSplats source texture regression

After building WASM and installing normal dependencies, run Vite on an unused local port:

```sh
npx vite --host 127.0.0.1 --port 5322 --strictPort
```

Install Playwright separately, or set `SPARK_PLAYWRIGHT_MODULE` to an existing module. With Chrome installed:

```sh
node scripts/test-ext-source-textures.mjs http://127.0.0.1:5322/test/browser/ext-source-textures.html /tmp/ext-texture-result
```

The test renders through the actual WebGL generator, reads the generated depth attachment, and compares both GPU source textures against their Uint32 input words. It covers constructed sources, in-place mutation, unchanged generations, nonzero byte offsets, growth and retirement, raw-array edits, and reinitialization. GPU source words are compared exactly; the depth check uses a small numeric tolerance and makes no sorting-equivalence claim.

Direct source edits require `extSplats.needsUpdate = true`. As with other source changes, mark `splatMesh.needsUpdate = true` to request a new accumulator generation. `setSplat`, `pushSplat`, and storage growth mark the source textures automatically.

# Ordering buffer browser check

This optional check constructs an original 1,024-splat fixture, renders changing camera poses, and reads the actual integer ordering texture before and after forced context loss/restoration. It verifies that committed CPU data matches GPU data, that current ordering survives restoration, and that repeated sorts need only two ordering allocations after warmup.

From the repository root, after installing dependencies and building WASM:

```sh
npm install --no-save --package-lock=false --ignore-scripts playwright@1.62.1
npx vite --host 127.0.0.1 --port 5311 --strictPort
```

With Google Chrome installed, run in another terminal:

```sh
node scripts/test-ordering-buffer-browser.mjs
```

The test uses Chrome's real GPU. It does not substitute software rendering. An unavailable GPU or failed readback fails the test. `SPARK_PLAYWRIGHT_MODULE` may point to an existing Playwright module. A different page URL can be passed as the first argument.

On Apple M4 Max / Chrome 152.0.7977.77 / Three.js r180, the candidate used two allocations and 19 reuses in the fixture, with identical pre-loss rendered pixels to upstream. Upstream's CPU ordering data differed from the actual GPU texture after camera changes; the candidate's data matched and restored exactly.

Full-scene restoration still loses accumulator textures on both untouched upstream and the candidate. That separate renderer issue produces an integer sampler/texture-format warning and is not claimed fixed here. This check deliberately reads the ordering texture directly to identify which resource was restored. This is correctness/allocation evidence, not an FPS benchmark.

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

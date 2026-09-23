# Optional native GPU radix sorting

This dependent change adds `webgpuSort: "radix"` on top of the native reference
integration. WebGL and the default native generated-depth/readback sorter retain
their existing behavior. No worker-center approximation, traversal scheduling
policy, reduced-quality default or projector is included.

The GPU sorter consumes the exact same generated depth words as the reference
WASM sorter. It sorts stably, keeps zero-opacity entries, excludes invalid depth
entries, and writes ordering plus indirect draw arguments into the generation's
working slot. The draw mode, generated data, ordering and count commit together.
An incomplete or stale generation cannot become active.

`getRenderStats()` returns `drawnSplats: null` when a radix count has not been read.
Call `readRenderStatsAsync()` to measure the current committed indirect count.
The result identifies the generation captured when called, even if a later
render reuses its slot before the promise settles. Late measurements cannot
replace current statistics. This explicit readback is outside the ordinary
render loop. The legacy `activeSplats` field is an upper bound until that
particular generation is measured; do not label it an actual draw count.

The portable four-bit kernels are adapted from PlayCanvas and its cited
kishimisu/WebGPU-Radix-Sort lineage. Source headers and packaged `NOTICE` preserve
attribution and license notices. The TypeScript dispatcher owns GPU resources
on the host Three.js device.

`npm run test:browser` adds actual-GPU radix tests at dispatch boundaries and up
to 2.5M elements, duplicate keys, partial tails and growth/shrinkage. It also adds
count-ownership tests with invalid entries, delayed readbacks, slot reuse and
disposal. `npm run test:wgsl` validates all eight composed shaders. Existing CI
and minimum/current Three package checks remain in the reference stack.

The combined precursor's real-GPU tests passed on Apple Metal / Chrome 152,
including a private huge-scene comparison against unmodified official SparkJS.
This split requires its own frozen-revision correctness and performance runs.
The inherited cross-API image exceptions remain open; one diagnostic timing run
per backend is not evidence for a public speedup claim. Five rotated repetitions,
matched quality/composition, instrumentation overhead and public licensed or
synthetic evidence are still required before publishing benchmarks or video.

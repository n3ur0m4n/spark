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

`npm run test:native:browser` adds actual-GPU radix tests at dispatch boundaries and up
to 2.5M elements, duplicate keys, partial tails and growth/shrinkage. It also adds
count-ownership tests with invalid entries, delayed readbacks, slot reuse and
disposal. `npm run test:wgsl` validates all eight composed shaders. Existing CI
and minimum/current Three package checks remain in the reference stack.

Prefix scratch is allocated for retained capacity. Active prefix levels, counts
and dispatches update without replacing buffers as the active count fluctuates.
The real-GPU test asserts zero scratch buffer allocations after a 2.5M warmup
while crossing dispatch and hierarchy boundaries. Generation clears only its
consumed texture-copy extent and initializes its logical depth span, including
mapping holes. Growth to 2.5M followed by small/empty generations is tested in
both sort modes; retained allocation is distinct from bytes cleared.

These are allocation/work-reduction changes, not a claim of higher steady-budget
FPS. Key fusion and submission consolidation remain excluded: the available
phase measurements did not justify their additional complexity. Timestamp
queries remain optional and unavailable samples are not zero-duration work.

This dependent branch on upstream `d7e7f8c` passes types, lint, unit and WGSL
checks, production/development builds, the real-GPU source browser suite and
CommonJS/ESM/TypeScript consumers on Three r180/r186. Hardware checks use
Apple M4 Max / Chrome 153. Packaged development/minified browser checks also pass on Three r180/r186. Other GPU vendors and remote CI
remain pending. Cross-API contour exceptions remain documented in the reference
integration; the radix and reference sort paths must agree exactly on identical
inputs. Public performance claims require a cleared or synthetic workload,
matched quality and complete run-level evidence.

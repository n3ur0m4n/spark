# Native WebGPU reference integration

This candidate adds an optional native backend on official SparkJS 2.2.0. The
existing WebGL constructor, synchronous rendering, LoD traversal, scheduling and
Rust code remain available. GPU radix sorting is a separate dependent change;
it is not included in this reference implementation. Experimental opacity-based
quad trimming, projectors and approximate traversal are excluded.

## Use the initialized host renderer

```ts
import { WebGPURenderer } from "three/webgpu";
import { SparkRenderer } from "@sparkjsdev/spark";

const renderer = new WebGPURenderer({ antialias: false });
await renderer.init();
const spark = new SparkRenderer({ renderer });
scene.add(spark);
await spark.renderAsync(scene, camera);
```

Spark uses the host's device. Each asynchronous render retains a camera snapshot
and resolves after generation, sorting and draw submission. It does not promise
GPU completion. Classic `THREE.WebGLRenderer` still supports synchronous
`spark.render(scene, camera): void` and retains its existing asynchronous update
and sorting behavior.

`getRenderStats()` returns counts and an identity for the committed generation.
`selectedSplats` counts generation input, while `drawnSplats` counts submitted
instances after invalid depths are excluded. Neither counts visible fragments.
The reference implementation knows the actual count from its existing WASM
sorter; `readRenderStatsAsync()` returns the same generation's counts.

## Runnable example

After building, open `/examples/native-webgpu/` on the development server. It uses
an original procedural 8,192-splat scene, an opaque Three.js occluder, orbit
controls and explicit Native WebGPU/Classic WebGL routes. It downloads no assets.
Camera and resize updates wait for the preceding submission. It demonstrates
integration; it is not a timing benchmark.

## Rendering and feature coverage

Generation supports ordinary PackedSplats/ExtSplats, source SH0–3, per-instance
positive uniform transforms, recolor/opacity, multiple instances, source
revisions and selected/full sources. The reference sorter consumes GPU-generated
world-space depth before output quantization. Sources can share uploaded storage;
each instance retains its own transforms and indices. Failed uploads cannot
commit an incomplete generation. Empty generations are valid and release
uploaded source allocations while retaining reusable output capacity.

Native rendering shares linear color and opaque depth with Three.js, then applies
one final color conversion. Ordinary upstream WebGL can blend directly in its
canvas output path, so default screenshots need not have identical colors.
Cross-API image qualification uses a separately identified external linear
WebGL composition control; ordinary upstream timings remain a distinct baseline. It supports resize (including during an awaited update), alpha, tone mapping, perspective
and orthographic cameras. Spark removes its device listeners/resources on dispose
and preserves the host device. Applications recreate the host renderer after
device loss. A shared device-loss observer has removable subscriptions; disposed
Spark instances can be collected while the host device remains alive.

Call `getSparkRendererCapabilities(renderer)` before choosing the route. Native
status remains `reference-preview` while the submission gates below are open.
Paging, XR, array cameras, modifiers, editing, skinning, covariance splats, custom generators,
GPU-generated sources, nonuniform/mirrored/sheared transforms, transparent or
transmissive host objects (including sprites), logarithmic or reversed depth,
MSAA, specialized targets, custom viewports, scissor
tests and manual clearing require classic
`THREE.WebGLRenderer`. Unsupported native combinations throw explicit errors;
they do not silently draw a partial scene. Three's `forceWebGL` case is rejected
explicitly and this candidate does not claim to resolve upstream issue #394.

Source and output buffers must fit the actual host device limits. Lowering the
selected LoD budget does not reduce full-source storage. Applications can request
higher supported limits when creating their Three.js renderer.

## Reproduction and evidence

Build WASM and install dependencies, then run:

```sh
npx tsc --noEmit
npm test
npm run test:wgsl
npm run build
npm run test:package
npm run test:native:browser
npm run test:package:browser
```

Shader validation requires Naga; a missing validator fails the check. Browser
tests require installed Chrome with a hardware WebGPU adapter. They start an
isolated Vite server on an available port, save raw results under ignored
`test-results/native/`, and close their own server/browser. Use
`SPARK_BROWSER_CHANNEL` or `SPARK_BROWSER_RESULTS` to change those settings.
Software GPU results are not hardware evidence.

The contribution's deterministic/seeded browser fixture can also run against a
separate pristine upstream checkout. It compares actual generated depth words,
ordering and counts, including SH, formats, invalid scales, zero opacity, moved
transforms and projection types. Package consumers exercise ESM, CommonJS and
TypeScript on Three r180/r186. The package uses the consumer's Three installation. The packed browser check
serves the npm archive through a plain HTTP server with the consumer's Three
installation, exercising both development and production bundles on r180/r186.
It runs the native lifecycle, forced-GC retention, counts, device, feature, host-mode
and composition fixtures;
no source alias can substitute checkout code. Raw results are retained under
`test-results/package-browser/`. It requires a real hardware WebGPU adapter.

The deterministic fixtures use generated or repository-provided test data. No
private diagnostic scene or footage is included. Generated attributes, depth
words, ordering and counts are checked separately from final images.

## Open submission gates

- The older 96-case and 24-case oracles retain ten single-pixel cutoff
  differences. GPU probes locate all ten within 0.000618 pixels of the ideal
  Gaussian contour. A diagnostic wider support isolates the cutoff, and all 48
  fresh seeded holdout cases pass the strict <=1-channel image comparison and
  exact depth/order/counts. Older strict failures remain explicit; other hardware
  must validate the boundary behavior independently.
- This independent reference branch based on upstream `d7e7f8c` passes
  types, lint, unit tests, WGSL validation, production/development builds, the
  source browser suite and CommonJS/ESM/TypeScript package consumers on Three
  r180/r186. Hardware checks use Chrome 153 / Apple M4 Max. Packaged development/minified browser
  checks also pass on both Three versions. Remote CI and other GPU/platform
  results remain pending; local checks do not establish them.
- Performance claims require matched quality, camera route, physical resolution,
  source/WASM identities and recording mode, with paired run-level results.
  Diagnostic phase timings and submission intervals are distinct from presentation
  FPS. This reference integration does not claim a public rendering speedup.
- Final qualification must retain the WebGL disposal regression check. The
  candidate now cancels pending scheduling, snapshots active accumulator ownership
  before clearing its map, and guards late callbacks. Source and packaged browser
  tests cover disposal during pending work; the earlier `No target` reproduction
  remains negative evidence for the unpatched upstream path.

Public figures, videos and posts require reproducible licensed or explicitly
synthetic evidence from the final contribution revisions.

Host-mode fixtures reproduce transparent-sprite overlap and logarithmic-depth
occlusion failures, reject unsupported native use before generation, recover to
supported native state, and verify the actual classic WebGL overlap pixels.
Reversed-depth hosts are also rejected; Three r180 does not expose that native
option, while the packed r186 fixture exercises it. Availability is recorded
explicitly, including the classic host's EXT_clip_control requirement.

Native rendering respects the Spark object's visibility, ancestor visibility and
camera layers. Its placeholder material is hidden without hiding child splats.
Hidden native submissions report zero drawn splats, and hiding during an awaited
update prevents that generation from drawing. Camera reversal is checked before
cloning because Three r180 does not copy that flag.

GPU radix pipeline setup uses a validation error scope before encoding any sort
commands. GPU validation errors return invalid pipeline objects rather than
throwing into JavaScript catch blocks. Initialization awaits that scope, releases
partial resources on failure and uses the generated-depth reference sorter.
The fallback fixture forces real validation errors in each optional pipeline,
a synchronous setup failure, repeated fallback frames, and disposal while setup
is awaiting validation. It verifies balanced scopes, resource release, exact
counts and continued use of the host device.


The dependent radix example exposes `?backend=webgpu&sort=radix`; the default
remains the native reference path. Its status reflects the sorter actually used,
including reference fallback, rather than only the requested option.

Array cameras are rejected before cloning or generation because their per-view
rendering is not implemented by the native route. The host-mode fixture checks
explicit update and render rejection, ordinary-camera recovery, and the classic
WebGL route with splats in two viewports.

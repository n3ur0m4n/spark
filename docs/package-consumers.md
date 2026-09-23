# CommonJS and ESM package consumers

The package declares `type: module`. CommonJS output must therefore use a `.cjs` extension. Publishing CommonJS statements in `spark.cjs.js` caused Node to load that entry as an ES module: a CommonJS consumer could fail or receive no Spark exports. The `main` and `exports.require` entries now point to `dist/spark.cjs`; the ESM/CDN entry stays `dist/spark.module.js`.

Build before checking consumers:

```sh
npm run build:wasm
npm run build
npm run test:package
```

The test creates an `npm pack` tarball and installs it in a temporary consumer project. It checks a real `require()` call from a `.cjs` file, an ESM import, and the existing synchronous WebGL API in TypeScript. It tests Three.js 0.180.0 and pinned 0.186.0, uses bundler resolution for browser TypeScript consumers, and rejects declarations that silently resolve to `any`. A failed consumer project is retained for reproduction.

Linux and Windows CI build the package before running this check. The distribution workflow also watches the package/build configuration so this fix produces the new CommonJS artifact after merging. Generated distribution files are not included in the source contribution.

The classic CommonJS entry bundles small Three addon helpers, while keeping the
host `three`, `three/webgpu` and `three/tsl` modules external. The consumer check
rejects an eager `require("three/...")` so newer Node ESM interop cannot mask a
regression on older hosts. The minimum Three r180 consumer also passes on Node
20.18.3. Three r186's own CommonJS entry requires newer Node ESM interop; r180
and r186 both pass on Node 22.23.2 and Node 23.10.0.

`npm run test:package:browser` checks actual rendering from the npm tarball with
a plain static server and pinned Three r180/r186. Both development and minified
production bundles run the lifecycle, counts, device limits, feature rejection,
color/depth/alpha, resize and disposal fixtures. It requires Chrome with a real
hardware WebGPU adapter and saves raw evidence under `test-results/package-browser/`.
This is a separate hardware check; software-only CI is not GPU validation.

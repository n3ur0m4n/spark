# WebGL ordering work reduction

Radial depth depends on view origin, not view direction. For ordinary built-in
PackedSplats/ExtSplats generation, rotation alone now reuses generated data and
ordering. Translation, directional sorting, source/selection revisions,
transforms and visibility continue to invalidate. prepareGenerate, frameUpdate
and LoD driving still execute.

Eligibility is conservative: custom generators, frameUpdate/onFrame callbacks,
subclasses, paging, covariance sources, modifiers, edits, skinning and other
unproven generation paths retain direction invalidation. Camera-relative packed
data is regenerated during translation; no precision shortcut is introduced.

The accompanying ordering pool retains explicit ownership across worker transfer,
commit and retirement. It does not change texture-upload lifecycle or sort
scheduling. Internal ordering diagnostics are disabled by default and associate
requests, commits and actual draws using mapping compatibility and lifecycle
epochs; timestamp age is reported separately from view displacement.

## Qualification and limits

The browser fixture checks eligible rotation with exact generated data, ordering
and images against forced regeneration; it also checks translation, directional
mode, source revisions, custom callbacks and visibility. Unit tests cover actual
transfer/detachment and pooled ownership. Run `npm run test:native:browser` and
`npm test`; package checks use `npm run test:package` and
`npm run test:package:browser`.

Rotation-only avoided readbacks are a demonstrated mechanism. Full-route paired
timings were variable and do not establish an isolated rendering speedup or
performance equivalence. Allocation reuse likewise has no established FPS gain.
These changes do not claim to eliminate WebGL readback stalls during translation.
CPU-center sorting is excluded because radial keys failed exact comparison with
the generated GPU-depth path. Existing GPU readback remains the default.

This branch depends on the native reference branch for shared integration and
source-eligibility helpers; its diff contains no radix sorter or traversal change.
Types, lint, unit, source-browser, production/development builds and packaged
consumer/browser checks pass on Three r180/r186. Local hardware coverage is
Apple M4 Max / Chrome 153. Other vendors and remote CI
remain unverified.

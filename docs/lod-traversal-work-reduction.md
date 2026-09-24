# Exact LoD traversal work reduction

The traversal keeps the existing binary heap, comparator, insertion/pop sequence,
terminal emission sites and final heap-array drain order. It changes two kinds of
bookkeeping:

- Emit directly into reusable per-instance vectors instead of constructing a
  global tuple list and counting/scattering it afterward.
- Track paging membership with dense epoch marks sized from each registered
  tree's page table, retaining a fallback set for out-of-range chunk requests.
  Instances sharing a tree share membership; distinct tree IDs remain distinct.

The ordered paging list and selection remain unchanged. Membership scratch is released when
its owning tree is disposed, and the remaining traversal output scratch is
released when the final registered tree is disposed. Retained Rust capacity and
WASM high-water memory are different measurements.

## Reproduce

Build an independent checkout of the base revision and this candidate using the
same Rust/WASM toolchain. Do not point both arguments at the same package.

```sh
cargo test --locked --manifest-path rust/Cargo.toml --workspace
npm run build:wasm
node scripts/compare-lod-traversal.mjs \
  --reference /absolute/base/rust/spark-rs/pkg \
  --reference-commit BASE_SHA --candidate-commit CANDIDATE_SHA \
  --benchmark --output traversal-comparison.json
```

The script generates a seeded synthetic tree; no external scene is required. Its
448 seeded compiled-WASM cases plus four empty/leaf cases compare ordered per-instance selections, duplicate
handling, paging order, counters and pixel limits. Native tests additionally
exercise 2,304 seeded configurations, shared-tree membership, epoch wrap,
out-of-range chunks, repeated reuse and tree disposal/re-registration.

Benchmark samples alternate candidate/reference execution order over five
repetitions at each budget. Report all pairs, including outliers, and record
both embedded WASM identities. The timed interval includes the synchronous WASM
call and returned JavaScript output construction; it is not pure Rust time,
worker queue time, rendering FPS or GPU latency.

The initial Apple M4 Max experiment showed roughly 3% typical traversal-time
reduction. This is a CPU/WASM result on a synthetic fixture, not an equal-quality
renderer speedup. Re-run against the exact contributed revision before using a
number publicly. Direct-output alone was noisy; cached frontier metadata was
neutral/slower and is excluded. No heap replacement, early-terminal bypass,
quality reduction or scheduling change is included.

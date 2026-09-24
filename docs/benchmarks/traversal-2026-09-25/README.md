# Synthetic traversal benchmark

Apple M4 Max, macOS, Node v24.18.0. Five alternating candidate/reference repetitions per budget. This measures synchronous WASM calls plus returned JavaScript output construction, not rendering FPS. All 452 differential cases passed. No private asset is used.

Exact source and WASM identities and every sample are in [raw.json](raw.json). The candidate source is `a9f1151e02d4cc6c58b65183bdefd46f08ad82f4`; this report-only follow-up does not change it.

| Requested budget | Actual selected | Reference mean ms | Candidate mean ms | Mean paired change |
|---|---|---|---|---|
| 1,500,000 | 1,500,000 | 356.44 | 341.34 | -4.05% |
| 2,500,000 | 2,463,347 | 564.40 | 555.19 | -1.63% |

A 2.5M sample regressed and is retained. This small local experiment supports modest traversal work reduction, not a universal speedup. Other hardware and operating systems are unmeasured. Reproduction commands and eligibility are in [the traversal notes](../../lod-traversal-work-reduction.md).

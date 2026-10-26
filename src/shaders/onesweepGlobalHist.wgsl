// Copyright © 2011-2026 PlayCanvas Ltd.
// Copyright © Thomas Smith (b0nes164/GPUSorting)
// SPDX-License-Identifier: MIT
//
// OneSweep GlobalHistogram: one read of the keys builds the 8-bit digit
// histograms of every radix pass. Adapted from playcanvas/engine
// 209ce5752527ab5050ec9632ca6dc1272542b70d (onesweep-global-hist.js); the
// key count is always read from b_count[0] (GPU or host written).

@group(0) @binding(0) var<storage, read> b_sort: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> b_globalHist: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> b_count: array<u32>;

override NUM_PASSES: u32 = 2u;

const RADIX: u32 = 256u;
const G_HIST_DIM: u32 = 128u;
const G_HIST_PART_SIZE: u32 = 32768u;
const G_HIST_PART_SIZE_VEC: u32 = G_HIST_PART_SIZE / 4u;

var<workgroup> g_gHist: array<atomic<u32>, 2048>;

fn histOffset(row: u32, pass_: u32) -> u32 {
    return row * RADIX + pass_ * 2u * RADIX;
}

fn addKey(key: u32, row: u32) {
    for (var p = 0u; p < NUM_PASSES; p = p + 1u) {
        atomicAdd(&g_gHist[((key >> (p * 8u)) & 0xFFu) + histOffset(row, p)], 1u);
    }
}

@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(local_invocation_index) gtid: u32,
    @builtin(workgroup_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let flatGid = gid.x + gid.y * nwg.x;
    let numKeys = b_count[0];

    for (var i = gtid; i < 2u * NUM_PASSES * RADIX; i = i + G_HIST_DIM) {
        atomicStore(&g_gHist[i], 0u);
    }
    workgroupBarrier();

    // Two shared rows halve atomic contention. Full vec4s take the fast loop;
    // the single ragged tail vec4 (numKeys % 4 != 0) is handled by one thread.
    let row = gtid / 64u;
    let numKeysVecFull = numKeys >> 2u;
    let partitionStartVec = flatGid * G_HIST_PART_SIZE_VEC;
    let partitionEndVec = min(partitionStartVec + G_HIST_PART_SIZE_VEC, numKeysVecFull);
    for (var i = partitionStartVec + gtid; i < partitionEndVec; i = i + G_HIST_DIM) {
        let q = b_sort[i];
        addKey(q.x, row);
        addKey(q.y, row);
        addKey(q.z, row);
        addKey(q.w, row);
    }

    let tailIdx = numKeysVecFull;
    if ((numKeys & 3u) != 0u &&
        tailIdx >= partitionStartVec &&
        tailIdx < partitionStartVec + G_HIST_PART_SIZE_VEC &&
        (tailIdx - partitionStartVec) % G_HIST_DIM == gtid) {
        let q = b_sort[tailIdx];
        let base = tailIdx << 2u;
        addKey(q.x, row);
        if (base + 1u < numKeys) { addKey(q.y, row); }
        if (base + 2u < numKeys) { addKey(q.z, row); }
    }
    workgroupBarrier();

    for (var i = gtid; i < RADIX; i = i + G_HIST_DIM) {
        for (var p = 0u; p < NUM_PASSES; p = p + 1u) {
            let total = atomicLoad(&g_gHist[i + histOffset(0u, p)]) +
                atomicLoad(&g_gHist[i + histOffset(1u, p)]);
            if (total != 0u) {
                atomicAdd(&b_globalHist[i + p * RADIX], total);
            }
        }
    }
}

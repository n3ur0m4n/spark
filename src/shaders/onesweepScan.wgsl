// Copyright © 2011-2026 PlayCanvas Ltd.
// Copyright © Thomas Smith (b0nes164/GPUSorting)
// SPDX-License-Identifier: MIT
//
// OneSweep Scan: exclusive scan of each pass's 256-digit global histogram,
// published to partition 0 of b_passHist as FLAG_INCLUSIVE so the chained
// lookback of DigitBinningPass terminates. One workgroup per radix pass.
// Adapted from playcanvas/engine 209ce5752527ab5050ec9632ca6dc1272542b70d
// (onesweep-scan.js).

enable subgroups;

@group(0) @binding(0) var<storage, read> b_globalHist: array<u32>;
@group(0) @binding(1) var<storage, read_write> b_passHist: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> b_count: array<u32>;

const RADIX: u32 = 256u;
const FLAG_INCLUSIVE: u32 = 2u;
// Host-patched: tile size of the binning pass and 256 / min subgroup size.
const PART_SIZE: u32 = 3584u;
const MAX_SUBGROUPS: u32 = 8u;

var<workgroup> sg_totals: array<u32, MAX_SUBGROUPS>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_index) gtid: u32,
    @builtin(workgroup_id) gid: vec3<u32>,
    @builtin(subgroup_invocation_id) sgInvId: u32,
    @builtin(subgroup_size) sgSize: u32,
) {
    let pass_ = gid.x;
    let threadBlocks = (b_count[0] + PART_SIZE - 1u) / PART_SIZE;
    let waveIndex = gtid / sgSize;

    let t = b_globalHist[gtid + pass_ * RADIX];
    let sgExcl = subgroupExclusiveAdd(t);
    let sgTotal = subgroupAdd(t);
    if (sgInvId == 0u) {
        sg_totals[waveIndex] = sgTotal;
    }
    workgroupBarrier();

    if (gtid == 0u) {
        var acc: u32 = 0u;
        for (var i = 0u; i < MAX_SUBGROUPS; i = i + 1u) {
            let v = sg_totals[i];
            sg_totals[i] = acc;
            acc = acc + v;
        }
    }
    workgroupBarrier();

    let excl = sgExcl + sg_totals[waveIndex];
    atomicStore(&b_passHist[pass_ * threadBlocks * RADIX + gtid], (excl << 2u) | FLAG_INCLUSIVE);
}

// Copyright © 2011-2026 PlayCanvas Ltd.
// Copyright © kishimisu (WebGPU-Radix-Sort)
// Copyright © 2025 WORLD LABS TECHNOLOGIES, INC.
// SPDX-License-Identifier: MIT
//
// Adapted from playcanvas/engine src/scene/shader-lib/wgsl/chunks/radix-sort/
// compute-radix-sort-reorder.js at 329a94115f5af7ff8ccf8d4348aded800b0e3fbd.
// Modifications: Spark GPUDevice bind groups, pipeline overrides instead of
// source templates, direct-dispatch only. See NOTICE.

@group(0) @binding(0) var<storage, read> inputKeys: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputKeys: array<u32>;
@group(0) @binding(2) var<storage, read> prefix_block_sum: array<u32>;
@group(0) @binding(3) var<storage, read> inputValues: array<u32>;
@group(0) @binding(4) var<storage, read_write> outputValues: array<u32>;

struct RadixSortUniforms {
    workgroupCount: u32,
    elementCount: u32,
};
@group(0) @binding(5) var<uniform> uniforms: RadixSortUniforms;

const THREADS_PER_WORKGROUP: u32 = 256u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const ELEMENTS_PER_THREAD: u32 = 8u;
const ELEMENTS_PER_WORKGROUP: u32 = THREADS_PER_WORKGROUP * ELEMENTS_PER_THREAD;

override CURRENT_BIT: u32 = 0u;
override IS_FIRST_PASS: u32 = 1u;
override IS_LAST_PASS: u32 = 0u;

var<workgroup> digit_masks: array<atomic<u32>, 128>;
var<workgroup> digit_offsets: array<u32, 16>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32,
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * ELEMENTS_PER_WORKGROUP;

    let word_idx = TID >> 5u;
    let bit_idx = TID & 31u;

    if (TID < 16u) {
        digit_offsets[TID] = 0u;
    }
    if (TID < 128u) {
        atomicStore(&digit_masks[TID], 0u);
    }
    workgroupBarrier();

    let elementCount = uniforms.elementCount;

    for (var round = 0u; round < ELEMENTS_PER_THREAD; round++) {
        let GID = WID + round * THREADS_PER_WORKGROUP + TID;
        let is_valid = GID < elementCount && WORKGROUP_ID < uniforms.workgroupCount;
        var k = 0u;
        var digit = 16u;
        var v = 0u;
        // select() evaluates both values: it cannot guard a storage load in
        // the partial final workgroup. Keep out-of-range invocations alive
        // through every barrier without touching keys or values.
        if (is_valid) {
            k = inputKeys[GID];
            digit = (k >> CURRENT_BIT) & 0xFu;
            if (IS_FIRST_PASS == 1u) {
                v = GID;
            } else {
                v = inputValues[GID];
            }
            atomicOr(&digit_masks[digit * 8u + word_idx], 1u << bit_idx);
        }
        workgroupBarrier();

        if (is_valid) {
            let base = digit * 8u;
            var local_prefix = digit_offsets[digit];
            for (var w = 0u; w < word_idx; w++) {
                local_prefix += countOneBits(atomicLoad(&digit_masks[base + w]));
            }
            local_prefix += countOneBits(atomicLoad(&digit_masks[base + word_idx]) & ((1u << bit_idx) - 1u));

            let pid = digit * uniforms.workgroupCount + WORKGROUP_ID;
            let sorted_position = prefix_block_sum[pid] + local_prefix;

            if (IS_LAST_PASS == 0u) {
                outputKeys[sorted_position] = k;
            }
            outputValues[sorted_position] = v;
        }

        if (round < ELEMENTS_PER_THREAD - 1u) {
            workgroupBarrier();
            if (TID < 16u) {
                var count = 0u;
                for (var w = 0u; w < 8u; w++) {
                    let idx = TID * 8u + w;
                    count += countOneBits(atomicLoad(&digit_masks[idx]));
                    atomicStore(&digit_masks[idx], 0u);
                }
                digit_offsets[TID] += count;
            }
            workgroupBarrier();
        }
    }
}

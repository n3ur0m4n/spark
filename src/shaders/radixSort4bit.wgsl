// Copyright © 2011-2026 PlayCanvas Ltd.
// Copyright © kishimisu (WebGPU-Radix-Sort)
// Copyright © 2025 WORLD LABS TECHNOLOGIES, INC.
// SPDX-License-Identifier: MIT
//
// Adapted from playcanvas/engine src/scene/shader-lib/wgsl/chunks/radix-sort/
// compute-radix-sort-4bit.js at 329a94115f5af7ff8ccf8d4348aded800b0e3fbd.
// Modifications: Spark GPUDevice bind groups, pipeline overrides instead of
// source templates, direct-dispatch only. See NOTICE.

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;

struct RadixSortUniforms {
    workgroupCount: u32,
    elementCount: u32,
};
@group(0) @binding(2) var<uniform> uniforms: RadixSortUniforms;

const THREADS_PER_WORKGROUP: u32 = 256u;
const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const ELEMENTS_PER_THREAD: u32 = 8u;
const ELEMENTS_PER_WORKGROUP: u32 = THREADS_PER_WORKGROUP * ELEMENTS_PER_THREAD;

override CURRENT_BIT: u32 = 0u;

var<workgroup> histogram: array<atomic<u32>, 16>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32,
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * ELEMENTS_PER_WORKGROUP;

    if (TID < 16u) {
        atomicStore(&histogram[TID], 0u);
    }
    workgroupBarrier();

    let elementCount = uniforms.elementCount;

    for (var r = 0u; r < ELEMENTS_PER_THREAD; r++) {
        let GID = WID + r * THREADS_PER_WORKGROUP + TID;
        let is_valid = GID < elementCount && WORKGROUP_ID < uniforms.workgroupCount;

        if (is_valid) {
            let elm = input[GID];
            let digit = (elm >> CURRENT_BIT) & 0xFu;
            atomicAdd(&histogram[digit], 1u);
        }
    }
    workgroupBarrier();

    if (TID < 16u && WORKGROUP_ID < uniforms.workgroupCount) {
        block_sums[TID * uniforms.workgroupCount + WORKGROUP_ID] = atomicLoad(&histogram[TID]);
    }
}

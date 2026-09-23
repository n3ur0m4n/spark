// Copyright © 2011-2026 PlayCanvas Ltd.
// Copyright © kishimisu (WebGPU-Radix-Sort)
// Copyright © 2025 WORLD LABS TECHNOLOGIES, INC.
// SPDX-License-Identifier: MIT
//
// Adapted from playcanvas/engine src/scene/shader-lib/wgsl/chunks/radix-sort/
// compute-prefix-sum.js at 329a94115f5af7ff8ccf8d4348aded800b0e3fbd.
// Parallel Prefix Sum (Scan) using Blelloch algorithm.
// Based on "Parallel Prefix Sum (Scan) with CUDA"
// https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
// See NOTICE.

@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

struct PrefixSumUniforms {
    elementCount: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(2) var<uniform> uniforms: PrefixSumUniforms;

const WORKGROUP_SIZE_X: u32 = 16u;
const WORKGROUP_SIZE_Y: u32 = 16u;
const THREADS_PER_WORKGROUP: u32 = 256u;
const ITEMS_PER_WORKGROUP: u32 = 512u;

var<workgroup> temp: array<u32, 1024>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32,
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    if (WORKGROUP_ID * ITEMS_PER_WORKGROUP >= uniforms.elementCount) { return; }
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID;

    let ELM_TID = TID * 2u;
    let ELM_GID = GID * 2u;

    temp[ELM_TID] = 0u;
    temp[ELM_TID + 1u] = 0u;
    if (ELM_GID < uniforms.elementCount) { temp[ELM_TID] = items[ELM_GID]; }
    if (ELM_GID + 1u < uniforms.elementCount) { temp[ELM_TID + 1u] = items[ELM_GID + 1u]; }

    var offset: u32 = 1u;

    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();

        if (TID < d) {
            let ai: u32 = offset * (ELM_TID + 1u) - 1u;
            let bi: u32 = offset * (ELM_TID + 2u) - 1u;
            temp[bi] += temp[ai];
        }

        offset *= 2u;
    }

    if (TID == 0u) {
        let last_offset = ITEMS_PER_WORKGROUP - 1u;
        blockSums[WORKGROUP_ID] = temp[last_offset];
        temp[last_offset] = 0u;
    }

    for (var d: u32 = 1u; d < ITEMS_PER_WORKGROUP; d *= 2u) {
        offset >>= 1u;
        workgroupBarrier();

        if (TID < d) {
            let ai: u32 = offset * (ELM_TID + 1u) - 1u;
            let bi: u32 = offset * (ELM_TID + 2u) - 1u;

            let t: u32 = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    if (ELM_GID < uniforms.elementCount) {
        items[ELM_GID] = temp[ELM_TID];
    }

    if (ELM_GID + 1u < uniforms.elementCount) {
        items[ELM_GID + 1u] = temp[ELM_TID + 1u];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32,
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID;

    let ELM_ID = GID * 2u;

    if (ELM_ID >= uniforms.elementCount) {
        return;
    }

    let blockSum = blockSums[WORKGROUP_ID];

    items[ELM_ID] += blockSum;

    if (ELM_ID + 1u >= uniforms.elementCount) {
        return;
    }

    items[ELM_ID + 1u] += blockSum;
}

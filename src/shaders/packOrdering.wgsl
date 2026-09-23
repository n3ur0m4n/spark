// Copyright © 2025 WORLD LABS TECHNOLOGIES, INC.
// SPDX-License-Identifier: MIT
//
// Pack a GPU-sorted u32 index buffer into Spark's RGBA32UI ordering texture
// (4 indices per texel, 4096-wide).

@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var orderTex: texture_storage_2d<rgba32uint, write>;

struct PackUniforms {
    numSplats: u32,
    width: u32,
};
@group(0) @binding(2) var<uniform> uniforms: PackUniforms;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    if (gid.x >= uniforms.width) {
        return;
    }
    let texelIndex = gid.y * uniforms.width + gid.x;
    let base = texelIndex * 4u;
    var packed = vec4u(0xFFFFFFFFu);
    if (base < uniforms.numSplats) {
        packed.x = indices[base];
    }
    if (base + 1u < uniforms.numSplats) {
        packed.y = indices[base + 1u];
    }
    if (base + 2u < uniforms.numSplats) {
        packed.z = indices[base + 2u];
    }
    if (base + 3u < uniforms.numSplats) {
        packed.w = indices[base + 3u];
    }
    textureStore(orderTex, vec2<i32>(i32(gid.x), i32(gid.y)), packed);
}

// Native counterpart of SplatAccumulator's ordinary SplatMesh generation.
// One dispatch per mesh writes its upstream row-aligned destination range.
// Depth is evaluated before output quantization, just like the WebGL MRT.
struct GenerateUniforms {
    base: u32, count: u32, sourceCount: u32, indexed: u32,
    sourceExt: u32, outputExt: u32, sortRadial: u32, numSh: u32,
    rotation: vec4f,
    translationScale: vec4f,
    viewOrigin: vec4f,
    viewDirection: vec4f,
    localViewOrigin: vec4f,
    encoding: vec4f,
    shMaxOpacity: vec4f,
    shOffsets: vec4u,
    recolor: vec4f,
};
@group(0) @binding(0) var<uniform> frame: GenerateUniforms;
@group(0) @binding(1) var<storage, read> primary: array<vec4u>;
@group(0) @binding(2) var<storage, read> secondary: array<vec4u>;
@group(0) @binding(3) var<storage, read> sh: array<u32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4u>;
@group(0) @binding(6) var<storage, read_write> output2: array<vec4u>;
@group(0) @binding(7) var<storage, read_write> depths: array<u32>;

fn sh4(offset: u32) -> vec4u {
    return vec4u(sh[offset], sh[offset + 1u], sh[offset + 2u], sh[offset + 3u]);
}

@compute @workgroup_size(256)
fn generate(@builtin(workgroup_id) group: vec3u,
            @builtin(local_invocation_id) local: vec3u,
            @builtin(num_workgroups) groups: vec3u) {
    let i = (group.y * groups.x + group.x) * 256u + local.x;
    if (i >= frame.count) { return; }
    let dst = frame.base + i;
    var src = i;
    if (frame.indexed != 0u) { src = indices[i]; }
    if (src >= frame.sourceCount) { return; }
    var splat: UnpackedSplat;
    if (frame.sourceExt != 0u) {
        splat = unpackSplatExt(primary[src], secondary[src]);
    } else {
        splat = unpackSplatEncoding(primary[src], frame.encoding);
        splat.rgba.a *= frame.shMaxOpacity.w;
    }
    // Upstream marks source activity by nonzero scales, not opacity. Zero
    // alpha still owns a finite depth/order entry and is discarded at draw.
    if (all(splat.scales == vec3f(0.0))) { return; }

    if (frame.numSh > 0u) {
        let dir = normalize(splat.center - frame.localViewOrigin.xyz);
        var rgb = vec3f(0.0);
        if (frame.sourceExt != 0u) {
            let a = sh4(frame.shOffsets.x + src * 4u);
            if (frame.numSh == 1u) {
                rgb = evaluateExtSH1(a, dir);
            } else {
                rgb = evaluateExtSH12(a, sh4(frame.shOffsets.y + src * 4u), dir);
                if (frame.numSh >= 3u) {
                    rgb += evaluateExtSH3(sh4(frame.shOffsets.z + src * 4u), sh4(frame.shOffsets.w + src * 4u), dir);
                }
            }
        } else {
            let off = frame.shOffsets.x + src * 2u;
            rgb = evaluatePackedSH1(vec2u(sh[off], sh[off + 1u]), dir, frame.shMaxOpacity.x);
            if (frame.numSh >= 2u) {
                rgb += evaluatePackedSH2(sh4(frame.shOffsets.y + src * 4u), dir, frame.shMaxOpacity.y);
            }
            if (frame.numSh >= 3u) {
                rgb += evaluatePackedSH3(sh4(frame.shOffsets.z + src * 4u), dir, frame.shMaxOpacity.z);
            }
        }
        // Match upstream source generation: signed SH colors survive until
        // output encoding. Extended output preserves negative half floats;
        // packed output performs its own format-specific clamping.
        splat.rgba = vec4f(splat.rgba.rgb + rgb, splat.rgba.a);
    }
    splat.rgba *= frame.recolor;
    let center = quatVec(frame.rotation, splat.center * frame.translationScale.w) + frame.translationScale.xyz;
    let delta = center - frame.viewOrigin.xyz;
    var metric = dot(delta, frame.viewDirection.xyz) + 100.0;
    if (frame.sortRadial != 0u) { metric = length(delta); }
    // Negative/NaN/infinite bit patterns are excluded by upstream sort32.
    depths[dst] = bitcast<u32>(metric);
    let scales = splat.scales * frame.translationScale.w;
    let rotation = quatQuat(frame.rotation, splat.quaternion);
    if (frame.outputExt != 0u) {
        let packed = packSplatExt(center, scales, rotation, splat.rgba);
        output[dst] = packed.packed;
        output2[dst] = packed.packed2;
    } else {
        // Upstream uses camera-relative half precision and expanded LoD alpha.
        output[dst] = packSplat(delta, scales, rotation, vec4f(splat.rgba.rgb, splat.rgba.a * 0.5));
    }
}

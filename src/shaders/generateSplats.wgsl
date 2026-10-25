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
// Bindings 1-3 (and 8-10 for paged sources) and the load*() accessors come
// from generateSourceBuffers.wgsl or generateSourcePaged.wgsl.
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<vec4u>;
@group(0) @binding(6) var<storage, read_write> output2: array<vec4u>;
@group(0) @binding(7) var<storage, read_write> depths: array<u32>;

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
        splat = unpackSplatExt(loadPrimary(src), loadSecondary(src));
    } else {
        splat = unpackSplatEncoding(loadPrimary(src), frame.encoding);
        splat.rgba.a *= frame.shMaxOpacity.w;
    }
    // Upstream marks source activity by nonzero scales, not opacity. Zero
    // alpha still owns a finite depth/order entry and is discarded at draw,
    // except for paged sources, whose readers drop it like zero scales.
    if (all(splat.scales == vec3f(0.0)) || (ZERO_ALPHA_INACTIVE && splat.rgba.a == 0.0)) { return; }

    if (frame.numSh > 0u) {
        let dir = normalize(splat.center - frame.localViewOrigin.xyz);
        var rgb = vec3f(0.0);
        if (frame.sourceExt != 0u) {
            let a = loadSh(0u, src);
            if (frame.numSh == 1u) {
                rgb = evaluateExtSH1(a, dir);
            } else {
                rgb = evaluateExtSH12(a, loadSh(1u, src), dir);
                if (frame.numSh >= 3u) {
                    rgb += evaluateExtSH3(loadSh(2u, src), loadSh(3u, src), dir);
                }
            }
        } else {
            rgb = evaluatePackedSH1(loadPackedSh1(src), dir, frame.shMaxOpacity.x);
            if (frame.numSh >= 2u) {
                rgb += evaluatePackedSH2(loadSh(1u, src), dir, frame.shMaxOpacity.y);
            }
            if (frame.numSh >= 3u) {
                rgb += evaluatePackedSH3(loadSh(2u, src), dir, frame.shMaxOpacity.z);
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

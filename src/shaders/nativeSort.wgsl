struct SortParams { count: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> params: SortParams;
@group(0) @binding(1) var<storage, read> depths: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read> sorted: array<u32>;
@group(0) @binding(4) var<storage, read_write> drawArgs: array<u32>;

@compute @workgroup_size(256)
fn prepareKeys(@builtin(workgroup_id) group: vec3u,
               @builtin(local_invocation_id) local: vec3u,
               @builtin(num_workgroups) groups: vec3u) {
    let i = (group.y * groups.x + group.x) * 256u + local.x;
    if (i >= params.count) { return; }
    let depth = depths[i];
    // Match sort32's validity and descending order, including valid zero and
    // stable ties. Invalid keys sort strictly after every finite positive key.
    keys[i] = select(0xffffffffu, 0x7fffffffu - depth, depth < 0x7f800000u);
}

@compute @workgroup_size(1)
fn countDraw() {
    // The sorted valid prefix ends at the first invalid depth. Binary search
    // avoids CPU readback and avoids an atomic increment for every splat.
    var lo = 0u;
    var hi = params.count;
    while (lo < hi) {
        let mid = lo + (hi - lo) / 2u;
        if (depths[sorted[mid]] < 0x7f800000u) { lo = mid + 1u; }
        else { hi = mid; }
    }
    drawArgs[0] = 6u;
    drawArgs[1] = lo;
    drawArgs[2] = 0u;
    drawArgs[3] = 0u;
    drawArgs[4] = 0u;
}

// 16-bit keys: depths quantized linearly between the view's valid min/max.
// range[0] holds ~min so a zero-cleared buffer is the identity for atomicMax.
@group(0) @binding(5) var<storage, read_write> range: array<atomic<u32>>;
var<workgroup> wgMin: atomic<u32>;
var<workgroup> wgMax: atomic<u32>;

@compute @workgroup_size(256)
fn depthRange(@builtin(workgroup_id) group: vec3u,
              @builtin(local_invocation_id) local: vec3u,
              @builtin(num_workgroups) groups: vec3u) {
    let base = (group.y * groups.x + group.x) * 2048u + local.x;
    var lo = 0xffffffffu;
    var hi = 0u;
    for (var k = 0u; k < 8u; k = k + 1u) {
        let i = base + k * 256u;
        if (i < params.count) {
            let depth = depths[i];
            if (depth < 0x7f800000u) {
                lo = min(lo, depth);
                hi = max(hi, depth);
            }
        }
    }
    if (local.x == 0u) {
        atomicStore(&wgMin, 0xffffffffu);
        atomicStore(&wgMax, 0u);
    }
    workgroupBarrier();
    atomicMin(&wgMin, lo);
    atomicMax(&wgMax, hi);
    workgroupBarrier();
    if (local.x == 0u) {
        atomicMax(&range[0], ~atomicLoad(&wgMin));
        atomicMax(&range[1], atomicLoad(&wgMax));
    }
}

@compute @workgroup_size(256)
fn prepareKeys16(@builtin(workgroup_id) group: vec3u,
                 @builtin(local_invocation_id) local: vec3u,
                 @builtin(num_workgroups) groups: vec3u) {
    let i = (group.y * groups.x + group.x) * 256u + local.x;
    if (i >= params.count) { return; }
    let depth = depths[i];
    // Positive finite f32 bits order like the floats; far splats sort first.
    let lo = bitcast<f32>(~atomicLoad(&range[0]));
    let hi = bitcast<f32>(atomicLoad(&range[1]));
    let scale = select(0.0, 65534.0 / (hi - lo), hi > lo);
    let q = u32(clamp((bitcast<f32>(depth) - lo) * scale, 0.0, 65534.0));
    keys[i] = select(0xffffu, 0xfffeu - q, depth < 0x7f800000u);
}

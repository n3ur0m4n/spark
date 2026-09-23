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

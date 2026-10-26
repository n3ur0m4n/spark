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

// OneSweep path: drop invalid (culled) splats before sorting and quantize the
// survivors' depth to 16 bits between the view's valid min/max. Compaction
// keeps index order, so the stable sort matches sorting every splat.
// range[0] holds ~min so a zero-cleared buffer is the identity for atomicMax.
@group(0) @binding(5) var<storage, read_write> range: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> blockCounts: array<u32>;
@group(0) @binding(7) var<storage, read_write> sortCount: array<u32>;
@group(0) @binding(8) var<storage, read_write> sortArgs: array<u32>;
@group(0) @binding(9) var<storage, read_write> values: array<u32>;
var<workgroup> wgMin: atomic<u32>;
var<workgroup> wgMax: atomic<u32>;
var<workgroup> scan: array<u32, 256>;

const BLOCK: u32 = 2048u;

fn isValid(i: u32) -> bool {
    return i < params.count && depths[i] < 0x7f800000u;
}

// Inclusive Hillis-Steele scan of `value` across the workgroup.
fn workgroupScan(local: u32, value: u32) -> u32 {
    scan[local] = value;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset = offset * 2u) {
        let add = select(0u, scan[local - offset], local >= offset);
        workgroupBarrier();
        scan[local] = scan[local] + add;
        workgroupBarrier();
    }
    return scan[local];
}

// Per 2048-splat block: valid count and the valid depth range.
@compute @workgroup_size(256)
fn compactCount(@builtin(workgroup_id) group: vec3u,
                @builtin(local_invocation_id) local: vec3u,
                @builtin(num_workgroups) groups: vec3u) {
    let block = group.y * groups.x + group.x;
    let base = block * BLOCK + local.x * 8u;
    var lo = 0xffffffffu;
    var hi = 0u;
    var valid = 0u;
    for (var k = 0u; k < 8u; k = k + 1u) {
        if (isValid(base + k)) {
            let depth = depths[base + k];
            lo = min(lo, depth);
            hi = max(hi, depth);
            valid = valid + 1u;
        }
    }
    if (local.x == 0u) {
        atomicStore(&wgMin, 0xffffffffu);
        atomicStore(&wgMax, 0u);
    }
    let total = workgroupScan(local.x, valid);
    atomicMin(&wgMin, lo);
    atomicMax(&wgMax, hi);
    workgroupBarrier();
    if (local.x == 255u && block * BLOCK < params.count) {
        blockCounts[block] = total;
        atomicMax(&range[0], ~atomicLoad(&wgMin));
        atomicMax(&range[1], atomicLoad(&wgMax));
    }
}

// One workgroup: exclusive block offsets, then the visible count drives the
// sort, pack and draw dispatches (params._a = OneSweep partition size).
@compute @workgroup_size(256)
fn compactScan(@builtin(local_invocation_id) local: vec3u) {
    let blocks = (params.count + BLOCK - 1u) / BLOCK;
    var carry = 0u;
    for (var start = 0u; start < blocks; start = start + 256u) {
        let i = start + local.x;
        let value = select(0u, blockCounts[i], i < blocks);
        let inclusive = workgroupScan(local.x, value);
        if (i < blocks) { blockCounts[i] = carry + inclusive - value; }
        carry = carry + scan[255];
        workgroupBarrier();
    }
    if (local.x == 0u) {
        let n = carry;
        sortCount[0] = n;
        sortArgs[0] = (n + params._a - 1u) / params._a;
        sortArgs[1] = 1u;
        sortArgs[2] = 1u;
        sortArgs[3] = (n + 32767u) / 32768u;
        sortArgs[4] = 1u;
        sortArgs[5] = 1u;
        // packOrdering: 16x16 texels (4 indices each) of a 4096-wide texture.
        sortArgs[6] = 256u;
        sortArgs[7] = ((n + 16383u) / 16384u + 15u) / 16u;
        sortArgs[8] = 1u;
        drawArgs[0] = 6u;
        drawArgs[1] = n;
        drawArgs[2] = 0u;
        drawArgs[3] = 0u;
        drawArgs[4] = 0u;
    }
}

@compute @workgroup_size(256)
fn compactScatter(@builtin(workgroup_id) group: vec3u,
                  @builtin(local_invocation_id) local: vec3u,
                  @builtin(num_workgroups) groups: vec3u) {
    let block = group.y * groups.x + group.x;
    let base = block * BLOCK + local.x * 8u;
    var valid = 0u;
    for (var k = 0u; k < 8u; k = k + 1u) {
        valid = valid + select(0u, 1u, isValid(base + k));
    }
    let inclusive = workgroupScan(local.x, valid);
    if (block * BLOCK >= params.count) { return; }
    var out = blockCounts[block] + inclusive - valid;
    // Positive finite f32 bits order like the floats; far splats sort first.
    let lo = bitcast<f32>(~atomicLoad(&range[0]));
    let hi = bitcast<f32>(atomicLoad(&range[1]));
    let scale = select(0.0, 65534.0 / (hi - lo), hi > lo);
    for (var k = 0u; k < 8u; k = k + 1u) {
        let i = base + k;
        if (isValid(i)) {
            let q = u32(clamp((bitcast<f32>(depths[i]) - lo) * scale, 0.0, 65534.0));
            keys[out] = 0xfffeu - q;
            values[out] = i;
            out = out + 1u;
        }
    }
}

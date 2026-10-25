// splatPickFragment.wgsl - WGSL port of splatPickFragment.glsl
// GPU pick (pickAsync) fragment shader, paired with the "//PICK" variant of
// splatVertex.wgsl. splatDefines.wgsl is prepended at runtime.

struct FragmentUniforms {
    near: f32,
    far: f32,
    encodeLinear: u32,
    time: f32,
    debugFlag: u32,
    maxStdDev: f32,
    minAlpha: f32,
    disableFalloff: u32,
    falloff: f32,
    stochastic: u32,
    renderHeight: u32,
}

struct PickUniforms {
    // Alpha above which a splat counts as a "solid" hit.
    alphaThreshold: f32,
    // 0 = weighted-average pass, 1 = first-solid-hit pass.
    solidMode: u32,
    // "No solid hit" sentinel; keep in sync with farBig in SparkRenderer.ts.
    farBig: f32,
}

@group(0) @binding(1) var<uniform> projectionMatrix: mat4x4f;
@group(0) @binding(5) var<uniform> fragUniforms: FragmentUniforms;
@group(0) @binding(6) var<uniform> pick: PickUniforms;

struct FragmentInput {
    @location(0) vRgba: vec4f,
    @location(1) vSplatUv: vec2f,
    @location(2) vNdc: vec3f,
    @location(4) @interpolate(flat) adjustedStdDev: f32,
    @location(5) vPickRay: vec3f,
    @location(6) @interpolate(flat) vPickCenter: vec3f,
    @location(7) @interpolate(flat) vPickSinvDiag: vec3f,
    @location(8) @interpolate(flat) vPickSinvOff: vec3f,
    @location(9) @interpolate(flat) vPickAux: vec2f,
}

@fragment
fn fs_pick(input: FragmentInput) -> @location(0) vec4f {
    var rgba = input.vRgba;

    let z2 = dot(input.vSplatUv, input.vSplatUv);
    if (z2 > (input.adjustedStdDev * input.adjustedStdDev)) {
        discard;
    }

    // Same cutoff/falloff logic as splatFragment.wgsl.
    if (rgba.a <= 1.0) {
        rgba.a = mix(rgba.a, rgba.a * exp(-0.5 * z2), fragUniforms.falloff);
    } else {
        let e = exp((rgba.a * rgba.a - 1.0) / 2.718281828459045);
        let alpha = 1.0 - pow(1.0 - exp(-0.5 * z2), e);
        rgba.a = mix(1.0, alpha, fragUniforms.falloff);
    }

    if (rgba.a < fragUniforms.minAlpha) {
        discard;
    }
    let a = rgba.a;

    // Linear view depth from WebGPU NDC z ([0, 1]).
    let p = projectionMatrix;
    var viewZ: f32;
    if (p[3][3] == 1.0) {
        viewZ = (input.vNdc.z - p[3][2]) / p[2][2];
    } else {
        viewZ = -p[3][2] / (input.vNdc.z + p[2][2]);
    }
    var d = -viewZ;

    if (input.vPickAux.y < 0.5) {
        // Max-density point along the ray p = t*r (r.z = -1, t = view depth):
        // t* = t0 + r^T S delta / r^T S r, delta = mu - t0*r (see the GLSL).
        let r = input.vPickRay;
        let sd = input.vPickSinvDiag;
        let so = input.vPickSinvOff;
        let sR = vec3f(
            dot(vec3f(sd.x, so.x, so.y), r),
            dot(vec3f(so.x, sd.y, so.z), r),
            dot(vec3f(so.y, so.z, sd.z), r),
        );
        let rSr = dot(r, sR);
        let t0 = -input.vPickCenter.z;
        d = t0;
        if (rSr > 0.0) {
            let delta = input.vPickCenter - t0 * r;
            let dt = dot(sR, delta) / rSr;
            let bound = input.adjustedStdDev * input.vPickAux.x;
            d = t0 + clamp(dt, -bound, bound);
        }
        d = max(d, fragUniforms.near);
    }

    if (pick.solidMode != 0u) {
        // MIN-blended: R ends up as the frontmost depth with a > alphaThreshold.
        return vec4f(select(pick.farBig, d, a > pick.alphaThreshold), 0.0, 0.0, 0.0);
    }
    // Premultiplied "over" blend, back to front: R = sum(w*d), A = sum(w).
    return vec4f(d * a, 0.0, 0.0, a);
}

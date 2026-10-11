precision highp float;
precision highp int;

#include <splatDefines>
// Pulls in packing.glsl's perspectiveDepthToViewZ / orthographicDepthToViewZ.
#include <packing>

uniform float near;
uniform float far;
uniform float maxStdDev;
uniform float minAlpha;
uniform float falloff;
// GPU-pick-only uniform: alpha above which a splat counts as a "solid" hit.
uniform float alphaThreshold;
// Selects which of pickAsync()'s two render passes this material is used
// for (two separate ShaderMaterial instances, each with this uniform fixed
// at creation time - see SparkRenderer.pickAsync()):
// false = weighted-average pass, true = first-solid-hit pass.
uniform bool pickSolidMode;

out vec4 fragColor;

in vec4 vRgba;
in vec2 vSplatUv;
in vec3 vNdc;
flat in uint vSplatIndex;
flat in float adjustedStdDev;

#ifdef SPARK_PICK_DEPTH
in vec3 vPickRay;
flat in vec3 vPickCenter;
flat in vec3 vPickSinvDiag;
flat in vec3 vPickSinvOff;
flat in vec2 vPickAux;
#endif

#include <logdepthbuf_pars_fragment>

// Sentinel written to the R channel (MIN-blended, solid-hit pass only) for
// fragments that never reach alphaThreshold, and used as that pass's clear
// value. Kept well inside the HalfFloatType range (max ~65504) and scaled
// off `far` so it stays unambiguously larger than any real splat depth,
// instead of an extreme value like 1e30 that would overflow to Infinity on
// a HalfFloatType target. Reading/clearing the *color* (R) channel rather
// than alpha also sidesteps any doubt about alpha-clear clamping.
// Keep this formula in sync with pickAsync()'s farBig in SparkRenderer.ts.
float farBig() {
    return min(far * 4.0, 60000.0);
}

void main() {
    vec4 rgba = vRgba;

    float z2 = dot(vSplatUv, vSplatUv);
    if (z2 > (adjustedStdDev * adjustedStdDev)) {
        discard;
    }

    // Same cutoff/falloff logic as splatFragment.glsl.
    if (rgba.a <= 1.0) {
        rgba.a = mix(rgba.a, rgba.a * exp(-0.5 * z2), falloff);
    } else {
        float a = exp((rgba.a * rgba.a - 1.0) / 2.718281828459045);
        float alpha = 1.0 - pow(1.0 - exp(-0.5 * z2), a);
        rgba.a = mix(1.0, alpha, falloff);
    }

    if (rgba.a < minAlpha) {
        discard;
    }

    float a = rgba.a;

    // Linear view-space depth (positive, world units) recovered from NDC z.
    // vNdc.z is in [-1, 1] (OpenGL NDC); perspectiveDepthToViewZ/
    // orthographicDepthToViewZ (from <packing>) expect depth in [0, 1].
    float depth01 = vNdc.z * 0.5 + 0.5;
    float viewZ = isOrthographic
        ? orthographicDepthToViewZ(depth01, near, far)
        : perspectiveDepthToViewZ(depth01, near, far);
    float d = -viewZ;

    #ifdef SPARK_PICK_DEPTH
    if (vPickAux.y < 0.5) {
        // Max-density point along the ray p = t*r (r.z = -1, so t is view
        // depth): maximizing exp(-1/2 (t r - mu)^T S (t r - mu)) gives
        // t* = r^T S mu / r^T S r. Written as t0 + r^T S delta / r^T S r with
        // t0 = center depth and delta = mu - t0*r (small, delta.z = 0) to
        // avoid cancellation on far splats.
        vec3 r = vPickRay;
        vec3 sR = vec3(
            dot(vec3(vPickSinvDiag.x, vPickSinvOff.x, vPickSinvOff.y), r),
            dot(vec3(vPickSinvOff.x, vPickSinvDiag.y, vPickSinvOff.z), r),
            dot(vec3(vPickSinvOff.y, vPickSinvOff.z, vPickSinvDiag.z), r)
        );
        float rSr = dot(r, sR);
        float t0 = -vPickCenter.z;
        d = t0;
        if (rSr > 0.0) {
            vec3 delta = vPickCenter - t0 * r;
            float dt = dot(sR, delta) / rSr;
            // Cauchy-Schwarz: |p*.z - mu.z| <= m * sqrt(cov zz), with m the
            // Mahalanobis distance of p*, bounded by the quad cutoff
            // adjustedStdDev. (A +-3 sigma-along-ray clamp would cancel the
            // correction on thin oblique splats.)
            float bound = adjustedStdDev * vPickAux.x;
            d = t0 + clamp(dt, -bound, bound);
        }
        d = max(d, near);
    }
    #endif

    if (pickSolidMode) {
        // Solid-hit pass: R is MIN-blended (set up in JS) across every
        // fragment covering this pixel, so it ends up holding the depth of
        // the frontmost splat with alpha > alphaThreshold, or farBig() if
        // none qualifies. G/B/A are unused.
        fragColor = vec4(a > alphaThreshold ? d : farBig(), 0.0, 0.0, 0.0);
    } else {
        // Weighted-average pass: standard premultiplied "over" compositing
        // (blend ONE/ONE_MINUS_SRC_ALPHA on both color and alpha - i.e. the
        // same blend THREE's NormalBlending uses when premultipliedAlpha is
        // true, exactly like splatFragment.glsl's PREMULTIPLIED_ALPHA
        // branch). Splats are drawn back-to-front (shared ordering texture),
        // so this correctly weighs each splat by true visibility
        // w_i = a_i * product((1 - a_j) for j in front of i), not a naive
        // sum: after compositing every splat covering this pixel,
        // R = sum(w*d) and A = sum(w). dMedia = R/A.
        fragColor = vec4(d * a, 0.0, 0.0, a);
    }

    #include <logdepthbuf_fragment>
}

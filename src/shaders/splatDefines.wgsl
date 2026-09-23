// splatDefines.wgsl - WGSL port of splatDefines.glsl
// Spark.js 3D Gaussian Splatting renderer - WebGPU shader defines
//
// Contains constants, packing/unpacking utilities, quaternion math, and
// splat encoding/decoding functions for the Gaussian splatting pipeline.

// ============================================================================
// Constants
// ============================================================================

const LN_SCALE_MIN: f32 = -12.0;
const LN_SCALE_MAX: f32 = 9.0;

const SPLAT_TEX_WIDTH_BITS: u32 = 11u;
const SPLAT_TEX_HEIGHT_BITS: u32 = 11u;
const SPLAT_TEX_DEPTH_BITS: u32 = 11u;
const SPLAT_TEX_LAYER_BITS: u32 = SPLAT_TEX_WIDTH_BITS + SPLAT_TEX_HEIGHT_BITS;

const SPLAT_TEX_WIDTH: u32 = 1u << SPLAT_TEX_WIDTH_BITS;
const SPLAT_TEX_HEIGHT: u32 = 1u << SPLAT_TEX_HEIGHT_BITS;
const SPLAT_TEX_DEPTH: u32 = 1u << SPLAT_TEX_DEPTH_BITS;

const SPLAT_TEX_WIDTH_MASK: u32 = SPLAT_TEX_WIDTH - 1u;
const SPLAT_TEX_HEIGHT_MASK: u32 = SPLAT_TEX_HEIGHT - 1u;
const SPLAT_TEX_DEPTH_MASK: u32 = SPLAT_TEX_DEPTH - 1u;

const F16_INF: u32 = 0x7c00u;
const PI: f32 = 3.1415926535897932384626433832795;

// naga doesn't support bitcast in const expressions, so use large sentinel values.
// These are used as "infinity" markers in splat depth output; actual IEEE inf is
// produced at runtime via bitcast<f32>(0x7F800000u) where needed.
const INFINITY_BITS: u32 = 0x7F800000u;
const NEG_INFINITY_BITS: u32 = 0xFF800000u;

// ============================================================================
// Result structs for functions that replace GLSL out parameters
// ============================================================================

struct UnpackedSplat {
    center: vec3f,
    scales: vec3f,
    quaternion: vec4f,
    rgba: vec4f,
}

struct UnpackedCovSplat {
    center: vec3f,
    rgba: vec4f,
    xxyyzz: vec3f,
    xyxzyz: vec3f,
}

struct PackedSplatExt {
    packed: vec4u,
    packed2: vec4u,
}

// ============================================================================
// Simple math utilities
// ============================================================================

fn sqr(x: f32) -> f32 {
    return x * x;
}

fn pow4(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2;
}

fn pow8(x: f32) -> f32 {
    let x4 = pow4(x);
    return x4 * x4;
}

fn srgbToLinear(rgb: vec3f) -> vec3f {
    return pow(rgb, vec3f(2.2));
}

fn linearToSrgb(rgb: vec3f) -> vec3f {
    return pow(rgb, vec3f(1.0 / 2.2));
}

// ============================================================================
// Texture coordinate helpers
// ============================================================================

fn splatTexCoord(index: i32) -> vec3i {
    let x = u32(index) & SPLAT_TEX_WIDTH_MASK;
    let y = (u32(index) >> SPLAT_TEX_WIDTH_BITS) & SPLAT_TEX_HEIGHT_MASK;
    let z = u32(index) >> SPLAT_TEX_LAYER_BITS;
    return vec3i(vec3u(x, y, z));
}

fn pagedSplatTexCoord(index: i32) -> vec3i {
    return vec3i(index & 255, (index >> 8) & 255, index >> 16);
}

// ============================================================================
// Byte conversion utilities
// ============================================================================

fn uintToVec4(val: u32) -> vec4f {
    let bytes = vec4u(
        val & 0xFFu,
        (val >> 8u) & 0xFFu,
        (val >> 16u) & 0xFFu,
        (val >> 24u) & 0xFFu
    );
    return vec4f(bytes) / 255.0;
}

fn floatToVec4(f: f32) -> vec4f {
    let val = bitcast<u32>(f);
    return uintToVec4(val);
}

fn debugColorHue(i: u32) -> vec3f {
    // Golden ratio conjugate; spreads hues evenly
    let hue = fract(f32(i) * 0.61803398875);
    // HSV to RGB with fixed S/V
    // mod(hue*6.0 + vec3(0,4,2), 6.0) in GLSL -> fmod for floats
    let h6 = hue * 6.0 + vec3f(0.0, 4.0, 2.0);
    let h6mod = h6 - 6.0 * floor(h6 / 6.0);
    let rgb = clamp(abs(h6mod - 3.0) - 1.0, vec3f(0.0), vec3f(1.0));
    return mix(vec3f(1.0), rgb, vec3f(0.85)); // saturation ~0.85, value ~1.0
}

// ============================================================================
// Quaternion math
// ============================================================================

// Rotate vector v by quaternion q
fn quatVec(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Apply quaternion q1 after quaternion q2
fn quatQuat(q1: vec4f, q2: vec4f) -> vec4f {
    return vec4f(
        q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    );
}

// NOTE: WGSL mat3x3f is constructed from 3 column vectors, matching GLSL mat3
// which also takes values in column-major order (column by column).
// GLSL mat3(c0r0, c0r1, c0r2, c1r0, c1r1, c1r2, c2r0, c2r1, c2r2)
// = WGSL mat3x3f(vec3f(c0r0, c0r1, c0r2), vec3f(c1r0, c1r1, c1r2), vec3f(c2r0, c2r1, c2r2))
fn quaternionToMatrix(q: vec4f) -> mat3x3f {
    return mat3x3f(
        vec3f(
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            2.0 * (q.x * q.y + q.w * q.z),
            2.0 * (q.x * q.z - q.w * q.y)
        ),
        vec3f(
            2.0 * (q.x * q.y - q.w * q.z),
            1.0 - 2.0 * (q.x * q.x + q.z * q.z),
            2.0 * (q.y * q.z + q.w * q.x)
        ),
        vec3f(
            2.0 * (q.x * q.z + q.w * q.y),
            2.0 * (q.y * q.z - q.w * q.x),
            1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        )
    );
}

// Compute the matrix of scaling by s then rotating by q
fn scaleQuaternionToMatrix(s: vec3f, q: vec4f) -> mat3x3f {
    return mat3x3f(
        vec3f(
            s.x * (1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
            s.x * (2.0 * (q.x * q.y + q.w * q.z)),
            s.x * (2.0 * (q.x * q.z - q.w * q.y))
        ),
        vec3f(
            s.y * (2.0 * (q.x * q.y - q.w * q.z)),
            s.y * (1.0 - 2.0 * (q.x * q.x + q.z * q.z)),
            s.y * (2.0 * (q.y * q.z + q.w * q.x))
        ),
        vec3f(
            s.z * (2.0 * (q.x * q.z + q.w * q.y)),
            s.z * (2.0 * (q.y * q.z - q.w * q.x)),
            s.z * (1.0 - 2.0 * (q.x * q.x + q.y * q.y))
        )
    );
}

// Spherical lerp between two quaternions
fn slerp(q1: vec4f, q2_in: vec4f, t: f32) -> vec4f {
    // Compute the cosine of the angle between the two vectors
    var cosHalfTheta = dot(q1, q2_in);

    // If q1=q2 or q1=-q2 then theta = 0 and we can return q1
    if (abs(cosHalfTheta) >= 0.999) {
        return q1;
    }

    // If q1 and q2 are more than 180 degrees apart,
    // we need to negate one to get the shortest path
    var q2 = q2_in;
    if (cosHalfTheta < 0.0) {
        q2 = -q2;
        cosHalfTheta = -cosHalfTheta;
    }

    // Calculate temporary values
    let halfTheta = acos(cosHalfTheta);
    let sinHalfTheta = sqrt(1.0 - cosHalfTheta * cosHalfTheta);

    // Calculate the interpolation factors
    let ratioA = sin((1.0 - t) * halfTheta) / sinHalfTheta;
    let ratioB = sin(t * halfTheta) / sinHalfTheta;

    // Calculate the interpolated quaternion
    return q1 * ratioA + q2 * ratioB;
}

// ============================================================================
// Quaternion encoding - Octahedral XY 8-8 with 8-bit rotation angle
// ============================================================================

// Spark's CPU encoders use Math.round, whose nonnegative half ties round up.
// WGSL round uses ties-to-even, changing octahedral bytes at 180-degree rotations.
// Compare the fractional part instead of adding 0.5, which can itself round a
// value just below a half tie upward in f32 arithmetic.
fn roundPositive(value: f32) -> f32 {
    return floor(value) + select(0.0, 1.0, fract(value) >= 0.5);
}

// Encode a quaternion (vec4f) into a 24-bit uint with folded octahedral mapping.
fn encodeQuatOctXy88R8(q_in: vec4f) -> u32 {
    // Ensure minimal representation: flip if q.w is negative.
    var q = q_in;
    if (q.w < 0.0) {
        q = -q;
    }
    // Compute rotation angle: theta = 2 * acos(q.w) in [0,pi]
    let theta = 2.0 * acos(q.w);
    let halfTheta = theta * 0.5;
    let s = sin(halfTheta);
    // Recover the rotation axis; use a default if nearly zero rotation.
    var axis: vec3f;
    if (abs(s) < 1e-6) {
        axis = vec3f(1.0, 0.0, 0.0);
    } else {
        axis = q.xyz / s;
    }

    // --- Folded Octahedral Mapping (inline) ---
    let sum = abs(axis.x) + abs(axis.y) + abs(axis.z);
    var p = vec2f(axis.x, axis.y) / sum;
    // If axis.z < 0, fold the mapping.
    if (axis.z < 0.0) {
        let oldPx = p.x;
        p.x = (1.0 - abs(p.y)) * select(-1.0, 1.0, p.x >= 0.0);
        p.y = (1.0 - abs(oldPx)) * select(-1.0, 1.0, p.y >= 0.0);
    }
    // Remap from [-1,1] to [0,1]
    let u_f = p.x * 0.5 + 0.5;
    let v_f = p.y * 0.5 + 0.5;
    // Quantize to 8 bits (0 to 255)
    let quantU = u32(clamp(roundPositive(u_f * 255.0), 0.0, 255.0));
    let quantV = u32(clamp(roundPositive(v_f * 255.0), 0.0, 255.0));

    // --- Angle Quantization ---
    // Quantize theta in [0,pi] to 8 bits (0 to 255)
    let angleInt = u32(clamp(roundPositive((theta / 3.14159265359) * 255.0), 0.0, 255.0));

    // Pack bits: bits [0-7]: quantU, [8-15]: quantV, [16-23]: angleInt.
    return (angleInt << 16u) | (quantV << 8u) | quantU;
}

// Decode a 24-bit encoded uint into a quaternion (vec4f) using the folded octahedral inverse.
fn decodeQuatOctXy88R8(encoded: u32) -> vec4f {
    // Extract the fields.
    let quantU = encoded & 0xFFu;                // bits 0-7
    let quantV = (encoded >> 8u) & 0xFFu;        // bits 8-15
    let angleInt = encoded >> 16u;                // bits 16-23

    // Recover u and v in [0,1], then map to [-1,1].
    let u_f = f32(quantU) / 255.0;
    let v_f = f32(quantV) / 255.0;
    let f = vec2f(u_f * 2.0 - 1.0, v_f * 2.0 - 1.0);

    var axis = vec3f(f.xy, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-axis.z, 0.0);
    axis.x += select(t, -t, axis.x >= 0.0);
    axis.y += select(t, -t, axis.y >= 0.0);
    axis = normalize(axis);

    // Decode the angle theta in [0,pi].
    let theta = (f32(angleInt) / 255.0) * 3.14159265359;
    let halfTheta = theta * 0.5;
    let s = sin(halfTheta);
    let w = cos(halfTheta);

    return vec4f(axis * s, w);
}

// ============================================================================
// Quaternion encoding - Octahedral XY 10-10 with 12-bit rotation angle
// ============================================================================

fn encodeQuatOctXy1010R12(q_in: vec4f) -> u32 {
    // Ensure minimal representation: flip if q.w is negative.
    var q = q_in;
    if (q.w < 0.0) {
        q = -q;
    }
    // Compute rotation angle: theta = 2 * acos(q.w) in [0,pi]
    let halfTheta = acos(q.w);
    let theta = 2.0 * halfTheta;
    let s = sin(halfTheta);
    // Recover the rotation axis; use a default if nearly zero rotation.
    var axis: vec3f;
    if (abs(s) < 1e-6) {
        axis = vec3f(1.0, 0.0, 0.0);
    } else {
        axis = q.xyz / s;
    }

    // --- Folded Octahedral Mapping (inline) ---
    let sum = abs(axis.x) + abs(axis.y) + abs(axis.z);
    var p = vec2f(axis.x, axis.y) / sum;
    // If axis.z < 0, fold the mapping.
    if (axis.z < 0.0) {
        let oldPx = p.x;
        p.x = (1.0 - abs(p.y)) * select(-1.0, 1.0, p.x >= 0.0);
        p.y = (1.0 - abs(oldPx)) * select(-1.0, 1.0, p.y >= 0.0);
    }
    // Remap from [-1,1] to [0,1]
    let u_f = p.x * 0.5 + 0.5;
    let v_f = p.y * 0.5 + 0.5;
    // Quantize to 10 bits (0 to 1023)
    let quantU = u32(clamp(roundPositive(u_f * 1023.0), 0.0, 1023.0));
    let quantV = u32(clamp(roundPositive(v_f * 1023.0), 0.0, 1023.0));

    // --- Angle Quantization ---
    // Quantize theta in [0,pi] to 12 bits (0 to 4095)
    let angleInt = u32(clamp(roundPositive((theta / PI) * 4095.0), 0.0, 4095.0));

    // Pack bits: bits [0-9]: quantU, [10-19]: quantV, [20-31]: angleInt.
    return (angleInt << 20u) | (quantV << 10u) | quantU;
}

fn decodeQuatOctXy1010R12(encoded: u32) -> vec4f {
    // Extract the fields.
    let quantU = encoded & 0x3FFu;               // bits 0-9
    let quantV = (encoded >> 10u) & 0x3FFu;      // bits 10-19
    let angleInt = encoded >> 20u;                // bits 20-31

    // Recover u and v in [0,1], then map to [-1,1].
    let u_f = f32(quantU) / 1023.0;
    let v_f = f32(quantV) / 1023.0;
    let f = vec2f(u_f * 2.0 - 1.0, v_f * 2.0 - 1.0);

    var axis = vec3f(f.xy, 1.0 - abs(f.x) - abs(f.y));
    let t = max(-axis.z, 0.0);
    axis.x += select(t, -t, axis.x >= 0.0);
    axis.y += select(t, -t, axis.y >= 0.0);
    axis = normalize(axis);

    // Decode the angle theta in [0,pi].
    let theta = (f32(angleInt) / 4095.0) * PI;
    let halfTheta = theta * 0.5;
    let s = sin(halfTheta);
    let w = cos(halfTheta);

    return vec4f(axis * s, w);
}

// ============================================================================
// Splat packing (standard 16-byte encoding)
// ============================================================================

// Pack a Gsplat into a vec4u
fn packSplatEncoding(
    center: vec3f, scales: vec3f, quaternion: vec4f, rgba: vec4f,
    rgbMinMaxLnScaleMinMax: vec4f
) -> vec4u {
    let rgbMin = rgbMinMaxLnScaleMinMax.x;
    let rgbMax = rgbMinMaxLnScaleMinMax.y;
    let encRgb = (rgba.rgb - vec3f(rgbMin)) / (rgbMax - rgbMin);
    let uRgba = vec4u(round(clamp(vec4f(encRgb, rgba.a) * 255.0, vec4f(0.0), vec4f(255.0))));

    let uQuat = encodeQuatOctXy88R8(quaternion);
    let uQuat3 = vec3u(uQuat & 0xFFu, (uQuat >> 8u) & 0xFFu, (uQuat >> 16u) & 0xFFu);

    // Encode scales in three uint8s, where 0=>0.0 and 1..=255 stores log scale
    let lnScaleMin = rgbMinMaxLnScaleMinMax.z;
    let lnScaleMax = rgbMinMaxLnScaleMinMax.w;
    let lnScaleScale = 254.0 / (lnScaleMax - lnScaleMin);
    let uScales = vec3u(
        select(u32(round(clamp((log(scales.x) - lnScaleMin) * lnScaleScale, 0.0, 254.0))) + 1u, 0u, scales.x == 0.0),
        select(u32(round(clamp((log(scales.y) - lnScaleMin) * lnScaleScale, 0.0, 254.0))) + 1u, 0u, scales.y == 0.0),
        select(u32(round(clamp((log(scales.z) - lnScaleMin) * lnScaleScale, 0.0, 254.0))) + 1u, 0u, scales.z == 0.0)
    );

    // Pack it all into 4 x u32
    let word0 = uRgba.r | (uRgba.g << 8u) | (uRgba.b << 16u) | (uRgba.a << 24u);
    let word1 = pack2x16float(center.xy);
    let word2 = pack2x16float(vec2f(center.z, 0.0)) | (uQuat3.x << 16u) | (uQuat3.y << 24u);
    let word3 = uScales.x | (uScales.y << 8u) | (uScales.z << 16u) | (uQuat3.z << 24u);
    return vec4u(word0, word1, word2, word3);
}

// Pack a Gsplat into a vec4u (convenience wrapper with default encoding range)
fn packSplat(center: vec3f, scales: vec3f, quaternion: vec4f, rgba: vec4f) -> vec4u {
    return packSplatEncoding(center, scales, quaternion, rgba, vec4f(0.0, 1.0, LN_SCALE_MIN, LN_SCALE_MAX));
}

// ============================================================================
// Splat unpacking (standard 16-byte encoding)
// ============================================================================

fn unpackSplatEncoding(packed: vec4u, rgbMinMaxLnScaleMinMax: vec4f) -> UnpackedSplat {
    var result: UnpackedSplat;

    let word0 = packed.x;
    let word1 = packed.y;
    let word2 = packed.z;
    let word3 = packed.w;

    let uRgba = vec4u(word0 & 0xFFu, (word0 >> 8u) & 0xFFu, (word0 >> 16u) & 0xFFu, (word0 >> 24u) & 0xFFu);
    let rgbMin = rgbMinMaxLnScaleMinMax.x;
    let rgbMax = rgbMinMaxLnScaleMinMax.y;
    var rgba = vec4f(uRgba) / 255.0;
    rgba = vec4f(rgba.rgb * (rgbMax - rgbMin) + rgbMin, rgba.a);
    result.rgba = rgba;

    // Unpack center from two half2x16 values
    let xy = unpack2x16float(word1);
    let zw = unpack2x16float(word2 & 0xFFFFu);
    result.center = vec3f(xy.x, xy.y, zw.x);

    let uScales = vec3u(word3 & 0xFFu, (word3 >> 8u) & 0xFFu, (word3 >> 16u) & 0xFFu);
    let lnScaleMin = rgbMinMaxLnScaleMinMax.z;
    let lnScaleMax = rgbMinMaxLnScaleMinMax.w;
    let lnScaleScale = (lnScaleMax - lnScaleMin) / 254.0;
    result.scales = vec3f(
        select(exp(lnScaleMin + f32(uScales.x - 1u) * lnScaleScale), 0.0, uScales.x == 0u),
        select(exp(lnScaleMin + f32(uScales.y - 1u) * lnScaleScale), 0.0, uScales.y == 0u),
        select(exp(lnScaleMin + f32(uScales.z - 1u) * lnScaleScale), 0.0, uScales.z == 0u)
    );

    let uQuat = ((word2 >> 16u) & 0xFFFFu) | ((word3 >> 8u) & 0xFF0000u);
    result.quaternion = decodeQuatOctXy88R8(uQuat);

    return result;
}

// Unpack a Gsplat from a vec4u (convenience wrapper with default encoding range)
fn unpackSplat(packed: vec4u) -> UnpackedSplat {
    return unpackSplatEncoding(packed, vec4f(0.0, 1.0, LN_SCALE_MIN, LN_SCALE_MAX));
}

// ============================================================================
// Splat covariance packing (standard 16-byte encoding)
// ============================================================================

fn packSplatCovEncoding(
    center: vec3f, rgba: vec4f, xxyyzz: vec3f, xyxzyz: vec3f,
    rgbMinMaxLnScaleMinMax: vec4f
) -> vec4u {
    let rgbMin = rgbMinMaxLnScaleMinMax.x;
    let rgbMax = rgbMinMaxLnScaleMinMax.y;
    let encRgb = (rgba.rgb - vec3f(rgbMin)) / (rgbMax - rgbMin);
    let uRgba = vec4u(round(clamp(vec4f(encRgb, rgba.a) * 255.0, vec4f(0.0), vec4f(255.0))));

    let lnScaleMin = rgbMinMaxLnScaleMinMax.z;
    let lnScaleMax = rgbMinMaxLnScaleMinMax.w;
    let diagScale = 255.0 / (2.0 * (lnScaleMax - lnScaleMin));
    let uXxyyzz = vec3u(round(clamp((log(xxyyzz) - 2.0 * lnScaleMin) * diagScale, vec3f(0.0), vec3f(255.0))));

    let xyxzyzCor = vec3f(
        clamp(xyxzyz.x / sqrt(xxyyzz.x * xxyyzz.y), -1.0, 1.0),
        clamp(xyxzyz.y / sqrt(xxyyzz.x * xxyyzz.z), -1.0, 1.0),
        clamp(xyxzyz.z / sqrt(xxyyzz.y * xxyyzz.z), -1.0, 1.0)
    );
    let iXyxzyzCor = vec3i(round(xyxzyzCor * 127.0));

    // Pack it all into 4 x u32
    let word0 = uRgba.r | (uRgba.g << 8u) | (uRgba.b << 16u) | (uRgba.a << 24u);
    let word1 = pack2x16float(center.xy);
    let word2 = pack2x16float(vec2f(center.z, 0.0)) |
        ((u32(iXyxzyzCor.y) & 0xFFu) << 16u) |
        ((u32(iXyxzyzCor.z) & 0xFFu) << 24u);
    let word3 =
        uXxyyzz.x | (uXxyyzz.y << 8u) | (uXxyyzz.z << 16u) |
        ((u32(iXyxzyzCor.x) & 0xFFu) << 24u);
    return vec4u(word0, word1, word2, word3);
}

fn unpackSplatCovEncoding(packed: vec4u, rgbMinMaxLnScaleMinMax: vec4f) -> UnpackedCovSplat {
    var result: UnpackedCovSplat;

    let word0 = packed.x;
    let word1 = packed.y;
    let word2 = packed.z;
    let word3 = packed.w;

    let uRgba = vec4u(word0 & 0xFFu, (word0 >> 8u) & 0xFFu, (word0 >> 16u) & 0xFFu, (word0 >> 24u) & 0xFFu);
    let rgbMin = rgbMinMaxLnScaleMinMax.x;
    let rgbMax = rgbMinMaxLnScaleMinMax.y;
    var rgba = vec4f(uRgba) / 255.0;
    rgba = vec4f(rgba.rgb * (rgbMax - rgbMin) + rgbMin, rgba.a);
    result.rgba = rgba;

    let xy = unpack2x16float(word1);
    let zw = unpack2x16float(word2 & 0xFFFFu);
    result.center = vec3f(xy.x, xy.y, zw.x);

    let uXxyyzz = vec3u(word3 & 0xFFu, (word3 >> 8u) & 0xFFu, (word3 >> 16u) & 0xFFu);
    // Sign-extending i8 values from packed bytes:
    // int(word3) >> 24 -> sign-extends the top byte
    // int(word2 << 8u) >> 24 -> sign-extends bits [16-23]
    // int(word2) >> 24 -> sign-extends the top byte
    let iXyxzyzCor = vec3i(i32(word3) >> 24u, i32(word2 << 8u) >> 24u, i32(word2) >> 24u);

    let lnScaleMin = rgbMinMaxLnScaleMinMax.z;
    let lnScaleMax = rgbMinMaxLnScaleMinMax.w;
    let diagScale = 2.0 * (lnScaleMax - lnScaleMin) / 255.0;
    result.xxyyzz = exp(2.0 * lnScaleMin + vec3f(uXxyyzz) * diagScale);

    let xyxzyzCor = vec3f(iXyxzyzCor) / 127.0;
    result.xyxzyz = xyxzyzCor * vec3f(
        sqrt(result.xxyyzz.x * result.xxyyzz.y),
        sqrt(result.xxyyzz.x * result.xxyyzz.z),
        sqrt(result.xxyyzz.y * result.xxyyzz.z)
    );

    return result;
}

// ============================================================================
// Extended splat packing (32-byte encoding, scale/quaternion)
// ============================================================================

fn packSplatExt(
    center: vec3f, scales: vec3f, quaternion: vec4f, rgba: vec4f
) -> PackedSplatExt {
    var result: PackedSplatExt;

    result.packed.x = bitcast<u32>(center.x);
    result.packed.y = bitcast<u32>(center.y);
    result.packed.z = bitcast<u32>(center.z);
    result.packed.w = pack2x16float(vec2f(rgba.a, 0.0));

    result.packed2.x = pack2x16float(rgba.rg);
    result.packed2.y = pack2x16float(vec2f(rgba.b, log(scales.x)));
    result.packed2.z = pack2x16float(log(scales.yz));
    result.packed2.w = encodeQuatOctXy1010R12(quaternion);

    return result;
}

fn unpackSplatExtCenterAlpha(packed: vec4u) -> vec4f {
    return vec4f(
        bitcast<f32>(packed.x),
        bitcast<f32>(packed.y),
        bitcast<f32>(packed.z),
        unpack2x16float(packed.w).x
    );
}

fn unpackSplatExtAlpha(packed: vec4u) -> f32 {
    return unpack2x16float(packed.w).x;
}

fn unpackSplatExt(packed: vec4u, packed2: vec4u) -> UnpackedSplat {
    var result: UnpackedSplat;

    result.center.x = bitcast<f32>(packed.x);
    result.center.y = bitcast<f32>(packed.y);
    result.center.z = bitcast<f32>(packed.z);
    result.rgba.a = unpack2x16float(packed.w).x;

    result.rgba.r = unpack2x16float(packed2.x).x;
    result.rgba.g = unpack2x16float(packed2.x).y;
    let split = unpack2x16float(packed2.y);
    result.rgba.b = split.x;
    result.scales.x = exp(split.y);
    let yz = unpack2x16float(packed2.z);
    result.scales.y = exp(yz.x);
    result.scales.z = exp(yz.y);
    result.quaternion = decodeQuatOctXy1010R12(packed2.w);

    return result;
}

// ============================================================================
// Extended splat covariance packing (32-byte encoding)
// ============================================================================

fn packSplatExtCov(
    center: vec3f, rgba: vec4f, xxyyzz: vec3f, xyxzyz: vec3f
) -> PackedSplatExt {
    var result: PackedSplatExt;

    result.packed.x = bitcast<u32>(center.x);
    result.packed.y = bitcast<u32>(center.y);
    result.packed.z = bitcast<u32>(center.z);
    result.packed.w = pack2x16float(vec2f(rgba.a, rgba.b));
    result.packed2.x = pack2x16float(rgba.rg);

    var xyxzyzCor = vec3f(
        clamp(xyxzyz.x / sqrt(xxyyzz.x * xxyyzz.y), -1.0, 1.0),
        clamp(xyxzyz.y / sqrt(xxyyzz.x * xxyyzz.z), -1.0, 1.0),
        clamp(xyxzyz.z / sqrt(xxyyzz.y * xxyyzz.z), -1.0, 1.0)
    );
    xyxzyzCor = sign(xyxzyzCor) * clamp(log(abs(xyxzyzCor)), vec3f(-100.0), vec3f(-0.0000001));
    let logXxyyzz = log(xxyyzz);

    result.packed2.y = pack2x16float(vec2f(logXxyyzz.x, logXxyyzz.y));
    result.packed2.z = pack2x16float(vec2f(logXxyyzz.z, xyxzyzCor.x));
    result.packed2.w = pack2x16float(vec2f(xyxzyzCor.y, xyxzyzCor.z));

    return result;
}

fn unpackSplatExtCov(packed: vec4u, packed2: vec4u) -> UnpackedCovSplat {
    var result: UnpackedCovSplat;

    result.center.x = bitcast<f32>(packed.x);
    result.center.y = bitcast<f32>(packed.y);
    result.center.z = bitcast<f32>(packed.z);

    let ab = unpack2x16float(packed.w);
    let rg = unpack2x16float(packed2.x);
    result.rgba = vec4f(rg, ab.y, ab.x);

    let xxyy = unpack2x16float(packed2.y);
    let zzxy = unpack2x16float(packed2.z);
    let xzyz = unpack2x16float(packed2.w);
    result.xxyyzz = exp(vec3f(xxyy.x, xxyy.y, zzxy.x));
    var xyxzyz_val = vec3f(zzxy.y, xzyz.x, xzyz.y);
    xyxzyz_val = -sign(xyxzyz_val) * exp(-abs(xyxzyz_val));
    result.xyxzyz = xyxzyz_val * vec3f(
        sqrt(result.xxyyzz.x * result.xxyyzz.y),
        sqrt(result.xxyyzz.x * result.xxyyzz.z),
        sqrt(result.xxyyzz.y * result.xxyyzz.z)
    );

    return result;
}

// ============================================================================
// Extended RGB encoding (HDR color in 32 bits)
// ============================================================================

fn encodeExtRgb(rgb: vec3f) -> u32 {
    let absRgb = abs(rgb);
    let maxAbs = max(absRgb.r, max(absRgb.g, absRgb.b));

    let base = clamp(i32(floor(log2(maxAbs))) + 15, 0, 31);
    let divisor = exp2(f32(base - 15)) / 255.0;

    let uRgb = vec3u(round(clamp(absRgb / divisor, vec3f(0.0), vec3f(255.0))));
    let expSigns = (u32(base) << 3u) |
        (select(0u, 0x1u, rgb.r < 0.0)) |
        (select(0u, 0x2u, rgb.g < 0.0)) |
        (select(0u, 0x4u, rgb.b < 0.0));
    return uRgb.r | (uRgb.g << 8u) | (uRgb.b << 16u) | (expSigns << 24u);
}

fn decodeExtRgb(encoded: u32) -> vec3f {
    let biasedBase = (encoded >> 27u) & 0x1Fu;
    let divisor = exp2(f32(i32(biasedBase) - 15)) / 255.0;

    var rgb = vec3f(vec3u(encoded & 0xFFu, (encoded >> 8u) & 0xFFu, (encoded >> 16u) & 0xFFu));
    rgb *= divisor;

    return vec3f(
        select(rgb.r, -rgb.r, (encoded & 0x1000000u) != 0u),
        select(rgb.g, -rgb.g, (encoded & 0x2000000u) != 0u),
        select(rgb.b, -rgb.b, (encoded & 0x4000000u) != 0u)
    );
}

// ============================================================================
// Spherical Harmonics evaluation (packed format)
// ============================================================================

fn evaluatePackedSH1(packed: vec2u, viewDir: vec3f, sh1Max: f32) -> vec3f {
    // Extract sint7 values packed into 2 x uint32
    let sh1_0 = vec3f(vec3i(
        i32(packed.x << 25u) >> 25,
        i32(packed.x << 18u) >> 25,
        i32(packed.x << 11u) >> 25
    ));
    let sh1_1 = vec3f(vec3i(
        i32(packed.x << 4u) >> 25,
        i32((packed.x >> 3u) | (packed.y << 29u)) >> 25,
        i32(packed.y << 22u) >> 25
    ));
    let sh1_2 = vec3f(vec3i(
        i32(packed.y << 15u) >> 25,
        i32(packed.y << 8u) >> 25,
        i32(packed.y << 1u) >> 25
    ));

    let rgb = sh1_0 * (-0.4886025 * viewDir.y)
        + sh1_1 * (0.4886025 * viewDir.z)
        + sh1_2 * (-0.4886025 * viewDir.x);
    return rgb * (sh1Max / 63.0);
}

fn evaluatePackedSH2(packed: vec4u, viewDir: vec3f, sh2Max: f32) -> vec3f {
    // Extract sint8 values packed into 4 x uint32
    let sh2_0 = vec3f(vec3i(
        i32(packed.x << 24u) >> 24,
        i32(packed.x << 16u) >> 24,
        i32(packed.x << 8u) >> 24
    ));
    let sh2_1 = vec3f(vec3i(
        i32(packed.x) >> 24,
        i32(packed.y << 24u) >> 24,
        i32(packed.y << 16u) >> 24
    ));
    let sh2_2 = vec3f(vec3i(
        i32(packed.y << 8u) >> 24,
        i32(packed.y) >> 24,
        i32(packed.z << 24u) >> 24
    ));
    let sh2_3 = vec3f(vec3i(
        i32(packed.z << 16u) >> 24,
        i32(packed.z << 8u) >> 24,
        i32(packed.z) >> 24
    ));
    let sh2_4 = vec3f(vec3i(
        i32(packed.w << 24u) >> 24,
        i32(packed.w << 16u) >> 24,
        i32(packed.w << 8u) >> 24
    ));

    let rgb = sh2_0 * (1.0925484 * viewDir.x * viewDir.y)
        + sh2_1 * (-1.0925484 * viewDir.y * viewDir.z)
        + sh2_2 * (0.3153915 * (2.0 * viewDir.z * viewDir.z - viewDir.x * viewDir.x - viewDir.y * viewDir.y))
        + sh2_3 * (-1.0925484 * viewDir.x * viewDir.z)
        + sh2_4 * (0.5462742 * (viewDir.x * viewDir.x - viewDir.y * viewDir.y));
    return rgb * (sh2Max / 127.0);
}

fn evaluatePackedSH3(packed: vec4u, viewDir: vec3f, sh3Max: f32) -> vec3f {
    // Extract sint6 values packed into 4 x uint32
    let sh3_0 = vec3f(vec3i(
        i32(packed.x << 26u) >> 26,
        i32(packed.x << 20u) >> 26,
        i32(packed.x << 14u) >> 26
    ));
    let sh3_1 = vec3f(vec3i(
        i32(packed.x << 8u) >> 26,
        i32(packed.x << 2u) >> 26,
        i32((packed.x >> 4u) | (packed.y << 28u)) >> 26
    ));
    let sh3_2 = vec3f(vec3i(
        i32(packed.y << 22u) >> 26,
        i32(packed.y << 16u) >> 26,
        i32(packed.y << 10u) >> 26
    ));
    let sh3_3 = vec3f(vec3i(
        i32(packed.y << 4u) >> 26,
        i32((packed.y >> 2u) | (packed.z << 30u)) >> 26,
        i32(packed.z << 24u) >> 26
    ));
    let sh3_4 = vec3f(vec3i(
        i32(packed.z << 18u) >> 26,
        i32(packed.z << 12u) >> 26,
        i32(packed.z << 6u) >> 26
    ));
    let sh3_5 = vec3f(vec3i(
        i32(packed.z) >> 26,
        i32(packed.w << 26u) >> 26,
        i32(packed.w << 20u) >> 26
    ));
    let sh3_6 = vec3f(vec3i(
        i32(packed.w << 14u) >> 26,
        i32(packed.w << 8u) >> 26,
        i32(packed.w << 2u) >> 26
    ));

    let xx = viewDir.x * viewDir.x;
    let yy = viewDir.y * viewDir.y;
    let zz = viewDir.z * viewDir.z;
    let xy = viewDir.x * viewDir.y;

    let rgb = sh3_0 * (-0.5900436 * viewDir.y * (3.0 * xx - yy))
        + sh3_1 * (2.8906114 * xy * viewDir.z)
        + sh3_2 * (-0.4570458 * viewDir.y * (4.0 * zz - xx - yy))
        + sh3_3 * (0.3731763 * viewDir.z * (2.0 * zz - 3.0 * xx - 3.0 * yy))
        + sh3_4 * (-0.4570458 * viewDir.x * (4.0 * zz - xx - yy))
        + sh3_5 * (1.4453057 * viewDir.z * (xx - yy))
        + sh3_6 * (-0.5900436 * viewDir.x * (xx - 3.0 * yy));
    return rgb * (sh3Max / 31.0);
}

// ============================================================================
// Spherical Harmonics evaluation (extended format)
// ============================================================================

fn evaluateExtSH1(packed: vec4u, viewDir: vec3f) -> vec3f {
    let sh1_0 = decodeExtRgb(packed.x);
    let sh1_1 = decodeExtRgb(packed.y);
    let sh1_2 = decodeExtRgb(packed.z);

    return sh1_0 * (-0.4886025 * viewDir.y)
        + sh1_1 * (0.4886025 * viewDir.z)
        + sh1_2 * (-0.4886025 * viewDir.x);
}

fn evaluateExtSH12(packed1: vec4u, packed2: vec4u, viewDir: vec3f) -> vec3f {
    let sh1_0 = decodeExtRgb(packed1.x);
    let sh1_1 = decodeExtRgb(packed1.y);
    let sh1_2 = decodeExtRgb(packed1.z);

    let sh2_0 = decodeExtRgb(packed1.w);
    let sh2_1 = decodeExtRgb(packed2.x);
    let sh2_2 = decodeExtRgb(packed2.y);
    let sh2_3 = decodeExtRgb(packed2.z);
    let sh2_4 = decodeExtRgb(packed2.w);

    let sh1Rgb = sh1_0 * (-0.4886025 * viewDir.y)
        + sh1_1 * (0.4886025 * viewDir.z)
        + sh1_2 * (-0.4886025 * viewDir.x);

    let sh2Rgb = sh2_0 * (1.0925484 * viewDir.x * viewDir.y)
        + sh2_1 * (-1.0925484 * viewDir.y * viewDir.z)
        + sh2_2 * (0.3153915 * (2.0 * viewDir.z * viewDir.z - viewDir.x * viewDir.x - viewDir.y * viewDir.y))
        + sh2_3 * (-1.0925484 * viewDir.x * viewDir.z)
        + sh2_4 * (0.5462742 * (viewDir.x * viewDir.x - viewDir.y * viewDir.y));

    return sh1Rgb + sh2Rgb;
}

fn evaluateExtSH3(packedA: vec4u, packedB: vec4u, viewDir: vec3f) -> vec3f {
    let sh3_0 = decodeExtRgb(packedA.x);
    let sh3_1 = decodeExtRgb(packedA.y);
    let sh3_2 = decodeExtRgb(packedA.z);
    let sh3_3 = decodeExtRgb(packedA.w);
    let sh3_4 = decodeExtRgb(packedB.x);
    let sh3_5 = decodeExtRgb(packedB.y);
    let sh3_6 = decodeExtRgb(packedB.z);

    let xx = viewDir.x * viewDir.x;
    let yy = viewDir.y * viewDir.y;
    let zz = viewDir.z * viewDir.z;
    let xy = viewDir.x * viewDir.y;

    return sh3_0 * (-0.5900436 * viewDir.y * (3.0 * xx - yy))
        + sh3_1 * (2.8906114 * xy * viewDir.z)
        + sh3_2 * (-0.4570458 * viewDir.y * (4.0 * zz - xx - yy))
        + sh3_3 * (0.3731763 * viewDir.z * (2.0 * zz - 3.0 * xx - 3.0 * yy))
        + sh3_4 * (-0.4570458 * viewDir.x * (4.0 * zz - xx - yy))
        + sh3_5 * (1.4453057 * viewDir.z * (xx - yy))
        + sh3_6 * (-0.5900436 * viewDir.x * (xx - 3.0 * yy));
}

// splatFragment.wgsl - WGSL port of splatFragment.glsl
// Spark.js 3D Gaussian Splatting renderer - WebGPU fragment shader
//
// This file is self-contained: splatDefines.wgsl content is prepended at build
// time (or concatenated at runtime). For naga validation the defines are
// included inline.

// ============================================================================
// Pipeline-overridable constants
// ============================================================================

override PREMULTIPLIED_ALPHA: bool = false;

// ============================================================================
// Fragment uniforms
// ============================================================================

struct FragmentUniforms {
    near: f32,
    far: f32,
    encodeLinear: u32,      // bool as u32 for uniform buffer
    time: f32,
    debugFlag: u32,         // bool as u32 for uniform buffer
    maxStdDev: f32,
    minAlpha: f32,
    disableFalloff: u32,    // bool as u32 for uniform buffer
    falloff: f32,
}

@group(0) @binding(5) var<uniform> fragUniforms: FragmentUniforms;

// ============================================================================
// Fragment input struct (matches vertex output, minus @builtin(position))
// ============================================================================

struct FragmentInput {
    @location(0) vRgba: vec4f,
    @location(1) vSplatUv: vec2f,
    @location(2) vNdc: vec3f,
    @location(3) @interpolate(flat) vSplatIndex: u32,
    @location(4) @interpolate(flat) adjustedStdDev: f32,
}

// ============================================================================
// Fragment shader entry point
// ============================================================================

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4f {
    var rgba = input.vRgba;

    let z2 = dot(input.vSplatUv, input.vSplatUv);
    if (z2 > (input.adjustedStdDev * input.adjustedStdDev)) {
        discard;
    }

    // New falloff function
    if (rgba.a <= 1.0) {
        rgba.a = mix(rgba.a, rgba.a * exp(-0.5 * z2), fragUniforms.falloff);
    } else {
        let a = exp((rgba.a * rgba.a - 1.0) / 2.718281828459045);
        let alpha = 1.0 - pow(1.0 - exp(-0.5 * z2), a);
        rgba.a = mix(1.0, alpha, fragUniforms.falloff);
    }

    if (rgba.a < fragUniforms.minAlpha) {
        discard;
    }

    if (fragUniforms.encodeLinear != 0u) {
        rgba = vec4f(srgbToLinear(rgba.rgb), rgba.a);
    }

    if (PREMULTIPLIED_ALPHA) {
        return vec4f(rgba.rgb * rgba.a, rgba.a);
    } else {
        return rgba;
    }
}

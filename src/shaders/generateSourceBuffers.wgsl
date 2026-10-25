// Ordinary sources: one storage buffer per attribute, SH bands concatenated.
const ZERO_ALPHA_INACTIVE = false;
@group(0) @binding(1) var<storage, read> primary: array<vec4u>;
@group(0) @binding(2) var<storage, read> secondary: array<vec4u>;
@group(0) @binding(3) var<storage, read> sh: array<u32>;

fn loadPrimary(src: u32) -> vec4u { return primary[src]; }
fn loadSecondary(src: u32) -> vec4u { return secondary[src]; }
fn loadSh(band: u32, src: u32) -> vec4u {
    let offset = frame.shOffsets[band] + src * 4u;
    return vec4u(sh[offset], sh[offset + 1u], sh[offset + 2u], sh[offset + 3u]);
}
fn loadPackedSh1(src: u32) -> vec2u {
    let offset = frame.shOffsets.x + src * 2u;
    return vec2u(sh[offset], sh[offset + 1u]);
}

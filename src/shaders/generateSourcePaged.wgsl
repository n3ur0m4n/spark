// Paged sources: SplatPager's 256x256 page layers, indexed like the WebGL
// pagedSplatTexCoord(). Selection indices are global pool indices.
// SplatPager readers also treat zero alpha as inactive.
const ZERO_ALPHA_INACTIVE = true;
@group(0) @binding(1) var primaryPages: texture_2d_array<u32>;
@group(0) @binding(2) var secondaryPages: texture_2d_array<u32>;
@group(0) @binding(3) var sh1Pages: texture_2d_array<u32>;
@group(0) @binding(8) var sh2Pages: texture_2d_array<u32>;
@group(0) @binding(9) var sh3Pages: texture_2d_array<u32>;
@group(0) @binding(10) var sh3bPages: texture_2d_array<u32>;

fn pageTexel(t: texture_2d_array<u32>, src: u32) -> vec4u {
    return textureLoad(t, vec2u(src & 255u, (src >> 8u) & 255u), src >> 16u, 0);
}
fn loadPrimary(src: u32) -> vec4u { return pageTexel(primaryPages, src); }
fn loadSecondary(src: u32) -> vec4u { return pageTexel(secondaryPages, src); }
fn loadSh(band: u32, src: u32) -> vec4u {
    switch band {
        case 0u: { return pageTexel(sh1Pages, src); }
        case 1u: { return pageTexel(sh2Pages, src); }
        case 2u: { return pageTexel(sh3Pages, src); }
        default: { return pageTexel(sh3bPages, src); }
    }
}
fn loadPackedSh1(src: u32) -> vec2u { return pageTexel(sh1Pages, src).xy; }

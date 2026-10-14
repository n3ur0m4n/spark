const DEPTH_INFINITY_F16: u32 = 0x7c00;
const DEPTH_SIZE_F16: usize = DEPTH_INFINITY_F16 as usize + 1;

#[derive(Default)]
pub struct SortBuffers {
    pub readback: Vec<u16>,
    pub ordering: Vec<u32>,
    pub buckets: Vec<u32>,
}

impl SortBuffers {
    pub fn ensure_size(&mut self, max_splats: usize) {
        if self.readback.len() < max_splats {
            self.readback.resize(max_splats, 0);
        }
        if self.ordering.len() < max_splats {
            self.ordering.resize(max_splats, 0);
        }
        if self.buckets.len() < DEPTH_SIZE_F16 {
            self.buckets.resize(DEPTH_SIZE_F16, 0);
        }
    }
}

pub fn sort_internal(buffers: &mut SortBuffers, num_splats: usize) -> Result<u32, String> {
    let SortBuffers { readback, ordering, buckets } = buffers;
    let readback = &readback[..num_splats];

    // Set the bucket counts to zero
    buckets.clear();
    buckets.resize(DEPTH_SIZE_F16, 0);

    // Count the number of splats in each bucket
    for &metric in readback.iter() {
        if (metric as u32) < DEPTH_INFINITY_F16 {
            buckets[metric as usize] += 1;
        }
    }

    // Compute bucket starting offset
    let mut active_splats = 0;
    for count in buckets.iter_mut().rev().skip(1) {
        let new_total = active_splats + *count;
        *count = active_splats;
        active_splats = new_total;
    }

    // Write out splat indices at the right location using bucket offsets
    for (index, &metric) in readback.iter().enumerate() {
        if (metric as u32) < DEPTH_INFINITY_F16 {
            ordering[buckets[metric as usize] as usize] = index as u32;
            buckets[metric as usize] += 1;
        }
    }

    // Sanity check
    if buckets[0] != active_splats {
        return Err(format!(
            "Expected {} active splats but got {}",
            active_splats,
            buckets[0]
        ));
    }
    Ok(active_splats)
}

const DEPTH_INFINITY_F32: u32 = 0x7f800000;
// 16-bit radix (2 passes)
const RADIX_BITS: u32 = 16;
const RADIX_BASE: usize = 1 << RADIX_BITS; // 65536
const RADIX_MASK: u32 = RADIX_BASE as u32 - 1;

#[derive(Default)]
pub struct Sort32Buffers {
    /// raw f32 bit‑patterns (one per splat)
    pub readback: Vec<u32>,
    /// output indices
    pub ordering: Vec<u32>,
    /// bucket counts / offsets (length == RADIX_BASE)
    pub buckets16lo: Vec<u32>,
    /// bucket counts / offsets (length == RADIX_BASE)
    pub buckets16hi: Vec<u32>,
    /// scratch space for (key, index)
    pub scratch: Vec<u64>,
}

impl Sort32Buffers {
    /// ensure all internal buffers are large enough for up to `max_splats`
    pub fn ensure_size(&mut self, max_splats: usize) {
        if self.readback.len() < max_splats {
            self.readback.resize(max_splats, 0);
        }
        if self.ordering.len() < max_splats {
            self.ordering.resize(max_splats, 0);
        }
        if self.scratch.len() < max_splats {
            self.scratch.resize(max_splats, 0);
        }
        if self.buckets16lo.len() < RADIX_BASE {
            self.buckets16lo.resize(RADIX_BASE, 0);
        }
        if self.buckets16hi.len() < RADIX_BASE {
            self.buckets16hi.resize(RADIX_BASE, 0);
        }
    }
}

fn prefix_sum_exclusive(buckets: &mut [u32]) -> u32 {
    let mut sum = 0u32;
    for b in buckets.iter_mut() {
        let tmp = *b;
        *b = sum;
        sum = sum.wrapping_add(tmp);
    }
    sum
}

/// Two‑pass radix sort (base 2¹⁶) of 32‑bit float bit‑patterns,
/// descending order (largest keys first).
pub fn sort32_internal(
    buffers: &mut Sort32Buffers,
    max_splats: usize,
    num_splats: usize,
) -> Result<u32, String> {
    // make sure our buffers can hold `max_splats`
    buffers.ensure_size(max_splats);

    let Sort32Buffers { readback, ordering, buckets16lo, buckets16hi, scratch } = buffers;
    let keys = &readback[..num_splats];

    // tally low and high buckets (branchless)
    buckets16lo.fill(0);
    buckets16hi.fill(0);

    macro_rules! tick {
        ($key:expr) => {{
            let valid = ($key < DEPTH_INFINITY_F32) as u32;
            let inv = !$key;
            let lo = inv & RADIX_MASK;
            let hi = inv >> RADIX_BITS;

            // by mask above: lo < 65536 == buckets16lo.len() == RADIX_BASE
            unsafe { *buckets16lo.get_unchecked_mut(lo as usize) += valid; }
            // by shift above: hi < 65536 == buckets16hi.len() == RADIX_BASE
            unsafe { *buckets16hi.get_unchecked_mut(hi as usize) += valid; }
        }};
    }

    let mut chunks = keys.chunks_exact(8);

    for chunk in chunks.by_ref() {
        tick!(chunk[0]);
        tick!(chunk[1]);
        tick!(chunk[2]);
        tick!(chunk[3]);
        tick!(chunk[4]);
        tick!(chunk[5]);
        tick!(chunk[6]);
        tick!(chunk[7]);
    }

    for &k in chunks.remainder() {
        tick!(k);
    }

    // exclusive prefix‑sum → starting offsets
    let active_splats = prefix_sum_exclusive(buckets16lo);
    prefix_sum_exclusive(buckets16hi);

    // ——— Pass #1: bucket by inv(low 16 bits) ———

    // scatter into scratch by low bits of inv
    macro_rules! place {
        ($key:expr, $idx:expr) => {{
            if $key < DEPTH_INFINITY_F32 {
                let inv = !$key;
                let lo = (inv & RADIX_MASK) as usize;
                // by mask above: lo < 65536 == buckets16lo.len() == RADIX_BASE
                let pos = unsafe { *buckets16lo.get_unchecked(lo) } as usize;
                let inv_idx = ((inv as u64) << 32) | ($idx as u64);

                // by design we have pos < active_splats <= max_splats <= scratch.len()
                unsafe { *scratch.get_unchecked_mut(pos) = inv_idx; }
                // by mask above: lo < 65536 == buckets16lo.len() == RADIX_BASE
                unsafe { *buckets16lo.get_unchecked_mut(lo) += 1; }
            }
        }};
    }

    let mut chunks = keys.chunks_exact(8);
    let mut i = 0;

    for chunk in chunks.by_ref() {
        place!(chunk[0], i);
        place!(chunk[1], i + 1);
        place!(chunk[2], i + 2);
        place!(chunk[3], i + 3);
        place!(chunk[4], i + 4);
        place!(chunk[5], i + 5);
        place!(chunk[6], i + 6);
        place!(chunk[7], i + 7);

        i += 8;
    }

    for &k in chunks.remainder() {
        place!(k, i);
        i += 1;
    }

    // ——— Pass #2: bucket by inv(high 16 bits) ———

    // scatter into final ordering by high bits of inv
    macro_rules! place2 {
        ($inv_idx:expr) => {{
            let idx = $inv_idx as u32;
            let hi = (($inv_idx >> 48) & RADIX_MASK as u64) as usize;
            // by mask above: hi < 65536 == buckets16hi.len() == RADIX_BASE
            let pos = unsafe { *buckets16hi.get_unchecked(hi) } as usize;

            // by design we have pos < active_splats <= max_splats <= ordering.len()
            unsafe { *ordering.get_unchecked_mut(pos) = idx; }
            // by mask above: hi < 65536 == buckets16hi.len() == RADIX_BASE
            unsafe { *buckets16hi.get_unchecked_mut(hi) += 1; }
        }};
    }

    let mut chunks = scratch[..active_splats as usize].chunks_exact(8);

    for chunk in chunks.by_ref() {
        place2!(chunk[0]);
        place2!(chunk[1]);
        place2!(chunk[2]);
        place2!(chunk[3]);
        place2!(chunk[4]);
        place2!(chunk[5]);
        place2!(chunk[6]);
        place2!(chunk[7]);
    }

    for &inv_idx in chunks.remainder() {
        place2!(inv_idx);
    }

    // sanity‑check: last bucket should have consumed all entries
    if buckets16hi[RADIX_BASE - 1] != active_splats {
        return Err(format!(
            "Expected {} active splats but got {}",
            active_splats,
            buckets16hi[RADIX_BASE - 1]
        ));
    }

    Ok(active_splats)
}

// ——— Parallel sort kernels ———
// N workers each run the single-threaded WASM; JS coordinates via
// SharedArrayBuffer (see src/sortParallel.ts): histogram of the high 16 bits,
// scatter (in JS) of (key, index) pairs grouped by high bits, then each worker
// sorts a contiguous range of high-bit buckets by the low 16 bits.
// Output order matches sort32_internal exactly: active keys (< DEPTH_INFINITY_F32)
// by !key ascending, ties by ascending index.

/// Buckets up to this size are comparison-sorted instead of radix-sorted.
const SMALL_BUCKET: usize = 64;

/// Step 1: add counts of the inverted high 16 bits of active keys to `hist`.
pub fn sort32_hist_hi(keys: &[u32], hist: &mut [u32]) {
    assert!(hist.len() >= RADIX_BASE);
    for &key in keys {
        let valid = (key < DEPTH_INFINITY_F32) as u32;
        // (!key >> 16) < 65536 <= hist.len() checked above
        unsafe { *hist.get_unchecked_mut((!key >> RADIX_BITS) as usize) += valid; }
    }
}

/// Step 4: `pairs` holds interleaved (key, index), all active, grouped by
/// inverted high 16 bits ascending and by index ascending within a group.
/// Sorts each group by inverted low 16 bits (stable) and writes the indices.
pub fn sort32_region(pairs: &[u32], tmp: &mut Vec<u64>, out: &mut [u32]) {
    let n = pairs.len() / 2;
    if tmp.len() < n {
        tmp.resize(n, 0);
    }
    let mut start = 0;
    while start < n {
        let hi = pairs[2 * start] >> RADIX_BITS;
        let mut end = start + 1;
        while end < n && pairs[2 * end] >> RADIX_BITS == hi {
            end += 1;
        }
        sort_bucket_lo(&pairs[2 * start..2 * end], &mut tmp[start..end], &mut out[start..end]);
        start = end;
    }
}

// One high-bits bucket, small enough to stay in cache: (inv_lo << 32 | index)
// sorted ascending is the stable order. Large buckets: two 8-bit LSD passes.
fn sort_bucket_lo(pairs: &[u32], tmp: &mut [u64], out: &mut [u32]) {
    let m = out.len();
    if m <= SMALL_BUCKET {
        for i in 0..m {
            tmp[i] = (((!pairs[2 * i] & RADIX_MASK) as u64) << 32) | pairs[2 * i + 1] as u64;
        }
        tmp.sort_unstable();
        for i in 0..m {
            out[i] = tmp[i] as u32;
        }
        return;
    }

    let mut c0 = [0u32; 256];
    let mut c1 = [0u32; 256];
    for i in 0..m {
        let inv = !pairs[2 * i];
        c0[(inv & 0xff) as usize] += 1;
        c1[((inv >> 8) & 0xff) as usize] += 1;
    }
    prefix_sum_exclusive(&mut c0);
    prefix_sum_exclusive(&mut c1);
    for i in 0..m {
        let inv = !pairs[2 * i];
        let b = (inv & 0xff) as usize;
        tmp[c0[b] as usize] = (((inv & RADIX_MASK) as u64) << 32) | pairs[2 * i + 1] as u64;
        c0[b] += 1;
    }
    for &lo_idx in tmp.iter() {
        let b = ((lo_idx >> 40) & 0xff) as usize;
        out[c1[b] as usize] = lo_idx as u32;
        c1[b] += 1;
    }
}

// ——— 16-bit sort of 32-bit keys ———
// The f32 depth keys are rounded to f16 (nearest, ties to even) and sorted by
// one counting pass, descending, ties by ascending index (as sort_internal).
// Keys the 32-bit sort excludes (>= DEPTH_INFINITY_F32: +inf, NaN and every
// negative value, whose sign bit makes the u32 large) stay excluded, so both
// sorts draw the same splats; finite keys above 65504 clamp to 0x7bff.
// The mapping is monotone non-decreasing on active keys.

/// f32 bit pattern -> f16 bit pattern for sorting (see above).
#[inline]
pub fn f32_key_to_f16(key: u32) -> u16 {
    // f16 biased exponent
    let exp = ((key >> 23) & 0xff) as i32 - 112;
    let mant = key & 0x7fffff;
    let (h, rem, half) = if exp >= 1 {
        (((exp as u32) << 10) | (mant >> 13), mant & 0x1fff, 0x1000)
    } else {
        // f16 subnormal or zero; shift 25 rounds everything to 0
        let shift = (14 - exp).min(25) as u32;
        let m = mant | 0x800000;
        (m >> shift, m & ((1 << shift) - 1), 1 << (shift - 1))
    };
    // a carry out of the mantissa correctly bumps the exponent
    let h = h + (rem > half || (rem == half && h & 1 == 1)) as u32;
    // select, not branch: excluded keys may be frequent and random (culling)
    if key >= DEPTH_INFINITY_F32 {
        DEPTH_INFINITY_F16 as u16
    } else {
        h.min(0x7bff) as u16
    }
}

/// Single worker: round `keys` to f16 and sort them into buffers.ordering,
/// returns the active count. Branchless: excluded keys are scattered after the
/// active ones, so ordering[active..keys.len()] is overwritten with them.
pub fn sort32_as16_internal(buffers: &mut SortBuffers, keys: &[u32]) -> u32 {
    let n = keys.len();
    buffers.ensure_size(n);
    let SortBuffers { readback, ordering, buckets } = buffers;
    buckets.resize(DEPTH_SIZE_F16, 0);
    sort16_hist(keys, &mut readback[..n], buckets);
    let active = offsets16(buckets);
    for (i, &h) in readback[..n].iter().enumerate() {
        let pos = &mut buckets[h as usize];
        ordering[*pos as usize] = i as u32;
        *pos += 1;
    }
    active
}

/// Counts -> start offsets: descending keys, then the excluded bucket.
/// Returns the active count.
pub fn offsets16(hist: &mut [u32]) -> u32 {
    let mut sum = 0;
    for b in (0..DEPTH_INFINITY_F16 as usize).rev() {
        let count = hist[b];
        hist[b] = sum;
        sum += count;
    }
    hist[DEPTH_INFINITY_F16 as usize] = sum;
    sum
}

/// Parallel step 1: round one slice of keys to f16 into `keys16` and count
/// them in `hist` (DEPTH_SIZE_F16 buckets, the last one counts excluded keys).
/// Step 2 (offsets per worker as offsets16) and the scatter run in JS.
pub fn sort16_hist(keys: &[u32], keys16: &mut [u16], hist: &mut [u32]) {
    assert!(hist.len() >= DEPTH_SIZE_F16 && keys16.len() >= keys.len());
    hist.fill(0);
    for (dst, &key) in keys16.iter_mut().zip(keys) {
        let h = f32_key_to_f16(key);
        *dst = h;
        hist[h as usize] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }
    }

    fn reference(keys: &[u32]) -> (u32, Vec<u32>) {
        let mut buffers = Sort32Buffers::default();
        buffers.ensure_size(keys.len());
        buffers.readback[..keys.len()].copy_from_slice(keys);
        let active = sort32_internal(&mut buffers, keys.len(), keys.len()).unwrap();
        (active, buffers.ordering[..active as usize].to_vec())
    }

    /// Mirrors src/sortParallel.ts (planSort32 + JS scatter) around the kernels.
    fn parallel(keys: &[u32], workers: usize) -> (u32, Vec<u32>) {
        let n = keys.len();
        let chunk = n.div_ceil(workers).max(1);
        let slices: Vec<(usize, usize)> = (0..workers)
            .map(|w| ((w * chunk).min(n), ((w + 1) * chunk).min(n)))
            .collect();

        let mut hists = vec![0u32; workers * RADIX_BASE];
        for (w, &(s, e)) in slices.iter().enumerate() {
            sort32_hist_hi(&keys[s..e], &mut hists[w * RADIX_BASE..(w + 1) * RADIX_BASE]);
        }

        // counts -> per-worker global offsets
        let mut sum = 0u32;
        let mut bucket_start = vec![0u32; RADIX_BASE + 1];
        for b in 0..RADIX_BASE {
            bucket_start[b] = sum;
            for w in 0..workers {
                let c = hists[w * RADIX_BASE + b];
                hists[w * RADIX_BASE + b] = sum;
                sum += c;
            }
        }
        let active = sum;
        bucket_start[RADIX_BASE] = active;

        // contiguous bucket ranges with ~equal counts
        let mut ranges = Vec::new();
        let mut b0 = 0;
        let mut k = 1;
        for b in 0..RADIX_BASE {
            if k >= workers {
                break;
            }
            if bucket_start[b + 1] as u64 * workers as u64 >= active as u64 * k as u64 {
                ranges.push((bucket_start[b0] as usize, bucket_start[b + 1] as usize));
                b0 = b + 1;
                k += 1;
            }
        }
        ranges.push((bucket_start[b0] as usize, active as usize));

        let mut pairs = vec![0u32; 2 * n];
        for (w, &(s, e)) in slices.iter().enumerate() {
            let offsets = &mut hists[w * RADIX_BASE..(w + 1) * RADIX_BASE];
            for i in s..e {
                let key = keys[i];
                if key < DEPTH_INFINITY_F32 {
                    let pos = &mut offsets[(!key >> RADIX_BITS) as usize];
                    pairs[2 * *pos as usize] = key;
                    pairs[2 * *pos as usize + 1] = i as u32;
                    *pos += 1;
                }
            }
        }

        let mut ordering = vec![0u32; active as usize];
        let mut tmp = Vec::new();
        for &(s, e) in &ranges {
            sort32_region(&pairs[2 * s..2 * e], &mut tmp, &mut ordering[s..e]);
        }
        (active, ordering)
    }

    fn check(keys: &[u32]) {
        let (active, expected) = reference(keys);
        for workers in 1..=5 {
            let (pactive, ordering) = parallel(keys, workers);
            assert_eq!(active, pactive, "workers {workers}");
            // both stable: identical ordering, not just key sequence
            assert_eq!(expected, ordering, "workers {workers}");
        }
    }

    #[test]
    fn parallel_matches_sort32_internal() {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        check(&[]);
        check(&[DEPTH_INFINITY_F32; 100]);
        check(&[0x3f800000; 1000]);
        check(&[1, 2, 3]);
        check(&[0x3f800000, DEPTH_INFINITY_F32]);
        check(&[0, 0x7f7fffff, 0x7f800000, 0xffffffff, 0x80000000, 0x0000ffff, 0x00010000]);
        for &n in &[7usize, 1000, 100_000, 1_000_000] {
            let random: Vec<u32> = (0..n)
                .map(|_| if rng.next() % 20 == 0 { DEPTH_INFINITY_F32 } else { rng.next() })
                .collect();
            check(&random);
            let depths: Vec<u32> = (0..n)
                .map(|_| (1.0f32 + (rng.next() % 10000) as f32 * 0.01).to_bits())
                .collect();
            check(&depths);
            // few distinct keys: huge buckets with many ties
            let few: Vec<u32> = (0..n).map(|_| 0x40000000 + rng.next() % 4).collect();
            check(&few);
            // one high bucket, all low bits used
            let one_hi: Vec<u32> = (0..n).map(|_| 0x40000000 | (rng.next() & 0xffff)).collect();
            check(&one_hi);
        }
    }

    fn f16_value(h: u32) -> f64 {
        if h < 0x400 {
            h as f64 * 2f64.powi(-24)
        } else {
            (1024 + (h & 0x3ff)) as f64 * 2f64.powi((h >> 10) as i32 - 25)
        }
    }

    /// Nearest f16 by search over all finite values, ties to even, clamped.
    fn f16_reference(key: u32) -> u16 {
        if key >= DEPTH_INFINITY_F32 {
            return DEPTH_INFINITY_F16 as u16;
        }
        let v = f32::from_bits(key) as f64;
        // largest h with value(h) <= v
        let (mut lo, mut hi) = (0u32, 0x7bffu32);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if f16_value(mid) <= v { lo = mid } else { hi = mid - 1 }
        }
        if lo == 0x7bff {
            return 0x7bff;
        }
        let (d0, d1) = (v - f16_value(lo), f16_value(lo + 1) - v);
        if d1 < d0 || (d1 == d0 && lo & 1 == 1) { (lo + 1) as u16 } else { lo as u16 }
    }

    #[test]
    fn f32_key_to_f16_rounds_to_nearest_even() {
        let mut keys: Vec<u32> = (0..0x7f800000u32).step_by(997).collect();
        // around every f16 value: exact, ties and neighbours
        for h in 0..0x7c00u32 {
            let v = f16_value(h);
            let mid = (v + f16_value(h + 1)) * 0.5;
            for x in [v as f32, mid as f32] {
                let b = x.to_bits();
                keys.extend_from_slice(&[b.saturating_sub(1), b, b + 1]);
            }
        }
        keys.extend_from_slice(&[0, 1, 0x7f7fffff, 65504f32.to_bits(), 65520f32.to_bits(), 65519.996f32.to_bits()]);
        keys.extend_from_slice(&[0x7f800000, 0x7fc00000, 0x80000000, (-1.0f32).to_bits(), 0xffffffff]);
        for &k in &keys {
            assert_eq!(f32_key_to_f16(k), f16_reference(k), "key {k:#x}");
        }
        // monotone on active keys
        let mut prev = 0;
        for k in (0..0x7f800000u32).step_by(101) {
            let h = f32_key_to_f16(k);
            assert!(h >= prev, "key {k:#x}");
            prev = h;
        }
    }

    /// Stable descending sort by f16 key, active (< 0x7c00) only.
    fn reference16(keys: &[u32]) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..keys.len() as u32)
            .filter(|&i| f16_reference(keys[i as usize]) < DEPTH_INFINITY_F16 as u16)
            .collect();
        idx.sort_by_key(|&i| std::cmp::Reverse(f16_reference(keys[i as usize])));
        idx
    }

    /// Mirrors src/sortParallel.ts (parallelSort16) around sort16_hist.
    fn parallel16(keys: &[u32], workers: usize) -> Vec<u32> {
        let n = keys.len();
        let chunk = n.div_ceil(workers).max(1);
        let slices: Vec<(usize, usize)> = (0..workers)
            .map(|w| ((w * chunk).min(n), ((w + 1) * chunk).min(n)))
            .collect();
        let mut keys16 = vec![0u16; n];
        let mut hists = vec![0u32; workers * DEPTH_SIZE_F16];
        for (w, &(s, e)) in slices.iter().enumerate() {
            sort16_hist(&keys[s..e], &mut keys16[s..e], &mut hists[w * DEPTH_SIZE_F16..(w + 1) * DEPTH_SIZE_F16]);
        }
        let mut sum = 0u32;
        let mut active = 0;
        let inf = DEPTH_INFINITY_F16 as usize;
        // descending active buckets, then the excluded one
        for b in (0..inf).rev().chain([inf]) {
            if b == inf {
                active = sum;
            }
            for w in 0..workers {
                let c = hists[w * DEPTH_SIZE_F16 + b];
                hists[w * DEPTH_SIZE_F16 + b] = sum;
                sum += c;
            }
        }
        let mut ordering = vec![0u32; n];
        for (w, &(s, e)) in slices.iter().enumerate() {
            let offsets = &mut hists[w * DEPTH_SIZE_F16..(w + 1) * DEPTH_SIZE_F16];
            for i in s..e {
                let h = keys16[i] as usize;
                ordering[offsets[h] as usize] = i as u32;
                offsets[h] += 1;
            }
        }
        ordering.truncate(active as usize);
        ordering
    }

    fn check16(keys: &[u32]) {
        let expected = reference16(keys);
        let mut buffers = SortBuffers::default();
        let active = sort32_as16_internal(&mut buffers, keys);
        assert_eq!(active as usize, expected.len());
        assert_eq!(&buffers.ordering[..active as usize], &expected[..]);
        for workers in 1..=5 {
            assert_eq!(parallel16(keys, workers), expected, "workers {workers}");
        }
    }

    #[test]
    fn sort16_matches_reference() {
        let mut rng = Rng(0x2545f4914f6cdd1d);
        check16(&[]);
        check16(&[DEPTH_INFINITY_F32; 100]);
        check16(&[0x3f800000; 1000]);
        // n < workers
        check16(&[1, 0x3f800000, 3]);
        // negatives, -0, NaN, > 65504, subnormals, zero
        check16(&[
            (-1.0f32).to_bits(), 0x80000000, 0x7fc00000, 70000f32.to_bits(), 65504f32.to_bits(),
            0x7f7fffff, 1, 0, 0x33800000, 0x33800001, 2.5f32.to_bits(),
        ]);
        for &n in &[7usize, 1000, 100_000] {
            let random: Vec<u32> = (0..n).map(|_| rng.next()).collect();
            check16(&random);
            let depths: Vec<u32> = (0..n)
                .map(|_| match rng.next() % 10 {
                    0 => DEPTH_INFINITY_F32,
                    1 => (-((rng.next() % 1000) as f32)).to_bits(),
                    _ => ((rng.next() % 100_000) as f32 * 0.01).to_bits(),
                })
                .collect();
            check16(&depths);
        }
    }
}

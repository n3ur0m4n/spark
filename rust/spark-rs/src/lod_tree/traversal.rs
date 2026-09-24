//! Selection core shared by the WASM entry point and native differential tests.
use super::*;

pub(super) type Instance<'a> = (
    u32,
    Ref<'a, Vec<LodSplat>>,
    &'a Vec<u32>,
    &'a Vec<u32>,
    Vec3A,
    Vec3A,
    f32,
    f32,
    f32,
    f32,
    f32,
);

#[derive(Default)]
pub(super) struct Buffers {
    frontier: Frontier<(OrderedFloat<f32>, u32, u32)>,
    pub(super) touched: Vec<(u32, u32)>,
    touched_set: AHashSet<(u32, u32)>,
    output: Vec<(u32, u32)>,
    counts: Vec<usize>,
    pub(super) instance_outputs: Vec<Vec<u32>>,
}

pub(super) struct Stats {
    pub(super) pixel_limit: f32,
    pub(super) emitted: usize,
    pub(super) drained: usize,
    pub(super) leaves: usize,
}

// Preserve the complete upstream emission sequence. Bypassing terminal heap
// entries or canonicalizing with a bitset changes equal-depth alpha blending.
pub(super) fn select(
    max_splats: usize,
    pixel_scale_limit: f32,
    root_pages: &[u32],
    instances: &[Instance<'_>],
    buffers: &mut Buffers,
) -> Stats {
    let num_instances = instances.len();
    let Buffers {
        frontier,
        touched,
        touched_set,
        output,
        counts,
        instance_outputs,
    } = buffers;
    let mut num_splats = 0;
    frontier.clear();
    output.clear();
    output.reserve(max_splats);
    touched.clear();
    touched_set.clear();

    for (inst_index, instance) in instances.iter().enumerate() {
        let (lod_id, splats, ..) = instance;
        let root_page = root_pages[inst_index];
        let root_page = if root_page == 0xFFFFFFFF {
            0
        } else {
            root_page
        };
        let root_index = root_page << 16;
        let pixel_scale = compute_pixel_scale(&splats[root_index as usize], instance);
        frontier.push((OrderedFloat(pixel_scale), inst_index as u32, root_index));
        num_splats += 1;

        if touched_set.insert((*lod_id, 0)) {
            touched.push((*lod_id, 0));
        }
    }

    let mut min_pixel_scale = f32::INFINITY;
    let mut leaf_count = 0;

    while let Some(&(OrderedFloat(pixel_scale), inst_index, paged_index)) = frontier.peek() {
        min_pixel_scale = min_pixel_scale.min(pixel_scale);
        if pixel_scale <= pixel_scale_limit {
            break;
        }

        let instance = &instances[inst_index as usize];
        let (lod_id, splats, _page_to_chunk, chunk_to_page, ..) = instance;
        let LodSplat {
            child_count,
            child_start,
            ..
        } = splats[paged_index as usize];

        if child_count == 0 {
            _ = frontier.pop();
            output.push((inst_index, paged_index));
            leaf_count += 1;
            continue;
        }

        let new_num_splats = num_splats - 1 + child_count as usize;
        if new_num_splats > max_splats {
            break;
        }

        _ = frontier.pop();

        let first_chunk = child_start >> 16;
        if touched_set.insert((*lod_id, first_chunk)) {
            touched.push((*lod_id, first_chunk));
        }

        let last_chunk = (child_start + child_count as u32 - 1) >> 16;
        if last_chunk != first_chunk && touched_set.insert((*lod_id, last_chunk)) {
            touched.push((*lod_id, last_chunk));
        }

        if last_chunk as usize >= chunk_to_page.len() {
            output.push((inst_index, paged_index));
            continue;
        }
        let first_page = chunk_to_page[first_chunk as usize];
        let last_page = chunk_to_page[last_chunk as usize];

        if first_page == 0xFFFFFFFF || last_page == 0xFFFFFFFF {
            output.push((inst_index, paged_index));
            continue;
        }

        for child in child_start..child_start + child_count as u32 {
            let child_chunk = (child >> 16) as usize;
            let child_page = chunk_to_page[child_chunk];
            let paged_index = (child_page << 16) | (child & 0xffff);
            let child_splat = &splats[paged_index as usize];
            let pixel_scale = compute_pixel_scale(child_splat, instance);
            if pixel_scale <= pixel_scale_limit {
                output.push((inst_index, paged_index));
            } else {
                frontier.push((OrderedFloat(pixel_scale), inst_index, paged_index));
            }
        }

        num_splats = new_num_splats;
    }

    let output_size = output.len();
    let frontier_size = frontier.len();

    for (_, inst_index, paged_index) in frontier.drain() {
        output.push((inst_index, paged_index));
    }

    counts.clear();
    counts.resize(num_instances, 0);
    for &(inst_index, _) in output.iter() {
        counts[inst_index as usize] += 1;
    }

    if instance_outputs.len() < num_instances {
        instance_outputs.resize_with(num_instances, Vec::new);
    }
    for (values, &count) in instance_outputs.iter_mut().zip(counts.iter()) {
        values.clear();
        values.reserve(count);
    }

    for &(inst_index, paged_index) in output.iter() {
        instance_outputs[inst_index as usize].push(paged_index);
    }

    Stats {
        pixel_limit: min_pixel_scale,
        emitted: output_size,
        drained: frontier_size,
        leaves: leaf_count,
    }
}

#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;

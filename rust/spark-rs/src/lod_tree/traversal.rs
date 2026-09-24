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
    membership_ids: AHashMap<u32, usize>,
    memberships: Vec<Membership>,
    instance_memberships: Vec<usize>,
    pub(super) instance_outputs: Vec<Vec<u32>>,
}

#[derive(Default)]
struct Membership {
    marks: Vec<u32>,
    epoch: u32,
    overflow: AHashSet<u32>,
}

impl Membership {
    fn begin(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.fill(0);
            self.epoch = 1;
        }
        self.overflow.clear();
    }

    fn first_touch(&mut self, chunk: u32) -> bool {
        if let Some(mark) = self.marks.get_mut(chunk as usize) {
            let first = *mark != self.epoch;
            *mark = self.epoch;
            first
        } else {
            self.overflow.insert(chunk)
        }
    }
}

impl Buffers {
    pub(super) fn forget_tree(&mut self, id: u32) {
        if let Some(index) = self.membership_ids.remove(&id) {
            self.memberships.swap_remove(index);
            let old_last = self.memberships.len();
            for slot in self.membership_ids.values_mut() {
                if *slot == old_last {
                    *slot = index;
                }
            }
            self.instance_memberships.clear();
        }
    }
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
        membership_ids,
        memberships,
        instance_memberships,
        instance_outputs,
    } = buffers;
    let mut num_splats = 0;
    frontier.clear();
    if instance_outputs.len() < num_instances {
        instance_outputs.resize_with(num_instances, Vec::new);
    }
    for values in instance_outputs.iter_mut() {
        values.clear();
    }
    let mut emitted = 0;
    touched.clear();
    for membership in memberships.iter_mut() {
        membership.begin();
    }
    instance_memberships.clear();
    for (lod_id, _, _, chunk_to_page, ..) in instances {
        let index = *membership_ids.entry(*lod_id).or_insert_with(|| {
            let mut membership = Membership::default();
            membership.begin();
            memberships.push(membership);
            memberships.len() - 1
        });
        memberships[index].marks.resize(chunk_to_page.len(), 0);
        instance_memberships.push(index);
    }

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

        if memberships[instance_memberships[inst_index]].first_touch(0) {
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
            instance_outputs[inst_index as usize].push(paged_index);
            emitted += 1;
            leaf_count += 1;
            continue;
        }

        let new_num_splats = num_splats - 1 + child_count as usize;
        if new_num_splats > max_splats {
            break;
        }

        _ = frontier.pop();

        let first_chunk = child_start >> 16;
        if memberships[instance_memberships[inst_index as usize]].first_touch(first_chunk) {
            touched.push((*lod_id, first_chunk));
        }

        let last_chunk = (child_start + child_count as u32 - 1) >> 16;
        if last_chunk != first_chunk
            && memberships[instance_memberships[inst_index as usize]].first_touch(last_chunk)
        {
            touched.push((*lod_id, last_chunk));
        }

        if last_chunk as usize >= chunk_to_page.len() {
            instance_outputs[inst_index as usize].push(paged_index);
            emitted += 1;
            continue;
        }
        let first_page = chunk_to_page[first_chunk as usize];
        let last_page = chunk_to_page[last_chunk as usize];

        if first_page == 0xFFFFFFFF || last_page == 0xFFFFFFFF {
            instance_outputs[inst_index as usize].push(paged_index);
            emitted += 1;
            continue;
        }

        for child in child_start..child_start + child_count as u32 {
            let child_chunk = (child >> 16) as usize;
            let child_page = chunk_to_page[child_chunk];
            let paged_index = (child_page << 16) | (child & 0xffff);
            let child_splat = &splats[paged_index as usize];
            let pixel_scale = compute_pixel_scale(child_splat, instance);
            if pixel_scale <= pixel_scale_limit {
                instance_outputs[inst_index as usize].push(paged_index);
                emitted += 1;
            } else {
                frontier.push((OrderedFloat(pixel_scale), inst_index, paged_index));
            }
        }

        num_splats = new_num_splats;
    }

    let output_size = emitted;
    let frontier_size = frontier.len();

    for (_, inst_index, paged_index) in frontier.drain() {
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

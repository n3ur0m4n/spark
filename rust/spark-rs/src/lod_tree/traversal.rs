//! Selection core shared by the WASM entry point and native differential tests.
//!
//! Same result as upstream's best-first loop over one global BinaryHeap, without
//! that heap. Key p(n) = (pixel scale, instance, paged index), unique per node.
//! e(n) = min of p on the path root->n. Upstream pops in decreasing e: a "group"
//! is a node m with p(m) < e(parent) plus the descendants it reaches through
//! nodes with p > p(m); a group pops contiguously, in local best-first order,
//! because every other heap entry is < p(m). So:
//! 1. Walk the groups with key > tau (no budget), each once in local order into
//!    a record (emissions, touched chunks, outer pushes, budget delta/peak);
//!    sort the group keys descending.
//! 2. Replay: copy whole records while the budget allows, step the stop group
//!    node by node, so emission order and paging (`touched`) order are exact.
//! 3. What upstream leaves in its heap is drained sorted by paged index per
//!    instance (the heap's internal order is history dependent).
//!    If the list runs out before the budget stops, tau was too high: halve it.
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

type Key = (OrderedFloat<f32>, u32, u32);

#[derive(Default)]
pub(super) struct Buffers {
    pub(super) touched: Vec<(u32, u32)>,
    membership_ids: AHashMap<u32, usize>,
    memberships: Vec<Membership>,
    instance_memberships: Vec<usize>,
    pub(super) instance_outputs: Vec<Vec<u32>>,
    pub(super) raycast_outputs: Vec<Vec<u32>>,
    scratch: Scratch,
    // Stop key of the previous selection, seeds the next DFS cut (0 = unknown).
    tau: f32,
}

#[derive(Default)]
struct Scratch {
    groups: Vec<(Key, u32)>,
    stack: Vec<Key>,
    local: BinaryHeap<Key>,
    pending: Vec<Key>,
    main_lens: Vec<usize>,
    records: Vec<Record>,
    emits: Vec<u32>,
    outer: Vec<Key>,
    chunks: Vec<u32>,
}

// One group processed in local order; ranges end where the next record starts.
#[derive(Default)]
struct Record {
    emits: u32,
    outer: u32,
    chunks: u32,
    leaves: usize,
    delta: usize,
    peak: usize,
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

struct Touch<'a> {
    touched: &'a mut Vec<(u32, u32)>,
    memberships: &'a mut [Membership],
    instance_memberships: &'a [usize],
}

impl Touch<'_> {
    fn touch(&mut self, inst_index: u32, lod_id: u32, chunk: u32) {
        if self.memberships[self.instance_memberships[inst_index as usize]].first_touch(chunk) {
            self.touched.push((lod_id, chunk));
        }
    }

    fn reset(&mut self, instances: &[Instance<'_>]) {
        self.touched.clear();
        for membership in self.memberships.iter_mut() {
            membership.begin();
        }
        for (inst_index, instance) in instances.iter().enumerate() {
            self.touch(inst_index as u32, instance.0, 0);
        }
    }
}

fn root_index(root_page: u32) -> u32 {
    (if root_page == 0xFFFFFFFF { 0 } else { root_page }) << 16
}

/// Children range of an expandable node, None for leaves and missing pages
/// (upstream emits those nodes themselves).
fn resident_children(node: &LodSplat, chunk_to_page: &[u32]) -> Option<std::ops::Range<u32>> {
    let LodSplat { child_count, child_start, .. } = *node;
    if child_count == 0 {
        return None;
    }
    let first_chunk = (child_start >> 16) as usize;
    let last_chunk = ((child_start + child_count as u32 - 1) >> 16) as usize;
    if last_chunk >= chunk_to_page.len()
        || chunk_to_page[first_chunk] == 0xFFFFFFFF
        || chunk_to_page[last_chunk] == 0xFFFFFFFF
    {
        return None;
    }
    Some(child_start..child_start + child_count as u32)
}

fn paged(child: u32, chunk_to_page: &[u32]) -> u32 {
    (chunk_to_page[(child >> 16) as usize] << 16) | (child & 0xffff)
}

fn listed(pixel_scale: f32, tau: Option<f32>) -> bool {
    tau.is_none_or(|tau| OrderedFloat(pixel_scale) > OrderedFloat(tau))
}

/// Group roots with key > tau (all when None) in upstream pop order, each
/// processed once in its local best-first order into a record: what upstream
/// emits, the chunks it touches, what it pushes outside the group, its budget
/// delta and the headroom it needs to complete.
fn collect(
    pixel_scale_limit: f32,
    root_pages: &[u32],
    instances: &[Instance<'_>],
    tau: Option<f32>,
    scratch: &mut Scratch,
) {
    let Scratch { groups, stack, local, records, emits, outer, chunks, .. } = scratch;
    groups.clear();
    records.clear();
    emits.clear();
    outer.clear();
    chunks.clear();
    for (inst_index, instance) in instances.iter().enumerate() {
        let (_, splats, _, chunk_to_page, ..) = instance;
        let inst = inst_index as u32;
        let root = root_index(root_pages[inst_index]);
        let pixel_scale = compute_pixel_scale(&splats[root as usize], instance);
        if pixel_scale <= pixel_scale_limit || !listed(pixel_scale, tau) {
            continue;
        }
        stack.push((OrderedFloat(pixel_scale), inst, root));
        while let Some(group) = stack.pop() {
            let mut record = Record {
                emits: emits.len() as u32,
                outer: outer.len() as u32,
                chunks: chunks.len() as u32,
                ..Default::default()
            };
            local.clear();
            local.push(group);
            while let Some((_, _, index)) = local.pop() {
                let node = &splats[index as usize];
                let child_count = node.child_count as usize;
                if child_count == 0 {
                    emits.push(index);
                    record.leaves += 1;
                    continue;
                }
                record.peak = record.peak.max(record.delta + child_count - 1);
                let first_chunk = node.child_start >> 16;
                let last_chunk = (node.child_start + child_count as u32 - 1) >> 16;
                chunks.push(first_chunk);
                if last_chunk != first_chunk {
                    chunks.push(last_chunk);
                }
                let Some(children) = resident_children(node, chunk_to_page) else {
                    emits.push(index);
                    continue;
                };
                for child in children {
                    let child = paged(child, chunk_to_page);
                    let pixel_scale = compute_pixel_scale(&splats[child as usize], instance);
                    if pixel_scale <= pixel_scale_limit {
                        emits.push(child);
                        continue;
                    }
                    let key = (OrderedFloat(pixel_scale), inst, child);
                    if key > group {
                        local.push(key);
                    } else {
                        outer.push(key);
                        if listed(pixel_scale, tau) {
                            stack.push(key);
                        }
                    }
                }
                record.delta += child_count - 1;
            }
            groups.push((group, records.len() as u32));
            records.push(record);
        }
    }
    records.push(Record {
        emits: emits.len() as u32,
        outer: outer.len() as u32,
        chunks: chunks.len() as u32,
        ..Default::default()
    });
    groups.sort_unstable_by_key(|g| Reverse(g.0));
}

/// Upstream's loop over the collected groups: whole groups are copied from
/// their records, the group where the budget stops is stepped node by node.
/// None when the groups ran out before upstream would stop (tau too high).
/// Returns the stop group key too.
fn replay(
    max_splats: usize,
    pixel_scale_limit: f32,
    root_pages: &[u32],
    instances: &[Instance<'_>],
    scratch: &mut Scratch,
    outputs: &mut Vec<Vec<u32>>,
    mut touch: Option<&mut Touch<'_>>,
) -> Option<(Stats, f32)> {
    let Scratch { groups, local, pending, main_lens, records, emits, outer, chunks, .. } = scratch;
    let num_instances = instances.len();
    if outputs.len() < num_instances {
        outputs.resize_with(num_instances, Vec::new);
    }
    for values in outputs.iter_mut() {
        values.clear();
    }
    if let Some(touch) = touch.as_deref_mut() {
        touch.reset(instances);
    }
    local.clear();
    pending.clear();
    for (inst_index, instance) in instances.iter().enumerate() {
        let root = root_index(root_pages[inst_index]);
        let pixel_scale = compute_pixel_scale(&instance.1[root as usize], instance);
        pending.push((OrderedFloat(pixel_scale), inst_index as u32, root));
    }

    let mut num_splats = num_instances;
    let mut min_pixel_scale = f32::INFINITY;
    let mut emitted = 0;
    let mut leaf_count = 0;
    let mut stop: Option<Key> = None;
    let mut last: Option<Key> = None;

    'groups: for &(group, record_index) in groups.iter() {
        let (OrderedFloat(group_scale), inst_index, _) = group;
        let record = &records[record_index as usize];
        let next = &records[record_index as usize + 1];
        // Upstream stops at the first expansion with num_splats - 1 + children
        // > max_splats; peak is the largest delta + children - 1 in the group.
        if num_splats + record.peak <= max_splats {
            // Every member key is > the group key: the group minimum is its root.
            min_pixel_scale = min_pixel_scale.min(group_scale);
            let new_emits = &emits[record.emits as usize..next.emits as usize];
            outputs[inst_index as usize].extend_from_slice(new_emits);
            emitted += new_emits.len();
            leaf_count += record.leaves;
            if let Some(touch) = touch.as_deref_mut() {
                let lod_id = instances[inst_index as usize].0;
                for &chunk in &chunks[record.chunks as usize..next.chunks as usize] {
                    touch.touch(inst_index, lod_id, chunk);
                }
            }
            pending.extend_from_slice(&outer[record.outer as usize..next.outer as usize]);
            num_splats += record.delta;
            last = Some(group);
            continue;
        }

        local.push(group);
        while let Some(&(OrderedFloat(pixel_scale), inst_index, paged_index)) = local.peek() {
            min_pixel_scale = min_pixel_scale.min(pixel_scale);
            let instance = &instances[inst_index as usize];
            let (lod_id, splats, _, chunk_to_page, ..) = instance;
            let node = &splats[paged_index as usize];
            let child_count = node.child_count;
            let output = &mut outputs[inst_index as usize];

            if child_count == 0 {
                local.pop();
                output.push(paged_index);
                emitted += 1;
                leaf_count += 1;
                continue;
            }

            let new_num_splats = num_splats - 1 + child_count as usize;
            if new_num_splats > max_splats {
                stop = Some(group);
                break 'groups;
            }
            local.pop();

            if let Some(touch) = touch.as_deref_mut() {
                let first_chunk = node.child_start >> 16;
                let last_chunk = (node.child_start + child_count as u32 - 1) >> 16;
                touch.touch(inst_index, *lod_id, first_chunk);
                if last_chunk != first_chunk {
                    touch.touch(inst_index, *lod_id, last_chunk);
                }
            }

            let Some(children) = resident_children(node, chunk_to_page) else {
                output.push(paged_index);
                emitted += 1;
                continue;
            };
            for child in children {
                let child = paged(child, chunk_to_page);
                let pixel_scale = compute_pixel_scale(&splats[child as usize], instance);
                if pixel_scale <= pixel_scale_limit {
                    output.push(child);
                    emitted += 1;
                } else {
                    let key = (OrderedFloat(pixel_scale), inst_index, child);
                    if key > group {
                        local.push(key);
                    } else {
                        pending.push(key);
                    }
                }
            }
            num_splats = new_num_splats;
        }
        last = Some(group);
    }

    // Upstream's leftover heap: pushed, not yet popped.
    let bound = stop.or(last);
    if let Some(bound) = bound {
        pending.retain(|key| *key < bound);
    }
    if stop.is_none() {
        // Groups exhausted: upstream peeks the best leftover, which must stop on
        // the pixel limit, else it would expand nodes this pass did not list.
        if let Some(&(OrderedFloat(pixel_scale), ..)) = pending.iter().max() {
            min_pixel_scale = min_pixel_scale.min(pixel_scale);
            #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN must count as "not stopped", like upstream
            if !(pixel_scale <= pixel_scale_limit) {
                return None;
            }
        }
    }

    main_lens.clear();
    main_lens.extend(outputs[..num_instances].iter().map(Vec::len));
    let drained = local.len() + pending.len();
    for (_, inst_index, paged_index) in local.drain().chain(pending.drain(..)) {
        outputs[inst_index as usize].push(paged_index);
    }
    for (output, &start) in outputs.iter_mut().zip(main_lens.iter()) {
        output[start..].sort_unstable();
    }

    let tau = bound.map_or(0.0, |(OrderedFloat(pixel_scale), ..)| pixel_scale);
    Some((
        Stats {
            pixel_limit: min_pixel_scale,
            emitted,
            drained,
            leaves: leaf_count,
        },
        tau,
    ))
}

fn usable(tau: f32, pixel_scale_limit: f32) -> Option<f32> {
    // At or below the pixel limit the cut lists everything anyway.
    (tau.is_finite() && tau > 1.0e-30 && tau > pixel_scale_limit).then_some(tau)
}

/// Collect + replay, lowering tau until the listed groups reach the stop.
#[allow(clippy::too_many_arguments)]
fn run(
    max_splats: usize,
    pixel_scale_limit: f32,
    root_pages: &[u32],
    instances: &[Instance<'_>],
    scratch: &mut Scratch,
    outputs: &mut Vec<Vec<u32>>,
    mut touch: Option<&mut Touch<'_>>,
    mut tau: Option<f32>,
) -> (Stats, f32) {
    loop {
        collect(pixel_scale_limit, root_pages, instances, tau, scratch);
        if let Some(result) = replay(
            max_splats,
            pixel_scale_limit,
            root_pages,
            instances,
            scratch,
            outputs,
            touch.as_deref_mut(),
        ) {
            return result;
        }
        // tau None lists every node, so replay cannot fail and this terminates.
        tau = tau.and_then(|tau| usable(tau * 0.5, pixel_scale_limit));
    }
}

pub(super) fn select(
    max_splats: usize,
    pixel_scale_limit: f32,
    root_pages: &[u32],
    instances: &[Instance<'_>],
    buffers: &mut Buffers,
) -> Stats {
    let Buffers {
        touched,
        membership_ids,
        memberships,
        instance_memberships,
        instance_outputs,
        scratch,
        tau,
        ..
    } = buffers;
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
    let mut touch = Touch {
        touched,
        memberships,
        instance_memberships,
    };

    // Warm: a little below the previous stop (None = list all). Cold: half the
    // best root, halving.
    let start = if *tau > 0.0 {
        usable(*tau * 0.8, pixel_scale_limit)
    } else {
        let best = instances
            .iter()
            .enumerate()
            .map(|(i, instance)| {
                compute_pixel_scale(&instance.1[root_index(root_pages[i]) as usize], instance)
            })
            .fold(0.0f32, f32::max);
        usable(best * 0.5, pixel_scale_limit)
    };
    let (stats, stop) = run(
        max_splats,
        pixel_scale_limit,
        root_pages,
        instances,
        scratch,
        instance_outputs,
        Some(&mut touch),
        start,
    );
    *tau = stop;
    stats
}

/// Selection at a smaller budget (raycast set) reusing the groups listed by the
/// last select() with the same inputs. Falls back to a fresh collect when the
/// smaller budget is not covered (only if it exceeds the main one).
pub(super) fn select_raycast(
    max_splats: usize,
    pixel_scale_limit: f32,
    root_pages: &[u32],
    instances: &[Instance<'_>],
    buffers: &mut Buffers,
) {
    let Buffers {
        raycast_outputs,
        scratch,
        tau,
        ..
    } = buffers;
    if replay(
        max_splats,
        pixel_scale_limit,
        root_pages,
        instances,
        scratch,
        raycast_outputs,
        None,
    )
    .is_none()
    {
        run(
            max_splats,
            pixel_scale_limit,
            root_pages,
            instances,
            scratch,
            raycast_outputs,
            None,
            usable(*tau * 0.8, pixel_scale_limit),
        );
    }
}

#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;

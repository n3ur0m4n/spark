use std::{array, cell::{Ref, RefCell}, cmp::Reverse, collections::BinaryHeap, rc::Rc};

use ahash::{AHashMap, AHashSet};
use glam::{Vec3, Vec3A};
use half::f16;
use itertools::izip;
use js_sys::{Array, Object, Reflect, Uint32Array};
use ordered_float::OrderedFloat;
use wasm_bindgen::prelude::*;

const MAX_SPLAT_CHUNK: usize = 65536;

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
struct FourHeap<T: Ord> {
    data: Vec<T>,
}

#[allow(dead_code)]
impl<T: Ord> FourHeap<T> {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn push(&mut self, value: T) {
        self.data.push(value);
        let mut index = self.data.len() - 1;
        while index > 0 {
            let parent = (index - 1) / 4;
            if self.data[parent] >= self.data[index] {
                break;
            }
            self.data.swap(parent, index);
            index = parent;
        }
    }

    fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn pop(&mut self) -> Option<T> {
        let last = self.data.pop()?;
        if self.data.is_empty() {
            return Some(last);
        }

        let root = std::mem::replace(&mut self.data[0], last);
        let len = self.data.len();
        let mut index = 0usize;
        loop {
            let child0 = index * 4 + 1;
            if child0 >= len {
                break;
            }

            let child_end = (child0 + 4).min(len);
            let mut max_child = child0;
            for child in (child0 + 1)..child_end {
                if self.data[child] > self.data[max_child] {
                    max_child = child;
                }
            }

            if self.data[index] >= self.data[max_child] {
                break;
            }
            self.data.swap(index, max_child);
            index = max_child;
        }

        Some(root)
    }

    fn drain(&mut self) -> std::vec::Drain<'_, T> {
        self.data.drain(..)
    }

    fn clear(&mut self) {
        self.data.clear();
    }
}

type Frontier<T> = BinaryHeap<T>;

#[derive(Debug, Clone, Default)]
struct LodSplat {
    center: [f16; 3],
    size: f16,
    child_start: u32,
    child_count: u16,
    // Raycast covering radius around `center`, see update_radii(). Fits in the
    // struct padding, so it costs no memory.
    radius: f16,
}

impl LodSplat {
    fn new_f16(center: [f16; 3], size: f16, child_start: u32, child_count: u16) -> Self {
        Self { center, size, child_start, child_count, radius: f16::ZERO }
    }

    // Decode the 4-word lodTree encoding (see spark_lib encode_lod_tree)
    fn from_words(words: [u32; 4]) -> Self {
        let center = [
            f16::from_bits((words[0] & 0xffff) as u16),
            f16::from_bits((words[0] >> 16) as u16),
            f16::from_bits((words[1] & 0xffff) as u16),
        ];
        let size = f16::from_bits((words[1] >> 16) as u16);
        let mut splat = Self::new_f16(center, size, words[3], (words[2] & 0xffff) as u16);
        // Own raycast ellipsoid bound, grown by update_radii() to cover descendants.
        // Adds the f16 center rounding error: half ulp <= |c| * 2^-11 (+ subnormal floor).
        let extent = f16::from_bits((words[2] >> 16) as u16).to_f32();
        let own = LOD_RAYCAST_MAX_SIGMA * extent + splat.center().length() / 2048.0 + 1.0e-7;
        splat.radius = f16_round_up(own * (1.0 + 1.0e-5));
        splat
    }

    #[allow(dead_code)]
    fn new(center: Vec3, size: f32, child_start: u32, child_count: u16) -> Self {
        let center = center.to_array().map(f16::from_f32);
        let size = f16::from_f32(size);
        Self::new_f16(center, size, child_start, child_count)
    }

    fn center(&self) -> Vec3A {
        Vec3A::from_array(self.center.map(|x| x.to_f32()))
    }

    fn size(&self) -> f32 {
        self.size.to_f32()
    }
}

// #[derive(Debug, Clone, Default)]
// struct LodSplat {
//     center: Vec3,
//     size: f32,
//     child_start: u32,
//     child_count: u16,
// }

// impl LodSplat {
//     fn new_f16(center: [f16; 3], size: f16, child_start: u32, child_count: u16) -> Self {
//         let center = Vec3::from_array(center.map(|x| x.to_f32()));
//         let size = size.to_f32();
//         Self::new(center, size, child_start, child_count)
//     }

//     fn new(center: Vec3, size: f32, child_start: u32, child_count: u16) -> Self {
//         Self { center, size, child_start, child_count }
//     }

//     fn center(&self) -> Vec3A {
//         self.center.to_vec3a()
//     }

//     fn size(&self) -> f32 {
//         self.size
//     }
// }

#[derive(Debug, Clone, Default)]
struct LodTree {
    splats: Rc<RefCell<Vec<LodSplat>>>,
    page_to_chunk: Vec<u32>,
    chunk_to_page: Vec<u32>,
    // tree index -> parent tree index (0xFFFFFFFF unknown), for paged radius
    // updates: an arrival recomputes only the ancestors of its nodes
    parents: Vec<u32>,
    // chunks arrived since the last radii flush, see flush_dirty_radii()
    dirty_chunks: Vec<u32>,
    // flush_dirty_radii() scratch, kept so updates do not allocate
    dirty_nodes: BinaryHeap<u32>,
}

struct LodState {
    next_id: u32,
    // Raycast radii are maintained only where enabled by set_lod_radii() (the
    // pick worker); the traversal-only LoD worker skips their cost entirely.
    radii: bool,
    lod_trees: AHashMap<u32, LodTree>,
    frontier: Frontier<(OrderedFloat<f32>, u32, u32)>,
    output: Vec<(u32, u32)>,
    touched: Vec<(u32, u32)>,
    touched_set: AHashSet<(u32, u32)>,
    buffer: Vec<u32>,
}

impl LodState {
    fn new() -> Self {
        Self {
            next_id: 1000,
            radii: false,
            lod_trees: AHashMap::new(),
            frontier: Frontier::new(),
            output: Vec::new(),
            touched: Vec::new(),
            touched_set: AHashSet::new(),
            buffer: Vec::new(),
        }
    }
}

thread_local! {
    static STATE: RefCell<LodState> = RefCell::new(LodState::new());
}

fn set_lod_tree_data(state: &mut LodState, lod_id: u32, page_base: u32, _chunk_base: u32, count: u32, lod_tree_data: &Uint32Array) {
    let lod_tree = state.lod_trees.get(&lod_id).unwrap();
    let mut splats = lod_tree.splats.borrow_mut();

    if state.buffer.is_empty() {
        state.buffer.resize(MAX_SPLAT_CHUNK * 4, 0);
    }

    if page_base + count > splats.len() as u32 {
        let new_size = (splats.len() * 2).max((page_base + count) as usize);
        splats.resize_with(new_size, Default::default);
    }

    let mut index = 0;
    while index < count {
        let chunk = (count - index).min(MAX_SPLAT_CHUNK as u32);
        let buffer = &mut state.buffer[0..(chunk * 4) as usize];
        lod_tree_data.subarray(index * 4, (index + chunk) * 4).copy_to(buffer);

        for i in 0..chunk {
            let i4 = i * 4;
            let words: [u32; 4] = array::from_fn(|j| buffer[i4 as usize + j]);
            splats[(page_base + index + i) as usize] = LodSplat::from_words(words);
        }
        index += chunk;
    }
}

#[wasm_bindgen]
pub fn new_lod_tree(capacity: u32) -> Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        let lod_id = state.next_id;
        let splats = Vec::with_capacity(capacity as usize);
        let splats = Rc::new(RefCell::new(splats));
        let page_capacity = capacity.div_ceil(65536);
        let page_to_chunk = Vec::with_capacity(page_capacity as usize);
        let chunk_to_page: Vec<u32> = Vec::with_capacity(page_capacity as usize);
        state.lod_trees.insert(lod_id, LodTree { splats, page_to_chunk, chunk_to_page, ..Default::default() });
        state.next_id += 1;

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(lod_id)).unwrap();

        Ok(result)
    })
}

#[wasm_bindgen]
pub fn new_shared_lod_tree(orig_lod_id: u32) -> Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        let lod_tree = state.lod_trees.get(&orig_lod_id).unwrap();
        let splats = lod_tree.splats.clone();
        let page_to_chunk = Vec::with_capacity(lod_tree.page_to_chunk.capacity());
        let chunk_to_page = Vec::with_capacity(lod_tree.chunk_to_page.capacity());

        let new_lod_id = state.next_id;
        state.next_id += 1;
        state.lod_trees.insert(new_lod_id, LodTree { splats, page_to_chunk, chunk_to_page, ..Default::default() });

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(new_lod_id)).unwrap();
        Ok(result)
    })
}

#[wasm_bindgen]
pub fn init_lod_tree(num_splats: u32, lod_tree: Uint32Array) -> Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        let lod_id = state.next_id;
        let pages = num_splats.div_ceil(65536);
        let splats = Vec::with_capacity(num_splats as usize);
        let splats = Rc::new(RefCell::new(splats));
        let page_to_chunk = (0..pages).collect();
        let chunk_to_page: Vec<u32> = (0..pages).collect();
        state.lod_trees.insert(lod_id, LodTree { splats, page_to_chunk, chunk_to_page, ..Default::default() });
        state.next_id += 1;

        set_lod_tree_data(state, lod_id, 0, 0, num_splats, &lod_tree);
        if state.radii {
            let LodTree { splats, chunk_to_page, .. } = &state.lod_trees[&lod_id];
            update_radii(&mut splats.borrow_mut(), chunk_to_page, 0, num_splats);
        }

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(lod_id)).unwrap();

        Ok(result)
    })
}

/// Enable raycast covering radii in this worker: computed for trees created
/// afterwards, and eagerly after every update_lod_trees(). Call before any tree
/// is created; raycast_lod_tree() requires it.
#[wasm_bindgen]
pub fn set_lod_radii(enabled: bool) {
    STATE.with_borrow_mut(|state| state.radii = enabled)
}

#[wasm_bindgen]
pub fn dispose_lod_tree(lod_id: u32) {
    STATE.with_borrow_mut(|state| {
        state.lod_trees.remove(&lod_id);
    })
}

#[wasm_bindgen]
pub fn update_lod_trees(lod_ids: &[u32], page_bases: &[u32], chunk_bases: &[u32], counts: &[u32], lod_trees: &Array) -> Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        for (&lod_id, &page_base, &chunk_base, &count, lod_tree_data) in izip!(lod_ids, page_bases, chunk_bases, counts, lod_trees.iter()) {
            let lod_tree = state.lod_trees.get_mut(&lod_id).unwrap();
            let pages = count.div_ceil(65536);
            let base_page = page_base >> 16;
            let base_chunk = chunk_base >> 16;

            if (base_page + pages) > lod_tree.page_to_chunk.len() as u32 {
                lod_tree.page_to_chunk.resize((base_page + pages) as usize, 0xFFFFFFFF);
            }
            if (base_chunk + pages) > lod_tree.chunk_to_page.len() as u32 {
                lod_tree.chunk_to_page.resize((base_chunk + pages) as usize, 0xFFFFFFFF);
            }

            if lod_tree_data.is_falsy() {
                for page in 0..pages {
                    lod_tree.page_to_chunk[(base_page + page) as usize] = 0xFFFFFFFF;
                    lod_tree.chunk_to_page[(base_chunk + page) as usize] = 0xFFFFFFFF;
                }    
            } else {
                for page in 0..pages {
                    lod_tree.page_to_chunk[(base_page + page) as usize] = base_chunk + page;
                    lod_tree.chunk_to_page[(base_chunk + page) as usize] = base_page + page;
                }

                let lod_tree_data = Uint32Array::from(lod_tree_data);
                set_lod_tree_data(state, lod_id, page_base, chunk_base, count, &lod_tree_data);
                if state.radii {
                    mark_chunks_arrived(state.lod_trees.get_mut(&lod_id).unwrap(), chunk_base, chunk_base + count);
                }
            }
        }
        // Eager: the next raycast finds radii ready instead of paying the flush
        flush_all_dirty_radii(&mut state.lod_trees);

        let result = Object::new();
        // for (&lod_id, lod_tree) in state.lod_trees.iter() {
        //     let entry = Object::new();
        //     Reflect::set(&entry, &JsValue::from_str("pageToChunk"), &JsValue::from(lod_tree.page_to_chunk.clone())).unwrap();
        //     Reflect::set(&entry, &JsValue::from_str("chunkToPage"), &JsValue::from(lod_tree.chunk_to_page.clone())).unwrap();
        //     Reflect::set(&result, &JsValue::from_str(lod_id.to_string().as_str()), &JsValue::from(entry)).unwrap();
        // }
        Ok(result)
    })
}

#[allow(dead_code)]
struct LodInstance<'a> {
    lod_id: u32,
    splats: Ref<'a, Vec<LodSplat>>,
    page_to_chunk: &'a [u32],
    chunk_to_page: &'a [u32],
    origin: Vec3A,
    forward: Vec3A,
    right: Vec3A,
    up: Vec3A,
    output: Vec<u32>,
    lod_scale: f32,
    outside_foveate: f32,
    behind_foveate: f32,
    cone_dot0: f32,
    cone_dot: f32,
    cone_foveate: f32,
}

#[allow(dead_code)]
fn children_resident(child_count: u16, child_start: u32, instance: &LodInstance) -> bool {
    // Check endpoints, okay since child_count <= 65535
    for child in [child_start, child_start + child_count as u32 - 1] {
        if !is_resident(child, instance) {
            return false;
        }
    }
    true
}

#[allow(dead_code)]
fn is_resident(index: u32, instance: &LodInstance) -> bool {
    let chunk = (index >> 16) as usize;
    if chunk >= instance.chunk_to_page.len() {
        false
    } else {
        instance.chunk_to_page[chunk] != 0xFFFFFFFF
    }
}

#[wasm_bindgen]
pub fn get_lod_tree_level(lod_id: u32, level: u32) -> anyhow::Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        let LodState { lod_trees, .. } = state;
        let lod_tree = lod_trees.get(&lod_id).unwrap();
        let splats = lod_tree.splats.borrow();

        let root_size = splats[0].size();
        let level_size = root_size / (1.25f32.powi(level as i32));

        let mut nodes = vec![0];
        let mut output_nodes = Vec::new();

        while !nodes.is_empty() {
            let mut new_nodes = Vec::new();
            for node in nodes {
                let splat = &splats[node as usize];
                let &LodSplat { child_count, child_start, .. } = splat;
                if splat.size() <= level_size {
                    output_nodes.push(node);
                } else {
                    for child in child_start..child_start + child_count as u32 {
                        new_nodes.push(child);
                    }
                }
            }
            nodes = new_nodes;
        }

        let output = Uint32Array::new_with_length(output_nodes.len() as u32);
        for (i, node) in output_nodes.into_iter().enumerate() {
            output.set_index(i as u32, node);
        }

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("indices"), &JsValue::from(output)).unwrap();
        Ok(result)
    })
}

#[wasm_bindgen]
pub fn traverse_lod_trees(
    max_splats: u32, pixel_scale_limit: f32, _last_pixel_limit: Option<f32>,
    lod_ids: &[u32], root_pages: &[u32],
    view_to_objects: &[f32], lod_scales: &[f32],
    behind_foveates: &[f32], cone_foveates: &[f32],
    cone_fov0s: &[f32], cone_fovs: &[f32],
) -> anyhow::Result<Object, JsValue> {
    let max_splats = max_splats as usize;
    let num_instances = lod_ids.len();
    if view_to_objects.len() != num_instances * 16 {
        return Err(JsValue::from_str("Invalid view_to_objects length"));
    }
    if lod_scales.len() != num_instances {
        return Err(JsValue::from_str("Invalid lod_scales length"));
    }
    if behind_foveates.len() != num_instances {
        return Err(JsValue::from_str("Invalid behind_foveates length"));
    }
    if cone_foveates.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_foveates length"));
    }
    if cone_fov0s.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_fov0s length"));
    }
    if cone_fovs.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_fovs length"));
    }

    STATE.with_borrow_mut(|state| {
        let LodState { lod_trees, frontier, output, touched, touched_set, .. } = state;
        let instances: Vec<_> = lod_ids.iter().enumerate().map(|(index, &lod_id)| {
            let lod_tree = lod_trees.get(&lod_id).unwrap();
            let LodTree { splats, page_to_chunk, chunk_to_page, .. } = &lod_tree;
            let i16 = index * 16;
            let forward = Vec3A::from_slice(&view_to_objects[(i16 + 8)..(i16 + 11)]).normalize().map(|x| -x);
            let origin = Vec3A::from_slice(&view_to_objects[(i16 + 12)..(i16 + 15)]);
            let lod_scale = lod_scales[index];
            let behind_foveate = behind_foveates[index];
            let cone_foveate = cone_foveates[index];
            let cone_dot0 = if cone_fov0s[index] > 0.0 { (0.5 * cone_fov0s[index].clamp(0.0, 180.0)).to_radians().cos() } else { 1.0 };
            let cone_dot = if cone_fovs[index] > 0.0 { (0.5 * cone_fovs[index].clamp(0.0, 180.0)).to_radians().cos() } else { 1.0 };
            let cone_dot = cone_dot.min(cone_dot0);
            (lod_id, splats.borrow(), page_to_chunk, chunk_to_page, origin, forward, lod_scale, behind_foveate, cone_foveate, cone_dot0, cone_dot)
        }).collect();

        let mut num_splats = 0;
        frontier.clear();
        output.clear();
        output.reserve(max_splats);
        touched.clear();
        touched_set.clear();

        for (inst_index, instance) in instances.iter().enumerate() {
            let (lod_id, splats, ..) = instance;
            let root_page = root_pages[inst_index];
            let root_page = if root_page == 0xFFFFFFFF { 0 } else { root_page };
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
            let LodSplat { child_count, child_start, .. } = splats[paged_index as usize];

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
                let pixel_scale = compute_pixel_scale(&splats[paged_index as usize], instance);
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

        let mut instance_counts = vec![0; num_instances];
        for &(inst_index, _) in output.iter() {
            instance_counts[inst_index as usize] += 1;
        }

        let mut instance_outputs = Vec::with_capacity(num_instances);
        for counts in instance_counts {
            instance_outputs.push(Vec::with_capacity(counts));
        }

        for &(inst_index, paged_index) in output.iter() {
            instance_outputs[inst_index as usize].push(paged_index);
        }

        let instance_indices = Array::new();

        for (inst_index, instance_output) in instance_outputs.iter_mut().enumerate() {
            // instance_output.sort_unstable();
            let rows = instance_output.len().div_ceil(16384);
            let capacity = rows * 16384;
            let output = Uint32Array::new_with_length(capacity as u32);
            output.subarray(0, instance_output.len() as u32).copy_from(instance_output);

            let result = Object::new();
            let lod_id = instances[inst_index].0;
            Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(lod_id)).unwrap();
            Reflect::set(&result, &JsValue::from_str("numSplats"), &JsValue::from(instance_output.len() as u32)).unwrap();
            Reflect::set(&result, &JsValue::from_str("indices"), &JsValue::from(output)).unwrap();
            instance_indices.push(&JsValue::from(result));
        }

        let out_chunks = Array::new();

        for &(inst_index, chunk) in touched.iter() {
            let pair = Array::new();
            pair.push(&JsValue::from(inst_index));
            pair.push(&JsValue::from(chunk));
            out_chunks.push(&JsValue::from(pair));
        }

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("pixelLimit"), &JsValue::from(min_pixel_scale)).unwrap();
        Reflect::set(&result, &JsValue::from_str("instanceIndices"), &JsValue::from(instance_indices)).unwrap();
        Reflect::set(&result, &JsValue::from_str("chunks"), &JsValue::from(out_chunks)).unwrap();
        Reflect::set(&result, &JsValue::from_str("outputSize"), &JsValue::from(output_size)).unwrap();
        Reflect::set(&result, &JsValue::from_str("frontierSize"), &JsValue::from(frontier_size)).unwrap();
        Reflect::set(&result, &JsValue::from_str("leafCount"), &JsValue::from(leaf_count)).unwrap();
        Ok(result)
    })
}

fn compute_pixel_scale<'a>(
    splat: &LodSplat,
    instance: &(u32, Ref<'a, Vec<LodSplat>>, &Vec<u32>, &Vec<u32>, Vec3A, Vec3A, f32, f32, f32, f32, f32),
) -> f32 {
    let &(_, _, _, _, origin, forward, lod_scale, behind_foveate, cone_foveate, cone_dot0, cone_dot) = instance;
    let center = splat.center();
    let delta = center - origin;
    let distance = delta.length().max(1.0e-6);
    let inv_distance = 1.0 / distance;
    let pixel_scale = splat.size() * inv_distance;
    let pixel_scale = pixel_scale * lod_scale;

    let forward_dot = delta.dot(forward);
    let foveate = if forward_dot <= 0.0 {
        behind_foveate
    } else {
        let dot = forward_dot * inv_distance;
        if dot >= cone_dot0 {
            1.0
        } else if dot >= cone_dot {
            let t = (dot - cone_dot) / (cone_dot0 - cone_dot);
            cone_foveate + (1.0 - cone_foveate) * t
        } else {
            let t = dot / cone_dot;
            behind_foveate + (cone_foveate - behind_foveate) * t
        }
    };
    foveate * pixel_scale
}

#[wasm_bindgen]
pub fn dynamic_traverse_lod_trees(
    max_splats: u32, pixel_scale_limit: f32, _last_pixel_limit: Option<f32>,
    // lod_instances: &Array,
    lod_ids: &[u32], root_pages: &[u32],
    view_to_objects: &[f32], lod_scales: &[f32],
    behind_foveates: &[f32], cone_foveates: &[f32],
    cone_fov0s: &[f32], cone_fovs: &[f32],
    // readback: Uint32Array,
    // flag: bool,
) -> anyhow::Result<Object, JsValue> {

    let max_splats = max_splats as usize;
    let num_instances = lod_ids.len();
    if view_to_objects.len() != num_instances * 16 {
        return Err(JsValue::from_str("Invalid view_to_objects length"));
    }
    if lod_scales.len() != num_instances {
        return Err(JsValue::from_str("Invalid lod_scales length"));
    }
    if behind_foveates.len() != num_instances {
        return Err(JsValue::from_str("Invalid behind_foveates length"));
    }
    if cone_foveates.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_foveates length"));
    }
    if cone_fov0s.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_fov0s length"));
    }
    if cone_fovs.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_fovs length"));
    }

    STATE.with_borrow_mut(|state| {
        let LodState { lod_trees, .. } = state;
        let instances: Vec<_> = lod_ids.iter().enumerate().map(|(index, &lod_id)| {
            let lod_tree = lod_trees.get(&lod_id).unwrap();
            let LodTree { splats, page_to_chunk, chunk_to_page, .. } = &lod_tree;
            let i16 = index * 16;
            let forward = Vec3A::from_slice(&view_to_objects[(i16 + 8)..(i16 + 11)]).normalize().map(|x| -x);
            let origin = Vec3A::from_slice(&view_to_objects[(i16 + 12)..(i16 + 15)]);
            let lod_scale = lod_scales[index];
            let behind_foveate = behind_foveates[index];
            let cone_foveate = cone_foveates[index];
            let cone_dot0 = if cone_fov0s[index] > 0.0 { (0.5 * cone_fov0s[index]).to_radians().cos() } else { 1.0 };
            let cone_dot = if cone_fovs[index] > 0.0 { (0.5 * cone_fovs[index]).to_radians().cos() } else { 1.0 };
            (lod_id, splats.borrow(), page_to_chunk, chunk_to_page, origin, forward, lod_scale, behind_foveate, cone_foveate, cone_dot0, cone_dot)
        }).collect();

        let mut lod_chunk_max: AHashMap<u32, Vec<f32>> = AHashMap::new();

        let mut outputs = Vec::with_capacity(num_instances);
        for (inst_index, instance) in instances.iter().enumerate() {
            let (lod_id, splats, ..) = instance;
            let root_page = root_pages[inst_index];
            let root_page = if root_page == 0xFFFFFFFF { 0 } else { root_page };
            let root_index = root_page << 16;
            let root_scale = compute_pixel_scale(&splats[root_index as usize], instance);
            let frontier = vec![(root_index, root_scale)];
            let instance_output = Vec::with_capacity(1000);
            outputs.push((instance_output, frontier));

            let chunk_max = lod_chunk_max.entry(*lod_id).or_default();
            if chunk_max.is_empty() {
                chunk_max.resize(1, 0.0);
            }
            chunk_max[0] = f32::INFINITY;
        }


        let mut leaf_count = 0;
        let mut missing_count = 0;
        let mut min_pixel_scale = f32::INFINITY;

        let mut current_scale = pixel_scale_limit * 100.0;

        loop {
            let iterator = instances.iter().zip(outputs);
            outputs = Vec::with_capacity(num_instances);

            let mut output_count = 0;

            for (instance, (mut instance_output, mut stack)) in iterator {
                let (lod_id, splats, _, chunk_to_page, ..) = instance;
                let chunk_max = lod_chunk_max.entry(*lod_id).or_default();
                let mut frontier = Vec::with_capacity(stack.len());

                while let Some((paged_index, pixel_scale)) = stack.pop() {
                    min_pixel_scale = min_pixel_scale.min(pixel_scale);
                    if pixel_scale <= current_scale {
                        frontier.push((paged_index, pixel_scale));
                        continue;
                    }

                    let LodSplat { child_count, child_start, .. } = splats[paged_index as usize];
                    if child_count == 0 {
                        instance_output.push((paged_index, pixel_scale));
                        leaf_count += 1;
                        continue;
                    }

                    let first_chunk = child_start >> 16;
                    let last_chunk = (child_start + child_count as u32 - 1) >> 16;

                    if last_chunk as usize >= chunk_max.len() {
                        chunk_max.resize(last_chunk as usize + 1, 0.0);
                    }
                    chunk_max[first_chunk as usize] = chunk_max[first_chunk as usize].max(pixel_scale);
                    chunk_max[last_chunk as usize] = chunk_max[last_chunk as usize].max(pixel_scale);
        
                    if last_chunk as usize >= chunk_to_page.len() {
                        instance_output.push((paged_index, pixel_scale));
                        missing_count += 1;
                        continue;
                    }
                    let first_page = chunk_to_page[first_chunk as usize];
                    let last_page = chunk_to_page[last_chunk as usize];
        
                    if first_page == 0xFFFFFFFF || last_page == 0xFFFFFFFF {
                        instance_output.push((paged_index, pixel_scale));
                        missing_count += 1;
                        continue;
                    }
        
                    for child in child_start..child_start + child_count as u32 {
                        let child_chunk = (child >> 16) as usize;
                        let child_page = chunk_to_page[child_chunk];
                        let paged_index = (child_page << 16) | (child & 0xffff);
                        let pixel_scale = compute_pixel_scale(&splats[paged_index as usize], instance);
                        if pixel_scale <= current_scale {
                            if pixel_scale <= pixel_scale_limit {
                                instance_output.push((paged_index, pixel_scale));
                            } else {
                                frontier.push((paged_index, pixel_scale));
                            }
                        } else {
                            stack.push((paged_index, pixel_scale));
                        }
                    }
                }

                output_count += instance_output.len() + frontier.len();
                outputs.push((instance_output, frontier));
            }

            let ratio = output_count as f32 / max_splats as f32;
            // let next_scale = (0.9 * current_scale * ratio.powf(1.0 / 1.5)).max(pixel_scale_limit);
            let next_scale = 0.99 * current_scale * ratio.powf(1.0 / 2.0);
            // let next_scale = (0.9 * current_scale).max(pixel_scale_limit);
            let next_scale = next_scale.max(0.5 * current_scale);
            let next_scale = next_scale.max(pixel_scale_limit);

            let no_frontier = outputs.iter().all(|(_, frontier)| frontier.is_empty());

            if no_frontier || (next_scale == current_scale) || output_count >= max_splats {
                break;
            }

            current_scale = next_scale;
            // loop
        };

        let mut touched: Vec<_> = lod_chunk_max.into_iter().flat_map(|x| {
            let (lod_id, chunk_max) = x;
            chunk_max.into_iter().enumerate().filter_map(move |(chunk, max)| {
                if max == 0.0 {
                    None
                } else {
                    Some((OrderedFloat(-max), lod_id, chunk as u32))
                }
            })
        }).collect();
        touched.sort_unstable();

        let instance_indices = Array::new();
        let mut output_size = 0;

        for (inst_index, (mut instance_output, frontier)) in outputs.into_iter().enumerate() {
            output_size += frontier.len();
            instance_output.extend(frontier);
            let rows = instance_output.len().div_ceil(16384);
            let capacity = rows * 16384;
            let output = Uint32Array::new_with_length(capacity as u32);
            let output_u32: Vec<u32> = instance_output.into_iter().map(|(paged_index, _)| paged_index).collect();
            output.subarray(0, output_u32.len() as u32).copy_from(&output_u32);

            let result = Object::new();
            let lod_id = instances[inst_index].0;
            Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(lod_id)).unwrap();
            Reflect::set(&result, &JsValue::from_str("numSplats"), &JsValue::from(output_u32.len() as u32)).unwrap();
            Reflect::set(&result, &JsValue::from_str("indices"), &JsValue::from(output)).unwrap();
            instance_indices.push(&JsValue::from(result));
        }

        let out_chunks = Array::new();

        for &(_, lod_id, chunk) in touched.iter() {
            let pair = Array::new();
            pair.push(&JsValue::from(lod_id));
            pair.push(&JsValue::from(chunk));
            out_chunks.push(&JsValue::from(pair));
        }

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("pixelLimit"), &JsValue::from(min_pixel_scale)).unwrap();
        Reflect::set(&result, &JsValue::from_str("instanceIndices"), &JsValue::from(instance_indices)).unwrap();
        Reflect::set(&result, &JsValue::from_str("chunks"), &JsValue::from(out_chunks)).unwrap();
        Reflect::set(&result, &JsValue::from_str("outputSize"), &JsValue::from(output_size)).unwrap();
        Reflect::set(&result, &JsValue::from_str("leafCount"), &JsValue::from(leaf_count)).unwrap();
        Reflect::set(&result, &JsValue::from_str("missingCount"), &JsValue::from(missing_count)).unwrap();
        Ok(result)
    })
}

// LoD tree raycasting. Every resident node stores a covering radius bounding
// its own ellipsoid and those of all resident descendants it can be traversed
// into, so a best-first walk ordered by covering-sphere entry distance visits
// stop nodes (leaves, or nodes whose children are not resident) in an order
// where any hit is at or beyond the sphere entry distance. The worker only has
// the 16-byte lodTree nodes, so it returns ordered candidates and the main
// thread, which holds the full splat data, runs the exact test.

/// Largest raycast `sigma` (ellipsoid semi-axes in standard deviations, times
/// the LoD opacity rescale) covered by the radii.
pub const LOD_RAYCAST_MAX_SIGMA: f32 = 2.0;

// Paged index (page << 16 | offset) of tree index `index`, if its chunk is resident
fn resident_index(index: u32, chunk_to_page: &[u32]) -> Option<u32> {
    let page = *chunk_to_page.get((index >> 16) as usize)?;
    (page != 0xFFFFFFFF).then_some((page << 16) | (index & 0xffff))
}

// Whether raycasting descends into the children of node `index`: same endpoint
// residency check as traverse_lod_trees, and children must follow their parent
// so a single reverse pass computes radii bottom-up.
fn descends(splat: &LodSplat, index: u32, chunk_to_page: &[u32]) -> bool {
    let LodSplat { child_start, child_count, .. } = *splat;
    child_count > 0
        && child_start > index
        && resident_index(child_start, chunk_to_page).is_some()
        && resident_index(child_start + child_count as u32 - 1, chunk_to_page).is_some()
}

fn f16_round_up(x: f32) -> f16 {
    let h = f16::from_f32(x);
    if h.to_f32() < x { f16::from_bits(h.to_bits() + 1) } else { h }
}

// Grow covering radii of resident tree indices [start, end) to cover their
// resident children, children first. Radii only grow (rounded up), so
// recomputing a range is safe and stale radii stay conservative; a page gets
// fresh radii when new data is decoded into it (LodSplat::from_words).
fn update_radii(splats: &mut [LodSplat], chunk_to_page: &[u32], start: u32, end: u32) {
    for index in (start..end).rev() {
        update_radius(splats, chunk_to_page, index);
    }
}

// Grow the covering radius of resident tree index `index` to cover its resident
// children (assumed final). Returns whether the radius changed, i.e. whether
// its parent must be recomputed too.
fn update_radius(splats: &mut [LodSplat], chunk_to_page: &[u32], index: u32) -> bool {
    let Some(paged) = resident_index(index, chunk_to_page) else { return false };
    let Some(splat) = splats.get(paged as usize).cloned() else { return false };
    if !descends(&splat, index, chunk_to_page) {
        return false;
    }
    let center = splat.center();
    let mut radius = splat.radius.to_f32();
    for child in splat.child_start..splat.child_start + splat.child_count as u32 {
        let child = resident_index(child, chunk_to_page).and_then(|paged| splats.get(paged as usize));
        let reach = child.map_or(f32::INFINITY, |child| center.distance(child.center()) + child.radius.to_f32());
        radius = radius.max(reach * (1.0 + 1.0e-5));
    }
    let radius = f16_round_up(radius);
    splats[paged as usize].radius = radius;
    radius != splat.radius
}

// Newly resident tree indices [start, end) of a paged tree: record the parent
// of their children and mark their chunks dirty. Radii are computed by
// flush_dirty_radii(), so loading pays O(count) only.
// Evictions need no update: radii then only cover more than needed.
fn mark_chunks_arrived(lod_tree: &mut LodTree, start: u32, end: u32) {
    let LodTree { splats, chunk_to_page, parents, dirty_chunks, .. } = lod_tree;
    let splats = splats.borrow();
    for index in start..end {
        let Some(paged) = resident_index(index, chunk_to_page) else { continue };
        let Some(&LodSplat { child_start, child_count, .. }) = splats.get(paged as usize) else { continue };
        // Same rule as descends(): children follow their parent, so parent
        // links strictly decrease and propagation always terminates.
        if child_count == 0 || child_start <= index {
            continue;
        }
        let children = child_start as usize..child_start as usize + child_count as usize;
        if children.end > parents.len() {
            parents.resize(children.end, 0xFFFFFFFF);
        }
        parents[children].fill(index);
    }
    dirty_chunks.extend((start >> 16)..end.div_ceil(65536));
}

// Flush every tree with pending arrivals. Trees sharing one splats Vec (paged)
// are borrowed one at a time.
fn flush_all_dirty_radii(lod_trees: &mut AHashMap<u32, LodTree>) {
    for lod_tree in lod_trees.values_mut() {
        flush_dirty_radii(lod_tree);
    }
}

// Radii of the dirty chunks, then of only those ancestor nodes whose radius
// actually changes: cost ~ arrived nodes + their changed ancestors, independent
// of tree size. Order: children always have a higher tree index than their
// parent, so processing dirty chunks and dirty nodes in one descending sweep
// finalizes every child before its parent. Radii only grow, so stopping where
// a radius is unchanged keeps every ancestor covering.
fn flush_dirty_radii(lod_tree: &mut LodTree) {
    if lod_tree.dirty_chunks.is_empty() {
        return;
    }
    let LodTree { splats, chunk_to_page, parents, dirty_chunks, dirty_nodes, .. } = lod_tree;
    let mut splats = splats.borrow_mut();
    dirty_chunks.sort_unstable();
    dirty_chunks.dedup();
    while let Some(chunk) = dirty_chunks.pop() {
        let (first, last) = (chunk << 16, (chunk << 16) | 0xFFFF);
        pop_dirty_nodes(&mut splats, chunk_to_page, parents, dirty_nodes, Some(last));
        if resident_index(first, chunk_to_page).is_none() {
            continue;
        }
        update_radii(&mut splats, chunk_to_page, first, last + 1);
        // Parents outside this chunk (siblings are contiguous: skip repeats)
        let mut prev = 0xFFFFFFFF;
        for &parent in parents.get(first as usize..(last as usize + 1).min(parents.len())).unwrap_or(&[]) {
            if parent != prev && parent < first {
                dirty_nodes.push(parent);
                prev = parent;
            }
        }
    }
    pop_dirty_nodes(&mut splats, chunk_to_page, parents, dirty_nodes, None);
}

// Recompute queued nodes with index > `above` (all if None), highest first,
// queueing the parent of each node whose radius changed.
fn pop_dirty_nodes(splats: &mut [LodSplat], chunk_to_page: &[u32], parents: &[u32], dirty_nodes: &mut BinaryHeap<u32>, above: Option<u32>) {
    let mut prev = 0xFFFFFFFF;
    while let Some(&index) = dirty_nodes.peek() {
        if above.is_some_and(|above| index <= above) {
            break;
        }
        dirty_nodes.pop();
        // Duplicates pop consecutively and a popped node is never queued again
        if index == prev {
            continue;
        }
        prev = index;
        if update_radius(splats, chunk_to_page, index) {
            if let Some(&parent) = parents.get(index as usize).filter(|&&parent| parent != 0xFFFFFFFF) {
                dirty_nodes.push(parent);
            }
        }
    }
}

struct LodRaycast {
    // Tree indices of stop nodes, in order of covering-sphere entry distance
    nodes: Vec<u32>,
    // Lower bound on the distance of any hit in a node not in `nodes`
    next_distance: f64,
    #[allow(dead_code)]
    visited: usize,
}

// Best-first walk of the resident tree along origin + t * dir (t in units of
// |dir|, like THREE.Raycaster after an object transform). Stops after
// `max_candidates` stop nodes.
#[allow(clippy::too_many_arguments)]
fn raycast_lod(
    splats: &[LodSplat], chunk_to_page: &[u32], root_page: u32,
    origin: [f64; 3], dir: [f64; 3], near: f64, far: f64, max_candidates: usize,
) -> LodRaycast {
    let length = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    let unit = dir.map(|x| x / length);
    // Covering sphere entry distance clamped to near, if the sphere overlaps [near, far].
    // Perpendicular distance from |v x unit| in f64: no cancellation for far nodes.
    let enter = |splat: &LodSplat| -> Option<f64> {
        let center = splat.center();
        let v: [f64; 3] = array::from_fn(|d| center[d] as f64 - origin[d]);
        let along = v[0] * unit[0] + v[1] * unit[1] + v[2] * unit[2];
        let cross = [v[1] * unit[2] - v[2] * unit[1], v[2] * unit[0] - v[0] * unit[2], v[0] * unit[1] - v[1] * unit[0]];
        let perp2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
        let radius = splat.radius.to_f32() as f64;
        if perp2 > radius * radius {
            return None;
        }
        let half = (radius * radius - perp2).sqrt();
        let (t0, t1) = ((along - half) / length, (along + half) / length);
        (t1 >= near && t0 <= far).then_some(t0.max(near))
    };

    let mut result = LodRaycast { nodes: Vec::new(), next_distance: f64::INFINITY, visited: 0 };
    let mut frontier = BinaryHeap::new();
    let root = root_page << 16;
    if let Some(t) = splats.get(root as usize).and_then(enter) {
        frontier.push(Reverse((OrderedFloat(t), 0u32, root)));
    }

    while let Some(&Reverse((OrderedFloat(t), index, paged))) = frontier.peek() {
        if result.nodes.len() >= max_candidates {
            result.next_distance = t;
            break;
        }
        frontier.pop();
        result.visited += 1;

        let splat = &splats[paged as usize];
        if !descends(splat, index, chunk_to_page) {
            result.nodes.push(index);
            continue;
        }
        for child in splat.child_start..splat.child_start + splat.child_count as u32 {
            let Some(child_paged) = resident_index(child, chunk_to_page) else { continue };
            if let Some(t) = splats.get(child_paged as usize).and_then(enter) {
                frontier.push(Reverse((OrderedFloat(t), child, child_paged)));
            }
        }
    }
    result
}

/// Ordered raycast candidates of a LoD tree, for SparkRenderer.raycastAsync().
/// `origin` and `dir` are in object space; distances are in units of |dir|.
/// Returns `{ nodes, nextDistance }`: tree indices (chunk << 16 | offset, mapped
/// to pages by the caller) of up to `max_candidates` stop nodes, and a lower
/// bound on the distance of any hit outside `nodes` (Infinity when the walk is
/// complete). An exact hit <= nextDistance among `nodes` is the closest hit.
#[wasm_bindgen]
pub fn raycast_lod_tree(
    lod_id: u32, root_page: u32,
    origin: &[f64], dir: &[f64], near: f64, far: f64, max_candidates: u32,
) -> Result<Object, JsValue> {
    if origin.len() != 3 || dir.len() != 3 {
        return Err(JsValue::from_str("Invalid origin or dir length"));
    }
    STATE.with_borrow_mut(|state| {
        if !state.radii {
            return Err(JsValue::from_str("LoD radii disabled in this worker, see set_lod_radii()"));
        }
        let lod_tree = state.lod_trees.get_mut(&lod_id).ok_or_else(|| JsValue::from_str("Invalid lod_id"))?;
        // Safety net: normally already flushed eagerly by update_lod_trees()
        flush_dirty_radii(lod_tree);
        let root_page = if root_page == 0xFFFFFFFF { 0 } else { root_page };
        let LodRaycast { nodes, next_distance, .. } = raycast_lod(
            &lod_tree.splats.borrow(), &lod_tree.chunk_to_page, root_page,
            [origin[0], origin[1], origin[2]], [dir[0], dir[1], dir[2]],
            near, far, max_candidates as usize,
        );

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("nodes"), &JsValue::from(Uint32Array::from(nodes.as_slice()))).unwrap();
        Reflect::set(&result, &JsValue::from_str("nextDistance"), &JsValue::from(next_distance)).unwrap();
        Ok(result)
    })
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use spark_lib::splat_encode::encode_lod_tree;

    use super::*;
    use crate::raycast::raycast_ellipsoid;

    const INVALID: u32 = 0xFFFFFFFF;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
        }
        fn range(&mut self, a: f32, b: f32) -> f32 {
            a + (b - a) * self.next()
        }
    }

    // Synthetic LoD tree in Spark order (BFS: contiguous children after parent),
    // with full splat data per node for the exact test.
    struct Tree {
        center: Vec<[f32; 3]>,
        scale: Vec<[f32; 3]>,
        quat: Vec<[f32; 4]>,
        opacity: Vec<f32>,
        parent: Vec<u32>,
        splats: Vec<LodSplat>,
        num_leaves: usize,
    }

    enum Node {
        Leaf(usize),
        Inner(Vec<Node>),
    }

    fn group(ids: &mut [usize], centers: &[[f32; 3]]) -> Node {
        if ids.len() == 1 {
            return Node::Leaf(ids[0]);
        }
        if ids.len() <= 4 {
            return Node::Inner(ids.iter().map(|&i| Node::Leaf(i)).collect());
        }
        let axis = (0..3)
            .max_by(|&a, &b| {
                let extent = |d: usize| {
                    let (lo, hi) = ids.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &i| (lo.min(centers[i][d]), hi.max(centers[i][d])));
                    hi - lo
                };
                extent(a).total_cmp(&extent(b))
            })
            .unwrap();
        ids.sort_by(|&a, &b| centers[a][axis].total_cmp(&centers[b][axis]));
        let quarter = ids.len().div_ceil(4);
        Node::Inner(ids.chunks_mut(quarter).map(|chunk| group(chunk, centers)).collect())
    }

    fn leaves_of(node: &Node, out: &mut Vec<usize>) {
        match node {
            Node::Leaf(i) => out.push(*i),
            Node::Inner(children) => children.iter().for_each(|c| leaves_of(c, out)),
        }
    }

    // `wall`: every other leaf is an opaque disk on the plane z = offset.z - 5,
    // the others fill a deep volume behind it (up to z = offset.z + 60),
    // hiding the rest of the scene from rays coming from -z.
    fn build_tree(num_leaves: usize, seed: u64, wall: bool) -> Tree {
        let mut rng = Rng(seed);
        let offset = [12.0, -3.0, 7.0];
        let mut leaf_center = Vec::new();
        let mut leaf_scale = Vec::new();
        let mut leaf_quat = Vec::new();
        let mut leaf_opacity = Vec::new();
        for i in 0..num_leaves {
            leaf_center.push(array::from_fn(|d| offset[d] + rng.range(-5.0, 5.0)));
            let mut scale: [f32; 3] = array::from_fn(|_| 10f32.powf(rng.range(-3.0, -1.3)));
            if i % 5 == 0 {
                scale[2] = 1.0e-5; // flat splat: disk branch
            }
            leaf_scale.push(scale);
            let q: [f32; 4] = array::from_fn(|_| rng.range(-1.0, 1.0));
            let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            leaf_quat.push(q.map(|x| x / norm));
            leaf_opacity.push(rng.range(0.1, 1.0));
            if wall && i % 2 == 0 {
                leaf_center[i][2] = offset[2] - 5.0;
                leaf_scale[i] = [0.06, 0.06, 1.0e-5];
                leaf_quat[i] = [0.0, 0.0, 0.0, 1.0];
                leaf_opacity[i] = 0.9;
            } else if wall {
                leaf_center[i][2] = offset[2] + rng.range(0.0, 60.0);
            }
        }

        let mut ids: Vec<usize> = (0..num_leaves).collect();
        let root = group(&mut ids, &leaf_center);

        let mut tree = Tree {
            center: Vec::new(), scale: Vec::new(), quat: Vec::new(), opacity: Vec::new(),
            parent: Vec::new(), splats: Vec::new(), num_leaves,
        };
        let mut queue = VecDeque::from([(root, INVALID)]);
        let mut next = 1u32;
        while let Some((node, parent)) = queue.pop_front() {
            let index = tree.center.len() as u32;
            let (center, scale, quat, opacity, child_count) = match &node {
                Node::Leaf(i) => (leaf_center[*i], leaf_scale[*i], leaf_quat[*i], leaf_opacity[*i], 0),
                Node::Inner(children) => {
                    let mut leaves = Vec::new();
                    leaves_of(&node, &mut leaves);
                    let n = leaves.len() as f32;
                    let center: [f32; 3] = array::from_fn(|d| leaves.iter().map(|&i| leaf_center[i][d]).sum::<f32>() / n);
                    let scale = array::from_fn(|d| {
                        let var = leaves.iter().map(|&i| (leaf_center[i][d] - center[d]).powi(2)).sum::<f32>() / n;
                        (0.5 * var.sqrt()).max(1.0e-3)
                    });
                    (center, scale, [0.0, 0.0, 0.0, 1.0], 1.2, children.len() as u16)
                }
            };
            let child_start = if child_count > 0 { next } else { 0 };
            let mut words = [0u32; 4];
            encode_lod_tree(&mut words, &center, opacity, &scale, child_count, child_start);
            tree.splats.push(LodSplat::from_words(words));
            tree.center.push(center);
            tree.scale.push(scale);
            tree.quat.push(quat);
            tree.opacity.push(opacity);
            tree.parent.push(parent);
            if let Node::Inner(children) = node {
                next += children.len() as u32;
                queue.extend(children.into_iter().map(|c| (c, index)));
            }
        }
        tree
    }

    fn identity_pages(tree: &Tree) -> Vec<u32> {
        (0..tree.splats.len().div_ceil(65536) as u32).collect()
    }

    struct Ray {
        origin: [f32; 3],
        dir: [f32; 3],
    }

    fn rays(tree: &Tree, count: usize, seed: u64) -> Vec<Ray> {
        let leaf_ids: Vec<usize> = (0..tree.splats.len()).filter(|&i| tree.splats[i].child_count == 0).collect();
        let mut rng = Rng(seed);
        (0..count)
            .map(|_| {
                let leaves = &leaf_ids;
                let target = tree.center[leaves[(rng.next() * (leaves.len() - 1) as f32) as usize]];
                let origin: [f32; 3] = array::from_fn(|d| target[d] + rng.range(-20.0, 20.0));
                let dir: [f32; 3] = array::from_fn(|d| target[d] + rng.range(-0.005, 0.005) - origin[d]);
                // non-unit dir, like a ray transformed into a scaled mesh
                let length = dir.iter().map(|x| x * x).sum::<f32>().sqrt() / rng.range(0.5, 2.0);
                Ray { origin, dir: dir.map(|x| x / length) }
            })
            .collect()
    }

    const MIN_OPACITY: f32 = 0.2;

    fn exact(tree: &Tree, i: usize, ray: &Ray, sigma: f32) -> Option<f32> {
        if tree.opacity[i] < MIN_OPACITY {
            return None;
        }
        raycast_ellipsoid(ray.origin, ray.dir, tree.opacity[i], tree.center[i], tree.scale[i], tree.quat[i], Some(sigma))
            .filter(|&t| t >= 0.0)
    }

    // Mirrors SparkRenderer.raycastAsync(): exact test of the candidates, then
    // retry with 8x more candidates and far clamped to the best hit so far, until
    // the closest hit is proven. Returns the hit and the total nodes visited.
    fn pick(tree: &Tree, splats: &[LodSplat], chunk_to_page: &[u32], ray: &Ray, sigma: f32, first_batch: usize) -> (Option<f32>, usize) {
        let (mut max_candidates, mut far, mut best, mut visited) = (first_batch, f64::INFINITY, None::<f32>, 0);
        loop {
            let result = raycast_lod(
                splats, chunk_to_page, 0,
                ray.origin.map(|x| x as f64), ray.dir.map(|x| x as f64), 0.0, far, max_candidates,
            );
            visited += result.visited;
            let hits = result.nodes.iter().filter_map(|&i| exact(tree, i as usize, ray, sigma));
            best = hits.chain(best).min_by(f32::total_cmp);
            if result.next_distance.is_infinite() || best.is_some_and(|t| t as f64 <= result.next_distance) {
                return (best, visited);
            }
            far = best.map_or(far, |t| t as f64);
            max_candidates *= 8;
        }
    }

    // Stop nodes of the resident tree (leaves or nodes whose children are not resident)
    fn resident_cut(tree: &Tree, chunk_to_page: &[u32]) -> Vec<usize> {
        let mut cut = Vec::new();
        let mut stack = vec![0u32];
        while let Some(index) = stack.pop() {
            let splat = &tree.splats[index as usize];
            if descends(splat, index, chunk_to_page) {
                stack.extend(splat.child_start..splat.child_start + splat.child_count as u32);
            } else {
                cut.push(index as usize);
            }
        }
        cut
    }

    fn brute_force(tree: &Tree, nodes: &[usize], ray: &Ray, sigma: f32) -> Option<f32> {
        nodes.iter().filter_map(|&i| exact(tree, i, ray, sigma)).min_by(f32::total_cmp)
    }

    #[test]
    fn lod_splat_stays_16_bytes() {
        assert_eq!(std::mem::size_of::<LodSplat>(), 16);
    }

    #[test]
    fn covering_radius_contains_descendants() {
        let mut tree = build_tree(3000, 1, false);
        let pages = identity_pages(&tree);
        let n = tree.splats.len() as u32;
        update_radii(&mut tree.splats, &pages, 0, n);
        for d in 0..tree.splats.len() {
            let max_scale = tree.scale[d].iter().copied().fold(0.0, f32::max);
            let own = LOD_RAYCAST_MAX_SIGMA * max_scale * (tree.opacity[d].max(1.0) * 4.0 - 3.0);
            let mut n = d as u32;
            while n != INVALID {
                let node = &tree.splats[n as usize];
                let reach = Vec3A::from_array(tree.center[d]).distance(node.center()) + own;
                assert!(reach <= node.radius.to_f32(), "node {n} radius {} < {reach} for descendant {d}", node.radius.to_f32());
                n = tree.parent[n as usize];
            }
        }
    }

    #[test]
    fn raycast_matches_brute_force() {
        let mut tree = build_tree(3000, 2, false);
        let pages = identity_pages(&tree);
        let n = tree.splats.len() as u32;
        update_radii(&mut tree.splats, &pages, 0, n);
        let cut = resident_cut(&tree, &pages);
        assert_eq!(cut.len(), tree.num_leaves);
        let mut hits = 0;
        for sigma in [1.0, LOD_RAYCAST_MAX_SIGMA] {
            for ray in rays(&tree, 200, 3) {
                let (t, _) = pick(&tree, &tree.splats, &pages, &ray, sigma, 16);
                assert_eq!(t, brute_force(&tree, &cut, &ray, sigma));
                hits += t.is_some() as usize;
            }
        }
        assert!(hits > 100, "too few hits to be meaningful: {hits}");
    }

    #[test]
    fn far_small_splat_no_false_hit() {
        // 1 mm splat 100 m away, flattened to 5% (still the ellipsoid branch):
        // in f32, b^2 - ac cancels at |o|/s ~ 1e5. A ray passing 2.5 sigma away
        // must miss, one passing 0.5 sigma away must hit at ~100 m.
        let (scale, quat, center) = ([1.0e-3, 1.0e-3, 5.0e-5], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 100.0]);
        for (offset, hit) in [(2.5e-3, false), (0.5e-3, true)] {
            for k in 0..64 {
                let angle = k as f32 * 0.1;
                let origin = [offset * angle.cos(), offset * angle.sin(), 0.0];
                let t = raycast_ellipsoid(origin, [0.0, 0.0, 1.0], 0.9, center, scale, quat, Some(1.0));
                assert_eq!(t.is_some(), hit, "offset {offset} angle {angle}: {t:?}");
                if let Some(t) = t {
                    assert!((t - 100.0).abs() < 1.0e-3);
                }
            }
        }
    }

    #[test]
    fn early_exit_visits_fewer_nodes() {
        // Rays from -z hit the wall first: the walk stops once the hit is proven
        // closest instead of visiting every node behind the wall
        let mut tree = build_tree(30000, 4, true);
        let pages = identity_pages(&tree);
        let n = tree.splats.len() as u32;
        update_radii(&mut tree.splats, &pages, 0, n);
        let mut rng = Rng(5);
        let (mut early, mut full, mut hits) = (0, 0, 0);
        for _ in 0..100 {
            let target = [12.0 + rng.range(-4.0, 4.0), -3.0 + rng.range(-4.0, 4.0), 7.0];
            let origin = [target[0] + rng.range(-3.0, 3.0), target[1] + rng.range(-3.0, 3.0), target[2] - 25.0];
            let ray = Ray { origin, dir: array::from_fn(|d| (target[d] - origin[d]) / 25.0) };
            let (t, visited) = pick(&tree, &tree.splats, &pages, &ray, 1.0, 1);
            hits += t.is_some() as usize;
            early += visited;
            full += raycast_lod(
                &tree.splats, &pages, 0,
                ray.origin.map(|x| x as f64), ray.dir.map(|x| x as f64), 0.0, f64::INFINITY, usize::MAX,
            ).visited;
        }
        assert!(hits > 50 && early * 2 < full, "hits {hits}: early {early} vs full {full}");
    }

    #[test]
    fn residency_stops_at_resident_level() {
        // > 65536 nodes so the deepest leaves live in chunk 1
        let mut tree = build_tree(70000, 6, false);
        let pages = identity_pages(&tree);
        assert_eq!(pages.len(), 2);
        let n = tree.splats.len() as u32;

        // Paged arrival in either chunk order gives the same radii as a full pass
        let mut full = tree.splats.clone();
        update_radii(&mut full, &pages, 0, n);
        for order in [[0u32, 1], [1, 0]] {
            let mut lod_tree = LodTree {
                splats: Rc::new(RefCell::new(tree.splats.clone())),
                chunk_to_page: vec![INVALID; 2],
                ..Default::default()
            };
            for chunk in order {
                lod_tree.chunk_to_page[chunk as usize] = chunk;
                mark_chunks_arrived(&mut lod_tree, chunk << 16, ((chunk + 1) << 16).min(n));
                flush_dirty_radii(&mut lod_tree);
            }
            let radii = |s: &[LodSplat]| s.iter().map(|x| x.radius.to_bits()).collect::<Vec<_>>();
            assert!(radii(&lod_tree.splats.borrow()) == radii(&full), "chunk order {order:?}");
        }

        // Chunk 1 evicted: stale radii stay conservative, nodes of chunk 0 with
        // children in chunk 1 become stop nodes and are tested themselves.
        tree.splats = full;
        let resident = [0, INVALID];
        let cut = resident_cut(&tree, &resident);
        let inner = cut.iter().filter(|&&i| tree.splats[i].child_count > 0).count();
        assert!(inner > 0 && cut.iter().all(|&i| i < 65536));
        for ray in rays(&tree, 64, 7) {
            let (t, _) = pick(&tree, &tree.splats, &resident, &ray, LOD_RAYCAST_MAX_SIGMA, 16);
            assert_eq!(t, brute_force(&tree, &cut, &ray, LOD_RAYCAST_MAX_SIGMA));
        }
    }

    // Every resident node the raycast descends from covers each resident child
    fn assert_covering(splats: &[LodSplat], chunk_to_page: &[u32], n: u32) {
        for index in 0..n {
            let Some(paged) = resident_index(index, chunk_to_page) else { continue };
            let node = &splats[paged as usize];
            if !descends(node, index, chunk_to_page) {
                continue;
            }
            for child in node.child_start..node.child_start + node.child_count as u32 {
                let child = &splats[resident_index(child, chunk_to_page).unwrap() as usize];
                let reach = node.center().distance(child.center()) + child.radius.to_f32();
                assert!(node.radius.to_f32() >= reach, "node {index} radius {} < {reach}", node.radius.to_f32());
            }
        }
    }

    #[test]
    fn lazy_radii_sparse_arrival_with_cycles() {
        // 5 chunks, arriving in sparse order with an eviction and a re-arrival,
        // raycasts (flushes) in between
        let tree = build_tree(200000, 8, false);
        let n = tree.splats.len() as u32;
        let chunks = n.div_ceil(65536);
        assert_eq!(chunks, 5);
        let mut lod_tree = LodTree {
            splats: Rc::new(RefCell::new(tree.splats.clone())),
            chunk_to_page: vec![INVALID; chunks as usize],
            ..Default::default()
        };
        let arrive = |lod_tree: &mut LodTree, chunk: u32| {
            lod_tree.chunk_to_page[chunk as usize] = chunk;
            let range = chunk << 16..((chunk + 1) << 16).min(n);
            for i in range.clone() {
                lod_tree.splats.borrow_mut()[i as usize] = tree.splats[i as usize].clone(); // fresh decode
            }
            mark_chunks_arrived(lod_tree, range.start, range.end);
        };
        for step in [vec![3, 0], vec![4, 1], vec![], vec![2]] {
            for chunk in step {
                arrive(&mut lod_tree, chunk);
            }
            flush_dirty_radii(&mut lod_tree);
            assert!(lod_tree.dirty_chunks.is_empty());
            assert_covering(&lod_tree.splats.borrow(), &lod_tree.chunk_to_page, n);
        }
        lod_tree.chunk_to_page[1] = INVALID;
        arrive(&mut lod_tree, 1);
        flush_dirty_radii(&mut lod_tree);
        let lazy = lod_tree.splats.borrow().clone();
        assert_covering(&lazy, &lod_tree.chunk_to_page, n);

        // Same radii and hits as the eager full pass
        let pages = identity_pages(&tree);
        let mut eager = tree.splats.clone();
        update_radii(&mut eager, &pages, 0, n);
        let radii = |s: &[LodSplat]| s.iter().map(|x| x.radius.to_bits()).collect::<Vec<_>>();
        assert!(radii(&lazy) == radii(&eager));
        let mut hits = 0;
        for ray in rays(&tree, 100, 9) {
            let (t, _) = pick(&tree, &lazy, &pages, &ray, LOD_RAYCAST_MAX_SIGMA, 16);
            assert_eq!(t, pick(&tree, &eager, &pages, &ray, LOD_RAYCAST_MAX_SIGMA, 16).0);
            hits += t.is_some() as usize;
        }
        assert!(hits > 50, "too few hits: {hits}");
    }

    #[test]
    fn many_chunk_arrivals_are_fast() {
        // 1.1M-node tree in 17 chunks arriving last-to-first: arrivals only mark
        // chunks dirty, one flush recomputes each chunk once
        let tree = build_tree(850000, 10, false);
        let n = tree.splats.len() as u32;
        let chunks = n.div_ceil(65536);
        assert!(chunks >= 17, "{chunks}");
        let mut lod_tree = LodTree {
            splats: Rc::new(RefCell::new(tree.splats.clone())),
            chunk_to_page: vec![INVALID; chunks as usize],
            ..Default::default()
        };
        let start = std::time::Instant::now();
        for chunk in (0..chunks).rev() {
            lod_tree.chunk_to_page[chunk as usize] = chunk;
            mark_chunks_arrived(&mut lod_tree, chunk << 16, ((chunk + 1) << 16).min(n));
        }
        let marked = start.elapsed();
        flush_dirty_radii(&mut lod_tree);
        let flushed = start.elapsed();
        assert!(marked.as_secs_f32() < 1.0 && flushed.as_secs_f32() < 2.0, "{marked:?} {flushed:?}");
        assert_covering(&lod_tree.splats.borrow(), &lod_tree.chunk_to_page, n);
    }

    #[test]
    fn flush_all_shared_splats() {
        // Pager-style: two trees sharing one splats Vec, each owning one page.
        // flush_all_dirty_radii() borrows them in turn and gives the full-pass radii.
        let tree = build_tree(70000, 6, false);
        let n = tree.splats.len() as u32;
        let splats = Rc::new(RefCell::new(tree.splats.clone()));
        let mut lod_trees = AHashMap::new();
        for chunk in 0..2u32 {
            let mut chunk_to_page = vec![INVALID; 2];
            chunk_to_page[chunk as usize] = chunk;
            let mut lod_tree = LodTree { splats: splats.clone(), chunk_to_page, ..Default::default() };
            mark_chunks_arrived(&mut lod_tree, chunk << 16, ((chunk + 1) << 16).min(n));
            lod_trees.insert(chunk, lod_tree);
        }
        flush_all_dirty_radii(&mut lod_trees);
        assert!(lod_trees.values().all(|t| t.dirty_chunks.is_empty()));
        // Each tree sees only its own page resident, so compare per page
        for chunk in 0..2u32 {
            let range = (chunk << 16) as usize..((chunk + 1) << 16).min(n) as usize;
            let mut own = tree.splats.clone();
            let mut chunk_to_page = vec![INVALID; 2];
            chunk_to_page[chunk as usize] = chunk;
            update_radii(&mut own, &chunk_to_page, chunk << 16, ((chunk + 1) << 16).min(n));
            let got: Vec<_> = splats.borrow()[range.clone()].iter().map(|x| x.radius.to_bits()).collect();
            let want: Vec<_> = own[range].iter().map(|x| x.radius.to_bits()).collect();
            assert!(got == want, "page {chunk}");
        }
    }

    // Same tree with the sibling groups of every level in random order (still
    // level by level, children contiguous and after their parent), like the
    // spatially batched build-lod layout: the parents of one chunk's nodes are
    // spread over all chunks of the levels above.
    fn shuffled_levels(tree: &Tree, seed: u64) -> Vec<LodSplat> {
        let mut rng = Rng(seed);
        // New order: old index and new parent index of every node
        let mut order: Vec<(usize, u32)> = vec![(0, INVALID)];
        let mut level = 0..1;
        while !level.is_empty() {
            let mut groups: Vec<(u32, usize, usize)> = order[level.clone()].iter().enumerate()
                .filter(|(_, &(old, _))| tree.splats[old].child_count > 0)
                .map(|(i, &(old, _))| {
                    let LodSplat { child_start, child_count, .. } = tree.splats[old];
                    ((level.start + i) as u32, child_start as usize, child_count as usize)
                })
                .collect();
            for i in (1..groups.len()).rev() {
                groups.swap(i, ((rng.next() * (i + 1) as f32) as usize).min(i));
            }
            let next = order.len();
            for (parent, start, count) in groups {
                order.extend((start..start + count).map(|child| (child, parent)));
            }
            level = next..order.len();
        }
        let mut children = vec![(0u32, 0u16); order.len()];
        for (index, &(_, parent)) in order.iter().enumerate() {
            if parent != INVALID {
                let (start, count) = &mut children[parent as usize];
                if *count == 0 {
                    *start = index as u32;
                }
                *count += 1;
            }
        }
        order.iter().zip(children).map(|(&(old, _), (child_start, child_count))| {
            let mut words = [0u32; 4];
            encode_lod_tree(&mut words, &tree.center[old], tree.opacity[old], &tree.scale[old], child_count, child_start);
            LodSplat::from_words(words)
        }).collect()
    }

    #[test]
    fn small_update_cost_independent_of_tree_size() {
        // One chunk re-arriving (fresh decode) into a fully resident tree costs
        // about the same in a 2-chunk and a 17-chunk tree with spread-out
        // parents (the old per-chunk closure recomputed nearly every chunk),
        // and gives the same radii as a full recompute.
        let mut best = Vec::new();
        for leaves in [100_000, 800_000] {
            let tree = build_tree(leaves, 13, false);
            let fresh = shuffled_levels(&tree, 14);
            let n = fresh.len() as u32;
            let chunks = n.div_ceil(65536);
            let pages: Vec<u32> = (0..chunks).collect();
            let mut lod_tree = LodTree {
                splats: Rc::new(RefCell::new(fresh.clone())),
                chunk_to_page: pages.clone(),
                ..Default::default()
            };
            mark_chunks_arrived(&mut lod_tree, 0, n);
            flush_dirty_radii(&mut lod_tree);
            let last = chunks - 1;
            let range = (last << 16) as usize..n as usize;
            let mut fastest = f64::INFINITY;
            for _ in 0..3 {
                lod_tree.splats.borrow_mut()[range.clone()].clone_from_slice(&fresh[range.clone()]);
                let start = std::time::Instant::now();
                mark_chunks_arrived(&mut lod_tree, range.start as u32, n);
                flush_dirty_radii(&mut lod_tree);
                fastest = fastest.min(start.elapsed().as_secs_f64());
            }
            let mut full = fresh.clone();
            update_radii(&mut full, &pages, 0, n);
            let radii = |s: &[LodSplat]| s.iter().map(|x| x.radius.to_bits()).collect::<Vec<_>>();
            assert!(radii(&lod_tree.splats.borrow()) == radii(&full), "{chunks} chunks");
            best.push((chunks, fastest));
        }
        let ((small_chunks, small), (big_chunks, big)) = (best[0], best[1]);
        assert!(small_chunks <= 3 && big_chunks >= 17, "{best:?}");
        assert!(big < small * 3.0 + 0.002, "{best:?}");
    }

    // Per-arrival radii cost on a realistic paged tree (~3.7M nodes): chunks
    // arrive coarse to fine (pager order), each followed by a flush like
    // update_lod_trees(), then single-chunk re-arrivals with everything resident.
    // cargo test --release radii_update_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn radii_update_bench() {
        let tree = build_tree(2_600_000, 11, false);
        let splats = shuffled_levels(&tree, 12);
        let n = splats.len() as u32;
        let chunks = n.div_ceil(65536);
        let mut lod_tree = LodTree {
            splats: Rc::new(RefCell::new(splats)),
            chunk_to_page: vec![INVALID; chunks as usize],
            ..Default::default()
        };
        let (mut total, mut worst) = (0.0f64, 0.0f64);
        for chunk in 0..chunks {
            let start = std::time::Instant::now();
            lod_tree.chunk_to_page[chunk as usize] = chunk;
            mark_chunks_arrived(&mut lod_tree, chunk << 16, ((chunk + 1) << 16).min(n));
            flush_dirty_radii(&mut lod_tree);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            total += ms;
            worst = worst.max(ms);
        }
        println!("nodes {n} chunks {chunks}: mean {:.2} ms worst {:.2} ms per arrival", total / chunks as f64, worst);
        for chunk in [chunks - 1, chunks / 2] {
            let start = std::time::Instant::now();
            mark_chunks_arrived(&mut lod_tree, chunk << 16, ((chunk + 1) << 16).min(n));
            flush_dirty_radii(&mut lod_tree);
            println!("re-arrival chunk {chunk}: {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);
        }
        assert_covering(&lod_tree.splats.borrow(), &lod_tree.chunk_to_page, n);
    }
}

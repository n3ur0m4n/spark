use super::*;
use std::time::Instant;

fn node(size: f32, first: u32, count: u16) -> LodSplat {
    LodSplat::new(Vec3::new(0.0, 0.0, -10.0), size, first, count)
}

fn tree(nodes: Vec<LodSplat>) -> LodTree {
    let pages = nodes.len().div_ceil(MAX_SPLAT_CHUNK) as u32;
    LodTree {
        splats: Rc::new(RefCell::new(nodes)),
        page_to_chunk: (0..pages).collect(),
        chunk_to_page: (0..pages).collect(),
        parents: Vec::new(),
        dirty_chunks: Vec::new(),
        dirty_nodes: BinaryHeap::new(),
    }
}

fn instances<'a>(trees: &'a [LodTree], seed: u32) -> Vec<Instance<'a>> {
    trees
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let angle = (seed % 11) as f32 * 0.17;
            (
                i as u32,
                t.splats.borrow(),
                &t.page_to_chunk,
                &t.chunk_to_page,
                Vec3A::new((seed % 7) as f32 - 3.0, i as f32, 0.0),
                Vec3A::new(angle.sin(), 0.0, -angle.cos()),
                0.5 + i as f32,
                if seed % 2 == 0 { 1.0 } else { 0.1 },
                if seed % 3 == 0 { 1.0 } else { 0.5 },
                0.8,
                0.3,
            )
        })
        .collect()
}

fn check(input: &[Instance<'_>], roots: &[u32], budget: usize, limit: f32, buffers: &mut Buffers) {
    let (expected, chunks, reference_stats) = reference::select_reference(
        budget,
        limit,
        roots,
        input,
        &mut reference::ReferenceBuffers::default(),
    );
    let stats = select(budget, limit, roots, input, buffers);
    assert_eq!(
        buffers.touched, chunks,
        "paging request order; budget={budget}, limit={limit}"
    );
    let mut total = 0;
    for (i, expected) in expected.iter().enumerate() {
        let actual = &buffers.instance_outputs[i];
        assert_eq!(
            actual, expected,
            "selected IDs for instance {i}; budget={budget}, limit={limit}"
        );
        assert_eq!(
            actual.iter().collect::<AHashSet<_>>().len(),
            actual.len(),
            "duplicate IDs"
        );
        total += actual.len();
    }
    assert_eq!(
        stats.pixel_limit.to_bits(),
        reference_stats.pixel_limit.to_bits()
    );
    assert_eq!(stats.emitted, reference_stats.emitted);
    assert_eq!(stats.drained, reference_stats.drained);
    assert_eq!(stats.leaves, reference_stats.leaves);
    assert_eq!(stats.emitted + stats.drained, total);
    // Upstream retains one root per instance even when the requested budget is
    // smaller. Do not silently turn this optimization into a budget policy fix.
    assert!(total <= budget.max(input.len()));
}

#[test]
fn empty_scene_and_root_budget_contract() {
    let mut scratch = Buffers::default();
    check(&[], &[], 0, 0.0, &mut scratch);
    let trees = [tree(vec![node(1.0, 0, 0)]), tree(vec![node(2.0, 0, 0)])];
    let input = instances(&trees, 0);
    for budget in [0, 1, 2, 10000] {
        for threshold in [0.0, 0.1, f32::INFINITY] {
            check(&input, &[0, u32::MAX], budget, threshold, &mut scratch);
        }
    }
}

#[test]
fn leaves_thresholds_ties_and_fanout_budget_stop() {
    let trees = [tree(vec![
        node(8.0, 1, 3),
        node(2.0, 4, 2),
        node(2.0, 6, 1),
        node(2.0, 0, 0),
        node(0.1, 0, 0),
        node(0.1, 0, 0),
        node(0.1, 0, 0),
    ])];
    let input = instances(&trees, 0);
    let mut scratch = Buffers::default();
    let root_scale = compute_pixel_scale(&input[0].1[0], &input[0]);
    for limit in [0.0, 0.01, 0.2, root_scale, f32::INFINITY] {
        for budget in 0..12 {
            check(&input, &[0], budget, limit, &mut scratch);
        }
    }
}

#[test]
fn shared_tree_instances_have_independent_outputs() {
    let shared = tree(vec![node(4.0, 1, 2), node(0.1, 0, 0), node(0.1, 0, 0)]);
    let trees = [shared.clone(), shared.clone(), shared];
    let mut input = instances(&trees, 0);
    for item in &mut input {
        item.0 = 7;
    }
    let mut scratch = Buffers::default();
    for budget in 0..10 {
        check(&input, &[0, 0, 0], budget, 0.0, &mut scratch);
    }
}

#[test]
fn paged_children_missing_resident_crossing_and_remapped() {
    let mut nodes = vec![node(0.1, 0, 0); MAX_SPLAT_CHUNK * 2];
    // Root is physically in page 1; children straddle logical chunks 0 and 1.
    nodes[MAX_SPLAT_CHUNK] = node(10.0, 65535, 2);
    let mut t = tree(nodes);
    t.page_to_chunk = vec![1, 0];
    let mut scratch = Buffers::default();
    for map in [
        vec![1, 0],
        vec![1, u32::MAX],
        vec![u32::MAX, 0],
        vec![1],
        vec![],
    ] {
        t.chunk_to_page = map;
        let trees = [t.clone()];
        let input = instances(&trees, 1);
        for budget in [0, 1, 2, 100] {
            check(&input, &[1], budget, 0.0, &mut scratch);
        }
    }
}

fn next(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    *state
}

fn generated_tree(seed: u32, count: usize) -> LodTree {
    let mut rng = seed;
    let mut nodes = Vec::with_capacity(count);
    for _ in 0..count {
        let xyz = std::array::from_fn(|_| (next(&mut rng) % 2001) as f32 * 0.02 - 20.0);
        nodes.push(LodSplat::new(
            Vec3::from_array(xyz),
            (next(&mut rng) % 65) as f32 * 0.125,
            0,
            0,
        ));
    }
    let mut first = 1;
    for i in 0..count {
        if first == count {
            break;
        }
        let children = (1 + next(&mut rng) as usize % 6).min(count - first);
        nodes[i].child_start = first as u32;
        nodes[i].child_count = children as u16;
        first += children;
    }
    tree(nodes)
}

#[test]
fn seeded_differential_and_reused_buffers_grow_shrink() {
    let mut scratch = Buffers::default();
    for seed in 0..64 {
        let shared = generated_tree(seed, if seed % 3 == 0 { 19 } else { 2049 });
        let trees = vec![shared; 1 + seed as usize % 3];
        let input = instances(&trees, seed);
        let roots = vec![0; trees.len()];
        for budget in [0, 1, 2, 3, 16, 63, 256, 2048, 10000] {
            for limit in [0.0, 0.03, 0.5, f32::INFINITY] {
                check(&input, &roots, budget, limit, &mut scratch);
            }
        }
    }
    check(&[], &[], 0, 0.0, &mut scratch);
}

#[test]
#[ignore = "reproducible CPU traversal benchmark; run with --release --ignored --nocapture"]
fn benchmark_synthetic_traversal() {
    let trees = [generated_tree(20260906, 4_000_001)];
    let input = instances(&trees, 0);
    let mut scratch = Buffers::default();
    let mut reference_scratch = reference::ReferenceBuffers::default();
    for budget in [1_500_000, 2_500_000] {
        check(&input, &[0], budget, 0.0, &mut scratch);
        for repetition in 0..6 {
            let mut times = [0.0; 2];
            for method in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                if method == 0 {
                    std::hint::black_box(reference::select_reference(
                        budget,
                        0.0,
                        &[0],
                        &input,
                        &mut reference_scratch,
                    ));
                } else {
                    std::hint::black_box(select(budget, 0.0, &[0], &input, &mut scratch));
                }
                times[method] = start.elapsed().as_secs_f64() * 1000.0;
            }
            if repetition > 0 {
                println!("{{\"fixture\":\"synthetic-lcg-20260906-v1\",\"nodes\":4000001,\"budget\":{budget},\"repetition\":{repetition},\"upstreamMs\":{},\"optimizedMs\":{},\"selected\":{}}}",
                    times[0], times[1], scratch.instance_outputs[0].len());
            }
        }
    }
}

#[test]
fn preserve_order_for_equal_depth_alpha_blending() {
    // Every selected center is at the same depth. Spark's stable depth sort
    // preserves traversal output order, so different order changes alpha color.
    let trees = [tree(vec![
        node(8.0, 1, 3),
        node(2.0, 4, 2),
        node(2.0, 6, 1),
        node(2.0, 0, 0),
        node(0.1, 0, 0),
        node(0.1, 0, 0),
        node(0.1, 0, 0),
    ])];
    let input = instances(&trees, 0);
    let (expected, _, _) = reference::select_reference(
        10,
        0.0,
        &[0],
        &input,
        &mut reference::ReferenceBuffers::default(),
    );
    let mut scratch = Buffers::default();
    select(10, 0.0, &[0], &input, &mut scratch);
    let composite = |ids: &[u32]| {
        ids.iter().fold(0.0, |background, id| {
            let color = if *id == 4 { 1.0 } else { 0.0 };
            color * 0.5 + background * 0.5
        })
    };
    assert_eq!(
        composite(&scratch.instance_outputs[0]),
        composite(&expected[0]),
        "equal-depth premultiplied alpha output changes"
    );
    assert_eq!(scratch.instance_outputs[0], expected[0]);
}

#[test]
fn output_allocations_are_reused_after_warmup() {
    let trees = [generated_tree(91, 2049), generated_tree(92, 2049)];
    let input = instances(&trees, 0);
    let mut scratch = Buffers::default();
    select(10000, 0.0, &[0, 0], &input, &mut scratch);
    let allocations: Vec<_> = scratch
        .instance_outputs
        .iter()
        .map(|v| (v.as_ptr(), v.capacity()))
        .collect();
    for budget in [2, 20, 200, 10000, 2, 10000] {
        check(&input, &[0, 0], budget, 0.0, &mut scratch);
        for (i, output) in scratch.instance_outputs.iter().enumerate() {
            assert_eq!((output.as_ptr(), output.capacity()), allocations[i]);
        }
    }
}

#[test]
fn disposing_last_tree_releases_pooled_selection_storage() {
    STATE.with_borrow_mut(|state| {
        *state = LodState::new();
        let tree = generated_tree(123, 2049);
        state.lod_trees.insert(42, tree.clone());
        let trees = [tree];
        let input = instances(&trees, 0);
        select(10000, 0.0, &[0], &input, &mut state.traversal);
        assert!(state.traversal.instance_outputs[0].capacity() > 0);
    });
    dispose_lod_tree(42);
    STATE.with_borrow(|state| {
        assert!(state.lod_trees.is_empty());
        assert_eq!(state.traversal.instance_outputs.capacity(), 0);
    });
}

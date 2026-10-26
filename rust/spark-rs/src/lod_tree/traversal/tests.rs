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

#[test]
fn membership_epochs_overflow_and_tree_identity_reuse() {
    let mut membership = Membership {
        marks: vec![0; 2],
        ..Default::default()
    };
    membership.begin();
    assert!(membership.first_touch(0));
    assert!(!membership.first_touch(0));
    assert!(membership.first_touch(u32::MAX));
    assert!(!membership.first_touch(u32::MAX));
    assert_eq!(membership.marks.len(), 2);
    membership.epoch = u32::MAX;
    membership.begin();
    assert!(membership.first_touch(0));
    assert!(membership.first_touch(u32::MAX));
    let trees = [tree(vec![node(1.0, 0, 0)]), tree(vec![node(1.0, 0, 0)])];
    let mut input = instances(&trees, 0);
    let mut scratch = Buffers::default();
    check(&input, &[0, 0], 2, 0.0, &mut scratch);
    assert_eq!(scratch.touched, vec![(0, 0), (1, 0)]);
    input[1].0 = 0;
    check(&input, &[0, 0], 2, 0.0, &mut scratch);
    assert_eq!(scratch.touched, vec![(0, 0)]);
    scratch.forget_tree(0);
    check(&input, &[0, 0], 2, 0.0, &mut scratch);
    assert_eq!(scratch.touched, vec![(0, 0)]);
    assert_eq!(scratch.membership_ids.len(), scratch.memberships.len());
}

fn posed_instances<'a>(trees: &'a [LodTree], rng: &mut u32) -> Vec<Instance<'a>> {
    let mut unit = || (next(rng) % 10001) as f32 / 10000.0;
    trees
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let origin = Vec3A::new(unit() * 50.0 - 25.0, unit() * 50.0 - 25.0, unit() * 50.0 - 25.0);
            let forward = Vec3A::new(unit() - 0.5, unit() - 0.5, unit() - 0.5).normalize_or(Vec3A::Z);
            let cone_dot0 = 0.5 + unit() * 0.5;
            (
                i as u32,
                t.splats.borrow(),
                &t.page_to_chunk,
                &t.chunk_to_page,
                origin,
                forward,
                0.25 + unit() * 4.0,
                0.05 + unit() * 0.95,
                0.2 + unit() * 0.8,
                cone_dot0,
                cone_dot0 * unit(),
            )
        })
        .collect()
}

// The raycast set reuses the groups of the preceding select().
fn check_raycast(input: &[Instance<'_>], roots: &[u32], budget: usize, limit: f32, buffers: &mut Buffers) {
    select_raycast(budget, limit, roots, input, buffers);
    let (expected, _, _) = reference::select_reference(
        budget,
        limit,
        roots,
        input,
        &mut reference::ReferenceBuffers::default(),
    );
    for (i, expected) in expected.iter().enumerate() {
        let mut actual = buffers.raycast_outputs[i].clone();
        let mut expected = expected.clone();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected, "raycast set instance {i}; budget={budget}");
    }
}

#[test]
fn non_monotone_children_closer_than_root_with_foveation() {
    // Camera at the origin looking down -Z. The root is far and big; children
    // are near the camera so their key exceeds the root's (e(n) != p(n)),
    // grandchildren sit behind the camera (foveated down).
    let at = |x: f32, z: f32, size: f32, first: u32, count: u16| {
        LodSplat::new(Vec3::new(x, 0.0, z), size, first, count)
    };
    let nodes = vec![
        at(0.0, -100.0, 10.0, 1, 4),
        at(0.5, -2.0, 1.0, 5, 3),
        at(-0.5, -2.0, 1.0, 8, 2),
        at(5.0, -1.0, 1.0, 10, 2),
        at(0.0, -80.0, 4.0, 0, 0),
        at(0.0, 3.0, 0.5, 12, 2),
        at(0.1, -1.0, 0.5, 0, 0),
        at(0.2, -1.0, 0.5, 0, 0),
        at(0.0, 4.0, 0.5, 0, 0),
        at(0.0, -0.5, 0.4, 14, 3),
        at(9.0, -1.0, 0.3, 0, 0),
        at(9.0, 1.0, 0.3, 0, 0),
        at(0.0, 2.0, 0.2, 0, 0),
        at(0.0, -2.0, 0.2, 0, 0),
        at(0.0, -0.3, 0.1, 0, 0),
        at(0.0, -0.3, 0.1, 0, 0),
        at(0.1, -0.3, 0.1, 0, 0),
    ];
    let trees = [tree(nodes.clone()), tree(nodes)];
    let mut input: Vec<Instance<'_>> = trees
        .iter()
        .enumerate()
        .map(|(i, t)| {
            (
                i as u32,
                t.splats.borrow(),
                &t.page_to_chunk,
                &t.chunk_to_page,
                Vec3A::new(i as f32 * 0.25, 0.0, 0.0),
                Vec3A::new(0.0, 0.0, -1.0),
                1.0,
                0.1,
                0.4,
                0.95,
                0.6,
            )
        })
        .collect();
    let root = compute_pixel_scale(&input[0].1[0], &input[0]);
    let child = compute_pixel_scale(&input[0].1[1], &input[0]);
    assert!(child > root, "fixture must be non-monotone");
    let mut scratch = Buffers::default();
    for instances in [1, 2] {
        for limit in [0.0, 0.05, root * 0.5, root] {
            for budget in 0..40 {
                check(&input[..instances], &[0, 0][..instances], budget, limit, &mut scratch);
                check_raycast(&input[..instances], &[0, 0][..instances], budget / 2, limit, &mut scratch);
            }
        }
    }
    input[1].0 = 0; // shared tree identity across instances
    for budget in 0..40 {
        check(&input, &[0, 0], budget, 0.0, &mut scratch);
    }
}

#[test]
fn budget_cuts_inside_a_group_multi_instance() {
    // Deep chains where every child grows (one group per instance), so most
    // budgets stop in the middle of a group.
    // Layout: 0 = root; level i: 2i+1 = chain node (closer, bigger), 2i+2 = far sibling.
    let levels = 29u32;
    let mut nodes = vec![LodSplat::new(Vec3::new(0.0, 0.0, -50.0), 1.0, 1, 2)];
    for i in 0..levels {
        let (first, count) = if i + 1 < levels { (2 * i + 3, 2) } else { (0, 0) };
        let z = -40.0 + i as f32;
        nodes.push(LodSplat::new(Vec3::new(0.0, 0.0, z), 1.0 + i as f32 * 0.1, first, count));
        nodes.push(LodSplat::new(Vec3::new(3.0, 0.0, -60.0), 0.05, 0, 0));
    }
    let trees = [tree(nodes.clone()), tree(nodes.clone()), tree(nodes)];
    let input = instances(&trees, 2);
    let roots = [0, 0, 0];
    let mut scratch = Buffers::default();
    for budget in 0..100 {
        check(&input, &roots, budget, 0.0, &mut scratch);
        check_raycast(&input, &roots, budget / 3, 0.0, &mut scratch);
    }
}

#[test]
fn randomized_poses_trees_and_paging_vs_reference() {
    let mut rng = 0x5eed_u32;
    let mut scratch = Buffers::default();
    for round in 0..120u32 {
        let count = [7, 40, 300, 2049, 70_000][round as usize % 5];
        let mut trees: Vec<LodTree> = (0..1 + next(&mut rng) % 3)
            .map(|k| generated_tree(next(&mut rng) ^ k, count))
            .collect();
        for t in trees.iter_mut() {
            // Random missing pages on multi-chunk trees (root chunk resident).
            if t.chunk_to_page.len() > 1 && next(&mut rng) % 2 == 0 {
                let missing = 1 + next(&mut rng) as usize % (t.chunk_to_page.len() - 1);
                t.chunk_to_page[missing] = u32::MAX;
            }
        }
        let input = posed_instances(&trees, &mut rng);
        let roots = vec![0; trees.len()];
        // Same buffers across rounds: warm tau from unrelated poses/budgets.
        for _ in 0..4 {
            let budget = [0, 1, 5, 64, 500, 3000, 20_000, 100_000][next(&mut rng) as usize % 8];
            let limit = [0.0, 0.01, 0.2, 1.0][next(&mut rng) as usize % 4];
            check(&input, &roots, budget, limit, &mut scratch);
            check_raycast(&input, &roots, (budget as f64 * 0.1).round() as usize, limit, &mut scratch);
            check_raycast(&input, &roots, budget * 2 + 7, limit, &mut scratch);
        }
    }
}

#[test]
#[ignore = "real .rad benchmark: SPARK_RAD=<file.rad> cargo test --release benchmark_real_rad -- --ignored --nocapture"]
fn benchmark_real_rad() {
    use spark_lib::decoder::{ChunkReceiver, SplatInit, SplatProps, SplatReceiver};
    #[derive(Default)]
    struct Rec {
        center: Vec<f32>,
        opacity: Vec<f32>,
        scale: Vec<f32>,
        child_count: Vec<u16>,
        child_start: Vec<usize>,
    }
    fn put<T: Copy + Default>(dst: &mut Vec<T>, base: usize, src: &[T]) {
        if dst.len() < base + src.len() {
            dst.resize(base + src.len(), T::default());
        }
        dst[base..base + src.len()].copy_from_slice(src);
    }
    impl SplatReceiver for Rec {
        fn init_splats(&mut self, init: &SplatInit) -> anyhow::Result<()> {
            self.child_count.reserve(init.num_splats);
            Ok(())
        }
        fn set_batch(&mut self, base: usize, count: usize, b: &SplatProps) {
            self.set_center(base, count, b.center);
            self.set_opacity(base, count, b.opacity);
            self.set_scale(base, count, b.scale);
            self.set_child_count(base, count, b.child_count);
            self.set_child_start(base, count, b.child_start);
        }
        fn set_center(&mut self, base: usize, _: usize, v: &[f32]) { put(&mut self.center, base * 3, v) }
        fn set_opacity(&mut self, base: usize, _: usize, v: &[f32]) { put(&mut self.opacity, base, v) }
        fn set_rgb(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_rgba(&mut self, base: usize, count: usize, v: &[f32]) {
            let alpha: Vec<f32> = v.chunks(4).map(|c| c[3]).collect();
            self.set_opacity(base, count, &alpha);
        }
        fn set_scale(&mut self, base: usize, _: usize, v: &[f32]) { put(&mut self.scale, base * 3, v) }
        fn set_quat(&mut self, _: usize, _: usize, _: &[f32]) {}
        fn set_child_count(&mut self, base: usize, _: usize, v: &[u16]) { put(&mut self.child_count, base, v) }
        fn set_child_start(&mut self, base: usize, _: usize, v: &[usize]) { put(&mut self.child_start, base, v) }
    }

    let Ok(path) = std::env::var("SPARK_RAD") else { return };
    let start = Instant::now();
    let bytes = std::fs::read(&path).unwrap();
    let mut decoder = spark_lib::rad::RadDecoder::new(Rec::default());
    for chunk in bytes.chunks(1 << 20) {
        decoder.push(chunk).unwrap();
    }
    decoder.finish().unwrap();
    drop(bytes);
    let rec = decoder.into_splats();
    let count = rec.child_count.len();
    let mut nodes = Vec::with_capacity(count);
    let mut words = [0u32; 4];
    let (mut lo, mut hi) = (Vec3A::splat(f32::MAX), Vec3A::splat(f32::MIN));
    for i in 0..count {
        let c = &rec.center[i * 3..i * 3 + 3];
        spark_lib::splat_encode::encode_lod_tree(
            &mut words, c, rec.opacity[i], &rec.scale[i * 3..i * 3 + 3],
            rec.child_count[i], rec.child_start[i] as u32,
        );
        nodes.push(LodSplat::from_words(words));
        if rec.child_count[i] == 0 {
            lo = lo.min(Vec3A::from_slice(c));
            hi = hi.max(Vec3A::from_slice(c));
        }
    }
    drop(rec);
    println!("loaded {count} nodes in {:.1}s, leaf bbox {lo:?}..{hi:?}", start.elapsed().as_secs_f64());
    let trees = [tree(nodes)];
    let mid = (lo + hi) * 0.5;
    let ext = hi - lo;
    let poses = [
        (mid, Vec3A::new(0.0, 0.0, -1.0)),
        (mid, Vec3A::new(1.0, 0.0, 0.0)),
        (lo + ext * Vec3A::new(0.2, 0.5, 0.2), Vec3A::new(1.0, 0.0, 1.0).normalize()),
        (lo + ext * Vec3A::new(0.8, 0.3, 0.6), Vec3A::new(-1.0, -0.2, 0.3).normalize()),
    ];
    let limit = 0.001;
    let cone = |deg: f32| (0.5 * deg).to_radians().cos();
    for budget in [1_000_000usize, 2_000_000, 3_375_000] {
        for (p, &(origin, forward)) in poses.iter().enumerate() {
            let make = |o: Vec3A| -> Vec<Instance<'_>> {
                vec![(0, trees[0].splats.borrow(), &trees[0].page_to_chunk, &trees[0].chunk_to_page,
                    o, forward, 1.0, 0.2, 0.4, cone(90.0), cone(120.0).min(cone(90.0)))]
            };
            let input = make(origin);
            let t = Instant::now();
            let (expected, chunks, rstats) = reference::select_reference(
                budget, limit, &[0], &input, &mut reference::ReferenceBuffers::default());
            let old_ms = t.elapsed().as_secs_f64() * 1e3;
            let mut buffers = Buffers::default();
            let t = Instant::now();
            let stats = select(budget, limit, &[0], &input, &mut buffers);
            let cold_ms = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(buffers.instance_outputs[0], expected[0], "pose {p} budget {budget}");
            assert_eq!(buffers.touched, chunks);
            assert_eq!(stats.pixel_limit.to_bits(), rstats.pixel_limit.to_bits());
            assert_eq!((stats.emitted, stats.drained, stats.leaves), (rstats.emitted, rstats.drained, rstats.leaves));
            // Warm: camera moved a little since the previous frame.
            let moved = make(origin + forward * (ext.length() * 0.002));
            let t = Instant::now();
            select(budget, limit, &[0], &moved, &mut buffers);
            let warm_ms = t.elapsed().as_secs_f64() * 1e3;
            let ray = 75_000usize.min((buffers.instance_outputs[0].len() as f64 * 0.1).round() as usize);
            let t = Instant::now();
            select_raycast(ray, limit, &[0], &moved, &mut buffers);
            let ray_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let (ray_expected, _, _) = reference::select_reference(
                ray, limit, &[0], &moved, &mut reference::ReferenceBuffers::default());
            let ray_old_ms = t.elapsed().as_secs_f64() * 1e3;
            let mut a = buffers.raycast_outputs[0].clone();
            let mut b = ray_expected[0].clone();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "raycast set pose {p} budget {budget}");
            println!("budget={budget} pose={p} selected={} drained={} old={old_ms:.0}ms new_cold={cold_ms:.0}ms new_warm={warm_ms:.0}ms raycast_old={ray_old_ms:.1}ms raycast_new={ray_ms:.1}ms identical=yes",
                expected[0].len(), rstats.drained);
        }
    }
}

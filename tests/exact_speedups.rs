//! Precomputed and vectorized queries against the original algorithms.
use altd_sim::{bsp::{Node, RayTree}, godot_math::F2, vec2::V2, world::{PathHit, Track, World, PATH_WINDOW}};
use serde_json::Value;

const SCENES: [&str; 5] = ["rally_a01_trained_scene", "rally_a05_trained_scene", "rally_a06_trained_scene", "rally_a07_contact_scene", "rally_b06_scene"];

fn world(name: &str) -> World {
    let path = format!("{}/scenes_exact/{name}.json", env!("CARGO_MANIFEST_DIR"));
    World::from_scene(&serde_json::from_str::<Value>(&std::fs::read_to_string(path).unwrap()).unwrap())
}

struct Random(u64);
impl Random {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// TrackManager's projection with a linear scan of the wrapped window.
fn reference_closest_path(track: &Track, point: V2, previous: Option<f64>) -> PathHit {
    let count = track.path.len() - 1;
    let length = track.path_length() as f32;
    let posmod = |x: f32| { let r = x % length; if r < 0.0 { r + length } else { r } };
    let index = |offset: f32| track.path_offsets.binary_search_by(|v| v.partial_cmp(&(offset as f64)).unwrap()).unwrap_or_else(|i| i.saturating_sub(1)) as isize;
    let (mut from, mut to) = (0isize, count as isize);
    if let Some(previous) = previous.filter(|_| length > 2.0 * PATH_WINDOW as f32) {
        let previous = previous as f32;
        let center = index(posmod(previous));
        from = index(posmod(previous - PATH_WINDOW as f32));
        to = index(posmod(previous + PATH_WINDOW as f32)) + 1;
        if from > center { from -= count as isize; }
        if to < center { to += count as isize; }
    }
    let point = F2::from(point);
    let mut distance = f32::MAX;
    let mut best = PathHit { offset: 0.0, near: V2::ZERO, tangent: V2::ZERO };
    for i in from..to {
        let i = i.rem_euclid(count as isize) as usize;
        let start = F2::from(track.path[i]);
        let delta = F2::from(track.path[i + 1]) - start;
        let len2 = delta.dot(delta);
        let fraction = if len2 > 0.0 { ((point - start).dot(delta) / len2).clamp(0.0, 1.0) } else { 0.0 };
        let near = start + delta * fraction;
        let diff = point - near;
        if diff.dot(diff) < distance {
            distance = diff.dot(diff);
            let begin = track.path_offsets[i] as f32;
            let end = track.path_offsets[i + 1] as f32;
            best = PathHit { offset: if length > 0.0 { posmod(begin + (end - begin) * fraction) as f64 } else { 0.0 }, near: near.into(), tangent: delta.normalized().into() };
        }
    }
    best
}

fn bits(hit: PathHit) -> [u64; 5] {
    [hit.offset.to_bits(), hit.near.x.to_bits(), hit.near.y.to_bits(), hit.tangent.x.to_bits(), hit.tangent.y.to_bits()]
}

#[test]
fn closest_path_matches_linear_projection() {
    let mut random = Random(0x9E3779B97F4A7C15);
    for name in SCENES {
        let world = world(name);
        let track = &world.track;
        let length = track.path_length();
        for n in 0..60_000 {
            // Near the path (the scoring case), on baked points, or anywhere.
            let base = track.path[(random.next() * track.path.len() as f64) as usize];
            let point = match n % 4 {
                0 => base,
                3 => V2::new(random.next() * 12000.0 - 2000.0, random.next() * 12000.0 - 2000.0),
                _ => V2::new(base.x + (random.next() - 0.5) * 600.0, base.y + (random.next() - 0.5) * 600.0),
            };
            let exact = reference_closest_path(track, point, None);
            let previous = match n % 5 {
                0 => None,
                1 => Some(random.next() * length),
                2 => Some([0.0, length, length - 1e-3, 449.99, length - 450.0][n / 5 % 5]),
                _ => Some(exact.offset + (random.next() - 0.5) * 1200.0),
            };
            assert_eq!(bits(track.closest_path(point, previous)), bits(reference_closest_path(track, point, previous)),
                "{name}: point {point:?} previous {previous:?}");
        }
    }
}

fn f32v(x: f64, y: f64) -> V2 {
    V2::new(x as f32 as f64, y as f32 as f64)
}

/// Rays across the track, from path points and trace poses, through wall
/// endpoints, along walls, parallel to walls at tiny offsets, and of zero length.
fn rays(path: &[V2], walls: &[altd_sim::world::Wall], poses: &[V2], random: &mut Random) -> Vec<(V2, V2)> {
    let (lo, hi) = walls.iter().fold(((f64::MAX, f64::MAX), (f64::MIN, f64::MIN)), |(l, h), w| {
        ((l.0.min(w.start.x), l.1.min(w.start.y)), (h.0.max(w.start.x), h.1.max(w.start.y)))
    });
    let polar = |p: V2, random: &mut Random, length: f64| {
        let angle = random.next() * std::f64::consts::TAU;
        f32v(p.x + angle.cos() * length, p.y + angle.sin() * length)
    };
    let length = |random: &mut Random| [50.0, 300.0, 700.0, 1500.0, 20000.0][(random.next() * 5.0) as usize];
    let mut out = Vec::new();
    for _ in 0..100_000 {
        let start = f32v(lo.0 - 300.0 + random.next() * (hi.0 - lo.0 + 600.0), lo.1 - 300.0 + random.next() * (hi.1 - lo.1 + 600.0));
        let l = length(random);
        out.push((start, polar(start, random, l)));
    }
    for &p in path.iter().step_by(3).chain(poses) {
        for _ in 0..13 {
            let l = length(random);
            out.push((p, polar(p, random, l)));
        }
    }
    for wall in walls {
        let (s, e) = (wall.start, wall.end);
        let mid = f32v((s.x + e.x) * 0.5, (s.y + e.y) * 0.5);
        let back = f32v(2.0 * s.x - e.x, 2.0 * s.y - e.y);
        let ahead = f32v(2.0 * e.x - s.x, 2.0 * e.y - s.y);
        out.extend([(back, e), (s, e), (e, s), (mid, mid), (back, ahead), (ahead, back), (e, ahead), (ahead, e)]);
        // Parallel and nearly parallel rays beside the wall.
        let (dx, dy) = (e.x - s.x, e.y - s.y);
        let n = (dx * dx + dy * dy).sqrt();
        for offset in [-2.0, -1.0, -0.5, -0.05, -1e-2, -1e-3, -1e-4, -1e-5, 0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.5, 1.0, 2.0] {
            let (ox, oy) = (-dy / n * offset, dx / n * offset);
            let tilt = (random.next() - 0.5) * 1e-3;
            out.push((f32v(back.x + ox, back.y + oy), f32v(ahead.x + ox + tilt * dy, ahead.y + oy - tilt * dx)));
            out.push((f32v(mid.x + ox, mid.y + oy), f32v(ahead.x + ox, ahead.y + oy)));
        }
        for _ in 0..6 {
            let l = 1.0 + random.next() * 400.0;
            out.push((s, polar(s, random, l)));
            out.push((mid, polar(mid, random, l)));
            let origin = polar(e, random, l);
            out.push((origin, e));
        }
    }
    out
}

/// `cast` against the original tree traversal on `rays`.
fn compare_ray_tree(name: &str, tree: &Node, path: &[V2], poses: &[V2], random: &mut Random, mut cast: impl FnMut(V2, V2) -> Option<V2>) -> (usize, usize) {
    let mut walls = Vec::new();
    tree.collect(&mut walls);
    let bits = |h: Option<V2>| h.map(|p| (p.x.to_bits(), p.y.to_bits()));
    let (mut checked, mut hits) = (0, 0);
    for (start, end) in rays(path, &walls, poses, random) {
        let want = tree.raycast(start, end);
        assert_eq!(bits(cast(start, end)), bits(want), "{name}: ray {start:?} -> {end:?}");
        checked += 1;
        hits += want.is_some() as usize;
    }
    (checked, hits)
}

#[test]
fn ray_tree_matches_bsp_traversal() {
    let mut random = Random(0xD1B54A32D192ED03);
    let trace: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/../docs/traces/rally_b06_reference.json", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
    let poses: Vec<V2> = trace["frames"].as_array().unwrap().iter().step_by(4).map(|f| altd_sim::world::vector(&f["position"])).collect();
    let (mut checked, mut hits) = (0, 0);
    let template = serde_json::from_str::<Value>(&std::fs::read_to_string(format!("{}/scenes_exact/rally_b06_scene.json", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
    let mut scenes: Vec<(String, Value)> = SCENES.iter().map(|name| {
        let path = format!("{}/scenes_exact/{name}.json", env!("CARGO_MANIFEST_DIR"));
        (name.to_string(), serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap())
    }).collect();
    let generated: Value = serde_json::from_str(include_str!("fixtures/windows_tracks.json")).unwrap();
    for case in generated.as_array().unwrap().iter().filter(|c| !c["result"].is_null()) {
        let track: altd_sim::random_track::GeneratedTrack = serde_json::from_value(case["result"].clone()).unwrap();
        scenes.push((format!("generated {}", case["i"]), track.to_scene(&template)));
    }
    for (name, scene) in &scenes {
        let world = World::from_scene(scene);
        let track = &world.track;
        let poses: &[V2] = if name == "rally_b06_scene" { &poses } else { &[] };
        let mut stamps = altd_sim::world::RayStamps::default();
        let (c, h) = compare_ray_tree(name, track.bsp.as_ref().unwrap(), &track.path, poses, &mut random, |a, b| track.raycast(a, b, &mut stamps));
        checked += c;
        hits += h;
    }
    // Far from the origin, where the skip margin grows with the coordinates.
    // Near 3e6, float32 spacing is 0.25 px. World's grids do not scale there,
    // so the trees are built directly.
    for (i, (name, scene)) in scenes.iter().enumerate() {
        for (x, y) in [(262144.5, -131072.25), (3145728.0, -2359296.0)] {
            if x > 1e6 && i >= SCENES.len() { continue; }
            let shift = |p: &Value| serde_json::json!([p[0].as_f64().unwrap() + x, p[1].as_f64().unwrap() + y]);
            let polygons: Vec<Value> = scene["track"]["polygons"].as_array().unwrap().iter()
                .map(|polygon| serde_json::json!({"points": polygon["points"].as_array().unwrap().iter().map(shift).collect::<Vec<_>>()}))
                .collect();
            let path: Vec<V2> = scene["track"]["path"].as_array().unwrap().iter().map(|p| altd_sim::world::vector(&shift(p))).collect();
            let tree = Node::from_polygons(&polygons).unwrap();
            let flat = RayTree::new(&tree);
            let (c, h) = compare_ray_tree(&format!("{name} shifted by {x}"), &tree, &path, &[], &mut random, |a, b| flat.raycast(a, b));
            checked += c;
            hits += h;
        }
    }
    assert!(checked > 2_000_000 && hits > 500_000, "{checked} rays, {hits} hits");
}

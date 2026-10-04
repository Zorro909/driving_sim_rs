//! Ordered raycast and closest-wall comparisons on the selected track.
use super::fixtures::scene_path;
use super::shared::{ok_line, Rng, Tally};
use altd_sim::gpu::hip::{Gpu, GpuWorld, Query};
use altd_sim::math::vec2::V2;
use altd_sim::track::world::World;

/// Points near the racing line (and a few far off it) for track queries.
fn track_points(world: &World, rng: &mut Rng, n: usize, spread: f64) -> Vec<V2> {
    let path = &world.track.path;
    (0..n)
        .map(|i| {
            let p = path[(rng.next() % path.len() as u64) as usize];
            let s = if i % 50 == 0 { spread * 20.0 } else { spread };
            V2::new(
                (p.x + rng.range(-s, s)) as f32 as f64,
                (p.y + rng.range(-s, s)) as f32 as f64,
            )
        })
        .collect()
}

pub(super) fn check_rays(gpu: &Gpu, world: &World, n: usize) -> bool {
    let mut rng = Rng(0xba5e);
    let gw = GpuWorld::new(gpu, world);
    let mut stamps = Default::default();
    let starts = track_points(world, &mut rng, n, 150.0);
    let mut input = Vec::with_capacity(n);
    for (i, s) in starts.iter().enumerate() {
        let angle = rng.range(-3.2, 3.2);
        let length = [200.0, 500.0, 800.0, 624.264, 1e-3, 0.0][i % 6] * rng.range(0.9, 1.1);
        input.push([
            s.x as f32,
            s.y as f32,
            (s.x + angle.cos() * length) as f32,
            (s.y + angle.sin() * length) as f32,
        ]);
    }
    // Rays ending exactly on a wall endpoint or along walls stress the ties.
    for (i, w) in world.track.walls.iter().enumerate().take(n / 10) {
        let s = starts[i];
        input[i] = [s.x as f32, s.y as f32, w.start.x as f32, w.start.y as f32];
    }
    let started = std::time::Instant::now();
    let out = gw.query(Query::Raycast, &input);
    let gpu_time = started.elapsed();
    let mut t = Tally::default();
    let mut hits = 0;
    for (q, o) in input.iter().zip(&out) {
        let want = world.track.raycast(
            V2::new(q[0] as f64, q[1] as f64),
            V2::new(q[2] as f64, q[3] as f64),
            &mut stamps,
        );
        let got = (o[2] != 0.0).then(|| V2::new(o[0] as f64, o[1] as f64));
        hits += want.is_some() as usize;
        let same = match (want, got) {
            (None, None) => true,
            (Some(a), Some(b)) => a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits(),
            _ => false,
        };
        t.record(0, same, || format!("ray {q:?}: cpu {want:?} gpu {got:?}"));
    }
    println!("track queries on {} ({n} each):", scene_path());
    ok_line(
        t.report("raycast"),
        &format!("{hits} hits, GPU {:.1} ms incl. copies", gpu_time.as_secs_f64() * 1e3),
    ) & {
        let points = track_points(world, &mut rng, n, 200.0);
        let input: Vec<[f32; 4]> = points.iter().map(|p| [p.x as f32, p.y as f32, 0.0, 0.0]).collect();
        let out = gw.query(Query::ClosestWall, &input);
        let mut t = Tally::default();
        for (p, o) in points.iter().zip(&out) {
            let want = world.track.closest_wall(*p).expect("BSP closest");
            let same = want.x.to_bits() == (o[0] as f64).to_bits() && want.y.to_bits() == (o[1] as f64).to_bits();
            t.record(0, same, || format!("point {p:?}: cpu {want:?} gpu {o:?}"));
        }
        t.report("closest_wall")
    }
}

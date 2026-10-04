//! The pruned nearest-segment searches must match the original linear scans
//! bit for bit, including far-away, on-path and exact-vertex queries.
use altd_sim::{
    math::godot_math::F2,
    math::vec2::V2,
    track::world::{Track, World, PATH_WINDOW},
};

#[path = "../examples/support/generated.rs"]
mod generated;

/// Original `Curve::closest_offset`.
fn curve_reference(points: &[F2], offsets: &[f32], query: F2) -> f32 {
    let (mut nearest, mut distance) = (0.0, -1.0);
    for i in 0..points.len() - 1 {
        let span = offsets[i + 1] - offsets[i];
        let origin = points[i];
        let v = points[i + 1] - origin;
        let direction = F2 {
            x: v.x / span,
            y: v.y / span,
        };
        let d = (query - origin).dot(direction).clamp(0.0, span);
        let projection = origin + direction * d;
        let delta = projection - query;
        let dist = delta.dot(delta);
        if distance < 0.0 || dist < distance {
            nearest = offsets[i] + d;
            distance = dist;
        }
    }
    nearest
}

/// Original `Track::closest_path(point, None)`.
fn path_reference(track: &Track, point: V2) -> (f64, V2, V2) {
    let count = track.path.len() - 1;
    let length = track.path_length() as f32;
    let posmod = |x: f32| {
        let r = x % length;
        if r < 0.0 {
            r + length
        } else {
            r
        }
    };
    let point = F2::from(point);
    let mut distance = f32::MAX;
    let mut best = (0.0, V2::ZERO, V2::ZERO);
    for i in 0..count {
        let start = F2::from(track.path[i]);
        let delta = F2::from(track.path[i + 1]) - start;
        let len2 = delta.dot(delta);
        let fraction = if len2 > 0.0 {
            ((point - start).dot(delta) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let near = start + delta * fraction;
        let diff = point - near;
        let d2 = diff.dot(diff);
        if d2 < distance {
            distance = d2;
            let begin = track.path_offsets[i] as f32;
            let end = track.path_offsets[i + 1] as f32;
            best = (
                if length > 0.0 {
                    posmod(begin + (end - begin) * fraction) as f64
                } else {
                    0.0
                },
                near.into(),
                delta.normalized().into(),
            );
        }
    }
    best
}

/// xorshift64*: deterministic query points without an RNG dependency.
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn queries(points: &[F2], rng: &mut Rng) -> Vec<F2> {
    let (lo, hi) = points
        .iter()
        .fold(((f32::MAX, f32::MAX), (f32::MIN, f32::MIN)), |(l, h), p| {
            ((l.0.min(p.x), l.1.min(p.y)), (h.0.max(p.x), h.1.max(p.y)))
        });
    let mut out = Vec::new();
    for _ in 0..20_000 {
        let x = lo.0 as f64 - 1500.0 + rng.unit() * (hi.0 - lo.0 + 3000.0) as f64;
        let y = lo.1 as f64 - 1500.0 + rng.unit() * (hi.1 - lo.1 + 3000.0) as f64;
        out.push(F2 {
            x: x as f32,
            y: y as f32,
        });
    }
    for (i, p) in points.iter().enumerate() {
        // Exact vertices, midpoints and sub-pixel jitter: ties and near-ties.
        out.push(*p);
        if let Some(q) = points.get(i + 1) {
            out.push(F2 {
                x: (p.x + q.x) * 0.5,
                y: (p.y + q.y) * 0.5,
            });
        }
        for scale in [1e-3, 0.5, 40.0, 300.0] {
            out.push(F2 {
                x: p.x + ((rng.unit() - 0.5) * scale) as f32,
                y: p.y + ((rng.unit() - 0.5) * scale) as f32,
            });
        }
    }
    out.push(F2 { x: 1e7, y: -1e7 });
    out
}

#[test]
fn pruned_path_searches_match_linear_scans() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut checked = 0;
    for slot in 0..5 {
        let world = World::from_scene(&generated::scene("rally", slot));
        let track = &world.track;
        let curve = track.curve.as_ref().expect("generated scenes carry the native curve");
        for q in queries(&curve.points, &mut rng) {
            assert_eq!(
                curve.closest_offset(q).to_bits(),
                curve_reference(&curve.points, &curve.offsets, q).to_bits(),
                "curve {q:?}"
            );
            checked += 1;
        }
        let path: Vec<F2> = track.path.iter().map(|&p| F2::from(p)).collect();
        assert!(
            track.path_length() > 2.0 * PATH_WINDOW,
            "exercise the unwindowed branch via previous=None"
        );
        for q in queries(&path, &mut rng) {
            let hit = track.closest_path(q.into(), None);
            let want = path_reference(track, q.into());
            assert_eq!(hit.offset.to_bits(), want.0.to_bits(), "path offset {q:?}");
            assert_eq!(
                (hit.near.x.to_bits(), hit.near.y.to_bits()),
                (want.1.x.to_bits(), want.1.y.to_bits()),
                "path near {q:?}"
            );
            assert_eq!(
                (hit.tangent.x.to_bits(), hit.tangent.y.to_bits()),
                (want.2.x.to_bits(), want.2.y.to_bits()),
                "path tangent {q:?}"
            );
            checked += 1;
        }
    }
    assert!(checked > 200_000, "{checked}");
}

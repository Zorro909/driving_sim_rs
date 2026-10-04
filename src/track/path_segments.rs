//! Baked path segments for `Track::closest_path`, stored as f32 columns.
//! Start, delta and squared length are the values TrackManager's projection
//! computes per query; precomputing them performs the same operations.
use crate::{math::godot_math::F2, math::vec2::V2};
use std::ops::Range;

#[derive(Default)]
pub(crate) struct PathSegments {
    x: Vec<f32>,
    y: Vec<f32>,
    dx: Vec<f32>,
    dy: Vec<f32>,
    len2: Vec<f32>,
}

/// One projection: squared distance, delta, clamped fraction and nearest point.
pub(crate) struct Projection {
    pub(crate) d2: f32,
    pub(crate) delta: F2,
    pub(crate) fraction: f32,
    pub(crate) near: F2,
}

impl PathSegments {
    pub(crate) fn new(path: &[V2]) -> PathSegments {
        let mut segments = PathSegments::default();
        for pair in path.windows(2) {
            let start = F2::from(pair[0]);
            let delta = F2::from(pair[1]) - start;
            segments.x.push(start.x);
            segments.y.push(start.y);
            segments.dx.push(delta.x);
            segments.dy.push(delta.y);
            segments.len2.push(delta.dot(delta));
        }
        segments
    }

    /// `[x, y, dx, dy, len2]` per segment (GPU export).
    pub(crate) fn rows(&self) -> Vec<[f32; 5]> {
        (0..self.len2.len())
            .map(|i| [self.x[i], self.y[i], self.dx[i], self.dy[i], self.len2[i]])
            .collect()
    }

    pub(crate) fn project(&self, point: F2, i: usize) -> Projection {
        let start = F2 {
            x: self.x[i],
            y: self.y[i],
        };
        let delta = F2 {
            x: self.dx[i],
            y: self.dy[i],
        };
        let len2 = self.len2[i];
        let fraction = if len2 > 0.0 {
            ((point - start).dot(delta) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let near = start + delta * fraction;
        let diff = point - near;
        Projection {
            d2: diff.dot(diff),
            delta,
            fraction,
            near,
        }
    }

    /// Visits `range` in order: a squared distance below `distance` replaces
    /// it and records the segment in `nearest`, so ties keep the first segment.
    pub(crate) fn scan(&self, point: F2, range: Range<usize>, distance: &mut f32, nearest: &mut Option<usize>) {
        assert!(range.end <= self.len2.len());
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            unsafe { self.scan_avx2(point, range, distance, nearest) };
            return;
        }
        self.scan_scalar(point, range, distance, nearest);
    }

    fn scan_scalar(&self, point: F2, range: Range<usize>, distance: &mut f32, nearest: &mut Option<usize>) {
        for i in range {
            let d2 = self.project(point, i).d2;
            if d2 < *distance {
                *distance = d2;
                *nearest = Some(i);
            }
        }
    }

    /// Eight segments per step with `project`'s operations; lanes are then
    /// compared in order, so the selected segment is the scalar one.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scan_avx2(&self, point: F2, range: Range<usize>, distance: &mut f32, nearest: &mut Option<usize>) {
        use std::arch::x86_64::*;
        let (px, py) = (_mm256_set1_ps(point.x), _mm256_set1_ps(point.y));
        let (zero, one) = (_mm256_setzero_ps(), _mm256_set1_ps(1.0));
        let mut i = range.start;
        while i + 8 <= range.end {
            let load = |column: &[f32]| _mm256_loadu_ps(column.as_ptr().add(i));
            let (sx, sy, dx, dy, len2) = (
                load(&self.x),
                load(&self.y),
                load(&self.dx),
                load(&self.dy),
                load(&self.len2),
            );
            let dot = _mm256_add_ps(
                _mm256_mul_ps(_mm256_sub_ps(px, sx), dx),
                _mm256_mul_ps(_mm256_sub_ps(py, sy), dy),
            );
            let q = _mm256_div_ps(dot, len2);
            // f32::clamp: NaN stays NaN.
            let q = _mm256_blendv_ps(q, zero, _mm256_cmp_ps::<_CMP_LT_OQ>(q, zero));
            let q = _mm256_blendv_ps(q, one, _mm256_cmp_ps::<_CMP_GT_OQ>(q, one));
            let fraction = _mm256_blendv_ps(zero, q, _mm256_cmp_ps::<_CMP_GT_OQ>(len2, zero));
            let ex = _mm256_sub_ps(px, _mm256_add_ps(sx, _mm256_mul_ps(dx, fraction)));
            let ey = _mm256_sub_ps(py, _mm256_add_ps(sy, _mm256_mul_ps(dy, fraction)));
            let d2 = _mm256_add_ps(_mm256_mul_ps(ex, ex), _mm256_mul_ps(ey, ey));
            if _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(d2, _mm256_set1_ps(*distance))) != 0 {
                let mut lanes = [0.0f32; 8];
                _mm256_storeu_ps(lanes.as_mut_ptr(), d2);
                for (lane, &d2) in lanes.iter().enumerate() {
                    if d2 < *distance {
                        *distance = d2;
                        *nearest = Some(i + lane);
                    }
                }
            }
            i += 8;
        }
        self.scan_scalar(point, i..range.end, distance, nearest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_scan_matches_scalar_scan() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x243F6A8885A308D3u64;
        for case in 0..400 {
            // Repeated points (zero-length segments), repeated segments (equal
            // distances) and a wide range of scales.
            let scale = [1e-3, 1.0, 64.0, 5000.0, 3e7][case % 5];
            let mut path: Vec<V2> = Vec::new();
            for _ in 0..(case % 97 + 2) {
                match (random(&mut state) * 6.0) as u32 {
                    0 if !path.is_empty() => path.push(*path.last().unwrap()),
                    1 if path.len() >= 2 => {
                        let n = path.len();
                        path.extend([path[n - 2], path[n - 1]]);
                    }
                    _ => path.push(V2::new(
                        (random(&mut state) - 0.5) * scale,
                        (random(&mut state) - 0.5) * scale,
                    )),
                }
            }
            if path.len() < 2 {
                path.push(path[0]);
            }
            let segments = PathSegments::new(&path);
            let count = path.len() - 1;
            for _ in 0..200 {
                let point = F2 {
                    x: ((random(&mut state) - 0.5) * scale * 1.5) as f32,
                    y: ((random(&mut state) - 0.5) * scale * 1.5) as f32,
                };
                let point = if random(&mut state) < 0.1 {
                    F2::from(path[(random(&mut state) * path.len() as f64) as usize])
                } else {
                    point
                };
                let a = (random(&mut state) * count as f64) as usize;
                let b = a + ((random(&mut state) * (count - a + 1) as f64) as usize).min(count - a);
                let initial = if random(&mut state) < 0.2 {
                    (random(&mut state) * scale * scale) as f32
                } else {
                    f32::MAX
                };
                let (mut d1, mut n1, mut d2, mut n2) = (initial, None, initial, None);
                segments.scan_scalar(point, a..b, &mut d1, &mut n1);
                unsafe { segments.scan_avx2(point, a..b, &mut d2, &mut n2) };
                assert_eq!((d1.to_bits(), n1), (d2.to_bits(), n2), "case {case} range {a}..{b}");
            }
        }
    }
}

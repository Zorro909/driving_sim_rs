//! Exact pruning for the game's linear nearest-segment scans.
//!
//! Godot's `Curve2D.get_closest_offset` and the track's path projection test
//! every baked segment. `SegmentGrid::candidates` returns, in ascending index
//! order, a superset of the segments whose float32 squared distance can be at
//! most the nearest one's. Callers rerun their original loop body on that
//! subset, so arithmetic, strict-less tie-breaking and NaN handling are
//! unchanged: every excluded segment is provably farther than an evaluated one.
use crate::math::godot_math::F2;
use std::cell::RefCell;

const CELL: f64 = 64.0;
const MAX_CELLS: usize = 1 << 22;

#[derive(Clone, Debug, Default)]
pub(crate) struct SegmentGrid {
    lo: (f64, f64),
    cell: f64,
    width: usize,
    height: usize,
    /// CSR offsets into `items`, one row per cell.
    start: Vec<u32>,
    items: Vec<u32>,
    boxes: Vec<[f64; 4]>,
    /// Absolute slack for float32 rounding of projected points.
    slack: f64,
}

thread_local! {
    static SCRATCH: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

impl SegmentGrid {
    /// Segment `i` joins `points[i]` and `points[i + 1]`. `None` if any point
    /// is not finite; callers then keep the linear scan.
    pub(crate) fn new(points: &[F2]) -> Option<SegmentGrid> {
        if points.len() < 2 || points.iter().any(|p| !p.x.is_finite() || !p.y.is_finite()) {
            return None;
        }
        let boxes: Vec<[f64; 4]> = points
            .windows(2)
            .map(|p| {
                let (a, b) = ((p[0].x as f64, p[0].y as f64), (p[1].x as f64, p[1].y as f64));
                [a.0.min(b.0), a.1.min(b.1), a.0.max(b.0), a.1.max(b.1)]
            })
            .collect();
        let lo = boxes
            .iter()
            .fold((f64::INFINITY, f64::INFINITY), |l, b| (l.0.min(b[0]), l.1.min(b[1])));
        let hi = boxes.iter().fold((f64::NEG_INFINITY, f64::NEG_INFINITY), |h, b| {
            (h.0.max(b[2]), h.1.max(b[3]))
        });
        let magnitude = [lo.0, lo.1, hi.0, hi.1].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut cell = CELL;
        let dims = |cell: f64| (((hi.0 - lo.0) / cell) as usize + 1, ((hi.1 - lo.1) / cell) as usize + 1);
        while dims(cell).0 * dims(cell).1 > MAX_CELLS {
            cell *= 2.0;
        }
        let (width, height) = dims(cell);
        let span = |b: &[f64; 4]| {
            let c = |v: f64, l: f64, n: usize| (((v - l) / cell) as usize).min(n - 1);
            (
                c(b[0], lo.0, width),
                c(b[1], lo.1, height),
                c(b[2], lo.0, width),
                c(b[3], lo.1, height),
            )
        };
        let mut counts = vec![0u32; width * height + 1];
        for b in &boxes {
            let (x0, y0, x1, y1) = span(b);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    counts[y * width + x + 1] += 1;
                }
            }
        }
        for i in 1..counts.len() {
            counts[i] += counts[i - 1];
        }
        let mut fill = counts.clone();
        let mut items = vec![0u32; *counts.last().unwrap() as usize];
        for (i, b) in boxes.iter().enumerate() {
            let (x0, y0, x1, y1) = span(b);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let slot = &mut fill[y * width + x];
                    items[*slot as usize] = i as u32;
                    *slot += 1;
                }
            }
        }
        Some(SegmentGrid {
            lo,
            cell,
            width,
            height,
            start: counts,
            items,
            boxes,
            slack: 1.0 + magnitude * 1e-4,
        })
    }

    fn cell_items(&self, x: usize, y: usize) -> &[u32] {
        let c = y * self.width + x;
        &self.items[self.start[c] as usize..self.start[c + 1] as usize]
    }

    /// Calls `visit` with the ascending candidate indices. `distance(i)` must be
    /// the caller's exact squared float32 distance to segment `i`. Returns
    /// `false` without visiting when no finite bound exists (use the full scan).
    pub(crate) fn scan(&self, query: F2, mut distance: impl FnMut(usize) -> f32, mut visit: impl FnMut(usize)) -> bool {
        let (qx, qy) = (query.x as f64, query.y as f64);
        if !qx.is_finite() || !qy.is_finite() {
            return false;
        }
        let clamp = |v: f64, l: f64, n: usize| (((v - l) / self.cell).max(0.0) as usize).min(n - 1);
        let (cx, cy) = (clamp(qx, self.lo.0, self.width), clamp(qy, self.lo.1, self.height));
        // Any evaluated finite distance bounds the nearest one from above.
        let mut bound = f32::INFINITY;
        let rings = self.width.max(self.height);
        for r in 0..=rings {
            let (x0, x1) = (cx.saturating_sub(r), (cx + r).min(self.width - 1));
            let (y0, y1) = (cy.saturating_sub(r), (cy + r).min(self.height - 1));
            for y in y0..=y1 {
                for x in x0..=x1 {
                    if x.abs_diff(cx) != r && y.abs_diff(cy) != r {
                        continue;
                    }
                    for &i in self.cell_items(x, y) {
                        let d = distance(i as usize);
                        if d < bound {
                            bound = d;
                        }
                    }
                }
            }
            if bound.is_finite() {
                break;
            }
        }
        if !bound.is_finite() {
            return false;
        }
        let radius = (bound as f64).sqrt() * (1.0 + 1e-4) + self.slack;
        let (x0, x1) = (
            clamp(qx - radius, self.lo.0, self.width),
            clamp(qx + radius, self.lo.0, self.width),
        );
        let (y0, y1) = (
            clamp(qy - radius, self.lo.1, self.height),
            clamp(qy + radius, self.lo.1, self.height),
        );
        SCRATCH.with(|scratch| {
            let mut candidates = scratch.borrow_mut();
            candidates.clear();
            for y in y0..=y1 {
                for x in x0..=x1 {
                    for &i in self.cell_items(x, y) {
                        let b = &self.boxes[i as usize];
                        let dx = (b[0] - qx).max(qx - b[2]).max(0.0);
                        let dy = (b[1] - qy).max(qy - b[3]).max(0.0);
                        if dx * dx + dy * dy <= radius * radius {
                            candidates.push(i);
                        }
                    }
                }
            }
            candidates.sort_unstable();
            candidates.dedup();
            candidates.iter().for_each(|&i| visit(i as usize));
        });
        true
    }
}

//! GPU lookup preparation must retain the original conservative candidates.
use altd_sim::gpu_sim::{near_grid, shape_grid};

fn distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let len2 = d[0] * d[0] + d[1] * d[1];
    let t = if len2 > 0.0 {
        (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / len2).clamp(0.0, 1.0)
    } else { 0.0 };
    let q = [a[0] + d[0] * t - p[0], a[1] + d[1] * t - p[1]];
    (q[0] * q[0] + q[1] * q[1]).sqrt()
}

fn random(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn gpu_grids_match_brute_force_lists_and_order() {
    let mut state = 0x243f6a8885a308d3;
    for case in 0..30 {
        let scale = [0.001, 1.0, 100000.0][case % 3];
        let mut segments = Vec::new();
        for i in 0..300 {
            let a = [(random(&mut state) - 0.5) * 512.0 * scale,
                     (random(&mut state) - 0.5) * 512.0 * scale];
            let b = if i % 5 == 0 { a } else {
                [a[0] + (random(&mut state) - 0.5) * 128.0 * scale,
                 a[1] + (random(&mut state) - 0.5) * 128.0 * scale]
            };
            segments.push((a, b));
            if i % 7 == 0 { segments.push((a, b)); }
        }
        let boxes: Vec<_> = segments.iter().map(|(a, b)|
            ([a[0].min(b[0]), a[1].min(b[1])], [a[0].max(b[0]), a[1].max(b[1])])).collect();
        let cell = 64.0 * scale;
        let margin = 128.0 * scale;
        let grid = near_grid(&segments, cell, margin).unwrap();
        let magnitude = [grid.x0, grid.y0, grid.x0 + grid.nx as f64 * cell, grid.y0 + grid.ny as f64 * cell]
            .iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let slack = 1.0 + magnitude * 1e-4;
        for cy in 0..grid.ny {
            for cx in 0..grid.nx {
                let bx = [grid.x0 + cx as f64 * cell - 1.0, grid.y0 + cy as f64 * cell - 1.0,
                          grid.x0 + (cx + 1) as f64 * cell + 1.0, grid.y0 + (cy + 1) as f64 * cell + 1.0];
                let corners = [[bx[0], bx[1]], [bx[2], bx[1]], [bx[0], bx[3]], [bx[2], bx[3]]];
                let bound = segments.iter().map(|&(a, b)|
                    corners.iter().map(|&p| distance(p, a, b)).fold(0.0f64, f64::max))
                    .fold(f64::INFINITY, f64::min);
                let limit = bound * (1.0 + 1e-4) + slack;
                let expected: Vec<_> = boxes.iter().enumerate().filter_map(|(i, (a, b))| {
                    let dx = (a[0] - bx[2]).max(bx[0] - b[0]).max(0.0);
                    let dy = (a[1] - bx[3]).max(bx[1] - b[1]).max(0.0);
                    ((dx * dx + dy * dy).sqrt() <= limit).then_some(i as u32)
                }).collect();
                let c = (cy * grid.nx + cx) as usize;
                assert_eq!(&grid.items[grid.start[c] as usize..grid.start[c + 1] as usize], expected,
                           "nearest grid case {case}, cell ({cx}, {cy})");
            }
        }
        let grid = shape_grid(&boxes, cell, margin).unwrap();
        for cy in 0..grid.ny {
            for cx in 0..grid.nx {
                let bx = [grid.x0 + cx as f64 * cell - margin, grid.y0 + cy as f64 * cell - margin,
                          grid.x0 + (cx + 1) as f64 * cell + margin, grid.y0 + (cy + 1) as f64 * cell + margin];
                let expected: Vec<_> = boxes.iter().enumerate().filter_map(|(i, (a, b))|
                    (a[0] - 1.0 <= bx[2] && b[0] + 1.0 >= bx[0] && a[1] - 1.0 <= bx[3] && b[1] + 1.0 >= bx[1])
                        .then_some(i as u32)).collect();
                let c = (cy * grid.nx + cx) as usize;
                assert_eq!(&grid.items[grid.start[c] as usize..grid.start[c + 1] as usize], expected,
                           "shape grid case {case}, cell ({cx}, {cy})");
            }
        }
    }
}

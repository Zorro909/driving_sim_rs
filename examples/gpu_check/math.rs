//! Float primitive comparisons, including nonfinite and exponent-wide inputs.
use super::shared::{Rng, Tally};
use altd_sim::gpu::hip::{Gpu, MathOp};
use altd_sim::{math::double_math, math::godot_math, math::native_math, nn::network};

/// Floats spread over every exponent, the simulator's typical ranges, and specials.
fn f32_inputs(rng: &mut Rng, n: usize, typical: f64) -> Vec<f32> {
    let mut v: Vec<f32> = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        1e-40,
        -1e-40,
    ]
    .into_iter()
    .chain([
        std::f32::consts::PI,
        -std::f32::consts::PI,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::FRAC_PI_4,
    ])
    .collect();
    while v.len() < n {
        v.push(match v.len() % 3 {
            0 => f32::from_bits(rng.next() as u32),
            1 => rng.range(-typical, typical) as f32,
            _ => rng.range(-1.0, 1.0) as f32,
        });
    }
    v
}

fn f64_inputs(rng: &mut Rng, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1e-310,
        709.78,
        -745.2,
        800.0,
        -800.0,
    ];
    while v.len() < n {
        v.push(if v.len().is_multiple_of(3) {
            f64::from_bits(rng.next())
        } else {
            rng.range(lo, hi)
        });
    }
    v
}

pub(super) fn check_math(gpu: &Gpu, n: usize, include_engine: bool) -> bool {
    let mut rng = Rng(0x5eed);
    let mut ok = true;
    println!("math primitives ({n} inputs each):");

    for (op, name, cpu) in [
        (MathOp::NativeSin, "native_sin", native_math::sin as fn(f32) -> f32),
        (MathOp::NativeCos, "native_cos", native_math::cos),
    ] {
        let input = f32_inputs(&mut rng, n, 40.0);
        let mut out = vec![0f32; n];
        let err = gpu.math(op, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            // The GPU flags angles that require the CPU's Payne-Hanek reduction.
            let want = cpu(input[i]);
            t.record(err[i], want.to_bits() == out[i].to_bits(), || {
                format!("{:e} cpu {want:e} gpu {:e}", input[i], out[i])
            });
        }
        ok &= t.report(name);
    }

    {
        let ys = f32_inputs(&mut rng, n, 1000.0);
        let xs = f32_inputs(&mut rng, n, 1000.0);
        let input: Vec<[f32; 2]> = ys.iter().zip(&xs).map(|(&y, &x)| [y, x]).collect();
        let mut out = vec![0f32; n];
        let err = gpu.math(MathOp::NativeAtan2, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let want = native_math::atan2(input[i][0], input[i][1]);
            t.record(err[i], want.to_bits() == out[i].to_bits(), || {
                format!("{:?} cpu {want:e} gpu {:e}", input[i], out[i])
            });
        }
        ok &= t.report("native_atan2");
    }

    {
        let input = f32_inputs(&mut rng, n, 20000.0);
        let mut out = vec![[0f32; 2]; n];
        let err = gpu.math(MathOp::ManagedSinCos, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let (s, c) = godot_math::managed_sin_cos(input[i]);
            let same = s.to_bits() == out[i][0].to_bits() && c.to_bits() == out[i][1].to_bits();
            t.record(err[i], same, || {
                format!("{:e} cpu ({s:e},{c:e}) gpu {:?}", input[i], out[i])
            });
        }
        ok &= t.report("managed_sin_cos");
    }

    {
        let input = f64_inputs(&mut rng, n, -760.0, 720.0);
        let mut out = vec![0f64; n];
        let err = gpu.math(MathOp::Exp, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let want = double_math::exp(input[i]);
            t.record(err[i], want.to_bits() == out[i].to_bits(), || {
                format!("{:e} cpu {want:e} gpu {:e}", input[i], out[i])
            });
        }
        ok &= t.report("exp");
    }

    {
        let xs = f64_inputs(&mut rng, n, 0.0, 1.0);
        let ys = f64_inputs(&mut rng, n, 0.05, 20.0);
        let mut input: Vec<[f64; 2]> = xs.iter().zip(&ys).map(|(&x, &y)| [x, y]).collect();
        // Negative bases with integer exponents, and the godot_ease shape 1 - pow(1 - v, 1 / c).
        for (i, pair) in input.iter_mut().enumerate().skip(16).step_by(7) {
            pair[0] = -pair[0] * 3.0;
            pair[1] = (i % 9) as f64;
        }
        let mut out = vec![0f64; n];
        let err = gpu.math(MathOp::Pow, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let want = double_math::pow(input[i][0], input[i][1]);
            t.record(err[i], want.to_bits() == out[i].to_bits(), || {
                format!("{:?} cpu {want:e} gpu {:e}", input[i], out[i])
            });
        }
        ok &= t.report("pow");
    }

    {
        let input = f64_inputs(&mut rng, n, -12.0, 12.0);
        let mut out = vec![0f64; n];
        let err = gpu.math(MathOp::GameTanh, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let want = network::game_tanh(input[i]);
            t.record(err[i], want.to_bits() == out[i].to_bits(), || {
                format!(
                    "{:e} ({:#018x}) cpu {want:e} ({:#018x}) gpu {:e} ({:#018x})",
                    input[i],
                    input[i].to_bits(),
                    want.to_bits(),
                    out[i],
                    out[i].to_bits()
                )
            });
        }
        ok &= t.report("game_tanh");
    }

    if include_engine {
        let mut input = f32_inputs(&mut rng, n, 16.0);
        for (i, x) in input.iter_mut().enumerate() {
            if i % 3 == 0 && x.is_finite() && x.abs() > 16.0 {
                *x = (i as f32 * 1e-3) % 32.0 - 16.0;
            }
        }
        let mut out = vec![[0f32; 2]; n];
        let err = gpu.math(MathOp::EngineSinCos, &input, &mut out);
        let mut t = Tally::default();
        for i in 0..n {
            let (s, c) = native_math::engine_sin_cos(input[i]);
            let same = s.to_bits() == out[i][0].to_bits() && c.to_bits() == out[i][1].to_bits();
            t.record(err[i], same, || {
                format!("{:e} cpu ({s:e},{c:e}) gpu {:?}", input[i], out[i])
            });
        }
        ok &= t.report("engine_sin_cos");
    }
    ok
}

//! Compares the crate's ucrt port (src/math/ucrt_math.rs) with the live Windows ucrtbase.dll.
//! Windows only. Run with `--release`; exits non-zero on any difference.
//! `--export PATH` also writes sample vectors from the DLL, tests/fixtures/ucrt_vectors.json.
#![allow(clippy::approx_constant)]
use altd_sim::math::ucrt_math as port;
use libloading::{Library, Symbol};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

struct Tally {
    name: String,
    checked: usize,
    bad: usize,
    first: Vec<String>,
}
impl Tally {
    fn new(name: &str) -> Tally {
        Tally {
            name: name.into(),
            checked: 0,
            bad: 0,
            first: vec![],
        }
    }
    fn record(&mut self, same: bool, what: impl FnOnce() -> String) {
        self.checked += 1;
        if !same {
            self.bad += 1;
            if self.first.len() < 40 {
                self.first.push(what());
            }
        }
    }
    fn report(&self) -> bool {
        println!(
            "{:<22} {:>9} checked {:>8} mismatches {}",
            self.name,
            self.checked,
            self.bad,
            if self.bad == 0 { "OK" } else { "FAIL" }
        );
        for line in &self.first {
            println!("    {line}");
        }
        self.bad == 0
    }
}

fn special32() -> Vec<f32> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
        1e-45,
        -1e-45,
        f32::MAX,
        f32::MIN,
    ];
    v.extend([
        f32::from_bits(0x7FA00001),
        f32::from_bits(0xFFC12345),
        f32::from_bits(0x7F800001),
        3.0e-5,
        1.0e-4,
        0.5,
        0.7853982,
        1.5707964,
        3.1415927,
        6.2831855,
    ]);
    v
}
fn special64() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::MIN_POSITIVE,
        5e-324,
        -5e-324,
        f64::MAX,
        f64::MIN,
        709.78,
        710.0,
        -745.0,
        -744.0,
        1e-300,
        1e300,
    ];
    v.extend([
        f64::from_bits(0x7FF4000000000001),
        f64::from_bits(0xFFF8000000000123),
        20.0,
        20.0000001,
        0.9,
        0.8999999,
        1e-9,
        3.0,
        0.1,
    ]);
    v
}

/// Every `STRIDE`th checked case, as hex bit patterns: inputs then the DLL's result.
struct Vectors(std::collections::BTreeMap<&'static str, Vec<Vec<String>>>);
impl Vectors {
    fn add(&mut self, name: &'static str, index: usize, bits: &[u64]) {
        // The specials come first in every list; keep all of them and a sample of the rest.
        if index < 250 || index.is_multiple_of(40_000) {
            self.0
                .entry(name)
                .or_default()
                .push(bits.iter().map(|b| format!("{b:#x}")).collect());
        }
    }
}

fn main() {
    let export = std::env::args().skip_while(|a| a != "--export").nth(1);
    let mut vectors = Vectors(Default::default());
    let lib = unsafe { Library::new("ucrtbase.dll") }.expect("ucrtbase.dll");
    let mut ok = true;
    unsafe {
        let sinf: Symbol<unsafe extern "C" fn(f32) -> f32> = lib.get(b"sinf\0").unwrap();
        let cosf: Symbol<unsafe extern "C" fn(f32) -> f32> = lib.get(b"cosf\0").unwrap();
        let atan2f: Symbol<unsafe extern "C" fn(f32, f32) -> f32> = lib.get(b"atan2f\0").unwrap();
        let exp: Symbol<unsafe extern "C" fn(f64) -> f64> = lib.get(b"exp\0").unwrap();
        let tanh: Symbol<unsafe extern "C" fn(f64) -> f64> = lib.get(b"tanh\0").unwrap();
        let pow: Symbol<unsafe extern "C" fn(f64, f64) -> f64> = lib.get(b"pow\0").unwrap();
        let n = 4_000_000;
        let mut rng = Rng(0x9e3779b97f4a7c15);
        let bits32 = |a: f32, b: f32| a.to_bits() == b.to_bits();
        let bits64 = |a: f64, b: f64| a.to_bits() == b.to_bits();

        for (name, ours, theirs, in_domain) in [
            (
                "sinf",
                port::sinf as fn(f32) -> f32,
                *sinf,
                port::sinf_in_domain as fn(f32) -> bool,
            ),
            (
                "cosf",
                port::cosf as fn(f32) -> f32,
                *cosf,
                port::cosf_in_domain as fn(f32) -> bool,
            ),
        ] {
            let mut t = Tally::new(name);
            let mut inputs = special32();
            for i in 0..n {
                inputs.push(match i % 5 {
                    0 => rng.range(-10.0, 10.0) as f32,
                    1 => rng.range(-10000.0, 10000.0) as f32,
                    2 => rng.range(-1.0e-3, 1.0e-3) as f32,
                    3 => rng.range(-1.0e7, 1.0e7) as f32,
                    _ => f32::from_bits(rng.next() as u32),
                });
            }
            for (index, x) in inputs.into_iter().enumerate() {
                if !in_domain(x) && x.is_finite() {
                    continue;
                }
                let (a, b) = (ours(x), theirs(x));
                vectors.add(
                    if name == "sinf" { "sinf" } else { "cosf" },
                    index,
                    &[x.to_bits() as u64, b.to_bits() as u64],
                );
                t.record(bits32(a, b), || {
                    format!(
                        "{x:e} ({:#x}) ours {:#x} ucrt {:#x}",
                        x.to_bits(),
                        a.to_bits(),
                        b.to_bits()
                    )
                });
            }
            ok &= t.report();
        }

        let mut t = Tally::new("atan2f");
        let mut pairs: Vec<(f32, f32)> = vec![];
        for &y in &special32() {
            for &x in &special32() {
                pairs.push((y, x));
            }
        }
        for i in 0..n {
            pairs.push(match i % 4 {
                0 => (rng.range(-100.0, 100.0) as f32, rng.range(-100.0, 100.0) as f32),
                1 => (rng.range(-1.0, 1.0) as f32, rng.range(-1.0e6, 1.0e6) as f32),
                2 => (f32::from_bits(rng.next() as u32), f32::from_bits(rng.next() as u32)),
                _ => (rng.range(-1.0e-30, 1.0e-30) as f32, rng.range(-1.0e-10, 1.0e10) as f32),
            });
        }
        for (index, (y, x)) in pairs.into_iter().enumerate() {
            let (a, b) = (port::atan2f(y, x), atan2f(y, x));
            vectors.add(
                "atan2f",
                index,
                &[y.to_bits() as u64, x.to_bits() as u64, b.to_bits() as u64],
            );
            t.record(bits32(a, b), || {
                format!("({y:e},{x:e}) ours {:#x} ucrt {:#x}", a.to_bits(), b.to_bits())
            });
        }
        ok &= t.report();

        let mut run1 =
            |name: &'static str, ours: &dyn Fn(f64) -> f64, theirs: &dyn Fn(f64) -> f64, ranges: &[(f64, f64)]| {
                let mut t = Tally::new(name);
                let mut inputs = special64();
                for i in 0..n {
                    let (lo, hi) = ranges[i % ranges.len()];
                    inputs.push(rng.range(lo, hi));
                    if i % 7 == 0 {
                        inputs.push(f64::from_bits(rng.next()));
                    }
                }
                for (index, x) in inputs.into_iter().enumerate() {
                    let (a, b) = (ours(x), theirs(x));
                    vectors.add(name, index, &[x.to_bits(), b.to_bits()]);
                    t.record(bits64(a, b), || {
                        format!(
                            "{x:e} ({:#x}) ours {:#x} ucrt {:#x}",
                            x.to_bits(),
                            a.to_bits(),
                            b.to_bits()
                        )
                    });
                }
                ok &= t.report();
            };
        run1(
            "exp",
            &port::exp,
            &|x| exp(x),
            &[(-745.0, 709.0), (-1.0, 1.0), (-1.0e-7, 1.0e-7), (-760.0, -700.0)],
        );
        run1(
            "tanh",
            &port::tanh,
            &|x| tanh(x),
            &[(-20.0, 20.0), (-1.0, 1.0), (-1e-8, 1e-8), (-30.0, 30.0)],
        );

        let mut t = Tally::new("pow");
        let mut pairs: Vec<(f64, f64)> = vec![];
        for &x in &special64() {
            for &y in &special64() {
                pairs.push((x, y));
            }
        }
        for i in 0..n {
            pairs.push(match i % 6 {
                0 => (rng.range(0.0, 10.0), rng.range(-30.0, 30.0)),
                1 => (rng.range(-3.0, 3.0), 2.0),
                2 => (rng.range(-3.0, 3.0), (rng.range(-10.0, 10.0)).round()),
                3 => (rng.range(0.0, 1.0e-5), rng.range(-2.0, 2.0)),
                4 => (rng.range(0.0, 1.0e5), rng.range(-400.0, 400.0)),
                _ => (f64::from_bits(rng.next()), f64::from_bits(rng.next())),
            });
        }
        for (index, (x, y)) in pairs.into_iter().enumerate() {
            let (a, b) = (port::pow(x, y), pow(x, y));
            vectors.add("pow", index, &[x.to_bits(), y.to_bits(), b.to_bits()]);
            t.record(bits64(a, b), || {
                format!("({x:e},{y:e}) ours {:#x} ucrt {:#x}", a.to_bits(), b.to_bits())
            });
        }
        ok &= t.report();
    }
    if let Some(path) = export {
        let json = serde_json::to_string_pretty(&vectors.0).unwrap();
        std::fs::write(
            &path,
            json + "
",
        )
        .unwrap();
        println!("wrote {path}");
    }
    if !ok {
        std::process::exit(1);
    }
}

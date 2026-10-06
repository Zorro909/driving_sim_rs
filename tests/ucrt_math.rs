//! The Windows UCRT port (src/math/ucrt_math.rs) against vectors recorded from ucrtbase.dll by
//! `cargo run --release --example ucrt_probe -- --export tests/fixtures/ucrt_vectors.json` (Windows).
//! The probe also compares the port with the live DLL over millions of inputs.
use altd_sim::math::{self, native_math, ucrt_math as port, Libm};
use serde_json::Value;

type Unary<T> = [(&'static str, fn(T) -> T); 2];

fn bits(value: &Value) -> u64 {
    u64::from_str_radix(value.as_str().unwrap().trim_start_matches("0x"), 16).unwrap()
}

#[test]
fn the_port_matches_ucrtbase() {
    let vectors: Value = serde_json::from_str(include_str!("fixtures/ucrt_vectors.json")).unwrap();
    let unary32: Unary<f32> = [("sinf", port::sinf), ("cosf", port::cosf)];
    let unary64: Unary<f64> = [("exp", port::exp), ("tanh", port::tanh)];
    let mut checked = 0;
    for (name, function) in unary32 {
        for row in vectors[name].as_array().unwrap() {
            let x = f32::from_bits(bits(&row[0]) as u32);
            assert_eq!(
                function(x).to_bits() as u64,
                bits(&row[1]),
                "{name}({:#x})",
                x.to_bits()
            );
            checked += 1;
        }
    }
    for row in vectors["atan2f"].as_array().unwrap() {
        let (y, x) = (
            f32::from_bits(bits(&row[0]) as u32),
            f32::from_bits(bits(&row[1]) as u32),
        );
        assert_eq!(
            port::atan2f(y, x).to_bits() as u64,
            bits(&row[2]),
            "atan2f({:#x}, {:#x})",
            y.to_bits(),
            x.to_bits()
        );
        checked += 1;
    }
    for (name, function) in unary64 {
        for row in vectors[name].as_array().unwrap() {
            let x = f64::from_bits(bits(&row[0]));
            assert_eq!(function(x).to_bits(), bits(&row[1]), "{name}({:#x})", x.to_bits());
            checked += 1;
        }
    }
    for row in vectors["pow"].as_array().unwrap() {
        let (x, y) = (f64::from_bits(bits(&row[0])), f64::from_bits(bits(&row[1])));
        assert_eq!(
            port::pow(x, y).to_bits(),
            bits(&row[2]),
            "pow({:#x}, {:#x})",
            x.to_bits(),
            y.to_bits()
        );
        checked += 1;
    }
    assert!(checked > 1000, "{checked} vectors");
}

#[test]
fn the_flavour_selects_the_ucrt_math() {
    // The Proton musl sinf and the UCRT's differ by one ulp here.
    let x = -4.70475f32;
    assert_eq!(math::libm(), Libm::Proton);
    assert_eq!(native_math::sin(x).to_bits(), 0x3f7ffe16);
    if cfg!(all(windows, target_env = "msvc")) {
        math::set_libm(Libm::Windows).unwrap();
        assert_eq!(math::libm(), Libm::Windows);
        assert_eq!(native_math::sin(x).to_bits(), 0x3f7ffe17);
        math::set_libm(Libm::Proton).unwrap();
        assert_eq!(native_math::sin(x).to_bits(), 0x3f7ffe16);
    } else {
        assert!(math::set_libm(Libm::Windows).is_err());
        assert_eq!(math::libm(), Libm::Proton);
    }
}

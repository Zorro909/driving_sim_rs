//! Native f32 division results for the browser's WGSL arithmetic regression.
//! cargo run --release --example webgpu_division_reference -- target/webgpu-division.bin
use std::io::{BufWriter, Write};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/webgpu-division.bin".into());
    if let Some(parent) = std::path::Path::new(&path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut out = BufWriter::new(std::fs::File::create(&path).expect("reference output"));
    let mut count = 0;
    let mut write = |a: u32, b: u32| {
        let quotient = f32::from_bits(a) / f32::from_bits(b);
        for bits in [a, b, quotient.to_bits(), 0] {
            out.write_all(&bits.to_le_bytes()).unwrap();
        }
        count += 1;
    };
    // Signed zero, subnormal/normal boundaries, rounding ties, and infinities.
    let edges = [
        0, 1, 2, 3, 0x3fffff, 0x7fffff, 0x800000, 0x800001, 0x3f000000, 0x3f7fffff, 0x3f800000, 0x3f800001, 0x3fc00000,
        0x40000000, 0x40400000, 0x7f7fffff, 0x7f800000,
    ];
    for a in edges {
        for b in edges {
            for sign_a in [0, 0x80000000] {
                for sign_b in [0, 0x80000000] {
                    write(a | sign_a, b | sign_b);
                }
            }
        }
    }
    // All exponent ranges, independent mantissas, deterministic signs.
    let mut state = 0x7155a123u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..1_048_576 {
        let a = next();
        let b = next();
        // Exclude NaN operands, retaining finite extremes and signed zeros.
        let finite = |bits: u32| {
            if (bits & 0x7fffffff) > 0x7f800000 {
                bits ^ 0x00800000
            } else {
                bits
            }
        };
        write(finite(a), finite(b));
    }
    out.flush().unwrap();
    println!("wrote {path}: {count} native divisions");
}

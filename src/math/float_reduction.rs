//! Wine/musl __rem_pio2_large, specialized to one float mantissa and prec=0.
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//! Developed at SunSoft, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this software is freely
//! granted, provided that this notice is preserved.

const TWO_OVER_PI: [i32; 66] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7,
    0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129, 0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C,
    0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11,
    0x8B5A0A, 0x6D1F6D, 0x367ECF, 0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7,
    0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3, 0x91615E,
    0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,
];
// These split coefficients retain the original reduction rounding.
#[allow(clippy::excessive_precision)]
const PI_OVER_TWO: [f64; 4] = [
    1.57079625129699707031,
    7.54978941586159635335e-8,
    5.39030252995776476554e-15,
    3.28200341580791294123e-22,
];
fn power_two(exponent: i32) -> f64 {
    f64::from_bits(((1023 + exponent) as u64) << 52)
}

pub(crate) fn reduce(x: f32) -> (i32, f64) {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let exponent = (magnitude >> 23) as i32 - 150;
    let mantissa = f32::from_bits(magnitude - ((exponent as u32) << 23)) as f64;
    let (n, y) = reduce_mantissa(mantissa, exponent);
    if bits >> 31 != 0 {
        (-n, -y)
    } else {
        (n, y)
    }
}

fn reduce_mantissa(x: f64, exponent: i32) -> (i32, f64) {
    let start = ((exponent - 3) / 24).max(0) as usize;
    let mut q0 = exponent - 24 * (start as i32 + 1);
    let mut q = [0.0; 20];
    let mut iq = [0i32; 20];
    let mut fq = [0.0; 20];
    for i in 0..=3 {
        q[i] = x * TWO_OVER_PI[start + i] as f64;
    }
    let mut end = 3usize;
    let (n, ih, mut z) = loop {
        let mut z = q[end];
        for i in 0..end {
            let whole = (z / 16777216.0) as i32 as f64;
            iq[i] = (z - 16777216.0 * whole) as i32;
            z = q[end - i - 1] + whole;
        }
        z *= power_two(q0);
        z -= 8.0 * (z * 0.125).floor();
        let mut n = z as i32;
        z -= n as f64;
        let mut ih = 0;
        if q0 > 0 {
            let high = iq[end - 1] >> (24 - q0);
            n += high;
            iq[end - 1] -= high << (24 - q0);
            ih = iq[end - 1] >> (23 - q0);
        } else if q0 == 0 {
            ih = iq[end - 1] >> 23;
        } else if z >= 0.5 {
            ih = 2;
        }
        if ih > 0 {
            n += 1;
            let mut carry = false;
            for chunk in &mut iq[..end] {
                if !carry {
                    if *chunk != 0 {
                        carry = true;
                        *chunk = 0x1000000 - *chunk;
                    }
                } else {
                    *chunk = 0xffffff - *chunk;
                }
            }
            if q0 == 1 {
                iq[end - 1] &= 0x7fffff;
            } else if q0 == 2 {
                iq[end - 1] &= 0x3fffff;
            }
            if ih == 2 {
                z = 1.0 - z;
                if carry {
                    z -= power_two(q0);
                }
            }
        }
        if z == 0.0 && iq[3..end].iter().all(|&v| v == 0) {
            let mut extra = 1;
            while iq[3 - extra] == 0 {
                extra += 1;
            }
            for i in end + 1..=end + extra {
                q[i] = x * TWO_OVER_PI[start + i] as f64;
            }
            end += extra;
            continue;
        }
        break (n, ih, z);
    };
    if z == 0.0 {
        end -= 1;
        q0 -= 24;
        while iq[end] == 0 {
            end -= 1;
            q0 -= 24;
        }
    } else {
        z *= power_two(-q0);
        if z >= 16777216.0 {
            let whole = (z / 16777216.0) as i32;
            iq[end] = (z - 16777216.0 * whole as f64) as i32;
            end += 1;
            q0 += 24;
            iq[end] = whole;
        } else {
            iq[end] = z as i32;
        }
    }
    let mut scale = power_two(q0);
    for i in (0..=end).rev() {
        q[i] = scale * iq[i] as f64;
        scale *= 1.0 / 16777216.0;
    }
    for i in (0..=end).rev() {
        let mut sum = 0.0;
        for k in 0..=3.min(end - i) {
            sum += PI_OVER_TWO[k] * q[i + k];
        }
        fq[end - i] = sum;
    }
    let mut y = 0.0;
    for i in (0..=end).rev() {
        y += fq[i];
    }
    (n & 7, if ih == 0 { y } else { -y })
}

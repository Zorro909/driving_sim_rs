//! Feed-forward networks with `tanh(input @ weights + bias)` per layer.
//!
//! Parameters are stored flat in `Network.vector()` order: for each layer the
//! input-neuron weight rows, then the bias.

use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    pub shape: Vec<usize>,
    pub params: Vec<f64>,
}

/// Reusable activation buffers for `Network::forward_into`.
#[derive(Default, Clone)]
pub struct ForwardScratch {
    values: Vec<f64>,
    next: Vec<f64>,
}

pub fn parameter_count(shape: &[usize]) -> usize {
    shape.windows(2).map(|w| (w[0] + 1) * w[1]).sum()
}

impl Network {
    pub fn xavier_game(shape: &[usize], rng: &mut crate::training::game_random::GameRandom) -> Network {
        let mut params = Vec::with_capacity(parameter_count(shape));
        for layer in shape.windows(2) {
            let (rows, cols) = (layer[0], layer[1]);
            let weights = rng.normal_array(rows * cols, (2.0 / (rows + cols) as f64).sqrt());
            for row in 0..rows {
                for col in 0..cols {
                    params.push(weights[col * rows + row]);
                }
            }
            params.extend(rng.normal_array(cols, 0.1));
        }
        Self::from_vector(shape, params)
    }
    // Column-major normal draws map onto row-major parameters by their original indices.
    #[allow(clippy::needless_range_loop)]
    pub fn mutate_xavier_game(&self, rate: f64, rng: &mut crate::training::game_random::GameRandom) -> Network {
        let mut output = self.clone();
        if rate <= 0.0 {
            return output;
        }
        let mut offset = 0;
        for layer in self.shape.windows(2) {
            let (rows, cols) = (layer[0], layer[1]);
            let noise = rng.normal_array(rows * cols, (2.0 / (rows + cols) as f64).sqrt() * rate);
            for row in 0..rows {
                for col in 0..cols {
                    output.params[offset + row * cols + col] += noise[col * rows + row];
                }
            }
            offset += (rows + 1) * cols;
        }
        offset = 0;
        for layer in self.shape.windows(2) {
            let (rows, cols) = (layer[0], layer[1]);
            let noise = rng.normal_array(cols, 0.1 * rate);
            for col in 0..cols {
                output.params[offset + rows * cols + col] += noise[col];
            }
            offset += (rows + 1) * cols;
        }
        output
    }
    pub fn from_vector(shape: &[usize], params: Vec<f64>) -> Network {
        assert!(
            shape.len() >= 2 && shape.iter().all(|&n| n > 0),
            "a network needs positive input and output layer sizes"
        );
        let expected = parameter_count(shape);
        assert_eq!(
            params.len(),
            expected,
            "expected {expected} parameters, got {}",
            params.len()
        );
        Network {
            shape: shape.to_vec(),
            params,
        }
    }

    /// `Network.xavier`: weights `gauss(0, sqrt(2 / (in + out)))`, biases
    /// `gauss(0, .1)`, drawn layer by layer in parameter order like the game's
    /// `FillRandomWithXavier`.
    pub fn xavier(shape: &[usize], rng: &mut crate::training::pyrandom::PyRandom) -> Network {
        let mut params = Vec::with_capacity(parameter_count(shape));
        for w in shape.windows(2) {
            let (inputs, outputs) = (w[0], w[1]);
            let deviation = (2.0 / (inputs + outputs) as f64).sqrt();
            params.extend((0..inputs * outputs).map(|_| rng.gauss(deviation)));
            params.extend((0..outputs).map(|_| rng.gauss(0.1)));
        }
        Network::from_vector(shape, params)
    }

    /// Load an exported network with shape, weights, and biases.
    pub fn from_game_export(data: &Value) -> Network {
        Self::try_from_game_export(data).unwrap_or_else(|e| panic!("{e}"))
    }

    /// `from_game_export` reporting malformed exports instead of panicking.
    pub(crate) fn try_from_game_export(data: &Value) -> Result<Network, String> {
        let summary = data.get("summary").unwrap_or(data);
        let shape: Vec<usize> = summary["shape"]
            .as_array()
            .ok_or("missing network shape")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or("network shape must hold positive integers")
            })
            .collect::<Result<_, _>>()?;
        if shape.len() < 2 || shape.contains(&0) {
            return Err(format!("invalid network shape {shape:?}"));
        }
        let weights = data["weights"].as_array().ok_or("missing weights")?;
        let biases = data["biases"].as_array().ok_or("missing biases")?;
        if weights.len() != shape.len() - 1 || biases.len() != weights.len() {
            return Err("the weight and bias layer count does not match shape".into());
        }
        let number = |v: &Value| v.as_f64().ok_or("network parameters must be numbers");
        let mut params = Vec::with_capacity(parameter_count(&shape));
        for (layer, (matrix, bias)) in weights.iter().zip(biases).enumerate() {
            let (matrix, bias) = (
                matrix.as_array().ok_or("weights must be nested arrays")?,
                bias.as_array().ok_or("biases must be arrays")?,
            );
            if matrix.len() != shape[layer] || bias.len() != shape[layer + 1] {
                return Err(format!("layer {layer} dimensions do not match shape"));
            }
            for row in matrix {
                let row = row.as_array().ok_or("weights must be nested arrays")?;
                if row.len() != bias.len() {
                    return Err(format!("layer {layer} weight row has the wrong size"));
                }
                for v in row {
                    params.push(number(v)?);
                }
            }
            for v in bias {
                params.push(number(v)?);
            }
        }
        Ok(Network::from_vector(&shape, params))
    }

    /// Nested network schema with `shape`, `weights`, and `biases`.
    pub fn to_json(&self) -> Value {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut position = 0;
        for w in self.shape.windows(2) {
            let (inputs, outputs) = (w[0], w[1]);
            let mut matrix = Vec::new();
            for _ in 0..inputs {
                matrix.push(json!(self.params[position..position + outputs]));
                position += outputs;
            }
            weights.push(Value::Array(matrix));
            biases.push(json!(self.params[position..position + outputs]));
            position += outputs;
        }
        json!({"shape": self.shape, "weights": weights, "biases": biases})
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        let mut scratch = ForwardScratch::default();
        self.forward_into(inputs, &mut scratch).to_vec()
    }

    /// `Network.forward`: each output is `tanh(sum(v_i * W[i][j]) + b[j])`
    /// where `sum` follows the game's managed MathNet accumulation order.
    pub fn forward_into<'a>(&self, inputs: &[f64], scratch: &'a mut ForwardScratch) -> &'a [f64] {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            assert_eq!(
                inputs.len(),
                self.shape[0],
                "expected {} sensor inputs, got {}",
                self.shape[0],
                inputs.len()
            );
            let ForwardScratch { values, next } = scratch;
            values.clear();
            values.extend_from_slice(inputs);
            // SAFETY: AVX2 is available; the kernel is bit-identical to the loop below.
            unsafe { crate::nn::network_simd::forward(&self.shape, &self.params, values, next) };
            return &values[..];
        }
        self.forward_into_scalar(inputs, scratch)
    }

    /// Portable reference for `forward_into`.
    pub(crate) fn forward_into_scalar<'a>(&self, inputs: &[f64], scratch: &'a mut ForwardScratch) -> &'a [f64] {
        assert_eq!(
            inputs.len(),
            self.shape[0],
            "expected {} sensor inputs, got {}",
            self.shape[0],
            inputs.len()
        );
        let ForwardScratch { values, next } = scratch;
        values.clear();
        values.extend_from_slice(inputs);
        let mut position = 0;
        for w in self.shape.windows(2) {
            let (n_in, n_out) = (w[0], w[1]);
            next.clear();
            next.resize(n_out, 0.0);
            let matrix = &self.params[position..position + n_in * n_out];
            let bias = &self.params[position + n_in * n_out..position + (n_in + 1) * n_out];
            // The game's managed MathNet provider adds products in input order.
            for (i, &value) in values.iter().enumerate() {
                let row = &matrix[i * n_out..(i + 1) * n_out];
                for j in 0..n_out {
                    next[j] += value * row[j];
                }
            }
            for j in 0..n_out {
                next[j] = game_tanh(next[j] + bias[j]);
            }
            std::mem::swap(values, next);
            position += (n_in + 1) * n_out;
        }
        &values[..]
    }
}

// tanh/expm1 adapted from Wine's musl math implementation (proton_11.0).
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this software is freely
// granted, provided that this notice is preserved.
// Preserve musl polynomial and range-reduction constants and arithmetic grouping.
#[allow(clippy::approx_constant, clippy::excessive_precision, clippy::manual_range_contains)]
pub(crate) fn game_expm1(mut x: f64) -> f64 {
    let high = ((x.to_bits() >> 32) as u32) & 0x7fff_ffff;
    let negative = x.is_sign_negative();
    if high >= 0x4043687a {
        if x.is_nan() {
            return x;
        }
        if x.is_infinite() {
            return if negative { -1.0 } else { x };
        }
        if negative {
            return -1.0;
        }
        if x > 7.09782712893383973096e2 {
            return f64::INFINITY;
        }
    }
    let (k, c);
    if high > 0x3fd62e42 {
        let (hi, lo);
        if high < 0x3ff0a2b2 {
            if !negative {
                hi = x - 6.93147180369123816490e-1;
                lo = 1.90821492927058770002e-10;
                k = 1;
            } else {
                hi = x + 6.93147180369123816490e-1;
                lo = -1.90821492927058770002e-10;
                k = -1;
            }
        } else {
            k = (1.44269504088896338700 * x + if negative { -0.5 } else { 0.5 }) as i32;
            hi = x - k as f64 * 6.93147180369123816490e-1;
            lo = k as f64 * 1.90821492927058770002e-10;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if high < 0x3c900000 {
        return x;
    } else {
        k = 0;
        c = 0.0;
    }
    let hfx = 0.5 * x;
    let hxs = x * hfx;
    let r1 = 1.0
        + hxs
            * (-3.33333333333331316428e-2
                + hxs
                    * (1.58730158725481460165e-3
                        + hxs
                            * (-7.93650757867487942473e-5
                                + hxs * (4.00821782732936239552e-6 + hxs * (-2.01099218183624371326e-7)))));
    let t = 3.0 - r1 * hfx;
    let mut e = hxs * ((r1 - t) / (6.0 - x * t));
    if k == 0 {
        return x - (x * e - hxs);
    }
    e = x * (e - c) - c;
    e -= hxs;
    if k == -1 {
        return 0.5 * (x - e) - 0.5;
    }
    if k == 1 {
        return if x < -0.25 {
            -2.0 * (e - (x + 0.5))
        } else {
            1.0 + 2.0 * (x - e)
        };
    }
    let twopk = f64::from_bits(((0x3ff + k) as u64) << 52);
    if k < 0 || k > 56 {
        let y = x - e + 1.0;
        return (if k == 1024 {
            y * 2.0 * f64::from_bits(0x7fe0000000000000)
        } else {
            y * twopk
        }) - 1.0;
    }
    let inverse = f64::from_bits(((0x3ff - k) as u64) << 52);
    if k < 20 {
        (x - e + (1.0 - inverse)) * twopk
    } else {
        (x - (e + inverse) + 1.0) * twopk
    }
}

/// Windows .NET 8.0.2 Math.Tanh under the game's Proton runtime.
pub fn game_tanh(value: f64) -> f64 {
    let x = value.abs();
    let high = (x.to_bits() >> 32) as u32;
    let t = if high > 0x3fe193ea {
        if high > 0x40340000 {
            1.0 - 0.0 / x
        } else {
            let t = game_expm1(2.0 * x);
            1.0 - 2.0 / (t + 2.0)
        }
    } else if high > 0x3fd058ae {
        let t = game_expm1(2.0 * x);
        t / (t + 2.0)
    } else if high >= 0x00100000 {
        let t = game_expm1(-2.0 * x);
        -t / (t + 2.0)
    } else {
        x
    };
    if value.is_sign_negative() {
        -t
    } else {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_layout_matches_python() {
        // Network.from_vector((2, 1), [w00, w10, b0]).forward([1, 2]) = tanh(w00 + 2*w10 + b0)
        let network = Network::from_vector(&[2, 1], vec![0.5, -0.25, 0.1]);
        let out = network.forward(&[1.0, 2.0]);
        assert_eq!(out, vec![(0.5f64 - 0.5 + 0.1).tanh()]);
    }

    #[test]
    fn xavier_matches_python() {
        // Network.xavier([20,16,16,16,16,12,12,8,5], random.Random(1)).vector()
        let shape = [20, 16, 16, 16, 16, 12, 12, 8, 5];
        let mut rng = crate::training::pyrandom::PyRandom::new(1);
        let v = Network::xavier(&shape, &mut rng).params;
        assert_eq!(v.len(), 1661);
        assert_eq!(
            (v[0], v[319], v[320], v[1660]),
            (
                0.3036280581257822,
                -0.15109602765744476,
                0.0006248398336010655,
                -0.05605915218919258
            )
        );
        // The checksum was captured with Linux CPython's host libm. On macOS,
        // compare every parameter and cached draw with CPython on that host:
        // both libm rounding and LLVM's sin/cos merging need an exact oracle.
        #[cfg(not(target_os = "macos"))]
        assert_eq!(crate::math::pymath::py_sum(v.iter().copied()), 2.8926482371919495);
        #[cfg(target_os = "macos")]
        {
            let reference = python_xavier_reference(&shape);
            let expected_bits: Vec<u64> = serde_json::from_value(reference["param_bits"].clone()).unwrap();
            assert_eq!(v.len(), expected_bits.len());
            for (index, (actual, &expected)) in v.iter().zip(&expected_bits).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected,
                    "parameter {index}: Rust {actual:?}, CPython {:?}",
                    f64::from_bits(expected)
                );
            }
            assert_eq!(rng.to_json(), reference["rng"], "Xavier RNG and cached Gaussian state");
        }
    }

    #[cfg(target_os = "macos")]
    fn python_xavier_reference(shape: &[usize]) -> Value {
        // CPython calls sin and cos separately. Keep the oracle outside Rust
        // so LLVM cannot merge its operations with those under test.
        const SCRIPT: &str = r#"
import json, math, random, struct, sys
shape = json.loads(sys.argv[1])
rng = random.Random(1)
params = []
for inputs, outputs in zip(shape, shape[1:]):
    deviation = math.sqrt(2.0 / (inputs + outputs))
    params.extend(rng.gauss(0.0, deviation) for _ in range(inputs * outputs))
    params.extend(rng.gauss(0.0, 0.1) for _ in range(outputs))
_, state, cached = rng.getstate()
def bits(value):
    return struct.unpack('!Q', struct.pack('!d', value))[0]
cached_bits = None if cached is None else bits(cached)
print(json.dumps({'param_bits': [bits(value) for value in params], 'rng': {
    'state': state[:-1], 'index': state[-1], 'gauss_next_bits': cached_bits
}}))
"#;
        let output = std::process::Command::new("python3")
            .args(["-c", SCRIPT, &serde_json::to_string(shape).unwrap()])
            .output()
            .expect("macOS Xavier fidelity test requires python3");
        assert!(
            output.status.success(),
            "CPython Xavier oracle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).expect("CPython Xavier oracle must return JSON")
    }
}

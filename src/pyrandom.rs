//! CPython's `random.Random` (MT19937) for the calls the trainer makes, so a
//! seeded Rust run reproduces `python -m driving_sim.train` exactly.

const N: usize = 624;
const M: usize = 397;

#[derive(Clone)]
pub struct PyRandom {
    state: [u32; N],
    index: usize,
    gauss_next: Option<f64>,
}

impl PyRandom {
    /// `random.Random(seed)` for an integer seed.
    pub fn new(seed: i64) -> PyRandom {
        let mut n = seed.unsigned_abs();
        let mut key = Vec::new();
        while n != 0 {
            key.push(n as u32);
            n >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        let mut rng = PyRandom { state: [0; N], index: N, gauss_next: None };
        rng.init_by_array(&key);
        rng
    }

    /// Complete generator state for checkpoints (`gauss_next` as raw bits).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "state": self.state.to_vec(),
            "index": self.index,
            "gauss_next_bits": self.gauss_next.map(f64::to_bits),
        })
    }

    pub fn from_json(data: &serde_json::Value) -> PyRandom {
        let words: Vec<u32> =
            data["state"].as_array().expect("rng state").iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let index = data["index"].as_u64().expect("rng index") as usize;
        assert!(words.len() == N && index <= N, "invalid random state");
        PyRandom {
            state: words.try_into().unwrap(),
            index,
            gauss_next: data["gauss_next_bits"].as_u64().map(f64::from_bits),
        }
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..N {
            let previous = self.state[i - 1];
            self.state[i] = 1812433253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..N.max(key.len()) {
            let previous = self.state[i - 1];
            self.state[i] = (self.state[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.state[0] = self.state[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let previous = self.state[i - 1];
            self.state[i] = (self.state[i]
                ^ (previous ^ (previous >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.state[0] = self.state[N - 1];
                i = 1;
            }
        }
        self.state[0] = 0x8000_0000;
    }

    fn twist(&mut self) {
        const MAG01: [u32; 2] = [0, 0x9908_b0df];
        let s = &mut self.state;
        for k in 0..N - M {
            let y = (s[k] & 0x8000_0000) | (s[k + 1] & 0x7fff_ffff);
            s[k] = s[k + M] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
        }
        for k in N - M..N - 1 {
            let y = (s[k] & 0x8000_0000) | (s[k + 1] & 0x7fff_ffff);
            s[k] = s[k + M - N] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
        }
        let y = (s[N - 1] & 0x8000_0000) | (s[0] & 0x7fff_ffff);
        s[N - 1] = s[M - 1] ^ (y >> 1) ^ MAG01[(y & 1) as usize];
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random.random()`.
    pub fn random(&mut self) -> f64 {
        let a = self.next_u32() >> 5;
        let b = self.next_u32() >> 6;
        (a as f64 * 67108864.0 + b as f64) * (1.0 / 9007199254740992.0)
    }

    /// `random.gauss(0, sigma)`, including the cached second variate.
    pub fn gauss(&mut self, sigma: f64) -> f64 {
        let z = match self.gauss_next.take() {
            Some(z) => z,
            None => {
                let x2pi = self.random() * std::f64::consts::TAU;
                let g2rad = (-2.0 * (1.0 - self.random()).ln()).sqrt();
                self.gauss_next = Some(x2pi.sin() * g2rad);
                x2pi.cos() * g2rad
            }
        };
        0.0 + z * sigma
    }

    /// The next `count` standard normal variates of `gauss`, identical to
    /// `count` sequential `gauss(1)` calls (the result is `z`, not `0.0 + z`).
    /// Uniform draws stay sequential; the transcendental math runs in parallel.
    pub fn standard_normals(&mut self, count: usize) -> Vec<f64> {
        use rayon::prelude::*;
        let mut out = Vec::with_capacity(count + 1);
        if count > 0 {
            if let Some(z) = self.gauss_next.take() {
                out.push(z);
            }
        }
        // Store each pair's uniforms in place, then transform the pairs in parallel.
        let offset = out.len();
        let pairs = (count - offset).div_ceil(2);
        for _ in 0..pairs {
            out.push(self.random());
            out.push(self.random());
        }
        out[offset..].par_chunks_exact_mut(2).with_min_len(4096).for_each(|pair| {
            let x2pi = pair[0] * std::f64::consts::TAU;
            let g2rad = (-2.0 * (1.0 - pair[1]).ln()).sqrt();
            pair[0] = x2pi.cos() * g2rad;
            pair[1] = x2pi.sin() * g2rad;
        });
        if out.len() > count {
            self.gauss_next = out.pop();
        }
        out
    }

    /// `random.getrandbits(k)` for `1 <= k <= 32`.
    fn getrandbits(&mut self, k: u32) -> u32 {
        self.next_u32() >> (32 - k)
    }

    /// `random.randrange(n)` for `1 <= n < 2**32`.
    pub fn randrange(&mut self, n: usize) -> usize {
        assert!(n >= 1 && n <= u32::MAX as usize, "randrange bound out of range");
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = self.getrandbits(k) as usize;
            if r < n {
                return r;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_cpython_seed_7() {
        // python3 -c "import random; r=random.Random(7); print(r.random(), r.random())"
        let mut rng = PyRandom::new(7);
        assert_eq!(rng.random(), 0.32383276483316237);
        assert_eq!(rng.random(), 0.15084917392450192);
    }

    #[test]
    fn gauss_randrange_and_seeds_match_cpython() {
        // r = random.Random(7); r.random(); r.random(); r.gauss(0, 1); r.gauss(0, 1);
        // r.randrange(300); r.randrange(5); r.gauss(0, .3)
        let mut rng = PyRandom::new(7);
        rng.random();
        rng.random();
        assert_eq!(rng.gauss(1.0), -0.2260961647831047);
        assert_eq!(rng.gauss(1.0), -0.3150684223311854);
        assert_eq!(rng.randrange(300), 274);
        assert_eq!(rng.randrange(5), 0);
        assert_eq!(rng.gauss(0.3), -0.06891764470260762);
        // Negative seeds use their absolute value; large seeds use several key words.
        assert_eq!(PyRandom::new(-3).random(), 0.23796462709189137);
        assert_eq!(PyRandom::new((1 << 40) + 5).random(), 0.5043802970418443);
    }

    #[test]
    fn json_state_round_trips() {
        let mut a = PyRandom::new(5);
        a.gauss(1.0);
        let mut b = PyRandom::from_json(&a.to_json());
        for _ in 0..1000 {
            assert_eq!(a.gauss(1.0).to_bits(), b.gauss(1.0).to_bits());
        }
    }

    #[test]
    fn batched_normals_match_sequential_gauss() {
        for (skip, count) in [(0, 0), (0, 1), (1, 7), (3, 10000), (2, 10001)] {
            let (mut a, mut b) = (PyRandom::new(11), PyRandom::new(11));
            for _ in 0..skip {
                assert_eq!(a.gauss(1.0), b.gauss(1.0));
            }
            let batch = a.standard_normals(count);
            for z in batch {
                assert_eq!((0.0 + z).to_bits(), b.gauss(1.0).to_bits());
            }
            assert_eq!(a.gauss(1.0).to_bits(), b.gauss(1.0).to_bits());
            assert_eq!(a.random().to_bits(), b.random().to_bits());
        }
    }
}

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
        let mut rng = PyRandom {
            state: [0; N],
            index: N,
            gauss_next: None,
        };
        rng.init_by_array(&key);
        rng
    }

    /// Complete generator state for checkpoints (`gauss_next` as raw bits).
    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "state": self.state.to_vec(),
            "index": self.index,
            "gauss_next_bits": self.gauss_next.map(f64::to_bits),
        })
    }

    pub(crate) fn from_json(data: &serde_json::Value) -> PyRandom {
        let words: Vec<u32> = data["state"]
            .as_array()
            .expect("rng state")
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
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
            self.state[i] =
                (self.state[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.state[0] = self.state[N - 1];
                i = 1;
            }
        }
        self.state[0] = 0x8000_0000;
    }

    fn twist(&mut self) {
        let s = &mut self.state;
        for k in 0..N - M {
            let y = (s[k] & 0x8000_0000) | (s[k + 1] & 0x7fff_ffff);
            s[k] = s[k + M] ^ (y >> 1) ^ (0u32.wrapping_sub(y & 1) & 0x9908_b0df);
        }
        for k in N - M..N - 1 {
            let y = (s[k] & 0x8000_0000) | (s[k + 1] & 0x7fff_ffff);
            s[k] = s[k + M - N] ^ (y >> 1) ^ (0u32.wrapping_sub(y & 1) & 0x9908_b0df);
        }
        let y = (s[N - 1] & 0x8000_0000) | (s[0] & 0x7fff_ffff);
        s[N - 1] = s[M - 1] ^ (y >> 1) ^ (0u32.wrapping_sub(y & 1) & 0x9908_b0df);
        self.index = 0;
    }

    #[inline]
    fn temper(mut y: u32) -> u32 {
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let y = self.state[self.index];
        self.index += 1;
        Self::temper(y)
    }

    /// `random.random()`.
    pub(crate) fn random(&mut self) -> f64 {
        let a = self.next_u32() >> 5;
        let b = self.next_u32() >> 6;
        (a as f64 * 67108864.0 + b as f64) * (1.0 / 9007199254740992.0)
    }

    /// Draw in MT state-sized blocks so tempering and conversion can be
    /// vectorized. Twist boundaries and the final generator state are exactly
    /// those of sequential `random()` calls, including odd starting indices.
    fn fill_uniforms(&mut self, mut out: &mut [f64]) {
        while !out.is_empty() {
            if self.index == N {
                self.twist();
            }
            if self.index == N - 1 {
                out[0] = self.random();
                out = &mut out[1..];
                continue;
            }
            let count = out.len().min((N - self.index) / 2);
            let words = &self.state[self.index..self.index + count * 2];
            for (value, pair) in out[..count].iter_mut().zip(words.chunks_exact(2)) {
                let a = Self::temper(pair[0]) >> 5;
                let b = Self::temper(pair[1]) >> 6;
                *value = (a as f64 * 67108864.0 + b as f64) * (1.0 / 9007199254740992.0);
            }
            self.index += count * 2;
            out = &mut out[count..];
        }
    }

    /// `random.gauss(0, sigma)`, including the cached second variate.
    pub(crate) fn gauss(&mut self, sigma: f64) -> f64 {
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
    #[cfg(test)]
    pub(crate) fn standard_normals(&mut self, count: usize) -> Vec<f64> {
        let mut out = Vec::new();
        self.standard_normals_into(count, &mut out);
        out
    }

    /// Reuse storage without clearing the elements that will be overwritten.
    ///
    /// The generator is sequential, but the Box-Muller transform of each
    /// filled block runs on the thread pool while the next block is drawn, so
    /// the transcendental work overlaps the draws instead of following them.
    /// Every element sees the same operations as before: `fill_uniforms` on
    /// consecutive blocks equals one call over the whole slice, and each pair
    /// is transformed on its own.
    pub(crate) fn standard_normals_into(&mut self, count: usize, out: &mut Vec<f64>) {
        let mut profile = crate::training::training_profile::Profile::new("normal_draws");
        let cached = if count > 0 { self.gauss_next.take() } else { None };
        // Store each pair's uniforms in place, then transform the pairs in parallel.
        let offset = usize::from(cached.is_some());
        let pairs = (count - offset).div_ceil(2);
        out.resize(offset + pairs * 2, 0.0);
        if let Some(z) = cached {
            out[0] = z;
        }
        const BLOCK: usize = 1 << 16; // doubles per block (even), about a millisecond of transform
        let body = &mut out[offset..];
        rayon::scope(|scope| {
            let mut rest = body;
            while !rest.is_empty() {
                let n = rest.len().min(BLOCK);
                let (block, tail) = std::mem::take(&mut rest).split_at_mut(n);
                self.fill_uniforms(block);
                scope.spawn(move |_| Self::transform_pairs(block));
                rest = tail;
            }
        });
        if out.len() > count {
            self.gauss_next = out.pop();
        }
        profile.mark("draws_and_transform");
    }

    /// Box-Muller on stored uniform pairs, in place (`random.gauss`).
    fn transform_pairs(block: &mut [f64]) {
        for pair in block.chunks_exact_mut(2) {
            let x2pi = pair[0] * std::f64::consts::TAU;
            let g2rad = (-2.0 * (1.0 - pair[1]).ln()).sqrt();
            pair[0] = x2pi.cos() * g2rad;
            pair[1] = x2pi.sin() * g2rad;
        }
    }

    /// `random.getrandbits(k)` for `1 <= k <= 32`.
    fn getrandbits(&mut self, k: u32) -> u32 {
        self.next_u32() >> (32 - k)
    }

    /// `random.randrange(n)` for `1 <= n < 2**32`.
    pub(crate) fn randrange(&mut self, n: usize) -> usize {
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

    #[test]
    fn bulk_uniforms_preserve_every_twist_boundary_and_checkpoint() {
        for seed in [0, 11, -3, (1 << 40) + 5] {
            for index in 0..=N {
                let mut expected = PyRandom::new(seed);
                expected.gauss(1.0); // Keep a cached normal across uniform draws.
                for _ in 0..index {
                    expected.next_u32();
                }
                let mut actual = expected.clone();
                for count in [0, 1, 2, 311, 312, 313, 625] {
                    let mut values = vec![0.0; count];
                    actual.fill_uniforms(&mut values);
                    for value in values {
                        assert_eq!(value.to_bits(), expected.random().to_bits());
                    }
                    assert_eq!(actual.to_json(), expected.to_json());
                }
            }
        }
    }

    #[test]
    fn reused_normal_storage_preserves_cached_values_and_rng_state() {
        let mut expected = PyRandom::new(123);
        let mut actual = expected.clone();
        let mut storage = vec![f64::NAN; 20000];
        // Counts around the pipelining block size (65536 doubles) and beyond several blocks.
        for count in [
            0, 1, 0, 7, 623, 624, 625, 10001, 2, 0, 3, 65535, 65536, 65537, 131073, 300001,
        ] {
            actual.standard_normals_into(count, &mut storage);
            assert_eq!(storage.len(), count);
            for &value in &storage {
                assert_eq!((0.0 + value).to_bits(), expected.gauss(1.0).to_bits());
            }
            assert_eq!(actual.to_json(), expected.to_json());
            assert_eq!(actual.randrange(57), expected.randrange(57));
        }
    }
}

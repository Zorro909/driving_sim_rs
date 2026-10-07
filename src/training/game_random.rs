//! The two independent random streams used by the shipped .NET 8 game.
//! Exact replay requires both states: the game seeds them independently.
use crate::math::profile::MathProfile;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct GameRandom {
    pub(crate) decision_state: [u64; 4],
    seed_array: [i32; 56],
    inext: usize,
    inextp: usize,
}
impl GameRandom {
    pub fn new(decision_state: [u64; 4], seed: i32) -> Self {
        assert!(decision_state.iter().any(|&x| x != 0));
        let mut seed_array = [0; 56];
        let subtraction = if seed == i32::MIN { i32::MAX } else { seed.abs() };
        let mut mj = 161803398 - subtraction;
        seed_array[55] = mj;
        let mut mk = 1;
        for i in 1..55 {
            let ii = 21 * i % 55;
            seed_array[ii] = mk;
            mk = mj.wrapping_sub(mk);
            if mk < 0 {
                mk += i32::MAX;
            }
            mj = seed_array[ii];
        }
        for _ in 0..4 {
            for i in 1..56 {
                seed_array[i] = seed_array[i].wrapping_sub(seed_array[1 + (i + 30) % 55]);
                if seed_array[i] < 0 {
                    seed_array[i] += i32::MAX;
                }
            }
        }
        Self {
            decision_state,
            seed_array,
            inext: 0,
            inextp: 21,
        }
    }
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.decision_state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * 1.1102230246251565e-16
    }
    pub fn randrange(&mut self, bound: usize) -> usize {
        assert!(bound <= i32::MAX as usize);
        let bound = bound as u32;
        let mut product = bound as u64 * (self.next_u64() >> 32);
        if (product as u32) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (product as u32) < threshold {
                product = bound as u64 * (self.next_u64() >> 32);
            }
        }
        (product >> 32) as usize
    }
    pub(crate) fn normal_uniform(&mut self) -> f64 {
        self.inext += 1;
        if self.inext >= 56 {
            self.inext = 1;
        }
        self.inextp += 1;
        if self.inextp >= 56 {
            self.inextp = 1;
        }
        let mut value = self.seed_array[self.inext] - self.seed_array[self.inextp];
        if value == i32::MAX {
            value -= 1;
        }
        if value < 0 {
            value += i32::MAX;
        }
        self.seed_array[self.inext] = value;
        value as f64 * (1.0 / i32::MAX as f64)
    }
    /// .NET's normal draws, whose Math.Log is the C runtime's `math.log`.
    pub fn normal_array(&mut self, count: usize, sigma: f64, math: MathProfile) -> Vec<f64> {
        if count == 0 {
            return Vec::new();
        }
        let mut prefetched = ((count * 4) as f64 * (1.0 / std::f64::consts::PI)).ceil() as usize;
        prefetched += prefetched % 2;
        let uniforms: Vec<_> = (0..prefetched).map(|_| self.normal_uniform()).collect();
        let mut result = Vec::with_capacity(count);
        let mut index = 0;
        while result.len() < count {
            let (a, b) = if index < prefetched {
                let pair = (uniforms[index], uniforms[index + 1]);
                index += 2;
                pair
            } else {
                (self.normal_uniform(), self.normal_uniform())
            };
            let u = 2.0 * a - 1.0;
            let v = 2.0 * b - 1.0;
            let s = u * u + v * v;
            if s >= 1.0 || s == 0.0 {
                continue;
            }
            let factor = ((-2.0 * math.log(s)) / s).sqrt();
            result.push(0.0 + sigma * (u * factor));
            if result.len() < count {
                result.push(0.0 + sigma * (v * factor));
            }
        }
        result
    }
    pub fn normal_state(&self) -> Value {
        json!({"algorithm":"dotnet_compat","seed_array":self.seed_array.as_slice(),"inext":self.inext,"inextp":self.inextp})
    }
    pub fn decision_json(&self) -> Value {
        json!({"algorithm":"xoshiro256starstar","state":self.decision_state})
    }
    pub fn to_json(&self) -> Value {
        json!({"backend":"game","decisions":self.decision_json(),"normals":self.normal_state()})
    }
    pub fn from_json(v: &Value) -> Self {
        Self::try_from_json(v).expect("valid game random state")
    }
    pub fn try_from_json(v: &Value) -> Result<Self, String> {
        let mut s = Self::new([1, 2, 3, 4], 0);
        let decisions = v["decisions"]["state"]
            .as_array()
            .filter(|a| a.len() == 4)
            .ok_or("invalid decision state length")?;
        for (i, x) in decisions.iter().enumerate() {
            s.decision_state[i] = x.as_u64().ok_or("invalid decision state word")?;
        }
        if s.decision_state.iter().all(|n| *n == 0) {
            return Err("decision state must not be all zero".into());
        }
        let normals = v["normals"]["seed_array"]
            .as_array()
            .filter(|a| a.len() == 56)
            .ok_or("invalid normal state length")?;
        for (i, x) in normals.iter().enumerate() {
            s.seed_array[i] = x
                .as_i64()
                .filter(|n| *n >= 0 && *n < i32::MAX as i64)
                .ok_or("invalid normal state word")? as i32;
        }
        if s.seed_array[1..].iter().all(|n| *n == 0) {
            return Err("normal state must not be all zero".into());
        }
        s.inext = v["normals"]["inext"]
            .as_u64()
            .filter(|n| *n < 56)
            .ok_or("invalid normal index")? as usize;
        s.inextp = v["normals"]["inextp"]
            .as_u64()
            .filter(|n| *n < 56)
            .ok_or("invalid normal index")? as usize;
        if (s.inextp + 55 - s.inext) % 55 != 21 {
            return Err("invalid normal index separation".into());
        }
        Ok(s)
    }
}

#[cfg(test)]
mod input_tests {
    use super::*;
    #[test]
    fn validated_game_states_round_trip_and_reject_bad_indices() {
        let mut source = GameRandom::new([1, 2, 3, 4], 19);
        for _ in 0..70 {
            source.normal_uniform();
        }
        let valid = source.to_json();
        let restored = GameRandom::try_from_json(&valid).unwrap();
        assert_eq!(restored.to_json(), valid);
        for field in ["inext", "inextp"] {
            let mut bad = valid.clone();
            bad["normals"][field] = serde_json::json!(56);
            assert!(GameRandom::try_from_json(&bad).is_err());
        }
        let mut bad = valid.clone();
        bad["decisions"]["state"] = serde_json::json!([0, 0, 0, 0]);
        assert!(GameRandom::try_from_json(&bad).is_err());
        bad = valid.clone();
        bad["normals"]["inextp"] = bad["normals"]["inext"].clone();
        assert!(GameRandom::try_from_json(&bad).is_err());
        bad = valid;
        bad["normals"]["seed_array"][1] = serde_json::json!(-1);
        assert!(GameRandom::try_from_json(&bad).is_err());
    }
}

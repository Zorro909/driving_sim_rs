//! `driving_sim.sim.Vec2` with the same operation order as the Python class.

use crate::pymath::{clamp, hypot};
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V2 {
    pub x: f64,
    pub y: f64,
}

impl V2 {
    pub const ZERO: V2 = V2 { x: 0.0, y: 0.0 };

    #[inline(always)]
    pub const fn new(x: f64, y: f64) -> V2 {
        V2 { x, y }
    }

    #[inline(always)]
    pub fn dot(self, o: V2) -> f64 {
        self.x * o.x + self.y * o.y
    }

    #[inline(always)]
    pub fn cross(self, o: V2) -> f64 {
        self.x * o.y - self.y * o.x
    }

    #[inline(always)]
    pub fn length(self) -> f64 {
        hypot(self.x, self.y)
    }

    /// Python `Vec2.__truediv__` multiplies by the reciprocal.
    #[inline(always)]
    pub fn div(self, value: f64) -> V2 {
        self * (1.0 / value)
    }

    #[inline(always)]
    pub fn normalized(self) -> V2 {
        let length = self.length();
        if length != 0.0 {
            self.div(length)
        } else {
            V2::ZERO
        }
    }

    #[inline(always)]
    pub fn rotated(self, angle: f64) -> V2 {
        let (c, s) = (angle.cos(), angle.sin());
        V2::new(c * self.x - s * self.y, s * self.x + c * self.y)
    }

    #[inline(always)]
    pub fn limit(self, maximum: f64) -> V2 {
        let length = self.length();
        if length > maximum {
            self * (maximum / length)
        } else {
            self
        }
    }

    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.x == 0.0 && self.y == 0.0
    }
}

impl Add for V2 {
    type Output = V2;
    #[inline(always)]
    fn add(self, o: V2) -> V2 {
        V2::new(self.x + o.x, self.y + o.y)
    }
}

impl Sub for V2 {
    type Output = V2;
    #[inline(always)]
    fn sub(self, o: V2) -> V2 {
        V2::new(self.x - o.x, self.y - o.y)
    }
}

impl Mul<f64> for V2 {
    type Output = V2;
    #[inline(always)]
    fn mul(self, value: f64) -> V2 {
        V2::new(self.x * value, self.y * value)
    }
}

/// Python negation is `self * -1.0`.
impl Neg for V2 {
    type Output = V2;
    #[inline(always)]
    fn neg(self) -> V2 {
        self * -1.0
    }
}

impl AddAssign for V2 {
    #[inline(always)]
    fn add_assign(&mut self, o: V2) {
        *self = *self + o;
    }
}

impl SubAssign for V2 {
    #[inline(always)]
    fn sub_assign(&mut self, o: V2) {
        *self = *self - o;
    }
}

/// `driving_sim.sim.closest_point`.
#[inline]
pub fn closest_point(point: V2, start: V2, end: V2) -> V2 {
    let delta = end - start;
    let length_sq = delta.dot(delta);
    let weight = if length_sq != 0.0 {
        clamp((point - start).dot(delta) / length_sq, 0.0, 1.0)
    } else {
        0.0
    };
    start + delta * weight
}

/// `driving_sim.sim.segment_intersection`: finite crossing including endpoints.
#[inline]
pub fn segment_intersection(a_start: V2, a_end: V2, b_start: V2, b_end: V2) -> Option<V2> {
    let r = a_end - a_start;
    let s = b_end - b_start;
    let divisor = r.cross(s);
    if divisor.abs() < 1e-8 {
        return None;
    }
    let t = (b_start - a_start).cross(s) / divisor;
    let u = (b_start - a_start).cross(r) / divisor;
    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        Some(a_start + r * t)
    } else {
        None
    }
}

//! Float64 vectors with the reference simulator's operation order.

use crate::math::pymath::{clamp, hypot};
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
    pub(crate) fn dot(self, o: V2) -> f64 {
        self.x * o.x + self.y * o.y
    }

    #[inline(always)]
    pub(crate) fn cross(self, o: V2) -> f64 {
        self.x * o.y - self.y * o.x
    }

    #[inline(always)]
    pub fn length(self) -> f64 {
        hypot(self.x, self.y)
    }

    /// Python `Vec2.__truediv__` multiplies by the reciprocal.
    #[inline(always)]
    // The named operation preserves reciprocal multiplication in the public interface.
    #[allow(clippy::should_implement_trait)]
    pub(crate) fn div(self, value: f64) -> V2 {
        self * (1.0 / value)
    }

    #[inline(always)]
    pub(crate) fn normalized(self) -> V2 {
        let length = self.length();
        if length != 0.0 {
            self.div(length)
        } else {
            V2::ZERO
        }
    }

    #[inline(always)]
    pub(crate) fn rotated(self, angle: f64) -> V2 {
        let (c, s) = (angle.cos(), angle.sin());
        V2::new(c * self.x - s * self.y, s * self.x + c * self.y)
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

/// Closest point on a segment, with a clamped projection weight.
#[inline]
pub(crate) fn closest_point(point: V2, start: V2, end: V2) -> V2 {
    let delta = end - start;
    let length_sq = delta.dot(delta);
    let weight = if length_sq != 0.0 {
        clamp((point - start).dot(delta) / length_sq, 0.0, 1.0)
    } else {
        0.0
    };
    start + delta * weight
}

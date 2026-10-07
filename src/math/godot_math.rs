//! Float32 operations at the Godot Vector2/Transform2D boundary.
//! Networks and the game's explicitly double-valued scalars keep using f64.
use crate::math::vec2::V2;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default)]
pub struct F2 {
    pub x: f32,
    pub y: f32,
}
impl From<V2> for F2 {
    fn from(v: V2) -> Self {
        Self {
            x: v.x as f32,
            y: v.y as f32,
        }
    }
}
impl From<F2> for V2 {
    fn from(v: F2) -> Self {
        Self::new(v.x as f64, v.y as f64)
    }
}
impl Add for F2 {
    type Output = Self;
    fn add(self, v: Self) -> Self {
        Self {
            x: self.x + v.x,
            y: self.y + v.y,
        }
    }
}
impl Sub for F2 {
    type Output = Self;
    fn sub(self, v: Self) -> Self {
        Self {
            x: self.x - v.x,
            y: self.y - v.y,
        }
    }
}
impl Mul<f32> for F2 {
    type Output = Self;
    fn mul(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }
}
impl Neg for F2 {
    type Output = Self;
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y }
    }
}
impl F2 {
    pub fn dot(self, v: Self) -> f32 {
        self.x * v.x + self.y * v.y
    }
    pub(crate) fn cross(self, v: Self) -> f32 {
        self.x * v.y - self.y * v.x
    }
    pub(crate) fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn normalized(self) -> Self {
        let n = self.length();
        if n == 0.0 {
            Self::default()
        } else {
            Self {
                x: self.x / n,
                y: self.y / n,
            }
        }
    }
    pub(crate) fn limit(self, max: f32) -> Self {
        let n = self.length();
        if n > max && n > 0.0 {
            // Godot C# Vector2.LimitLength divides before multiplying.
            Self {
                x: self.x / n,
                y: self.y / n,
            } * max
        } else {
            self
        }
    }
    pub(crate) fn rotated(self, angle: f32) -> Self {
        self.rotated_sin_cos(managed_sin_cos(angle))
    }
    /// `rotated` by the angle whose `managed_sin_cos` is `(s, c)`.
    #[inline]
    pub(crate) fn rotated_sin_cos(self, (s, c): (f32, f32)) -> Self {
        Self {
            x: self.x * c - self.y * s,
            y: self.x * s + self.y * c,
        }
    }
}

/// Windows .NET 8.0.2 MathF.SinCos, as used by the game's managed transforms.
/// COMSingle::SinCos uses the compiler's packed sine routine, including a
/// rounded pi/2 shift for cosine. Native Godot transforms use separate sin/cos.
/// Constants and operation order are from the shipped coreclr.dll, RVA 169f90
/// and 16b620. No fused multiply-adds; large lanes use the original tables.
// Negated float comparisons reproduce the managed branch behavior, including NaNs.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn managed_sin_cos(angle: f32) -> (f32, f32) {
    // COMISS takes the wrapper's tiny branch for NaNs. Sine retains the input
    // payload; cosine quiets the absolute-value payload through multiplication.
    if angle.is_nan() {
        return (angle, f32::from_bits((angle.to_bits() & 0x7fffffff) | 0x00400000));
    }
    if !(angle.abs() >= f32::from_bits(0x39000000)) {
        return (angle, 1.0 - angle.abs() * angle.abs() * 0.5);
    }
    fn packed_sine(value: f32) -> f32 {
        let abs = value.abs();
        if !(abs <= 10000.0) {
            return crate::math::managed_trig::sine_large(value);
        }
        let rounded = abs * f32::from_bits(0x3ea2f983) + f32::from_bits(0x4b400000);
        let n = rounded - f32::from_bits(0x4b400000);
        let mut r = abs - n * f32::from_bits(0x40490000);
        r -= n * f32::from_bits(0x3a7da000);
        r -= n * f32::from_bits(0x34222000);
        r -= n * f32::from_bits(0x2cb4611a);
        let square = r * r;
        r = f32::from_bits(r.to_bits() ^ (rounded.to_bits() << 31));
        let mut p = f32::from_bits(0x362edef8) * square;
        p += f32::from_bits(0xb94fb7ff);
        p *= square;
        p += f32::from_bits(0x3c088766);
        p *= square;
        p += f32::from_bits(0xbe2aaaa6);
        let result = r + (square * p) * r;
        f32::from_bits(result.to_bits() ^ (value.to_bits() & 0x80000000))
    }
    (
        packed_sine(angle),
        packed_sine(angle.abs() + std::f32::consts::FRAC_PI_2),
    )
}

/// `managed_sin_cos(-PI / 2)`, the quarter turn from a body's right to its front.
pub(crate) const QUARTER_TURN_BACK: (f32, f32) = (f32::from_bits(0xbf7f_ffff), f32::from_bits(0xb3bb_bd2e));

pub fn basis(rotation: f64) -> (F2, F2) {
    let (s, c) = crate::math::native_math::engine_sin_cos(rotation as f32);
    (F2 { x: c, y: s }, F2 { x: -s, y: c })
}

pub(crate) fn transform_point(position: V2, (x, y): (F2, F2), local: V2) -> V2 {
    let p = F2::from(local);
    // Transform2D operator * evaluates the two products before the origin.
    (x * p.x + y * p.y + F2::from(position)).into()
}

pub(crate) fn combine(brake: F2, drive: F2, brake_max: f32, drive_max: f32) -> F2 {
    let n = brake.length();
    if n < 0.0001 || brake_max <= 0.0 {
        return brake + drive;
    }
    let allowed = n + drive_max * (1.0 - (n / brake_max).min(1.0));
    let result = brake + drive;
    let axis = brake.normalized();
    let projection = result.dot(axis);
    if projection > allowed {
        result - axis * (projection - allowed)
    } else {
        result
    }
}

/// C# Mathf.Remap(value, -maximum, maximum, -1f, 1f), then Clamp.
/// Keeping the InverseLerp/Lerp operations preserves their float rounding.
pub(crate) fn signed_sensor(value: f32, maximum: f32) -> f64 {
    let from = 0.0f32 - maximum;
    let weight = (value - from) / (maximum - from);
    (-1.0f32 + 2.0f32 * weight).clamp(-1.0, 1.0) as f64
}

/// BspTreeRaycaster.TryGetIntersection(wall, ray). The scene export already
/// contains BSP-split wall segments. The returned point lies on the wall.
pub(crate) fn ray_intersection(start: V2, end: V2, wall_start: V2, wall_end: V2) -> Option<V2> {
    let (start, end) = (F2::from(start), F2::from(end));
    let (ws, we) = (F2::from(wall_start), F2::from(wall_end));
    ray_wall_hit(start, end - start, ws, we - ws).map(V2::from)
}

/// `ray_intersection` with the ray `start` and `b = end - start` and the wall
/// `ws` and `a = we - ws` already in float32.
#[inline]
pub(crate) fn ray_wall_hit(start: F2, b: F2, ws: F2, a: F2) -> Option<F2> {
    let denominator = a.x * b.y - b.x * a.y;
    if denominator.abs() < 1e-6 {
        return None;
    }
    let delta = ws - start;
    let ray_fraction = ((0.0 - a.y) * delta.x + a.x * delta.y) / denominator;
    let wall_fraction = (b.x * delta.y - b.y * delta.x) / denominator;
    if !(0.0..=1.0).contains(&ray_fraction) || !(0.0..=1.0).contains(&wall_fraction) {
        return None;
    }
    Some(ws + a * wall_fraction)
}

/// Vehicle.Reset constructs a managed transform; BodySetState orthonormalizes it.
pub(crate) fn reset_basis(angle: f32) -> (F2, F2) {
    let (s, c) = managed_sin_cos(angle);
    let x = F2 { x: c, y: s }.normalized();
    let y = F2 { x: -s, y: c };
    (x, (y - x * x.dot(y)).normalized())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_turn_back_is_managed_sin_cos() {
        let (s, c) = managed_sin_cos(-std::f32::consts::FRAC_PI_2);
        assert_eq!(
            (s.to_bits(), c.to_bits()),
            (QUARTER_TURN_BACK.0.to_bits(), QUARTER_TURN_BACK.1.to_bits())
        );
        assert_eq!(std::f32::consts::FRAC_PI_2, std::f32::consts::PI / 2.0);
    }
}

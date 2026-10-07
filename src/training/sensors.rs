//! Ordered game sensor descriptors and input layout.

use crate::math::profile::MathProfile;
use crate::math::vec2::V2;
use crate::physics::car::{Car, Sensor, SensorScratch};
use crate::track::world::World;
use serde_json::Value;

/// Map compact model input names to game sensor types.
pub(crate) fn sensor_type(name: &str) -> Option<&'static str> {
    Some(match name {
        "AccF" => "accelerationFront",
        "AccS" => "accelerationSide",
        "Wall" => "distanceFromWall",
        "AngVel" => "angularVelocity",
        "Boost" => "boostCapacity",
        "Dir" => "correctDirection",
        "Grip" => "grip",
        "VelF" => "velocityFront",
        "VelS" => "velocitySide",
        "Curve" => "trackCurvature",
        "Speed" | "VelAbs" => "speed",
        "Wheel" => "wheelAngle",
        _ => return None,
    })
}

/// Parse `"↑ -35°"` into its angle.
pub(crate) fn vision_angle(name: &str) -> Option<f64> {
    name.strip_prefix("↑ ")?.strip_suffix('°')?.trim().parse().ok()
}

/// `SensorsEditor.CalculateSensorLength`: the editor gives every vision ray
/// `200 + 600 * |cos(degrees)|` pixels, in float32 with the game's UCRT cosf.
pub fn vision_length(degrees: f32, math: MathProfile) -> f32 {
    200.0 + 600.0 * math.cos(degrees * (std::f32::consts::PI / 180.0)).abs()
}

#[derive(Clone, Debug)]
pub struct SensorLayout {
    pub names: Vec<String>,
    pub sensors: Vec<Sensor>,
}

impl SensorLayout {
    /// Exact ordered descriptors captured from instantiated game sensors.
    /// Names may repeat; each descriptor retains its own float parameters.
    pub fn from_ordered(value: &Value) -> Result<SensorLayout, String> {
        let rows = value["sensors"].as_array().ok_or("missing ordered sensors")?;
        let names: Vec<String> = value["names"]
            .as_array()
            .ok_or("missing ordered names")?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "invalid sensor name".to_owned())
            })
            .collect::<Result<_, _>>()?;
        if names.len() != rows.len() {
            return Err("sensor/name count differs".into());
        }
        let mut sensors = Vec::with_capacity(rows.len());
        for row in rows {
            let number = |key: &str| {
                row[key]
                    .as_f64()
                    .filter(|x| (*x as f32).is_finite())
                    .map(|x| x as f32 as f64)
                    .ok_or_else(|| format!("missing sensor parameter {key}"))
            };
            let kind = row["$type"].as_str().ok_or("missing sensor type")?;
            sensors.push(match kind {
                "raycast" => Sensor::Raycast {
                    degrees: number("Degrees")?,
                    length: number("Length")?,
                },
                "accelerationFront" => Sensor::AccelerationFront {
                    max_acceleration: number("MaxAcceleration")?,
                },
                "accelerationSide" => Sensor::AccelerationSide {
                    max_acceleration: number("MaxAcceleration")?,
                },
                "distanceFromWall" => Sensor::DistanceFromWall {
                    max_distance: number("MaxDistance")?,
                },
                "trackCurvature" => {
                    let min = number("MinLookaheadDistance")?.max(0.0);
                    Sensor::TrackCurvature {
                        min_lookahead: min,
                        max_lookahead: number("MaxLookaheadDistance")?.max(min),
                    }
                }
                "grip" => {
                    let offset = &row["PositionOffset"];
                    let (x, y) = if offset.is_array() {
                        (&offset[0], &offset[1])
                    } else {
                        (&offset["X"], &offset["Y"])
                    };
                    let component = |v: &Value| {
                        v.as_f64()
                            .filter(|x| (*x as f32).is_finite())
                            .map(|x| x as f32 as f64)
                            .ok_or("missing grip offset component")
                    };
                    Sensor::Grip {
                        offset: V2::new(component(x)?, component(y)?),
                    }
                }
                "speed" | "velocityFront" | "velocitySide" | "angularVelocity" | "wheelAngle" | "boostCapacity"
                | "correctDirection" => Sensor::by_name(kind),
                _ => return Err(format!("unknown sensor type {kind}")),
            });
        }
        Ok(SensorLayout { names, sensors })
    }

    /// `SensorLayout.from_exports(live_network, model)`. Vision rays the model
    /// does not list get the editor's length under `math`.
    pub fn from_exports(live_network: &Value, model: &Value, math: MathProfile) -> SensorLayout {
        Self::try_from_exports(live_network, model, math).expect("valid sensor exports")
    }

    pub fn try_from_exports(live_network: &Value, model: &Value, math: MathProfile) -> Result<SensorLayout, String> {
        if let Some(exact) = model.get("sensor_layout") {
            let layout = Self::from_ordered(exact)?;
            let inputs: Vec<_> = live_network["inputs"]
                .as_array()
                .ok_or("missing network inputs")?
                .iter()
                .map(|v| v.as_str().ok_or("invalid input name"))
                .collect::<Result<_, _>>()?;
            if !layout.names.iter().map(String::as_str).eq(inputs) {
                return Err("ordered sensor inputs differ from network".into());
            }
            return Ok(layout);
        }
        let vision: Vec<(f64, f64)> = model["vision"]
            .as_array()
            .ok_or("missing model vision")?
            .iter()
            .map(|item| {
                Ok((
                    item["angle"]
                        .as_f64()
                        .filter(|n| (*n as f32).is_finite())
                        .ok_or("invalid vision angle")?,
                    item["length"]
                        .as_f64()
                        .filter(|n| (*n as f32).is_finite())
                        .ok_or("invalid vision length")?,
                ))
            })
            .collect::<Result<_, String>>()?;
        let names: Vec<String> = live_network["inputs"]
            .as_array()
            .ok_or("missing network inputs")?
            .iter()
            .map(|v| v.as_str().map(str::to_owned).ok_or("invalid input name"))
            .collect::<Result<_, _>>()?;
        let sensors = names
            .iter()
            .map(|name| match vision_angle(name) {
                Some(angle) => {
                    if !(angle as f32).is_finite() {
                        return Err("invalid vision angle".into());
                    }
                    // Python dict: the last vision entry with this angle wins.
                    // Angles the model does not list get the editor's length.
                    let length = vision
                        .iter()
                        .rev()
                        .find(|(a, _)| *a == angle)
                        .map_or_else(|| vision_length(angle as f32, math) as f64, |&(_, length)| length);
                    Ok(Sensor::Raycast { degrees: angle, length })
                }
                None => sensor_type(name)
                    .map(Sensor::by_name)
                    .ok_or_else(|| format!("unknown sensor: {name}")),
            })
            .collect::<Result<_, _>>()?;
        Ok(SensorLayout { names, sensors })
    }

    pub fn read_into(&self, world: &World, car: &Car, scratch: &mut SensorScratch, out: &mut Vec<f64>) {
        scratch.invalidate();
        out.clear();
        out.extend(self.sensors.iter().map(|&sensor| car.sensor(world, sensor, scratch)));
    }

    /// The number of `Sensor::Raycast` entries.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn ray_count(&self) -> usize {
        self.sensors
            .iter()
            .filter(|s| matches!(s, Sensor::Raycast { .. }))
            .count()
    }

    /// `read_into` with the ray sensors' track raycasts supplied by an external
    /// raycaster: one `[hit.x, hit.y, hit != 0, _]` per `Sensor::Raycast`, in
    /// layout order, for the ray `TrainingRunner::ray_queries` produced.
    #[cfg(any(test, all(target_arch = "wasm32", feature = "wasm")))]
    pub(crate) fn read_into_with_hits(
        &self,
        world: &World,
        car: &Car,
        scratch: &mut SensorScratch,
        out: &mut Vec<f64>,
        hits: &[[f32; 4]],
    ) {
        scratch.invalidate();
        out.clear();
        let mut hits = hits.iter();
        out.extend(self.sensors.iter().map(|&sensor| match sensor {
            Sensor::Raycast { degrees, length } => {
                let (end, length) = car.ray_end(degrees, length);
                let h = hits.next().expect("one hit per ray sensor");
                let hit = (h[2] != 0.0).then(|| V2::from(crate::math::godot_math::F2 { x: h[0], y: h[1] }));
                Car::ray_value(car.position, end, length, hit)
            }
            other => car.sensor(world, other, scratch),
        }));
    }
}

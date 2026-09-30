//! Experimental Godot body, contact, and sensor arithmetic.
//! See ../ROUND2.md for measured accuracy and remaining limitations.

use crate::godot_math::{self, F2};
use crate::collision::{solve_wall_contacts, CollisionScratch, Contact};
use crate::pymath::{f32r, clamp, py_max, py_min, py_mod};
use crate::vec2::V2;
use crate::world::{RayStamps, Surface, Track, World, ASPHALT};
use std::f64::consts::PI;

pub const DT: f64 = 1.0 / 60.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Controls {
    pub acceleration: f64,
    pub steering: f64,
    pub brake: f64,
    pub handbrake: f64,
    pub boost: f64,
}

impl Controls {
    pub fn normalized(&self) -> Controls {
        Controls {
            acceleration: clamp(self.acceleration, -1.0, 1.0),
            steering: clamp(self.steering, -1.0, 1.0),
            brake: clamp(self.brake, -1.0, 1.0),
            handbrake: clamp(self.handbrake, 0.0, 1.0),
            boost: clamp(self.boost, 0.0, 1.0),
        }
    }

    /// `Controls(**{name.lower(): value})`; panics on unknown names like Python.
    pub fn set(&mut self, name: &str, value: f64) {
        match name.to_lowercase().as_str() {
            "acceleration" => self.acceleration = value,
            "steering" => self.steering = value,
            "brake" => self.brake = value,
            "handbrake" => self.handbrake = value,
            "boost" => self.boost = value,
            other => panic!("unknown control output: {other}"),
        }
    }

    pub fn get(&self, name: &str) -> f64 {
        match name.to_lowercase().as_str() {
            "acceleration" => self.acceleration,
            "steering" => self.steering,
            "brake" => self.brake,
            "handbrake" => self.handbrake,
            "boost" => self.boost,
            other => panic!("unknown control output: {other}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WheelState {
    pub angle_deg: f64,
    pub previous_position: Option<V2>,
    pub previous_surface: Surface,
}

impl Default for WheelState {
    fn default() -> Self {
        WheelState { angle_deg: 0.0, previous_position: None, previous_surface: ASPHALT }
    }
}

#[derive(Clone)]
struct WheelInitialization {
    grip: f64,
    steering_speed: f64,
    center_steering: bool,
    positions: Vec<V2>,
}

/// Mutable state of one car. Static data lives in the shared `World`.
#[derive(Clone)]
pub struct Car {
    pub position: V2,
    pub rotation: f64,
    /// Angle that produced the stored body basis; may differ from atan2(basis).
    pub transform_angle: f64,
    pub body_basis: (F2, F2),
    pub velocity: V2,
    pub angular_velocity: f64,
    /// Game value: velocity difference per tick.
    pub acceleration: V2,
    pub boost_energy: f64,
    pub wheels: Vec<WheelState>,
    wheel_initialization: WheelInitialization,
    pub active: bool,
    pub frozen: bool,
    frozen_velocity: (V2, f64),
    pending_velocity_store: bool,
    pending_velocity_reset: Option<(V2, f64)>,
    reset_acceleration: V2,
    reset_right: bool,
    pending_transform_reset: Option<(V2, f64)>,
    pending_callback: Option<NativeCallback>,
    pub collision_count: u64,
    pub wall_contacts: Vec<Contact>,
    previous_contacts: Vec<Contact>,
    scratch: CollisionScratch,
    pub tick: u64,
}

#[derive(Clone, Copy)]
struct NativeCallback {
    old_velocity: F2,
    collided: bool,
    was_active: bool,
    eliminate_on_wall: bool,
}

pub fn godot_ease(value: f64, curve: f64) -> f64 {
    let value = clamp(value, 0.0, 1.0);
    1.0 - crate::double_math::pow(1.0 - value, 1.0 / curve)
}

pub fn combine_brake_and_drive(braking: V2, drive: V2, brake_max: f64, drive_max: f64) -> V2 {
    let magnitude = braking.length();
    if magnitude < 0.0001 || brake_max <= 0.0 {
        return braking + drive;
    }
    let allowed_projection = magnitude + drive_max * (1.0 - py_min(1.0, magnitude / brake_max));
    let result = braking + drive;
    let axis = braking.normalized();
    let excess = result.dot(axis) - allowed_projection;
    if excess > 0.0 {
        result - axis * excess
    } else {
        result
    }
}

/// Sensor kinds of `Simulator.sensor`, with their keyword arguments resolved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sensor {
    Raycast { degrees: f64, length: f64 },
    DistanceFromWall { max_distance: f64 },
    Speed,
    VelocityFront,
    VelocitySide,
    AccelerationFront { max_acceleration: f64 },
    AccelerationSide { max_acceleration: f64 },
    AngularVelocity,
    WheelAngle,
    BoostCapacity,
    Grip { offset: V2 },
    CorrectDirection,
    TrackCurvature { min_lookahead: f64, max_lookahead: f64 },
}

impl Sensor {
    /// Sensor by `Simulator.sensor` name with default keyword arguments.
    pub fn by_name(name: &str) -> Sensor {
        match name {
            "raycast" => Sensor::Raycast { degrees: 0.0, length: 300.0 },
            "distanceFromWall" => Sensor::DistanceFromWall { max_distance: 150.0 },
            "speed" => Sensor::Speed,
            "velocityFront" => Sensor::VelocityFront,
            "velocitySide" => Sensor::VelocitySide,
            "accelerationFront" => Sensor::AccelerationFront { max_acceleration: 10.0 },
            "accelerationSide" => Sensor::AccelerationSide { max_acceleration: 12.0 },
            "angularVelocity" => Sensor::AngularVelocity,
            "wheelAngle" => Sensor::WheelAngle,
            "boostCapacity" => Sensor::BoostCapacity,
            "grip" => Sensor::Grip { offset: V2::ZERO },
            "correctDirection" => Sensor::CorrectDirection,
            "trackCurvature" => Sensor::TrackCurvature { min_lookahead: 256.0, max_lookahead: 512.0 },
            other => panic!("unknown sensor: {other}"),
        }
    }
}

/// Per-thread scratch for sensor reads.
#[derive(Default, Clone)]
pub struct SensorScratch {
    pub stamps: RayStamps,
    /// `(sensor point, offset)` of the last path projection for this car pose.
    path_cache: Option<(V2, f64)>,
}

impl SensorScratch {
    /// Forget cached path projections (call when the car moves).
    pub fn invalidate(&mut self) {
        self.path_cache = None;
    }
}

impl Car {
    pub(crate) fn update_native_shape(&mut self,world:&World)->crate::broadphase::Aabb {
        self.scratch.update_shape(&world.vehicle,self.position,self.body_basis);
        self.scratch.shape_bounds()
    }
    pub(crate) fn native_shape_bounds(&self)->crate::broadphase::Aabb {self.scratch.shape_bounds()}
    pub(crate) fn set_native_pairs(&mut self,pairs:&[usize],wall_first:&[bool],epochs:&[u64]) {
        self.scratch.set_external_pairs(pairs,wall_first,epochs,&mut self.previous_contacts);
    }
    /// `Simulator(config, track, position, rotation)`. The per-tick
    /// `Simulator.score` tracker does not affect training, so it is omitted.
    pub fn new(world: &World, position: V2, rotation: f64) -> Car {
        Car {
            position,
            rotation: crate::native_math::atan2(crate::native_math::engine_sin(rotation as f32), crate::native_math::engine_cos(rotation as f32)) as f64,
            transform_angle: rotation as f32 as f64,
            body_basis: godot_math::basis(rotation),
            velocity: V2::ZERO,
            angular_velocity: 0.0,
            acceleration: V2::ZERO,
            boost_energy: 1.0,
            wheels: vec![WheelState::default(); world.vehicle.wheels.len()],
            wheel_initialization: WheelInitialization {
                grip: world.vehicle.grip,
                steering_speed: world.vehicle.steering_speed,
                center_steering: world.vehicle.center_steering,
                positions: world.vehicle.wheels.iter().map(|wheel| wheel.position).collect(),
            },
            active: true,
            frozen: false,
            frozen_velocity: (V2::ZERO, 0.0),
            pending_velocity_store: false,
            pending_velocity_reset: None,
            reset_acceleration: V2::ZERO,
            reset_right: false,
            pending_transform_reset: None,
            pending_callback: None,
            collision_count: 0,
            wall_contacts: Vec::new(),
            previous_contacts: Vec::new(),
            scratch: CollisionScratch::default(),
            tick: 0,
        }
    }

    /// `Simulator.reset(position, rotation, reused_vehicle)`.
    pub fn reset(&mut self, world: &World, position: V2, rotation: f64, reused_vehicle: bool) {
        let scratch = std::mem::take(&mut self.scratch);
        let previous_contacts = std::mem::take(&mut self.previous_contacts);
        let initialized = self.wheel_initialization.clone();
        *self = Car::new(world, position, rotation);
        if reused_vehicle { self.wheel_initialization = initialized; }
        self.scratch = scratch;
        self.previous_contacts = previous_contacts;
        let (x, y) = godot_math::reset_basis(rotation as f32);
        self.set_body_basis(x, y);
        if !world.track.shapes.is_empty(){self.scratch.update_shape(&world.vehicle,self.position,self.body_basis);}
        if reused_vehicle {
            // Wheel.Reset sets _lastPosition to the car origin, unlike _Ready.
            for wheel in &mut self.wheels {
                wheel.previous_position = Some(position);
            }
        }
    }

    pub fn set_transform_angle(&mut self, angle: f64) {
        self.transform_angle = angle as f32 as f64;
        self.body_basis = godot_math::basis(self.transform_angle);
        let (x, _) = self.body_basis;
        self.rotation = crate::native_math::atan2(x.y, x.x) as f64;
    }

    /// Vehicle.SetIsActive(false) queues its reset for the next native callback.
    pub fn deactivate(&mut self) {
        self.set_active(false);
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
        self.reset_velocity(V2::ZERO, 0.0);
    }

    /// Freezing stores native velocity on the next callback, then resets it once.
    /// Native integration continues, including while scripted driving is frozen.
    pub fn set_frozen(&mut self, frozen: bool) {
        let previous = self.frozen;
        self.frozen = frozen;
        if self.active {
            if frozen && !previous {
                self.pending_velocity_store = true;
                self.reset_velocity(V2::ZERO, 0.0);
            } else if !frozen && previous {
                self.reset_velocity(self.frozen_velocity.0, self.frozen_velocity.1);
            }
        }
    }

    fn reset_velocity(&mut self, linear: V2, angular: f64) {
        self.pending_velocity_reset = Some((linear, angular));
        self.reset_acceleration = self.acceleration;
    }

    pub fn has_pending_request(&self) -> bool {
        self.pending_velocity_reset.is_some() || self.pending_transform_reset.is_some()
    }

    pub fn actual_position(&self) -> V2 {
        self.pending_transform_reset.map_or(self.position, |(position, _)| position)
    }

    pub fn actual_velocity(&self) -> V2 {
        self.pending_velocity_reset.map_or(self.velocity, |(linear, _)| linear)
    }

    pub fn actual_angular_velocity(&self) -> f64 {
        self.pending_velocity_reset.map_or(self.angular_velocity, |(_, angular)| angular)
    }

    pub fn actual_acceleration(&self) -> V2 {
        if self.pending_velocity_reset.is_some() { self.reset_acceleration } else { self.acceleration }
    }

    fn cached_right(&self) -> F2 {
        if self.reset_right { F2::default() } else { self.body_basis.0.normalized() }
    }

    /// Vehicle.GetVelocity, used for display while physics is frozen.
    pub fn display_velocity(&self) -> V2 {
        if !self.active { V2::ZERO } else if self.frozen { self.frozen_velocity.0 } else { self.velocity }
    }

    /// Vehicle.Reset updates wheel history immediately, then applies the body
    /// transform and zero velocity in the next _IntegrateForces callback.
    pub fn queue_reset(&mut self, position: V2, rotation: f64) {
        self.active = true;
        self.reset_right = true;
        self.boost_energy = 1.0;
        self.reset_velocity(V2::ZERO, 0.0);
        self.frozen_velocity = (V2::ZERO, 0.0);
        self.pending_transform_reset = Some((position, rotation));
        self.tick = 0;
        self.collision_count = 0;
        for wheel in &mut self.wheels {
            wheel.angle_deg = 0.0;
            wheel.previous_position = Some(position);
            wheel.previous_surface = ASPHALT;
        }
    }

    /// Shape indices and warm contacts belong to the track that created them.
    pub(crate) fn clear_track_contacts(&mut self) {
        self.wall_contacts.clear();
        self.previous_contacts.clear();
        self.scratch = CollisionScratch::default();
    }

    pub fn set_body_basis(&mut self, x: F2, y: F2) {
        self.body_basis = (x, y);
        self.rotation = crate::native_math::atan2(x.y, x.x) as f64;
        self.transform_angle = self.rotation;
    }

    pub fn right(&self) -> V2 {
        V2::new(1.0, 0.0).rotated(self.rotation)
    }

    pub fn forward(&self) -> V2 {
        V2::new(0.0, -1.0).rotated(self.rotation)
    }

    /// `Simulator.step(raw, dt, eliminate_on_wall)`. Returns wall contact status.
    pub fn step(&mut self, world: &World, raw: &Controls, dt: f64, eliminate_on_wall: bool) -> bool {
        self.step_internal(world, raw, dt, eliminate_on_wall, true, false)
    }

    /// Native physics continues while the generation waits, without wheel updates.
    pub fn passive_step(&mut self, world: &World, dt: f64, eliminate_on_wall: bool) -> bool {
        self.step_internal(world, &Controls::default(), dt, eliminate_on_wall, false, false)
    }

    pub(crate) fn native_integrate(&mut self, world: &World, raw: &Controls, dt: f64, eliminate_on_wall: bool, drive: bool) {
        assert!(self.pending_callback.is_none());
        self.step_internal(world, raw, dt, eliminate_on_wall, drive, true);
    }

    pub(crate) fn native_callback(&mut self, world: &World) -> bool {
        let callback=self.pending_callback.take().expect("native integration must precede callback");
        self.finish_callback(world, callback)
    }

    fn step_internal(&mut self, world: &World, raw: &Controls, dt: f64, eliminate_on_wall: bool, drive: bool, defer_callback: bool) -> bool {
        assert!(dt > 0.0, "dt must be positive");
        let was_active = self.active;
        let drive = drive && !self.frozen;
        let cfg = &world.vehicle;
        let track = &world.track;
        let control = raw.normalized();
        let mut impulse = F2::default();
        let mut torque = 0.0f32;
        let old_velocity = F2::from(self.velocity);
        if was_active && drive {
        // VehicleState.UpdateBoost stores energy and boost strength as floats.
        let boost = control.boost as f32;
        let mut energy = self.boost_energy as f32;
        let boost_strength = if boost > 0.0 && energy > 0.0 {
            energy = (energy - boost * 0.01f32).max(0.0);
            boost * 0.5f32
        } else {
            if boost <= 0.0 { energy = (energy + 0.002f32).min(1.0); }
            0.0
        };
        self.boost_energy = energy as f64;
        let drive = (control.acceleration * if control.acceleration > 0.0 {
            1.0 + boost_strength as f64
        } else { 1.0 }) as f32;
        let steering_multiplier = 1.0 / (0.002 * F2::from(self.velocity).length() as f64 + 1.0);
        let (body_x, body_y) = self.body_basis;
        for ((spec, wheel), &local_position) in cfg.wheels.iter().zip(self.wheels.iter_mut()).zip(&self.wheel_initialization.positions) {
            let position = godot_math::transform_point(self.position, self.body_basis, local_position);
            let offset = F2::from(position) - F2::from(self.position);
            let wheel_velocity = F2::from(position) - F2::from(wheel.previous_position.unwrap_or(position));
            wheel.previous_position = Some(position);
            // ProcessVariables composes body and wheel transforms, normalizes
            // their axes, then applies the previous surface's extra steering.
            let angle = wheel.angle_deg as f32 * (std::f32::consts::PI / 180.0);
            let (s, c) = godot_math::managed_sin_cos(angle);
            let mut forward = -(body_x * -s + body_y * c).normalized();
            let mut right = (body_x * c + body_y * s).normalized();
            let extra = angle * (wheel.previous_surface.steering as f32 - 1.0);
            if extra != 0.0 {
                forward = forward.rotated(extra);
                right = right.rotated(extra);
            }
            let surface = track.vehicle_surface(position, cfg);
            wheel.previous_surface = surface;
            let direction = wheel_velocity.normalized();
            let braking = (-direction * (control.handbrake as f32 * spec.handbrake_power as f32)
                + -direction * (control.brake as f32 * spec.brake_power as f32))
                .limit(spec.brake_power.max(spec.handbrake_power) as f32);
            let drive_force = forward * (drive * spec.power as f32);
            let wheel_drive = godot_math::combine(braking, drive_force,
                spec.brake_power.max(spec.handbrake_power) as f32, spec.power as f32)
                * surface.power as f32;
            impulse = impulse + wheel_drive;
            torque += offset.cross(wheel_drive);

            let lateral_speed = wheel_velocity.dot(right);
            let handbrake = if spec.handbrake_power.abs() < 0.00001 { 0.0 } else { control.handbrake };
            let lateral_grip = 0.20000000298023224
                + (0.8 + (0.1 - 0.8) * godot_ease(handbrake, 0.3))
                    / (1.0 + crate::double_math::exp((wheel_velocity.length() as f64 - 450.0) * 0.00800000037997961));
            let effective_grip = (self.wheel_initialization.grip as f32 as f64 * lateral_grip) as f32;
            let lateral = right * (-effective_grip * lateral_speed) * surface.grip as f32;
            // ImpulseAccumulator receives Drive and ApplyLateralForces separately.
            impulse = impulse + lateral;
            torque += offset.cross(lateral);
            if spec.steering {
                let mut target = (steering_multiplier * control.steering) * spec.max_angle_deg;
                if control.steering == 0.0 && !self.wheel_initialization.center_steering { target = wheel.angle_deg; }
                wheel.angle_deg = f32r(wheel.angle_deg + (target - wheel.angle_deg)
                    * (self.wheel_initialization.steering_speed * steering_multiplier));
            }
        }
        impulse = impulse + (old_velocity * 0.005f32) * (-cfg.air_resistance as f32);
        }
        let mut velocity = old_velocity + impulse * (1.0 / cfg.mass as f32);
        // ClampVelocity observes the cached body velocity before this tick's
        // impulses have reached _IntegrateForces.
        if was_active && drive && old_velocity.length() as f64 > cfg.max_velocity {
            velocity = old_velocity.normalized() * cfg.max_velocity as f32;
        }
        let previous_velocity=velocity;
        let previous_angular=self.angular_velocity as f32 + torque * (1.0 / cfg.inertia as f32);
        if !cfg.custom_integrator {
        // Native force accumulation retains the mass round trip and zero force additions.
        velocity = velocity * (1.0f32 - cfg.linear_damp as f32 * dt as f32).max(0.0)
            + ((F2::from(cfg.gravity) * cfg.mass as f32 + F2::default())
                + F2::default()) * (1.0 / cfg.mass as f32) * dt as f32;
        }
        self.velocity = velocity.into();
        self.angular_velocity = if cfg.custom_integrator { previous_angular as f64 } else { (previous_angular
            * (1.0f32 - cfg.angular_damp as f32 * dt as f32).max(0.0)) as f64 };
        let collided;
        if !track.shapes.is_empty() {
            let result = solve_wall_contacts(
                cfg,
                track,
                self.position,
                self.transform_angle,
                self.body_basis,
                self.velocity,
                self.angular_velocity,
                previous_velocity,
                previous_angular,
                dt,
                &mut self.scratch,
                &mut self.wall_contacts,
                &mut self.previous_contacts,
            );
            self.velocity = result.velocity;
            self.angular_velocity = result.angular_velocity;
            self.velocity = V2::new(f32r(self.velocity.x), f32r(self.velocity.y));
            self.angular_velocity = f32r(self.angular_velocity);
            self.position = V2::new(
                f32r(self.position.x + f32r(f32r(self.velocity.x + result.bias_velocity.x) * f32r(dt))),
                f32r(self.position.y + f32r(f32r(self.velocity.y + result.bias_velocity.y) * f32r(dt))),
            );
            let angle_delta=f32r(f32r(self.angular_velocity + result.bias_angular) * f32r(dt));
            self.apply_center_of_mass_displacement(cfg, angle_delta as f32);
            self.set_transform_angle(f32r(self.rotation + angle_delta));
            collided = result.collided;
        } else {
            let old_position = self.position;
            self.position = (F2::from(self.position) + F2::from(self.velocity) * dt as f32).into();
            let angle_delta=f32r(self.angular_velocity * f32r(dt));
            self.apply_center_of_mass_displacement(cfg, angle_delta as f32);
            self.set_transform_angle(f32r(self.rotation + angle_delta));
            collided = self.resolve_walls(world, old_position);
        }
        if drive { self.tick += 1; }
        if !track.shapes.is_empty() || self.scratch.has_shape(){self.scratch.update_shape(cfg,self.position,self.body_basis);}
        let callback=NativeCallback{old_velocity,collided,was_active,eliminate_on_wall};
        if defer_callback {self.pending_callback=Some(callback);return false;}
        self.finish_callback(world,callback)
    }

    fn finish_callback(&mut self, world: &World, callback: NativeCallback) -> bool {
        let NativeCallback{old_velocity,collided,was_active,eliminate_on_wall}=callback;
        let cfg=&world.vehicle;let track=&world.track;
        let report_collision = collided && was_active && self.pending_velocity_reset.is_none() && self.pending_transform_reset.is_none();
        if report_collision {
            self.collision_count += 1;
            if eliminate_on_wall {
                self.set_active(false);
            }
        }
        if self.pending_velocity_store {
            self.frozen_velocity = (self.velocity, self.angular_velocity);
            self.pending_velocity_store = false;
        }
        if let Some((linear, angular)) = self.pending_velocity_reset.take() {
            self.velocity = linear;
            self.angular_velocity = angular;
        }
        if let Some((position, rotation)) = self.pending_transform_reset.take() {
            self.position = position;
            let (x, y) = godot_math::reset_basis(rotation as f32);
            self.set_body_basis(x, y);
            if !track.shapes.is_empty() || self.scratch.has_shape() { self.scratch.update_shape(cfg, self.position, self.body_basis); }
        }
        self.reset_right = false;
        self.acceleration = (F2::from(self.velocity) - old_velocity).into();
        report_collision
    }

    fn apply_center_of_mass_displacement(&mut self, cfg: &crate::world::VehicleConfig, angle_delta: f32) {
        let center=self.body_basis.0*cfg.center_of_mass.x as f32+self.body_basis.1*cfg.center_of_mass.y as f32;
        if center.dot(center) as f64>1e-5f64*1e-5f64 {
            let s=crate::native_math::engine_sin(angle_delta);let c=crate::native_math::engine_cos(angle_delta);
            let rotated=F2{x:center.x*c-center.y*s,y:center.x*s+center.y*c};
            self.position=(F2::from(self.position)+(center-rotated)).into();
        }
    }

    /// `Simulator._resolve_walls`: prototype rectangle versus wall segments.
    fn resolve_walls(&mut self, world: &World, old_position: V2) -> bool {
        let cfg = &world.vehicle;
        let mut contact = false;
        for wall in &world.track.walls {
            let tangent = (wall.end - wall.start).normalized();
            if tangent.length() == 0.0 {
                continue;
            }
            let normal = V2::new(-tangent.y, tangent.x);
            let old_side = (old_position - wall.start).dot(normal);
            let side = (self.position - wall.start).dot(normal);
            let toward_car = normal * if old_side >= 0.0 { 1.0 } else { -1.0 };
            let (right, forward) = (self.right(), self.forward());
            let support = toward_car.dot(right).abs() * cfg.half_width + toward_car.dot(forward).abs() * cfg.half_length;
            let along = (self.position - wall.start).dot(tangent);
            let length = (wall.end - wall.start).length();
            let extent = tangent.dot(right).abs() * cfg.half_width + tangent.dot(forward).abs() * cfg.half_length;
            if -extent > along || along > length + extent {
                continue;
            }
            let signed_distance = if old_side >= 0.0 { side } else { -side };
            if signed_distance >= support
                || old_side * side < 0.0 && old_side.abs() > support + (self.position - old_position).length()
            {
                continue;
            }
            let penetration = support - signed_distance;
            self.position += toward_car * py_max(0.0, penetration);
            let normal_speed = self.velocity.dot(toward_car);
            if normal_speed < 0.0 {
                self.velocity -= toward_car * ((1.0 + cfg.bounce) * normal_speed);
            }
            contact = true;
        }
        contact
    }

    /// Offset along the path of the tile-connection sensor point, cached per pose.
    fn sensor_path_offset(&self, track: &Track, scratch: &mut SensorScratch) -> f64 {
        let point = track.path_sensor_point(self.actual_position());
        if let Some((cached_point, offset)) = scratch.path_cache {
            if cached_point.x.to_bits() == point.x.to_bits() && cached_point.y.to_bits() == point.y.to_bits() {
                return offset;
            }
        }
        let offset = track.curve.as_ref().map_or_else(|| track.closest_path(point, None).offset, |c| c.closest_offset(point.into()) as f64);
        scratch.path_cache = Some((point, offset));
        offset
    }

    /// `Simulator.sensor(name, **kwargs)`.
    pub fn sensor(&self, world: &World, sensor: Sensor, scratch: &mut SensorScratch) -> f64 {
        let cfg = &world.vehicle;
        let track = &world.track;
        match sensor {
            Sensor::Raycast { degrees, length } => {
                assert!(length > 0.0, "ray length must be positive");
                let length = length as f32;
                let angle = (degrees as f32 - 90.0f32) * (std::f32::consts::PI / 180.0);
                let local = F2 { x: crate::native_math::cos(angle) * length, y: crate::native_math::sin(angle) * length };
                let end = godot_math::transform_point(self.position, self.body_basis, local.into());
                match track.raycast(self.position, end, &mut scratch.stamps) {
                    None => 0.0,
                    Some(hit) if hit == end => 0.0,
                    Some(hit) => (1.0f32 - (F2::from(hit) - F2::from(self.position)).length() / length) as f64,
                }
            }
            Sensor::DistanceFromWall { max_distance } => {
                let position = self.actual_position();
                match track.closest_wall(position) {
                    None => 1.0,
                    Some(wall) => ((F2::from(position) - F2::from(wall)).length() / max_distance as f32).clamp(0.0, 1.0) as f64,
                }
            }
            Sensor::Speed => (F2::from(self.velocity).length() / cfg.max_velocity as f32).clamp(0.0, 1.0) as f64,
            Sensor::VelocityFront => {
                let right = self.cached_right();
                let front = right.rotated(-std::f32::consts::PI / 2.0);
                godot_math::signed_sensor(F2::from(self.actual_velocity()).dot(front), cfg.max_velocity as f32)
            },
            Sensor::VelocitySide => {
                let right = self.cached_right();
                godot_math::signed_sensor(F2::from(self.actual_velocity()).dot(right), cfg.max_velocity as f32 / 2.0)
            },
            Sensor::AccelerationFront { max_acceleration } => {
                let front = self.cached_right().rotated(-std::f32::consts::FRAC_PI_2);
                godot_math::signed_sensor(F2::from(self.actual_acceleration()).dot(front), max_acceleration as f32)
            }
            Sensor::AccelerationSide { max_acceleration } => {
                godot_math::signed_sensor(F2::from(self.actual_acceleration()).dot(self.cached_right()), max_acceleration as f32)
            }
            Sensor::AngularVelocity => godot_math::signed_sensor(self.actual_angular_velocity() as f32, 2.0),
            Sensor::WheelAngle => {
                let angle = cfg
                    .wheels
                    .iter()
                    .zip(&self.wheels)
                    .find(|(spec, _)| spec.steering)
                    .expect("wheel angle sensor requires a steering wheel").1.angle_deg;
                godot_math::signed_sensor(angle as f32, 45.0)
            }
            Sensor::BoostCapacity => clamp(self.boost_energy, 0.0, 1.0),
            Sensor::Grip { offset } => track.vehicle_surface((F2::from(self.position) + F2::from(offset).rotated(self.rotation as f32)).into(), cfg).grip as f32 as f64,
            Sensor::CorrectDirection | Sensor::TrackCurvature { .. } => {
                if track.path.len() < 2 {
                    return 0.0;
                }
                let here = self.sensor_path_offset(track, scratch);
                if let Some(curve) = &track.curve {
                    let tangent = curve.direction(here as f32).normalized();
                    let Sensor::TrackCurvature { min_lookahead, max_lookahead } = sensor else {
                        let forward = self.cached_right().rotated(-std::f32::consts::FRAC_PI_2).normalized();
                        return forward.dot(tangent).clamp(-1.0, 1.0) as f64;
                    };
                    if curve.length() <= 0.0 { return 0.0; }
                    let speed = (F2::from(self.actual_velocity()).length() / cfg.max_velocity as f32).clamp(0.0, 1.0);
                    let min = (min_lookahead as f32).max(0.0);
                    let max = (max_lookahead as f32).max(min);
                    let lookahead = min + (max-min)*speed;
                    let offset = (here as f32+lookahead)%curve.length();
                    let ahead = curve.direction(offset).normalized();
                    return (crate::native_math::atan2(tangent.cross(ahead),tangent.dot(ahead)).abs()/std::f32::consts::PI).clamp(0.0,1.0) as f64;
                }
                let tangent = track.path_direction(here);
                let Sensor::TrackCurvature { min_lookahead, max_lookahead } = sensor else {
                    return clamp(self.forward().dot(tangent), -1.0, 1.0);
                };
                let path_length = track.path_length();
                if path_length <= 0.0 {
                    return 0.0;
                }
                let lookahead = min_lookahead + (max_lookahead - min_lookahead) * self.sensor(world, Sensor::Speed, scratch);
                let target_offset = py_mod(here + lookahead, path_length);
                let ahead = track.path_direction(target_offset);
                tangent.cross(ahead).atan2(tangent.dot(ahead)).abs() / PI
            }
        }
    }
}

impl Car {
    /// The GPU car state (`gpu_sim::GpuCar`) with its collision scratch;
    /// errors on states the GPU port does not model.
    pub fn gpu_export(&self, vehicle: &crate::world::VehicleConfig, surfaces: &crate::gpu_sim::SurfaceTable) -> Result<crate::gpu_sim::GpuCar, String> {
        use crate::gpu_sim::*;
        let init = &self.wheel_initialization;
        if init.grip != vehicle.grip || init.steering_speed != vehicle.steering_speed || init.center_steering != vehicle.center_steering
            || !init.positions.iter().eq(vehicle.wheels.iter().map(|w| &w.position)) {
            return Err("wheel initialization differs from the vehicle config".into());
        }
        if self.frozen || self.pending_velocity_store {
            return Err("frozen cars are not supported".into());
        }
        if self.pending_transform_reset.is_some() || self.pending_callback.is_some() {
            return Err("pending transform reset or native callback".into());
        }
        if self.wheels.len() > MAX_WHEELS {
            return Err(format!("{} wheels exceed MAX_WHEELS", self.wheels.len()));
        }
        let mut c = GpuCar::zeroed();
        c.position = exact2(self.position, "position")?;
        c.velocity = exact2(self.velocity, "velocity")?;
        c.acceleration = exact2(self.acceleration, "acceleration")?;
        c.reset_acceleration = exact2(self.reset_acceleration, "reset_acceleration")?;
        c.basis_x = f2(self.body_basis.0);
        c.basis_y = f2(self.body_basis.1);
        c.rotation = exact(self.rotation, "rotation")?;
        c.transform_angle = exact(self.transform_angle, "transform_angle")?;
        c.angular_velocity = exact(self.angular_velocity, "angular_velocity")?;
        c.boost_energy = exact(self.boost_energy, "boost_energy")?;
        c.flags = if self.active { CAR_ACTIVE } else { 0 } | if self.reset_right { CAR_RESET_RIGHT } else { 0 };
        match self.pending_velocity_reset {
            None => {}
            Some((linear, angular)) if linear == V2::ZERO && angular == 0.0 => c.flags |= CAR_PENDING_VELOCITY,
            Some(_) => return Err("pending non-zero velocity reset".into()),
        }
        c.collision_count = u32::try_from(self.collision_count).map_err(|_| "collision count overflows u32")?;
        c.tick = self.tick;
        for (i, wheel) in self.wheels.iter().enumerate() {
            c.wheel_angle[i] = exact(wheel.angle_deg, "wheel angle")?;
            if let Some(p) = wheel.previous_position {
                // Only ever read through F2::from.
                c.wheel_previous[i] = f2(F2::from(p));
                c.wheel_has_previous |= 1 << i;
            }
            c.wheel_surface[i] = surfaces.by_value(wheel.previous_surface)?;
        }
        self.scratch.gpu_export(&mut c, &self.previous_contacts)?;
        Ok(c)
    }

    /// Sets the state exported by `gpu_export` from a GPU car. `wall_contacts`
    /// (a per-step report) is cleared.
    pub fn gpu_import(&mut self, c: &crate::gpu_sim::GpuCar, surfaces: &crate::gpu_sim::SurfaceTable) -> Result<(), String> {
        use crate::gpu_sim::*;
        if c.error != 0 {
            return Err(format!("GPU error bits {:#x}", c.error));
        }
        self.position = v2(c.position);
        self.velocity = v2(c.velocity);
        self.acceleration = v2(c.acceleration);
        self.reset_acceleration = v2(c.reset_acceleration);
        self.body_basis = (F2 { x: c.basis_x[0], y: c.basis_x[1] }, F2 { x: c.basis_y[0], y: c.basis_y[1] });
        self.rotation = c.rotation as f64;
        self.transform_angle = c.transform_angle as f64;
        self.angular_velocity = c.angular_velocity as f64;
        self.boost_energy = c.boost_energy as f64;
        self.active = c.flags & CAR_ACTIVE != 0;
        self.reset_right = c.flags & CAR_RESET_RIGHT != 0;
        self.pending_velocity_reset = (c.flags & CAR_PENDING_VELOCITY != 0).then_some((V2::ZERO, 0.0));
        self.collision_count = c.collision_count as u64;
        self.tick = c.tick;
        for (i, wheel) in self.wheels.iter_mut().enumerate() {
            wheel.angle_deg = c.wheel_angle[i] as f64;
            wheel.previous_position = (c.wheel_has_previous >> i & 1 != 0).then(|| v2(c.wheel_previous[i]));
            wheel.previous_surface = *surfaces.values.get(c.wheel_surface[i] as usize).ok_or("wheel surface index out of range")?;
        }
        self.wall_contacts.clear();
        self.scratch.gpu_import(c, &mut self.previous_contacts);
        Ok(())
    }
}

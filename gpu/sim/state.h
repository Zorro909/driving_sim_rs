// Host/device layouts of the simulation state, mirrored field by field by
// #[repr(C)] structs in src/gpu/simulation.rs. Every car value the CPU keeps as f64 is
// float32-representable (the exporter checks), so cars store f32.
#pragma once
#include <cstdint>

namespace altd {

constexpr int MAX_WHEELS = 4;
constexpr int MAX_PAIRS = 16;        // broad-phase pairs per car
constexpr int MAX_CONTACTS = 32;     // retained pair contacts per car (two per pair)
constexpr int MAX_SHAPE_POINTS = 16; // points per physics shape
constexpr int MAX_SENSORS = 32;
constexpr int MAX_OUTPUTS = 8;
constexpr int MAX_LAYERS = 12;
constexpr int MAX_WIDTH = 32;
constexpr int RECENT = 10;           // TrainingStats.recent_score_diffs capacity

// Per-car error bits; the math bits of math.h (1, 2, 4) are included.
enum CarError : uint32_t {
    ERR_PAIRS = 8,        // more than MAX_PAIRS broad-phase pairs
    ERR_CONTACTS = 16,    // more than MAX_CONTACTS retained contacts
    ERR_PATH_NAN = 32,    // NaN path offset (the CPU panics)
};

struct Vec2 { float x, y; };

enum ContactFlags : uint32_t { CONTACT_WALL_FIRST = 1, CONTACT_USED = 2 };

// collision::Contact
struct Contact {
    Vec2 normal, point, wall_point, local_point, wall_local_point;
    float depth;
    uint32_t shape, flags;
    float normal_mass, tangent_mass, bias, bounce, friction;
    float acc_normal, acc_tangent, acc_bias, acc_bias_center;
};

enum CarFlags : uint32_t {
    CAR_ACTIVE = 1,
    CAR_RESET_RIGHT = 2,
    CAR_PENDING_VELOCITY = 4,  // pending_velocity_reset = Some((0, 0), 0)
    CAR_HAS_SHAPE = 8,         // scratch.shape_aabb is Some
    CAR_HAS_LEAF = 16,         // scratch.leaf_aabb is Some
    CAR_PAIR_CHECK = 32,       // scratch.pair_check_pending
};

// car::Car with its CollisionScratch. `contacts` are the retained
// previous_contacts; `axes[i]` is separating_axes[pairs[i]].
struct Car {
    Vec2 position, velocity, acceleration, reset_acceleration, basis_x, basis_y;
    float rotation, transform_angle, angular_velocity, boost_energy;
    uint32_t flags, collision_count;
    uint64_t tick;
    float wheel_angle[MAX_WHEELS];
    Vec2 wheel_previous[MAX_WHEELS];
    uint32_t wheel_surface[MAX_WHEELS];  // index into TrackDesc::surfaces
    uint32_t wheel_has_previous;         // bit per wheel
    uint32_t pair_count, contact_count, error;
    Vec2 shape_position, shape_size;     // scratch.shape_aabb (FloatRect)
    Vec2 leaf_min, leaf_max;             // scratch.leaf_aabb
    uint32_t pairs[MAX_PAIRS];
    Vec2 axes[MAX_PAIRS];
    Contact contacts[MAX_CONTACTS];
};

enum AgentFlags : uint32_t {
    AGENT_HAS_PREVIOUS_OFFSET = 1,
    AGENT_WENT_BACKWARDS = 2,
    AGENT_HAS_LAP_CROSSING = 4,
    AGENT_PENDING_CONTACT = 8,
    AGENT_HAS_BEST_LAP = 16,
    AGENT_HAS_ALL_LAPS = 32,
    AGENT_HAS_DEACTIVATED = 64,
};

// training::TrainingAgent without its network and car: controls,
// TrainingStats with its ScoreTracker, and the window bookkeeping.
struct Agent {
    double controls[5];  // acceleration, steering, brake, handbrake, boost
    double previous_offset, tracker_score, lap_crossing_fraction;
    uint64_t lap_count;
    double network_novelty;
    uint64_t update_count, collision_count;
    double total_score, total_speed, total_abs_steering, total_actual_distance, total_drifting_amount,
        total_momentum_change_frequency, total_throttle_change_frequency, total_braking_frequency,
        total_distance_from_wall, total_distance_from_center;
    double best_lap_time;
    uint64_t idle_ticks;
    double previous_speed, previous_throttle, previous_score;
    uint64_t previous_lap_count, lap_start_tick, drift_ticks;
    double current_slip_angle_degrees, continuous_drift_time, max_continuous_drift_time, all_laps_time_sum;
    double recent[RECENT];  // ring buffer, oldest at recent_start
    uint32_t recent_start, recent_len;
    uint64_t deactivated_at;
    uint32_t flags, pad;
};

// world::WheelConfig with the conversions step_internal applies.
struct WheelDesc {
    Vec2 position;           // F2::from(local position)
    uint32_t steering;
    uint32_t handbrake_off;  // handbrake_power.abs() < 0.00001
    float power, brake_power, handbrake_power, brake_max;  // as f32; brake_max = max(brake, handbrake) as f32
    double max_angle_deg;
};

// world::VehicleConfig and the car's wheel initialization.
struct VehicleDesc {
    WheelDesc wheels[MAX_WHEELS];
    uint32_t wheel_count, custom_integrator, center_steering, solver_iterations;
    uint32_t max_contacts_reported, pad0;
    double steering_speed, max_velocity, dt, contact_max_separation;
    float grip, air_resistance, mass, inertia;
    float linear_damp, angular_damp, dt_f, max_velocity_f;
    Vec2 gravity, shape_basis_x, shape_basis_y, shape_size, shape_half, shape_position, center_of_mass;
    float friction, bounce, contact_bias, allowed_penetration;
    float recycle_radius, max_separation_f, pad1, pad2;
};

enum SensorKind : uint32_t {
    S_RAYCAST = 0,           // a = degrees, b = length
    S_DISTANCE_FROM_WALL,    // a = max distance
    S_SPEED,
    S_VELOCITY_FRONT,
    S_VELOCITY_SIDE,
    S_ACCELERATION_FRONT,    // a = max acceleration
    S_ACCELERATION_SIDE,     // a = max acceleration
    S_ANGULAR_VELOCITY,
    S_WHEEL_ANGLE,
    S_BOOST_CAPACITY,
    S_GRIP,                  // offset
    S_CORRECT_DIRECTION,
    S_TRACK_CURVATURE,       // a = min lookahead, b = max lookahead
};

struct SensorDesc {
    uint32_t kind;
    float a, b;
    Vec2 offset;
    uint32_t pad;
};

}  // namespace altd

// Device regression checks. Build the library first, then run:
// hipcc -std=c++17 gpu/tests/world_replacement.cpp -ldl -o /tmp/altd-world-replacement
// /tmp/altd-world-replacement target/gpu/libaltd_gpu.so
#include <hip/hip_runtime.h>
#include "../sim/world.h"
#include <dlfcn.h>
#include <cassert>
#include <cstdio>
using namespace altd;
struct Sim;

template<class T> T symbol(void* library, const char* name) {
    auto function = reinterpret_cast<T>(dlsym(library, name));
    assert(function);
    return function;
}

int main(int argc, char** argv) {
    assert(argc == 2);
    void* library = dlopen(argv[1], RTLD_NOW);
    if (!library) { std::puts(dlerror()); return 1; }
    auto make_world = symbol<World*(*)(const World*)>(library, "altd_gpu_world_create");
    auto free_world = symbol<void(*)(World*)>(library, "altd_gpu_world_free");
    auto make_sim = symbol<Sim*(*)(const World*, const VehicleDesc*, const SensorDesc*, uint32_t, uint32_t)>(library, "altd_gpu_sim_create");
    auto free_sim = symbol<void(*)(Sim*)>(library, "altd_gpu_sim_free");
    auto upload = symbol<int(*)(Sim*, uint32_t, const Car*, const Agent*)>(library, "altd_gpu_sim_upload");
    auto set_world = symbol<int(*)(Sim*, const World*)>(library, "altd_gpu_sim_set_world");
    auto step = symbol<int(*)(Sim*, uint32_t, uint32_t, uint64_t)>(library, "altd_gpu_sim_step");
    auto download = symbol<int(*)(Sim*, Car*, Agent*)>(library, "altd_gpu_sim_download");

    SurfaceDev surface{1, 1, 1, 0};
    Vec2 points[4] = {{-50, -50}, {50, -50}, {50, 50}, {-50, 50}};
    Vec2 normals[4] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
    ShapeDev shape{};
    shape.min_x = shape.min_y = -50; shape.max_x = shape.max_y = 50;
    shape.count = 4; shape.wall_first = 1;
    ShapeDev shapes[2] = {shape, shape};
    shapes[0].origin = {1000, 0}; shapes[0].min_x += 1000; shapes[0].max_x += 1000;
    World description{};
    description.surface_count = 1; description.surfaces = &surface;
    description.shape_count = 2; description.shape_point_count = 4;
    description.shapes = shapes; description.shape_local = points; description.shape_normals = normals;
    World* world = make_world(&description);
    assert(world);
    description.shape_count = 1;
    World* smaller_world = make_world(&description);
    assert(smaller_world);

    VehicleDesc vehicle{};
    vehicle.custom_integrator = 1; vehicle.mass = vehicle.inertia = 1;
    vehicle.dt = 1.0 / 60; vehicle.dt_f = static_cast<float>(vehicle.dt);
    vehicle.solver_iterations = 8; vehicle.max_contacts_reported = 4;
    vehicle.contact_max_separation = vehicle.max_separation_f = 10;
    vehicle.contact_bias = .2f; vehicle.allowed_penetration = .3f; vehicle.recycle_radius = 1;
    vehicle.shape_basis_x = {1, 0}; vehicle.shape_basis_y = {0, 1};
    vehicle.shape_size = {20, 20}; vehicle.shape_half = {10, 10};
    Agent agent{};
    Car initial{};
    initial.position = {45, 0}; initial.basis_x = {1, 0}; initial.basis_y = {0, 1};
    initial.flags = CAR_ACTIVE | CAR_HAS_SHAPE | CAR_HAS_LEAF | CAR_PAIR_CHECK;
    initial.shape_position = initial.leaf_min = {35, -10};
    initial.shape_size = {20, 20}; initial.leaf_max = {55, 10};
    initial.pair_count = 1; initial.pairs[0] = 1;

    auto tick = [&](World* next, const VehicleDesc& v, Car car, const Agent& a, bool drive) {
        Sim* sim = make_sim(world, &v, nullptr, 0, 1);
        assert(sim);
        assert(upload(sim, 1, &car, &a) == 0);
        if (next) assert(set_world(sim, next) == 0);
        assert(step(sim, drive, 0, 1) == 0);
        assert(download(sim, &car, nullptr) == 0);
        assert(car.error == 0);
        free_sim(sim);
        return car;
    };

    // A retained initialized AABB must discover the new world's pairs on tick one.
    Car baseline = tick(nullptr, vehicle, initial, agent, false);
    Car switched = tick(world, vehicle, initial, agent, false);
    assert(baseline.contact_count == 2 && baseline.collision_count == 1);
    assert(switched.contact_count == baseline.contact_count);
    assert(switched.collision_count == baseline.collision_count);
    assert(switched.position.x == baseline.position.x && switched.position.y == baseline.position.y);
    Car removed = tick(smaller_world, vehicle, initial, agent, false);
    assert(removed.pair_count == 0 && removed.contact_count == 0 && removed.collision_count == 0);

    // Wheel position history must keep braking impulses continuous across a switch.
    vehicle.wheel_count = 1; vehicle.mass = 1000; vehicle.grip = 1;
    vehicle.max_velocity = vehicle.max_velocity_f = 1e9;
    vehicle.wheels[0].brake_power = vehicle.wheels[0].brake_max = 1000;
    agent.controls[2] = 1;
    initial = {};
    initial.position = {150, 0}; initial.velocity = {60, 0};
    initial.wheel_previous[0] = {149, 0}; initial.wheel_has_previous = 1;
    initial.basis_x = {1, 0}; initial.basis_y = {0, 1}; initial.flags = CAR_ACTIVE;
    baseline = tick(nullptr, vehicle, initial, agent, true);
    switched = tick(world, vehicle, initial, agent, true);
    assert(baseline.velocity.x < initial.velocity.x);
    assert(switched.velocity.x == baseline.velocity.x && switched.velocity.y == baseline.velocity.y);
    assert(switched.position.x == baseline.position.x && switched.position.y == baseline.position.y);

    free_world(smaller_world); free_world(world); dlclose(library);
    std::puts("world replacement: first-tick contacts, smaller world, and wheel motion passed");
}

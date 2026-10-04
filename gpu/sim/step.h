// `Car::step` (src/physics/car.rs) with its wall collision (src/physics/collision.rs) for
// tracks with physics shapes and no native shared broadphase: one thread per
// car. Every f64 expression of the CPU code whose operands are float32 values
// and whose result is rounded to float32 at once (f32r, dot_f32, ...) is a
// plain float operation here: +, -, *, / and sqrt rounded through double
// equal the float operation.
#pragma once
#include "track.h"

namespace altd {

#ifdef ALTD_STEP_TIMING
// Diagnostic build: summed per-thread cycles of the car_step sections.
__device__ unsigned long long g_step_cycles[16];
#define STEP_MARK(var) uint64_t var = wall_clock64()
#define STEP_SECTION(k, var) do { uint64_t _now = wall_clock64(); atomicAdd(&g_step_cycles[k], _now - var); var = _now; } while (0)
#define STEP_COUNT(k) atomicAdd(&g_step_cycles[k], 1ull)
#define STEP_ADD(k, x) atomicAdd(&g_step_cycles[k], (unsigned long long)(x))
#else
#define STEP_MARK(var)
#define STEP_SECTION(k, var)
#define STEP_COUNT(k)
#define STEP_ADD(k, x)
#endif

struct Controls { double acceleration, steering, brake, handbrake, boost; };

// `Controls::normalized`
__device__ inline Controls normalized(const double* c) {
    return Controls{py_clamp(c[0], -1.0, 1.0), py_clamp(c[1], -1.0, 1.0), py_clamp(c[2], -1.0, 1.0),
                    py_clamp(c[3], 0.0, 1.0), py_clamp(c[4], 0.0, 1.0)};
}

// Vector2 == Vector2 (so -0 equals 0).
__device__ inline bool is_zero(Vec2 a) { return a.x == 0.0f && a.y == 0.0f; }

// ---- vehicle ----

// `godot_ease`
__device__ inline double godot_ease(double value, double curve) {
    return 1.0 - dpow(1.0 - py_clamp(value, 0.0, 1.0), 1.0 / curve);
}

// `godot_math::combine`
__device__ inline Vec2 combine(Vec2 brake, Vec2 drive, float brake_max, float drive_max) {
    float n = length(brake);
    if (n < 0.0001f || brake_max <= 0.0f) return brake + drive;
    float allowed = n + drive_max * (1.0f - rmin(n / brake_max, 1.0f));
    Vec2 result = brake + drive;
    Vec2 axis = normalized(brake);
    float projection = dot(result, axis);
    return projection > allowed ? result - axis * (projection - allowed) : result;
}

// `Car::set_transform_angle`
__device__ inline void set_transform_angle(Car& car, float angle, uint32_t& err) {
    float s, c;
    engine_sin_cos(angle, s, c, err);
    car.transform_angle = angle;
    car.basis_x = Vec2{c, s};
    car.basis_y = Vec2{-s, c};
    car.rotation = native_atan2(s, c);
}

// ---- CollisionScratch::update_shape ----

__device__ inline void update_shape(const VehicleDesc& v, Car& car) {
    Vec2 bx = car.basis_x, by = car.basis_y;
    Vec2 x = bx * v.shape_basis_x.x + by * v.shape_basis_x.y;
    Vec2 y = bx * v.shape_basis_y.x + by * v.shape_basis_y.y;
    Vec2 size = v.shape_size;
    Vec2 shape_origin = bx * v.shape_position.x + by * v.shape_position.y + car.position;
    Vec2 pos = x * (-size.x * 0.5f) + y * (-size.y * 0.5f) + shape_origin;
    Vec2 dx = x * size.x, dy = y * size.y;
    // FloatRect::expand from an empty rect at pos, keeping (position, size).
    Vec2 rp = pos, rs = Vec2{0.0f, 0.0f};
    auto expand_rect = [&](Vec2 p) {
        Vec2 e = rp + rs;
        Vec2 b = Vec2{rmin(rp.x, p.x), rmin(rp.y, p.y)};
        Vec2 f = Vec2{rmax(e.x, p.x), rmax(e.y, p.y)};
        rp = b;
        rs = f - b;
    };
    expand_rect(pos + dx);
    expand_rect(pos + dy);
    expand_rect((pos + dx) + dy);
    float margin = (car.flags & CAR_HAS_SHAPE)
        ? (float)((double)(car.shape_size.x + car.shape_size.y) * 0.5 * 0.05) : 0.0f;
    rp = rp - Vec2{margin, margin};
    rs = rs + Vec2{margin * 2.0f, margin * 2.0f};
    car.shape_position = rp;
    car.shape_size = rs;
    car.flags |= CAR_HAS_SHAPE;
    Vec2 raw_min = rp, raw_max = rp + rs;
    if (!(car.flags & CAR_HAS_LEAF)) {
        // item_add inserts the unexpanded box.
        car.leaf_min = raw_min;
        car.leaf_max = raw_max;
        car.flags |= CAR_HAS_LEAF | CAR_PAIR_CHECK;
        return;
    }
    float threshold = (float)((double)0.1f * 2.0 * 2.0 * (double)1.1f);
    float density = (float)((double)car.pair_count * (1.0 / 9.0));
    float grow = 0.1f * (1.0f - rmin(density, 1.0f));
    Vec2 old_size = car.leaf_max - car.leaf_min;
    Vec2 rt_max = car.leaf_min + old_size;  // roundtrip_rect
    bool encloses = raw_min.x >= car.leaf_min.x && raw_min.y >= car.leaf_min.y && raw_max.x <= rt_max.x && raw_max.y <= rt_max.y;
    if (encloses && (old_size.x + old_size.y) - (rs.x + rs.y) < threshold) return;
    car.leaf_min = raw_min - Vec2{grow, grow};
    car.leaf_max = raw_max + Vec2{grow, grow};
    car.flags |= CAR_PAIR_CHECK;
}

// ---- SAT contacts (find_contacts_with_basis) ----

struct Support { Vec2 p[2]; uint32_t n; };

__device__ inline Vec2 project_on_line(Vec2 point, Vec2 e0, Vec2 e1) {
    Vec2 direction = e1 - e0, offset = point - e0;
    float ls = dot(direction, direction);
    if (ls < 1e-20f) return e0;
    float fraction = dot(direction, offset) / ls;
    return e0 + direction * fraction;
}

__device__ inline Support rectangle_supports(const Vec2* car, Vec2 direction, Vec2 bx, Vec2 by) {
    Vec2 local = normalized(Vec2{dot(bx, direction), dot(by, direction)});
    if ((double)fabsf(local.x) > 0.99998)
        return local.x > 0.0f ? Support{{car[2], car[1]}, 2} : Support{{car[3], car[0]}, 2};
    if ((double)fabsf(local.y) > 0.99998)
        return local.y > 0.0f ? Support{{car[2], car[3]}, 2} : Support{{car[1], car[0]}, 2};
    if (local.x < 0.0f) return Support{{local.y < 0.0f ? car[0] : car[3], Vec2{0, 0}}, 1};
    return Support{{local.y < 0.0f ? car[1] : car[2], Vec2{0, 0}}, 1};
}

// The loop returns at the first edge facing `direction`, after updating the
// maximum up to it: that maximum is only used without such an edge.
__device__ inline Support polygon_supports(const World& w, const ShapeDev& shape, Vec2 wall_offset, Vec2 direction) {
    direction = normalized(direction);
    const Vec2* local = w.shape_local + shape.first;
    const Vec2* normals = w.shape_normals + shape.first;
    float maximum = -INFINITY;
    uint32_t best = 0, edge = NONE, n = shape.count;
    // Unrolled for constant indices; leaving at n skips the predicated no-op
    // iterations (most track shapes have 3 or 4 points).
#pragma unroll
    for (uint32_t i = 0; i < MAX_SHAPE_POINTS; i++) {
        if (i >= n) break;
        float projection = dot(local[i], direction);
        bool higher = projection > maximum;
        maximum = higher ? projection : maximum;
        best = higher ? i : best;
        edge = edge == NONE && (double)dot(normals[i], direction) > 0.99998 ? i : edge;
    }
    if (edge != NONE) return Support{{local[edge] + wall_offset, local[edge + 1 == n ? 0 : edge + 1] + wall_offset}, 2};
    return Support{{local[best] + wall_offset, Vec2{0, 0}}, 1};
}

// `contact_pairs`: (car point, wall point) pairs.
__device__ inline uint32_t contact_pairs(const Support& a, const Support& b, Vec2 push, Vec2 (*out)[2]) {
    if (a.n == 1 && b.n == 1) { out[0][0] = a.p[0]; out[0][1] = b.p[0]; return 1; }
    if (a.n == 1) { out[0][0] = a.p[0]; out[0][1] = project_on_line(a.p[0], b.p[0], b.p[1]); return 1; }
    if (b.n == 1) { out[0][0] = project_on_line(b.p[0], a.p[0], a.p[1]); out[0][1] = b.p[0]; return 1; }
    Vec2 tangent = Vec2{push.y, -push.x};
    float distance_a = dot(push, a.p[0]), distance_b = dot(push, b.p[0]);
    float key[4] = {dot(a.p[0], tangent), dot(a.p[1], tangent), dot(b.p[0], tangent), dot(b.p[1], tangent)};
    uint32_t order[4] = {0, 1, 2, 3};  // 0, 1: car points; 2, 3: wall points
    for (int i = 1; i < 4; i++)
        for (int j = i; j > 0 && key[order[j]] < key[order[j - 1]]; j--) {
            uint32_t t = order[j]; order[j] = order[j - 1]; order[j - 1] = t;
        }
    uint32_t count = 0;
    for (int k = 1; k < 3; k++) {
        uint32_t o = order[k];
        bool from_car = o < 2;
        Vec2 point = from_car ? a.p[o] : b.p[o - 2];
        float distance = dot(push, point) - (from_car ? distance_b : distance_a);
        Vec2 other = point - push * distance;
        Vec2 p0 = from_car ? point : other, p1 = from_car ? other : point;
        if ((double)dot(push, p0) <= (double)dot(push, p1) - 1e-5) {
            out[count][0] = p0;
            out[count][1] = p1;
            count++;
        }
    }
    return count;
}

__device__ inline bool shape_overlaps(const ShapeDev& s, double left, double right, double top, double bottom) {
    return !(s.max_x < left || s.min_x > right || s.max_y < top || s.min_y > bottom);
}

// Broad pairs update (pair_check_pending, no shared broadphase): retain the
// overlapping pairs with their separating axes, then append overlapping
// shapes in ascending index order.
__device__ inline void update_pairs(const World& w, Car& car, double left, double right, double top, double bottom) {
    uint32_t kept = 0;
    for (uint32_t i = 0; i < car.pair_count; i++) {
        uint32_t index = car.pairs[i];
        if (shape_overlaps(w.shapes[index], left, right, top, bottom)) {
            car.pairs[kept] = index;
            car.axes[kept] = car.axes[i];
            kept++;
        }
    }
    uint32_t retained = kept;
    auto visit = [&](uint32_t index) {
        if (!shape_overlaps(w.shapes[index], left, right, top, bottom)) return;
        for (uint32_t j = 0; j < retained; j++)
            if (car.pairs[j] == index) return;
        if (kept == MAX_PAIRS) { car.error |= ERR_PAIRS; return; }
        car.pairs[kept] = index;
        car.axes[kept] = Vec2{0.0f, 0.0f};
        kept++;
    };
    // Cell lists hold every shape whose box, grown by one unit, meets the
    // cell grown by shape_margin: complete for queries inside that region.
    const NearGrid& g = w.shape_grid;
    double fx = floor((left - g.x0) / g.cell), fy = floor((top - g.y0) / g.cell);
    bool inside = fx >= 0.0 && fy >= 0.0 && fx < (double)g.nx && fy < (double)g.ny
        && right < g.x0 + (fx + 1.0) * g.cell + w.shape_margin && bottom < g.y0 + (fy + 1.0) * g.cell + w.shape_margin;
    if (inside) {
        uint32_t cell = (uint32_t)fy * g.nx + (uint32_t)fx;
        for (uint32_t k = g.start[cell], e = g.start[cell + 1]; k < e; k++) visit(g.items[k]);
    } else {
        STEP_COUNT(6);
        for (uint32_t i = 0; i < w.shape_count; i++) visit(i);
    }
    STEP_COUNT(7);
    car.pair_count = kept;
    car.flags &= ~CAR_PAIR_CHECK;
}

// The car side of `find_contacts_with_basis`: the collision shape's basis,
// its corners relative to the car origin and the car's SAT axes.
struct CarFrame { Vec2 bx, by, basis_x, basis_y, local[4], axis_x, axis_y; };

__device__ inline CarFrame car_frame(const VehicleDesc& v, const Car& car) {
    CarFrame f;
    f.bx = car.basis_x;
    f.by = car.basis_y;
    f.basis_x = f.bx * v.shape_basis_x.x + f.by * v.shape_basis_x.y;
    f.basis_y = f.bx * v.shape_basis_y.x + f.by * v.shape_basis_y.y;
    const float sxs[4] = {-1.0f, 1.0f, 1.0f, -1.0f}, sys[4] = {-1.0f, -1.0f, 1.0f, 1.0f};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float x = sxs[i] * v.shape_half.x, y = sys[i] * v.shape_half.y;
        f.local[i] = Vec2{f.basis_x.x * x + f.basis_y.x * y, f.basis_x.y * x + f.basis_y.y * y};
    }
    f.axis_x = normalized(f.basis_x);
    f.axis_y = normalized(f.basis_y);
    return f;
}

// One broad pair, relative to the pair origin: the wall's origin if the wall
// comes first, else the car's.
struct PairFrame { Vec2 origin, car_offset, wall_offset, car[4]; bool wall_first; };

__device__ inline PairFrame pair_frame(const VehicleDesc& v, const Car& car, const CarFrame& f, const ShapeDev& shape) {
    PairFrame p;
    p.wall_first = shape.wall_first != 0;
    p.origin = p.wall_first ? shape.origin : car.position;
    p.car_offset = car.position - p.origin;
    p.wall_offset = shape.origin - p.origin;
    Vec2 shape_origin = f.bx * v.shape_position.x + f.by * v.shape_position.y + p.car_offset;
#pragma unroll
    for (int i = 0; i < 4; i++) p.car[i] = f.local[i] + shape_origin;
    return p;
}

// The SAT test of broad pair ci of `find_contacts_with_basis`. It reads and
// writes only the pair's cached axis, so the pairs are independent. Returns
// the push axis of the least penetration, or zero if the shapes are separated
// (caching the separating axis) or no axis qualifies.
__device__ inline Vec2 sat_pair(const World& w, const VehicleDesc& v, Car& car, uint32_t ci) {
    CarFrame f = car_frame(v, car);
    const ShapeDev& shape = w.shapes[car.pairs[ci]];
    PairFrame p = pair_frame(v, car, f, shape);
    uint32_t n = shape.count;
    const Vec2* local = w.shape_local + shape.first;
    Vec2 wall[MAX_SHAPE_POINTS];  // registers: constant indices only
#pragma unroll
    for (uint32_t i = 0; i < MAX_SHAPE_POINTS; i++) wall[i] = i < n ? local[i] + p.wall_offset : Vec2{0.0f, 0.0f};
    // Axes: the cached separating axis (if any), the car axes, the wall edge normals.
    Vec2& separating = car.axes[ci];
    Vec2 cached = separating;
    uint32_t axis_count = 3 + n;
    float best_depth = 1e15f;
    Vec2 push = Vec2{0.0f, 0.0f};
    STEP_COUNT(12);
    STEP_ADD(15, n);
    for (uint32_t k = is_zero(cached) ? 1 : 0; k < axis_count; k++) {
        STEP_COUNT(13);
        Vec2 axis;
        if (k >= 3) {
            uint32_t e = k - 3;
            Vec2 en = normalized((local[e + 1 == n ? 0 : e + 1] + p.wall_offset) - (local[e] + p.wall_offset));
            axis = Vec2{en.y, -en.x};
        } else {
            axis = k == 0 ? cached : k == 1 ? f.axis_x : f.axis_y;
        }
        if ((double)fabsf(axis.x) < 1e-5 && (double)fabsf(axis.y) < 1e-5) axis = Vec2{0.0f, 1.0f};
        float min_a = INFINITY, max_a = -INFINITY, min_b = INFINITY, max_b = -INFINITY;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float d = dot(p.car[i], axis);
            min_a = d < min_a ? d : min_a;
            max_a = d > max_a ? d : max_a;
        }
#pragma unroll
        for (uint32_t i = 0; i < MAX_SHAPE_POINTS; i++) {
            if (i >= n) break;  // constant indices keep `wall` in registers; iterations past n did nothing
            float d = dot(wall[i], axis);
            min_b = d < min_b ? d : min_b;
            max_b = d > max_b ? d : max_b;
        }
        float width_a = (max_a - min_a) * 0.5f, center_a = (min_a + max_a) * 0.5f;
        float dmin = (min_b - width_a) - center_a, dmax = (max_b + width_a) - center_a;
        if (dmin > 0.0f || dmax < 0.0f) {
            STEP_COUNT(14);
            separating = axis;
            return Vec2{0.0f, 0.0f};
        }
        bool use_max = dmax < fabsf(dmin);
        float depth = use_max ? dmax : fabsf(dmin);
        if (depth < best_depth) {
            best_depth = depth;
            push = use_max ? axis : -axis;
        }
    }
    if (!is_zero(push)) separating = Vec2{0.0f, 0.0f};
    return push;
}

// The contacts of broad pair ci for its (nonzero) push axis, `add`ed in
// order; returns how many.
template <class Add>
__device__ inline uint32_t pair_contacts(const World& w, const VehicleDesc& v, const Car& car, const CarFrame& f, uint32_t ci,
                                         Vec2 push, Add&& add) {
    uint32_t shape_index = car.pairs[ci];
    const ShapeDev& shape = w.shapes[shape_index];
    PairFrame p = pair_frame(v, car, f, shape);
    Support support_car = rectangle_supports(p.car, -push, f.basis_x, f.basis_y);
    Support support_wall = polygon_supports(w, shape, p.wall_offset, push);
    Vec2 pairs[2][2];
    uint32_t made = contact_pairs(support_car, support_wall, push, pairs), added = 0;
    for (uint32_t k = 0; k < made; k++) {
        Vec2 car_point = pairs[k][0], wall_point = pairs[k][1];
        Vec2 delta = p.wall_first ? wall_point - car_point : car_point - wall_point;
        float depth = sqrtf(dot(delta, delta));
        if (!(depth > 0.0f)) continue;
        Vec2 normal = Vec2{delta.x / depth, delta.y / depth};
        if (!p.wall_first) normal = -normal;
        Vec2 relative = car_point - p.car_offset;
        Vec2 local = Vec2{dot(f.bx, relative), dot(f.by, relative)};
        Vec2 global_a = (f.bx * local.x + f.by * local.y) + p.car_offset;
        Vec2 wall_local = wall_point - p.wall_offset;
        Vec2 global_b = wall_local + p.wall_offset;
        Contact c{};
        c.normal = normal;
        c.point = global_a + p.origin;
        c.wall_point = global_b + p.origin;
        c.local_point = local;
        c.wall_local_point = wall_local;
        c.depth = dot(global_b - global_a, normal);
        c.shape = shape_index;
        c.flags = (p.wall_first ? CONTACT_WALL_FIRST : 0u) | CONTACT_USED;
        add(c);
        added++;
    }
    return added;
}

// ---- solve_wall_contacts ----

struct SolveResult { Vec2 velocity, bias_velocity; float angular, bias_angular; bool collided; };

// `refresh_contact`; returns the separation vector of the validation.
__device__ inline Vec2 refresh_contact(const World& w, Contact& c, Vec2 position, Vec2 bx, Vec2 by) {
    const ShapeDev& shape = w.shapes[c.shape];
    Vec2 origin = (c.flags & CONTACT_WALL_FIRST) ? shape.origin : position;
    Vec2 car_offset = position - origin, wall_offset = shape.origin - origin;
    Vec2 car = (bx * c.local_point.x + by * c.local_point.y) + car_offset;
    Vec2 wall = c.wall_local_point + wall_offset;
    c.depth = dot(wall - car, c.normal);
    c.point = car + origin;
    c.wall_point = wall + origin;
    return (wall - c.normal * c.depth) - car;
}

__device__ inline void remove_contact(Car& car, uint32_t index) {
    for (uint32_t j = index; j + 1 < car.contact_count; j++) car.contacts[j] = car.contacts[j + 1];
    car.contact_count--;
}

// Lever arm of `contact_arm`.
__device__ inline Vec2 contact_arm(const World& w, const Contact& c, Vec2 position, Vec2 bx, Vec2 by, Vec2 center) {
    Vec2 local = bx * c.local_point.x + by * c.local_point.y;
    Vec2 offset = (c.flags & CONTACT_WALL_FIRST) ? position - w.shapes[c.shape].origin : Vec2{0.0f, 0.0f};
    return ((local + offset) - center) - offset;
}

// The start of `solve_wall_contacts`: validates the retained contacts, then
// updates the broad pairs (`find_contacts_with_basis` before its SAT tests)
// and keeps the contacts of the paired shapes.
__device__ inline void prepare_contacts(const World& w, const VehicleDesc& v, Car& car) {
    STEP_MARK(ts);
    Vec2 position = car.position, bx = car.basis_x, by = car.basis_y;
    float max_sep2 = v.max_separation_f * v.max_separation_f;
    for (uint32_t i = 0; i < car.contact_count;) {
        Contact& c = car.contacts[i];
        Vec2 separation = refresh_contact(w, c, position, bx, by);
        if (!(c.flags & CONTACT_USED) || (double)c.depth < -v.contact_max_separation || dot(separation, separation) > max_sep2) {
            uint32_t last = i;
            for (uint32_t j = car.contact_count; j-- > i;)
                if (car.contacts[j].shape == c.shape) { last = j; break; }
            car.contacts[i] = car.contacts[last];
            remove_contact(car, last);
        } else {
            c.flags &= ~CONTACT_USED;
            i++;
        }
    }
    STEP_SECTION(1, ts);
    if (!(car.flags & CAR_HAS_SHAPE)) update_shape(v, car);
    // leaf.roundtrip_rect()
    Vec2 qmin = car.leaf_min, qmax = car.leaf_min + (car.leaf_max - car.leaf_min);
    if (car.flags & CAR_PAIR_CHECK) update_pairs(w, car, (double)qmin.x, (double)qmax.x, (double)qmin.y, (double)qmax.y);
    STEP_SECTION(10, ts);
    // previous.retain(in broad_pairs)
    uint32_t kept = 0;
    for (uint32_t i = 0; i < car.contact_count; i++) {
        bool paired = false;
        for (uint32_t j = 0; j < car.pair_count; j++) paired |= car.pairs[j] == car.contacts[i].shape;
        if (paired) car.contacts[kept++] = car.contacts[i];
    }
    car.contact_count = kept;
}

// `merge` of one fresh contact: finding reads no contacts, so merging each
// as it is found matches merging the collected list in order.
__device__ inline void merge_contact(const VehicleDesc& v, Car& car, const Contact& nw) {
    float radius2 = v.recycle_radius * v.recycle_radius;
    int32_t found = -1;
    uint32_t same = 0;
    for (uint32_t i = 0; i < car.contact_count; i++) {
        const Contact& old = car.contacts[i];
        if (old.shape != nw.shape) continue;
        same++;
        Vec2 a = old.local_point - nw.local_point, b = old.wall_local_point - nw.wall_local_point;
        if (found < 0 && dot(a, a) < radius2 && dot(b, b) < radius2) found = (int32_t)i;
    }
    if (found >= 0) {
        Contact& old = car.contacts[found];
        Contact r = nw;
        r.acc_normal = old.acc_normal;
        r.acc_tangent = old.acc_tangent;
        r.acc_bias = old.acc_bias;
        r.acc_bias_center = old.acc_bias_center;
        old = r;
    } else if (same < 2) {
        if (car.contact_count == MAX_CONTACTS) car.error |= ERR_CONTACTS;
        else car.contacts[car.contact_count++] = nw;
    } else {
        int32_t least = -1;
        float depth = nw.depth;
        for (uint32_t i = 0; i < car.contact_count; i++)
            if (car.contacts[i].shape == nw.shape && car.contacts[i].depth < depth) { depth = car.contacts[i].depth; least = (int32_t)i; }
        if (least >= 0) car.contacts[least] = nw;
    }
}

// The rest of `solve_wall_contacts` once this tick's contacts are merged;
// `made_contacts` has a bit per broad pair that made contacts.
__device__ inline SolveResult solve_contacts(const World& w, const VehicleDesc& v, Car& car, uint32_t made_contacts, Vec2 velocity,
                                             float angular, Vec2 initial_velocity, float initial_angular) {
    STEP_MARK(ts);
    Vec2 position = car.position, bx = car.basis_x, by = car.basis_y;
    // Active contacts in broad-pair order: positive depth, shape collided this tick.
    uint8_t active[MAX_CONTACTS];
    uint32_t active_count = 0;
    for (uint32_t p = 0; p < car.pair_count; p++) {
        uint32_t shape = car.pairs[p];
        bool collided = false;
        for (uint32_t q = 0; q < car.pair_count; q++) collided |= ((made_contacts >> q) & 1) && car.pairs[q] == shape;
        if (!collided) continue;
        for (uint32_t i = 0; i < car.contact_count; i++)
            if (car.contacts[i].shape == shape && car.contacts[i].depth > 0.0f) active[active_count++] = (uint8_t)i;
    }
    STEP_SECTION(3, ts);
    if (active_count == 0) return SolveResult{velocity, Vec2{0.0f, 0.0f}, angular, 0.0f, false};
    STEP_COUNT(9);
    float inv_mass = 1.0f / v.mass, inv_inertia = 1.0f / v.inertia, inv_dt = 1.0f / v.dt_f;
    Vec2 vel = velocity;
    float wv = angular;
    Vec2 center = bx * v.center_of_mass.x + by * v.center_of_mass.y;
    for (uint32_t k = 0; k < active_count; k++) {
        Contact& c = car.contacts[active[k]];
        const ShapeDev& shape = w.shapes[c.shape];
        Vec2 arm = contact_arm(w, c, position, bx, by, center);
        Vec2 n = c.normal, t = Vec2{-n.y, n.x};
        float rn = dot(arm, n), rt = dot(arm, t);
        c.normal_mass = 1.0f / (inv_mass + inv_inertia * (dot(arm, arm) - rn * rn));
        c.tangent_mass = 1.0f / (inv_mass + inv_inertia * (dot(arm, arm) - rt * rt));
        c.bias = -v.contact_bias * inv_dt * rmin(-c.depth + v.allowed_penetration, 0.0f);
        Vec2 rotational = Vec2{-arm.y * initial_angular, arm.x * initial_angular};
        c.bounce = rclamp(v.bounce + shape.bounce, 0.0f, 1.0f);
        if (c.bounce != 0.0f) c.bounce *= dot(initial_velocity + rotational, n);
        c.friction = fabsf(rmin(v.friction, shape.friction));
        Vec2 impulse = n * c.acc_normal + t * c.acc_tangent;
        vel = vel + impulse * inv_mass;
        wv += cross((arm + center) - center, impulse) * inv_inertia;
    }
    Vec2 bv = Vec2{0.0f, 0.0f};
    float bw = 0.0f;
    float bias_limit = (float)(0.39269908169872414 / (double)v.dt_f);
    for (uint32_t iteration = 0; iteration < v.solver_iterations; iteration++) {
        for (uint32_t k = 0; k < active_count; k++) {
            Contact& c = car.contacts[active[k]];
            Vec2 arm = contact_arm(w, c, position, bx, by, center);
            Vec2 impulse_arm = (arm + center) - center;
            Vec2 n = c.normal, t = Vec2{-n.y, n.x};
            Vec2 rotational = Vec2{-arm.y, arm.x};
            Vec2 at_contact = vel + rotational * wv;
            float normal_speed = dot(at_contact, n), tangent_speed = dot(at_contact, t);
            float biased = dot(bv + rotational * bw, n);
            float bias_impulse = (c.bias - biased) * c.normal_mass;
            float previous_bias = c.acc_bias;
            c.acc_bias = rmax(previous_bias + bias_impulse, 0.0f);
            Vec2 impulse = n * (c.acc_bias - previous_bias);
            bv = bv + impulse * inv_mass;
            bw = rclamp(bw + cross(impulse_arm, impulse) * inv_inertia, -bias_limit, bias_limit);
            biased = dot(bv + rotational * bw, n);
            if (fabsf(c.bias - biased) > 0.001f) {
                float center_impulse = (c.bias - biased) / inv_mass;
                float previous_center = c.acc_bias_center;
                c.acc_bias_center = rmax(previous_center + center_impulse, 0.0f);
                bv = bv + (n * (c.acc_bias_center - previous_center)) * inv_mass;
            }
            float normal_impulse = -(normal_speed + c.bounce) * c.normal_mass;
            float previous_normal = c.acc_normal;
            c.acc_normal = rmax(previous_normal + normal_impulse, 0.0f);
            float friction_limit = c.friction * c.acc_normal;
            float tangent_impulse = -tangent_speed * c.tangent_mass;
            float previous_tangent = c.acc_tangent;
            c.acc_tangent = rclamp(previous_tangent + tangent_impulse, -friction_limit, friction_limit);
            impulse = n * (c.acc_normal - previous_normal) + t * (c.acc_tangent - previous_tangent);
            vel = vel + impulse * inv_mass;
            wv += cross(impulse_arm, impulse) * inv_inertia;
        }
    }
    STEP_SECTION(4, ts);
    // Reports are capped at max_contacts_reported, so any active contact collides unless the cap is 0.
    return SolveResult{vel, bv, wv, bw, v.max_contacts_reported > 0};
}

// ---- Car::step ----
//
// In parts: `step_begin` (driving, contact validation, broad pairs), then
// `sat_pair` for every broad pair, then `step_end` (contacts, merge, solve,
// integration, finish_callback). `car_step` runs them in a row.

// A step's state between its parts.
struct StepCarry {
    Vec2 old_velocity, velocity, previous_velocity;
    float angular, previous_angular;
    uint32_t was_active;
    uint32_t base;  // first SAT item of the car (the split kernels)
};

// `Car::step` (drive) or `Car::passive_step` up to the SAT tests.
__device__ inline StepCarry step_begin(const World& w, const VehicleDesc& v, Car& car, const double* raw, bool drive) {
    STEP_MARK(tc);
    STEP_COUNT(8);
    uint32_t err = 0;
    bool was_active = car.flags & CAR_ACTIVE;
    Controls control = normalized(raw);
    Vec2 impulse = Vec2{0.0f, 0.0f};
    float torque = 0.0f;
    Vec2 old_velocity = car.velocity;
    bool driving = was_active && drive;
    if (driving) {
        float boost = (float)control.boost, energy = car.boost_energy, strength = 0.0f;
        if (boost > 0.0f && energy > 0.0f) {
            energy = rmax(energy - boost * 0.01f, 0.0f);
            strength = boost * 0.5f;
        } else if (boost <= 0.0f) {
            energy = rmin(energy + 0.002f, 1.0f);
        }
        car.boost_energy = energy;
        float drive_amount = (float)(control.acceleration * (control.acceleration > 0.0 ? 1.0 + (double)strength : 1.0));
        double steering_multiplier = 1.0 / (0.002 * (double)length(car.velocity) + 1.0);
        Vec2 body_x = car.basis_x, body_y = car.basis_y;
        // godot_ease(handbrake, 0.3) takes one of two inputs per step (the
        // control, or 0.0 for wheels without handbrake power), so each pow
        // is evaluated at most once and shared by the wheels with that input.
        double ease[2];
        bool eased[2] = {false, false};
        for (uint32_t i = 0; i < v.wheel_count; i++) {
            const WheelDesc& spec = v.wheels[i];
            Vec2 position = transform_point(car.position, body_x, body_y, spec.position);
            Vec2 offset = position - car.position;
            Vec2 wheel_velocity = position - ((car.wheel_has_previous >> i & 1) ? car.wheel_previous[i] : position);
            car.wheel_previous[i] = position;
            car.wheel_has_previous |= 1u << i;
            float angle = car.wheel_angle[i] * (PI_F / 180.0f);
            float s, c;
            managed_sin_cos(angle, s, c, err);
            Vec2 forward = -normalized(body_x * -s + body_y * c);
            Vec2 right = normalized(body_x * c + body_y * s);
            float extra = angle * (w.surfaces[car.wheel_surface[i]].steering - 1.0f);
            if (extra != 0.0f) {
                forward = rotated(forward, extra, err);
                right = rotated(right, extra, err);
            }
            uint32_t surface_index = vehicle_surface(w, position);
            const SurfaceDev& surface = w.surfaces[surface_index];
            car.wheel_surface[i] = surface_index;
            Vec2 direction = normalized(wheel_velocity);
            Vec2 braking = limit(-direction * ((float)control.handbrake * spec.handbrake_power)
                                     + -direction * ((float)control.brake * spec.brake_power), spec.brake_max);
            Vec2 drive_force = forward * (drive_amount * spec.power);
            Vec2 wheel_drive = combine(braking, drive_force, spec.brake_max, spec.power) * surface.power;
            impulse = impulse + wheel_drive;
            torque += cross(offset, wheel_drive);
            float lateral_speed = dot(wheel_velocity, right);
            uint32_t off = spec.handbrake_off ? 1 : 0;
            if (!eased[off]) {
                ease[off] = godot_ease(off ? 0.0 : control.handbrake, 0.3);
                eased[off] = true;
            }
            double lateral_grip = 0.20000000298023224
                + (0.8 + (0.1 - 0.8) * ease[off])
                    / (1.0 + dexp(((double)length(wheel_velocity) - 450.0) * 0.00800000037997961));
            float effective_grip = (float)((double)v.grip * lateral_grip);
            Vec2 lateral = right * (-effective_grip * lateral_speed) * surface.grip;
            impulse = impulse + lateral;
            torque += cross(offset, lateral);
            if (spec.steering) {
                double current = (double)car.wheel_angle[i];
                double target = (steering_multiplier * control.steering) * spec.max_angle_deg;
                if (control.steering == 0.0 && !v.center_steering) target = current;
                car.wheel_angle[i] = (float)(current + (target - current) * (v.steering_speed * steering_multiplier));
            }
        }
        impulse = impulse + (old_velocity * 0.005f) * (-v.air_resistance);
    }
    Vec2 velocity = old_velocity + impulse * (1.0f / v.mass);
    if (driving && (double)length(old_velocity) > v.max_velocity) velocity = normalized(old_velocity) * v.max_velocity_f;
    Vec2 previous_velocity = velocity;
    float previous_angular = car.angular_velocity + torque * (1.0f / v.inertia);
    float angular = previous_angular;
    if (!v.custom_integrator) {
        Vec2 zero = Vec2{0.0f, 0.0f};
        velocity = velocity * rmax(1.0f - v.linear_damp * v.dt_f, 0.0f)
            + (((v.gravity * v.mass + zero) + zero) * (1.0f / v.mass)) * v.dt_f;
        angular = previous_angular * rmax(1.0f - v.angular_damp * v.dt_f, 0.0f);
    }
    STEP_SECTION(0, tc);
    car.error |= err;
    prepare_contacts(w, v, car);
    return StepCarry{old_velocity, velocity, previous_velocity, angular, previous_angular, was_active, 0};
}

// The rest of the step with `push_of(ci)`, the result of `sat_pair` for
// broad pair ci. Returns the reported wall contact.
template <class PushOf>
__device__ inline bool step_end(const World& w, const VehicleDesc& v, Car& car, const StepCarry& k, PushOf&& push_of, bool drive,
                                bool eliminate_on_wall) {
    STEP_MARK(tc);
    uint32_t err = 0;
    static_assert(MAX_PAIRS <= 32, "pair mask");
    // The car frame (two normalizations) is only needed once a pair pushes;
    // most steps have no penetrating pair. The basis does not change here.
    CarFrame f;
    bool framed = false;
    uint32_t made_contacts = 0;
    for (uint32_t ci = 0; ci < car.pair_count; ci++) {
        Vec2 push = push_of(ci);
        if (is_zero(push)) continue;
        if (!framed) {
            f = car_frame(v, car);
            framed = true;
        }
        uint32_t made = pair_contacts(w, v, car, f, ci, push, [&](const Contact& c) {
            STEP_MARK(tm);
            merge_contact(v, car, c);
            STEP_SECTION(11, tm);
        });
        made_contacts |= made ? 1u << ci : 0u;
    }
    STEP_SECTION(2, tc);
    SolveResult r = solve_contacts(w, v, car, made_contacts, k.velocity, k.angular, k.previous_velocity, k.previous_angular);
    STEP_MARK(tr);
    car.velocity = r.velocity;
    car.angular_velocity = r.angular;
    car.position = Vec2{car.position.x + (r.velocity.x + r.bias_velocity.x) * v.dt_f,
                        car.position.y + (r.velocity.y + r.bias_velocity.y) * v.dt_f};
    float angle_delta = (r.angular + r.bias_angular) * v.dt_f;
    // apply_center_of_mass_displacement
    Vec2 center = car.basis_x * v.center_of_mass.x + car.basis_y * v.center_of_mass.y;
    if ((double)dot(center, center) > 1e-5 * 1e-5) {
        float s, c;
        engine_sin_cos(angle_delta, s, c, err);
        Vec2 turned = Vec2{center.x * c - center.y * s, center.x * s + center.y * c};
        car.position = car.position + (center - turned);
    }
    set_transform_angle(car, car.rotation + angle_delta, err);
    car.tick += drive ? 1 : 0;
    update_shape(v, car);

    // finish_callback (no pending transform reset; frozen cars are not modeled)
    bool pending_velocity = car.flags & CAR_PENDING_VELOCITY;
    bool report = r.collided && k.was_active && !pending_velocity;
    if (report) {
        car.collision_count++;
        if (eliminate_on_wall) {
            // set_active(false): reset_velocity((0, 0), 0)
            car.flags &= ~CAR_ACTIVE;
            car.reset_acceleration = car.acceleration;
            pending_velocity = true;
        }
    }
    if (pending_velocity) {
        car.velocity = Vec2{0.0f, 0.0f};
        car.angular_velocity = 0.0f;
    }
    car.flags &= ~(CAR_PENDING_VELOCITY | CAR_RESET_RIGHT);
    car.acceleration = car.velocity - k.old_velocity;
    car.error |= err;
    STEP_SECTION(5, tr);
    return report;
}

// `Car::step` (drive) or `Car::passive_step`, then `finish_callback`.
// Returns the reported wall contact.
__device__ inline bool car_step(const World& w, const VehicleDesc& v, Car& car, const double* raw, bool drive,
                                bool eliminate_on_wall) {
    StepCarry k = step_begin(w, v, car, raw, drive);
    return step_end(w, v, car, k, [&](uint32_t ci) { return sat_pair(w, v, car, ci); }, drive, eliminate_on_wall);
}

}  // namespace altd

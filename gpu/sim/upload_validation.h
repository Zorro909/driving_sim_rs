// Host-side validation before copying external state to fixed device arrays.
#pragma once
#include "state.h"
namespace altd {
inline bool valid_car_upload(const Car& c, uint32_t shapes, uint32_t surfaces, uint32_t wheels) {
    if (c.pair_count > MAX_PAIRS || c.contact_count > MAX_CONTACTS || wheels > MAX_WHEELS
        || (c.wheel_has_previous >> MAX_WHEELS) != 0) return false;
    for (uint32_t i = 0; i < c.pair_count; ++i) if (c.pairs[i] >= shapes) return false;
    for (uint32_t i = 0; i < c.contact_count; ++i) if (c.contacts[i].shape >= shapes) return false;
    for (uint32_t i = 0; i < wheels; ++i) if (c.wheel_surface[i] >= surfaces) return false;
    return true;
}
inline bool valid_agent_upload(const Agent& a) {
    return a.recent_len <= RECENT && a.recent_start < RECENT;
}
}

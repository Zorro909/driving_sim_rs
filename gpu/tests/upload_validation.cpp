// Run without a GPU: clang++ -std=c++17 -Wall -Wextra -Werror
// gpu/tests/upload_validation.cpp -o /tmp/altd-upload-validation && /tmp/altd-upload-validation
#include "../sim/upload_validation.h"
#include <cassert>
using namespace altd;
int main() {
    Car c{};
    assert(valid_car_upload(c, 1, 1, 4));
    c.pair_count = MAX_PAIRS + 1;
    assert(!valid_car_upload(c, 1, 1, 4));
    c = {}; c.contact_count = MAX_CONTACTS + 1;
    assert(!valid_car_upload(c, 1, 1, 4));
    c = {}; c.pair_count = 1; c.pairs[0] = 1;
    assert(!valid_car_upload(c, 1, 1, 4));
    assert(valid_car_upload(c, 2, 1, 4));
    c = {}; c.contact_count = 1; c.contacts[0].shape = 1;
    assert(!valid_car_upload(c, 1, 1, 4));
    c = {}; c.wheel_surface[0] = 1;
    assert(!valid_car_upload(c, 1, 1, 4));
    c = {}; c.wheel_has_previous = 1u << MAX_WHEELS;
    assert(!valid_car_upload(c, 1, 1, 4));
    c = {}; c.pairs[MAX_PAIRS-1] = UINT32_MAX;
    assert(valid_car_upload(c, 0, 1, 4));
    assert(!valid_car_upload(c, 0, 1, MAX_WHEELS + 1));
    Agent a{};
    assert(valid_agent_upload(a));
    a.recent_len = RECENT; a.recent_start = RECENT - 1;
    assert(valid_agent_upload(a));
    a.recent_len++;
    assert(!valid_agent_upload(a));
    a.recent_len = 0; a.recent_start = RECENT;
    assert(!valid_agent_upload(a));
}

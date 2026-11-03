#pragma once
#include <stdint.h>

// Range of a cell's points inside the sorted prim_ids array.
struct GridCell {
  uint32_t offset;
  uint32_t count;
};

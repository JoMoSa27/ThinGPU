#pragma once
#include <stdint.h>

// k-d tree node. A leaf (count > 0, split_dim = -1) owns prim_ids[offset, offset + count);
// an inner node (count = 0) has its children at nodes[offset] and nodes[offset + 1].
// count is 32-bit because a leaf of identical points can be very large.
template<int Dim>
struct __align__(32) KDNode {
    float    split_pos;
    int16_t  split_dim;
    uint32_t count;
    uint32_t offset;
    float    bb_min[Dim];
    float    bb_max[Dim];
};

using KDNode2D = KDNode<2>;

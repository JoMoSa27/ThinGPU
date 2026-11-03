#pragma once
#include <stdint.h>
#include "KDNode.cuh"
#include "SpatialTags.cuh"
#include "SpatialContract.cuh"

// Temporary device memory needed by build_kdtree_gpu for n_prims points.
template<int Dim>
size_t kd_scratch_bytes(uint32_t n_prims);

// Builds a k-d tree breadth first on the GPU, splitting each node at the midpoint
// of its widest axis, then groups points by leaf with a radix sort. d_nodes needs
// room for 4n + 4 nodes; num_nodes returns the number used.
template<int Dim>
void build_kdtree_gpu(
    KDNode<Dim>*   d_nodes,
    uint32_t*      d_prim_ids,
    CoordPtrs<Dim> coords,
    uint32_t       n_prims,
    uint32_t&      num_nodes,
    void*          scratch,
    cudaStream_t   stream = 0);

// 2D convenience overload that allocates its own scratch memory.
inline void build_kdtree_gpu(
    KDNode<2>*   d_nodes,
    uint32_t*    d_prim_ids,
    const float* d_c1,
    const float* d_c2,
    uint32_t     n_prims,
    uint32_t&    num_nodes)
{
    void* s = nullptr;
    cudaMalloc(&s, kd_scratch_bytes<2>(n_prims));
    build_kdtree_gpu<2>(d_nodes, d_prim_ids, CoordPtrs<2>{d_c1, d_c2},
                        n_prims, num_nodes, s);
    cudaFree(s);
}

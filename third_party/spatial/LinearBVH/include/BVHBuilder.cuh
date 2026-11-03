#pragma once
#include "BVHNode.cuh"
#include "SpatialTags.cuh"
#include "SpatialContract.cuh"

// Maps coordinates to [0, 1] before computing Morton codes.
template<int Dim>
struct MortonNorm {
    float mn[Dim];
    float inv[Dim];
};

// Temporary device memory needed by build_bvh_gpu for n points.
template<int Dim>
size_t bvh_scratch_bytes(uint32_t n);

// Builds a linear BVH over n points inside [bmin, bmax]: points are sorted by
// Morton code and placed as leaves of an implicit binary heap of 2n - 1 nodes.
template<int Dim>
void build_bvh_gpu(
    BVHNode<Dim>*  d_nodes,
    CoordPtrs<Dim> coords,
    uint32_t       n,
    const float*   bmin,
    const float*   bmax,
    void*          scratch,
    cudaStream_t   stream = 0);

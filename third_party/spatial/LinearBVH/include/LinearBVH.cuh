#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include "Metric.cuh"
#include "SpatialTags.cuh"
#include "BVHNode.cuh"
#include "BVHBuilder.cuh"
#include "BBoxGpu.cuh"

// GPU linear BVH with one point per leaf (see BVHNode.cuh for the layout).
// Same build and query interface as KDTree, plus ray_trace.
template<typename Metric = CartesianMetric, int Dim = 2>
struct LinearBVH : TagDeviceNN, TagDeviceRange, TagDeviceCount, TagDeviceRay, TagMetric {

    static constexpr int STACK_MAX = 64;

    BVHNode<Dim>* d_nodes         = nullptr;
    const float*  d_coords[Dim]   = {};
    uint32_t      n_points        = 0;
    uint32_t      n_nodes         = 0;
    Metric        metric          = {};

    bool owns_coords = false;
    bool owns_nodes  = false;

    // Device memory needed for n points: 2n - 1 nodes and the build scratch.
    static SpatialSizes reserve(uint32_t n) {
        SpatialSizes s{};
        if (n == 0) return s;
        s.nodes    = (2ull * n - 1ull) * sizeof(BVHNode<Dim>);
        s.prim_ids = 0;
        s.scratch  = bvh_scratch_bytes<Dim>(n);
        return s;
    }

    // Build on caller-owned buffers. The bounding box is computed on the GPU and
    // widened slightly in flat dimensions.
    void build_device(CoordPtrs<Dim> d_in, uint32_t n,
                      BVHNode<Dim>* nodes, void* scratch,
                      cudaStream_t stream, Metric m = {}) {
        if (n == 0) return;
        metric   = m;
        n_points = n;
        n_nodes  = 2 * n - 1;
        d_nodes  = nodes;

        for (int d = 0; d < Dim; d++) d_coords[d] = d_in.c[d];
        owns_coords = false;
        owns_nodes  = false;

        float mn[Dim], mx[Dim];
        compute_bbox_gpu<Dim>(d_in, n, mn, mx);

        const float eps = 1e-6f;
        for (int d = 0; d < Dim; d++)
            if (mx[d] - mn[d] < eps) mx[d] = mn[d] + eps;

        build_bvh_gpu<Dim>(d_nodes, d_in, n, mn, mx, scratch, stream);
    }

    // Build on device coordinates, allocating the nodes (freed by clear()).
    void build_device(CoordPtrs<Dim> d_in, uint32_t n, Metric m = {}) {
        if (n == 0) return;

        const SpatialSizes s = reserve(n);
        BVHNode<Dim>* nodes = nullptr;
        void*         scratch = nullptr;
        cudaMalloc(&nodes,   s.nodes);
        cudaMalloc(&scratch, s.scratch);

        build_device(d_in, n, nodes, scratch, 0, m);

        cudaFree(scratch);
        owns_nodes = true;
    }

    // Build from host coordinates, which are copied to the device.
    void build(CoordPtrs<Dim> coords, uint32_t n, Metric m = {}) {
        if (n == 0) return;

        CoordPtrs<Dim> d_in;
        for (int d = 0; d < Dim; d++) {
            float* p = nullptr;
            cudaMalloc(&p, n * sizeof(float));
            cudaMemcpy(p, coords.c[d], n * sizeof(float), cudaMemcpyHostToDevice);
            d_in.c[d] = p;
        }
        build_device(d_in, n, m);
        owns_coords = true;
    }

    template<int D = Dim, typename = std::enable_if_t<D == 2>>
    void build(const float* c1, const float* c2, uint32_t n, Metric m = {}) {
        build(CoordPtrs<2>{c1, c2}, n, m);
    }

    void clear() {
        if (owns_nodes && d_nodes) cudaFree(d_nodes);
        d_nodes = nullptr;
        if (owns_coords)
            for (int d = 0; d < Dim; d++)
                if (d_coords[d]) cudaFree(const_cast<float*>(d_coords[d]));
        for (int d = 0; d < Dim; d++) d_coords[d] = nullptr;
        owns_coords = false;
        owns_nodes  = false;
        n_points = n_nodes = 0;
    }

    __host__ __device__ uint32_t size()  const { return n_points; }
    __host__ __device__ bool     empty() const { return n_points == 0; }

    __device__ void load_point(uint32_t idx, float* out) const {
        #pragma unroll
        for (int d = 0; d < Dim; d++) out[d] = d_coords[d][idx];
    }

    // Closest point to q; visits the nearer child first and skips the farther one
    // when its box is already beyond the best distance.
    __device__ uint32_t nearest_neighbor(const float* q, float& best_dist_sq) const
    {
        uint32_t best = 0;
        best_dist_sq  = 1e30f;

        int stack[STACK_MAX], top = 0;
        stack[top++] = 0;

        while (top > 0) {
            const BVHNode<Dim>& node = d_nodes[stack[--top]];

            if (metric.min_dist_sq(q, node.mn, node.mx) >= best_dist_sq)
                continue;

            if (node.leftChild == -1) {
                const uint32_t idx = (uint32_t)node.rightChild;
                float pt[Dim];
                load_point(idx, pt);
                const float d = metric.distance_sq(q, pt);
                if (d < best_dist_sq) { best_dist_sq = d; best = idx; }
            } else {

                const int L = node.leftChild;
                const int R = node.rightChild;
                const BVHNode<Dim>& nl = d_nodes[L];
                const BVHNode<Dim>& nr = d_nodes[R];
                const float dl = metric.min_dist_sq(q, nl.mn, nl.mx);
                const float dr = metric.min_dist_sq(q, nr.mn, nr.mx);

                const bool  l_near = (dl <= dr);
                const int   near   = l_near ? L : R;
                const int   far    = l_near ? R : L;
                const float d_far  = l_near ? dr : dl;

                if (top + 2 <= STACK_MAX) {
                    if (d_far < best_dist_sq) stack[top++] = far;
                    stack[top++] = near;
                }
            }
        }
        return best;
    }

    template<int D = Dim, typename = std::enable_if_t<D == 2>>
    __device__ uint32_t nearest_neighbor(float q1, float q2, float& best_dist_sq) const {
        const float q[2] = {q1, q2};
        return nearest_neighbor(q, best_dist_sq);
    }

    // Points with distance <= radius, collected into hits_out or only counted.
    // overflow is set if the traversal stack fills up.
    template<bool Collect>
    __device__ uint32_t range_impl(
        const float* q, float radius,
        uint32_t* hits_out, uint32_t max_results,
        bool* overflow = nullptr) const
    {
        uint32_t found = 0;
        const float r_sq = radius * radius;

        int stack[STACK_MAX], top = 0;
        stack[top++] = 0;

        while (top > 0) {
            if (Collect && found >= max_results) break;

            const BVHNode<Dim>& node = d_nodes[stack[--top]];

            if (metric.min_dist_sq(q, node.mn, node.mx) > r_sq)
                continue;

            if (node.leftChild == -1) {
                const uint32_t idx = (uint32_t)node.rightChild;
                float pt[Dim];
                load_point(idx, pt);
                if (metric.distance_sq(q, pt) <= r_sq) {
                    if constexpr (Collect) hits_out[found] = idx;
                    ++found;
                }
            } else {
                if (top + 2 <= STACK_MAX) {
                    stack[top++] = node.leftChild;
                    stack[top++] = node.rightChild;
                } else if (overflow) {
                    *overflow = true;
                }
            }
        }
        return found;
    }

    __device__ uint32_t range_query(
        const float* q, float radius,
        uint32_t* hits_out, uint32_t max_results,
        bool* overflow = nullptr) const
    {
        return range_impl<true>(q, radius, hits_out, max_results, overflow);
    }

    __device__ uint32_t count_in_radius(
        const float* q, float radius, bool* overflow = nullptr) const
    {
        return range_impl<false>(q, radius, nullptr, 0u, overflow);
    }

    template<int D = Dim, typename = std::enable_if_t<D == 2>>
    __device__ uint32_t count_in_radius(
        float q1, float q2, float radius, bool* overflow = nullptr) const
    {
        const float q[2] = {q1, q2};
        return count_in_radius(q, radius, overflow);
    }

    template<int D = Dim, typename = std::enable_if_t<D == 2>>
    __device__ uint32_t range_query(
        float q1, float q2, float radius,
        uint32_t* hits_out, uint32_t max_results) const
    {
        const float q[2] = {q1, q2};
        return range_query(q, radius, hits_out, max_results);
    }

    // First point within hit_radius of the ray ro + t * rd (t > 0); returns -1 if none.
    __device__ int ray_trace(
        const float* ro, const float* rd,
        float& closest_t, float hit_radius = 0.05f) const
    {
        int   stack[STACK_MAX], top = 0;
        float bestT  = 1e30f;
        int   bestPt = -1;
        stack[top++] = 0;

        float len_sq = 0.f;
        #pragma unroll
        for (int d = 0; d < Dim; d++) len_sq += rd[d] * rd[d];

        while (top > 0) {
            const BVHNode<Dim>& node = d_nodes[stack[--top]];

            float tMin = 0.f, tMax = 1e30f;
            if (!rayAABBIntersect<Dim>(ro, rd, node.mn, node.mx, tMin, tMax))
                continue;
            if (tMin > bestT) continue;

            if (node.leftChild == -1) {
                const uint32_t idx = (uint32_t)node.rightChild;
                if (len_sq > 0.f) {
                    float pt[Dim];
                    load_point(idx, pt);

                    float dot = 0.f;
                    #pragma unroll
                    for (int d = 0; d < Dim; d++) dot += (pt[d] - ro[d]) * rd[d];
                    const float t = dot / len_sq;

                    if (t > 0.f && t < bestT) {

                        float perp_sq = 0.f;
                        #pragma unroll
                        for (int d = 0; d < Dim; d++) {
                            const float e = pt[d] - (ro[d] + t * rd[d]);
                            perp_sq += e * e;
                        }
                        if (perp_sq < hit_radius * hit_radius) {
                            bestT  = t;
                            bestPt = (int)idx;
                        }
                    }
                }
            } else {
                if (top + 2 <= STACK_MAX) {
                    stack[top++] = node.leftChild;
                    stack[top++] = node.rightChild;
                }
            }
        }
        closest_t = bestT;
        return bestPt;
    }

    template<int D = Dim, typename = std::enable_if_t<D == 2>>
    __device__ int ray_trace(
        float ro1, float ro2, float rd1, float rd2,
        float& closest_t, float hit_radius = 0.05f) const
    {
        const float ro[2] = {ro1, ro2};
        const float rd[2] = {rd1, rd2};
        return ray_trace(ro, rd, closest_t, hit_radius);
    }
};

template<typename M = CartesianMetric>
using LinearBVH2D = LinearBVH<M, 2>;

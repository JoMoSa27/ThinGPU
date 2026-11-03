#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include "Metric.cuh"
#include "KDBuilder.cuh"
#include "SpatialTags.cuh"
#include "SpatialContract.cuh"

// GPU k-d tree over Dim-dimensional points. The struct is passed by value to
// kernels; the device queries below run one query per thread with an explicit stack.
template<typename Metric = CartesianMetric, int Dim = 2>
struct KDTree : TagDeviceNN, TagDeviceRange, TagDeviceCount, TagMetric {

    KDNode<Dim>* d_nodes    = nullptr;
    uint32_t*    d_prim_ids = nullptr;
    uint32_t     n_nodes    = 0;
    uint32_t     n_points   = 0;
    Metric       metric     = {};
    float*       d_coords[Dim];

    bool owns_coords   = false;
    bool owns_nodes    = false;
    bool owns_prim_ids = false;

    KDTree() {
        for (int d = 0; d < Dim; d++) d_coords[d] = nullptr;
    }

    // Device memory needed for n points.
    static SpatialSizes reserve(uint32_t n) {
        SpatialSizes s{};
        if (n == 0) return s;
        const uint32_t max_nodes = 4 * n + 4;
        s.nodes    = (size_t)max_nodes * sizeof(KDNode<Dim>);
        s.prim_ids = (size_t)n         * sizeof(uint32_t);
        s.scratch  = kd_scratch_bytes<Dim>(n);
        return s;
    }

    // Build on caller-owned device buffers sized with reserve(n). Coordinates stay
    // owned by the caller and must outlive the tree.
    void build_device(CoordPtrs<Dim> d_in, uint32_t n,
                      KDNode<Dim>* nodes, uint32_t* prim_ids,
                      void* scratch, cudaStream_t stream, Metric m = {})
    {
        if (n == 0) return;
        metric     = m;
        n_points   = n;
        d_nodes    = nodes;
        d_prim_ids = prim_ids;
        for (int d = 0; d < Dim; d++)
            d_coords[d] = const_cast<float*>(d_in.c[d]);
        owns_coords   = false;
        owns_nodes    = false;
        owns_prim_ids = false;
        build_kdtree_gpu<Dim>(d_nodes, d_prim_ids, d_in, n, n_nodes, scratch, stream);
    }

    // Build on device coordinates, allocating nodes and ids (freed by clear()).
    void build_device(CoordPtrs<Dim> d_in, uint32_t n, Metric m = {}) {
        if (n == 0) return;
        const SpatialSizes s = reserve(n);
        KDNode<Dim>* nodes    = nullptr;
        uint32_t*    prim_ids = nullptr;
        void*        scratch  = nullptr;
        cudaMalloc(&nodes,    s.nodes);
        cudaMalloc(&prim_ids, s.prim_ids);
        cudaMalloc(&scratch,  s.scratch);
        build_device(d_in, n, nodes, prim_ids, scratch, 0, m);
        cudaFree(scratch);

        cudaStreamSynchronize(0);
        owns_nodes    = true;
        owns_prim_ids = true;
    }

    // Build from host coordinates, which are copied to the device.
    void build(CoordPtrs<Dim> coords, uint32_t n, Metric m = {}) {
        if (n == 0) return;

        CoordPtrs<Dim> d_in;
        for (int d = 0; d < Dim; d++) {
            float* p = nullptr;
            cudaMalloc(&p, n * sizeof(float));
            cudaMemcpy(p, coords.c[d], n * sizeof(float),
                       cudaMemcpyHostToDevice);
            d_in.c[d] = p;
        }

        build_device(d_in, n, m);
        owns_coords = true;
    }

    void build(const float* c1, const float* c2, uint32_t n, Metric m = {}) {
        static_assert(Dim == 2, "2D overload only for KDTree<M, 2> (KDTree2D)");
        build(CoordPtrs<2>{c1, c2}, n, m);
    }

    // Frees only the buffers the tree allocated itself.
    void clear() {
        if (owns_coords) {
            for (int d = 0; d < Dim; d++)
                if (d_coords[d]) cudaFree(d_coords[d]);
        }
        for (int d = 0; d < Dim; d++) d_coords[d] = nullptr;
        owns_coords = false;

        if (owns_nodes    && d_nodes)    cudaFree(d_nodes);
        if (owns_prim_ids && d_prim_ids) cudaFree(d_prim_ids);
        d_nodes = nullptr; d_prim_ids = nullptr;
        owns_nodes = owns_prim_ids = false;
        n_nodes = n_points = 0;
    }

    // Index of the closest point to q; best_dist_sq returns its squared distance.
    // The near child is pushed last so it is visited first.
    __device__ uint32_t nearest_neighbor(
        const float* q, float& best_dist_sq) const
    {
        uint32_t best_hit = 0;
        best_dist_sq      = 1e30f;

        uint32_t stack[64];
        int      top = 0;
        stack[top++] = 0;

        while (top > 0) {
            const uint32_t idx = stack[--top];
            if (idx >= n_nodes) continue;

            const KDNode<Dim>& node = d_nodes[idx];

            if (metric.min_dist_sq(q, node.bb_min, node.bb_max) >= best_dist_sq)
                continue;

            if (node.split_dim < 0) {

                for (uint32_t i = 0; i < node.count; i++) {
                    const uint32_t hit = d_prim_ids[node.offset + i];
                    float pt[Dim];
                    for (int d = 0; d < Dim; d++) pt[d] = d_coords[d][hit];
                    const float dist = metric.distance_sq(q, pt);
                    if (dist < best_dist_sq) {
                        best_dist_sq = dist;
                        best_hit     = hit;
                    }
                }
            } else {

                const float q_dim    = q[node.split_dim];
                const bool  go_left  = (q_dim < node.split_pos);
                const uint32_t near  = node.offset + (go_left ? 0 : 1);
                const uint32_t far   = node.offset + (go_left ? 1 : 0);
                if (far  < n_nodes) stack[top++] = far;
                if (near < n_nodes) stack[top++] = near;
            }
        }

        return best_hit;
    }

    __device__ uint32_t nearest_neighbor(
        float q1, float q2, float& best_dist_sq) const
    {
        static_assert(Dim == 2, "2D overload only for KDTree<M, 2>");
        const float q[2] = {q1, q2};
        return nearest_neighbor(q, best_dist_sq);
    }

    // Points with distance <= radius. With Collect, ids are written to hits_out
    // (at most max_results); otherwise they are only counted. overflow is set
    // if the traversal stack fills up, in which case the result is incomplete.
    template<bool Collect>
    __device__ uint32_t range_impl(
        const float* q, float radius,
        uint32_t* hits_out, uint32_t max_results,
        bool* overflow = nullptr) const
    {
        uint32_t    found = 0;
        const float r_sq  = radius * radius;

        constexpr int STACK_MAX = 64;
        uint32_t stack[STACK_MAX];
        int      top = 0;
        stack[top++] = 0;

        while (top > 0) {
            if (Collect && found >= max_results) break;

            const uint32_t idx = stack[--top];
            if (idx >= n_nodes) continue;

            const KDNode<Dim>& node = d_nodes[idx];

            if (metric.min_dist_sq(q, node.bb_min, node.bb_max) > r_sq)
                continue;

            if (node.split_dim < 0) {
                for (uint32_t i = 0; i < node.count; i++) {
                    if (Collect && found >= max_results) break;
                    const uint32_t hit = d_prim_ids[node.offset + i];
                    float pt[Dim];
                    for (int d = 0; d < Dim; d++) pt[d] = d_coords[d][hit];
                    if (metric.distance_sq(q, pt) <= r_sq) {
                        if constexpr (Collect) hits_out[found] = hit;
                        ++found;
                    }
                }
            } else {

                if (top + 2 > STACK_MAX) {
                    if (overflow) *overflow = true;
                    continue;
                }
                if (node.offset     < n_nodes) stack[top++] = node.offset;
                if (node.offset + 1 < n_nodes) stack[top++] = node.offset + 1;
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

    __device__ uint32_t count_in_radius(
        float q1, float q2, float radius, bool* overflow = nullptr) const
    {
        static_assert(Dim == 2, "2D overload only for KDTree<M, 2>");
        const float q[2] = {q1, q2};
        return count_in_radius(q, radius, overflow);
    }

    __device__ uint32_t range_query(
        float q1, float q2, float radius,
        uint32_t* hits_out, uint32_t max_results) const
    {
        static_assert(Dim == 2, "2D overload only for KDTree<M, 2>");
        const float q[2] = {q1, q2};
        return range_query(q, radius, hits_out, max_results);
    }

    __host__ __device__ uint32_t size()  const { return n_points; }
    __host__ __device__ bool     empty() const { return n_points == 0; }
};

template<typename M = CartesianMetric>
using KDTree2D = KDTree<M, 2>;

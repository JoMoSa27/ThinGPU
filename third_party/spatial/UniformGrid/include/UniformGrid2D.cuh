#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "Metric.cuh"
#include "GridCell.cuh"
#include "GridBuilder.cuh"
#include "SpatialTags.cuh"
#include "BBoxGpu.cuh"
#include "SpatialContract.cuh"

// Optional build timing, enabled with the GRID_PROFILE environment variable.
namespace ug_profile {

inline bool on() {
    static const bool v = (getenv("GRID_PROFILE") != nullptr);
    return v;
}
inline std::chrono::high_resolution_clock::time_point tic() {
    if (on()) cudaDeviceSynchronize();
    return std::chrono::high_resolution_clock::now();
}
inline void toc(std::chrono::high_resolution_clock::time_point t0, const char* label) {
    if (!on()) return;
    cudaDeviceSynchronize();
    printf("    [grid-profile] %-22s %8.3f ms\n", label,
           std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now() - t0).count());
}

}

// GPU uniform grid: points sorted by cell, each cell a range of d_prim_ids.
// Fast on evenly spread data; cells over dense clusters become long lists.
template<typename Metric = CartesianMetric, int Dim = 2>
struct UniformGrid : TagDeviceNN, TagDeviceRange, TagDeviceCount, TagDeviceAnnular, TagMetric {

    GridCell* d_cells    = nullptr;
    uint32_t* d_prim_ids = nullptr;
    float*    d_coords[Dim];

    uint32_t n_points = 0;
    GridParams<Dim> gp = {};
    Metric   metric   = {};

    bool owns_coords   = false;
    bool owns_cells    = false;
    bool owns_prim_ids = false;

    UniformGrid() {
        for (int d = 0; d < Dim; d++) d_coords[d] = nullptr;
    }

    // Upper bound for the automatic cell size (about n / 8 cells).
    static SpatialSizes reserve(uint32_t n) {
        SpatialSizes s{};
        if (n == 0) return s;
        s.nodes    = (size_t)n * sizeof(GridCell);
        s.prim_ids = (size_t)n * sizeof(uint32_t);
        s.scratch  = grid_scratch_bytes(n);
        return s;
    }

    // Build on caller-owned buffers with about 8 points per cell on average.
    void build_device(CoordPtrs<Dim> d_in, uint32_t n,
                      GridCell* cells, uint32_t* prim_ids,
                      void* scratch, cudaStream_t stream, Metric m = {})
    {
        if (n == 0) return;
        float mn[Dim], mx[Dim];
        compute_bbox_gpu<Dim>(d_in, n, mn, mx);

        const float cells_per_axis = std::max(1.f,
            powf((float)n / 8.f, 1.f / Dim));
        GridParams<Dim> p;
        for (int d = 0; d < Dim; d++) {
            const float sz = (mx[d] - mn[d]) / cells_per_axis;
            p.cell_sz[d]  = sz > 0.f ? sz : 1.f;
            p.mn[d]       = mn[d];
            p.n_cells[d]  = (uint32_t)((mx[d] - mn[d]) / p.cell_sz[d]) + 1;
        }
        metric   = m;
        n_points = n;
        gp       = p;
        d_cells    = cells;
        d_prim_ids = prim_ids;
        for (int d = 0; d < Dim; d++)
            d_coords[d] = const_cast<float*>(d_in.c[d]);
        owns_coords = owns_cells = owns_prim_ids = false;
        build_grid_gpu<Dim>(d_cells, d_prim_ids, d_in, n, p, scratch, stream);
    }

    // Same cell size rule, allocating the buffers (freed by clear()).
    void build_device(CoordPtrs<Dim> d_in, uint32_t n, Metric m = {}) {
        if (n == 0) return;

        float mn[Dim], mx[Dim];
        auto _t = ug_profile::tic();
        compute_bbox_gpu<Dim>(d_in, n, mn, mx);
        ug_profile::toc(_t, "compute_bbox_gpu");

        const float cells_per_axis = std::max(1.f,
            powf((float)n / 8.f, 1.f / Dim));

        GridParams<Dim> p;
        for (int d = 0; d < Dim; d++) {
            const float sz = (mx[d] - mn[d]) / cells_per_axis;
            p.cell_sz[d]  = sz > 0.f ? sz : 1.f;
            p.mn[d]       = mn[d];
            p.n_cells[d]  = (uint32_t)((mx[d] - mn[d]) / p.cell_sz[d]) + 1;
        }

        _build_internal(d_in, n, p, m);
    }

    // Build with a given cell size.
    void build_device(CoordPtrs<Dim> d_in, uint32_t n,
                      const float cell_sz[Dim], Metric m = {}) {
        if (n == 0) return;

        float mn[Dim], mx[Dim];
        compute_bbox_gpu<Dim>(d_in, n, mn, mx);

        GridParams<Dim> p;
        for (int d = 0; d < Dim; d++) {
            p.cell_sz[d] = cell_sz[d];
            p.mn[d]      = mn[d];
            p.n_cells[d] = (uint32_t)((mx[d] - mn[d]) / cell_sz[d]) + 1;
        }
        _build_internal(d_in, n, p, m);
    }

    // Build on caller-owned buffers with explicit grid parameters; cells must hold
    // the product of p.n_cells.
    void build_device(CoordPtrs<Dim> d_in, uint32_t n,
                      GridCell* cells, uint32_t* prim_ids,
                      GridParams<Dim> p,
                      void* scratch, cudaStream_t stream, Metric m = {}) {
        if (n == 0) return;
        metric     = m;
        n_points   = n;
        gp         = p;
        d_cells    = cells;
        d_prim_ids = prim_ids;
        for (int d = 0; d < Dim; d++)
            d_coords[d] = const_cast<float*>(d_in.c[d]);
        owns_coords = owns_cells = owns_prim_ids = false;
        build_grid_gpu<Dim>(d_cells, d_prim_ids, d_in, n, p, scratch, stream);
    }

    // Builds from host coordinates, which are copied to the device.
    void build(CoordPtrs<Dim> h_coords, uint32_t n, Metric m = {}) {
        if (n == 0) return;
        CoordPtrs<Dim> d_in = _upload(h_coords, n);
        build_device(d_in, n, m);
        owns_coords = true;
    }

    void build(CoordPtrs<Dim> h_coords, uint32_t n,
               const float cell_sz[Dim], Metric m = {}) {
        if (n == 0) return;
        CoordPtrs<Dim> d_in = _upload(h_coords, n);
        build_device(d_in, n, cell_sz, m);
        owns_coords = true;
    }

    void build(const float* c1, const float* c2, uint32_t n, Metric m = {}) {
        static_assert(Dim == 2, "2D overload only for UniformGrid<M, 2>");
        build(CoordPtrs<2>{c1, c2}, n, m);
    }

    void build(const float* c1, const float* c2, uint32_t n,
               float cell_sz1, float cell_sz2, Metric m = {}) {
        static_assert(Dim == 2, "2D overload only for UniformGrid<M, 2>");
        const float sz[2] = {cell_sz1, cell_sz2};
        build(CoordPtrs<2>{c1, c2}, n, sz, m);
    }

    void clear() {
        if (owns_coords) {
            for (int d = 0; d < Dim; d++)
                if (d_coords[d]) cudaFree(d_coords[d]);
        }
        for (int d = 0; d < Dim; d++) d_coords[d] = nullptr;
        owns_coords = false;

        if (owns_cells    && d_cells)    cudaFree(d_cells);
        if (owns_prim_ids && d_prim_ids) cudaFree(d_prim_ids);
        d_cells = nullptr; d_prim_ids = nullptr;
        owns_cells = owns_prim_ids = false;
        n_points = 0;
        gp = {};
    }

    // Row-major cell of q, clamped to the grid.
    __device__ uint32_t cell_of(const float* q) const {
        uint32_t cell = 0, stride = 1;
        for (int d = 0; d < Dim; d++) {
            int idx = (int)floorf((q[d] - gp.mn[d]) / gp.cell_sz[d]);
            idx = max(0, min(idx, (int)gp.n_cells[d] - 1));
            cell  += (uint32_t)idx * stride;
            stride *= gp.n_cells[d];
        }
        return cell;
    }

    // Closest point to q, searching rings of cells outwards from q's cell and
    // stopping when the next ring cannot hold anything closer.
    __device__ uint32_t nearest_neighbor(
        const float* q, float& best_dist_sq) const
    {
        uint32_t best_hit = 0;
        best_dist_sq = 1e30f;

        int cx[Dim];
        for (int d = 0; d < Dim; d++) {
            cx[d] = (int)floorf((q[d] - gp.mn[d]) / gp.cell_sz[d]);
            cx[d] = max(0, min(cx[d], (int)gp.n_cells[d] - 1));
        }

        int max_ring = 0;
        for (int d = 0; d < Dim; d++)
            max_ring = max(max_ring, (int)gp.n_cells[d]);

        for (int ring = 0; ring <= max_ring; ring++) {

            if (best_dist_sq < 1e30f) {
                float min_cell_sz = gp.cell_sz[0];
                for (int d = 1; d < Dim; d++)
                    min_cell_sz = fminf(min_cell_sz, gp.cell_sz[d]);
                const float ring_dist = (ring - 1) * min_cell_sz;
                if (ring_dist > 0.f && ring_dist * ring_dist >= best_dist_sq) break;
            }

            // Visit only the shell of the (2 ring + 1)^Dim block of cells.
            uint32_t box_size = 1;
            for (int d = 0; d < Dim; d++) box_size *= (2 * ring + 1);

            for (uint32_t flat = 0; flat < box_size; flat++) {
                int idx[Dim];
                bool on_shell = false;
                uint32_t tmp = flat;
                for (int d = 0; d < Dim; d++) {
                    int local = (int)(tmp % (uint32_t)(2 * ring + 1));
                    tmp /= (2 * ring + 1);
                    idx[d] = cx[d] - ring + local;

                    if (local == 0 || local == 2 * ring) on_shell = true;
                }

                if (ring > 0 && !on_shell) continue;

                bool valid = true;
                uint32_t cell_id = 0, stride = 1;
                for (int d = 0; d < Dim; d++) {
                    if (idx[d] < 0 || idx[d] >= (int)gp.n_cells[d]) { valid = false; break; }
                    cell_id += (uint32_t)idx[d] * stride;
                    stride  *= gp.n_cells[d];
                }
                if (!valid) continue;

                const GridCell& cell = d_cells[cell_id];
                for (uint32_t k = 0; k < cell.count; k++) {
                    const uint32_t hit = d_prim_ids[cell.offset + k];
                    float pt[Dim];
                    for (int d = 0; d < Dim; d++) pt[d] = d_coords[d][hit];
                    const float dist = metric.distance_sq(q, pt);
                    if (dist < best_dist_sq) { best_dist_sq = dist; best_hit = hit; }
                }
            }
        }
        return best_hit;
    }

    __device__ uint32_t nearest_neighbor(
        float q1, float q2, float& best_dist_sq) const
    {
        static_assert(Dim == 2, "2D overload only for UniformGrid<M, 2>");
        const float q[2] = {q1, q2};
        return nearest_neighbor(q, best_dist_sq);
    }

    // Points with distance <= radius among the cells overlapping the query box.
    // overflow is set (and 0 returned) if that box has more than 2^32 cells.
    template<bool Collect>
    __device__ uint32_t range_impl(
        const float* q, float radius,
        uint32_t* hits_out, uint32_t max_results,
        bool* overflow = nullptr) const
    {
        uint32_t found = 0;
        const float r_sq = radius * radius;

        int lo[Dim], hi[Dim];
        for (int d = 0; d < Dim; d++) {
            lo[d] = max(0, (int)floorf((q[d]-radius - gp.mn[d]) / gp.cell_sz[d]));
            hi[d] = min((int)gp.n_cells[d]-1,
                        (int)floorf((q[d]+radius - gp.mn[d]) / gp.cell_sz[d]));

            if (hi[d] < lo[d]) return 0;
        }

        unsigned long long box_size = 1ull;
        for (int d = 0; d < Dim; d++)
            box_size *= (unsigned long long)(hi[d] - lo[d] + 1);

        if (box_size > 0xFFFFFFFFull) {
            if (overflow) *overflow = true;
            return 0;
        }

        for (unsigned long long flat = 0; flat < box_size; flat++) {
            if (Collect && found >= max_results) break;

            uint32_t cell_id = 0, stride = 1;
            unsigned long long tmp = flat;
            for (int d = 0; d < Dim; d++) {
                const uint32_t span  = (uint32_t)(hi[d] - lo[d] + 1);
                const uint32_t local = (uint32_t)(tmp % span);
                tmp /= span;
                cell_id += (uint32_t)(lo[d] + local) * stride;
                stride  *= gp.n_cells[d];
            }
            const GridCell& cell = d_cells[cell_id];
            for (uint32_t k = 0; k < cell.count; k++) {
                if (Collect && found >= max_results) break;
                const uint32_t hit = d_prim_ids[cell.offset + k];
                float pt[Dim];
                for (int d = 0; d < Dim; d++) pt[d] = d_coords[d][hit];
                if (metric.distance_sq(q, pt) <= r_sq) {
                    if constexpr (Collect) hits_out[found] = hit;
                    ++found;
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

    __device__ uint32_t count_in_radius(
        float q1, float q2, float radius, bool* overflow = nullptr) const
    {
        static_assert(Dim == 2, "2D overload only for UniformGrid<M, 2>");
        const float q[2] = {q1, q2};
        return count_in_radius(q, radius, overflow);
    }

    __device__ uint32_t range_query(
        float q1, float q2, float radius,
        uint32_t* hits_out, uint32_t max_results) const
    {
        static_assert(Dim == 2, "2D overload only for UniformGrid<M, 2>");
        const float q[2] = {q1, q2};
        return range_query(q, radius, hits_out, max_results);
    }

    // Points with r_inner <= distance <= r_outer, at most max_results.
    __device__ uint32_t annular_query(
        const float* q, float r_inner, float r_outer,
        uint32_t* hits_out, uint32_t max_results) const
    {
        uint32_t found = 0;
        const float r_in_sq  = r_inner * r_inner;
        const float r_out_sq = r_outer * r_outer;

        int lo[Dim], hi[Dim];
        for (int d = 0; d < Dim; d++) {
            lo[d] = max(0, (int)floorf((q[d]-r_outer - gp.mn[d]) / gp.cell_sz[d]));
            hi[d] = min((int)gp.n_cells[d]-1,
                        (int)floorf((q[d]+r_outer - gp.mn[d]) / gp.cell_sz[d]));
        }

        uint32_t box_size = 1;
        for (int d = 0; d < Dim; d++) box_size *= (uint32_t)(hi[d] - lo[d] + 1);

        for (uint32_t flat = 0; flat < box_size && found < max_results; flat++) {
            uint32_t cell_id = 0, stride = 1, tmp = flat;
            for (int d = 0; d < Dim; d++) {
                int local = (int)(tmp % (uint32_t)(hi[d] - lo[d] + 1));
                tmp /= (hi[d] - lo[d] + 1);
                cell_id += (uint32_t)(lo[d] + local) * stride;
                stride  *= gp.n_cells[d];
            }
            const GridCell& cell = d_cells[cell_id];
            for (uint32_t k = 0; k < cell.count && found < max_results; k++) {
                const uint32_t hit = d_prim_ids[cell.offset + k];
                float pt[Dim];
                for (int d = 0; d < Dim; d++) pt[d] = d_coords[d][hit];
                const float dist = metric.distance_sq(q, pt);
                if (dist >= r_in_sq && dist <= r_out_sq)
                    hits_out[found++] = hit;
            }
        }
        return found;
    }

    __device__ uint32_t annular_query(
        float q1, float q2, float r_inner, float r_outer,
        uint32_t* hits_out, uint32_t max_results) const
    {
        static_assert(Dim == 2, "2D overload only for UniformGrid<M, 2>");
        const float q[2] = {q1, q2};
        return annular_query(q, r_inner, r_outer, hits_out, max_results);
    }

    __host__ __device__ uint32_t size()  const { return n_points; }
    __host__ __device__ bool     empty() const { return n_points == 0; }

private:

    CoordPtrs<Dim> _upload(CoordPtrs<Dim> h_coords, uint32_t n) {
        CoordPtrs<Dim> d_in;
        for (int d = 0; d < Dim; d++) {
            float* p = nullptr;
            cudaMalloc(&p, n * sizeof(float));
            cudaMemcpy(p, h_coords.c[d], n * sizeof(float),
                       cudaMemcpyHostToDevice);
            d_in.c[d] = p;
        }
        return d_in;
    }

    void _build_internal(CoordPtrs<Dim> d_in, uint32_t n,
                         GridParams<Dim> p, Metric m)
    {
        if (d_cells) clear();
        metric   = m;
        n_points = n;
        gp       = p;

        uint32_t n_cells = 1;
        for (int d = 0; d < Dim; d++) n_cells *= p.n_cells[d];

        for (int d = 0; d < Dim; d++)
            d_coords[d] = const_cast<float*>(d_in.c[d]);
        owns_coords = false;

        auto _t1 = ug_profile::tic();
        cudaMalloc(&d_cells,    n_cells * sizeof(GridCell));
        ug_profile::toc(_t1, "cudaMalloc d_cells");

        auto _t2 = ug_profile::tic();
        cudaMalloc(&d_prim_ids, n       * sizeof(uint32_t));
        ug_profile::toc(_t2, "cudaMalloc d_prim_ids");

        owns_cells    = true;
        owns_prim_ids = true;
        build_grid_gpu<Dim>(d_cells, d_prim_ids, d_in, n, p);
    }
};

template<typename M = CartesianMetric>
using UniformGrid2D = UniformGrid<M, 2>;

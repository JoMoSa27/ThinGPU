#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "../include/GridBuilder.cuh"
#include "SpatialContract.cuh"

namespace {

// Row-major cell index of a point; points outside the grid go to the border cells.
template<int Dim>
__host__ __device__ inline uint32_t compute_cell_idx(
    CoordPtrs<Dim> coords, uint32_t id, GridParams<Dim> gp)
{
    uint32_t cell = 0, stride = 1;
    for (int d = 0; d < Dim; d++) {
        int idx = (int)floorf((coords.c[d][id] - gp.mn[d]) / gp.cell_sz[d]);
        idx = max(0, min(idx, (int)gp.n_cells[d] - 1));
        cell  += (uint32_t)idx * stride;
        stride *= gp.n_cells[d];
    }
    return cell;
}

// Pass 1: number of points per cell.
template<int Dim>
__global__ void grid_count_k(
    uint32_t* cell_count,
    CoordPtrs<Dim> coords, uint32_t n_prims, GridParams<Dim> gp)
{
    const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n_prims) return;
    atomicAdd(&cell_count[compute_cell_idx<Dim>(coords, id, gp)], 1u);
}

// Pass 3: scatter point ids into their cell's range.
template<int Dim>
__global__ void grid_fill_k(
    uint32_t* prim_ids, uint32_t* slot_counter, const GridCell* cells,
    CoordPtrs<Dim> coords, uint32_t n_prims, GridParams<Dim> gp)
{
    const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n_prims) return;
    const uint32_t c    = compute_cell_idx<Dim>(coords, id, gp);
    const uint32_t slot = atomicAdd(&slot_counter[c], 1u);
    prim_ids[cells[c].offset + slot] = id;
}

// Pass 2 (after the scan): store each cell's offset and count.
__global__ void grid_write_cells_k(
    GridCell* cells, const uint32_t* cell_offset, const uint32_t* cell_count,
    uint32_t n_cells)
{
    const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n_cells) return;
    cells[id].offset = cell_offset[id];
    cells[id].count  = cell_count[id];
}

}

namespace {

// Optional per-phase timing, enabled with the GRID_PROFILE environment variable.
struct PhaseTimer {
    bool on;
    std::chrono::high_resolution_clock::time_point t0;

    PhaseTimer() : on(getenv("GRID_PROFILE") != nullptr) {}

    void tic() {
        if (!on) return;
        cudaDeviceSynchronize();
        t0 = std::chrono::high_resolution_clock::now();
    }
    void toc(const char* label) {
        if (!on) return;
        cudaDeviceSynchronize();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        printf("    [grid-profile] %-22s %8.3f ms\n", label, ms);
    }
};

}

namespace {

// Scratch buffers of the build, laid out by plan_grid_scratch.
struct GridBufs {
    uint32_t* cell_count;
    uint32_t* cell_offset;
    uint32_t* slot_counter;
    void*     cub_tmp;
    size_t    cub_bytes;
};

inline size_t cub_excl_sum_bytes(uint32_t n_cells)
{
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, bytes,
        (const uint32_t*)nullptr, (uint32_t*)nullptr,
        (int)n_cells);
    return bytes;
}

GridBufs plan_grid_scratch(ScratchLayout& L, uint32_t n_cells)
{
    GridBufs b;
    b.cell_count   = L.take<uint32_t>(n_cells);
    b.cell_offset  = L.take<uint32_t>(n_cells);
    b.slot_counter = L.take<uint32_t>(n_cells);
    b.cub_bytes    = cub_excl_sum_bytes(n_cells);
    b.cub_tmp      = L.take_bytes(b.cub_bytes);
    return b;
}

}

// Measures the scratch size with the same plan used by the build.
size_t grid_scratch_bytes(uint32_t n_cells)
{
    if (n_cells == 0) return 0;
    ScratchLayout sizer;
    plan_grid_scratch(sizer, n_cells);
    return sizer.total();
}

template<int Dim>
void build_grid_gpu(
    GridCell*       d_cells,
    uint32_t*       d_prim_ids,
    CoordPtrs<Dim>  coords,
    uint32_t        n_prims,
    GridParams<Dim> gp,
    void*           scratch,
    cudaStream_t    stream)
{
    uint32_t n_cells = 1;
    for (int d = 0; d < Dim; d++) n_cells *= gp.n_cells[d];

    const uint32_t block = 256;
    auto blk = [&](uint32_t n){ return (n + block - 1) / block; };

    ScratchLayout layout(scratch);
    GridBufs b = plan_grid_scratch(layout, n_cells);

    cudaMemsetAsync(b.cell_count,   0, n_cells * sizeof(uint32_t), stream);
    cudaMemsetAsync(b.slot_counter, 0, n_cells * sizeof(uint32_t), stream);

    grid_count_k<Dim><<<blk(n_prims), block, 0, stream>>>(b.cell_count, coords, n_prims, gp);

    cub::DeviceScan::ExclusiveSum(
        b.cub_tmp, b.cub_bytes,
        b.cell_count, b.cell_offset,
        (int)n_cells, stream);

    grid_write_cells_k<<<blk(n_cells), block, 0, stream>>>(
        d_cells, b.cell_offset, b.cell_count, n_cells);

    grid_fill_k<Dim><<<blk(n_prims), block, 0, stream>>>(
        d_prim_ids, b.slot_counter, d_cells, coords, n_prims, gp);

}

template void build_grid_gpu<2>(GridCell*, uint32_t*, CoordPtrs<2>, uint32_t, GridParams<2>, void*, cudaStream_t);
template void build_grid_gpu<3>(GridCell*, uint32_t*, CoordPtrs<3>, uint32_t, GridParams<3>, void*, cudaStream_t);

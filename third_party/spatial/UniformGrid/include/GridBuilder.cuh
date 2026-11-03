#pragma once
#include <stdint.h>
#include "GridCell.cuh"
#include "SpatialTags.cuh"
#include "SpatialContract.cuh"

// Grid shape: number of cells, origin and cell size along each dimension.
template<int Dim>
struct GridParams {
    uint32_t n_cells[Dim];
    float    mn[Dim];
    float    cell_sz[Dim];
};

// Temporary device memory needed by build_grid_gpu.
size_t grid_scratch_bytes(uint32_t n_cells_total);

// Sorts points by cell (count, scan, scatter) and fills the cell ranges.
template<int Dim>
void build_grid_gpu(
    GridCell*       d_cells,
    uint32_t*       d_prim_ids,
    CoordPtrs<Dim>  coords,
    uint32_t        n_prims,
    GridParams<Dim> gp,
    void*           scratch,
    cudaStream_t    stream = 0);

// Overload that allocates its own scratch memory.
template<int Dim>
inline void build_grid_gpu(
    GridCell*       d_cells,
    uint32_t*       d_prim_ids,
    CoordPtrs<Dim>  coords,
    uint32_t        n_prims,
    GridParams<Dim> gp)
{
    uint32_t n_cells = 1;
    for (int d = 0; d < Dim; d++) n_cells *= gp.n_cells[d];
    void* s = nullptr;
    cudaMalloc(&s, grid_scratch_bytes(n_cells));
    build_grid_gpu<Dim>(d_cells, d_prim_ids, coords, n_prims, gp, s);
    cudaFree(s);
}

// 2D overload with scalar parameters.
inline void build_grid_gpu(
    GridCell*    d_cells,     uint32_t*    d_prim_ids,
    const float* d_coords1,   const float* d_coords2,
    uint32_t n_prims,
    uint32_t nx,    uint32_t ny,
    float min1,     float min2,
    float cell_sz1, float cell_sz2)
{
    GridParams<2> gp;
    gp.n_cells[0] = nx;       gp.n_cells[1] = ny;
    gp.mn[0]      = min1;     gp.mn[1]      = min2;
    gp.cell_sz[0] = cell_sz1; gp.cell_sz[1] = cell_sz2;
    build_grid_gpu<2>(d_cells, d_prim_ids,
                      CoordPtrs<2>{d_coords1, d_coords2},
                      n_prims, gp);
}

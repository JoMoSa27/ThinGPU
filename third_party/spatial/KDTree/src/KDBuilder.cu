#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cmath>
#include <cstring>
#include "../include/KDBuilder.cuh"
#include "AtomicFloat.cuh"
#include "SpatialContract.cuh"

#define CUDA_CHECK_KERNEL() do { \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) \
        fprintf(stderr, "[KD] kernel error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
} while(0)

// k-d tree build, breadth first. Each level runs two kernels: kd_select_splits
// decides for every open node whether to split (midpoint of the widest axis) or
// become a leaf, and kd_update_prims moves every point to its child. Points are
// then grouped by leaf with a radix sort. No recursion and no allocation in kernels.

namespace {

// A point during the build: its original id and the node that currently owns it.
struct PrimState {
    uint32_t prim_id;
    uint32_t node_id;
    bool     done;
};

// OPEN_BRANCH: bbox and count being accumulated; OPEN_NODE: split chosen,
// points not yet routed; DONE_NODE: leaf or finished inner node.
enum NodePhase : int8_t { OPEN_BRANCH = 0, OPEN_NODE = 1, DONE_NODE = 2 };

// Bounding box updated with integer atomics on order-preserving float encodings.
template<int Dim>
struct AtomicBBox {
    int32_t mn[Dim], mx[Dim];

    __host__ __device__ static int32_t encode(float f)   { return atomic_float::encode(f); }
    __host__ __device__ static float   decode(int32_t b) { return atomic_float::decode(b); }

    __host__ __device__ void set_empty() {
        for (int d = 0; d < Dim; d++) {
            mn[d] = encode(atomic_float::EMPTY_MIN);
            mx[d] = encode(atomic_float::EMPTY_MAX);
        }
    }

    __device__ void grow(CoordPtrs<Dim> coords, uint32_t id) {
        for (int d = 0; d < Dim; d++) {
            const int32_t e = encode(coords.c[d][id]);
            atomicMin(&mn[d], e);
            atomicMax(&mx[d], e);
        }
    }

    __host__ __device__ float get_min(int d) const { return decode(mn[d]); }
    __host__ __device__ float get_max(int d) const { return decode(mx[d]); }
};

// Mutable node state during the build, converted to KDNode at the end.
template<int Dim>
struct TempNode {
    AtomicBBox<Dim> bbox;
    uint32_t        count;
    float           split_pos;
    int             split_dim;
    uint32_t        child_offset;

    uint32_t        depth;
};

struct BuildState { uint32_t num_nodes; };

// Nodes with this many points or fewer become leaves.
constexpr unsigned LEAF_THRESHOLD = 8;

// Depth cap, well below the 64-entry traversal stacks of the queries.
constexpr uint32_t MAX_DEPTH = 48;

// Root at slot 0; slot 1 is left unused so children always come in pairs. Single thread.
template<int Dim>
__global__ void kd_init_state(
    BuildState* build_state, NodePhase* node_phases, TempNode<Dim>* temp_nodes)
{
    build_state->num_nodes      = 2;
    node_phases[0]              = OPEN_BRANCH;
    temp_nodes[0].count         = 0;
    temp_nodes[0].depth         = 0;
    temp_nodes[0].child_offset  = 0xFFFFFFFF;
    temp_nodes[0].bbox.set_empty();

    node_phases[1]              = DONE_NODE;
    temp_nodes[1].count         = 0;
    temp_nodes[1].depth         = 0;
    temp_nodes[1].child_offset  = 0xFFFFFFFF;
    temp_nodes[1].bbox.set_empty();
}

// Every point starts at the root, which accumulates the global bounding box.
template<int Dim>
__global__ void kd_init_prims(
    TempNode<Dim>* temp_nodes, PrimState* prim_states,
    CoordPtrs<Dim> coords, uint32_t n_prims)
{
    const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n_prims) return;

    prim_states[id].prim_id = id;
    prim_states[id].node_id = 0;
    prim_states[id].done    = false;

    atomicAdd(&temp_nodes[0].count, 1u);
    temp_nodes[0].bbox.grow(coords, id);
}

// One thread per node. Child pairs are reserved with an atomic counter.
template<int Dim>
__global__ void kd_select_splits(
    BuildState* build_state, NodePhase* node_phases,
    TempNode<Dim>* temp_nodes, uint32_t num_nodes, uint32_t max_nodes)
{
    const uint32_t node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (node_id >= num_nodes) return;

    NodePhase&     phase = node_phases[node_id];
    TempNode<Dim>& node  = temp_nodes[node_id];

    if (phase == DONE_NODE) return;

    if (phase == OPEN_NODE) { phase = DONE_NODE; return; }

    #define MAKE_LEAF() do { phase = DONE_NODE; \
                             node.child_offset = 0xFFFFFFFF; return; } while (0)

    if (node.count <= LEAF_THRESHOLD) MAKE_LEAF();

    if (node.depth >= MAX_DEPTH) MAKE_LEAF();

    float best_range = 0.f;
    int   dim        = 0;
    for (int d = 0; d < Dim; d++) {
        const float r = node.bbox.get_max(d) - node.bbox.get_min(d);
        if (r > best_range) { best_range = r; dim = d; }
    }

    const float lo        = node.bbox.get_min(dim);
    const float hi        = node.bbox.get_max(dim);
    const float split_pos = 0.5f * (lo + hi);

    // The midpoint does not separate anything (all points equal along every axis,
    // or lo and hi are adjacent floats): this is what guarantees termination.
    if (!(split_pos > lo)) MAKE_LEAF();

    const uint32_t child_base = atomicAdd(&build_state->num_nodes, 2u);
    if (child_base + 1 >= max_nodes) MAKE_LEAF();

    node.split_dim    = dim;
    node.split_pos    = split_pos;
    node.child_offset = child_base;
    for (int side = 0; side < 2; side++) {
        node_phases[child_base + side]              = OPEN_BRANCH;
        temp_nodes [child_base + side].count        = 0;
        temp_nodes [child_base + side].depth        = node.depth + 1;
        temp_nodes [child_base + side].child_offset = 0xFFFFFFFF;
        temp_nodes [child_base + side].bbox.set_empty();
    }
    phase = OPEN_NODE;

    #undef MAKE_LEAF
}

// One thread per point: move it to the child on its side of the split.
template<int Dim>
__global__ void kd_update_prims(
    NodePhase* node_phases, TempNode<Dim>* temp_nodes, PrimState* prim_states,
    CoordPtrs<Dim> coords, uint32_t n_prims)
{
    const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n_prims) return;

    PrimState& me = prim_states[id];
    if (me.done) return;

    if (node_phases[me.node_id] == DONE_NODE) { me.done = true; return; }

    const TempNode<Dim>& node  = temp_nodes[me.node_id];
    const float          coord = coords.c[node.split_dim][id];
    const int            side  = (coord >= node.split_pos) ? 1 : 0;
    const uint32_t       child = node.child_offset + side;

    me.node_id = child;
    atomicAdd(&temp_nodes[child].count, 1u);
    temp_nodes[child].bbox.grow(coords, id);
}

// Converts TempNode to KDNode. Leaf offsets are filled by kd_write_prim_ids_nd.
template<int Dim>
__global__ void kd_write_nodes(
    KDNode<Dim>* final_nodes, const TempNode<Dim>* temp_nodes,
    const uint32_t* d_num_nodes)
{
    const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= *d_num_nodes) return;

    const TempNode<Dim>& src = temp_nodes[id];
    KDNode<Dim>&         dst = final_nodes[id];

    for (int d = 0; d < Dim; d++) {
        dst.bb_min[d] = src.bbox.get_min(d);
        dst.bb_max[d] = src.bbox.get_max(d);
    }

    if (src.child_offset == 0xFFFFFFFF) {
        dst.split_dim = -1;
        dst.split_pos = 0.f;
        dst.count     = src.count;

        dst.offset    = 0xFFFFFFFF;
    } else {
        dst.split_dim = (int16_t)src.split_dim;
        dst.split_pos = src.split_pos;
        dst.count     = 0;
        dst.offset    = src.child_offset;
    }
}

// After sorting points by leaf, each leaf's offset is its first position.
template<int Dim>
__global__ void kd_write_prim_ids_nd(
    KDNode<Dim>* final_nodes, uint32_t* prim_ids,
    const uint32_t* sorted_node_ids,
    const uint32_t* sorted_prim_ids,
    uint32_t n_prims)
{
    const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos >= n_prims) return;
    prim_ids[pos] = sorted_prim_ids[pos];
    atomicMin(&final_nodes[sorted_node_ids[pos]].offset, pos);
}

// Sort keys (leaf id) and values (point id) for the radix sort.
__global__ void kd_extract_kv(
    const PrimState* prim_states,
    uint32_t* keys, uint32_t* vals,
    uint32_t n)
{
    const uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;
    keys[id] = prim_states[id].node_id;
    vals[id] = prim_states[id].prim_id;
}

}

namespace {

template<int Dim>
struct KdBufs {
    BuildState*     build_state;
    NodePhase*      node_phases;
    TempNode<Dim>*  temp_nodes;
    PrimState*      prim_states;
    uint32_t*       keys_in;
    uint32_t*       vals_in;
    uint32_t*       keys_out;
    uint32_t*       vals_out;
    void*           cub_tmp;
    size_t          cub_bytes;
};

inline size_t cub_kd_sort_bytes(uint32_t n)
{
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, bytes,
        (const uint32_t*)nullptr, (uint32_t*)nullptr,
        (const uint32_t*)nullptr, (uint32_t*)nullptr,
        (int)n);
    return bytes;
}

// One plan for both measuring and laying out the scratch buffer.
template<int Dim>
KdBufs<Dim> plan_kd_scratch(ScratchLayout& L, uint32_t n_prims)
{
    const uint32_t max_nodes = 4 * n_prims + 4;
    KdBufs<Dim> b;
    b.build_state = L.take<BuildState>(1);
    b.node_phases = L.take<NodePhase>(max_nodes);
    b.temp_nodes  = L.take<TempNode<Dim>>(max_nodes);
    b.prim_states = L.take<PrimState>(n_prims);
    b.keys_in     = L.take<uint32_t>(n_prims);
    b.vals_in     = L.take<uint32_t>(n_prims);
    b.keys_out    = L.take<uint32_t>(n_prims);
    b.vals_out    = L.take<uint32_t>(n_prims);
    b.cub_bytes   = cub_kd_sort_bytes(n_prims);
    b.cub_tmp     = L.take_bytes(b.cub_bytes);
    return b;
}

}

template<int Dim>
size_t kd_scratch_bytes(uint32_t n_prims)
{
    if (n_prims == 0) return 0;
    ScratchLayout sizer;
    plan_kd_scratch<Dim>(sizer, n_prims);
    return sizer.total();
}

template<int Dim>
void build_kdtree_gpu(
    KDNode<Dim>*   d_nodes,
    uint32_t*      d_prim_ids,
    CoordPtrs<Dim> coords,
    uint32_t       n_prims,
    uint32_t&      num_nodes,
    void*          scratch,
    cudaStream_t   stream)
{
    const uint32_t max_nodes = 4 * n_prims + 4;
    const uint32_t block     = 256;
    auto grid = [&](uint32_t n){ return (n + block - 1) / block; };

    ScratchLayout layout(scratch);
    KdBufs<Dim> b = plan_kd_scratch<Dim>(layout, n_prims);

    cudaMemsetAsync(b.node_phases, 2,
                    max_nodes * sizeof(NodePhase), stream);

    kd_init_state<Dim><<<1, 1, 0, stream>>>(b.build_state, b.node_phases, b.temp_nodes);
    CUDA_CHECK_KERNEL();

    kd_init_prims<Dim><<<grid(n_prims), block, 0, stream>>>(
        b.temp_nodes, b.prim_states, coords, n_prims);
    CUDA_CHECK_KERNEL();

    // A fixed number of levels avoids a device-to-host sync per level; nodes still
    // open at the end become leaves.
    const int max_iters = (n_prims > 1)
        ? (int)ceilf(log2f((float)n_prims)) + 4
        : 1;

    for (int iter = 0; iter < max_iters; iter++) {
        kd_select_splits<Dim><<<grid(max_nodes), block, 0, stream>>>(
            b.build_state, b.node_phases, b.temp_nodes, max_nodes, max_nodes);
        CUDA_CHECK_KERNEL();

        kd_update_prims<Dim><<<grid(n_prims), block, 0, stream>>>(
            b.node_phases, b.temp_nodes, b.prim_states, coords, n_prims);
        CUDA_CHECK_KERNEL();
    }

    kd_extract_kv<<<grid(n_prims), block, 0, stream>>>(
        b.prim_states, b.keys_in, b.vals_in, n_prims);
    CUDA_CHECK_KERNEL();

    cub::DeviceRadixSort::SortPairs(
        b.cub_tmp, b.cub_bytes,
        b.keys_in,  b.keys_out,
        b.vals_in,  b.vals_out,
        (int)n_prims, 0, (int)(sizeof(uint32_t) * 8), stream);

    kd_write_nodes<Dim><<<grid(max_nodes), block, 0, stream>>>(
        d_nodes, b.temp_nodes, &b.build_state->num_nodes);
    CUDA_CHECK_KERNEL();

    kd_write_prim_ids_nd<Dim><<<grid(n_prims), block, 0, stream>>>(
        d_nodes, d_prim_ids, b.keys_out, b.vals_out, n_prims);
    CUDA_CHECK_KERNEL();

    cudaMemcpyAsync(&num_nodes, &b.build_state->num_nodes,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);

}

template size_t kd_scratch_bytes<2>(uint32_t);
template size_t kd_scratch_bytes<3>(uint32_t);
template void build_kdtree_gpu<2>(KDNode<2>*, uint32_t*, CoordPtrs<2>, uint32_t, uint32_t&, void*, cudaStream_t);
template void build_kdtree_gpu<3>(KDNode<3>*, uint32_t*, CoordPtrs<3>, uint32_t, uint32_t&, void*, cudaStream_t);

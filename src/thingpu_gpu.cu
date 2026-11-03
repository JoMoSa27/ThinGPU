// thingpu-gpu: per-species spatial thinning and target-group density on the GPU.
//
// Called by the Python package (thingpu/gpu.py), which writes the inputs as raw
// little-endian arrays and reads the output back:
//
//   thingpu-gpu thin   --n N --xyz xyz.f32 --group group.i32 --distance-m D
//                      [--rule mindeg|random] [--seed S] [--index kdtree|lbvh|grid] --out kept.u8
//   thingpu-gpu effort --n N --xyz xyz.f32 --distance-m D [--index ...] --out count.i32
//
// xyz.f32 holds N x 3 float32 Cartesian coordinates in metres (centred on their
// mean), group.i32 the species id of each record (-1 = not thinned). The last
// line printed on stdout is a JSON summary.
//
// Thinning materialises the same-species neighbour graph in CSR form (count,
// scan, fill, compact) and then decides records in rounds: a record is kept when
// no undecided neighbour has a higher priority, and its neighbours are dropped.
// With a fixed priority order this returns exactly the sequential greedy set
// (Blelloch, Fineman and Shun, SPAA 2012); the priorities match thingpu/thinning.py.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "SpatialTags.cuh"
#include "Metric.cuh"
#include "KDTree2D.cuh"
#include "LinearBVH.cuh"
#include "UniformGrid2D.cuh"

using Metric3 = Cartesian<3>;

#define CUDA_CHECK(expr) do {                                                  \
    cudaError_t e_ = (expr);                                                   \
    if (e_ != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                     cudaGetErrorString(e_));                                  \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

// Geometry

static constexpr double EARTH_RADIUS_M = 6371008.8;

// Chord radius for a geodesic distance, lowered by one float32 step so that the
// library's <= comparison behaves as the strict < used by spThin and GeoThinneR.
static float strict_chord_radius_f32(double distance_m) {
    const double chord = 2.0 * EARTH_RADIUS_M * std::sin(distance_m / (2.0 * EARTH_RADIUS_M));
    return std::nextafterf(static_cast<float>(chord), 0.0f);
}

// Priorities: splitmix64 (Steele, Lea and Flood, 2014), as in thingpu/thinning.py

__host__ __device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__host__ __device__ __forceinline__ uint64_t record_hash(uint64_t id, uint64_t seed) {
    return splitmix64(id ^ splitmix64(seed));
}

enum : int8_t { UNDECIDED = 0, KEPT = 1, DROPPED = 2 };

// Lexicographic comparison of priority keys, higher first:
//   random: (hash, id)    mindeg: (-degree, hash, id)
__device__ __forceinline__ bool outranks(uint32_t u, uint32_t v, const uint64_t* h,
                                         const int32_t* deg, bool use_degree) {
    if (use_degree && deg[u] != deg[v]) return deg[u] < deg[v];
    if (h[u] != h[v]) return h[u] > h[v];
    return u > v;
}

// Kernels

template <typename DS>
__global__ void count_k(DS ds, CoordPtrs<3> pts, float r, uint32_t* cnt, uint32_t* overflow, uint32_t n) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float q[3] = {pts.c[0][i], pts.c[1][i], pts.c[2][i]};
    bool ov = false;
    cnt[i] = ds.count_in_radius(q, r, &ov);
    if (ov) atomicAdd(overflow, 1u);
}

template <typename DS>
__global__ void fill_k(DS ds, CoordPtrs<3> pts, float r, const uint64_t* off, const uint32_t* cnt,
                       uint32_t* csr, uint32_t n) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    float q[3] = {pts.c[0][i], pts.c[1][i], pts.c[2][i]};
    ds.range_query(q, r, csr + off[i], cnt[i]);
}

// Keeps, in place, only neighbours of the same species (and not the record itself).
// Safe in place because every row shrinks and is written before the next row is read.
__global__ void compact_k(const uint64_t* off, const uint32_t* cnt, const int32_t* group,
                          uint32_t* csr, int32_t* deg, uint32_t n) {
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n) return;
    const uint64_t o = off[v];
    const int32_t g = group[v];
    uint32_t k = 0;
    for (uint32_t j = 0; j < cnt[v]; j++) {
        const uint32_t u = csr[o + j];
        if (u != v && group[u] == g) csr[o + k++] = u;
    }
    deg[v] = static_cast<int32_t>(k);
}

__global__ void move_rows_k(const uint64_t* off_old, const uint64_t* off_new, const int32_t* deg,
                            const uint32_t* src, uint32_t* dst, uint32_t n) {
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n) return;
    for (int32_t j = 0; j < deg[v]; j++) dst[off_new[v] + j] = src[off_old[v] + j];
}

__global__ void init_k(const int32_t* group, uint64_t seed, uint64_t* h, int8_t* state, uint32_t n) {
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n) return;
    h[v] = record_hash(v, seed);
    state[v] = group[v] < 0 ? DROPPED : UNDECIDED;
}

__global__ void select_k(const uint64_t* off, const int32_t* deg, const uint32_t* csr,
                         const int8_t* state, const uint64_t* h, bool use_degree, uint8_t* sel, uint32_t n) {
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n) return;
    sel[v] = 0;
    if (state[v] != UNDECIDED) return;
    for (int32_t j = 0; j < deg[v]; j++) {
        const uint32_t u = csr[off[v] + j];
        if (state[u] == UNDECIDED && outranks(u, v, h, deg, use_degree)) return;
    }
    sel[v] = 1;
}

__global__ void keep_k(const uint8_t* sel, int8_t* state, uint32_t n) {
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v < n && sel[v]) state[v] = KEPT;
}

__global__ void drop_k(const uint64_t* off, const int32_t* deg, const uint32_t* csr, int8_t* state,
                       uint32_t* undecided, uint32_t n) {
    const uint32_t v = threadIdx.x + blockIdx.x * blockDim.x;
    if (v >= n || state[v] != UNDECIDED) return;
    for (int32_t j = 0; j < deg[v]; j++)
        if (state[csr[off[v] + j]] == KEPT) { state[v] = DROPPED; return; }
    atomicAdd(undecided, 1u);
}

__global__ void effort_k(const uint32_t* cnt, int32_t* out, uint32_t n) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = static_cast<int32_t>(cnt[i]) - 1;     // minus the record itself
}

// Host helpers

template <typename T>
static std::vector<T> read_raw(const std::string& path, size_t count) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<T> v(count);
    const size_t got = std::fread(v.data(), sizeof(T), count, f);
    std::fclose(f);
    if (got != count) throw std::runtime_error(path + ": expected " + std::to_string(count) + " values");
    return v;
}

template <typename T>
static void write_raw(const std::string& path, const std::vector<T>& v) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    std::fwrite(v.data(), sizeof(T), v.size(), f);
    std::fclose(f);
}

static void exclusive_scan(const uint32_t* in, uint64_t* out, uint32_t n) {
    void* tmp = nullptr; size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, n + 1);
    CUDA_CHECK(cudaMalloc(&tmp, bytes));
    cub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, n + 1);
    CUDA_CHECK(cudaFree(tmp));
}

static void exclusive_scan(const int32_t* in, uint64_t* out, uint32_t n) {
    void* tmp = nullptr; size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, n + 1);
    CUDA_CHECK(cudaMalloc(&tmp, bytes));
    cub::DeviceScan::ExclusiveSum(tmp, bytes, in, out, n + 1);
    CUDA_CHECK(cudaFree(tmp));
}

struct Points {
    uint32_t n = 0;
    float* d[3] = {nullptr, nullptr, nullptr};
    CoordPtrs<3> ptrs{};
    float lo[3]{}, hi[3]{};

    Points(const std::vector<float>& xyz, uint32_t count) : n(count) {
        std::vector<float> c(n);
        for (int k = 0; k < 3; k++) {
            lo[k] = 1e30f; hi[k] = -1e30f;
            for (uint32_t i = 0; i < n; i++) {
                c[i] = xyz[3 * i + k];
                lo[k] = std::min(lo[k], c[i]); hi[k] = std::max(hi[k], c[i]);
            }
            CUDA_CHECK(cudaMalloc(&d[k], n * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d[k], c.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            ptrs.c[k] = d[k];
        }
    }
    ~Points() { for (auto p : d) if (p) cudaFree(p); }
};

// A dense uniform grid with cells as wide as the radius can exceed memory for
// continental or global data; it is refused here with a clear message.
static bool grid_fits(const Points& p, float r) {
    unsigned long long cells = 1;
    for (int k = 0; k < 3; k++) cells *= static_cast<unsigned long long>((p.hi[k] - p.lo[k]) / r) + 1ull;
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    return cells <= 0xFFFFFFFFull && cells * sizeof(GridCell) < 0.9 * free_b;
}

struct Timer {
    cudaEvent_t a{}, b{};
    Timer() { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); CUDA_CHECK(cudaEventRecord(a)); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    float ms() {
        CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
        float t = 0; CUDA_CHECK(cudaEventElapsedTime(&t, a, b)); return t;
    }
};

// Neighbour counts for every record; the caller frees *d_cnt.
template <typename DS>
static uint32_t* count_neighbours(DS& ds, const Points& p, float r) {
    const uint32_t n = p.n, TPB = 256, blocks = (n + TPB - 1) / TPB;
    uint32_t *d_cnt = nullptr, *d_ov = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cnt, (static_cast<size_t>(n) + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ov, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_cnt + n, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_ov, 0, sizeof(uint32_t)));
    count_k<DS><<<blocks, TPB>>>(ds, p.ptrs, r, d_cnt, d_ov, n);
    CUDA_CHECK(cudaGetLastError());
    uint32_t ov = 0;
    CUDA_CHECK(cudaMemcpy(&ov, d_ov, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_ov);
    if (ov) throw std::runtime_error("neighbour count overflowed in " + std::to_string(ov) + " records");
    return d_cnt;
}

template <typename DS>
static std::string run_thin(DS& ds, const Points& p, const std::vector<int32_t>& group, float r,
                            bool use_degree, uint64_t seed, std::vector<uint8_t>& kept, float build_ms) {
    const uint32_t n = p.n, TPB = 256, blocks = (n + TPB - 1) / TPB;
    Timer t;
    int32_t* d_group = nullptr;
    CUDA_CHECK(cudaMalloc(&d_group, n * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_group, group.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));

    uint32_t* d_cnt = count_neighbours(ds, p, r);
    uint64_t *d_off = nullptr, *d_off2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_off, (static_cast<size_t>(n) + 1) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_off2, (static_cast<size_t>(n) + 1) * sizeof(uint64_t)));
    exclusive_scan(d_cnt, d_off, n);
    uint64_t total = 0;
    CUDA_CHECK(cudaMemcpy(&total, d_off + n, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    uint32_t* d_csr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_csr, std::max<uint64_t>(total, 1) * sizeof(uint32_t)));
    fill_k<DS><<<blocks, TPB>>>(ds, p.ptrs, r, d_off, d_cnt, d_csr, n);
    int32_t* d_deg = nullptr;
    CUDA_CHECK(cudaMalloc(&d_deg, (static_cast<size_t>(n) + 1) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(d_deg + n, 0, sizeof(int32_t)));
    compact_k<<<blocks, TPB>>>(d_off, d_cnt, d_group, d_csr, d_deg, n);
    CUDA_CHECK(cudaGetLastError());
    exclusive_scan(d_deg, d_off2, n);
    uint64_t edges = 0;
    CUDA_CHECK(cudaMemcpy(&edges, d_off2 + n, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint32_t* d_graph = nullptr;
    CUDA_CHECK(cudaMalloc(&d_graph, std::max<uint64_t>(edges, 1) * sizeof(uint32_t)));
    move_rows_k<<<blocks, TPB>>>(d_off, d_off2, d_deg, d_csr, d_graph, n);
    CUDA_CHECK(cudaGetLastError());
    cudaFree(d_csr); cudaFree(d_cnt); cudaFree(d_off);

    uint64_t* d_h = nullptr; int8_t* d_state = nullptr; uint8_t* d_sel = nullptr; uint32_t* d_left = nullptr;
    CUDA_CHECK(cudaMalloc(&d_h, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_state, n * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_sel, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_left, sizeof(uint32_t)));
    init_k<<<blocks, TPB>>>(d_group, seed, d_h, d_state, n);
    int rounds = 0;
    for (uint32_t left = 1; left > 0;) {
        if (++rounds > 100000) throw std::runtime_error("thinning did not converge");
        select_k<<<blocks, TPB>>>(d_off2, d_deg, d_graph, d_state, d_h, use_degree, d_sel, n);
        keep_k<<<blocks, TPB>>>(d_sel, d_state, n);
        CUDA_CHECK(cudaMemset(d_left, 0, sizeof(uint32_t)));
        drop_k<<<blocks, TPB>>>(d_off2, d_deg, d_graph, d_state, d_left, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&left, d_left, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
    const float thin_ms = t.ms();

    std::vector<int8_t> state(n);
    CUDA_CHECK(cudaMemcpy(state.data(), d_state, n * sizeof(int8_t), cudaMemcpyDeviceToHost));
    kept.resize(n);
    long long n_kept = 0;
    for (uint32_t i = 0; i < n; i++) { kept[i] = state[i] == KEPT; n_kept += kept[i]; }
    for (void* q : {(void*)d_group, (void*)d_off2, (void*)d_deg, (void*)d_graph, (void*)d_h,
                    (void*)d_state, (void*)d_sel, (void*)d_left}) cudaFree(q);
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "{\"kept\": %lld, \"rounds\": %d, \"edges\": %llu, \"build_ms\": %.3f, \"thin_ms\": %.3f}",
                  n_kept, rounds, static_cast<unsigned long long>(edges / 2), build_ms, thin_ms);
    return buf;
}

template <typename DS>
static std::string run_effort(DS& ds, const Points& p, float r, std::vector<int32_t>& out, float build_ms) {
    const uint32_t n = p.n, TPB = 256, blocks = (n + TPB - 1) / TPB;
    Timer t;
    uint32_t* d_cnt = count_neighbours(ds, p, r);
    int32_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int32_t)));
    effort_k<<<blocks, TPB>>>(d_cnt, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    const float ms = t.ms();
    out.resize(n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_cnt); cudaFree(d_out);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "{\"build_ms\": %.3f, \"count_ms\": %.3f}", build_ms, ms);
    return buf;
}

// Builds the requested index and runs `job` on it.
template <typename Job>
static std::string with_index(const std::string& index, const Points& p, float r, Job job) {
    if (index == "kdtree") {
        KDTree<Metric3, 3> ds;
        Timer t; ds.build_device(p.ptrs, p.n, Metric3{}); const float b = t.ms();
        std::string s = job(ds, b); ds.clear(); return s;
    }
    if (index == "lbvh") {
        LinearBVH<Metric3, 3> ds;
        Timer t; ds.build_device(p.ptrs, p.n, Metric3{}); const float b = t.ms();
        std::string s = job(ds, b); ds.clear(); return s;
    }
    if (index == "grid") {
        if (!grid_fits(p, r))
            throw std::runtime_error("the uniform grid does not fit in GPU memory for this extent and "
                                     "distance; use --index kdtree");
        UniformGrid<Metric3, 3> ds;
        const float cell[3] = {r, r, r};
        Timer t; ds.build_device(p.ptrs, p.n, cell, Metric3{}); const float b = t.ms();
        std::string s = job(ds, b); ds.clear(); return s;
    }
    throw std::runtime_error("unknown index: " + index);
}

static void usage() {
    std::fprintf(stderr,
        "usage: thingpu-gpu thin   --n N --xyz F --group F --distance-m D [--rule mindeg|random]\n"
        "                          [--seed S] [--index kdtree|lbvh|grid] --out F\n"
        "       thingpu-gpu effort --n N --xyz F --distance-m D [--index ...] --out F\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    const std::string mode = argv[1];
    std::string xyz_path, group_path, out_path, rule = "mindeg", index = "kdtree";
    unsigned long long n = 0, seed = 0;
    double distance_m = 0;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        if (i + 1 >= argc) { usage(); return 2; }
        const char* v = argv[++i];
        if (a == "--n") n = std::strtoull(v, nullptr, 10);
        else if (a == "--xyz") xyz_path = v;
        else if (a == "--group") group_path = v;
        else if (a == "--out") out_path = v;
        else if (a == "--distance-m") distance_m = std::atof(v);
        else if (a == "--rule") rule = v;
        else if (a == "--seed") seed = std::strtoull(v, nullptr, 10);
        else if (a == "--index") index = v;
        else { usage(); return 2; }
    }
    if (n == 0 || n > 0xFFFFFFFFull || xyz_path.empty() || out_path.empty() || distance_m <= 0 ||
        (mode == "thin" && group_path.empty()) || (rule != "mindeg" && rule != "random")) {
        usage(); return 2;
    }
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            throw std::runtime_error("no CUDA device available");
        const uint32_t count = static_cast<uint32_t>(n);
        Points p(read_raw<float>(xyz_path, 3ull * count), count);
        const float r = strict_chord_radius_f32(distance_m);
        std::string summary;
        if (mode == "thin") {
            const std::vector<int32_t> group = read_raw<int32_t>(group_path, count);
            std::vector<uint8_t> kept;
            summary = with_index(index, p, r, [&](auto& ds, float b) {
                return run_thin(ds, p, group, r, rule == "mindeg", seed, kept, b); });
            write_raw(out_path, kept);
        } else if (mode == "effort") {
            std::vector<int32_t> counts;
            summary = with_index(index, p, r, [&](auto& ds, float b) { return run_effort(ds, p, r, counts, b); });
            write_raw(out_path, counts);
        } else {
            usage(); return 2;
        }
        std::printf("%s\n", summary.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "thingpu-gpu: %s\n", e.what());
        return 1;
    }
    return 0;
}

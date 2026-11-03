#pragma once
#include <type_traits>

// Structure-of-arrays coordinates: c[d] points to the d-th coordinate of all points.
template<int Dim>
struct CoordPtrs { const float* c[Dim]; };

// Capability tags. A structure inherits from the tags of the device queries it
// implements, and kernels check them at compile time with the REQUIRE_* macros.
struct TagDeviceNN {};

struct TagDeviceRange {};

struct TagDeviceCount {};

struct TagDeviceAnnular {};

struct TagDeviceRay {};

struct TagMetric {};

template<typename DS>
inline constexpr bool supports_nn    = std::is_base_of_v<TagDeviceNN,    DS>;

template<typename DS>
inline constexpr bool supports_range = std::is_base_of_v<TagDeviceRange, DS>;

template<typename DS>
inline constexpr bool supports_count = std::is_base_of_v<TagDeviceCount, DS>;

template<typename DS>
inline constexpr bool supports_annular = std::is_base_of_v<TagDeviceAnnular, DS>;

template<typename DS>
inline constexpr bool supports_ray = std::is_base_of_v<TagDeviceRay, DS>;

template<typename DS>
inline constexpr bool supports_metric = std::is_base_of_v<TagMetric,     DS>;

#define REQUIRE_NN(DS) \
  static_assert(supports_nn<DS>, \
    #DS " must implement __device__ nearest_neighbor(q1, q2, dist_sq)")

#define REQUIRE_RANGE(DS) \
  static_assert(supports_range<DS>, \
    #DS " must implement __device__ range_query(q1, q2, r, out, max)")

#define REQUIRE_COUNT(DS) \
  static_assert(supports_count<DS>, \
    #DS " must implement __device__ count_in_radius(q, radius)")

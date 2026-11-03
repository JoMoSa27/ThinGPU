#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "SpatialTags.cuh"
#include "Distributions.h"

// Euclidean metric. min_dist_sq is the squared distance from q to a box,
// used to prune nodes during searches.
template<int Dim = 2>
struct Cartesian : TagMetric {

    __host__ __device__ float distance_sq(const float* a, const float* b) const {
        float s = 0.f;
        for (int d = 0; d < Dim; d++) {
            const float diff = a[d] - b[d];
            s += diff * diff;
        }
        return s;
    }

    __host__ __device__ float min_dist_sq(
        const float* q, const float* mn, const float* mx) const
    {
        float s = 0.f;
        for (int d = 0; d < Dim; d++) {
            const float clamped = fmaxf(mn[d], fminf(q[d], mx[d]));
            const float diff    = q[d] - clamped;
            s += diff * diff;
        }
        return s;
    }
};

using CartesianMetric = Cartesian<2>;

// Metric on (r, phi) with periodic phi; r_scale converts angle to distance.
struct CylindricalMetric : TagMetric {

    float r_scale;

    __host__ __device__ CylindricalMetric(float scale = 1.f) : r_scale(scale) {}

    __host__ __device__ float distance_sq(const float* a, const float* b) const {
        const float T    = 6.28318530f;
        const float dr   = a[0] - b[0];
        float dphi = a[1] - b[1];
        dphi -= T * rintf(dphi / T);
        return dr*dr + (r_scale * dphi) * (r_scale * dphi);
    }

    __host__ __device__ float min_dist_sq(
        const float* q, const float* mn, const float* mx) const
    {
        const float T  = 6.28318530f;
        const float d0 = fmaxf(0.f, fmaxf(mn[0] - q[0], q[0] - mx[0]));

        const float qp = q[1];
        const float d1  = fmaxf(0.f, fmaxf(mn[1]   - qp, qp - mx[1]));
        const float d1a = fmaxf(0.f, fmaxf(mn[1]+T  - qp, qp - (mx[1]+T)));
        const float d1b = fmaxf(0.f, fmaxf(mn[1]-T  - qp, qp - (mx[1]-T)));
        const float dp  = fminf(d1, fminf(d1a, d1b));
        return d0*d0 + (r_scale * dp) * (r_scale * dp);
    }
};

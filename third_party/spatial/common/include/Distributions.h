#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

// Synthetic point distributions for tests and benchmarks (host side, rand()-based).

namespace dist_detail {

inline float randf() { return (rand() + 1.f) / (RAND_MAX + 2.f); }

// Standard normal sample from two uniforms.
inline float box_muller(float u, float v) {
    return sqrtf(-2.f * logf(u)) * cosf(2.f * 3.14159265f * v);
}

inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

}

// Uniform in a box.
struct UniformDistribution {
    float lower_bound1 =  0.f;
    float upper_bound1 =  1.f;
    float lower_bound2 =  0.f;
    float upper_bound2 =  1.f;
    float lower_bound3 =  0.f;
    float upper_bound3 =  1.f;
    const char* name   = "Uniform";

    void fill(std::vector<float>& c1, std::vector<float>& c2, uint32_t seed) {
        srand(seed);
        const float span1 = upper_bound1 - lower_bound1;
        const float span2 = upper_bound2 - lower_bound2;
        for (size_t i = 0; i < c1.size(); ++i) {
            c1[i] = lower_bound1 + span1 * dist_detail::randf();
            c2[i] = lower_bound2 + span2 * dist_detail::randf();
        }
    }

    void fill3(std::vector<float>& c1, std::vector<float>& c2,
               std::vector<float>& c3, uint32_t seed) {
        srand(seed);
        const float s1 = upper_bound1 - lower_bound1;
        const float s2 = upper_bound2 - lower_bound2;
        const float s3 = upper_bound3 - lower_bound3;
        for (size_t i = 0; i < c1.size(); ++i) {
            c1[i] = lower_bound1 + s1 * dist_detail::randf();
            c2[i] = lower_bound2 + s2 * dist_detail::randf();
            c3[i] = lower_bound3 + s3 * dist_detail::randf();
        }
    }
};

// One Gaussian blob, clamped to the box.
struct GaussianDistribution {
    float cx    = 0.5f;
    float cy    = 0.5f;
    float sigma = 0.15f;
    float lower_bound1 =  0.f;
    float upper_bound1 =  1.f;
    float lower_bound2 =  0.f;
    float upper_bound2 =  1.f;
    float cz           = 0.5f;
    float lower_bound3 =  0.f;
    float upper_bound3 =  1.f;
    const char* name   = "Gaussian";

    void fill(std::vector<float>& c1, std::vector<float>& c2, uint32_t seed) {
        srand(seed);
        for (size_t i = 0; i < c1.size(); ++i) {
            float z1 = dist_detail::box_muller(dist_detail::randf(), dist_detail::randf());
            float z2 = dist_detail::box_muller(dist_detail::randf(), dist_detail::randf());
            c1[i] = dist_detail::clampf(cx + sigma * z1, lower_bound1, upper_bound1);
            c2[i] = dist_detail::clampf(cy + sigma * z2, lower_bound2, upper_bound2);
        }
    }

    void fill3(std::vector<float>& c1, std::vector<float>& c2,
               std::vector<float>& c3, uint32_t seed) {
        srand(seed);
        for (size_t i = 0; i < c1.size(); ++i) {
            float z1 = dist_detail::box_muller(dist_detail::randf(), dist_detail::randf());
            float z2 = dist_detail::box_muller(dist_detail::randf(), dist_detail::randf());
            float z3 = dist_detail::box_muller(dist_detail::randf(), dist_detail::randf());
            c1[i] = dist_detail::clampf(cx + sigma * z1, lower_bound1, upper_bound1);
            c2[i] = dist_detail::clampf(cy + sigma * z2, lower_bound2, upper_bound2);
            c3[i] = dist_detail::clampf(cz + sigma * z3, lower_bound3, upper_bound3);
        }
    }
};

// Mixture of K Gaussian clusters with random centres, clamped to the box.
struct GMMDistribution {
    int   K             = 8;
    float cluster_sigma = 0.05f;
    float lower_bound1  =  0.f;
    float upper_bound1  =  1.f;
    float lower_bound2  =  0.f;
    float upper_bound2  =  1.f;
    float lower_bound3  =  0.f;
    float upper_bound3  =  1.f;
    const char* name    = "GMM";

    void fill(std::vector<float>& c1, std::vector<float>& c2, uint32_t seed) {
        srand(seed);
        const float s1 = upper_bound1 - lower_bound1;
        const float s2 = upper_bound2 - lower_bound2;
        std::vector<float> kc1(K), kc2(K);
        for (int k = 0; k < K; ++k) {
            kc1[k] = lower_bound1 + s1 * dist_detail::randf();
            kc2[k] = lower_bound2 + s2 * dist_detail::randf();
        }
        for (size_t i = 0; i < c1.size(); ++i) {
            const int k  = rand() % K;
            const float z1 = dist_detail::box_muller(dist_detail::randf(), dist_detail::randf());
            const float z2 = dist_detail::box_muller(dist_detail::randf(), dist_detail::randf());
            c1[i] = dist_detail::clampf(kc1[k] + cluster_sigma * z1, lower_bound1, upper_bound1);
            c2[i] = dist_detail::clampf(kc2[k] + cluster_sigma * z2, lower_bound2, upper_bound2);
        }
    }

    void fill3(std::vector<float>& c1, std::vector<float>& c2,
               std::vector<float>& c3, uint32_t seed) {
        srand(seed);
        const float s1 = upper_bound1 - lower_bound1;
        const float s2 = upper_bound2 - lower_bound2;
        const float s3 = upper_bound3 - lower_bound3;
        std::vector<float> kc1(K), kc2(K), kc3(K);
        for (int k = 0; k < K; ++k) {
            kc1[k] = lower_bound1 + s1 * dist_detail::randf();
            kc2[k] = lower_bound2 + s2 * dist_detail::randf();
            kc3[k] = lower_bound3 + s3 * dist_detail::randf();
        }
        for (size_t i = 0; i < c1.size(); ++i) {
            const int k  = rand() % K;
            const float z1 = dist_detail::box_muller(dist_detail::randf(), dist_detail::randf());
            const float z2 = dist_detail::box_muller(dist_detail::randf(), dist_detail::randf());
            const float z3 = dist_detail::box_muller(dist_detail::randf(), dist_detail::randf());
            c1[i] = dist_detail::clampf(kc1[k] + cluster_sigma * z1, lower_bound1, upper_bound1);
            c2[i] = dist_detail::clampf(kc2[k] + cluster_sigma * z2, lower_bound2, upper_bound2);
            c3[i] = dist_detail::clampf(kc3[k] + cluster_sigma * z3, lower_bound3, upper_bound3);
        }
    }
};

// Uniform in area on an annulus r_min <= r <= r_max, in (r, phi) or (r, phi, z).
struct AnnularDistribution {
    float r_min =   5.f;
    float r_max =  40.f;
    float z_min = -300.f;
    float z_max =  300.f;
    const char* name = "Annular";

    void fill(std::vector<float>& r, std::vector<float>& phi, uint32_t seed) {
        srand(seed);
        const float r2_min = r_min * r_min;
        const float r2_max = r_max * r_max;
        for (size_t i = 0; i < r.size(); ++i) {
            r[i]   = sqrtf(r2_min + dist_detail::randf() * (r2_max - r2_min));
            phi[i] = -3.14159265f + 2.f * 3.14159265f * dist_detail::randf();
        }
    }

    void fill3(std::vector<float>& r, std::vector<float>& phi,
               std::vector<float>& z, uint32_t seed) {
        srand(seed);
        const float r2_min = r_min * r_min;
        const float r2_max = r_max * r_max;
        const float z_span = z_max - z_min;
        for (size_t i = 0; i < r.size(); ++i) {
            r[i]   = sqrtf(r2_min + dist_detail::randf() * (r2_max - r2_min));
            phi[i] = -3.14159265f + 2.f * 3.14159265f * dist_detail::randf();
            z[i]   = z_min + z_span * dist_detail::randf();
        }
    }
};

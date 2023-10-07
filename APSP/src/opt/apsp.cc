#include "graph.hh"
#include <iostream>
#include <immintrin.h>
#include <avx512fintrin.h>

#define THREAD_NUM 128
#define SIMD_WIDTH 16
#define MAX_VNUM 4096

__m512i k_cache[MAX_VNUM / SIMD_WIDTH];


Graph Graph::apsp() {
    Graph result(*this);
    int stride_i = vertex_num_ / THREAD_NUM;
    for (int k = 0; k < vertex_num_; ++k) {

#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int j = 0; j < vertex_num_ / SIMD_WIDTH; ++j) {
                k_cache[j] = _mm512_loadu_si512(&result(k, 0) + j * SIMD_WIDTH);
            }
#pragma omp for schedule(static)
            for (int i = 0; i < vertex_num_; i += stride_i) {
                for (int ii = i; ii < i + stride_i; ++ii) {
                    __m512i tmp3 = _mm512_set1_epi32(result(ii, k));
                    static constexpr int blk_simd = SIMD_WIDTH; // 512 / 32 == 16
                    for (int j = 0; j < vertex_num_; j += blk_simd) {
                        __m512i tmp2 = k_cache[j / blk_simd];
                        __m512i tmp4 = _mm512_add_epi32(tmp2, tmp3);
                        _mm512_storeu_si512((__m512i *) &result(ii, j), _mm512_min_epi32(tmp4, _mm512_loadu_si512((__m512i *) &result(ii, j))));
                        // result(ii, j) = std::min(result(ii, j), tmp1 + result(k, j));
                    }
                }
            }
        }

    }
    return result;
}
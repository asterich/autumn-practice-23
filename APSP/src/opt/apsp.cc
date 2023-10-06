#include "graph.hh"
#include <iostream>
#include <immintrin.h>

#define THREAD_NUM 128

Graph Graph::apsp() {
    Graph result(*this);
    int stride_i = vertex_num_ / THREAD_NUM / 2;
    for (int k = 0; k < vertex_num_; ++k) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < vertex_num_; i += stride_i) {
            for (int ii = i; ii < i + stride_i; ++ii) {
                __m512i tmp3 = _mm512_set1_epi32(result(ii, k));
                int blk_simd = 16; // 512 / 32 == 16
                for (int j = 0; j < vertex_num_; j += blk_simd) {
                    __m512i tmp2 = _mm512_loadu_si512((__m512i *) &result(k, j));
                    __m512i tmp4 = _mm512_add_epi32(tmp2, tmp3);
                    _mm512_storeu_si512((__m512i *) &result(ii, j), _mm512_min_epi32(tmp4, _mm512_loadu_si512((__m512i *) &result(ii, j))));
                    // result(ii, j) = std::min(result(ii, j), tmp1 + result(k, j));
                }
                // __m256i tmp3 = _mm256_set1_epi32(result(ii, k));
                // int blk_simd = 8; // 256 / 32 == 8
                // for (int j = 0; j < vertex_num_; j += blk_simd) {
                //     __m256i tmp2 = _mm256_loadu_si256((__m256i *) &result(k, j));
                //     __m256i tmp4 = _mm256_add_epi32(tmp2, tmp3);
                //     _mm256_storeu_si256((__m256i *) &result(ii, j), _mm256_min_epi32(tmp4, _mm256_loadu_si256((__m256i *) &result(ii, j))));
                //     // result(ii, j) = std::min(result(ii, j), tmp1 + result(k, j));
                // }
            }
        }
    }
    return result;
}
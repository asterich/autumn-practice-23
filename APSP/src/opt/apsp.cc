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

void apsp_in_block(Graph &g1, Graph &g2, Graph &g3, int blk_size,
                    int offset_i, int offset_j, int offset_k) {
    for (int k = 0; k < blk_size; ++k) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < blk_size; i += 1) {
            __m512i tmp3 = _mm512_set1_epi32(g2(i + offset_i, k + offset_k));
            int blk_simd = 16; // 512 / 32 == 16
            for (int j = 0; j < blk_size; j += blk_simd) {
                __m512i tmp2 = _mm512_loadu_si512((__m512i *) &g3(k + offset_k, j + offset_j));
                __m512i tmp4 = _mm512_add_epi32(tmp2, tmp3);
                _mm512_storeu_si512(
                    (__m512i *)&g1(i + offset_i, j + offset_j), 
                    _mm512_min_epi32(
                        tmp4, 
                        _mm512_loadu_si512((__m512i *) &g1(i + offset_i, j + offset_j))));
                // g1(i, j) = std::min(g1(i, j), g2(i, k) + g3(k, j));
            }
        }
    }
}

void apsp_top_level(Graph &g, int vertex_num_) {
    int block_sz = 128;
    int block_num = vertex_num_ / block_sz;
    for (int k = 0; k < block_num; ++k) {
        int b = k * block_sz;
        apsp_in_block(g, g, g, block_sz, b, b, b);
        for (int j = 0; j < block_num; ++j) {
            int a = j * block_sz;
            if (j != k)
                apsp_in_block(g, g, g, block_sz, b, b, a);
        }
        for (int i = 0; i < block_num; ++i) {
            int a = i * block_sz;
            if (i != k)
                apsp_in_block(g, g, g, block_sz, a, b, b);
        }
        for (int i = 0; i < block_num; ++i) {
            int a = i * block_sz;
            for (int j = 0; j < block_num; ++j) {
                int b = j * block_sz;
                if (i != k && j != k)
                    apsp_in_block(g, g, g, block_sz, a, b, b);
            }
        }
    }
}
#include "graph.hh"
#include <iostream>
#include <immintrin.h>

#define THREAD_NUM 128

void apsp_top_level(Graph &g, int vertex_num_);

// Graph Graph::apsp() {
//     Graph result(*this);
//     int stride_i = vertex_num_ / THREAD_NUM / 2;
//     for (int k = 0; k < vertex_num_; ++k) {
// #pragma omp parallel for schedule(static)
//         for (int i = 0; i < vertex_num_; i += stride_i) {
//             for (int ii = i; ii < i + stride_i; ++ii) {
//                 __m512i tmp3 = _mm512_set1_epi32(result(ii, k));
//                 int blk_simd = 16; // 512 / 32 == 16
//                 for (int j = 0; j < vertex_num_; j += blk_simd) {
//                     __m512i tmp2 = _mm512_loadu_si512((__m512i *) &result(k, j));
//                     __m512i tmp4 = _mm512_add_epi32(tmp2, tmp3);
//                     _mm512_storeu_si512((__m512i *) &result(ii, j), _mm512_min_epi32(tmp4, _mm512_loadu_si512((__m512i *) &result(ii, j))));
//                     // result(ii, j) = std::min(result(ii, j), tmp1 + result(k, j));
//                 }
//                 // __m256i tmp3 = _mm256_set1_epi32(result(ii, k));
//                 // int blk_simd = 8; // 256 / 32 == 8
//                 // for (int j = 0; j < vertex_num_; j += blk_simd) {
//                 //     __m256i tmp2 = _mm256_loadu_si256((__m256i *) &result(k, j));
//                 //     __m256i tmp4 = _mm256_add_epi32(tmp2, tmp3);
//                 //     _mm256_storeu_si256((__m256i *) &result(ii, j), _mm256_min_epi32(tmp4, _mm256_loadu_si256((__m256i *) &result(ii, j))));
//                 //     // result(ii, j) = std::min(result(ii, j), tmp1 + result(k, j));
//                 // }
//             }
//         }
//     }
//     return result;
// }

Graph Graph::apsp() {
    Graph result(*this);
    apsp_top_level(result, vertex_num_);
    return result;
}


void apsp_in_block0(int *g1, int *g2, int *g3, int blk_size,
                    int vertex_num_) {
    for (int k = 0; k < blk_size; ++k) {
        int kth = k * vertex_num_;
        for (int i = 0; i < blk_size; i += 1) {
            __m512i tmp3 = _mm512_set1_epi32(g2[i * vertex_num_ + k]);
            int blk_simd = 16; // 512 / 32 == 16
#pragma unroll
            for (int j = 0; j < blk_size; j += blk_simd) {
                __m512i tmp2 = _mm512_loadu_si512((__m512i *) &g3[kth + j]);
                __m512i tmp4 = _mm512_add_epi32(tmp2, tmp3);
                tmp2 = _mm512_loadu_si512((__m512i *) &g1[i * vertex_num_ + j]);
                _mm512_storeu_si512((__m512i *)&g1[i * vertex_num_ + j],  _mm512_min_epi32(tmp4, tmp2));
                // g1(i, j) = std::min(g1(i, j), g2(i, k) + g3(k, j));
            }
        }
    }
}

void apsp_in_block1(int *g1, int *g2, int *g3, int blk_size,
                    int vertex_num_) {
    for (int k = 0; k < blk_size; k++) {
        int kth = k * vertex_num_;
        for (int i = 0; i < blk_size; i++) {
#pragma omp simd
#pragma unroll
            for (int j = 0; j < blk_size; j++) {
                int sum = g2[i * vertex_num_ + k] + g3[kth + j];
                if (g1[i * vertex_num_ + j] > sum) {
                    g1[i * vertex_num_ + j] = sum;
                }
            }
        }
    }
}

void apsp_in_block1_kk(int *g1, int *g2, int *g3, int blk_size,
                    int vertex_num_) {
    for (int k = 0; k < blk_size; k++) {
        int kth = k * vertex_num_;
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < blk_size; i++) {
#pragma omp simd
#pragma unroll
            for (int j = 0; j < blk_size; j++) {
                int sum = g2[i * vertex_num_ + k] + g3[kth + j];
                if (g1[i * vertex_num_ + j] > sum) {
                    g1[i * vertex_num_ + j] = sum;
                }
            }
        }
    }
}

void apsp_top_level(Graph &g, int vertex_num_) {
    int block_sz = 32;
    int block_num = vertex_num_ / block_sz;
    for (int k = 0; k < block_num; ++k) {
        apsp_in_block1(
            &g(k * block_sz, k * block_sz), 
            &g(k * block_sz, k * block_sz), 
            &g(k * block_sz, k * block_sz), 
            block_sz, vertex_num_);
#pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < block_num; ++j) {
            int offs_j = j * block_sz;
            if (j != k)
                apsp_in_block1(
                    &g(k * block_sz, j * block_sz), 
                    &g(k * block_sz, k * block_sz), 
                    &g(k * block_sz, j * block_sz), 
                    block_sz, vertex_num_);
        }
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < block_num; ++i) {
            int offs_i = i * block_sz;
            if (i != k)
                apsp_in_block1(
                    &g(i * block_sz, k * block_sz), 
                    &g(i * block_sz, k * block_sz), 
                    &g(k * block_sz, k * block_sz), 
                    block_sz, vertex_num_);
        }
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < block_num; ++i) {
            for (int j = 0; j < block_num; ++j) {
                int offs_i = i * block_sz;
                int offs_j = j * block_sz;
                if (i != k && j != k)
                    apsp_in_block1(
                        &g(i * block_sz, j * block_sz), 
                        &g(i * block_sz, k * block_sz), 
                        &g(k * block_sz, j * block_sz), 
                        block_sz, vertex_num_);
            }
        }
    }
}
#include "graph.hh"


Graph Graph::apsp() {
    Graph result(*this);
    for (int k = 0; k < vertex_num_; ++k) {
#pragma omp parallel for
        for (int i = 0; i < vertex_num_; ++i) {
            auto tmp = result(i, k);
            for (int j = 0; j < vertex_num_; ++j) {
                result(i, j) = std::min(result(i, j), tmp + result(k, j));
            }
        }
    }
    return result;
}
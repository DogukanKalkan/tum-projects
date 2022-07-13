#include "dgemm.h"
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <climits>

void dgemm(float alpha, const float *a, const float *b, float beta, float *c) {
    int num_bits = 8 - (MATRIX_SIZE % 8);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            c[i * MATRIX_SIZE + j] *= beta;
            __m256 partial_sum = _mm256_set1_ps(0);
            float partial_sum_array[8] = {0, 0, 0, 0, 0, 0, 0, 0};

            for (int k = 0; k < MATRIX_SIZE; k+=8) {
                __m256i mask ;
                uint bits[8]={0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};

                if(k+8 > MATRIX_SIZE){
                    for(int b = 0; b < num_bits; b++){
                        bits[b] = 0;
                    }
                }

                mask = _mm256_set_epi32(bits[0],bits[1],bits[2],bits[3],bits[4],bits[5],bits[6],bits[7]);

                //std::cout << i << "," << j << "," << k << std::endl;
                //std::cout << "3rd Loop" << std::endl;
                // TODO: define a __m256 type and load 8 float values in the matrix row into it
                __m256 a_row = _mm256_maskload_ps(a + i*MATRIX_SIZE + k, mask);
                //std::cout << "X" << std::endl;
                // TODO: define a __m256 type and load 8 float values in the vector into it
                __m256 b_col = _mm256_maskload_ps(b + j*MATRIX_SIZE + k, mask);
                // TODO: perform element-wise product between the above two __m256 type and store it in a new __m256 type
                __m256 mul_result = _mm256_mul_ps(a_row, b_col);
                // TODO: add partial_sum and the product result, and assign the result to partial_sum
                partial_sum = _mm256_add_ps(partial_sum, mul_result);
                //c[i * MATRIX_SIZE + j] += alpha * a[i * MATRIX_SIZE + k] * b[j * MATRIX_SIZE + k];
            }

            // TODO: store the partial_sum into partial_sum_array
            _mm256_store_ps(partial_sum_array, partial_sum);
            for(int t = 0; t<8; t++){
                c[i*MATRIX_SIZE + j] += alpha * partial_sum[t];
            }
        }
    }
}


int main(int, char **) {
    float alpha, beta;
    // mem allocations
    int mem_size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    auto a = (float *) aligned_alloc(32,mem_size);
    auto b = (float *) aligned_alloc(32,mem_size);
    auto c = (float *) aligned_alloc(32,mem_size);

    // check if allocated
    if (nullptr == a || nullptr == b || nullptr == c) {
        printf("Memory allocation failed\n");
        if (nullptr != a) free(a);
        if (nullptr != b) free(b);
        if (nullptr != c) free(c);
        return 0;
    }

    generateProblemFromInput(alpha, a, b, beta, c);

    std::cerr << "Launching dgemm step." << std::endl;
    // matrix-multiplication
    dgemm(alpha, a, b, beta, c);

    outputSolution(c);

    free(a);
    free(b);
    free(c);
    return 0;
}

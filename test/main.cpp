#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>
#include <cstring>
#include <algorithm>

#define LOWER_B 0.0
#define UPPER_B 1.0

#define THRESHOLD 200
#define OMP_NUM_THREAD 16

class Timer{
    std::chrono::high_resolution_clock::time_point start_;
    public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    float elapse(){
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float>(end - start_).count();
    }
};

std::vector<float> createRandomMatrix(int size, int seed){
    std::vector<float> matrix(size*size);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(LOWER_B, UPPER_B);
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = dist(rng);
    }
    return matrix;
}



void naiveAddMultiply(
    int n, 
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
){

    for(int i = 0; i < n; ++i){
        for(int k = 0; k < n; ++k){
            float a_ik = A[i * lda + k];
            
            #pragma omp simd
            for (int j = 0; j < n; ++j){
                C[i * ldc + j] += a_ik * B[k * ldb + j];
            }
        }
    }
}

// void naiveAddMultiply2(
//     int n, 
//     const float* A, int lda,
//     const float* B, int ldb,
//     float* C, int ldc
// ){
//     for(int i = 0; i < n; ++i){
//         for(int k = 0; k < n; ++k){
//             float a_ik = A[i * lda + k];
            
//             for (int j = 0; j < n; ++j){
//                 C[i * ldc + j] += a_ik * B[k * ldb + j];
//             }
//         }
//     }
// }

// void naiveAddMultiple3(
//     int n, 
//     const float* A, int lda,
//     const float* B, int ldb,
//     float* C, int ldc
// ){
//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < n ; i++){
//         for (int j = 0; j < n; j++){
//             float sum = 0.0f;
        
//             #pragma omp simd reduction(+:sum)
//             for (int k = 0; k < n; k++){
//                 sum += A[i*lda + k] * B[k*lda + j];
//             }

//             C[i*ldc + j] = sum;
//         }
//     }
// }



void naiveAddMultiply4(
    int n, 
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc
){
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < n; ++i){
        for(int k = 0; k < n; ++k){
            float a_ik = A[i * lda + k];
            
            #pragma omp simd
            for (int j = 0; j < n; ++j){
                C[i * ldc + j] += a_ik * B[k * ldb + j];
            }
        }
    }
}

int main() {

    omp_set_num_threads(OMP_NUM_THREAD);

    std::vector<int> sizes = {100, 1000, 10000};
    const int trials = 10;

    for (int N : sizes) {
        std::cout << "\n===== Matrix Size: " << N << " x " << N << " =====\n";

        auto A = createRandomMatrix(N, 123);
        auto B = createRandomMatrix(N, 456);

        for (int version = 1; version <= 2; version++) {
            float total_time = 0.0f;

            std::cout << "Testing naiveAddMultiply" << version << " ...\n";

            for (int t = 0; t < trials; t++) {
                std::vector<float> C(N * N, 0.0f);
                Timer timer;
                timer.start();

                if (version == 1)
                    naiveAddMultiply4(N, A.data(), N, B.data(), N, C.data(), N);
                else
                    naiveAddMultiply(N, A.data(), N, B.data(), N, C.data(), N);

                double end_local_time = timer.elapse();
                total_time += end_local_time;
                 std::cout << "Local time: trial " << t + 1 << " . Version: "<< version << " Time Exe: " << end_local_time << "s" << std::endl;
            }

            std::cout << "Avg time: " << (total_time / trials) << "s" << std::endl;
        }
    }

    return 0;
}
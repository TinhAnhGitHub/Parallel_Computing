#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <iomanip>

#define LOWER_B 0.0
#define UPPER_B 1.0

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

void serialVerify(int n, const float* A, const float* B, float* C){
    std::fill(C, C + n*n, 0.0f);
    for(int i = 0; i < n; ++i){
        for(int k = 0; k < n; ++k){
            float a_ik = A[i * n + k];
            for (int j = 0; j < n; ++j){
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
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

void recursiveMatMul(
    int n, 
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    int threshold
){
    // Handle non-power-of-2 sizes and small matrices
    if (n <= threshold) {
        naiveAddMultiply(n, A, lda, B, ldb, C, ldc);
        return;
    }

    // For odd sizes, use naive multiplication
    if (n % 2 != 0) {
        naiveAddMultiply(n, A, lda, B, ldb, C, ldc);
        return;
    }

    int m = n / 2;

    const float* A11 = A;
    const float* A12 = A + m;
    const float* A21 = A + m * lda;
    const float* A22 = A + m * lda + m;

    const float* B11 = B;
    const float* B12 = B + m;
    const float* B21 = B + m * ldb;
    const float* B22 = B + m * ldb + m;

    float* C11 = C;
    float* C12 = C + m;
    float* C21 = C + m * ldc;
    float* C22 = C + m * ldc + m;
    
    #pragma omp taskgroup
    {
        // First wave: C11 += A11*B11, C12 += A11*B12, C21 += A21*B11, C22 += A21*B12
        #pragma omp task
        recursiveMatMul(m, A11, lda, B11, ldb, C11, ldc, threshold);
        
        #pragma omp task
        recursiveMatMul(m, A11, lda, B12, ldb, C12, ldc, threshold);

        #pragma omp task
        recursiveMatMul(m, A21, lda, B11, ldb, C21, ldc, threshold);

        #pragma omp task
        recursiveMatMul(m, A21, lda, B12, ldb, C22, ldc, threshold);
        
        #pragma omp taskwait 

        // Second wave: C11 += A12*B21, C12 += A12*B22, C21 += A22*B21, C22 += A22*B22
        #pragma omp task
        recursiveMatMul(m, A12, lda, B21, ldb, C11, ldc, threshold);

        #pragma omp task
        recursiveMatMul(m, A12, lda, B22, ldb, C12, ldc, threshold);

        #pragma omp task
        recursiveMatMul(m, A22, lda, B21, ldb, C21, ldc, threshold);

        #pragma omp task
        recursiveMatMul(m, A22, lda, B22, ldb, C22, ldc, threshold);
    }
}

void parallelDCMatMul(
    int n, 
    const std::vector<float>& A, 
    const std::vector<float>& B, 
    std::vector<float>& C,
    int num_threads,
    int threshold
){
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            recursiveMatMul(n, A.data(), n, B.data(), n, C.data(), n, threshold);
        }
    }
}

// Pad matrix to nearest power of 2 (optional enhancement)
int nextPowerOf2(int n) {
    int power = 1;
    while (power < n) power *= 2;
    return power;
}

void printBenchmarkHeader() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "OpenMP Divide & Conquer Matrix Multiplication" << std::endl;
    std::cout << "================================================" << std::endl;
}

void printResults(int n, int threads, int threshold, float time, bool show_gflops = true) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Size: " << n << "x" << n 
              << " | Threads: " << threads 
              << " | Threshold: " << threshold
              << " | Time: " << time << "s";
    
    if (show_gflops) {
        double flops = 2.0 * n * n * n;
        double gflops = (flops / time) / 1e9;
        std::cout << " | " << std::setprecision(2) << gflops << " GFLOPS";
    }
    std::cout << std::endl;
}

int main(int argc, char ** argv){
    if (argc < 2){
        std::cerr << "Usage: " << argv[0] << " <matrix_size> [check_error] [num_threads] [threshold]" << std::endl;
        std::cerr << "  matrix_size: Size of square matrix (e.g., 1000)" << std::endl;
        std::cerr << "  check_error: 0=skip, 1=verify (default: 1)" << std::endl;
        std::cerr << "  num_threads: Number of OpenMP threads (default: max)" << std::endl;
        std::cerr << "  threshold: Base case size (default: 128)" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    int check = (argc > 2) ? std::atoi(argv[2]) : 1;
    int num_threads = (argc > 3) ? std::atoi(argv[3]) : omp_get_max_threads();
    int threshold = (argc > 4) ? std::atoi(argv[4]) : 128;

    printBenchmarkHeader();
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Threshold: " << threshold << std::endl;
    std::cout << "Max threads available: " << omp_get_max_threads() << std::endl;
    
    // Warn about non-power-of-2 sizes
    int next_pow2 = nextPowerOf2(N);
    if (next_pow2 != N) {
        std::cout << "⚠ Warning: Size " << N << " is not a power of 2." << std::endl;
        std::cout << "  Algorithm will fall back to naive multiplication for odd-sized blocks." << std::endl;
        std::cout << "  Nearest power of 2: " << next_pow2 << std::endl;
    }
    std::cout << "================================================" << std::endl;

    std::cout << "\nInitializing matrices..." << std::endl;
    auto A = createRandomMatrix(N, 123);
    auto B = createRandomMatrix(N, 456);
    std::vector<float> C(N * N, 0.0f);

    std::cout << "Starting Parallel Divide & Conquer..." << std::endl;
    Timer t;
    t.start();
    
    parallelDCMatMul(N, A, B, C, num_threads, threshold);
    
    float time = t.elapse();
    
    std::cout << "\n================================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "================================================" << std::endl;
    printResults(N, num_threads, threshold, time, true);

    if (check == 0){
        std::cout << "================================================" << std::endl;
        return 0;
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << "Verifying Correctness..." << std::endl;
    std::cout << "================================================" << std::endl;
    
    Timer verify_timer;
    verify_timer.start();
    std::vector<float> CC(N * N, 0.0f);
    serialVerify(N, A.data(), B.data(), CC.data());
    float verify_time = verify_timer.elapse();
    
    std::cout << "Serial verification time: " << verify_time << "s" << std::endl;
    
    float diff_sum = 0.0, ref_sum = 0.0;
    int max_errors_to_show = 0.01;
    int error_count = 0;
    
    for (int i = 0; i < N * N; ++i) {
        float diff = std::abs(C[i] - CC[i]);
        if (diff > 1e-3 && error_count < max_errors_to_show) {
            int row = i / N;
            int col = i % N;
            std::cout << "  Error at (" << row << "," << col << "): "
                      << "got " << C[i] << ", expected " << CC[i] 
                      << ", diff=" << diff << std::endl;
            error_count++;
        }
        diff_sum += (C[i] - CC[i]) * (C[i] - CC[i]);
        ref_sum += CC[i] * CC[i];
    }
    
    float rel_error = std::sqrt(diff_sum / (ref_sum + 1e-12));
    std::cout << "\nRelative L2 error: " << std::scientific << rel_error << std::endl;
    
    if (rel_error < 1e-4) {
        std::cout << "✓ PASSED - Results are correct!" << std::endl;
    } else {
        std::cout << "✗ FAILED - Results differ significantly!" << std::endl;
    }
    
    std::cout << "\nSpeedup vs Serial: " << std::fixed << std::setprecision(2) 
              << verify_time / time << "x" << std::endl;
    std::cout << "================================================" << std::endl;

    return 0;
}
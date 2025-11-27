#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>
#include <cstring>

#define LOWER_B 0.0
#define UPPER_B 1.0

#define THRESHOLD 128
#define MAX_DEPTH 4 

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


void naiveMultiply(
    int n, 
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float* C,
    int ldc
){
    #pragma omp parallel for schedule(static) default(none) shared(n, A, B, C, lda, ldb, ldc)
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

void addMatrix(
    int n, const float* A, int lda, const float* B, int ldb , float* C, int ldc
){
    
    for(int i = 0; i < n; i++){
        #pragma omp simd
        for(int j =0; j < n; j++){
            C[i * ldc + j] = A[i * lda + j] + B[i * ldb + j];
        }
    }
}

void subtractMatrix(
    int n, const float* A, int lda, const float* B, int ldb , float* C, int ldc
){
    for(int i = 0; i < n; i++){
        #pragma omp simd
        for(int j =0; j < n; j++){
            C[i * ldc + j] = A[i * lda + j] - B[i * ldb + j];
        }
    }
}

void strassenSerial(
    int n, 
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    float* work
){
    if (n <= THRESHOLD || n % 2 != 0) {
        for (int i = 0; i < n; i++) {
            std::memset(C + i * ldc, 0, n * sizeof(float));
        }
        naiveMultiply(n, A, lda, B, ldb, C, ldc);
        return;
    }
    int m = n / 2;
    float* M1 = work; 
    float* M2 = work + m*m; 
    float* M3 = work + 2*m*m; 
    float* M4 = work + 3*m*m;
    float* M5 = work + 4*m*m; 
    float* M6 = work + 5*m*m;
    float* M7 = work + 6*m*m; 
    float* T1 = work + 7*m*m; 
    float* T2 = work + 8*m*m;

    float* nextWork = work + 9*m*m;

    const float* A11 = A;             
    const float* A12 = A + m;
    const float* A21 = A + m * lda;   
    const float* A22 = A + m * lda + m;
    
    const float* B11 = B;             
    const float* B12 = B + m;
    const float* B21 = B + m * ldb;   
    const float* B22 = B + m * ldb + m;

    // M1 = (A11 + A22)(B11 + B22)
    addMatrix(m, A11, lda, A22, lda, T1, m);
    addMatrix(m, B11, ldb, B22, ldb, T2, m);
    strassenSerial(m, T1, m, T2, m, M1, m, nextWork);

    // M2 = (A21 + A22)B11
    addMatrix(m, A21, lda, A22, lda, T1, m);
    strassenSerial(m, T1, m, B11, ldb, M2, m, nextWork);

    // M3 = A11(B12 - B22)
    subtractMatrix(m, B12, ldb, B22, ldb, T1, m);
    strassenSerial(m, A11, lda, T1, m, M3, m, nextWork);

    // M4 = A22(B21 - B11)
    subtractMatrix(m, B21, ldb, B11, ldb, T1, m);
    strassenSerial(m, A22, lda, T1, m, M4, m, nextWork);

    // M5 = (A11 + A12)B22
    addMatrix(m, A11, lda, A12, lda, T1, m);
    strassenSerial(m, T1, m, B22, ldb, M5, m, nextWork);

    // M6 = (A21 - A11)(B11 + B12)
    subtractMatrix(m, A21, lda, A11, lda, T1, m);
    addMatrix(m, B11, ldb, B12, ldb, T2, m);
    strassenSerial(m, T1, m, T2, m, M6, m, nextWork);

    // M7 = (A12 - A22)(B21 + B22)
    subtractMatrix(m, A12, lda, A22, lda, T1, m);
    addMatrix(m, B21, ldb, B22, ldb, T2, m);
    strassenSerial(m, T1, m, T2, m, M7, m, nextWork);

    #pragma omp simd collapse(2)
    for (int i = 0; i < m ; i++){   
        for (int j = 0; j < m; j++){
            int k = i * m + j;
            C[i * ldc + j] = M1[k] + M4[k] - M5[k] + M7[k]; // C11
            C[i * ldc + (j + m)] = M3[k] + M5[k]; // C12
            C[(i + m) * ldc + j] = M2[k] + M4[k]; // C21
            C[(i + m) * ldc + (j + m)] = M1[k] - M2[k] + M3[k] + M6[k];
        }
    }
}

void strassenParallel(
    int n, 
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    int depth,
    int max_depth
){
    if (depth >= max_depth || n % 2 != 0) {
        
        if (n % 2 == 0) {
            size_t stackSize = (size_t)(3 * n * n);
            std::vector<float> serialStack(stackSize);
            strassenSerial(n, A, lda, B, ldb, C, ldc, serialStack.data());
        } else {
            naiveMultiply(n, A, lda, B, ldb, C, ldc);
        }
        return;
    }

    int m = n / 2;
    std::vector<float> results(7 * m * m);
    float* M1 = &results[0];         
    float* M2 = &results[m*m];
    float* M3 = &results[2*m*m];     
    float* M4 = &results[3*m*m];
    float* M5 = &results[4*m*m];     
    float* M6 = &results[5*m*m];
    float* M7 = &results[6*m*m];


    const float* A11 = A;             
    const float* A12 = A + m;
    const float* A21 = A + m * lda;   
    const float* A22 = A + m * lda + m;
    
    const float* B11 = B;             
    const float* B12 = B + m;
    const float* B21 = B + m * ldb;   
    const float* B22 = B + m * ldb + m;

    #pragma omp taskgroup
    {

        #pragma omp task shared(results)
        {
             // M2 = (A21 + A22)B11
            std::vector<float> T(m * m);
            addMatrix(m, A21, lda, A22, lda, T.data(), m);
            strassenParallel(m, T.data(), m, B11, ldb, M2, m, depth + 1, max_depth);
        }

        #pragma omp task shared(results)
        {
            // M3 = A11(B12 - B22)
            std::vector<float> T(m * m);
            subtractMatrix(m, B12, ldb, B22, ldb, T.data(), m);
            strassenParallel(m, A11, lda, T.data(), m, M3, m, depth + 1, max_depth);
        }
        
        #pragma omp task shared(results)
        {
            // M4 = A22(B21 - B11)
            std::vector<float> T(m * m);
            subtractMatrix(m, B21, ldb, B11, ldb, T.data(), m);
            strassenParallel(m, A22, lda, T.data(), m, M4, m, depth + 1, max_depth);
        }

        #pragma omp task shared(results)
        {
            std::vector<float> T(m * m);
            addMatrix(m, A11, lda, A12, lda, T.data(), m);
            strassenParallel(m, T.data(), m, B22, ldb, M5, m, depth + 1, max_depth);
        }

        #pragma omp task shared(results)
        {
            std::vector<float> T(2 * m * m);
            float* t1 = &T[0]; float* t2 = &T[m*m];
            subtractMatrix(m, A21, lda, A11, lda, t1, m);
            addMatrix(m, B11, ldb, B12, ldb, t2, m);
            strassenParallel(m, t1, m, t2, m, M6, m, depth + 1, max_depth);
        }

        #pragma omp task shared(results)
        {
            std::vector<float> T(2 * m * m);
            float* t1 = &T[0]; float* t2 = &T[m*m];
            subtractMatrix(m, A12, lda, A22, lda, t1, m);
            addMatrix(m, B21, ldb, B22, ldb, t2, m);
            strassenParallel(m, t1, m, t2, m, M7, m, depth + 1, max_depth);
        }

        {
            std::vector<float> T(2 * m * m);
            float* t1 = &T[0]; float* t2 = &T[m*m];
            addMatrix(m, A11, lda, A22, lda, t1, m);
            addMatrix(m, B11, ldb, B22, ldb, t2, m);
            strassenParallel(m, t1, m, t2, m, M1, m, depth + 1, max_depth);
        }
    }

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < m ; i++){
        for (int j = 0; j < m; j++){
            int k = i * m + j;
            C[i * ldc + j] = M1[k] + M4[k] - M5[k] + M7[k]; // C11
            C[i * ldc + (j + m)] = M3[k] + M5[k]; // C12
            C[(i + m) * ldc + j] = M2[k] + M4[k]; // C21
            C[(i + m) * ldc + (j + m)] = M1[k] - M2[k] + M3[k] + M6[k];
        }
    }
}

void strassenMatMul(
    int n, 
    const std::vector<float>& A, 
    const std::vector<float>& B, 
    std::vector<float>& C
){
    int k = THRESHOLD;
    int paddedSize = ((n + k - 1) / k) * k;
    int temp = paddedSize;
    while (temp > THRESHOLD) { // Limit depth
        temp /= 2;
    }    
    

    std::cout << "N=" << n << ", Padded=" << paddedSize 
              << ", Depth=" << MAX_DEPTH << std::endl;

    if (paddedSize == n){
        #pragma omp parallel
        {
            #pragma omp single
            {
                strassenParallel(n, A.data(), n, B.data(), n, C.data(), n, 0, MAX_DEPTH);
            }
        }
    } else {
        std::vector<float> AP(paddedSize * paddedSize, 0.0f);
        std::vector<float> BP(paddedSize * paddedSize, 0.0f);
        std::vector<float> CP(paddedSize * paddedSize, 0.0f);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            std::copy(A.begin() + i * n, A.begin() + (i + 1) * n, AP.begin() + i * paddedSize);
            std::copy(B.begin() + i * n, B.begin() + (i + 1) * n, BP.begin() + i * paddedSize);
        }

        #pragma omp parallel
        {
            #pragma omp single
            {
                strassenParallel(paddedSize, AP.data(), paddedSize, BP.data(), paddedSize, CP.data(), paddedSize, 0, MAX_DEPTH);
            } 
        }

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            std::copy(CP.begin() + i * paddedSize, CP.begin() + i * paddedSize + n, C.begin() + i * n);
        }
    }
}

int main(int argc, char ** argv){

    if (argc < 3){
        std::cerr << "Usage: " << " <matrix_size>" << "  <check err>  ";
        return 1;
    }
    int N = std::atoi(argv[1]);

    std::cout << "Initializing..." << std::endl;
    omp_set_num_threads(16); 
    auto A = createRandomMatrix(N, 123);
    auto B = createRandomMatrix(N, 456);
    std::vector<float> C(N * N, 0.0);

    std::cout << "Starting Strassen..." << std::endl;
    Timer t;
    t.start();
    
    strassenMatMul(N, A, B, C);
    
    float time = t.elapse();
    std::cout << "Time: " << time << "s" << std::endl;

    int check = std::atoi(argv[2]);
    if (check==0){
        return 0;
    }

    std::vector<float> CC(N * N, 0.0f);
    naiveMultiply(N, A.data(), N , B.data(), N, CC.data(), N);
    float diff_sum = 0.0, ref_sum = 0.0;
    for (int i = 0; i < N * N; ++i) {
        float diff = C[i] - CC[i];
        diff_sum += diff * diff;
        ref_sum += CC[i] * CC[i];
    }
    float rel_error = std::sqrt(diff_sum / (ref_sum + 1e-12));
    std::cout << "Relative L2 error between Strassen and naive: " << rel_error << "\n";

    return 0;
}
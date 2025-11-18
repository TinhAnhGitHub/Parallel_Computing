#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>
#include <cstring>
#define LOWER_B 0.0
#define UPPER_B 1.0
#define THRESHOLD 1024
#define MAX_DEPTH 1

class Timer{
    std::chrono::high_resolution_clock::time_point start_;
    public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double elapse(){
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

std::vector<float> transposeMatrix(int size, const std::vector<float>& M){
    std::vector<float> MT(size*size);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            MT[j * size + i] = M[i * size + j];
        }
    }
    return MT;
}
void naiveMultiply(
    int size,
    const std::vector<float>& A,
    const std::vector<float>& BT,
    std::vector<float>& C
){
    #pragma omp parallel for collapse(2) schedule(guided) default(none) shared(size, A, BT, C)
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            float sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < size; k++){
                sum += A[i * size + k] * BT[j * size + k];
            }
            C[i * size + j] = sum;
        }
    }
}
inline void addMatrix(
    int size,
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& result
){
    #pragma omp parallel for
    for(int i = 0; i < size * size; i++) {
        result[i] = A[i] + B[i];
    }
}

inline void subtract2Matrix(
    int size,
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& result
){
    #pragma omp parallel for
    for (int i = 0; i < size * size; i++){
        result[i] = A[i] - B[i];
    }
}

    void extractSubmat(
    int size,
    const std::vector<float>& source,
    int startRow,
    int startCol,
    std::vector<float>& dest,
    int destSize
){
    for(int i = 0; i < destSize; i++){
        for (int j = 0; j < destSize; j++){
            dest[i * destSize + j] = source[(i+startRow) * size + (j + startCol)];
        }
    }
}


inline void combine4SubMat(
    int m,
    const std::vector<float>& C11,
    const std::vector<float>& C12,
    const std::vector<float>& C21,
    const std::vector<float>& C22,
    std::vector<float>& result
){
    int N = 2 * m;
    #pragma omp parallel for default(none) shared(m, N, C11, C12, C21, C22, result)
    for (int i = 0; i < m; i++){
        std::copy(C11.begin() + i * m, C11.begin() + (i + 1) * m, result.begin() + i * N); // A11
        std::copy(C12.begin() + i * m, C12.begin() + (i + 1) * m, result.begin() + i * N + m); // A12
        std::copy(C21.begin() + i * m, C21.begin() + (i + 1) * m, result.begin() + (i + m) * N);
        std::copy(C22.begin() + i * m, C22.begin() + (i + 1) * m, result.begin() + (i + m) * N + m);
    }
}

void strassen(
    int size,
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int depth
){
    if (size < THRESHOLD){
        naiveMultiply(size, A, B, C);
        return;
    }
    if (size % 2 != 0){
        int newSize = size + 1;
        std::vector<float> A_padded(newSize * newSize, 0.0);
        std::vector<float> B_padded(newSize * newSize, 0.0);
        std::vector<float> C_padded(newSize * newSize, 0.0);

        for (int i = 0; i < size; i++){
            std::copy(A.begin() + i * size, A.begin() + (i + 1) * size, A_padded.begin() + i * newSize);
            std::copy(B.begin() + i * size, B.begin() + (i + 1) * size, B_padded.begin() + i * newSize);
        }
        strassen(newSize, A_padded, B_padded, C_padded, depth + 1);
        for (int i = 0; i < size; i++){
            std::copy(
                C_padded.begin() + i * newSize,
                C_padded.begin() + i * newSize + size,
                C.begin() + i * size
            );
        }
        return;
    }

    int m = size / 2;

    std::vector<float> A11(m * m), A12(m * m), A21(m * m), A22(m * m);
    std::vector<float> B11(m * m), B12(m * m), B21(m * m), B22(m * m);


    extractSubmat(size, A, 0, 0, A11, m);
    extractSubmat(size, A, 0, m, A12, m);
    extractSubmat(size, A, m, 0, A21, m);
    extractSubmat(size, A, m, m, A22, m);

    extractSubmat(size, B, 0, 0, B11, m);
    extractSubmat(size, B, 0, m, B12, m);
    extractSubmat(size, B, m, 0, B21, m);
    extractSubmat(size, B, m, m, B22, m);

    std::vector<float> M1(m * m, 0.0), M2(m * m, 0.0), M3(m * m, 0.0), M4(m * m, 0.0);
    std::vector<float> M5(m * m, 0.0), M6(m * m, 0.0), M7(m * m, 0.0);

    if (depth < MAX_DEPTH){
        

        #pragma omp parallel default(none) shared(depth, m, A11, A12, A21, A22, B11, B12, B21, B22, M1, M2, M3, M4, M5, M6, M7)
        {
            #pragma omp single nowait
            {
                #pragma omp task
                {
                    std::vector<float> temp1(m * m), temp2(m * m);
                    addMatrix(m, A11, A22, temp1); 
                    addMatrix(m, B11, B22, temp2); 
                    strassen(m, temp1, temp2, M1, depth+1);
                }

                #pragma omp task
                {
                    std::vector<float> temp(m * m);
                    addMatrix(m, A21, A22, temp);
                    strassen(m, temp, B11, M2, depth+1);
                }
                
                #pragma omp task
                {
                    std::vector<float> temp(m * m);
                    subtract2Matrix(m, B12, B22, temp); 
                    strassen(m, A11, temp, M3, depth + 1);
                }

                #pragma omp task
                {
                    std::vector<float> temp(m * m);
                    subtract2Matrix(m, B21, B11, temp); 
                    strassen(m, A22, temp, M4, depth+1); 
                }

                #pragma omp task
                {
                    std::vector<float> temp(m * m);
                    addMatrix(m, A11, A12, temp); 
                    strassen(m, temp, B22, M5, depth+1);
                }

                #pragma omp task
                {
                    std::vector<float> temp1(m * m), temp2(m * m);
                    subtract2Matrix(m, A21, A11, temp1); 
                    addMatrix(m, B11, B12, temp2);
                    strassen(m, temp1, temp2, M6, depth+1);
                }

                #pragma omp task
                {
                    std::vector<float> temp1(m * m), temp2(m * m);
                    subtract2Matrix(m, A12, A22, temp1); 
                    addMatrix(m, B21, B22, temp2);
                    strassen(m, temp1, temp2, M7, depth+1);
                }

            }
        }
    } else {
        std::vector<float> temp1(m * m), temp2(m * m);

        addMatrix(m, A11, A22, temp1);
        addMatrix(m, B11, B22, temp2);
        strassen(m, temp1, temp2, M1, depth + 1);

        addMatrix(m, A21, A22, temp1);
        strassen(m, temp1, B11, M2, depth+1);

        subtract2Matrix(m, B12, B22, temp2); 
        strassen(m, A11, temp2, M3, depth + 1);
        
        subtract2Matrix(m, B21, B11, temp2); 
        strassen(m, A22, temp2, M4, depth+1); 
        
        addMatrix(m, A11, A12, temp1); 
        strassen(m, temp1, B22, M5, depth+1);

        subtract2Matrix(m, A21, A11, temp1); 
        addMatrix(m, B11, B12, temp2);
        strassen(m, temp1, temp2, M6, depth+1);

        subtract2Matrix(m, A12, A22, temp1); 
        addMatrix(m, B21, B22, temp2);
        strassen(m, temp1, temp2, M7, depth+1);
    }


    std::vector<float> temp1(m * m), temp2(m * m);
    std::vector<float> C11(m * m), C12(m * m), C21(m * m), C22(m * m);
    #pragma omp parallel sections default(none) shared(m, M1, M2, M3, M4, M5, M6, M7, C11, C12, C21, C22, temp1, temp2)
    {
        #pragma omp section
        { // C11 = M1 + M4 - M5 + M7
            addMatrix(m, M1, M4, temp1);
            subtract2Matrix(m, temp1, M5, temp2);
            addMatrix(m, temp2, M7, C11);
        }


        #pragma omp section
        { // C12 = M3 + M5
            addMatrix(m, M3, M5, C12);
        }
        
        #pragma omp section
        { // C21 = M2 + M4
            addMatrix(m, M2, M4, C21);
        }
        
        #pragma omp section
        { // C22 = M1 - M2 + M3 + M6
            subtract2Matrix(m, M1, M2, temp1);
            addMatrix(m, temp1, M3, temp2);
            addMatrix(m, temp2, M6, C22);
        }

    }

    combine4SubMat(m, C11, C12, C21, C22, C);
};
int main() {
    const int SIZE = 10000;
    std::vector<float> A = createRandomMatrix(SIZE, 123);
    std::vector<float> B = createRandomMatrix(SIZE, 456);
    std::vector<float> BT = transposeMatrix(SIZE, B);
    std::vector<float> C(SIZE * SIZE, 0.0f);

    Timer t;
    t.start();
    strassen(SIZE, A, BT, C, 0);
    double time = t.elapse();

    std::cout << "Time: " << time << " seconds\n";
    return 0;
}

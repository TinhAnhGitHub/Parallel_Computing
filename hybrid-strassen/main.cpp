#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>

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
    #pragma omp parallel for  schedule(static) default(none) shared(n, A, B, C, lda, ldb, ldc)
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



void strassenSerial(int n, const float* A, int lda, const float* B, int ldb, float* C, int ldc, float* work) {
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

    #pragma omp parallel for collapse(2)
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
        
        #pragma omp task shared(results)
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


void strassen_mpi_wrapper(
    int N,  
    int rank,
    int numProcs,
    int* sendcounts,
    int* displs,
    const float* A, 
    int lda,
    const float* B,
    int ldb,
    float* C,
    int ldc,
    int recvCount,
    float* recvbuf,
    Timer* timer,
    int max_depth
){
  
    int totalSend = 0;
    int m = N / 2;
    for(int i = 0; i < numProcs; i++) totalSend += sendcounts[i];
    
    
    std::vector<float> scatterBuf(totalSend, 0.0f);

    const float* A11;
    const float* A12;
    const float* A21;
    const float* A22;
    const float* B11;
    const float* B12;
    const float* B21;
    const float* B22;
    
    if (rank == 0){
        A11 = A;
        A12 = A + m;
        A21 = A + m * lda;   
        A22 = A + m * lda + m;
        
        B11 = B;             
        B12 = B + m;
        B21 = B + m * ldb;   
        B22 = B + m * ldb + m;

        auto pack_submatrix = [&](float* dest, const float* src, int rows, int lddest, int ldsrc){
            for(int i = 0; i < rows; i++){
                for(int j = 0; j < rows; j++){
                    dest[i * lddest + j] = src[i * ldsrc + j];
                }
            }
        };
        pack_submatrix(scatterBuf.data() + displs[1], A11, m, m, lda);
        pack_submatrix(scatterBuf.data() + displs[1] + m*m , A22, m, m, lda);
        pack_submatrix(scatterBuf.data() + displs[1] + 2*m*m, B11, m,m,ldb);
        pack_submatrix(scatterBuf.data() + displs[1] + 3*m*m, B22, m,m,ldb);

        pack_submatrix(scatterBuf.data() + displs[2], A21, m,m,lda);
        pack_submatrix(scatterBuf.data() + displs[2] + m*m, A22, m,m,lda);
        pack_submatrix(scatterBuf.data() + displs[2]+ 2*m*m, B11, m,m,ldb);

        pack_submatrix(scatterBuf.data() + displs[3], A11, m,m,lda);
        pack_submatrix(scatterBuf.data() + displs[3] + m*m, B12, m,m,ldb);
        pack_submatrix(scatterBuf.data() + displs[3]+ 2*m*m, B22, m,m,ldb);

        pack_submatrix(scatterBuf.data() + displs[4], A22, m,m,lda);
        pack_submatrix(scatterBuf.data() + displs[4] + m*m, B21, m,m,ldb);
        pack_submatrix(scatterBuf.data() + displs[4]+ 2*m*m, B11, m,m,ldb);

        pack_submatrix(scatterBuf.data() + displs[5], A11, m,m,lda);
        pack_submatrix(scatterBuf.data() + displs[5] + m*m, A12, m,m,lda);
        pack_submatrix(scatterBuf.data() + displs[5]+ 2*m*m, B22, m,m,ldb);

        pack_submatrix(scatterBuf.data() + displs[6], A21, m, m, lda);
        pack_submatrix(scatterBuf.data() + displs[6] + m*m , A11, m, m, lda);
        pack_submatrix(scatterBuf.data() + displs[6] + 2*m*m, B11, m,m, ldb);
        pack_submatrix(scatterBuf.data() + displs[6] + 3*m*m, B12, m,m, ldb);
    }
    


    MPI_Scatterv(
        scatterBuf.data(), 
        sendcounts, 
        displs, 
        MPI_FLOAT,
        recvbuf, 
        recvCount, 
        MPI_FLOAT,
        0, 
        MPI_COMM_WORLD
    );

    std::vector<float> M(m*m, 0.0f);

    if (rank == 0){
        // calculate M7 = (A12 - A22)(B21 + B22)
        std::vector<float>T(2*m*m);
        float* t1 = &T[0]; float* t2 = &T[m*m];

        subtractMatrix(m, A12, lda, A22, lda, t1, m);
        addMatrix(m, B21, ldb, B22, ldb, t2, m);
        strassenParallel(m, t1, m, t2, m, M.data(),m , 0 , max_depth);
    }
    else if (rank == 1){
        // M1 = (A11 + A22)(B11 + B22)
        A11 = &recvbuf[0];
        A22 = &recvbuf[m*m];
        B11 = &recvbuf[2*m*m];
        B22 = &recvbuf[3*m*m];

        std::vector<float>T(2*m*m);
        float* t1 = &T[0]; float* t2 = &T[m*m];
        addMatrix(m, A11, m, A22, m, t1, m);
        addMatrix(m, B11, m, B22, m, t2, m);
        strassenParallel(m, t1, m, t2, m, M.data(), m, 0, max_depth);
        
        // std::cout<<"HI"<<std::endl;
        // printMatrix(m, M.data(), m);
        // std::cout<< std::endl;


    }else if (rank == 2){
        A21 = &recvbuf[0];
        A22 = &recvbuf[m*m];
        B11 = &recvbuf[2*m*m];

        std::vector<float>T(m*m);
        addMatrix(m, A21, m, A22, m, T.data(), m);
        strassenParallel(m, T.data(), m, B11, m, M.data(), m, 0, max_depth);

    }else if (rank == 3){
        A11 = &recvbuf[0];
        B12 = &recvbuf[m*m];
        B22 = &recvbuf[2*m*m];

        std::vector<float>T(m*m);
        subtractMatrix(m, B12, m, B22, m, T.data(), m);
        strassenParallel(m, A11, m, T.data(), m, M.data(), m , 0, max_depth);
    }else if (rank == 4){
        A22 = &recvbuf[0];
        B21 = &recvbuf[m*m];
        B11 = &recvbuf[2*m*m];

        std::vector<float>T(m*m);
        subtractMatrix(m, B21, m, B11, m, T.data(), m);
        strassenParallel(m, A22, m, T.data(), m, M.data(), m, 0, max_depth);
    }else if (rank == 5){
        A11 = &recvbuf[0];
        A12 = &recvbuf[m*m];
        B22 = &recvbuf[2*m*m];

        std::vector<float>T(m*m);
        addMatrix(m, A11, m, A12, m, T.data(), m);
        strassenParallel(m, T.data(), m, B22, m, M.data(), m, 0, max_depth);
    }else if (rank == 6){
        A21 = &recvbuf[0];
        A11 = &recvbuf[m*m];
        B11 = &recvbuf[2*m*m];
        B12 = &recvbuf[3*m*m];

        std::vector<float>T(2*m*m);
        float* t1 = &T[0]; float* t2 = &T[m*m];

        subtractMatrix(m, A21, m, A11, m, t1, m);
        addMatrix(m, B11, m, B12, m, t2, m);
        strassenParallel(m, t1, m, t2, m, M.data() ,m , 0 , max_depth);
    }
    
    std::vector<float> gatherBuf;
    if (rank == 0) gatherBuf.resize(numProcs * m * m, 0.0f);
    MPI_Gather(
        M.data(), 
        m*m, 
        MPI_FLOAT, 
        gatherBuf.data(), 
        m*m, 
        MPI_FLOAT,
        0, 
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        std::cout<< gatherBuf.size() << std::endl;
        float* M7 = gatherBuf.data();
        float* M1 = M7 + m*m;
        float* M2 = M1 + m*m;
        float* M3 = M2 + m*m;
        float* M4 = M3 + m*m;
        float* M5 = M4 + m*m;
        float* M6 = M5 + m*m;
       

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
        float totalTime = timer->elapse();
        std::cout << "Strassen completed in " << totalTime << " seconds.\n";
    }   
}

int main(int argc, char**argv){
    int rank, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (numProcs != 7){
        if (rank == 0) {
            std::cerr << "Error: This implementation requires exactly 7 processes.\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << " <matrix_size>" << "  <check err>  ";
        }
        MPI_Finalize();
        return 1;
    }

    int N = std::atoi(argv[1]);
    Timer timer;

    int k = THRESHOLD;
    int paddedSize = ((N + k - 1) / k) * k;

    int temp = paddedSize;
    while (temp > THRESHOLD) { // Limit depth
        temp /= 2;
    } 
    
    if (rank == 0){
        std::cout << "N=" << N << ", Padded=" << paddedSize 
            << ", Depth=" << MAX_DEPTH << std::endl;
    }

    std::vector<float> A, B, C(N*N, 0.0);
    if (paddedSize != N){
        std::vector<float> A_padded, B_padded, C_padded;
        if (rank == 0){

            
            A = createRandomMatrix(N, 123);
            B = createRandomMatrix(N, 456);
            
            A_padded.resize(paddedSize * paddedSize, 0.0);
            B_padded.resize(paddedSize * paddedSize, 0.0);
            C_padded.resize(paddedSize * paddedSize, 0.0);
            
            
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    A_padded[i * paddedSize + j] = A[i * N + j];
                    B_padded[i * paddedSize + j] = B[i * N + j];
                }
            }
            A.clear();
            B.clear();
            
        }
        

        int m = paddedSize / 2;
        timer.start();
        std::vector<int> sendcounts (7, 0.0);
        sendcounts[0]= 0;
        sendcounts[1] = 4*m*m;
        sendcounts[2] = 3*m*m;
        sendcounts[3] = 3*m*m;
        sendcounts[4] = 3*m*m;
        sendcounts[5] = 3*m*m;
        sendcounts[6] = 4*m*m;
    
        std::vector<int> displs(7, 0);
        for(int i = 1; i < 7; i++){
            displs[i] = sendcounts[i-1] + displs[i-1];
        }
        MPI_Bcast(sendcounts.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(displs.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);

        int recvCount = sendcounts[rank];
        std::vector<float> recvbuf(recvCount, 0.0);

        strassen_mpi_wrapper(
            paddedSize,
            rank,
            numProcs, 
            sendcounts.data(),
            displs.data(),
            A_padded.data(),
            paddedSize,
            B_padded.data(),
            paddedSize,
            C_padded.data(),
            paddedSize,
            recvCount,
            recvbuf.data(),
            &timer,
            MAX_DEPTH
        );

        
        if (rank == 0){
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                std::copy(C_padded.begin() + i * paddedSize, C_padded.begin() + i * paddedSize + N, C.begin() + i * N);
            }
        }

    } else {

        int m = N / 2;
        timer.start();
        std::vector<int> sendcounts (7, 0.0);
        sendcounts[0]= 0;
        sendcounts[1] = 4*m*m;
        sendcounts[2] = 3*m*m;
        sendcounts[3] = 3*m*m;
        sendcounts[4] = 3*m*m;
        sendcounts[5] = 3*m*m;
        sendcounts[6] = 4*m*m;
    
        std::vector<int> displs(7, 0);
        for(int i = 1; i < 7; i++){
            displs[i] = sendcounts[i-1] + displs[i-1];
        }
        MPI_Bcast(sendcounts.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(displs.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);

        int recvCount = sendcounts[rank];
        std::vector<float> recvbuf(recvCount, 0.0);

        if (rank == 0){
            A = createRandomMatrix(N, 123);
            B = createRandomMatrix(N, 456);
        }

        strassen_mpi_wrapper(
            N,
            rank,
            numProcs, 
            sendcounts.data(),
            displs.data(),
            A.data(),
            N,
            B.data(),
            N,
            C.data(),
            N,
            recvCount,
            recvbuf.data(),
            &timer,
            MAX_DEPTH
        );
    }

    if (rank == 0){
        int check = std::atoi(argv[2]);
        if (check == 1) {

            std::vector<float> A, B;
            A = createRandomMatrix(N, 123);
            B = createRandomMatrix(N, 456);
            std::vector<float> CC(N*N, 0.0f);
            Timer naive_timer;
            naive_timer.start();
            naiveMultiply(N, A.data(), N, B.data(), N, CC.data(), N);
            float naiveTime = naive_timer.elapse();
            std::cout << "Naive completed in " << naiveTime << " seconds.\n";


            float diff_sum = 0.0, ref_sum = 0.0;
            for (int i = 0; i < N*N; ++i) {
                float d = C[i] - CC[i];
                diff_sum += d*d;
                ref_sum += CC[i]*CC[i];
            }
            float rel_error = std::sqrt(diff_sum / (ref_sum + 1e-12));
            std::cout << "Relative L2 error: " << rel_error << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
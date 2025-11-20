
//Assumption: Running the MPI Strassen algorithm on 8 process.
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>

#define LOWER_B 0.0
#define UPPER_B 1.0
#define THRESHOLD 128


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


void naiveMultiply(
    int size, 
    const std::vector<float>& A, 
    const std::vector<float>& B,
    std::vector<float>& C
){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            float sum = 0.0;
            for (int k = 0; k < size; k++){
                sum += A[i * size + k] * B[k* size + j];
            }
            C[i * size + j] = sum;
        }
    }
}

std::vector<float> createRandomMatrix(int size, int seed){
    std::vector<float> matrix(size*size);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(LOWER_B, UPPER_B);
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = dist(rng);
    }
    return matrix;
}


inline void add2Matrix(
    int size, 
    const std::vector<float>& A, 
    const std::vector<float>& B,
    std::vector<float>& result
){
    for (int i = 0; i < size * size; i++){
        result[i] = A[i] + B[i];
    }
}


inline void subtract2Matrix(
    int size, 
    const std::vector<float>& A, 
    const std::vector<float>& B,
    std::vector<float>& result
){
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

void combine4SubMat(
    int m,
    const std::vector<float>& C11,
    const std::vector<float>& C12,
    const std::vector<float>& C21,
    const std::vector<float>& C22,
    std::vector<float>& result
){
    int N = 2 * m;
    for (int i = 0; i < m; i++){
        std::copy(C11.begin() + i * m, C11.begin() + (i + 1) * m, result.begin() + i * N); // A11
        std::copy(C12.begin() + i * m, C12.begin() + (i + 1) * m, result.begin() + i * N + m); // A12
    }
    for (int i = 0; i < m; i++){
        std::copy(C21.begin() + i * m, C21.begin() + (i + 1) * m, result.begin() + (i + m) * N);
        std::copy(C22.begin() + i * m, C22.begin() + (i + 1) * m, result.begin() + (i + m) * N + m);
    }
}

void strassen(
    int size, 
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C
){
    if (size < THRESHOLD){
        naiveMultiply(size, A, B, C);
        return;
    }

    if (size % 2 != 0){
        // padded
        int newSize = size + 1;
        std::vector<float> A_padded(newSize * newSize, 0.0);
        std::vector<float> B_padded(newSize * newSize, 0.0);
        std::vector<float> C_padded(newSize * newSize, 0.0);

        for (int i = 0; i < size; i++){
            std::copy(A.begin() + i * size, A.begin() + (i + 1) * size, A_padded.begin() + i * newSize);
            std::copy(B.begin() + i * size, B.begin() + (i + 1) * size, B_padded.begin() + i * newSize);
        }
        strassen(newSize, A_padded, B_padded, C_padded);
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

    std::vector<float> temp1(m * m), temp2(m * m), temp3(m * m);

    // M1 = (A11 + A22) * (B11 + B22)
    add2Matrix(m, A11, A22, temp1);
    add2Matrix(m, B11, B22, temp2);
    strassen(m, temp1, temp2, M1);

    // M2 = (A21 + A22) * B11
    add2Matrix(m, A21, A22, temp1);
    strassen(m, temp1, B11, M2);
    
    // M3 = A11 * (B12 - B22)
    subtract2Matrix(m, B12, B22, temp2);
    strassen(m, A11, temp2, M3);
    
    // M4 = A22 * (B21 - B11)
    subtract2Matrix(m, B21, B11, temp2);
    strassen(m, A22, temp2, M4);
    
    // M5 = (A11 + A12) * B22
    add2Matrix(m, A11, A12, temp1);
    strassen(m, temp1, B22, M5);
    
    // M6 = (A21 - A11) * (B11 + B12)
    subtract2Matrix(m, A21, A11, temp1);
    add2Matrix(m, B11, B12, temp2);
    strassen(m, temp1, temp2, M6);

    // M7 = (A12 - A22) * (B21 + B22)
    subtract2Matrix(m, A12, A22, temp1);  
    add2Matrix(m, B21, B22, temp2);
    strassen(m, temp1, temp2, M7);

    std::vector<float> C11(m * m), C12(m * m), C21(m * m), C22(m * m);
    
    // C11 = M1 + M4 - M5 + M7
    add2Matrix(m, M1, M4, temp1);
    subtract2Matrix(m, temp1, M5, temp2);
    add2Matrix(m, temp2, M7, C11);
    
    // C12 = M3 + M5
    add2Matrix(m, M3, M5, C12);
    
    // C21 = M2 + M4
    add2Matrix(m, M2, M4, C21);
    
    // C22 = M1 - M2 + M3 + M6
    subtract2Matrix(m, M1, M2, temp1);
    add2Matrix(m, temp1, M3, temp2);
    add2Matrix(m, temp2, M6, C22);
    
    // Combine into final result
    combine4SubMat(m, C11, C12, C21, C22, C);

}



void worker(int rank, int N){
    int m = N / 2;

    MPI_Status status;

    int numMatrices = (rank == 1 || rank == 6 || rank == 7) ? 4 : 3;
    std::vector<float> buffer(numMatrices * m * m);
    std::vector<float> M(m * m, 0.0);

    MPI_Recv(buffer.data(), numMatrices * m * m, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
    
    switch(rank){
        case 1: {
            // M1 = (A11 + A22) * (B11 + B22)
            std::vector<float> A11(m*m), A22(m*m), B11(m*m), B22(m*m);
            A11.assign(buffer.begin(), buffer.begin() + m*m);
            A22.assign(buffer.begin() + m*m, buffer.begin() + 2*m*m);
            B11.assign(buffer.begin() + 2*m*m, buffer.begin() + 3*m*m);
            B22.assign(buffer.begin() + 3*m*m, buffer.end());

            std::vector<float> sumA(m * m), sumB(m * m);
            add2Matrix(m, A11, A22, sumA);
            add2Matrix(m, B11, B22, sumB);
            strassen(m, sumA, sumB, M);
            break;
        }
        case 2: { // M2 = (A21 + A22) * B11
            std::vector<float> A21(m * m), A22(m * m), B11(m * m);
            A21.assign(buffer.begin(), buffer.begin() + m*m);
            A22.assign(buffer.begin() + m*m, buffer.begin() + 2*m*m);
            B11.assign(buffer.begin() + 2*m*m, buffer.end());
            
            std::vector<float> sumA(m * m);
            add2Matrix(m, A21, A22, sumA);
            strassen(m, sumA, B11, M);
            break;
        }
        case 3: { // M3 = A11 * (B12 - B22)
            std::vector<float> A11(m * m), B12(m * m), B22(m * m);
            A11.assign(buffer.begin(), buffer.begin() + m*m);
            B12.assign(buffer.begin() + m*m, buffer.begin() + 2*m*m);
            B22.assign(buffer.begin() + 2*m*m, buffer.end());
            
            std::vector<float> diffB(m * m);
            subtract2Matrix(m, B12, B22, diffB);
            strassen(m, A11, diffB, M);
            break;
        }
        case 4 : { // M4 = A22 * (B21 - B11)
            std::vector<float> A22(m * m), B21(m * m), B11(m * m);
            A22.assign(buffer.begin(), buffer.begin() + m*m);
            B21.assign(buffer.begin() + m*m, buffer.begin() + 2*m*m);
            B11.assign(buffer.begin() + 2*m*m, buffer.end());
            
            std::vector<float> diffB(m * m);
            subtract2Matrix(m, B21, B11, diffB);
            strassen(m, A22, diffB, M);
            break;
        }
        case 5 : { // M5 = (A11 + A12) * B22
            std::vector<float> A11(m * m), A12(m * m), B22(m * m);
            A11.assign(buffer.begin(), buffer.begin() + m*m);
            A12.assign(buffer.begin() + m*m, buffer.begin() + 2*m*m);
            B22.assign(buffer.begin() + 2*m*m, buffer.end());
            
            std::vector<float> sumA(m * m);
            add2Matrix(m, A11, A12, sumA);
            strassen(m, sumA, B22, M);
            break;
        }
        case 6: { // M6 = (A21 - A11) * (B11 + B12)
            std::vector<float> A21(m * m), A11(m * m), B11(m * m), B12(m * m);
            A21.assign(buffer.begin(), buffer.begin() + m*m);
            A11.assign(buffer.begin() + m*m, buffer.begin() + 2*m*m);
            B11.assign(buffer.begin() + 2*m*m, buffer.begin() + 3*m*m);
            B12.assign(buffer.begin() + 3*m*m, buffer.end());
            
            std::vector<float> diffA(m * m), sumB(m * m);
            subtract2Matrix(m, A21, A11, diffA);
            add2Matrix(m, B11, B12, sumB);
            strassen(m, diffA, sumB, M);
            break;
        }
        case 7 : {  // M7 = (A12 - A22) * (B21 + B22)
            std::vector<float> A12(m * m), A22(m * m), B21(m * m), B22(m * m);
            A12.assign(buffer.begin(), buffer.begin() + m*m);
            A22.assign(buffer.begin() + m*m, buffer.begin() + 2*m*m);
            B21.assign(buffer.begin() + 2*m*m, buffer.begin() + 3*m*m);
            B22.assign(buffer.begin() + 3*m*m, buffer.end());
            
            
            std::vector<float> diffA(m * m), sumB(m * m);
            subtract2Matrix(m, A12, A22, diffA);
            add2Matrix(m, B21, B22, sumB);
            strassen(m, diffA, sumB, M);
            break;
        }
    }

    MPI_Send(M.data(), m*m, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
}


// N will always be 100, 1000, 10000
// ./main <size matrix: 100> <test mat correct:0 or 1 for mat <= 100>
int main(int argc, char ** argv){
    int rank, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (numProcs != 8){
        if (rank == 0) {
            std::cerr << "Error: This implementation requires exactly 8 processes.\n";
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

    Timer timer;
    double totalTime = 0.0;

    int N = std::atoi(argv[1]);
    std::vector<float> C(N * N, 0.0f);
    std::vector<float> A, B; 

    if (rank == 0){
        std::cout << "Initializing matrices of size " << N << "x" << N << "...\n";
        A = createRandomMatrix(N, 123);
        B = createRandomMatrix(N, 456);
        int m = N / 2; 
        timer.start();
        std::vector<float> A11(m*m), A12(m*m), A21(m*m), A22(m*m);
        std::vector<float> B11(m*m), B12(m*m), B21(m*m), B22(m*m);

        extractSubmat(N, A, 0, 0, A11, m);
        extractSubmat(N, A, 0, m, A12, m);
        extractSubmat(N, A, m, 0, A21, m);
        extractSubmat(N, A, m, m, A22, m);
        
        extractSubmat(N, B, 0, 0, B11, m);
        extractSubmat(N, B, 0, m, B12, m);
        extractSubmat(N, B, m, 0, B21, m);
        extractSubmat(N, B, m, m, B22, m);

        MPI_Request recvReqs[7];
        std::vector<float> M1(m*m), M2(m*m), M3(m*m), M4(m*m), M5(m*m), M6(m*m), M7(m*m);
        
        MPI_Irecv(M1.data(), m*m, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &recvReqs[0]);
        MPI_Irecv(M2.data(), m*m, MPI_FLOAT, 2, 2, MPI_COMM_WORLD, &recvReqs[1]);
        MPI_Irecv(M3.data(), m*m, MPI_FLOAT, 3, 3, MPI_COMM_WORLD, &recvReqs[2]);
        MPI_Irecv(M4.data(), m*m, MPI_FLOAT, 4, 4, MPI_COMM_WORLD, &recvReqs[3]);
        MPI_Irecv(M5.data(), m*m, MPI_FLOAT, 5, 5, MPI_COMM_WORLD, &recvReqs[4]);
        MPI_Irecv(M6.data(), m*m, MPI_FLOAT, 6, 6, MPI_COMM_WORLD, &recvReqs[5]);
        MPI_Irecv(M7.data(), m*m, MPI_FLOAT, 7, 7, MPI_COMM_WORLD, &recvReqs[6]);

        MPI_Request sendReqs[7];

        std::vector<float> buf1 (4*m*m);
        std::copy(A11.begin(), A11.end(), buf1.begin());
        std::copy(A22.begin(), A22.end(), buf1.begin() + m*m);
        std::copy(B11.begin(), B11.end(), buf1.begin() + m*m*2);
        std::copy(B22.begin(), B22.end(), buf1.begin() + m*m*3);
        MPI_Isend(buf1.data(), 4*m*m, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &sendReqs[0]);

        std::vector<float> buf2(3*m*m);
        std::copy(A21.begin(), A21.end(), buf2.begin());
        std::copy(A22.begin(), A22.end(), buf2.begin() + m*m);
        std::copy(B11.begin(), B11.end(), buf2.begin() + 2*m*m);
        MPI_Isend(buf2.data(), 3*m*m, MPI_FLOAT, 2, 0, MPI_COMM_WORLD, &sendReqs[1]);

        std::vector<float> buf3(3*m*m);
        std::copy(A11.begin(), A11.end(), buf3.begin());
        std::copy(B12.begin(), B12.end(), buf3.begin() + m*m);
        std::copy(B22.begin(), B22.end(), buf3.begin() + 2*m*m);
        MPI_Isend(buf3.data(), 3*m*m, MPI_FLOAT, 3, 0, MPI_COMM_WORLD, &sendReqs[2]);

        std::vector<float> buf4(3*m*m);
        std::copy(A22.begin(), A22.end(), buf4.begin());
        std::copy(B21.begin(), B21.end(), buf4.begin() + m*m);
        std::copy(B11.begin(), B11.end(), buf4.begin() + 2*m*m);
        MPI_Isend(buf4.data(), 3*m*m, MPI_FLOAT, 4, 0, MPI_COMM_WORLD, &sendReqs[3]);

        std::vector<float> buf5(3*m*m);
        std::copy(A11.begin(), A11.end(), buf5.begin());
        std::copy(A12.begin(), A12.end(), buf5.begin() + m*m);
        std::copy(B22.begin(), B22.end(), buf5.begin() + 2*m*m);
        MPI_Isend(buf5.data(), 3*m*m, MPI_FLOAT, 5, 0, MPI_COMM_WORLD, &sendReqs[4]);

        std::vector<float> buf6(4*m*m);
        std::copy(A21.begin(), A21.end(), buf6.begin());
        std::copy(A11.begin(), A11.end(), buf6.begin() + m*m);
        std::copy(B11.begin(), B11.end(), buf6.begin() + 2*m*m);
        std::copy(B12.begin(), B12.end(), buf6.begin() + 3*m*m);
        MPI_Isend(buf6.data(), 4*m*m, MPI_FLOAT, 6, 0, MPI_COMM_WORLD, &sendReqs[5]);

        std::vector<float> buf7(4*m*m);
        std::copy(A12.begin(), A12.end(), buf7.begin());
        std::copy(A22.begin(), A22.end(), buf7.begin() + m*m);
        std::copy(B21.begin(), B21.end(), buf7.begin() + 2*m*m);
        std::copy(B22.begin(), B22.end(), buf7.begin() + 3*m*m);
        MPI_Isend(buf7.data(), 4*m*m, MPI_FLOAT, 7, 0, MPI_COMM_WORLD, &sendReqs[6]);

        int recvCount = 0;
        MPI_Status statuses[7];

        while (recvCount < 7) {   
            int idx;
            MPI_Waitany(7, recvReqs, &idx, MPI_STATUS_IGNORE);
            if (idx != MPI_UNDEFINED) {
                recvCount++;
            }
        }

        MPI_Waitall(7, sendReqs, MPI_STATUSES_IGNORE);
        
        std::vector<float> C11(m*m), C12(m*m), C21(m*m), C22(m*m);
        std::vector<float> temp1(m*m), temp2(m*m);

        // C11 = M1 + M4 - M5 + M7
        add2Matrix(m, M1, M4, temp1);
        subtract2Matrix(m, temp1, M5, temp2);
        add2Matrix(m, temp2, M7, C11);
        
        // C12 = M3 + M5
        add2Matrix(m, M3, M5, C12);
        
        // C21 = M2 + M4
        add2Matrix(m, M2, M4, C21);
        
        // C22 = M1 - M2 + M3 + M6
        subtract2Matrix(m, M1, M2, temp1);
        add2Matrix(m, temp1, M3, temp2);
        add2Matrix(m, temp2, M6, C22);
        
        combine4SubMat(m, C11, C12, C21, C22, C);
        
        totalTime = timer.elapse();
        std::cout << "Strassen completed in " << totalTime << " seconds.\n";
    } else {
        worker(rank, N);
    }
    

    if (rank == 0){

        int check = std::atoi(argv[2]);
        if (check == 0) {
            return 0;
        }
        
        std::vector<float> CC(N * N, 0.0f);
        
        
        Timer naive_timer;
        naive_timer.start();
        naiveMultiply(N, A, B, CC);
        double naiveTotalTime = naive_timer.elapse();
        std::cout << "Naive completed in " << naiveTotalTime << " seconds.\n";
        
        double diff_sum = 0.0, ref_sum = 0.0;
        for (int i = 0; i < N * N; ++i) {
                double diff = C[i] - CC[i];
                diff_sum += diff * diff;
                ref_sum += CC[i] * CC[i];
            }
        double rel_error = std::sqrt(diff_sum / (ref_sum + 1e-12));
        std::cout << "Relative L2 error between Strassen and naive: " << rel_error << "\n";
            
    }
    MPI_Finalize();

    return 0;
}
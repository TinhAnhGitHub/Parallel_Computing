#include <iostream>
#include <cmath>
#include <chrono>
#include <mpi.h>

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


void generateMatrix(std::vector<float>& mat, int N, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(LOWER_B, UPPER_B);
    for (int i = 0; i < N * N; ++i) {
        mat[i] = dist(rng);
    }
}

void naiveAddMultiply(
    int rows,       
    int cols,       
    const float* A, 
    const float* B, 
    float* C        
){
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < cols; ++k) {
            
            float a_ik = A[i * cols + k];
            
            #pragma omp simd
            for (int j = 0; j < cols; ++j) {
                C[i * cols + j] += a_ik * B[k * cols + j];
            }
        }
    }
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank, num_procs; 

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <N> [check=0/1]\n";
        }
        MPI_Finalize();
        return 1;
    }

    int N = std::atoi(argv[1]);
    int check = (argc > 2) ? std::atoi(argv[2]) : 0;

    int base_rows = N / num_procs;
    int remainder = N % num_procs;

    int local_rows = base_rows + (rank < remainder? 1 : 0);

    std::vector<int> sendcounts;
    std::vector<int> displs;

    if (rank == 0){
        sendcounts.resize(num_procs);
        displs.resize(num_procs);

        int offset = 0;
        for(int i = 0; i < num_procs; i++){
            int rows = base_rows + (i < remainder? 1 : 0);
            sendcounts[i] = rows * N;
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }


    std::vector<float> full_A;
    std::vector<float> full_C;
    std::vector<float> full_B(N * N);

    std::vector<float> local_A (local_rows* N);
    std::vector<float> local_C (local_rows *N, 0.0);

    if (rank == 0) {
        std::cout << "Initializing Hybrid MPI+OpenMP..." << std::endl;
        std::cout << "MPI Ranks: " << num_procs << std::endl;
        std::cout << "Matrix Size: " << N << "x" << N << std::endl;
        
        full_A.resize(N * N);
        full_C.resize(N * N);
        
        generateMatrix(full_A, N, 123);
        generateMatrix(full_B, N, 456);



    }

    double start_time = MPI_Wtime();
    MPI_Scatterv(
        (rank == 0? full_A.data() : nullptr),
        (rank == 0 ? sendcounts.data() : nullptr),
        (rank == 0 ? displs.data() : nullptr),
        MPI_FLOAT,
        local_A.data(),
        local_rows * N,
        MPI_FLOAT,
        0, 
        MPI_COMM_WORLD
    );

    MPI_Bcast(
        full_B.data(),
        N * N,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    double local_compute_start = MPI_Wtime();
    
    naiveAddMultiply(local_rows, N, local_A.data(), full_B.data(), local_C.data());
    double local_compute_end = MPI_Wtime();
    double local_duration = local_compute_end - local_compute_start;

    MPI_Gatherv(
        local_C.data(),
        local_rows * N, 
        MPI_FLOAT,
        (rank == 0 ? full_C.data() : nullptr),
        (rank == 0 ? sendcounts.data() : nullptr),
        (rank == 0 ? displs.data() : nullptr),
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    double end_time = MPI_Wtime();

    


    if (rank == 0) {
        std::cout << "Total Time: " << (end_time - start_time) << "s" << std::endl;
        
        if (check) {
            std::cout << "Verifying results (Serial check)..." << std::endl;
            std::vector<float> ref_C(N * N, 0.0f);
            
            for(int i = 0; i < N; ++i){
                for(int k = 0; k < N; ++k){
                    float val = full_A[i * N + k];
                    for (int j = 0; j < N; ++j){
                        ref_C[i * N + j] += val * full_B[k * N + j];
                    }
                }
            }

            float diff_sum = 0.0, ref_sum = 0.0;
            for (int i = 0; i < N * N; ++i) {
                float diff = full_C[i] - ref_C[i];
                diff_sum += diff * diff;
                ref_sum += ref_C[i] * ref_C[i];
            }
            float rel_error = std::sqrt(diff_sum / (ref_sum + 1e-12));
            std::cout << "Relative Error: " << rel_error << std::endl;
        }
    }

    double max_compute_time = 0.0;
    MPI_Reduce(&local_duration, &max_compute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Max local compute time: " << max_compute_time << "s" << std::endl;
    }
    MPI_Finalize();
    return 0;



    
}



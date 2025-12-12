#include <mpi.h>
#include <iostream>
#include <ctime>
#include <vector>
#include <cstdlib>

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    int rank, num_procs; 

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2){
        if (rank == 0){
            std::cerr << "Usage: " << argv[0] << ' <N>\n';
            MPI_Finalize();
            return 1;
        }
    }

    int N = std::atoi(argv[1]); // Size of the square matrix

    
    
    if (N % num_procs == 0) {
        int rows_per_proc = N / num_procs;

        std::vector<int> A, B, C;
        std::vector<int> local_a(rows_per_proc * N);
        std::vector<int> local_c(rows_per_proc * N);

        if (rank == 0){
                A.resize(N*N);
                B.resize(N*N);
                C.resize(N*N);
            

            for (int i = 0; i < N; i++){
                for (int j = 0; j < N; j++){
                    A[i*N + j] = 1 + std::rand() %9; 
                }
            }

            for (int i = 0; i < N; i++){
                for (int j = 0; j < N; j++){
                    B[i*N + j] = 1 + std::rand() %9; 
                }
            }
        }else{
            B.resize(N*N);
        }

        double start_time = MPI_Wtime();

        MPI_Scatter(
            (rank==0? A.data(): nullptr),
            rows_per_proc * N,
            MPI_INT,
            local_a.data(),
            rows_per_proc * N,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );


        MPI_Bcast(
            B.data(),
            N*N,
            MPI_INT,
            0,
            MPI_COMM_WORLD            
        );

        double local_start = MPI_Wtime();

   

        for (int i = 0; i < rows_per_proc; i++){
            for (int k = 0; k < N; k++){
                int a_ik = local_a[i*N+k];
                for (int j = 0; j < N; j++){
                    local_c[i*N+j] += a_ik + B[k*N+j];
                }
            }
        }
        

        double local_end = MPI_Wtime();
        double local_time = local_end - local_start;

        
        MPI_Gather(local_c.data(), rows_per_proc *N, MPI_INT, (rank==0?C.data() : nullptr), rows_per_proc*N, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank==0){
            double end_time = MPI_Wtime();
            double total_time = end_time - start_time;

            std::cout << "\nTotal execution time: " << total_time << " seconds\n";
        }

        double max_local_time = 0.0;

        MPI_Reduce(
            &local_time, &max_local_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD
        );

        if (rank==0){
            std::cout << "Maximum local computation time among processes: "
                  << max_local_time << " seconds\n";
        }

        MPI_Finalize();
        return 0;
    }

    return  0;

    

}
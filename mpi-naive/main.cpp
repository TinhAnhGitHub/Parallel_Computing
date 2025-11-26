#include "mpi-naive.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <N> [verify]\n";
            std::cerr << "  N: Matrix size\n";
            std::cerr << "  verify: 0=skip, 1=verify (default: 0)\n";
        }
        MPI_Finalize();
        return 1;
    }

    int N = std::atoi(argv[1]); // Size of the square matrix
    int verify = (argc > 2) ? std::atoi(argv[2]) : 0;

    if (N % num_procs != 0)
    {
        if (rank == 0)
            std::cerr << "Error: N must be divisible by number of processes\n";
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / num_procs;
    std::vector<int> A, B, C;
    std::vector<int> local_a(rows_per_proc * N);
    std::vector<int> local_c(rows_per_proc * N, 0);

    initializeMatrices(N, rank, A, B, C);
    double start_time = MPI_Wtime();
    distributeMatrices(N, rank, A, local_a, B, rows_per_proc);

    double local_time;
    localMatrixComputation(N, rows_per_proc, local_a, B, local_c, local_time);

    gatherResults(N, rank, rows_per_proc, local_c, C);

    if (rank == 0)
    {
        double total_time = MPI_Wtime() - start_time; // Re-measure total outside
        std::cout << "\nTotal execution time: " << total_time << " seconds\n";
    }

    double max_local_time = computeMaxLocalTime(local_time, rank);
    if (rank == 0)
    {
        std::cout << "Maximum local computation time among processes: " << max_local_time << " seconds\n";
    }

    // Verification (optional)
    if (verify && rank == 0)
    {
        std::cout << "\n================================================" << std::endl;
        std::cout << "Verifying Correctness..." << std::endl;
        std::cout << "================================================" << std::endl;

        double verify_start = MPI_Wtime();
        std::vector<int> C_verify(N * N, 0);
        serialVerify(N, A, B, C_verify);
        double verify_time = MPI_Wtime() - verify_start;

        std::cout << "Serial verification time: " << verify_time << "s" << std::endl;

        bool passed = verifyResults(N, C, C_verify, rank);

        if (passed)
        {
            double total_time = MPI_Wtime() - start_time;
            std::cout << "\nSpeedup vs Serial: " << std::fixed << std::setprecision(2)
                      << verify_time / total_time << "x" << std::endl;
        }
        std::cout << "================================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}

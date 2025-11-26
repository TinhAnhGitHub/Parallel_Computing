#ifndef MPI_NAIVE_H
#define MPI_NAIVE_H

#include <mpi.h>
#include <iostream>
#include <ctime>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <iomanip>

void initializeMatrices(int N, int rank, std::vector<int>& A, std::vector<int>& B, std::vector<int>& C);

void distributeMatrices(int N, int rank, const std::vector<int>& A, std::vector<int>& local_a, std::vector<int>& B, int rows_per_proc);

void localMatrixComputation(int N, int rows_per_proc, const std::vector<int>& local_a, const std::vector<int>& B, std::vector<int>& local_c, double& local_time);

void gatherResults(int N, int rank, int rows_per_proc, const std::vector<int>& local_c, std::vector<int>& C);

double computeMaxLocalTime(double local_time, int rank);

void serialVerify(int N, const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C_verify);

bool verifyResults(int N, const std::vector<int>& C, const std::vector<int>& C_verify, int rank);

#endif 

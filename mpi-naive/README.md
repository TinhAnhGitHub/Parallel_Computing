# MPI Naive Square Matrix Multiply (Scaffold)

Tiny scaffold for a straightforward MPI implementation of C = A × B for square matrices (N×N, double).

## Goal
- Compute C[N×N] from A[N×N], B[N×N] with MPI.
- Row partitioning: each rank gets rows of A; all ranks get B; root gathers C.

## Prerequisites
- OpenMPI or MPICH (`mpicc`, `mpirun`).
- C compiler available via `mpicc`.

## Layout (planned)
- `mpi-naive/README.md` (this file)
- `mpi-naive/main.c` (to be created)
- `mpi-naive/Makefile` (to be created)

## Checkpoints
- [-] Init MPI, parse: `N [--seed S] [--verify]`.
- [ ] Rank 0: create A[N×N], B[N×N] (pattern or random).
- [ ] Compute rows per rank (block rows; handle remainder with `Scatterv`).
- [ ] Broadcast B to all ranks.
- [ ] Scatter rows of A to each rank.
- [ ] Each rank: allocate `C_local[rows_i×N]` and run naive triple loop.
- [ ] Gather `C_local` at rank 0 to form C.
- [ ] Rank 0: optional serial verify for small N (`--verify`).
- [ ] Rank 0: print timing and a checksum.

## Interface (planned)
- Executable: `./mm_mpi_naive N [--seed S] [--verify]`.
- Data type: `double`.

## Build (planned)
```
CC=mpicc
CFLAGS=-O2 -Wall -Wextra

all: mm_mpi_naive

mm_mpi_naive: main.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f mm_mpi_naive
```

Once `main.c` and `Makefile` exist:
- Build: `make -C mpi-naive`

## Run (planned)
- `mpirun -np 4 ./mm_mpi_naive 512`
- `mpirun -np 8 ./mm_mpi_naive 1024 --seed 0 --verify`

## Notes
- Use `MPI_Scatterv`/`MPI_Gatherv` for uneven row counts.
- Add `MPI_Barrier` around compute for clearer timings.
- Prefer printing a checksum instead of matrices.

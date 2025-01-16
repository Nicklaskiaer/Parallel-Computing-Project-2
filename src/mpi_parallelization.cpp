#include <mpi.h>

#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

float** allocate_and_init_matrix(int n) {
    float** matrix = new float*[n];
    for (int i = 0; i < n; i++) {
        matrix[i] = new float[n];
        for (int j = 0; j < n; j++) {
            matrix[i][j] = i + j;
        }
    }
    return matrix;
}

void flatten_matrix(float** matrix, float* flat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            flat[i * n + j] = matrix[i][j];
        }
    }
}

void unflatten_matrix(const float* flat, float** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = flat[i * n + j];
        }
    }
}

// Parallel check symmetry. Each rank checks a portion of rows and does
// local checks. Then MPI_Reduce (logical AND) to get the final result.
bool checkSymMPI(float** matrix, int n, int rank, int size) {
    int rows_per_rank = n / size;
    int start_row = rank * rows_per_rank;
    int end_row = (rank == size - 1) ? n : (rank + 1) * rows_per_rank;

    bool local_sym = true;
    for (int i = start_row; i < end_row && local_sym; i++) {
        for (int j = 0; j < n && local_sym; j++) {
            if (matrix[i][j] != matrix[j][i]) {
                local_sym = false;
            }
        }
    }

    bool global_sym = false;
    MPI_Allreduce(&local_sym, &global_sym, 1, MPI_C_BOOL, MPI_LAND,
                  MPI_COMM_WORLD);

    return global_sym;
}

float** transposeMPI(float** matrix, int n, int rank, int size) {
    // Allocate a local transposed slice:
    // We will store the transposed rows for [start_row .. end_row).
    float** local_trans = new float*[n];
    for (int i = 0; i < n; i++) {
        local_trans[i] = new float[n];
    }

    // Determine this rank's rows
    int rows_per_rank = n / size;
    int start_row = rank * rows_per_rank;
    int end_row = (rank == size - 1) ? n : (rank + 1) * rows_per_rank;

    // Perform local transpose
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            local_trans[j][i] = matrix[i][j];
        }
    }

    // Now, rank 0 will gather everything into a final transposed matrix
    float** result = nullptr;
    if (rank == 0) {
        result = new float*[n];
        for (int i = 0; i < n; i++) {
            result[i] = new float[n];
        }
    }

    // We can gather via a flattened approach. Flatten local_trans, gather into
    // a big array on rank 0, then unflatten it. Each process is responsible
    // for the columns [start_row..end_row) in the transposed matrix.
    std::vector<float> flat_local(n * n, 0.0f);
    std::vector<float> flat_global(n * n, 0.0f);

    // Flatten local trans
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            flat_local[i * n + j] = local_trans[i][j];
        }
    }

    // All ranks gather their flattened slices into flat_global on rank 0
    MPI_Reduce(&flat_local[0], &flat_global[0], n * n, MPI_FLOAT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    // On rank 0, place flat_global into the result matrix
    if (rank == 0) {
        // “Summation” gather works here because only one rank writes each cell
        // at a time, so effectively it is just copying that data.
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = flat_global[i * n + j];
            }
        }
    }

    // Cleanup local_trans
    for (int i = 0; i < n; i++) {
        delete[] local_trans[i];
    }
    delete[] local_trans;

    return result;  // On non-root ranks, this will be nullptr
}

// -----------------------------------------------------------------------------
// This function will run the parallel version for a given n.
// -----------------------------------------------------------------------------
int run_parallel(int n, int rank, int size) {
    // We measure time only on rank 0
    if (rank == 0) {
        std::cout << "\nRunning parallel for n = 2^" << std::log2(n) << "\n";
    }

    // Rank 0 allocates and initializes the matrix
    float** matrix = nullptr;
    if (rank == 0) {
        matrix = allocate_and_init_matrix(n);
    }

    // Everyone allocates the same shape locally (to hold the broadcast matrix)
    // or you can do a partial distribution for large n. For demonstration:
    if (rank != 0) {
        matrix = new float*[n];
        for (int i = 0; i < n; i++) {
            matrix[i] = new float[n];
        }
    }

    // Flatten on rank 0, broadcast, unflatten
    std::vector<float> flat(n * n);
    if (rank == 0) {
        flatten_matrix(matrix, flat.data(), n);
    }

    // Broadcast the entire flattened matrix from rank 0
    MPI_Bcast(flat.data(), n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Everyone unflattens into their local 2D matrix
    unflatten_matrix(flat.data(), matrix, n);

    // 1) Check Symmetry (parallel)
    double start = 0.0, end = 0.0;
    if (rank == 0) start = MPI_Wtime();  // More accurate than clock() for MPI

    bool sym = checkSymMPI(matrix, n, rank, size);

    if (rank == 0) {
        end = MPI_Wtime();
        std::cout << "Time taken by checkSymMPI: " << (end - start)
                  << " seconds\n";
        // std::cout << "Matrix is symmetric? " << (sym ? "YES" : "NO")
        //           << std::endl;
    }

    // 2) Transpose (parallel)
    if (rank == 0) start = MPI_Wtime();

    float** new_matrix = transposeMPI(matrix, n, rank, size);

    if (rank == 0) {
        end = MPI_Wtime();
        std::cout << "Time taken by matTransposeMPI: " << (end - start)
                  << " seconds\n";

        // Optional: verify the transpose or do something with new_matrix...
        // Cleanup result
        for (int i = 0; i < n; i++) {
            delete[] new_matrix[i];
        }
        delete[] new_matrix;
    }

    // Cleanup
    for (int i = 0; i < n; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;

    return 0;
}

// -----------------------------------------------------------------------------
// Main entry point: Initialize MPI, run tests for powers of two, finalize.
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Example: run for n = 2^4 to 2^12
    for (int i = 4; i <= 12; i++) {
        int n = (int)std::pow(2, i);
        run_parallel(n, rank, size);
    }

    MPI_Finalize();
    return 0;
}

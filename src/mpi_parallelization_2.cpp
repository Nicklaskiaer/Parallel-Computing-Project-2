#include <mpi.h>

#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

// -----------------------------------------------------------------------------
// Helper: Allocate and initialize an n x n matrix (allocated on rank 0 only).
// -----------------------------------------------------------------------------
float** allocate_and_init_matrix(int n) {
    float** matrix = new float*[n];
    for (int i = 0; i < n; i++) {
        matrix[i] = new float[n];
        for (int j = 0; j < n; j++) {
            matrix[i][j] = i + j;  // Simple init
        }
    }
    return matrix;
}

// -----------------------------------------------------------------------------
// Flatten a 2D array into a 1D array.
// -----------------------------------------------------------------------------
void flatten_matrix(float** matrix, float* flat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            flat[i * n + j] = matrix[i][j];
        }
    }
}

// -----------------------------------------------------------------------------
// Un-flatten a 1D array into a 2D array.
// -----------------------------------------------------------------------------
void unflatten_matrix(const float* flat, float** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = flat[i * n + j];
        }
    }
}

// -----------------------------------------------------------------------------
// Parallel check symmetry: each rank checks a slice of rows, then we combine
// the booleans with MPI_Allreduce (logical AND).
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Parallel transpose: each rank computes only the portion [start_row..end_row)
// of the transposed matrix. Then we gather these partial results back to rank 0
// into one final matrix.
// -----------------------------------------------------------------------------
float** transposeMPI(float** matrix, int n, int rank, int size) {
    // Calculate which rows (in the final matrix) this rank handles.
    // In the transposed matrix, "row" corresponds to the original "col".
    // We'll define each rank is responsible for row range [start_row..end_row).
    int rows_per_rank = n / size;
    int start_row = rank * rows_per_rank;
    int end_row = (rank == size - 1) ? n : (rank + 1) * rows_per_rank;

    // -------------------------------------------------------------------------
    // 1) Build local_data (flattened sub-block) for the portion of the
    // transpose
    //    that this rank is responsible for.
    //
    //    final_transposed[i][j] = matrix[j][i]
    //
    //    - i in [start_row..end_row)
    //    - j in [0..n)
    //    => local data size = (end_row - start_row) * n
    // -------------------------------------------------------------------------
    int local_height = end_row - start_row;  // # rows in transposed sub-block
    int local_size = local_height * n;       // total # floats in sub-block
    std::vector<float> local_data(local_size);

    // Fill the local_data array
    // local_data stores transposed sub-block in row-major:
    //   sub-block row = i - start_row, sub-block col = j
    //   index = (i - start_row) * n + j
    //   value = matrix[j][i]
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            int idx = (i - start_row) * n + j;
            local_data[idx] = matrix[j][i];
        }
    }

    // -------------------------------------------------------------------------
    // 2) Gather all sub-blocks on rank 0 into a flattened array "result_flat"
    //    of size n*n. We use MPI_Gatherv because each rank has a different
    //    sub-block size (in general, if n % size != 0).
    // -------------------------------------------------------------------------
    std::vector<int> recvcounts(size), displs(size);
    int total_size = n * n;  // total elements in final transposed matrix

    if (rank == 0) {
        // Calculate recvcounts and displacements for each rank
        for (int r = 0; r < size; r++) {
            int r_start = r * rows_per_rank;
            int r_end = (r == size - 1) ? n : (r + 1) * rows_per_rank;
            int height = r_end - r_start;
            recvcounts[r] = height * n;  // each row has n columns
        }
        // displs[r] = offset in the flattened array where rank r's block goes
        // The block for rank r starts at (r_start)*n in the row-major index:
        // i.e. row = r_start.
        int cum = 0;
        for (int r = 0; r < size; r++) {
            displs[r] = cum;
            cum += recvcounts[r];
        }
    }

    // Buffer on rank 0 to receive the final transposed matrix in flattened form
    std::vector<float> result_flat;
    if (rank == 0) {
        result_flat.resize(total_size);
    }

    // Gatherv
    MPI_Gatherv(local_data.data(), local_size, MPI_FLOAT, result_flat.data(),
                recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // -------------------------------------------------------------------------
    // 3) On rank 0, unflatten "result_flat" into a final 2D array "result"
    // -------------------------------------------------------------------------
    float** result = nullptr;
    if (rank == 0) {
        result = new float*[n];
        for (int i = 0; i < n; i++) {
            result[i] = new float[n];
        }
        // Unflatten
        for (int idx = 0; idx < total_size; idx++) {
            int row = idx / n;
            int col = idx % n;
            result[row][col] = result_flat[idx];
        }
    }

    // Return the newly allocated transposed matrix pointer (nullptr on others)
    return result;
}

// -----------------------------------------------------------------------------
// This function will run the parallel version for a given n.
// -----------------------------------------------------------------------------
int run_parallel(int n, int rank, int size) {
    if (rank == 0) {
        std::cout << "\nRunning parallel for n = 2^" << std::log2(n) << " on "
                  << size << " ranks\n";
    }

    // -------------------------------------------------------------------------
    // Allocate matrix on rank 0, and broadcast to all
    // -------------------------------------------------------------------------
    float** matrix = nullptr;
    if (rank == 0) {
        matrix = allocate_and_init_matrix(n);
    } else {
        // Allocate matching shape on other ranks (they receive data via Bcast)
        matrix = new float*[n];
        for (int i = 0; i < n; i++) {
            matrix[i] = new float[n];
        }
    }

    // Flatten on rank 0, then broadcast
    std::vector<float> flat(n * n);
    if (rank == 0) {
        flatten_matrix(matrix, flat.data(), n);
    }
    MPI_Bcast(flat.data(), n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Unflatten into local matrix
    unflatten_matrix(flat.data(), matrix, n);

    // -------------------------------------------------------------------------
    // 1) Check Symmetry
    // -------------------------------------------------------------------------
    double start = 0.0, end = 0.0;
    if (rank == 0) start = MPI_Wtime();

    bool sym = checkSymMPI(matrix, n, rank, size);

    if (rank == 0) {
        end = MPI_Wtime();
        std::cout << "Time taken by checkSymMPI: " << (end - start)
                  << " seconds\n";
        // Optionally: std::cout << "Matrix is symmetric? "
        //             << (sym ? "YES\n" : "NO\n");
    }

    // -------------------------------------------------------------------------
    // 2) Transpose
    // -------------------------------------------------------------------------
    if (rank == 0) start = MPI_Wtime();

    float** new_matrix = transposeMPI(matrix, n, rank, size);

    if (rank == 0) {
        end = MPI_Wtime();
        std::cout << "Time taken by matTransposeMPI: " << (end - start)
                  << " seconds\n";

        // OPTIONAL: do something with new_matrix, verify correctness, etc.
        // Cleanup
        for (int i = 0; i < n; i++) {
            delete[] new_matrix[i];
        }
        delete[] new_matrix;
    }

    // Cleanup old matrix
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

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
    int rows_per_rank = n / size;
    int start_row = rank * rows_per_rank;
    int end_row = (rank == size - 1) ? n : (rank + 1) * rows_per_rank;

    int local_height = end_row - start_row;  // # rows in transposed sub-block
    int local_size = local_height * n;       // total elements in sub-block
    std::vector<float> local_data(local_size);

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < n; j++) {
            int idx = (i - start_row) * n + j;
            local_data[idx] = matrix[j][i];
        }
    }

    std::vector<int> recvcounts(size), displs(size);
    int total_size = n * n;

    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            int r_start = r * rows_per_rank;
            int r_end = (r == size - 1) ? n : (r + 1) * rows_per_rank;
            int height = r_end - r_start;
            recvcounts[r] = height * n;
        }

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

    MPI_Gatherv(local_data.data(), local_size, MPI_FLOAT, result_flat.data(),
                recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

    float** result = nullptr;
    if (rank == 0) {
        result = new float*[n];
        for (int i = 0; i < n; i++) {
            result[i] = new float[n];
        }
        for (int idx = 0; idx < total_size; idx++) {
            int row = idx / n;
            int col = idx % n;
            result[row][col] = result_flat[idx];
        }
    }
    return result;
}

int run_parallel(int n, int rank, int size) {
    if (rank == 0) {
        std::cout << "\nRunning parallel for n = 2^" << std::log2(n) << " on "
                  << size << " ranks\n";
    }

    float** matrix = nullptr;
    if (rank == 0) {
        matrix = allocate_and_init_matrix(n);
    } else {
        matrix = new float*[n];
        for (int i = 0; i < n; i++) {
            matrix[i] = new float[n];
        }
    }

    std::vector<float> flat(n * n);
    if (rank == 0) {
        flatten_matrix(matrix, flat.data(), n);
    }
    MPI_Bcast(flat.data(), n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

    unflatten_matrix(flat.data(), matrix, n);

    double start = 0.0, end = 0.0;
    if (rank == 0) start = MPI_Wtime();

    bool sym = checkSymMPI(matrix, n, rank, size);

    if (rank == 0) {
        end = MPI_Wtime();
        std::cout << "Time taken by checkSymMPI: " << (end - start)
                  << " seconds\n";
        // std::cout << "Matrix is symmetric? "
        //             << (sym ? "YES\n" : "NO\n");
    }

    if (rank == 0) start = MPI_Wtime();

    float** new_matrix = transposeMPI(matrix, n, rank, size);

    if (rank == 0) {
        end = MPI_Wtime();
        std::cout << "Time taken by matTransposeMPI: " << (end - start)
                  << " seconds\n";

        // Cleanup
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int i = 4; i <= 12; i++) {
        int n = (int)std::pow(2, i);
        run_parallel(n, rank, size);
    }

    MPI_Finalize();
    return 0;
}

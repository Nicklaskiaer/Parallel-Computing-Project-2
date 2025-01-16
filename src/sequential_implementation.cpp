#include <ctime>
#include <iostream>

float **allocate_and_init_matrix(int n) {
    float **matrix = new float *[n];
    for (int i = 0; i < n; i++) {
        matrix[i] = new float[n];
        for (int j = 0; j < n; j++) {
            matrix[i][j] = i + j;
        }
    }
    return matrix;
}

bool checkSym(float **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i][j] != matrix[j][i]) {
                return false;
            }
        }
    }
    return true;
}

float **transpose(float **matrix, int n) {
    float **new_matrix = new float *[n];
    for (int i = 0; i < n; i++) {
        new_matrix[i] = new float[n];
        for (int j = 0; j < n; j++) {
            new_matrix[i][j] = matrix[j][i];
        }
    }
    return new_matrix;
}

int run_sequential(int n) {
    std::cout << "\nRunning for n = 2^" << std::log2(n) << "\n";

    float **matrix = allocate_and_init_matrix(n);

    // Sequential implementation:

    clock_t start, end;
    double sym_duration;

    // Start timing
    start = clock();
    bool sym = checkSym(matrix, n);
    end = clock();

    sym_duration = (double)(end - start) / CLOCKS_PER_SEC;

    std::cout << "Time taken by checkSymSeq: " << sym_duration << " seconds"
              << std::endl;

    double trans_duration;
    // Start timing
    start = clock();
    float **new_matrix = transpose(matrix, n);
    end = clock();

    trans_duration = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout << "Time taken by matTransposeSeq: " << trans_duration
              << " seconds" << std::endl;
    return 0;
}

int main() {
    for (int i = 4; i <= 12; i++) {
        run_sequential((int)std::pow(2, i));
    }
    return 0;
}
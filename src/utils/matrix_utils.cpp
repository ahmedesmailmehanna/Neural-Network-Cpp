#include "utils.hpp"
#include <vector>
#include <vector>
#include "../math/matrix.hpp"

// Takes square matrix and flattens it to a 1D matrix
Matrix utils::flatten(const Matrix& m) {
    int n = m.cols;
    Matrix flattened(1, n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            flattened.data[0][i * n + j] = m.data[i][j];
        }
    }
    return flattened;  // Replace with flattened version
}

// Creates a target matrix for MNIST dataset, with 1 in the index of the label and 0 elsewhere
Matrix utils::createMNISTTargetMatrix(int label){
    Matrix res = Matrix(1, 10);
    res.data[0][label] = 1;
    return res;
}
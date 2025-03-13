#include "matrix.hpp"

// Default constructor: Initializes empty matrix
Matrix::Matrix() : rows(0), cols(0), data(nullptr) {}

// Constructor: Initializes matrix with given rows and columns
Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    data = new double*[rows];
    for (int i = 0; i < rows; i++) {
        data[i] = new double[cols];
    }
}

// Destructor: Frees allocated memory to prevent memory leaks
Matrix::~Matrix() {
    for (int i = 0; i < rows; i++) {
        delete[] data[i];
    }
    delete[] data;
}

// Random Initialization (For weights)
void Matrix::randomize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = dist(gen);
        }
    }
}

// Operator Overloading for Matrix Addition
Matrix Matrix::operator+(const Matrix &other) {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

// Operator Overloading for Matrix Subtraction
Matrix Matrix::operator-(const Matrix &other) {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

// Operator Overloading for Matrix Multiplication
Matrix Matrix::operator*(const Matrix &other) {
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            result.data[i][j] = 0;
            for (int k = 0; k < cols; k++) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }
    return result;
}

// Operator Overloading for Scalar Matrix Multiplication
Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }
    return result;
}

// Transposing the Matrix
Matrix Matrix::transpose() {
    Matrix transposed(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            transposed.data[j][i] = data[i][j];
        }
    }
    return transposed;
}

// Apply Function (Activation)
Matrix Matrix::applyFunction(std::function<double(double)> func) {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = func(data[i][j]);  // Apply function to each element
        }
    }
    return result;
}

// Print Matrix
void Matrix::print() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::fixed << std::setprecision(3) << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

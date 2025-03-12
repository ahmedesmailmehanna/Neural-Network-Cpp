// matrix.hpp - Matrix class for neural network

#ifndef MATRIX_HPP // Include guards, to prevent multiple inclusions/ redefinitions
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip> // For printing formatting
#include <random>
#include <functional> // For using lamnda function to pass member functions as pointer parameters

class Matrix {
public:
    int rows, cols;
    double** data;

    Matrix() : rows(0), cols(0), data(nullptr) {}

    // Constructor
    Matrix(int r, int c) : rows(r), cols(c) {
        data = new double*[rows];
        for (int i = 0; i < rows; i++) {
            data[i] = new double[cols];
        }
    }

    // Destructor (Frees memory)
    ~Matrix() {
        for (int i = 0; i < rows; i++) {
            delete[] data[i];
        }
        delete[] data;
    }

    // Random Initialization (for weights)
    void randomize() {
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
    Matrix operator+(const Matrix &other) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    // Operator Overloading for Matrix Subtraction
    Matrix operator-(const Matrix &other) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    // Operator Overloading for Matrix Multiplication
    Matrix operator*(const Matrix &other) {
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
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * scalar;
            }
        }
        return result;  // Return the new scaled matrix
    }    

    // Transposing the Matrix
    Matrix transpose() {
        Matrix transposed(cols, rows); // New matrix with swapped dimensions
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed.data[j][i] = data[i][j];
            }
        }
        return transposed;
    }

    // Apply Function (Activation)
    Matrix applyFunction(std::function<double(double)> func) { // So it can accept lambda functions
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = func(data[i][j]);  // Apply function to each element
            }
        }
        return result;
    }

    // Print Matrix
    void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << std::fixed << std::setprecision(3) << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

#endif // MATRIX_HPP

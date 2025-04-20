#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <vector>
#include <iomanip>  // For printing formatting
#include <random>
#include <functional>  // For using lambda function to pass member functions as pointer parameters

class Matrix {
public:
    int rows, cols;
    double** data;

    Matrix();
    Matrix(int r, int c);
    Matrix(const Matrix &other);
    ~Matrix();

    void randomize(double lowerLimit = -0.1, double upperLimit = 0.1);
    void fill(double value);
    bool isEqual(const Matrix& other) const;
    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix elementWiseMultiply(const Matrix &other) const;
    Matrix sumRows() const;
    Matrix operator*(const Matrix &other) const;
    Matrix operator*(double scalar) const;
    Matrix& operator=(const Matrix &other);
    Matrix transpose();
    Matrix applyFunction(std::function<std::vector<double>(std::vector<double>&)> func);
    void print();
};

#endif // MATRIX_HPP

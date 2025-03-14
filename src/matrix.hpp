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

    void randomize();
    Matrix operator+(const Matrix &other);
    Matrix operator-(const Matrix &other);
    Matrix operator*(const Matrix &other);
    Matrix operator*(double scalar) const;
    Matrix& operator=(const Matrix &other);
    Matrix transpose();
    Matrix applyFunction(std::function<double(double)> func);
    void print();
};

#endif // MATRIX_HPP

#ifndef SIGMOID_FUNCTION_HPP
#define SIGMOID_FUNCTION_HPP

#include <cmath>
#include "activation_function.hpp"

class SigmoidFunction : public ActivationFunction {
public:
    double activate(double x) override {
        return 1.0 / (1.0 + exp(-x));
    }

    double derivative(double x) override {
        return x * (1 - x);  // Sigmoid derivative
    }
};

#endif // SIGMOID_FUNCTION_HPP

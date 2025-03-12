#ifndef RELU_FUNCTION_HPP
#define RELU_FUNCTION_HPP

#include "activation.hpp"

class ReLUFunction : public ActivationFunction {
public:
    double activate(double x) override {
        return (x > 0) ? x : 0;
    }

    double derivative(double x) override {
        return (x > 0) ? 1 : 0;
    }
};

#endif // RELU_FUNCTION_HPP

#ifndef SIGMOID_FUNCTION_HPP
#define SIGMOID_FUNCTION_HPP

#include <cmath>
#include <vector>
#include "activation_function.hpp"

class SigmoidFunction : public ActivationFunction {
public:

    std::vector<double> activate(const std::vector<double> &x) {
        std::vector<double> y(x.size());
        for (int i = 0; i < x.size(); i++) {
            y[i] = 1.0 / (1.0 + exp(-(x[i])));
        }
        return y;
    }

    std::vector<double> derivative(const std::vector<double> &x) {
        std::vector<double> y(x.size());
        for (int i = 0; i < x.size(); i++) {
            y[i] = x[i] * (1 - x[i]);
        }
        return y;
    }

};

#endif // SIGMOID_FUNCTION_HPP

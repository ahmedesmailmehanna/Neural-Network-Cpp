#ifndef SIGMOID_FUNCTION_HPP
#define SIGMOID_FUNCTION_HPP

#include "activation_function.hpp"
#include <cmath>
#include <vector>

class SigmoidFunction : public ActivationFunction {
    public:
        std::vector<double> activate(const std::vector<double> &x) {
            std::vector<double> y(x.size());
            for (int i = 0; i < x.size(); i++) {
                y[i] = 1.0 / (1.0 + exp(-x[i]));
            }
            return y;
        }
    
        std::vector<double> derivative(const std::vector<double> &x) {
            std::vector<double> y(x.size());
            for (int i = 0; i < x.size(); i++) {
                double sigmoid_x = 1.0 / (1.0 + exp(-x[i]));
                y[i] = sigmoid_x * (1 - sigmoid_x);
            }
            return y;
        }
    };
    

#endif // SIGMOID_FUNCTION_HPP

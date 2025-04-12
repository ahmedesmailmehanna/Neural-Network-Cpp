#ifndef RELU_FUNCTION_HPP
#define RELU_FUNCTION_HPP

#include "activation_function.hpp"
#include <vector>

class ReLUFunction : public ActivationFunction {
    public:
        std::vector<double> activate(const std::vector<double> &x) {
            std::vector<double> y(x.size());
            for (int i = 0; i < x.size(); i++) {
                y[i] = (x[i] > 0) ? x[i] : 0.01 * x[i];  // small slope for x < 0
            }
            return y;
        }
    
        std::vector<double> derivative(const std::vector<double> &x) {
            std::vector<double> y(x.size());
            for (int i = 0; i < x.size(); i++) {
                y[i] = (x[i] > 0) ? 1 : 0.01;  // small gradient for x < 0
            }
            return y;
        }
    };
    

#endif // RELU_FUNCTION_HPP

#ifndef SOFTMAX_FUNCTION_HPP
#define SOFTMAX_FUNCTION_HPP

#include "activation_function.hpp"
#include <iostream>
#include <vector>
#include <cmath>

class SoftmaxFunction : public ActivationFunction {
public:



// Function to apply Softmax across a vector of inputs (used for layers with multiple neurons)
    std::vector<double> activate(const std::vector<double>& input) {
        double sum = 0.0;
        // Calculate the sum of the exponentials of all inputs
        for (auto val : input) {
            sum += exp(val);
        }
        
        // Apply Softmax formula to each input element
        std::vector<double> output;
        for (auto val : input) {
            output.push_back(exp(val) / sum);
        }
        
        return output;
    }

    // Compute the derivative of the Softmax function
    std::vector<double> derivative(const std::vector<double>& input) {
        // here we would typically compute the Jacobian matrix of the Softmax function
        // but for simplicity, we will return a vector of zeros
        // since the derivative of Softmax is not straightforward and depends on the output

        // raise error use softmax functions only on output layer
        std::cerr << "Softmax derivative is not implemented. Softmax is typically used as an output layer activation function." << std::endl;
        std::vector<double> derivative(input.size(), 0.0);
        return derivative;
    }
};

#endif // SOFTMAX_FUNCTION_HPP

#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <vector>

class ActivationFunction {
public:
    virtual std::vector<double> activate(const std::vector<double> &x) = 0;        // Pure virtual function
    virtual std::vector<double> derivative(const std::vector<double> &x) = 0;      // For backpropagation
    virtual ~ActivationFunction() {}              // Virtual destructor
};

#endif // ACTIVATION_FUNCTION_HPP

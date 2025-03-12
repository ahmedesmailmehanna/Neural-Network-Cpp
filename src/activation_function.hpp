#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

class ActivationFunction {
public:
    virtual double activate(double x) = 0;        // Pure virtual function
    virtual double derivative(double x) = 0;      // For backpropagation
    virtual ~ActivationFunction() {}              // Virtual destructor
};

#endif // ACTIVATION_FUNCTION_HPP

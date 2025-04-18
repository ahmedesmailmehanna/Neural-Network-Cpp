#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <vector>

class ActivationFunction {
public:
    virtual std::vector<double> activate(const std::vector<double> &x) = 0;
    virtual std::vector<double> derivative(const std::vector<double> &x) = 0;
    virtual ~ActivationFunction() {}
};

#endif // ACTIVATION_FUNCTION_HPP

// can't make them virtual and static at the same time.
// in other words, can't make derived classes static and we need to pass a new instance each time
// to the layer.
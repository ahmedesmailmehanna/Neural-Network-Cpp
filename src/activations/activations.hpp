#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "ReLU_function.hpp"
#include "sigmoid_function.hpp"
#include "softmax_function.hpp"

namespace activations { // Namespace for activation functions
    using ReLU = ReLUFunction;
    using Sigmoid = SigmoidFunction;
    using Softmax = SoftmaxFunction;
}

#endif // ACTIVATIONS_HPP
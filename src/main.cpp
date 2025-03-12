#include "dense_layer.hpp"
#include "sigmoid_function.hpp"
#include "relu_function.hpp"

int main() {
    ActivationFunction* sigmoid = new SigmoidFunction();
    ActivationFunction* relu = new ReLUFunction();

    DenseLayer layer1(10, 5, sigmoid);
    DenseLayer layer2(5, 3, relu);

    return 0;
}

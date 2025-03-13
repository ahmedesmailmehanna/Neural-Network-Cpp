#include "neural_network.hpp"
#include "dense_layer.hpp"
#include "sigmoid_function.hpp"

int main() {
    ActivationFunction* sigmoid = new SigmoidFunction();
    NeuralNetwork nn;

    nn.addLayer(new DenseLayer(784, 128, sigmoid));
    nn.addLayer(new DenseLayer(128, 10, sigmoid));

    Matrix input(1, 784);  // Dummy input
    Matrix target(1, 10);  // Dummy target

    nn.train(input, target, 10, 0.01);
    nn.saveToFile("model_weights.bin");

    return 0;
}

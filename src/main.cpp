#include "neural_network.hpp"
#include "dense_layer.hpp"
#include "sigmoid_function.hpp"

int main() {
    NeuralNetwork<DenseLayer> nn;  // Neural Network with only Dense Layers
    ActivationFunction* sigmoid = new SigmoidFunction();

    nn.addLayer(new DenseLayer(784, 128, sigmoid));
    nn.addLayer(new DenseLayer(128, 10, sigmoid));

    Matrix input(1, 784);
    Matrix target(1, 10);

    nn.train(input, target, 10, 0.01);
    nn.saveToFile("model_weights.bin");

    return 0;
}

#include "neural_network.hpp"
#include "dense_layer.hpp"
#include "sigmoid_function.hpp"

int main() {
    NeuralNetwork<DenseLayer> nn;
    ActivationFunction* sigmoid = new SigmoidFunction();

    nn.addLayer(new DenseLayer(2, 2, sigmoid));  // Small test network
    nn.addLayer(new DenseLayer(2, 1, sigmoid));

    Matrix input(1, 2);
    Matrix target(1, 1);

    input.data[0][0] = 0.5;  // Add simple test values
    input.data[0][1] = -0.2;
    target.data[0][0] = 1.0;

    std::cout << "Starting training...\n";
    nn.train(input, target, 100, 0.1);

    std::cout << "Testing forward pass...\n";
    Matrix result = nn.forward(input);
    result.print();

    nn.saveToFile("elbees");



    return 0;
}


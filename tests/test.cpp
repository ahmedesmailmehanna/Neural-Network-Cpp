#include "../src/math/matrix.hpp"
#include "../src/layers/dense_layer.hpp"
#include "../src/core/neural_network.hpp"
#include "../src/activations/activations.hpp"
#include "../src/utils/utils.hpp"
#include "./test_runner.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>

using namespace std;

// Test for Matrix Addition
bool testMatrixAddition() {
    Matrix m1(2, 2);
    m1.data[0][0] = 1; m1.data[0][1] = 2;
    m1.data[1][0] = 3; m1.data[1][1] = 4;

    Matrix m2(2, 2);
    m2.data[0][0] = 5; m2.data[0][1] = 6;
    m2.data[1][0] = 7; m2.data[1][1] = 8;

    Matrix sum = m1 + m2;
    Matrix expectedSum(2, 2);
    expectedSum.data[0][0] = 6; expectedSum.data[0][1] = 8;
    expectedSum.data[1][0] = 10; expectedSum.data[1][1] = 12;

    return sum.isEqual(expectedSum);
}

// Test for Matrix Subtraction
bool testMatrixSubtraction() {
    Matrix m1(2, 2);
    m1.data[0][0] = 5; m1.data[0][1] = 6;
    m1.data[1][0] = 7; m1.data[1][1] = 8;

    Matrix m2(2, 2);
    m2.data[0][0] = 1; m2.data[0][1] = 2;
    m2.data[1][0] = 3; m2.data[1][1] = 4;

    Matrix diff = m1 - m2;
    Matrix expectedDiff(2, 2);
    expectedDiff.data[0][0] = 4; expectedDiff.data[0][1] = 4;
    expectedDiff.data[1][0] = 4; expectedDiff.data[1][1] = 4;

    return diff.isEqual(expectedDiff);
}

// Test for Matrix Multiplication
bool testMatrixMultiplication() {
    Matrix m1(2, 3);
    m1.data[0][0] = 1; m1.data[0][1] = 2; m1.data[0][2] = 3;
    m1.data[1][0] = 4; m1.data[1][1] = 5; m1.data[1][2] = 6;

    Matrix m2(3, 2);
    m2.data[0][0] = 7; m2.data[0][1] = 8;
    m2.data[1][0] = 9; m2.data[1][1] = 10;
    m2.data[2][0] = 11; m2.data[2][1] = 12;

    Matrix product = m1 * m2;
    Matrix expectedProduct(2, 2);
    expectedProduct.data[0][0] = 58; expectedProduct.data[0][1] = 64;
    expectedProduct.data[1][0] = 139; expectedProduct.data[1][1] = 154;

    return product.isEqual(expectedProduct);
}

// Test for Matrix Transposition
bool testMatrixTranspose() {
    Matrix m(2, 3);
    m.data[0][0] = 1; m.data[0][1] = 2; m.data[0][2] = 3;
    m.data[1][0] = 4; m.data[1][1] = 5; m.data[1][2] = 6;

    Matrix transposed = m.transpose();
    Matrix expectedTranspose(3, 2);
    expectedTranspose.data[0][0] = 1; expectedTranspose.data[0][1] = 4;
    expectedTranspose.data[1][0] = 2; expectedTranspose.data[1][1] = 5;
    expectedTranspose.data[2][0] = 3; expectedTranspose.data[2][1] = 6;

    return transposed.isEqual(expectedTranspose);
}

// Test for Scalar Multiplication
bool testMatrixScalarMultiplication() {
    Matrix m(2, 2);
    m.data[0][0] = 1; m.data[0][1] = 2;
    m.data[1][0] = 3; m.data[1][1] = 4;

    Matrix scaled = m * 2.0;
    Matrix expectedScaled(2, 2);
    expectedScaled.data[0][0] = 2; expectedScaled.data[0][1] = 4;
    expectedScaled.data[1][0] = 6; expectedScaled.data[1][1] = 8;

    return scaled.isEqual(expectedScaled);
}

// Test for Random Initialization
bool testMatrixRandomInitialization() {
    Matrix m(2, 2);
    m.randomize();

    // Check if all elements are within the expected range (-0.1 to 0.1)
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            if (m.data[i][j] < -0.1 || m.data[i][j] > 0.1) {
                return false;
            }
        }
    }
    return true;
}

// Layer Tests
bool testDenseLayerForward() {
    ActivationFunction* sigmoid = new SigmoidFunction();
    DenseLayer layer(2, 2, sigmoid);
    
    // Set specific weights and biases for testing
    layer.weights.data[0][0] = 0.5; layer.weights.data[0][1] = 0.5;
    layer.weights.data[1][0] = 0.5; layer.weights.data[1][1] = 0.5;
    layer.biases.data[0][0] = 0.1; layer.biases.data[0][1] = 0.1;

    Matrix input(1, 2);
    input.data[0][0] = 1.0; input.data[0][1] = 1.0;

    layer.forward(input);
    
    // The output should be sigmoid(1.0 * 0.5 + 1.0 * 0.5 + 0.1) for both neurons
    double expected = 1.0 / (1.0 + exp(-1.1)); // sigmoid(1.1)
    
    return abs(layer.output.data[0][0] - expected) < 1e-6 && 
           abs(layer.output.data[0][1] - expected) < 1e-6;
}

// Neural Network Tests
bool testNeuralNetworkForward() {
    NeuralNetwork<DenseLayer> nn;
    ActivationFunction* sigmoid = new SigmoidFunction();
    
    nn.addLayer(new DenseLayer(2, 2, sigmoid));
    nn.addLayer(new DenseLayer(2, 1, sigmoid));
    
    Matrix input(1, 2);
    input.data[0][0] = 1.0; input.data[0][1] = 1.0;
    

    Matrix output = nn.forward(input);
        
    delete sigmoid;
    
    return output.rows == 1 && output.cols == 1;
}

// MNIST Data Tests
bool testMNISTDataLoading() {
    std::string images_file = "./data/train-images-idx3-ubyte";
    std::string labels_file = "./data/train-labels-idx1-ubyte";

    std::vector<Matrix> images = utils::loadMNISTImages(images_file);
    std::vector<int> labels = utils::loadMNISTLabels(labels_file);

    return !images.empty() && !labels.empty() && 
           images.size() == labels.size() &&
           images[0].rows == 28 && images[0].cols == 28;
}

// Model Save/Load Tests
bool testModelSaveLoad() {
    NeuralNetwork<DenseLayer> nn1;
    
    nn1.addLayer(new DenseLayer(2, 2, new activations::Sigmoid()));
    nn1.addLayer(new DenseLayer(2, 1, new activations::Sigmoid()));

    // Save the model
    nn1.saveToFile("./tests/test_model");

    // Create a new network and load the saved model
    NeuralNetwork<DenseLayer> nn2;
    nn2.addLayer(new DenseLayer(2, 2, new activations::Sigmoid()));
    nn2.addLayer(new DenseLayer(2, 1, new activations::Sigmoid()));
    nn2.loadFromFile("./tests/test_model");

    // nn1.layers[1]->weights.print();
    // nn1.layers[1]->biases.print();
    // nn2.layers[1]->weights.print();
    // nn2.layers[1]->biases.print();

    // Compare the weights and biases of the layers
    return nn1.layers[0]->isEqual(*nn2.layers[0]) && 
           nn1.layers[1]->isEqual(*nn2.layers[1]);
}

int main() {
    TestRunner runner;

    std::cout << "Running Matrix Tests..." << std::endl;
    runner.runTest("Matrix Addition", testMatrixAddition);
    runner.runTest("Matrix Subtraction", testMatrixSubtraction);
    runner.runTest("Matrix Multiplication", testMatrixMultiplication);
    runner.runTest("Matrix Transpose", testMatrixTranspose);
    runner.runTest("Matrix Scalar Multiplication", testMatrixScalarMultiplication);
    runner.runTest("Matrix Random Initialization", testMatrixRandomInitialization);


    // std::cout << "\nRunning Layer Tests..." << std::endl;
    // runner.runTest("Dense Layer Forward Pass", testDenseLayerForward);

    // std::cout << "\nRunning Neural Network Tests..." << std::endl;
    // runner.runTest("Neural Network Forward Pass", testNeuralNetworkForward);


    std::cout << "\nRunning Data Loading Tests..." << std::endl;
    runner.runTest("MNIST Data Loading", testMNISTDataLoading);

    std::cout << "\nRunning Model Save/Load Tests..." << std::endl;
    runner.runTest("Model Save and Load", testModelSaveLoad);

    runner.printSummary();

    return 0;
} 
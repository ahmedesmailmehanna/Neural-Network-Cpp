#include "../src/math/matrix.hpp"
#include "../src/layers/dense_layer.hpp"
#include "../src/layers/conv_layer.hpp"
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

// Test for Matrix Apply Function
bool testMatrixApplyFunction() {
    // Create a 2x2 matrix with specific values
    Matrix m(2, 2);
    m.data[0][0] = 1.0; m.data[0][1] = -2.0;
    m.data[1][0] = 3.0; m.data[1][1] = -4.0;

    // Define a lambda function to square each element
    auto squareFunction = [](std::vector<double>& row) {
        std::vector<double> result(row.size());
        for (size_t i = 0; i < row.size(); i++) {
            result[i] = row[i] * row[i];
        }
        return result;
    };

    // Apply the function to the matrix
    Matrix result = m.applyFunction(squareFunction);

    // Expected output matrix
    Matrix expected(2, 2);
    expected.data[0][0] = 1.0; expected.data[0][1] = 4.0;
    expected.data[1][0] = 9.0; expected.data[1][1] = 16.0;

    // Compare the result with the expected matrix
    return result.isEqual(expected);
}

// Layer Tests
bool testDenseLayerForward() {
    ActivationFunction* sig = new SigmoidFunction();
    DenseLayer layer(2, 2, sig);
    
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

bool testConvLayerForward() {
    ActivationFunction* relu = new ReLUFunction();
    ConvLayer layer(3, 1, 0, relu); // 3x3 kernel, stride=1, no padding

    // Initialize kernel with specific values for testing
    layer.kernel.data[0][0] = 1; layer.kernel.data[0][1] = 0; layer.kernel.data[0][2] = -1;
    layer.kernel.data[1][0] = 1; layer.kernel.data[1][1] = 0; layer.kernel.data[1][2] = -1;
    layer.kernel.data[2][0] = 1; layer.kernel.data[2][1] = 0; layer.kernel.data[2][2] = -1;

    // Input matrix
    Matrix input(5, 5);
    input.fill(1.0); // Fill with ones for simplicity

    // Perform forward pass
    layer.forward(input);

    // Expected output size: (5 - 3 + 2*0) / 1 + 1 = 3x3
    Matrix expectedOutput(3, 3);
    expectedOutput.fill(0.0); // Fill with expected values based on kernel and input

    return layer.output.isEqual(expectedOutput);
}

// Neural Network Tests
bool testNeuralNetworkForward() {
    NeuralNetwork nn;
    
    nn.addLayer(std::make_unique<DenseLayer>(2, 2, new activations::Sigmoid()));
    nn.addLayer(std::make_unique<DenseLayer>(2, 1, new activations::Softmax(), true)); // Output layer
    
    Matrix input(1, 2);
    input.data[0][0] = 1.0; input.data[0][1] = 1.0;

    Matrix output = nn.forward(input);
    
    // The output should be a 1x1 matrix since the last layer has 1 neuron
    // Check if the output is a 1x1 matrix and the value is between 0 and 1 (softmax output)
    if (output.data[0][0] < 0 || output.data[0][0] > 1) {
        return false;
    }
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
    // Create first network
    NeuralNetwork nn1;
    
    nn1.addLayer(std::make_unique<DenseLayer>(2, 2, new activations::Sigmoid()));
    nn1.addLayer(std::make_unique<DenseLayer>(2, 1, new activations::Sigmoid()));

    // Save the model
    nn1.saveToFile("./tests/test_model");

    // Create a new network and load the saved model
    NeuralNetwork nn2;
    nn2.addLayer(std::make_unique<DenseLayer>(2, 2, new activations::Sigmoid()));
    nn2.addLayer(std::make_unique<DenseLayer>(2, 1, new activations::Sigmoid()));
    nn2.loadFromFile("./tests/test_model");

    // Cast the Layer pointers to DenseLayer pointers
    auto* l1 = dynamic_cast<DenseLayer*>(nn1.layers[0].get());
    auto* l2 = dynamic_cast<DenseLayer*>(nn2.layers[0].get());
    auto* l3 = dynamic_cast<DenseLayer*>(nn1.layers[1].get());
    auto* l4 = dynamic_cast<DenseLayer*>(nn2.layers[1].get());

    // Check if the cast was successful
    // If the cast fails, l1, l2, l3, or l4 will be nullptr
    // Or we could have just used a static_cast, but we want to be sure
    // that the layers are indeed DenseLayers for testing purposes and avoiding undefined behavior
    if (!l1 || !l2 || !l3 || !l4) {
        std::cerr << "Dynamic cast failed: One or more layers are not DenseLayers" << std::endl;
        return false;
    }

    // Compare weights and biases
    bool weightsMatch = l1->weights.isEqual(l2->weights) && 
                        l3->weights.isEqual(l4->weights);
    bool biasesMatch = l1->biases.isEqual(l2->biases) && 
                       l3->biases.isEqual(l4->biases);

    return weightsMatch && biasesMatch;
}

// Simple Model Accuracy Test with XOR Problem
bool testModelAccuracy() {
    NeuralNetwork nn;
    nn.addLayer(std::make_unique<DenseLayer>(2, 4, new activations::Sigmoid()));
    nn.addLayer(std::make_unique<DenseLayer>(4, 2, new activations::Softmax(), true));

    // test data (XOR problem)
    std::vector<Matrix> inputs = {
        Matrix(1, 2), Matrix(1, 2),
        Matrix(1, 2), Matrix(1, 2)
    };
    std::vector<Matrix> targets = {
        Matrix(1, 2), Matrix(1, 2),
        Matrix(1, 2), Matrix(1, 2)
    };

    // 0 XOR 0 = 0, 0 XOR 1 = 1, 1 XOR 0 = 1, 1 XOR 1 = 0
    // Initialize XOR data                              // 0                       // 1
    inputs[0].data[0][0] = 0; inputs[0].data[0][1] = 0; targets[0].data[0][0] = 1; targets[0].data[0][1] = 0;
    inputs[1].data[0][0] = 0; inputs[1].data[0][1] = 1; targets[1].data[0][0] = 0; targets[1].data[0][1] = 1;
    inputs[2].data[0][0] = 1; inputs[2].data[0][1] = 0; targets[2].data[0][0] = 0; targets[2].data[0][1] = 1;
    inputs[3].data[0][0] = 1; inputs[3].data[0][1] = 1; targets[3].data[0][0] = 1; targets[3].data[0][1] = 0;


    nn.loadFromFile("./src/models/xor_model");



    // for (int i = 0; i < inputs.size(); i++) {
    //     nn.train(inputs[i], targets[i], 1000, 0.1);
    // }

    // for (int epoch = 0; epoch < 100; epoch++) {

    //     // Train on each sample
    //     for (int i = 0; i < inputs.size(); i++) {
    //         nn.train(inputs[i], targets[i], 20, 0.01);
    //     }
        
    // }

    // nn.saveToFile("./src/models/xor_model");
    
        
    int correct = 0;
    for (int i = 0; i < inputs.size(); i++) {
        Matrix output = nn.forward(inputs[i]);

        // For Softmax output with 2 neurons:
        // output[0] represents probability of class 0
        // output[1] represents probability of class 1

        cout<<"Predicted: " << output.data[0][0] << ", " << output.data[0][1] << endl;
        cout<<"Actual: " << targets[i].data[0][0] << ", " << targets[i].data[0][1] << endl;
        
        bool predicted = output.data[0][1] > output.data[0][0]; // predict 1 if index 1 is greater than index 0, else 0
        bool actual = targets[i].data[0][1] > targets[i].data[0][0]; // actual 1 if index 1 is greater than index 0, else 0
        if (predicted == actual) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / inputs.size();
    std::cout << "\nFinal accuracy: " << accuracy * 100 << "%\n";

    return accuracy >= 0.75; // Expect at least 75% accuracy
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
    runner.runTest("Matrix Apply Function", testMatrixApplyFunction);
    

    std::cout << "\nRunning Layer Tests..." << std::endl;
    runner.runTest("Dense Layer Forward Pass", testDenseLayerForward);
    runner.runTest("Conv Layer Forward Pass", testConvLayerForward);


    std::cout << "\nRunning Neural Network Tests..." << std::endl;
    runner.runTest("Neural Network Forward Pass", testNeuralNetworkForward);


    std::cout << "\nRunning Data Loading Tests..." << std::endl;
    runner.runTest("MNIST Data Loading", testMNISTDataLoading);


    std::cout << "\nRunning Model Save/Load Tests..." << std::endl;
    runner.runTest("Model Save and Load", testModelSaveLoad);

    std::cout << "\nRunning Model Accuracy Tests..." << std::endl;
    runner.runTest("Model Accuracy", testModelAccuracy);

    runner.printSummary();

    return 0;
}
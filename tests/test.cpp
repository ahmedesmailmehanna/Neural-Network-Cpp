#include "../src/math/matrix.hpp"
#include "../src/layers/dense_layer.hpp"
#include "../src/core/neural_network.hpp"
#include "../src/activations/sigmoid_function.hpp"
#include "../src/activations/softmax_function.hpp"
#include "../src/utils/utils.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>

using namespace std;

class TestRunner {
private:
    int totalTests = 0;
    int passedTests = 0;
    std::string currentTestName;

    void printTestResult(bool passed) {
        cout <<"hellooooo"<< endl;
        totalTests++;
        if (passed) {
            passedTests++;
            std::cout << currentTestName << " PASSED" << std::endl;
        } else {
            std::cout << currentTestName << " FAILED" << std::endl;
        }
    }

public:
    void runTest(const std::string name, std::function<bool()> test) {
        currentTestName = name;
        bool result = test();

        printTestResult(result);
    }

    void printSummary() {
        std::cout << "\nTest Summary:" << std::endl;
        std::cout << "Total Tests: " << totalTests << std::endl;
        std::cout << "Passed: " << passedTests << std::endl;
        std::cout << "Failed: " << (totalTests - passedTests) << std::endl;
        std::cout << "Success Rate: " << (static_cast<double>(passedTests) / totalTests * 100) << "%" << std::endl;
    }
};

// Matrix Tests
bool testMatrixOperations() {
    Matrix m1(2, 2);
    m1.data[0][0] = 1; m1.data[0][1] = 2;
    m1.data[1][0] = 3; m1.data[1][1] = 4;

    Matrix m2(2, 2);
    m2.data[0][0] = 1; m2.data[0][1] = 2;
    m2.data[1][0] = 3; m2.data[1][1] = 4;

    Matrix sum = m1 + m2;
    Matrix expectedSum(2, 2);
    expectedSum.data[0][0] = 2; expectedSum.data[0][1] = 4;
    expectedSum.data[1][0] = 6; expectedSum.data[1][1] = 8;

    return sum.isEqual(expectedSum);
}

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

    cout << "hi5" << endl;;
    std::string images_file = "../data/train-images-idx3-ubyte";
    std::string labels_file = "../data/train-labels-idx1-ubyte";

    std::vector<Matrix> images = utils::loadMNISTImages(images_file);
    std::vector<int> labels = utils::loadMNISTLabels(labels_file);

    return !images.empty() && !labels.empty() && 
           images.size() == labels.size() &&
           images[0].rows == 28 && images[0].cols == 28;
}

// Model Save/Load Tests
bool testModelSaveLoad() {
    NeuralNetwork<DenseLayer> nn1;
    ActivationFunction* sigmoid = new SigmoidFunction();
    
    nn1.addLayer(new DenseLayer(2, 2, sigmoid));
    nn1.addLayer(new DenseLayer(2, 1, sigmoid));

    // Save the model
    nn1.saveToFile("test_model");

    // Create a new network and load the saved model
    NeuralNetwork<DenseLayer> nn2;
    nn2.addLayer(new DenseLayer(2, 2, sigmoid));
    nn2.addLayer(new DenseLayer(2, 1, sigmoid));
    nn2.loadFromFile("test_model");

    // Compare the weights and biases of the first layer
    return nn1.layers[0]->isEqual(*nn2.layers[0]);
}

int main() {
    TestRunner runner;

    std::cout << "Running Matrix Tests..." << std::endl;
    runner.runTest("Matrix Addition", testMatrixOperations);
    runner.runTest("Matrix Multiplication", testMatrixMultiplication);

    std::cout << "\nRunning Layer Tests..." << std::endl;
    runner.runTest("Dense Layer Forward Pass", testDenseLayerForward);

    std::cout << "\nRunning Neural Network Tests..." << std::endl;
    runner.runTest("Neural Network Forward Pass", testNeuralNetworkForward);

    std::cout << "\nRunning Data Loading Tests..." << std::endl;
    runner.runTest("MNIST Data Loading", testMNISTDataLoading);

    std::cout << "\nRunning Model Save/Load Tests..." << std::endl;
    runner.runTest("Model Save and Load", testModelSaveLoad);

    runner.printSummary();

    return 0;
} 
#include "neural_network.hpp"
#include "dense_layer.hpp"
#include "ReLU_function.hpp"
#include "sigmoid_function.hpp"
#include "softmax_function.hpp"
#include "mnist_loader.hpp"
#include <vector>

void testSaveAndLoad(const std::string &filename);
void testLoadMNIST(const std::string &images_file, const std::string &labels_file);
void testVisualizeMNIST(const std::string &images_file, const std::string &labels_file, int image_index);

int main() {
    std::string images_file = "../data/train-images-idx3-ubyte";
    std::string labels_file = "../data/train-labels-idx1-ubyte";

    NeuralNetwork<DenseLayer> nn;
    ActivationFunction* ReLU = new ReLUFunction();
    ActivationFunction* softmax = new SoftmaxFunction();
    
    nn.addLayer(new DenseLayer(784, 10, ReLU));
    DenseLayer* outputLayer = new DenseLayer(10, 10, softmax);
    outputLayer->isOutputLayer = true;
    nn.addLayer(outputLayer);

    std::vector<Matrix> input = loadMNISTImages(images_file);
    // For flattening the input data (from 28x28 to 1x784) in place
    for (auto& img : input) {
        Matrix flattened(1, 784);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                flattened.data[0][i * 28 + j] = img.data[i][j];
            }
        }
        img = flattened;  // Replace with flattened version
    }

    
    std::vector<int> t = loadMNISTLabels(labels_file);
    std::vector<Matrix> target(t.size());
    // Converting from int to Matrix(1, 10), with 1 in the correct index
    for (int i = 0; i < t.size(); i++) {
        target[i] = Matrix(1, 10);
        target[i].data[0][t[i]] = 1;
    } 
    // input[2].print();

    nn.train(input[1], target[1], 10, 0.001);

    // auto inputIt = input.begin();
    // auto targetIt = target.begin();

    // while (inputIt != input.end() && targetIt != target.end()) {
    //     nn.train(*inputIt, *targetIt, 100, 0.1);
    //     ++inputIt;
    //     ++targetIt;
    // }

    // nn.saveToFile("model_v1");
    
    

    // testSaveAndLoad("testfile");
    // testLoadMNIST(images_file, labels_file); // Load a few images for testing 
    // testVisualizeMNIST(images_file, labels_file, 0); // Index of the image to visualize

    return 0;
}

// Function to test if data saves/loads correctly
void testSaveAndLoad(const std::string &filename) {
    std::cout << "Running Save & Load Test...\n";

    // Create and initialize a DenseLayer
    ActivationFunction* sigmoid = new SigmoidFunction();
    DenseLayer layer1(3, 2, sigmoid);
    
    std::cout << "Initial Weights & Biases (Before Saving):\n";
    layer1.weights.print();
    layer1.biases.print();

    // Save weights
    layer1.saveToFile(filename);

    // Create another DenseLayer with the same shape 
    DenseLayer layer2(3, 2, sigmoid);

    // Load saved weights into the new layer
    layer2.loadFromFile(filename);

    // ✅ Verify that loaded weights match saved weights
    if (layer1.isEqual(layer2)) {
        std::cout << "Test Passed: Weights & Biases match after loading!\n";
    } else {
        std::cout << "Test Failed: Weights & Biases do NOT match after loading!\n";
    }
}

// Function to test if MNIST data loads correctly
void testLoadMNIST(const std::string &images_file, const std::string &labels_file) {
    std::cout << "Running MNIST Data Load Test...\n";

    // Load images and labels
    std::vector<Matrix> images = loadMNISTImages(images_file);
    std::vector<int> labels = loadMNISTLabels(labels_file);

    // Check if images and labels were loaded successfully
    if (!images.empty() && !labels.empty()) {
        std::cout << "MNIST Data Loaded Successfully!\n";
        std::cout << "Loaded " << images.size() << " images and " << labels.size() << " labels.\n";
    } else {
        std::cout << "Error: MNIST data loading failed.\n";
    }
}

// Function to visually verify an MNIST image (prints a matrix)
void testVisualizeMNIST(const std::string &images_file, const std::string &labels_file, int image_index) {
    std::cout << "\nRunning MNIST Visualization Test...\n";

    // Load one image and its label
    std::vector<Matrix> images = loadMNISTImages(images_file);
    std::vector<int> labels = loadMNISTLabels(labels_file);

    // Check if data exists
    if (images.empty() || labels.empty()) {
        std::cout << "Error: Could not load MNIST data for visualization.\n";
        return;
    }

    // Print the first image and its label
    std::cout << "Label of the Image: " << labels[image_index] << "\n";
    std::cout << "Image Data (Normalized):\n";
    images[image_index].print();
}
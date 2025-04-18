#include "./src/core/neural_network.hpp"
#include "./src/layers/dense_layer.hpp"
#include "./src/activations/activations.hpp"
#include "./src/utils/utils.hpp"
#include <vector>
#include <chrono> // To Measure time
#include <string>
#include <stack>


void testSaveAndLoad(const std::string &filename);
void testLoadMNIST(const std::string &images_file, const std::string &labels_file);
void testVisualizeMNIST(const std::string &images_file, const std::string &labels_file, int image_index);

int main() {
    std::string images_file = "./data/train-images-idx3-ubyte";
    std::string labels_file = "./data/train-labels-idx1-ubyte";

    NeuralNetwork<DenseLayer> nn;
    ActivationFunction* sigmoid = new activations::Sigmoid();
    ActivationFunction* softmax = new activations::Softmax();
    
    nn.addLayer(new DenseLayer(784, 16, sigmoid));
    nn.addLayer(new DenseLayer(16, 16, sigmoid));
    nn.addLayer(new DenseLayer(16, 10, softmax, true)); // Output layer
    
    std::vector<Matrix> input = utils::loadMNISTImages(images_file);
    // For flattening the input data (from 28x28 to 1x784) in place
    for (auto& img : input) {
        img = utils::flatten(img);  // Replace with flattened version
    }
    
    std::vector<int> labels = utils::loadMNISTLabels(labels_file);
    std::vector<Matrix> target(labels.size());
    // Converting from int to Matrix(1, 10), with 1 in the correct index
    for (int i = 0; i < labels.size(); i++) {
        target[i] = utils::createMNISTTargetMatrix(labels[i]);
    } 

    
    // Data training

    nn.loadFromFile("./src/models/model_v2.1");

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 5000; i++) {
        std::cout << "Trianing number: " << i << std::endl;
        nn.train(input[i], target[i], 300, 0.01);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() / 60.0;
    std::cout << "Training completed in " << duration << " minutes.\n";

    nn.saveToFile("./src/models/model_v2.1");

    // ==================================================

    nn.loadFromFile("./src/models/model_v2.1");

    int n = 99;
    for (int i = 0; i < 3; i++) {
        nn.train(input[n + i], target[n + i], 10, 0.01);
    }

    Matrix out = nn.forward(input[n]);

    out.print();
    target[n].print();

    nn.saveToFile("./src/models/model_v2.1");

    // ===================================================
    
    return 0;
}

// Function to test if data saves/loads correctly
bool testSaveAndLoad() {
    DenseLayer layer1(3, 2, new activations::Sigmoid());
    layer1.saveToFile("filename");
    DenseLayer layer2(3, 2, new activations::Sigmoid());
    layer2.loadFromFile("filename");
    
    return layer1.isEqual(layer2);
}
    

// Function to test if MNIST data loads correctly
void testLoadMNIST(const std::string &images_file, const std::string &labels_file) {
    std::cout << "Running MNIST Data Load Test...\n";

    // load images and labels
    std::vector<Matrix> images = utils::loadMNISTImages(images_file);
    std::vector<int> labels = utils::loadMNISTLabels(labels_file);

    // check if images and labels were loaded successfully
    if (!images.empty() && !labels.empty()) {
        std::cout << "MNIST Data Loaded Successfully!\n";
        std::cout << "Loaded " << images.size() << " images and " << labels.size() << " labels.\n";
    } else {
        std::cout << "Error: MNIST data loading failed.\n";
    }
}

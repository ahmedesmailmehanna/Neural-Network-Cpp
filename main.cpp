#include "./src/core/neural_network.hpp"
#include "./src/layers/dense_layer.hpp"
#include "./src/activations/ReLU_function.hpp"
#include "./src/activations/sigmoid_function.hpp"
#include "./src/activations/softmax_function.hpp"
#include "./src/utils/mnist_loader.hpp"
#include <vector>
#include <chrono> // To Measure time
#include <string>
#include <stack>


void testSaveAndLoad(const std::string &filename);
void testLoadMNIST(const std::string &images_file, const std::string &labels_file);
void testVisualizeMNIST(const std::string &images_file, const std::string &labels_file, int image_index);
Matrix createTargetMatrix(double label);

int main() {
    std::stack<std::string> s;
    std::string images_file = "./src/data/train-images-idx3-ubyte";
    std::string labels_file = "./src/data/train-labels-idx1-ubyte";

    NeuralNetwork<DenseLayer> nn;
    ActivationFunction* sigmoid = new SigmoidFunction();
    ActivationFunction* softmax = new SoftmaxFunction();
    
    nn.addLayer(new DenseLayer(784, 16, sigmoid));
    nn.addLayer(new DenseLayer(16, 16, sigmoid));
    nn.addLayer(new DenseLayer(16, 10, softmax, true));
    
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

    
    // Data training
    // nn.loadFromFile("./src/models/model_v2.1");

    // auto start_time = std::chrono::high_resolution_clock::now();

    // for (int i = 0; i < 200; i++) {
    //     std::cout << "Trianing: " << i << "=================================================================================================================================================================================================================================================================================================================================" << std::endl;
    //     nn.train(input[i], target[i], 300, 0.01);
    // }
    
    // auto end_time = std::chrono::high_resolution_clock::now();
    // double duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() / 60.0;
    // std::cout << "Training completed in " << duration << " minutes.\n";

    // nn.saveToFile("./src/models/model_v2.1");

    // ==================================================

    nn.loadFromFile("./src/models/model_v2.1");
    // nn.loadFromFile("test");

    int n = 99;

    for (int i = 0; i < 3; i++) {
        nn.train(input[n + i], target[n + i], 10, 0.01);
    }

    Matrix out = nn.forward(input[n]);

    out.print();
    target[n].print();

    nn.saveToFile("./src/models/model_v2.1");
    // nn.saveToFile("test");

    // ===================================================
    

    // testSaveAndLoad("testfile");
    // testLoadMNIST(images_file, labels_file); // Load a few images for testing 
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

// Matrix createMNISTTargetMatrix(double label) {

// }
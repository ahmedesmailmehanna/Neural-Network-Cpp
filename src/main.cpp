#include "neural_network.hpp"
#include "dense_layer.hpp"
#include "sigmoid_function.hpp"
#include "mnist_loader.hpp"

void testSaveAndLoad(const std::string &filename);
void testLoadMNIST(const std::string &images_file, const std::string &labels_file, int num_images);
void testVisualizeMNIST(const std::string &images_file, const std::string &labels_file, int image_index);

int main() {
    // NeuralNetwork<DenseLayer> nn;
    // ActivationFunction* sigmoid = new SigmoidFunction();

    // nn.addLayer(new DenseLayer(2, 2, sigmoid));  // Small test network
    // nn.addLayer(new DenseLayer(2, 1, sigmoid));

    // Matrix input(1, 2);
    // Matrix target(1, 1);

    // input.data[0][0] = 0.5;  // Add simple test values
    // input.data[0][1] = -0.2;
    // target.data[0][0] = 1.0;

    // std::cout << "Starting training...\n";
    // nn.train(input, target, 100, 0.1);

    // std::cout << "Testing forward pass...\n";
    // Matrix result = nn.forward(input);
    // result.print();

    //testSaveAndLoad("testfile");

    std::string images_file = "train-images-idx3-ubyte";
    std::string labels_file = "train-labels-idx1-ubyte";

    testLoadMNIST(images_file, labels_file, 5); // Load a few images for testing 
    testVisualizeMNIST(images_file, labels_file, 0); // Index of the image to visualize

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
void testLoadMNIST(const std::string &images_file, const std::string &labels_file, int num_images) {
    std::cout << "Running MNIST Data Load Test...\n";

    // Load images and labels
    std::vector<Matrix> images = loadMNISTImages(images_file, num_images);
    std::vector<int> labels = loadMNISTLabels(labels_file, num_images);

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
    std::vector<Matrix> images = loadMNISTImages(images_file, image_index + 1);
    std::vector<int> labels = loadMNISTLabels(labels_file, image_index + 1);

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
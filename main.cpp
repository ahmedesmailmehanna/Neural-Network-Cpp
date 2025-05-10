#include "./src/core/neural_network.hpp"
#include "./src/layers/dense_layer.hpp"
#include "./src/activations/activations.hpp"
#include "./src/utils/utils.hpp"
#include <vector>
#include <chrono> // To Measure time

int main() {
    std::string images_file = "./data/train-images-idx3-ubyte";
    std::string labels_file = "./data/train-labels-idx1-ubyte";

    NeuralNetwork nn;
    nn.addLayer(std::make_unique<DenseLayer>(784, 16, new activations::Sigmoid()));
    nn.addLayer(std::make_unique<DenseLayer>(16, 16, new activations::Sigmoid()));
    nn.addLayer(std::make_unique<DenseLayer>(16, 10, new activations::Softmax(), true)); // Output layer
    
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

    // nn.loadFromFile("./src/models/model_v3.1");

    // auto start_time = std::chrono::high_resolution_clock::now();

    // for (int i = 0; i < 5000; i++) {
    //     std::cout << "Trianing number: " << i << std::endl;
    //     nn.train(input[i], target[i], 200, 0.01);
    // }
    
    // auto end_time = std::chrono::high_resolution_clock::now();

    // double duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() / 60.0;
    // std::cout << "Training completed in " << duration << " minutes.\n";
    
    // nn.saveToFile("./src/models/model_v3.1");
    
    // ==================================================
    // Batch data training
    
    // nn.loadFromFile("./src/models/model_v3.1");
    
    int n = 100; // Number of samples in the batch
    std::vector<Matrix> batch_input(n);
    std::vector<Matrix> batch_target(n);
    
    for (int i = 0; i < n; i++) {
        batch_input[i] = input[i];
        batch_target[i] = target[i];
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    nn.train_batch(batch_input, batch_target, 100, 0.01);
    auto end_time = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() / 60.0;
    std::cout << "Training completed in " << duration << " minutes.\n";
    
    nn.saveToFile("./src/models/model_v3.1");

    // ==================================================
    // first n inputs accuracy

    nn.loadFromFile("./src/models/model_v3.1");

    n = 100; // Number of samples to test
    int correct = 0;
    for (int i = 0; i < n; i++) {
        Matrix output = nn.forward(input[i]);

        int max = 0;
        for (int j = 0; j < output.cols; j++) {
            if (output.data[0][j] > output.data[0][max]) {
                max = j; // maximum probability
            }
        }
        std::cout << "Predicted: " << max << ", Actual: " << labels[i] << std::endl;
        if (max == labels[i]) {
            correct++;
        }
    }

    double accuracy = static_cast<double>(correct) / n;
    std::cout << "\nFinal accuracy: " << accuracy * 100 << "%\n";

    // nn.saveToFile("./src/models/model_v3.1");

    // ===================================================
    
    return 0;
}   
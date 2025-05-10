#include "neural_network.hpp"
#include <iostream>

void NeuralNetwork::addLayer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer)); // move ownership of the layer to the vector
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    Matrix curr = input;
    for (auto& layer : layers) {
        layer->forward(curr);
        curr = layer->output;

        std::cout << "Layer Output:";  
        curr.print();  
    }
    return curr;
}

void NeuralNetwork::train(Matrix &input, Matrix &target, int epochs, double learning_rate) {
    std::cout << "Training started for " << epochs << " epochs...\n";
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch: " << epoch + 1 << '\n';
        
        // Forward pass
        Matrix output = forward(input);

        // Calculate error (loss) between output and target
        Matrix error = output - target;
        
        // std::cout << "Error: ";  
        // error.print();  // Show the final error matrix
        
        // Backward pass (iterate from last to first layer)
        Matrix d_input = error;  // Start with error at output layer
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            d_input = (*it)->backward(d_input, learning_rate);  // Pass the new gradient
        }
    }

    std::cout << "Training completed!\n";
}

void NeuralNetwork::train_batch(std::vector<Matrix> &inputs, std::vector<Matrix> &targets, int epochs, double learning_rate) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }

    std::cout << "Training started for " << epochs << " epochs...\n";
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Batch epoch: " << epoch + 1 << '\n';
        // Iterate over each input-target pair
        for (int i = 0; i < inputs.size(); i++) {
        
            // Forward pass
            Matrix output = forward(inputs[i]);

            // Calculate error (loss) between output and target
            Matrix error = output - targets[i];

            // std::cout << "Error: ";  
            // error.print();  // Show the final error matrix
        
            // Backward pass (iterate from last to first layer)
            Matrix d_input = error;  // Start with error at output layer
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            d_input = (*it)->backward(d_input, learning_rate);  // Pass the new gradient
            }
        }
    }

    std::cout << "Training completed!\n";
}


void NeuralNetwork::saveToFile(const std::string &filename) {
    try {
        if (filename.empty()) {
            throw std::invalid_argument("Filename cannot be empty");
        }
        
        for (size_t i = 0; i < layers.size(); i++) {
            std::string layerFilename = filename + "_layer_" + std::to_string(i) + ".dat";
            layers[i]->saveToFile(layerFilename);
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to save model: " << e.what() << std::endl;
        throw;
    }
}

void NeuralNetwork::loadFromFile(const std::string &filename) {
    try {
        if (filename.empty()) {
            throw std::invalid_argument("Filename cannot be empty");
        }
        
        for (size_t i = 0; i < layers.size(); i++) {
            std::string layerFilename = filename + "_layer_" + std::to_string(i) + ".dat";
            layers[i]->loadFromFile(layerFilename);
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        throw;
    }
}

// Destructor is not needed as unique_ptr will automatically clean up the memory
// ~NeuralNetwork() { 
//     for (auto layer : layers) {
//         delete layer;
//     }
// }

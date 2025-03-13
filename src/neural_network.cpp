#include "neural_network.hpp"
#include <fstream>

// Add a layer of type LayerType
template <typename LayerType>
void NeuralNetwork<LayerType>::addLayer(LayerType* layer) {
    layers.push_back(layer);
}

// Perform forward propagation
template <typename LayerType>
Matrix NeuralNetwork<LayerType>::forward(Matrix input) {
    for (auto layer : layers) {
        layer->forward(input);
        input = layer->output;
    }
    return input;
}

// Train the network using backpropagation
template <typename LayerType>
void NeuralNetwork<LayerType>::train(Matrix &input, Matrix &target, int epochs, double learning_rate) {
    for (int i = 0; i < epochs; i++) {
        Matrix output = forward(input);
        Matrix error = target - output;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            (*it)->backward(error, learning_rate);
        }
    }
}

// Save all layers to a file
template <typename LayerType>
void NeuralNetwork<LayerType>::saveToFile(const std::string &filename) {
    for (auto layer : layers) {
        layer->saveToFile(filename);
    }
}

// Load all layers from a file
template <typename LayerType>
void NeuralNetwork<LayerType>::loadFromFile(const std::string &filename) {
    for (auto layer : layers) {
        layer->loadFromFile(filename);
    }
}

// Destructor to free memory
template <typename LayerType>
NeuralNetwork<LayerType>::~NeuralNetwork() {
    for (auto layer : layers) {
        delete layer;
    }
}

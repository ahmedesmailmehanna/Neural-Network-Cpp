#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include "trainable.hpp"
#include "serializable.hpp"

// Template class for neural network layers
template <typename LayerType>
class NeuralNetwork : public Trainable, public Serializable {
public:
    std::vector<LayerType*> layers;

    void addLayer(LayerType* layer) {
        layers.push_back(layer);
    }

    Matrix forward(Matrix input) {
        for (auto layer : layers) {
            layer->forward(input);
            input = layer->output;
        }
        return input;
    }

    void train(Matrix &input, Matrix &target, int epochs, double learning_rate) override {
        for (int i = 0; i < epochs; i++) {
            Matrix output = forward(input);
            Matrix error = target - output;
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                (*it)->backward(error, learning_rate);
            }
        }
    }

    void saveToFile(const std::string &filename) override {
        for (auto layer : layers) {
            layer->saveToFile(filename);
        }
    }

    void loadFromFile(const std::string &filename) override {
        for (auto layer : layers) {
            layer->loadFromFile(filename);
        }
    }

    ~NeuralNetwork() {
        for (auto layer : layers) {
            delete layer;
        }
    }
};

#endif  // NEURAL_NETWORK_HPP

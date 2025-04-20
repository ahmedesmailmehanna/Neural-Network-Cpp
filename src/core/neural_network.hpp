#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <memory>
#include "trainable.hpp"
#include "../layers/layer.hpp"
#include "../math/matrix.hpp"
#include "serializable.hpp"

// cancel the usage of templates
// we are going to use polymorphism and smart pointers instead
// to be able to make a cnn and a dnn in the same class

// template<typename LayerType>
class NeuralNetwork : public Trainable, public Serializable {
public:
    std::vector<std::unique_ptr<Layer>> layers;

    void addLayer(std::unique_ptr<Layer> layer) {
        layers.push_back(std::move(layer));
    }

    Matrix forward(const Matrix& input) {
        Matrix curr = input;
        for (auto& layer : layers) {
            layer->forward(curr);
            curr = layer->output;

            std::cout << "Layer Output:";  
            curr.print();  
        }
        return curr;
    }

    void train(Matrix &input, Matrix &target, int epochs, double learning_rate) override {
        std::cout << "Training started for " << epochs << " epochs...\n";
        
        for (int i = 0; i < epochs; i++) {
            std::cout << "Epoch: " << i + 1 << '\n';
            
            // Forward pass
            Matrix output = forward(input);
            Matrix error = output - target;  // simple Loss gradient
            
            std::cout << "Error: ";  
            error.print();  // Show the error matrix
            
            // Backward pass (iterate from last to first layer)
            Matrix d_input = error;  // Start with error at output layer
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                d_input = (*it)->backward(d_input, learning_rate);  // Pass the new gradient
            }
        }
    
        std::cout << "Training completed!\n";
    }
    

    void saveToFile(const std::string &filename) override {
        for (size_t i = 0; i < layers.size(); i++) {
            std::string layerFilename = filename + "_layer_" + std::to_string(i) + ".dat";
            layers[i]->saveToFile(layerFilename);
        }
    }

    void loadFromFile(const std::string &filename) override {
        for (size_t i = 0; i < layers.size(); i++) {
            std::string layerFilename = filename + "_layer_" + std::to_string(i) + ".dat";
            layers[i]->loadFromFile(layerFilename);
        }
    }

    // Destructor is not needed as unique_ptr will automatically clean up the memory
    // ~NeuralNetwork() { 
    //     for (auto layer : layers) {
    //         delete layer;
    //     }
    // }
};

#endif  // NEURAL_NETWORK_HPP

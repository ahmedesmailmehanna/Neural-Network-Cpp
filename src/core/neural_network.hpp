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
private:
    std::vector<std::unique_ptr<Layer>> layers;
public:

    // add a layer to the network
    void addLayer(std::unique_ptr<Layer> layer);

    // Forward pass through the network
    Matrix forward(const Matrix& input);

    // Train a single input data for number of epochs
    void train(Matrix &input, Matrix &target, int epochs, double learning_rate) override;
    // Train a batch of input data for number of epochs 
    void train_batch(std::vector<Matrix> &inputs, std::vector<Matrix> &targets, int epochs, double learning_rate) override;

    void saveToFile(const std::string &filename) override;
    void loadFromFile(const std::string &filename) override;
};

#endif  // NEURAL_NETWORK_HPP

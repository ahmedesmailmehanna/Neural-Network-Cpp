#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include "trainable.hpp"
#include "serializable.hpp"
#include "neural_network.cpp"


// Template class to support different layer types
template <typename LayerType>
class NeuralNetwork : public Trainable, public Serializable {
public:
    std::vector<LayerType*> layers;  // Store only one layer type

    void addLayer(LayerType* layer);
    Matrix forward(Matrix input);
    void train(Matrix &input, Matrix &target, int epochs, double learning_rate) override;
    void saveToFile(const std::string &filename) override;
    void loadFromFile(const std::string &filename) override;
    ~NeuralNetwork();
};

#endif  // NEURAL_NETWORK_HPP

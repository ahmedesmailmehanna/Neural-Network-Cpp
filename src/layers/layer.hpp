#ifndef LAYER_HPP
#define LAYER_HPP

#include "../math/matrix.hpp"
#include "../core/serializable.hpp"
#include "../activations/activation_function.hpp"
#include <iostream>
#include <memory>

// Abstract class for all layers
class Layer : public Serializable {
public:
    Matrix input, output;
    std::unique_ptr<ActivationFunction> activation;
    bool isOutputLayer; // For softmax

    // Constructors 
    Layer() {};
    Layer(ActivationFunction* activationFunc) : activation(activationFunc) {}
    Layer(ActivationFunction* activationFunc, bool isOutputLayer) : activation(activationFunc), isOutputLayer(isOutputLayer) {}

    virtual void forward(Matrix &input) = 0;
    virtual Matrix backward(Matrix &d_output, double learning_rate) = 0;

    virtual void saveToFile(const std::string &filename) = 0;
    virtual void loadFromFile(const std::string &filename) = 0;

    virtual ~Layer() = default;
};

#endif  // LAYER_HPP

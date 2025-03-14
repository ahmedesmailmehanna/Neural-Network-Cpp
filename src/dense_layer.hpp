#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "base_layer.hpp"
#include "serializable.hpp"
#include "matrix.hpp"
#include "activation_function.hpp"  // Include activation base class

class DenseLayer : public BaseLayer, public Serializable {
public:
    Matrix weights, biases;
    Matrix input, output;
    ActivationFunction* activation;  // Pointer to activation function

    DenseLayer(int input_size, int output_size, ActivationFunction* activationFunc);
    void forward(Matrix &input) override;
    void backward(Matrix &error, double learning_rate) override;

    void saveToFile(const std::string &filename) override;
    void loadFromFile(const std::string &filename) override;

    bool isEqual(DenseLayer &other); // For testing

    ~DenseLayer();
};

#endif // DENSE_LAYER_HPP

#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "layer.hpp"
#include "../core/serializable.hpp"
#include "../math/matrix.hpp"
#include "../activations/activation_function.hpp"

class DenseLayer : public Layer {
public:
    Matrix weights, biases; // weight => which neuron/pixels take action // biases => how high before getting active
    Matrix input, output;
    ActivationFunction* activation;
    bool isOutputLayer; // For softmax

    DenseLayer(int input_size, int output_size, ActivationFunction* activationFunc);
    DenseLayer(int input_size, int output_size, ActivationFunction* activationFunc, bool isOutputLayer);
    
    void forward(Matrix &input) override;
    Matrix backward(Matrix &d_output, double learning_rate) override;

    void saveToFile(const std::string &filename) override;
    void loadFromFile(const std::string &filename) override;

    bool isEqual(DenseLayer &other); // For testing

    ~DenseLayer();
};

#endif // DENSE_LAYER_HPP

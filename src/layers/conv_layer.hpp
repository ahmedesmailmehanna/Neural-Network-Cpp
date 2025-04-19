#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include "layer.hpp"
#include "../math/matrix.hpp"
#include "../activations/activations.hpp"
#include "../activations/activation_function.hpp"
#include "../core/trainable.hpp"
#include "../core/serializable.hpp"

class ConvLayer : public Layer {
public:
    int kernel_size, stride, padding;
    Matrix input;  // Input matrix (image)
    Matrix output; // Output matrix (feature map)
    Matrix kernel;  // Convolution kernel (filter)
    ActivationFunction* activation;

    ConvLayer(int kernel_size, int stride, int padding, ActivationFunction* activationFunc);
    void forward(Matrix &input) override;
    Matrix backward(Matrix &error, double learning_rate) override;
    void saveToFile(const std::string &filename) override;
    void loadFromFile(const std::string &filename) override;
    ~ConvLayer();
};

#endif  // CONV_LAYER_HPP

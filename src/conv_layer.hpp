#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include "base_layer.hpp"
#include "matrix.hpp"
#include "activation_function.hpp"
#include "serializable.hpp"

class ConvLayer : public BaseLayer, public Serializable {
public:
    int kernel_size, stride, padding;
    Matrix kernel;  // Convolution kernel (filter)
    ActivationFunction* activation;

    ConvLayer(int kernel_size, int stride, int padding, ActivationFunction* activationFunc);
    void forward(Matrix &input) override;
    void backward(Matrix &error, double learning_rate) override;
    void saveToFile(const std::string &filename) override;
    void loadFromFile(const std::string &filename) override;
    ~ConvLayer();
};

#endif  // CONV_LAYER_HPP

#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include "layer.hpp"
#include "../math/matrix.hpp"
#include "../activations/activation_function.hpp"
#include <memory>

class ConvLayer : public Layer {
public:
    int kernel_size, stride, padding;
    Matrix kernel;

    ConvLayer(int kernel_size, int stride, int padding, ActivationFunction* activationFunc);
    ConvLayer(int kernel_size, int stride, int padding, ActivationFunction* activationFunc, bool isOutputLayer);

    void forward(Matrix &input) override;
    Matrix backward(Matrix &d_output, double learning_rate) override;
    void saveToFile(const std::string &filename) override;
    void loadFromFile(const std::string &filename) override;
    ~ConvLayer();
};

#endif  // CONV_LAYER_HPP

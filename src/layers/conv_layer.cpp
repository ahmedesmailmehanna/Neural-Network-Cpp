#include "conv_layer.hpp"
#include <fstream>
#include <typeinfo>
#include "../activations/softmax_function.hpp"

ConvLayer::ConvLayer(int kernel_size, int stride, int padding, ActivationFunction* activationFunc)
    : kernel_size(kernel_size), stride(stride), padding(padding), Layer(activationFunc),
      kernel(kernel_size, kernel_size) {
    kernel.randomize();
    isOutputLayer = false;
}

ConvLayer::ConvLayer(int kernel_size, int stride, int padding, ActivationFunction* activationFunc, bool isOutputLayer)
    : kernel_size(kernel_size), stride(stride), padding(padding), Layer(activationFunc, isOutputLayer),
      kernel(kernel_size, kernel_size) {
    kernel.randomize();
}

void ConvLayer::forward(Matrix &input) {
    this->input = input;
    int output_size = (input.rows - kernel_size + 2 * padding) / stride + 1;
    output = Matrix(output_size, output_size);

    // Check if the activation function is Softmax and enforce output layer usage
    if (typeid(*activation) == typeid(SoftmaxFunction) && !isOutputLayer) {
        throw std::logic_error("SoftmaxFunction can only be used in the output layer: new ConvLayer(..., true)");
    }

    // Compute convolution
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            double sum = 0.0;
            for (int ki = 0; ki < kernel_size; ki++) {
                for (int kj = 0; kj < kernel_size; kj++) {
                    int x = i * stride + ki - padding;
                    int y = j * stride + kj - padding;
                    
                    if (x >= 0 && x < input.rows && y >= 0 && y < input.cols) {
                        sum += input.data[x][y] * kernel.data[ki][kj];
                    }
                }
            }
            output.data[i][j] = sum;
        }
    }

    // Apply activation function
    std::vector<double> row(output.cols);
    output = output.applyFunction([this](std::vector<double> x) { 
        return activation->activate(x); 
    });
}

Matrix ConvLayer::backward(Matrix &d_output, double learning_rate) {
    Matrix d_input(input.rows, input.cols);
    Matrix d_kernel(kernel_size, kernel_size);

    if (isOutputLayer) {
        // For output layer, d_output is already the error
        d_input = d_output;
    } else {
        // For hidden layers, compute the derivative of the activation function
        Matrix d_activation = output.applyFunction([this](std::vector<double> x) {
            return activation->derivative(x);
        });
        d_input = d_output.elementWiseMultiply(d_activation);
    }

    // Update kernel weights using gradient descent
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            kernel.data[i][j] -= learning_rate * d_kernel.data[i][j];
        }
    }

    return d_input;
}

void ConvLayer::saveToFile(const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // Save kernel dimensions and data
    file.write((char*)&kernel_size, sizeof(kernel_size));
    for (int i = 0; i < kernel_size; i++) {
        file.write((char*)kernel.data[i], kernel_size * sizeof(double));
    }

    file.close();
}

void ConvLayer::loadFromFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    // Load kernel dimensions and data
    int loaded_kernel_size;
    file.read((char*)&loaded_kernel_size, sizeof(loaded_kernel_size));
    if (loaded_kernel_size != kernel_size) {
        throw std::runtime_error("Kernel size mismatch in file: " + filename);
    }

    for (int i = 0; i < kernel_size; i++) {
        file.read((char*)kernel.data[i], kernel_size * sizeof(double));
    }

    file.close();
}

ConvLayer::~ConvLayer() {
    // delete activation; // not needed as it's managed by the Layer class
}

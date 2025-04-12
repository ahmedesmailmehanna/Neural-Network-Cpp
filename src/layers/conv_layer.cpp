#include "conv_layer.hpp"
#include <fstream>

// Constructor: Initializes kernel and parameters
ConvLayer::ConvLayer(int kernel_size, int stride, int padding, ActivationFunction* activationFunc)
    : kernel_size(kernel_size), stride(stride), padding(padding), activation(activationFunc),
      kernel(kernel_size, kernel_size) {  // Kernel is square
    kernel.randomize();
}

// Forward Propagation: Applies convolution
void ConvLayer::forward(Matrix &input) {
    int output_size = (input.rows - kernel_size + 2 * padding) / stride + 1;
    output = Matrix(output_size, output_size);

    // First compute all the convolution sums
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

    // Then apply activation function to the entire output matrix
    output = output.applyFunction([this](const std::vector<double>& x) {
        return activation->activate(x);
    });
}

// Backward Propagation: Computes gradients (Placeholder for now)
Matrix ConvLayer::backward(Matrix &error, double learning_rate) {
    // TODO: Implement gradient calculations for ConvLayers
}

// Save & Load Functions
void ConvLayer::saveToFile(const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return;
    for (int i = 0; i < kernel_size; i++) {
        file.write((char*)kernel.data[i], kernel_size * sizeof(double));
    }
    file.close();
}

void ConvLayer::loadFromFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return;
    for (int i = 0; i < kernel_size; i++) {
        file.read((char*)kernel.data[i], kernel_size * sizeof(double));
    }
    file.close();
}

// Destructor
ConvLayer::~ConvLayer() {
    delete activation;
}

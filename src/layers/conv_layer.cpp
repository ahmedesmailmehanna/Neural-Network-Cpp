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

// Backward Propagation: Computes gradients
Matrix ConvLayer::backward(Matrix &d_output, double learning_rate) {
    int input_size = input.rows; // Assuming square input
    int output_size = output.rows; // Assuming square output

    // Initialize gradients for the kernel
    Matrix d_kernel(kernel_size, kernel_size);
    d_kernel.fill(0.0);

    // Initialize gradients for the input (to propagate back)
    Matrix d_input(input_size, input_size);
    d_input.fill(0.0);

    // Compute gradients for the kernel
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            for (int ki = 0; ki < kernel_size; ki++) {
                for (int kj = 0; kj < kernel_size; kj++) {
                    int x = i * stride + ki - padding;
                    int y = j * stride + kj - padding;

                    if (x >= 0 && x < input_size && y >= 0 && y < input_size) {
                        d_kernel.data[ki][kj] += d_output.data[i][j] * input.data[x][y];
                        d_input.data[x][y] += d_output.data[i][j] * kernel.data[ki][kj];
                    }
                }
            }
        }
    }

    // Update the kernel using the gradients
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            kernel.data[i][j] -= learning_rate * d_kernel.data[i][j];
        }
    }

    return d_input; // Propagate the error back to the previous layer
}

// Save & Load Functions
void ConvLayer::saveToFile(const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not create file " << filename << std::endl;
        return;
    }

    std::cout << "Saving kernel to " << filename << std::endl;

    // Save kernel data
    for (int i = 0; i < kernel_size; i++) {
        file.write((char*)kernel.data[i], kernel_size * sizeof(double));
    }

    file.close();
    std::cout << "File saved successfully!\n";
}

void ConvLayer::loadFromFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for loading!" << std::endl;
        return;
    }

    std::cout << "Loading kernel from " << filename << std::endl;

    // Load kernel data
    for (int i = 0; i < kernel_size; i++) {
        file.read((char*)kernel.data[i], kernel_size * sizeof(double));
    }

    file.close();
    std::cout << "File loaded successfully!\n";
}

// Destructor
ConvLayer::~ConvLayer() {
    delete activation;
}

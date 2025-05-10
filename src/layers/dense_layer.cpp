#include "dense_layer.hpp"
#include "../activations/softmax_function.hpp" // for last layer logic
#include <fstream>

// Weights Matrix has input_size rows and output_size cols
// Each neuron has 1 bias so the rows are 1
DenseLayer::DenseLayer(int input_size, int output_size, ActivationFunction* activationFunc) 
    : weights(input_size, output_size), biases(1, output_size), Layer(activationFunc) {
    // xavier/glorot initialization
    double limit = sqrt(6.0 / (input_size + output_size));
    weights.randomize(-limit, limit);
    biases.randomize(-0.1, 0.1);
    isOutputLayer = false;
}
DenseLayer::DenseLayer(int input_size, int output_size, ActivationFunction* activationFunc, bool isOutputLayer) 
    : weights(input_size, output_size), biases(1, output_size), Layer(activationFunc, isOutputLayer) {
    // xavier/glorot initialization
    double limit = sqrt(6.0 / (input_size + output_size));
    weights.randomize(-limit, limit);
    biases.randomize(-0.1, 0.1);
}

// Forward pass: Computes output = (input * weights) + biases
void DenseLayer::forward(Matrix &input) {
// Store the input for backpropagation
    this->input = input;
    
    // Compute the forward pass
    output = input * weights;  // Matrix multiplication
    output = output + biases;  // Add biases

    // Check if the activation function is Softmax and enforce output layer usage
    if (typeid(*activation) == typeid(SoftmaxFunction) && !isOutputLayer) {
        throw std::logic_error("SoftmaxFunction can only be used in the output layer: new DenseLayer(..., true)");
    }
    
    // Apply activation function
    output = output.applyFunction([this](std::vector<double> x) { 
return activation->activate(x); 
}); 
}

// Backpropagation: Compute weight and bias updates
// d_output is the gradient of the loss with respect to the output of this layer
// learning_rate is the step size for updating weights and biases
// The output of this layer is the input to the next layer, so we need to propagate the error back
Matrix DenseLayer::backward(Matrix &d_output, double learning_rate) {
    Matrix delta;
    
    if (isOutputLayer) {
        // For Softmax output layer, we assume d_output = predictions - target
        delta = d_output;  // No need to multiply by activation derivative
    } else {
        // Compute derivative of activation
        Matrix d_activation = output.applyFunction([this](std::vector<double> x) { return activation->derivative(x); });

        // Compute delta for backpropagation
        // delta = d_output * activation_derivative(output)
        delta = d_output.elementWiseMultiply(d_activation);
    }

    // Compute gradients
    Matrix d_weights = input.transpose() * delta; // Transpose input for correct multiplication
    // d_weights has shape (input_size, output_size)
    // d_weights = input^T * delta

    Matrix d_biases = delta.sumRows();  // Sum across rows to get bias gradients
    // d_biases has shape (1, output_size)
    
    // Update parameters
    weights = weights - (d_weights * learning_rate);
    biases = biases - (d_biases * learning_rate);

    // Propagate error to the previous layer
    Matrix d_input = delta * weights.transpose(); // d_input has shape (batch_size, input_size)
    
    return d_input;
}


// Save weights and biases to file
void DenseLayer::saveToFile(const std::string &filename) {
    try {
        if (filename.empty()) {
            throw std::invalid_argument("Filename cannot be empty");
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Could not create file " << filename << std::endl;
            return;
        }

        std::cout << "Saving weights and biases to " << filename << std::endl;

        for (int i = 0; i < weights.rows; i++) {
            file.write((char*)weights.data[i], weights.cols * sizeof(double));
        }
        for (int i = 0; i < biases.rows; i++) {
            file.write((char*)biases.data[i], biases.cols * sizeof(double));
        }

        file.close();
        std::cout << "File saved successfully!\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving layer: " << e.what() << std::endl;
        throw;
    }
}

// Load weights and biases from file
void DenseLayer::loadFromFile(const std::string &filename) {
    try {
        if (filename.empty()) {
            throw std::invalid_argument("Filename cannot be empty");
        }

        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Could not open file " << filename << " for loading!" << std::endl;
            return;
        }

        std::cout << "Loading weights and biases from " << filename << std::endl;

        for (int i = 0; i < weights.rows; i++) {
            file.read((char*)weights.data[i], weights.cols * sizeof(double));
        }
        for (int i = 0; i < biases.rows; i++) {
            file.read((char*)biases.data[i], biases.cols * sizeof(double));
        }

        file.close();
        std::cout << "File loaded successfully!\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading layer: " << e.what() << std::endl;
        throw;
    }
}

// Compares two layers for testing
bool DenseLayer::isEqual(DenseLayer &other) {
    for (int i = 0; i < weights.rows; i++) {
        for (int j = 0; j < weights.cols; j++) {
            if (weights.data[i][j] != other.weights.data[i][j]) {
                return false;  // Mismatch found
            }
        }
    }

    for (int i = 0; i < biases.rows; i++) {
        for (int j = 0; j < biases.cols; j++) {
            if (biases.data[i][j] != other.biases.data[i][j]) {
                return false;
            }
        }
    }
    
    return true;  // All values match
}

// Destructor: Prevent memory leak by deleting activation function
DenseLayer::~DenseLayer() {
    // delete activation; // not needed as it's managed by Layer class
}

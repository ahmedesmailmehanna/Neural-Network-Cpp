#include "dense_layer.hpp"
#include <fstream>

// Weights Matrix has input_size rows and output_size cols
// Each neuron has 1 bias so the rows are 1
DenseLayer::DenseLayer(int input_size, int output_size, ActivationFunction* activationFunc) 
    : weights(input_size, output_size), biases(1, output_size), activation(activationFunc) {
    weights.randomize();
    biases.randomize();
}

// Forward pass: Computes output = (input * weights) + biases
void DenseLayer::forward(Matrix &input) {
    this->input = input;
    output = (input * weights) + biases;
    
    // Lambda function to use member function
    output = output.applyFunction([this](double x) { return activation->activate(x); }); 
}

// Backpropagation: Compute weight and bias updates
void DenseLayer::backward(Matrix &error, double learning_rate) {
    // Apply activation derivative element-wise (Lambda function)
    Matrix d_output = error.applyFunction([this](double x) { return activation->derivative(x); });

    // Compute gradients: d_weights = input.T * d_output
    Matrix d_weights = input.transpose() * d_output;

    // Update weights: weights = weights - (d_weights * learning_rate)
    weights = weights - (d_weights * learning_rate);

    // Update biases: biases = biases - (d_output * learning_rate)
    biases = biases - (d_output * learning_rate);
}

// Save weights and biases to file
void DenseLayer::saveToFile(const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return;  

    for (int i = 0; i < weights.rows; i++) {
        file.write((char*)weights.data[i], weights.cols * sizeof(double));
    }
    for (int i = 0; i < biases.rows; i++) {
        file.write((char*)biases.data[i], biases.cols * sizeof(double));
    }

    file.close();
}

// Load weights and biases from file
void DenseLayer::loadFromFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return;  

    for (int i = 0; i < weights.rows; i++) {
        file.read((char*)weights.data[i], weights.cols * sizeof(double));
    }
    for (int i = 0; i < biases.rows; i++) {
        file.read((char*)biases.data[i], biases.cols * sizeof(double));
    }

    file.close();
}

// Destructor: Prevent memory leak by deleting activation function
DenseLayer::~DenseLayer() {
    delete activation;
}

#ifndef DENSE_LAYER_HPP
#define DENSE_LAYER_HPP

#include "base_layer.hpp"
#include "matrix.hpp"
#include "activation_function.hpp"  // Include activation base class

class DenseLayer : public BaseLayer {
public:
    Matrix weights, biases;
    Matrix input, output;
    ActivationFunction* activation;  // Pointer to activation function

    // Weights Matrix has input_size rows and output_size cols
    // Each neuron has 1 bias so the rows are 1
    DenseLayer(int input_size, int output_size, ActivationFunction* activationFunc) 
        : weights(input_size, output_size), biases(1, output_size), activation(activationFunc) {
        weights.randomize();
        biases.randomize();
    }

    void forward(Matrix &input) override {
        this->input = input;
        output = (input * weights) + biases;
        output = output.applyFunction([this](double x) { return activation->activate(x); }); // Lambda function to use member function
    }    

    void backward(Matrix &error, double learning_rate) override {
        Matrix d_output = error.applyFunction([this](double x) { return activation->derivative(x); }); // Lambda function
        Matrix d_weights = input.transpose() * d_output;

        weights = weights - (d_weights * learning_rate);
        biases = biases -  (d_output * learning_rate);
    }

    ~DenseLayer() {
        delete activation;  // Prevent memory leak
    }
};

#endif // DENSE_LAYER_HPP

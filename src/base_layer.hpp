#ifndef BASE_LAYER_HPP
#define BASE_LAYER_HPP

#include "matrix.hpp"
#include "serializable.hpp"

// Abstract class for all layers
class BaseLayer{
public:
    Matrix input, output;

    BaseLayer(); // Definition for default contructor so linker doesn't get confused
    virtual void forward(Matrix &input) = 0;
    virtual Matrix backward(Matrix &d_output, double learning_rate) = 0;
    virtual ~BaseLayer();  // Virtual destructor with definition for linker
};

#endif  // BASE_LAYER_HPP

#ifndef BASE_LAYER_HPP
#define BASE_LAYER_HPP

#include "../src/matrix.hpp"  // Include the Matrix class

// Layer's abstract class, you cannot create objects
class BaseLayer {
public:
    virtual void forward(Matrix &input) = 0;  // Pure virtual function so it's not called from the derived class
    virtual void backward(Matrix &error, double learning_rate) = 0;
    virtual ~BaseLayer() {}  // Virtual destructor for proper cleanup
};

#endif  // BASE_LAYER_HPP

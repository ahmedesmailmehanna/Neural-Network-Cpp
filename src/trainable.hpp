#ifndef TRAINABLE_HPP
#define TRAINABLE_HPP

#include "matrix.hpp"

class Trainable {
public:
    virtual void train(Matrix &input, Matrix &target, int epochs, double learning_rate) = 0;  
    virtual ~Trainable() {}  
};

#endif  // TRAINABLE_HPP

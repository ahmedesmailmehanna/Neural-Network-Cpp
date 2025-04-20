#ifndef TRAINABLE_HPP
#define TRAINABLE_HPP

#include "../math/matrix.hpp"
#include <vector>

class Trainable {
public:
    virtual void train(Matrix &input, Matrix &target, int epochs, double learning_rate) = 0;  
    virtual void train_batch(std::vector<Matrix> &input, std::vector<Matrix> &target, int epochs, double learning_rate) = 0;  
    virtual ~Trainable() {}  
};

#endif  // TRAINABLE_HPP

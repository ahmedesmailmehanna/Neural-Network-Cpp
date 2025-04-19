#ifndef LAYER_HPP
#define LAYER_HPP

#include "../math/matrix.hpp"
#include "../core/serializable.hpp"

// Abstract class for all layers
class Layer : public Serializable {
public:
    Matrix input, output;

    Layer();
    virtual void forward(Matrix &input) = 0;
    virtual Matrix backward(Matrix &d_output, double learning_rate) = 0;

    virtual void saveToFile(const std::string &filename) = 0;
    virtual void loadFromFile(const std::string &filename) = 0;

    virtual ~Layer();
};

#endif  // LAYER_HPP

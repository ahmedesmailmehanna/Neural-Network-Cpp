#ifndef BASE_LAYER_HPP
#define BASE_LAYER_HPP

#include "matrix.hpp"
#include "serializable.hpp"

// Abstract class for all layers
class BaseLayer : public Serializable {
public:
    Matrix input, output;

    BaseLayer();
    virtual void forward(Matrix &input) = 0;
    virtual void backward(Matrix &error, double learning_rate) = 0;
    virtual ~BaseLayer();  // Virtual destructor

    // Default implementation for serialization (to be overridden)
    virtual void saveToFile(const std::string &filename) override;
    virtual void loadFromFile(const std::string &filename) override;
};

#endif  // BASE_LAYER_HPP

#ifndef SERIALIZABLE_HPP
#define SERIALIZABLE_HPP

#include <fstream>
#include "matrix.hpp"

class Serializable {
public:
    virtual void saveToFile(const std::string &filename) = 0;
    virtual void loadFromFile(const std::string &filename) = 0;
    virtual ~Serializable() {}  // Virtual destructor
};

#endif  // SERIALIZABLE_HPP

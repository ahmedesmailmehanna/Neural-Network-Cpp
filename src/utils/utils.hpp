#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>
#include "../math/matrix.hpp"

class utils { // Utility class for various functions
    public:
    // MNIST dataset
    static std::vector<Matrix> loadMNISTImages(const std::string &filename);
    static std::vector<int> loadMNISTLabels(const std::string &filename);
    
    // Matrix utilities
    static Matrix flatten(const Matrix& m);
    static Matrix createMNISTTargetMatrix(int label);
};




#endif  // UTILS_HPP

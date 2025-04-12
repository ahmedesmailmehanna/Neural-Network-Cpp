#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <vector>
#include <string>
#include "../math/matrix.hpp"

std::vector<Matrix> loadMNISTImages(const std::string &filename);
std::vector<int> loadMNISTLabels(const std::string &filename);

#endif  // MNIST_LOADER_HPP

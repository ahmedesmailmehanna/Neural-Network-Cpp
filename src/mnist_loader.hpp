#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <vector>
#include <string>
#include "matrix.hpp"

std::vector<Matrix> loadMNISTImages(const std::string &filename, int num_images);
std::vector<int> loadMNISTLabels(const std::string &filename, int num_labels);

#endif  // MNIST_LOADER_HPP

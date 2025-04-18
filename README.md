# Neural Network Framework in C++

A lightweight neural network library implemented in modern C++ with dense layers and MNIST support.

Still in development.

## Features

- 🧠 Dense (fully-connected) layers
- ⚡ ReLU/Sigmoid/Softmax activations
- 📊 MNIST dataset loader
- 💾 Model serialization

## Getting Started

### Requirements
- C++17 compiler (g++/clang++)
- Standard library only (no external dependencies)

### Compilation
```bash
# Compile all source files directly
g++ g++ -std=c++17 -O3 -o main main.cpp src/math/matrix.cpp src/layers/dense_layer.cpp src/layers/base_layer.cpp src/utils/mnist_loader.cpp src/utils/matrix_utils.cpp -I./

## Getting Started


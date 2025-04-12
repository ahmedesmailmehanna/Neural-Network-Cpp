# Neural Network Framework in C++

A lightweight neural network library implemented in modern C++ with dense layers and MNIST support.

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
g++ -std=c++17 -O3 -o neuralnet \
    src/matrix.cpp \
    src/dense_layer.cpp \
    src/mnist_loader.cpp \
    src/main.cpp